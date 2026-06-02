from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class CustomMetricError(RuntimeError):
    """Base class for custom stain metric errors."""


class CustomMetricInputError(CustomMetricError):
    """Raised when custom stain metric inputs are invalid."""


DEFAULT_STAIN_MATRIX = np.array(
    [
        [0.65, 0.07],
        [0.70, 0.99],
        [0.29, 0.11],
    ],
    dtype=np.float64,
)

DEFAULT_STAIN_VECTOR = DEFAULT_STAIN_MATRIX[:, :1]


@dataclass
class StainMetricConfig:
    alpha: float = 1.0
    beta: float = 0.15
    io: float = 240.0
    eps: float = 1e-8
    max_fit_pixels: int = 200_000
    random_seed: int = 0
    max_stain_cosine: float = 0.9999
    min_angle_denominator_deg: float = 10.0
    precision: int = 6


@dataclass
class CustomStainMetric:
    target_patch: np.ndarray | Any
    config: StainMetricConfig = field(default_factory=StainMetricConfig)

    def __post_init__(self) -> None:
        self._target_rgb = _to_bhwc_rgb(self.target_patch)
        self._target_stain_matrix = estimate_stain_matrix(
            self._target_rgb,
            config=self.config,
        )
        self._scores: list[dict[str, float]] = []

    def evaluate(
        self,
        source_patch: np.ndarray | Any,
        output_patch: np.ndarray | Any,
    ) -> None:
        self._scores.append(
            calculate_custom_stain_metrics(
                source_patch=source_patch,
                normalized_patch=output_patch,
                target_patch=self._target_rgb,
                target_stain_matrix=self._target_stain_matrix,
                config=self.config,
            )
        )

    def finalize(self) -> dict[str, float | None]:
        if not self._scores:
            return _empty_scores()

        keys = self._scores[0].keys()
        scores = {
            key: _round_or_none(
                float(np.nanmean([score[key] for score in self._scores])),
                self.config.precision,
            )
            for key in keys
        }
        if (
            scores["source_target_stain_angle_deg"] is not None
            and scores["normalized_target_stain_angle_deg"] is not None
        ):
            scores["stain_angle_improvement_deg"] = round(
                scores["source_target_stain_angle_deg"]
                - scores["normalized_target_stain_angle_deg"],
                self.config.precision,
            )

        return scores


def calculate_custom_stain_metrics(
    source_patch: np.ndarray | Any,
    normalized_patch: np.ndarray | Any,
    target_patch: np.ndarray | Any,
    target_stain_matrix: np.ndarray | None = None,
    config: StainMetricConfig | None = None,
) -> dict[str, float]:
    """Calculate four OD-space metrics for A, normalized A, and target patches.

    Returned keys:
    - stain_preservation_corr: Pearson correlation between source and normalized
      concentration maps. Range is -1 to 1, higher is better.
    - normalized_target_stain_angle_deg: mean matched stain-vector angle between
      normalized and target patches. Range is 0 to 90 degrees, lower is better.
      If one patch has only one stable stain axis, the best one-axis match is used.
    - source_target_stain_angle_deg: same angle for source and target patches.
    - stain_angle_improvement_deg: source-target angle minus normalized-target
      angle. Positive means normalization moved stain vectors toward the target.
    - custom_structure_score: clipped structure coordinate, range 0 to 1.
    - custom_color_score: clipped stain-alignment coordinate, range -1 to 1.
    - source_stain_rank, normalized_stain_rank, target_stain_rank: number of
      stable stain axes used for each patch, either 1 or 2.
    """
    config = config or StainMetricConfig()
    source_rgb = _to_bhwc_rgb(source_patch)
    normalized_rgb = _to_bhwc_rgb(normalized_patch)
    target_rgb = _to_bhwc_rgb(target_patch)

    if source_rgb.shape != normalized_rgb.shape:
        raise CustomMetricInputError(
            "source_patch와 normalized_patch의 shape이 같아야 합니다. "
            f"입력 shape: {source_rgb.shape} vs {normalized_rgb.shape}"
        )

    source_stain_matrix = estimate_stain_matrix(source_rgb, config=config)
    normalized_stain_matrix = estimate_stain_matrix(normalized_rgb, config=config)
    if target_stain_matrix is None:
        target_stain_matrix = estimate_stain_matrix(target_rgb, config=config)
    else:
        target_stain_matrix = _normalize_stain_matrix(
            np.asarray(target_stain_matrix, dtype=np.float64),
            config=config,
        )

    preservation_corr = _concentration_pearson(
        source_rgb,
        normalized_rgb,
        source_stain_matrix,
        normalized_stain_matrix,
        config=config,
    )
    normalized_target_angle = matched_stain_angle_deg(
        normalized_stain_matrix,
        target_stain_matrix,
        config=config,
    )
    source_target_angle = matched_stain_angle_deg(
        source_stain_matrix,
        target_stain_matrix,
        config=config,
    )
    angle_improvement = source_target_angle - normalized_target_angle
    structure_score = np.clip(preservation_corr, 0.0, 1.0)
    color_score = np.clip(
        angle_improvement
        / max(source_target_angle, config.min_angle_denominator_deg),
        -1.0,
        1.0,
    )

    return {
        "stain_preservation_corr": round(preservation_corr, config.precision),
        "normalized_target_stain_angle_deg": round(
            normalized_target_angle,
            config.precision,
        ),
        "source_target_stain_angle_deg": round(source_target_angle, config.precision),
        "stain_angle_improvement_deg": round(angle_improvement, config.precision),
        "custom_structure_score": round(float(structure_score), config.precision),
        "custom_color_score": round(float(color_score), config.precision),
        "source_stain_rank": float(source_stain_matrix.shape[1]),
        "normalized_stain_rank": float(normalized_stain_matrix.shape[1]),
        "target_stain_rank": float(target_stain_matrix.shape[1]),
    }


def estimate_stain_matrix(
    rgb: np.ndarray | Any,
    config: StainMetricConfig | None = None,
) -> np.ndarray:
    config = config or StainMetricConfig()
    rgb = _to_bhwc_rgb(rgb)
    od = _prepare_od(rgb, config=config)

    _, _, vh = np.linalg.svd(od, full_matrices=False)
    top_vectors = vh[:2].T

    projected = od @ top_vectors
    angles = np.arctan2(projected[:, 1], projected[:, 0])

    min_angle = np.percentile(angles, config.alpha)
    max_angle = np.percentile(angles, 100.0 - config.alpha)

    v1 = top_vectors @ np.array([np.cos(min_angle), np.sin(min_angle)])
    v2 = top_vectors @ np.array([np.cos(max_angle), np.sin(max_angle)])

    if v1[0] > v2[0]:
        stain_matrix = np.stack([v1, v2], axis=1)
    else:
        stain_matrix = np.stack([v2, v1], axis=1)

    stain_matrix = _normalize_stain_matrix(stain_matrix, config=config)
    if not _is_stable_stain_matrix(stain_matrix, config=config):
        return _estimate_single_stain_vector(od, config=config)

    return stain_matrix


def matched_stain_angle_deg(
    first_stain_matrix: np.ndarray,
    second_stain_matrix: np.ndarray,
    config: StainMetricConfig | None = None,
) -> float:
    config = config or StainMetricConfig()
    first = _normalize_stain_matrix(first_stain_matrix, config=config)
    second = _normalize_stain_matrix(second_stain_matrix, config=config)
    matched_pairs = _match_stain_columns(first, second, config=config)
    angles = [
        _vector_angle_deg(first[:, first_index], second[:, second_index], config=config)
        for first_index, second_index in matched_pairs
    ]
    if not angles:
        return 0.0

    return float(np.mean(angles))


def _to_bhwc_rgb(image: np.ndarray | Any) -> np.ndarray:
    if hasattr(image, "detach") and callable(image.detach):
        image = image.detach().cpu().numpy()

    rgb = np.asarray(image)
    if rgb.ndim == 3:
        if rgb.shape[-1] == 3:
            rgb = rgb[np.newaxis, ...]
        elif rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))[np.newaxis, ...]
        else:
            raise CustomMetricInputError(
                f"RGB patch는 3채널이어야 합니다. 입력 shape: {rgb.shape}"
            )
    elif rgb.ndim == 4:
        if rgb.shape[-1] == 3:
            pass
        elif rgb.shape[1] == 3:
            rgb = np.transpose(rgb, (0, 2, 3, 1))
        else:
            raise CustomMetricInputError(
                f"RGB patch batch는 3채널이어야 합니다. 입력 shape: {rgb.shape}"
            )
    else:
        raise CustomMetricInputError(
            f"RGB patch 또는 batch가 필요합니다. 입력 차원: {rgb.ndim}D"
        )

    rgb = np.ascontiguousarray(rgb, dtype=np.float64)
    if np.nanmax(rgb) <= 1.0:
        rgb = rgb * 255.0

    return np.clip(rgb, 0.0, 255.0)


def _rgb_to_od(rgb: np.ndarray, config: StainMetricConfig) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float64), 1.0, config.io)
    return -np.log((rgb + config.eps) / config.io)


def _prepare_od(rgb: np.ndarray, config: StainMetricConfig) -> np.ndarray:
    flat_rgb = _to_bhwc_rgb(rgb).reshape(-1, 3)
    od = _rgb_to_od(flat_rgb, config=config).reshape(-1, 3)
    od_norm = np.linalg.norm(od, axis=1)

    rgb_unit = flat_rgb / 255.0
    channel_max = np.max(rgb_unit, axis=1)
    channel_min = np.min(rgb_unit, axis=1)
    saturation = (channel_max - channel_min) / (channel_max + config.eps)

    valid = (od_norm > config.beta) & (channel_max < 0.98) & (saturation > 0.02)
    if not np.any(valid):
        valid = od_norm > config.beta
    if not np.any(valid):
        raise CustomMetricInputError("유효한 조직 픽셀을 찾지 못했습니다. beta 값을 낮춰보세요.")

    od = od[valid]
    if len(od) > config.max_fit_pixels:
        rng = np.random.default_rng(config.random_seed)
        indices = rng.choice(len(od), size=config.max_fit_pixels, replace=False)
        od = od[indices]

    return od


def _estimate_concentrations(
    rgb: np.ndarray,
    stain_matrix: np.ndarray,
    config: StainMetricConfig,
) -> np.ndarray:
    od = _rgb_to_od(_to_bhwc_rgb(rgb), config=config).reshape(-1, 3).T
    concentrations = np.linalg.pinv(stain_matrix) @ od
    return np.clip(concentrations, 0.0, None)


def _estimate_single_stain_vector(
    od: np.ndarray,
    config: StainMetricConfig,
) -> np.ndarray:
    _, _, vh = np.linalg.svd(od, full_matrices=False)
    stain_vector = vh[:1].T
    return _normalize_stain_matrix(stain_vector, config=config)


def _concentration_pearson(
    source_rgb: np.ndarray,
    normalized_rgb: np.ndarray,
    source_stain_matrix: np.ndarray,
    normalized_stain_matrix: np.ndarray,
    config: StainMetricConfig,
) -> float:
    source_conc = _estimate_concentrations(
        source_rgb,
        source_stain_matrix,
        config=config,
    )
    normalized_conc = _estimate_concentrations(
        normalized_rgb,
        normalized_stain_matrix,
        config=config,
    )
    matched_pairs = _match_stain_columns(
        source_stain_matrix,
        normalized_stain_matrix,
        config=config,
    )

    tissue_mask = _tissue_mask(source_rgb, config=config)
    if not np.any(tissue_mask):
        tissue_mask = np.ones(source_conc.shape[1], dtype=bool)

    correlations = [
        _pearson(
            source_conc[source_index, tissue_mask],
            normalized_conc[normalized_index, tissue_mask],
            config,
        )
        for source_index, normalized_index in matched_pairs
    ]
    valid = [score for score in correlations if np.isfinite(score)]
    if not valid:
        return 0.0

    return float(np.mean(valid))


def _match_stain_columns(
    first_stain_matrix: np.ndarray,
    second_stain_matrix: np.ndarray,
    config: StainMetricConfig,
) -> list[tuple[int, int]]:
    first_count = first_stain_matrix.shape[1]
    second_count = second_stain_matrix.shape[1]
    if first_count == 1 and second_count == 1:
        return [(0, 0)]

    if first_count == 1:
        best_index = min(
            range(second_count),
            key=lambda index: _vector_angle_deg(
                first_stain_matrix[:, 0],
                second_stain_matrix[:, index],
                config=config,
            ),
        )
        return [(0, best_index)]

    if second_count == 1:
        best_index = min(
            range(first_count),
            key=lambda index: _vector_angle_deg(
                first_stain_matrix[:, index],
                second_stain_matrix[:, 0],
                config=config,
            ),
        )
        return [(best_index, 0)]

    angle_matrix = np.array(
        [
            [
                _vector_angle_deg(
                    first_stain_matrix[:, first_index],
                    second_stain_matrix[:, second_index],
                    config=config,
                )
                for second_index in range(second_count)
            ]
            for first_index in range(first_count)
        ],
        dtype=np.float64,
    )
    same_order = angle_matrix[0, 0] + angle_matrix[1, 1]
    swapped_order = angle_matrix[0, 1] + angle_matrix[1, 0]
    if same_order <= swapped_order:
        return [(0, 0), (1, 1)]
    return [(0, 1), (1, 0)]


def _tissue_mask(rgb: np.ndarray, config: StainMetricConfig) -> np.ndarray:
    flat_rgb = _to_bhwc_rgb(rgb).reshape(-1, 3)
    od = _rgb_to_od(flat_rgb, config=config).reshape(-1, 3)
    return np.linalg.norm(od, axis=1) > config.beta


def _normalize_stain_matrix(
    stain_matrix: np.ndarray,
    config: StainMetricConfig,
) -> np.ndarray:
    stain_matrix = np.asarray(stain_matrix, dtype=np.float64).copy()
    if stain_matrix.ndim == 1:
        stain_matrix = stain_matrix[:, np.newaxis]
    if stain_matrix.ndim != 2 or stain_matrix.shape[0] != 3:
        raise CustomMetricInputError(
            f"stain_matrix는 (3, K) shape이어야 합니다. 입력 shape: {stain_matrix.shape}"
        )
    if stain_matrix.shape[1] not in {1, 2}:
        raise CustomMetricInputError(
            f"stain_matrix는 1개 또는 2개의 stain column을 가져야 합니다. 입력 shape: {stain_matrix.shape}"
        )

    for index in range(stain_matrix.shape[1]):
        if np.sum(stain_matrix[:, index]) < 0:
            stain_matrix[:, index] *= -1.0

    stain_matrix = np.maximum(stain_matrix, config.eps)
    norms = np.linalg.norm(stain_matrix, axis=0, keepdims=True) + config.eps
    return stain_matrix / norms


def _is_stable_stain_matrix(
    stain_matrix: np.ndarray,
    config: StainMetricConfig,
) -> bool:
    if stain_matrix.ndim != 2 or stain_matrix.shape[0] != 3:
        return False
    if stain_matrix.shape[1] == 1:
        return True
    if stain_matrix.shape[1] != 2:
        return False
    if not np.isfinite(stain_matrix).all():
        return False

    cosine = abs(float(np.dot(stain_matrix[:, 0], stain_matrix[:, 1])))
    return cosine < config.max_stain_cosine


def _vector_angle_deg(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
    config: StainMetricConfig,
) -> float:
    first = np.asarray(first_vector, dtype=np.float64)
    second = np.asarray(second_vector, dtype=np.float64)
    first = first / (np.linalg.norm(first) + config.eps)
    second = second / (np.linalg.norm(second) + config.eps)
    cosine = abs(float(np.dot(first, second)))
    cosine = np.clip(cosine, 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _pearson(first: np.ndarray, second: np.ndarray, config: StainMetricConfig) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = (
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered) + config.eps
    )
    if denominator <= config.eps * 10:
        return np.nan

    return float(np.dot(first_centered, second_centered) / denominator)


def _round_or_none(value: float, precision: int) -> float | None:
    if not np.isfinite(value):
        return None
    return round(value, precision)


def _empty_scores() -> dict[str, None]:
    return {
        "stain_preservation_corr": None,
        "normalized_target_stain_angle_deg": None,
        "source_target_stain_angle_deg": None,
        "stain_angle_improvement_deg": None,
        "custom_structure_score": None,
        "custom_color_score": None,
        "source_stain_rank": None,
        "normalized_stain_rank": None,
        "target_stain_rank": None,
    }
