from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment


class GaussianColorDistError(RuntimeError):
    """Base class for Gaussian color distance metric errors."""


class GaussianColorDistInputError(GaussianColorDistError):
    """Raised when metric input cannot provide valid OD pixels."""


@dataclass(frozen=True)
class GaussianColorDistConfig:
    n_components: int = 6
    io: float = 240.0
    eps: float = 1e-8
    covariance_eps: float = 1e-4
    max_target_pixels: int = 120_000
    max_domain_pixels: int = 120_000
    max_update_pixels: int = 20_000
    max_em_iter: int = 80
    em_tol: float = 1e-4
    random_seed: int = 0
    precision: int = 6


@dataclass
class GaussianMixture:
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    n_iter: int
    log_likelihood: float


@dataclass
class GaussianColorDistMetric:
    target_patch: np.ndarray | Any
    config: GaussianColorDistConfig = field(default_factory=GaussianColorDistConfig)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.config.random_seed)
        self._source_pixels = np.empty((0, 3), dtype=np.float64)
        self._normalized_pixels = np.empty((0, 3), dtype=np.float64)
        target_pixels = self._sample_od_pixels(
            self.target_patch,
            max_pixels=self.config.max_target_pixels,
        )
        self._target_gmm = fit_gmm(
            target_pixels,
            config=self.config,
            rng=self._rng,
        )

    def evaluate(
        self,
        source_patch: np.ndarray | Any,
        output_patch: np.ndarray | Any,
    ) -> None:
        self._source_pixels = self._merge_samples(
            self._source_pixels,
            self._sample_od_pixels(source_patch),
        )
        self._normalized_pixels = self._merge_samples(
            self._normalized_pixels,
            self._sample_od_pixels(output_patch),
        )

    def finalize(self) -> dict[str, float | None]:
        if len(self._source_pixels) == 0 or len(self._normalized_pixels) == 0:
            return _empty_scores()

        n_components = int(self._target_gmm.means.shape[0])
        source_gmm = fit_gmm(
            self._source_pixels,
            config=self.config,
            rng=self._rng,
            n_components=n_components,
        )
        normalized_gmm = fit_gmm(
            self._normalized_pixels,
            config=self.config,
            rng=self._rng,
            n_components=n_components,
        )

        source_target_dist = gaussian_color_distance(source_gmm, self._target_gmm)
        normalized_target_dist = gaussian_color_distance(
            normalized_gmm,
            self._target_gmm,
        )
        color_gain = source_target_dist - normalized_target_dist

        return {
            "gaussian_color_dist": round(
                normalized_target_dist,
                self.config.precision,
            ),
            "gaussian_color_gain": round(color_gain, self.config.precision),
        }

    def _sample_od_pixels(
        self,
        patch: np.ndarray | Any,
        max_pixels: int | None = None,
    ) -> np.ndarray:
        rgb = _to_bhwc_rgb(patch)
        od = _rgb_to_od(rgb, self.config).reshape(-1, 3)

        rgb_flat = rgb.reshape(-1, 3).astype(np.float64, copy=False) / 255.0
        channel_max = np.max(rgb_flat, axis=1)
        channel_min = np.min(rgb_flat, axis=1)
        saturation = (channel_max - channel_min) / (channel_max + self.config.eps)
        od_norm = np.linalg.norm(od, axis=1)

        valid = (
            np.isfinite(od).all(axis=1)
            & (od_norm > 0.15)
            & (channel_max < 0.98)
            & (saturation > 0.02)
        )
        od = od[valid]
        if len(od) == 0:
            fallback = np.isfinite(od_norm) & (od_norm > 0.15)
            od = _rgb_to_od(rgb, self.config).reshape(-1, 3)[fallback]

        if len(od) == 0:
            raise GaussianColorDistInputError("No valid tissue OD pixels found.")

        max_pixels = max_pixels or self.config.max_update_pixels
        if len(od) > max_pixels:
            indices = self._rng.choice(len(od), size=max_pixels, replace=False)
            od = od[indices]

        return od.astype(np.float64, copy=False)

    def _merge_samples(
        self,
        existing: np.ndarray,
        update: np.ndarray,
    ) -> np.ndarray:
        merged = np.concatenate([existing, update], axis=0)
        if len(merged) <= self.config.max_domain_pixels:
            return merged

        indices = self._rng.choice(
            len(merged),
            size=self.config.max_domain_pixels,
            replace=False,
        )
        return merged[indices]


def fit_gmm(
    pixels: np.ndarray,
    config: GaussianColorDistConfig,
    rng: np.random.Generator,
    n_components: int | None = None,
) -> GaussianMixture:
    pixels = _validate_pixels(pixels)
    k = int(n_components or config.n_components)
    k = max(1, min(k, len(pixels)))

    means = _init_means_kmeans_plus_plus(pixels, k, rng)
    global_cov = np.cov(pixels, rowvar=False)
    if global_cov.ndim == 0:
        global_cov = np.eye(3) * float(global_cov)
    global_cov = _regularize_covariance(global_cov, config.covariance_eps)
    covariances = np.repeat(global_cov[np.newaxis, :, :], k, axis=0)
    weights = np.full(k, 1.0 / k, dtype=np.float64)

    previous_ll = -np.inf
    log_likelihood = -np.inf
    for iteration in range(1, config.max_em_iter + 1):
        responsibilities, log_likelihood = _expectation(
            pixels,
            means,
            covariances,
            weights,
            config,
        )
        nk = responsibilities.sum(axis=0) + config.eps
        weights = nk / len(pixels)
        means = (responsibilities.T @ pixels) / nk[:, None]

        for component in range(k):
            centered = pixels - means[component]
            weighted = centered * responsibilities[:, component:component + 1]
            cov = weighted.T @ centered / nk[component]
            covariances[component] = _regularize_covariance(
                cov,
                config.covariance_eps,
            )

        if abs(log_likelihood - previous_ll) < config.em_tol:
            break
        previous_ll = log_likelihood

    return GaussianMixture(
        means=means,
        covariances=covariances,
        weights=weights,
        n_iter=iteration,
        log_likelihood=float(log_likelihood),
    )


def gaussian_color_distance(
    source: GaussianMixture,
    target: GaussianMixture,
) -> float:
    cost = np.zeros((len(source.weights), len(target.weights)), dtype=np.float64)
    for source_index in range(len(source.weights)):
        for target_index in range(len(target.weights)):
            cost[source_index, target_index] = gaussian_wasserstein_distance(
                source.means[source_index],
                source.covariances[source_index],
                target.means[target_index],
                target.covariances[target_index],
            )

    row_indices, col_indices = linear_sum_assignment(cost)
    matched_cost = 0.0
    matched_weight = 0.0
    for row, col in zip(row_indices, col_indices):
        weight = 0.5 * (source.weights[row] + target.weights[col])
        matched_cost += weight * cost[row, col]
        matched_weight += weight

    return float(matched_cost / max(matched_weight, 1e-12))


def gaussian_wasserstein_distance(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
) -> float:
    mean_term = float(np.sum((mean_a - mean_b) ** 2))
    sqrt_product = sqrtm(covariance_a @ covariance_b)
    if np.iscomplexobj(sqrt_product):
        sqrt_product = sqrt_product.real

    covariance_term = float(
        np.trace(covariance_a + covariance_b - 2.0 * sqrt_product)
    )
    return float(np.sqrt(max(mean_term + covariance_term, 0.0)))


def _expectation(
    pixels: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
    config: GaussianColorDistConfig,
) -> tuple[np.ndarray, float]:
    log_probs = np.empty((len(pixels), len(weights)), dtype=np.float64)
    for component in range(len(weights)):
        log_probs[:, component] = (
            np.log(weights[component] + config.eps)
            + _log_multivariate_normal(
                pixels,
                means[component],
                covariances[component],
                config,
            )
        )

    max_log = np.max(log_probs, axis=1, keepdims=True)
    stabilized = np.exp(log_probs - max_log)
    normalizer = np.sum(stabilized, axis=1, keepdims=True) + config.eps
    responsibilities = stabilized / normalizer
    log_likelihood = float(np.sum(max_log[:, 0] + np.log(normalizer[:, 0])))
    return responsibilities, log_likelihood


def _log_multivariate_normal(
    pixels: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    config: GaussianColorDistConfig,
) -> np.ndarray:
    covariance = _regularize_covariance(covariance, config.covariance_eps)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        covariance = _regularize_covariance(covariance, config.covariance_eps * 10)
        sign, logdet = np.linalg.slogdet(covariance)

    inv_cov = np.linalg.pinv(covariance)
    centered = pixels - mean
    mahalanobis = np.sum((centered @ inv_cov) * centered, axis=1)
    dim = pixels.shape[1]
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + mahalanobis)


def _init_means_kmeans_plus_plus(
    pixels: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    means = np.empty((k, pixels.shape[1]), dtype=np.float64)
    means[0] = pixels[rng.integers(0, len(pixels))]

    distances = np.sum((pixels - means[0]) ** 2, axis=1)
    for index in range(1, k):
        total = float(distances.sum())
        if total <= 0:
            means[index:] = pixels[rng.choice(len(pixels), size=k - index)]
            break

        probabilities = distances / total
        means[index] = pixels[rng.choice(len(pixels), p=probabilities)]
        new_distances = np.sum((pixels - means[index]) ** 2, axis=1)
        distances = np.minimum(distances, new_distances)

    return means


def _regularize_covariance(covariance: np.ndarray, eps: float) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (3, 3):
        covariance = np.eye(3, dtype=np.float64) * eps
    covariance = 0.5 * (covariance + covariance.T)
    return covariance + np.eye(3, dtype=np.float64) * eps


def _validate_pixels(pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise GaussianColorDistInputError(
            f"pixels는 shape [N, 3]이어야 합니다. 입력 shape: {pixels.shape}"
        )
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    if len(pixels) == 0:
        raise GaussianColorDistInputError("No valid pixels found.")
    return pixels


def _rgb_to_od(
    rgb: np.ndarray,
    config: GaussianColorDistConfig,
) -> np.ndarray:
    rgb = rgb.astype(np.float64, copy=False)
    if np.nanmax(rgb) <= 1.0:
        rgb = rgb * 255.0

    rgb = np.clip(rgb, 1.0, config.io)
    return -np.log((rgb + config.eps) / config.io)


def _to_bhwc_rgb(patch: np.ndarray | Any) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None

    if torch is not None and isinstance(patch, torch.Tensor):
        patch = patch.detach().cpu().numpy()
    patch = np.asarray(patch)

    if patch.ndim == 3:
        if patch.shape[0] == 3:
            patch = patch[np.newaxis, ...].transpose(0, 2, 3, 1)
        elif patch.shape[-1] == 3:
            patch = patch[np.newaxis, ...]
        else:
            raise GaussianColorDistInputError(
                f"RGB patch shape을 해석할 수 없습니다: {patch.shape}"
            )
    elif patch.ndim == 4:
        if patch.shape[1] == 3:
            patch = patch.transpose(0, 2, 3, 1)
        elif patch.shape[-1] != 3:
            raise GaussianColorDistInputError(
                f"RGB patch batch shape을 해석할 수 없습니다: {patch.shape}"
            )
    else:
        raise GaussianColorDistInputError(
            f"patch는 3D 또는 4D여야 합니다. 입력 shape: {patch.shape}"
        )

    return np.ascontiguousarray(patch)


def _empty_scores() -> dict[str, None]:
    return {
        "gaussian_color_dist": None,
        "gaussian_color_gain": None,
    }
