"""
dataset.py discovers files like scc_01_nz210.tif, splits by specimen/case, samples tissue patches from WSIs, 
returns unpaired source/target patches for CycleGAN training, and optionally returns aligned canonical patches for validation/evaluation only
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from ai.models.multistain.config import MultiStainCycleGANConfig
from ai.samplers.patch_sampler import NoTissueFoundError, NoValidPatchError, PatchSampler
from ai.wsi.handle import WSIHandle
from ai.wsi.loader import load_patch, open_wsi_handle
from ai.wsi.patch_ref import PatchRef


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MIN_PLAUSIBLE_MPP = 0.05
MAX_PLAUSIBLE_MPP = 5.0


@dataclass(frozen=True, slots=True)
class SlideRecord:
    """One WSI belonging to one tissue sample and one scanner domain."""

    sample_id: str
    scanner_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class PatchItem:
    """A sampled patch reference tied to its source WSI record."""

    record: SlideRecord
    ref: PatchRef


class MultiStainPatchDataset(Dataset):
    """
    Patch dataset for many-source-to-one MultiStain-CycleGAN training.

    Each item returns an unpaired source-domain patch and canonical target-domain
    patch. The same public dataset contains matched scans of each specimen, but
    the training pair is intentionally unpaired because CycleGAN-style training
    should not require exact one-to-one supervision.
    """

    def __init__(
        self,
        config: MultiStainCycleGANConfig,
        sample_ids: list[str] | None = None,
        split_name: str = "train",
        include_aligned_target: bool = False,
    ) -> None:
        self.config = config
        self.split_name = split_name
        self.include_aligned_target = bool(include_aligned_target)

        self.dataset_dir = Path(config.dataset_dir)
        self.canonical_domain = config.canonical_domain
        self.source_domains = tuple(config.source_domains)
        self.domain_to_index = {
            domain: index for index, domain in enumerate(config.all_domains)
        }
        self.rng = random.Random(config.split_seed + _stable_int(split_name))
        self.handles: dict[Path, WSIHandle] = {}

        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.records_by_sample = discover_multiscanner_records(
            self.dataset_dir,
            recursive=config.recursive,
        )
        if sample_ids is not None:
            wanted = set(sample_ids)
            self.records_by_sample = {
                sample_id: records
                for sample_id, records in self.records_by_sample.items()
                if sample_id in wanted
            }

        self.sample_ids = sorted(self.records_by_sample)
        if not self.sample_ids:
            raise ValueError(f"No samples found for split {split_name!r}.")

        self._validate_samples()

        self.source_items: list[PatchItem] = []
        self.target_items: list[PatchItem] = []
        self.canonical_records = {
            sample_id: records[self.canonical_domain]
            for sample_id, records in self.records_by_sample.items()
        }

        cache_path = self._cache_path()
        if config.use_patch_cache and cache_path.is_file():
            self._log(f"Loading cached patch refs: {cache_path}")
            self._load_patch_cache(cache_path)
        else:
            self._build_patch_items()
            if config.use_patch_cache:
                self._save_patch_cache(cache_path)
                self._log(f"Saved patch-ref cache: {cache_path}")

        if not self.source_items:
            raise ValueError("No source-domain patches were sampled.")
        if not self.target_items:
            raise ValueError("No canonical target-domain patches were sampled.")

        self._log(
            f"Ready: samples={len(self.sample_ids)} source_patches={len(self.source_items)} "
            f"target_patches={len(self.target_items)}"
        )

    @property
    def source_scanner_ids(self) -> list[str]:
        return sorted({item.record.scanner_id for item in self.source_items})

    def __len__(self) -> int:
        return len(self.source_items)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int | str]:
        source_item = self.source_items[index % len(self.source_items)]
        target_item = self.target_items[
            self.rng.randrange(len(self.target_items))
        ]

        source = self._load_normalized_patch(source_item.ref)
        target = self._load_normalized_patch(target_item.ref)

        batch: dict[str, np.ndarray | int | str] = {
            "source": source,
            "target": target,
            "identity": target,
            "source_domain": source_item.record.scanner_id,
            "target_domain": target_item.record.scanner_id,
            "source_domain_idx": self.domain_to_index[source_item.record.scanner_id],
            "target_domain_idx": self.domain_to_index[target_item.record.scanner_id],
            "sample_id": source_item.record.sample_id,
            "source_path": str(source_item.record.path),
            "target_path": str(target_item.record.path),
        }

        if self.include_aligned_target:
            aligned_ref = self._aligned_canonical_ref(source_item)
            batch["aligned_target"] = self._load_normalized_patch(aligned_ref)

        return batch

    def _validate_samples(self) -> None:
        required = set(self.source_domains) | {self.canonical_domain}
        bad_samples: list[str] = []
        for sample_id, records in self.records_by_sample.items():
            missing = required - set(records)
            if missing:
                bad_samples.append(f"{sample_id}: missing {sorted(missing)}")
        if bad_samples:
            preview = "; ".join(bad_samples[:5])
            raise ValueError(
                "Every selected specimen must contain all configured scanners. "
                f"Problems: {preview}"
            )

    def _build_patch_items(self) -> None:
        sampler = PatchSampler(
            patch_size=self._native_patch_size_for_mpp(self.config.target_mpp),
            stride=self._native_patch_size_for_mpp(self.config.target_mpp),
            read_level=self.config.read_level,
            mask_longest_side=self.config.mask_longest_side,
            strict_mpp_check=self.config.strict_mpp_check,
            result_dir=Path("result") / "multistain_patch_sampler" / self.split_name,
            verbose=self.config.verbose,
            log_to_file=False,
        )

        sample_iter = tqdm(
            self.sample_ids,
            desc=f"MultiStain {self.split_name} patch sampling",
            unit="sample",
            leave=False,
            disable=not self.config.verbose,
        )
        for sample_id in sample_iter:
            records = self.records_by_sample[sample_id]

            canonical_record = records[self.canonical_domain]
            canonical_refs = self._sample_record(
                sampler=sampler,
                record=canonical_record,
                max_patches=self.config.patches_per_target_slide,
                seed=self.config.split_seed + _stable_int(sample_id, "target"),
            )
            self.target_items.extend(PatchItem(canonical_record, ref) for ref in canonical_refs)

            for scanner_id in self.source_domains:
                source_record = records[scanner_id]
                refs = self._sample_record(
                    sampler=sampler,
                    record=source_record,
                    max_patches=self.config.patches_per_source_slide,
                    seed=self.config.split_seed + _stable_int(sample_id, scanner_id),
                )
                self.source_items.extend(PatchItem(source_record, ref) for ref in refs)

    def _sample_record(
        self,
        sampler: PatchSampler,
        record: SlideRecord,
        max_patches: int,
        seed: int,
    ) -> list[PatchRef]:
        handle = self._open(record)
        patch_size = self._read_size_for_handle(handle, record.scanner_id)
        sampler.patch_size = patch_size
        sampler.stride = patch_size

        original_mask_longest_side = sampler.mask_longest_side
        retry_sides = sorted(
            {
                original_mask_longest_side,
                original_mask_longest_side * 2,
                original_mask_longest_side * 4,
                1024,
                2048,
            }
        )
        last_error: Exception | None = None

        for mask_longest_side in retry_sides:
            sampler.mask_longest_side = mask_longest_side
            try:
                refs = sampler.sample(
                    handle,
                    mode="training",
                    max_patches=max_patches,
                    seed=seed,
                    save_debug=False,
                )
                sampler.mask_longest_side = original_mask_longest_side
                self._log(
                    f"sample={record.sample_id} scanner={record.scanner_id} "
                    f"patches={len(refs)} read_size={patch_size}"
                )
                return refs
            except (NoTissueFoundError, NoValidPatchError) as exc:
                last_error = exc

        sampler.mask_longest_side = original_mask_longest_side
        self._log(
            f"skipped sample={record.sample_id} scanner={record.scanner_id}: {last_error}"
        )
        return []

    def _aligned_canonical_ref(self, source_item: PatchItem) -> PatchRef:
        source_record = source_item.record
        canonical_record = self.canonical_records[source_record.sample_id]
        source_handle = self._open(source_record)
        canonical_handle = self._open(canonical_record)

        source_mpp_x, source_mpp_y = source_handle.mpp
        canonical_mpp_x, canonical_mpp_y = canonical_handle.mpp
        read_size = self._read_size_for_handle(canonical_handle, canonical_record.scanner_id)
        level0_span = int(
            round(read_size * float(canonical_handle.level_downsamples[self.config.read_level]))
        )

        x = int(round(source_item.ref.x * source_mpp_x / canonical_mpp_x))
        y = int(round(source_item.ref.y * source_mpp_y / canonical_mpp_y))
        x = max(0, min(x, canonical_handle.dim[0] - level0_span))
        y = max(0, min(y, canonical_handle.dim[1] - level0_span))

        return PatchRef(
            image_path=canonical_handle.image_path,
            x=x,
            y=y,
            width=read_size,
            height=read_size,
            read_level=self.config.read_level,
            downsample=int(canonical_handle.level_downsamples[self.config.read_level]),
            mpp_x=canonical_mpp_x,
            mpp_y=canonical_mpp_y,
        )

    def _open(self, record: SlideRecord) -> WSIHandle:
        handle = self.handles.get(record.path)
        if handle is not None:
            return handle

        start = time.perf_counter()
        handle = open_wsi_handle(record.path)
        handle = self._with_scanner_mpp_fallback(handle, record.scanner_id)
        self.handles[record.path] = handle
        self._log(
            f"opened {record.path.name} in {time.perf_counter() - start:.2f}s "
            f"dims={handle.level_dimensions} mpp={handle.mpp}"
        )
        return handle

    def _with_scanner_mpp_fallback(self, handle: WSIHandle, scanner_id: str) -> WSIHandle:
        mpp_x, mpp_y = handle.mpp
        if self._is_plausible_mpp(mpp_x) and self._is_plausible_mpp(mpp_y):
            return handle

        fallback = self.config.scanner_mpp.get(scanner_id)
        if fallback is None:
            if self.config.strict_mpp_check:
                raise ValueError(
                    f"Invalid MPP metadata for {handle.image_path}: {handle.mpp}. "
                    f"No scanner fallback configured for {scanner_id!r}."
                )
            fallback = float(self.config.target_mpp)

        self._log(
            f"using scanner MPP fallback: file={handle.image_path.name} "
            f"scanner={scanner_id} fallback={fallback}"
        )
        return WSIHandle(
            image_path=handle.image_path,
            dim=handle.dim,
            mpp=(fallback, fallback),
            level_dimensions=handle.level_dimensions,
            level_downsamples=handle.level_downsamples,
        )

    def _read_size_for_handle(self, handle: WSIHandle, scanner_id: str) -> int:
        mpp_x, mpp_y = handle.mpp
        if not self._is_plausible_mpp(mpp_x) or not self._is_plausible_mpp(mpp_y):
            fallback = self.config.scanner_mpp.get(scanner_id, self.config.target_mpp)
            mpp_x = mpp_y = float(fallback)
        downsample = float(handle.level_downsamples[self.config.read_level])
        read_level_mpp = ((mpp_x + mpp_y) / 2.0) * downsample
        return self._native_patch_size_for_mpp(read_level_mpp)

    def _native_patch_size_for_mpp(self, mpp: float | None) -> int:
        if mpp is None or mpp <= 0:
            raise ValueError(f"mpp must be > 0, got {mpp}")
        return max(1, int(round(self.config.image_size * float(self.config.target_mpp) / mpp)))

    def _load_normalized_patch(self, ref: PatchRef) -> np.ndarray:
        patch = load_patch(ref).img
        if patch.shape[1] != self.config.image_size or patch.shape[2] != self.config.image_size:
            image = Image.fromarray(np.transpose(patch, (1, 2, 0)), mode="RGB")
            image = image.resize(
                (self.config.image_size, self.config.image_size),
                Image.BILINEAR,
            )
            patch = np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1))

        patch_np = patch.astype(np.float32) / 255.0
        patch_np = (patch_np - 0.5) * 2.0
        return patch_np.astype(np.float32)

    def _cache_path(self) -> Path:
        self.config.patch_cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_dir": str(self.dataset_dir),
            "split_name": self.split_name,
            "sample_ids": self.sample_ids,
            "canonical_domain": self.canonical_domain,
            "source_domains": self.source_domains,
            "image_size": self.config.image_size,
            "target_mpp": self.config.target_mpp,
            "read_level": self.config.read_level,
            "patches_per_source_slide": self.config.patches_per_source_slide,
            "patches_per_target_slide": self.config.patches_per_target_slide,
            "mask_longest_side": self.config.mask_longest_side,
            "scanner_mpp": self.config.scanner_mpp,
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return self.config.patch_cache_dir / f"multistain_{self.split_name}_{digest}.json"

    def _save_patch_cache(self, cache_path: Path) -> None:
        payload = {
            "source_items": [self._item_to_dict(item) for item in self.source_items],
            "target_items": [self._item_to_dict(item) for item in self.target_items],
        }
        tmp_path = cache_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file)
        tmp_path.replace(cache_path)

    def _load_patch_cache(self, cache_path: Path) -> None:
        with cache_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        self.source_items = [
            self._item_from_dict(row) for row in payload.get("source_items", [])
        ]
        self.target_items = [
            self._item_from_dict(row) for row in payload.get("target_items", [])
        ]

    def _item_to_dict(self, item: PatchItem) -> dict[str, object]:
        return {
            "record": {
                "sample_id": item.record.sample_id,
                "scanner_id": item.record.scanner_id,
                "path": str(item.record.path),
            },
            "ref": item.ref.to_dict(),
        }

    def _item_from_dict(self, row: dict[str, object]) -> PatchItem:
        record_data = row["record"]
        ref_data = row["ref"]
        if not isinstance(record_data, dict) or not isinstance(ref_data, dict):
            raise ValueError("Invalid patch cache format.")
        record = SlideRecord(
            sample_id=str(record_data["sample_id"]),
            scanner_id=str(record_data["scanner_id"]),
            path=Path(str(record_data["path"])),
        )
        ref = PatchRef(
            image_path=Path(str(ref_data["image_path"])),
            x=int(ref_data["x"]),
            y=int(ref_data["y"]),
            width=int(ref_data["width"]),
            height=int(ref_data["height"]),
            read_level=int(ref_data["read_level"]),
            downsample=int(ref_data["downsample"]),
            mpp_x=float(ref_data["mpp_x"]),
            mpp_y=float(ref_data["mpp_y"]),
        )
        return PatchItem(record=record, ref=ref)

    def _is_plausible_mpp(self, mpp: float) -> bool:
        return MIN_PLAUSIBLE_MPP <= float(mpp) <= MAX_PLAUSIBLE_MPP

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[MultiStainPatchDataset:{self.split_name}] {message}", flush=True)


def discover_multiscanner_records(
    dataset_dir: str | Path,
    recursive: bool = False,
) -> dict[str, dict[str, SlideRecord]]:
    dataset_dir = Path(dataset_dir)
    pattern = "**/*" if recursive else "*"
    records: dict[str, dict[str, SlideRecord]] = {}

    for path in sorted(dataset_dir.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        parsed = parse_multiscanner_filename(path)
        if parsed is None:
            continue
        sample_id, scanner_id = parsed
        records.setdefault(sample_id, {})[scanner_id] = SlideRecord(
            sample_id=sample_id,
            scanner_id=scanner_id,
            path=path,
        )

    return records


def parse_multiscanner_filename(path: Path) -> tuple[str, str] | None:
    match = re.match(r"^scc_(?P<sample>.+)_(?P<scanner>[^_]+)$", path.stem)
    if match is None:
        return None
    return match.group("sample"), match.group("scanner")


def split_sample_ids(
    dataset_dir: str | Path,
    train_count: int = 36,
    val_count: int = 8,
    seed: int = 0,
    recursive: bool = False,
) -> tuple[list[str], list[str]]:
    """Split by specimen/case so scanner replicas of the same tissue cannot leak."""

    records = discover_multiscanner_records(dataset_dir, recursive=recursive)
    sample_ids = sorted(records)
    if train_count + val_count > len(sample_ids):
        raise ValueError(
            f"Requested {train_count + val_count} samples but found only {len(sample_ids)}."
        )

    rng = random.Random(seed)
    shuffled = sample_ids[:]
    rng.shuffle(shuffled)
    train_ids = sorted(shuffled[:train_count])
    val_ids = sorted(shuffled[train_count:train_count + val_count])
    return train_ids, val_ids


def create_datasets(
    config: MultiStainCycleGANConfig,
) -> tuple[MultiStainPatchDataset, MultiStainPatchDataset]:
    train_ids, val_ids = split_sample_ids(
        config.dataset_dir,
        train_count=config.train_sample_count,
        val_count=config.val_sample_count,
        seed=config.split_seed,
        recursive=config.recursive,
    )
    train_dataset = MultiStainPatchDataset(
        config=config,
        sample_ids=train_ids,
        split_name="train",
        include_aligned_target=False,
    )
    val_dataset = MultiStainPatchDataset(
        config=config,
        sample_ids=val_ids,
        split_name="val",
        include_aligned_target=True,
    )
    return train_dataset, val_dataset


def _stable_int(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
