from __future__ import annotations

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

from ai.samplers.patch_sampler import PatchSampler
from ai.wsi.handle import WSIHandle
from ai.wsi.loader import load_patch, open_wsi_handle
from ai.wsi.patch_ref import PatchRef


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_SCANNER_MPP = {
    "cs2": 0.25,
    "nz210": 0.22,
    "nz20": 0.23,
    "p1000": 0.25,
    "gt450": 0.26,
}


@dataclass(frozen=True, slots=True)
class SlideRecord:
    sample_id: str
    scanner_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class StainGANPatchSample:
    source: np.ndarray
    canonical: np.ndarray
    identity: np.ndarray
    source_domain: str
    target_domain: str
    sample_id: str
    source_path: str
    canonical_path: str


class MultiDomainWSIPatchDataset(Dataset):
    """
    Many-source-to-one StainGAN dataset.

    A training item contains:
    - a tissue patch from one non-canonical scanner
    - the same physical location from the canonical scanner for that sample
    - a canonical-domain patch for identity loss
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        canonical_domain: str = "nz210",
        sample_ids: list[str] | None = None,
        image_size: int = 256,
        target_mpp: float = 0.25,
        read_level: int = 0,
        patches_per_source_slide: int = 128,
        strict_mpp_check: bool = True,
        recursive: bool = False,
        seed: int = 0,
        scanner_mpp: dict[str, float] | None = None,
        sampler_result_dir: str | Path = "result/staingan_patch_sampler",
        verbose: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.canonical_domain = canonical_domain
        self.image_size = int(image_size)
        self.target_mpp = float(target_mpp)
        self.read_level = int(read_level)
        self.patches_per_source_slide = int(patches_per_source_slide)
        self.strict_mpp_check = bool(strict_mpp_check)
        self.recursive = bool(recursive)
        self.seed = int(seed)
        self.scanner_mpp = {**DEFAULT_SCANNER_MPP, **(scanner_mpp or {})}
        self.verbose = bool(verbose)

        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")
        if self.target_mpp <= 0:
            raise ValueError(f"target_mpp must be > 0, got {self.target_mpp}")
        if self.patches_per_source_slide <= 0:
            raise ValueError(
                f"patches_per_source_slide must be > 0, got {self.patches_per_source_slide}"
            )

        self.records_by_sample = self._discover_records()
        self._log(
            f"Discovered {len(self.records_by_sample)} sample(s) in {self.dataset_dir}"
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
            raise ValueError("No samples found for the requested split.")
        self._log(
            f"Using {len(self.sample_ids)} sample(s): {', '.join(self.sample_ids[:8])}"
            + (" ..." if len(self.sample_ids) > 8 else "")
        )

        self.handles: dict[Path, WSIHandle] = {}
        self.source_items: list[tuple[SlideRecord, PatchRef]] = []
        self.canonical_records: dict[str, SlideRecord] = {}

        sampler = PatchSampler(
            patch_size=self._native_patch_size_for_mpp(self.target_mpp),
            stride=self._native_patch_size_for_mpp(self.target_mpp),
            read_level=self.read_level,
            strict_mpp_check=self.strict_mpp_check,
            result_dir=sampler_result_dir,
            verbose=self.verbose,
        )

        sample_iter = tqdm(
            self.sample_ids,
            desc="StainGAN dataset sample scan",
            unit="sample",
            leave=False,
            disable=not self.verbose,
        )
        for sample_id in sample_iter:
            records = self.records_by_sample[sample_id]
            canonical = records.get(self.canonical_domain)
            if canonical is None:
                raise ValueError(
                    f"Sample {sample_id} does not have canonical domain {self.canonical_domain!r}"
                )
            self.canonical_records[sample_id] = canonical

            for scanner_id, record in sorted(records.items()):
                if scanner_id == self.canonical_domain:
                    continue
                self._log(
                    f"Sampling patches: sample={sample_id} source={scanner_id} "
                    f"target={self.canonical_domain} path={record.path.name}"
                )
                handle = self._open(record)
                patch_size = self._read_size_for_handle(handle, scanner_id)
                sampler.patch_size = patch_size
                sampler.stride = patch_size
                refs = sampler.sample(
                    handle,
                    mode="training",
                    max_patches=self.patches_per_source_slide,
                    seed=self.seed + len(self.source_items),
                    save_debug=False,
                )
                self.source_items.extend((record, ref) for ref in refs)
                self._log(
                    f"  sampled {len(refs)} patch(es), total_source_patches={len(self.source_items)}"
                )

        if not self.source_items:
            raise ValueError("No non-canonical source patches were sampled.")
        self._log(f"Finished dataset initialization with {len(self.source_items)} source patch item(s).")

    @property
    def source_domains(self) -> list[str]:
        domains = {
            record.scanner_id
            for record, _ in self.source_items
            if record.scanner_id != self.canonical_domain
        }
        return sorted(domains)

    @property
    def scanner_ids(self) -> list[str]:
        domains: set[str] = set()
        for records in self.records_by_sample.values():
            domains.update(records)
        return sorted(domains)

    def __len__(self) -> int:
        return len(self.source_items)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str]:
        source_record, source_ref = self.source_items[index % len(self.source_items)]
        canonical_record = self.canonical_records[source_record.sample_id]
        source_handle = self._open(source_record)
        canonical_handle = self._open(canonical_record)

        canonical_ref = self._paired_ref(
            source_ref=source_ref,
            source_handle=source_handle,
            source_scanner=source_record.scanner_id,
            canonical_handle=canonical_handle,
            canonical_scanner=canonical_record.scanner_id,
        )
        return {
            "source": self._load_normalized_patch(source_ref),
            "canonical": self._load_normalized_patch(canonical_ref),
            "identity": self._load_normalized_patch(canonical_ref),
            "source_domain": source_record.scanner_id,
            "target_domain": canonical_record.scanner_id,
            "sample_id": source_record.sample_id,
            "source_path": str(source_record.path),
            "canonical_path": str(canonical_record.path),
        }

    def _discover_records(self) -> dict[str, dict[str, SlideRecord]]:
        pattern = "**/*" if self.recursive else "*"
        records: dict[str, dict[str, SlideRecord]] = {}
        for path in sorted(self.dataset_dir.glob(pattern)):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            parsed = self._parse_name(path)
            if parsed is None:
                continue
            sample_id, scanner_id = parsed
            records.setdefault(sample_id, {})[scanner_id] = SlideRecord(
                sample_id=sample_id,
                scanner_id=scanner_id,
                path=path,
            )
        return records

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[MultiDomainWSIPatchDataset] {message}", flush=True)

    def _parse_name(self, path: Path) -> tuple[str, str] | None:
        match = re.match(r"^scc_(?P<sample>.+)_(?P<scanner>[^_]+)$", path.stem)
        if match is None:
            return None
        return match.group("sample"), match.group("scanner")

    def _open(self, record: SlideRecord) -> WSIHandle:
        handle = self.handles.get(record.path)
        if handle is None:
            self._log(f"Opening WSI: scanner={record.scanner_id} path={record.path.name}")
            start = time.perf_counter()
            handle = open_wsi_handle(record.path)
            handle = self._with_scanner_mpp_fallback(handle, record.scanner_id)
            self.handles[record.path] = handle
            self._log(
                f"Opened {record.path.name} in {time.perf_counter() - start:.2f}s "
                f"dims={handle.level_dimensions} downsamples={handle.level_downsamples} "
                f"mpp={handle.mpp}"
            )
        return handle

    def _with_scanner_mpp_fallback(self, handle: WSIHandle, scanner_id: str) -> WSIHandle:
        mpp_x, mpp_y = handle.mpp
        if mpp_x > 0 and mpp_y > 0:
            return handle
        fallback = self.scanner_mpp.get(scanner_id)
        if fallback is None:
            if self.strict_mpp_check:
                raise ValueError(
                    f"Missing MPP metadata for {handle.image_path} and no fallback for {scanner_id!r}"
                )
            fallback = self.target_mpp
        return WSIHandle(
            image_path=handle.image_path,
            dim=handle.dim,
            mpp=(fallback, fallback),
            level_dimensions=handle.level_dimensions,
            level_downsamples=handle.level_downsamples,
        )

    def _native_patch_size_for_mpp(self, mpp: float) -> int:
        return max(1, int(round(self.image_size * self.target_mpp / mpp)))

    def _read_size_for_handle(self, handle: WSIHandle, scanner_id: str) -> int:
        mpp = self._read_level_mpp(handle, scanner_id)
        return self._native_patch_size_for_mpp(mpp)

    def _read_level_mpp(self, handle: WSIHandle, scanner_id: str) -> float:
        mpp_x, mpp_y = handle.mpp
        if mpp_x <= 0 or mpp_y <= 0:
            fallback = self.scanner_mpp.get(scanner_id, self.target_mpp)
            mpp_x = mpp_y = fallback
        downsample = float(handle.level_downsamples[self.read_level])
        return ((mpp_x + mpp_y) / 2.0) * downsample

    def _paired_ref(
        self,
        source_ref: PatchRef,
        source_handle: WSIHandle,
        source_scanner: str,
        canonical_handle: WSIHandle,
        canonical_scanner: str,
    ) -> PatchRef:
        source_mpp_x, source_mpp_y = source_handle.mpp
        canonical_mpp_x, canonical_mpp_y = canonical_handle.mpp
        read_size = self._read_size_for_handle(canonical_handle, canonical_scanner)
        level0_span = int(round(read_size * float(canonical_handle.level_downsamples[self.read_level])))
        x = int(round(source_ref.x * source_mpp_x / canonical_mpp_x))
        y = int(round(source_ref.y * source_mpp_y / canonical_mpp_y))
        x = max(0, min(x, canonical_handle.dim[0] - level0_span))
        y = max(0, min(y, canonical_handle.dim[1] - level0_span))
        return PatchRef(
            image_path=canonical_handle.image_path,
            x=x,
            y=y,
            width=read_size,
            height=read_size,
            read_level=self.read_level,
            downsample=int(canonical_handle.level_downsamples[self.read_level]),
            mpp_x=canonical_mpp_x,
            mpp_y=canonical_mpp_y,
        )

    def _random_canonical_ref(
        self,
        canonical_handle: WSIHandle,
        canonical_scanner: str,
        index: int,
    ) -> PatchRef:
        read_size = self._read_size_for_handle(canonical_handle, canonical_scanner)
        level0_span = int(round(read_size * float(canonical_handle.level_downsamples[self.read_level])))
        rng = random.Random(self.seed + index)
        max_x = max(0, canonical_handle.dim[0] - level0_span)
        max_y = max(0, canonical_handle.dim[1] - level0_span)
        return PatchRef(
            image_path=canonical_handle.image_path,
            x=rng.randint(0, max_x),
            y=rng.randint(0, max_y),
            width=read_size,
            height=read_size,
            read_level=self.read_level,
            downsample=int(canonical_handle.level_downsamples[self.read_level]),
            mpp_x=canonical_handle.mpp[0],
            mpp_y=canonical_handle.mpp[1],
        )

    def _load_normalized_patch(self, ref: PatchRef) -> np.ndarray:
        patch = load_patch(ref).img
        if patch.shape[1] != self.image_size or patch.shape[2] != self.image_size:
            image = Image.fromarray(np.transpose(patch, (1, 2, 0)), mode="RGB")
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            patch = np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1))
        patch_np = patch.astype(np.float32) / 255.0
        patch_np = (patch_np - 0.5) * 2.0
        return patch_np.astype(np.float32)


def split_sample_ids(
    dataset_dir: str | Path,
    train_count: int = 36,
    val_count: int = 8,
    seed: int = 0,
    recursive: bool = False,
) -> tuple[list[str], list[str]]:
    dataset_dir = Path(dataset_dir)
    pattern = "**/*" if recursive else "*"
    sample_ids = sorted(
        {
            match.group("sample")
            for path in dataset_dir.glob(pattern)
            if path.is_file()
            for match in [re.match(r"^scc_(?P<sample>.+)_(?P<scanner>[^_]+)$", path.stem)]
            if match is not None
        }
    )
    rng = random.Random(seed)
    shuffled = sample_ids[:]
    rng.shuffle(shuffled)
    train_ids = sorted(shuffled[:train_count])
    val_ids = sorted(shuffled[train_count:train_count + val_count])
    return train_ids, val_ids
