from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ai.models.staingan.multidomain_wsi_dataset import (
    MultiDomainWSIPatchDataset,
    SlideRecord,
    split_sample_ids,
)
from ai.pipelines.staingan import StainGANInferenceConfig, StainGANPipeline
from ai.wsi.loader import load_patch


def chw_to_pil(patch: np.ndarray) -> Image.Image:
    return Image.fromarray(np.transpose(patch, (1, 2, 0)), mode="RGB")


def resize_chw_uint8(patch: np.ndarray, image_size: int) -> np.ndarray:
    if patch.shape[1] == image_size and patch.shape[2] == image_size:
        return patch
    image = chw_to_pil(patch)
    image = image.resize((image_size, image_size), Image.BILINEAR)
    return np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1))


def filter_source_items(
    dataset: MultiDomainWSIPatchDataset,
    source_domains: set[str],
) -> None:
    if not source_domains:
        return
    dataset.source_items = [
        item
        for item in dataset.source_items
        if item[0].scanner_id in source_domains
    ]
    if not dataset.source_items:
        raise ValueError(
            "No source patches left after filtering to source domains: "
            f"{sorted(source_domains)}"
        )


def build_dataset(
    args: argparse.Namespace,
    sample_ids: list[str],
    split_name: str,
    patches_per_source_slide: int,
    source_domains: set[str],
) -> MultiDomainWSIPatchDataset:
    dataset = MultiDomainWSIPatchDataset(
        dataset_dir=args.dataset_dir,
        canonical_domain=args.canonical_domain,
        sample_ids=sample_ids,
        image_size=args.image_size,
        target_mpp=args.target_mpp,
        read_level=args.read_level,
        patches_per_source_slide=patches_per_source_slide,
        mask_longest_side=args.mask_longest_side,
        strict_mpp_check=args.strict_mpp_check,
        recursive=args.recursive,
        seed=args.split_seed + (0 if split_name == "train" else 10_000),
        sampler_result_dir=args.output_dir / "_sampler_logs" / split_name,
        cache_dir=args.patch_cache_dir / split_name,
        use_patch_cache=args.use_patch_cache,
        verbose=args.verbose,
    )
    filter_source_items(dataset, source_domains)
    return dataset


def export_split(
    dataset: MultiDomainWSIPatchDataset,
    staingan: StainGANPipeline,
    output_dir: Path,
    split_name: str,
    image_size: int,
    batch_size: int,
) -> None:
    source_dir = output_dir / split_name / "source"
    target_dir = output_dir / split_name / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset.source_items)
    print(f"Exporting {split_name}: {total} pseudo-pair(s)", flush=True)

    for start in range(0, total, batch_size):
        batch_items = dataset.source_items[start:start + batch_size]
        batch_patches = [
            resize_chw_uint8(load_patch(ref).img, image_size)
            for _, ref in batch_items
        ]
        normalized_patches = staingan._normalize_patch_batch(batch_patches)

        for offset, ((record, ref), source_patch, target_patch) in enumerate(
            zip(batch_items, batch_patches, normalized_patches)
        ):
            index = start + offset
            pair_key = (
                f"{record.sample_id}_{record.scanner_id}_"
                f"x{ref.x}_y{ref.y}_{index:06d}"
            )
            chw_to_pil(source_patch).save(source_dir / f"A{pair_key}.png")
            chw_to_pil(target_patch).save(target_dir / f"H{pair_key}.png")

        processed = min(start + len(batch_items), total)
        print(f"  {split_name}: {processed}/{total}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate StainNet pseudo-pairs using a trained StainGAN teacher."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--staingan-checkpoint-path", type=Path, default=None)
    parser.add_argument("--staingan-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--canonical-domain", type=str, default="nz210")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-mpp", type=float, default=0.25)
    parser.add_argument("--read-level", type=int, default=0)
    parser.add_argument("--patches-per-source-slide", type=int, default=128)
    parser.add_argument("--val-patches-per-source-slide", type=int, default=32)
    parser.add_argument("--mask-longest-side", type=int, default=1024)
    parser.add_argument(
        "--patch-cache-dir",
        type=Path,
        default=Path("/mnt/Disk1/dpsn_patch_cache/stainnet_staingan_pseudopairs"),
    )
    parser.add_argument("--use-patch-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-sample-count", type=int, default=36)
    parser.add_argument("--val-sample-count", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-mpp-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-source-domains", type=str, nargs="*", default=[])
    parser.add_argument("--val-source-domains", type=str, nargs="*", default=[])
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    train_ids, val_ids = split_sample_ids(
        args.dataset_dir,
        train_count=args.train_sample_count,
        val_count=args.val_sample_count,
        seed=args.split_seed,
        recursive=args.recursive,
    )
    if not train_ids:
        raise ValueError("Training split is empty.")
    if not val_ids:
        raise ValueError("Validation split is empty.")

    train_dataset = build_dataset(
        args=args,
        sample_ids=train_ids,
        split_name="train",
        patches_per_source_slide=args.patches_per_source_slide,
        source_domains=set(args.train_source_domains),
    )
    val_dataset = build_dataset(
        args=args,
        sample_ids=val_ids,
        split_name="val",
        patches_per_source_slide=args.val_patches_per_source_slide,
        source_domains=set(args.val_source_domains),
    )

    staingan_config = StainGANInferenceConfig(
        checkpoint_path=args.staingan_checkpoint_path,
        checkpoint_dir=(
            args.staingan_checkpoint_dir
            if args.staingan_checkpoint_dir is not None
            else StainGANInferenceConfig.checkpoint_dir
        ),
        patch_size=args.image_size,
        stride=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        verbose=args.verbose,
    )
    staingan = StainGANPipeline(logger=None, config=staingan_config)

    export_split(
        dataset=train_dataset,
        staingan=staingan,
        output_dir=args.output_dir,
        split_name="train",
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    export_split(
        dataset=val_dataset,
        staingan=staingan,
        output_dir=args.output_dir,
        split_name="val",
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    print(f"Saved pseudo-pairs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
