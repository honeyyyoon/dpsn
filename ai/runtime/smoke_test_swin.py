"""
Run example:
./.venv/bin/python -m ai.runtime.smoke_test_swin

Output:
1. output .zarr
2. input_thumbnail.png
3. output_thumbnail.png
4. metrics.txt
5. smoke_test_swin.log
"""

from __future__ import annotations

from pathlib import Path
import shutil

from PIL import Image

from ai.pipelines.stainswin import StainSWINInferenceConfig, StainSWINPipeline
from ai.runtime.task import Task
from ai.runtime.worker import Worker
from ai.wsi.loader import load_patch, open_wsi_handle


# Edit these before running.
INPUT_WSI_PATH = Path("/absolute/path/to/input_wsi.tiff")
OUTPUT_ZARR_PATH = Path("/absolute/path/to/output/slide_stainswin.zarr")
CHECKPOINT_PATH: Path | None = None


class SmokeTestWorker(Worker):
    def __init__(
        self,
        output_zarr_path: Path,
        checkpoint_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.output_zarr_path = Path(output_zarr_path)
        self.checkpoint_path = checkpoint_path

    def _create_pipeline(self, model_id: int):
        if model_id != 6:
            return super()._create_pipeline(model_id)

        config = StainSWINInferenceConfig(
            checkpoint_path=self.checkpoint_path,
            verbose=True,
        )
        log_path = self.output_zarr_path.parent / "smoke_test_swin.log"
        return StainSWINPipeline(self._build_logger(log_path), config=config)


def main() -> None:
    _validate_paths()

    output_dir = OUTPUT_ZARR_PATH.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    task = Task(
        src_img_path=INPUT_WSI_PATH,
        target_img_path=None,
        result_path=OUTPUT_ZARR_PATH,
        model_id=6,
    )
    worker = SmokeTestWorker(
        output_zarr_path=OUTPUT_ZARR_PATH,
        checkpoint_path=CHECKPOINT_PATH,
    )

    print("[smoke_test_swin] Starting Worker.run() for StainSWIN...", flush=True)
    result = worker.run(task)
    generated_output_zarr = Path(result.result_img_path)
    generated_output_thumbnail = generated_output_zarr.parent / "out_image.png"

    final_output_zarr = OUTPUT_ZARR_PATH
    final_output_thumbnail = output_dir / "output_thumbnail.png"
    final_input_thumbnail = output_dir / "input_thumbnail.png"
    metrics_txt_path = output_dir / "metrics.txt"

    _move_zarr_output(generated_output_zarr, final_output_zarr)
    if generated_output_thumbnail.exists():
        shutil.copy2(generated_output_thumbnail, final_output_thumbnail)

    _save_input_thumbnail(INPUT_WSI_PATH, final_input_thumbnail)
    _write_metrics_text(
        metrics_txt_path=metrics_txt_path,
        input_wsi_path=INPUT_WSI_PATH,
        output_zarr_path=final_output_zarr,
        input_thumbnail_path=final_input_thumbnail,
        output_thumbnail_path=final_output_thumbnail,
        ssim=result.metrics.ssim,
        psnr=result.metrics.psnr,
        fid=result.metrics.fid,
    )

    print("[smoke_test_swin] Done.", flush=True)
    print(f"input_wsi_path: {INPUT_WSI_PATH}", flush=True)
    print(f"output_zarr_path: {final_output_zarr}", flush=True)
    print(f"input_thumbnail_path: {final_input_thumbnail}", flush=True)
    print(f"output_thumbnail_path: {final_output_thumbnail}", flush=True)
    print(f"metrics_txt_path: {metrics_txt_path}", flush=True)


def _validate_paths() -> None:
    if not INPUT_WSI_PATH.is_file():
        raise FileNotFoundError(f"Input WSI not found: {INPUT_WSI_PATH}")
    if CHECKPOINT_PATH is not None and not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if OUTPUT_ZARR_PATH.suffix != ".zarr":
        raise ValueError(
            f"OUTPUT_ZARR_PATH must end with .zarr, got {OUTPUT_ZARR_PATH}"
        )


def _move_zarr_output(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Generated output Zarr not found: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    if src.resolve() == dst.resolve():
        return
    shutil.move(str(src), str(dst))


def _save_input_thumbnail(input_wsi_path: Path, thumbnail_path: Path) -> None:
    wsi_handle = open_wsi_handle(input_wsi_path)
    thumb_ref = wsi_handle.make_ref((0, 0), wsi_handle.max_level)
    thumb_patch = load_patch(thumb_ref)
    thumb_hwc = thumb_patch.img.transpose(1, 2, 0)
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(thumb_hwc, mode="RGB").save(thumbnail_path)


def _write_metrics_text(
    metrics_txt_path: Path,
    input_wsi_path: Path,
    output_zarr_path: Path,
    input_thumbnail_path: Path,
    output_thumbnail_path: Path,
    ssim: float,
    psnr: float,
    fid: float,
) -> None:
    metrics_txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"input_wsi_path: {input_wsi_path}",
        f"output_zarr_path: {output_zarr_path}",
        f"input_thumbnail_path: {input_thumbnail_path}",
        f"output_thumbnail_path: {output_thumbnail_path}",
        f"ssim: {ssim}",
        f"psnr: {psnr}",
        f"fid: {fid}",
    ]
    metrics_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
