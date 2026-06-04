from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openslide
from PIL import Image, ImageDraw
import tifffile


# Edit these two paths before running the script.
INPUT_FILE_PATH = Path("/mnt/Disk1/dpsn_datasets/camelyon16/normal_074.tif")
OUTPUT_FILE_PATH = Path("/mnt/Disk1/dpsn_datasets/inf_result_stainnet/normal_074_stainnet.tiff")

# The side-by-side preview will be saved here.
PREVIEW_OUTPUT_PATH = Path(__file__).with_name("see_img_preview.png")

# Maximum preview width/height used for visual comparison.
PREVIEW_MAX_SIZE = 1024


@dataclass(slots=True)
class WSIReport:
    path: Path
    exists: bool
    tifffile_ok: bool
    openslide_ok: bool
    tif_shape: tuple[int, ...] | None
    tif_dtype: str | None
    tif_pages: int | None
    tif_subifds: int | None
    openslide_vendor: str | None
    openslide_level_count: int | None
    openslide_level_dimensions: tuple[tuple[int, int], ...] | None
    preview_shape: tuple[int, int, int] | None
    preview_mean: float | None
    preview_std: float | None
    content_warning: str | None
    warnings: list[str]


def main() -> None:
    print("[see_img] Starting WSI inspection...", flush=True)
    _validate_input_globals()

    input_report = inspect_wsi(INPUT_FILE_PATH)
    output_report = inspect_wsi(OUTPUT_FILE_PATH)

    print_report("INPUT", input_report)
    print_report("OUTPUT", output_report)
    print_comparison_summary(input_report, output_report)

    input_preview = load_preview(INPUT_FILE_PATH, PREVIEW_MAX_SIZE)
    output_preview = load_preview(OUTPUT_FILE_PATH, PREVIEW_MAX_SIZE)
    save_side_by_side_preview(
        input_preview=input_preview,
        output_preview=output_preview,
        output_path=PREVIEW_OUTPUT_PATH,
        input_path=INPUT_FILE_PATH,
        output_path_label=OUTPUT_FILE_PATH,
    )

    print(f"[see_img] Saved side-by-side preview to: {PREVIEW_OUTPUT_PATH}", flush=True)


def _validate_input_globals() -> None:
    for name, path in (
        ("INPUT_FILE_PATH", INPUT_FILE_PATH),
        ("OUTPUT_FILE_PATH", OUTPUT_FILE_PATH),
    ):
        if not isinstance(path, Path):
            raise TypeError(f"{name}은 pathlib.Path 타입이어야 합니다. 입력 타입: {type(path).__name__}")


def inspect_wsi(path: Path) -> WSIReport:
    warnings: list[str] = []

    if not path.exists():
        return WSIReport(
            path=path,
            exists=False,
            tifffile_ok=False,
            openslide_ok=False,
            tif_shape=None,
            tif_dtype=None,
            tif_pages=None,
            tif_subifds=None,
            openslide_vendor=None,
            openslide_level_count=None,
            openslide_level_dimensions=None,
            preview_shape=None,
            preview_mean=None,
            preview_std=None,
            content_warning="file does not exist",
            warnings=["missing file"],
        )

    tifffile_ok = False
    tif_shape = None
    tif_dtype = None
    tif_pages = None
    tif_subifds = None
    try:
        with tifffile.TiffFile(path) as tif:
            tifffile_ok = True
            page0 = tif.pages[0]
            tif_shape = tuple(int(x) for x in page0.shape)
            tif_dtype = str(page0.dtype)
            tif_pages = len(tif.pages)
            tif_subifds = len(page0.pages) if getattr(page0, "pages", None) is not None else 0
            if tif_subifds == 0:
                warnings.append("TIFF page 0 has no SubIFDs; pyramid structure may be absent.")
    except Exception as exc:
        warnings.append(f"tifffile open/read failed: {exc}")

    openslide_ok = False
    openslide_vendor = None
    openslide_level_count = None
    openslide_level_dimensions = None
    try:
        openslide_vendor = openslide.OpenSlide.detect_format(str(path))
        if openslide_vendor:
            with openslide.OpenSlide(str(path)) as slide:
                openslide_ok = True
                openslide_level_count = int(slide.level_count)
                openslide_level_dimensions = tuple(
                    (int(w), int(h)) for (w, h) in slide.level_dimensions
                )
                if slide.level_count < 2:
                    warnings.append("OpenSlide reports only one level; not a full multilevel WSI.")
        else:
            warnings.append("OpenSlide does not recognize this file as a supported WSI.")
    except Exception as exc:
        warnings.append(f"OpenSlide open/read failed: {exc}")

    preview = load_preview(path, PREVIEW_MAX_SIZE)
    preview_shape = tuple(int(x) for x in preview.shape)
    preview_mean = float(preview.mean())
    preview_std = float(preview.std())
    content_warning = analyze_preview_content(preview)
    if content_warning is not None:
        warnings.append(content_warning)

    return WSIReport(
        path=path,
        exists=True,
        tifffile_ok=tifffile_ok,
        openslide_ok=openslide_ok,
        tif_shape=tif_shape,
        tif_dtype=tif_dtype,
        tif_pages=tif_pages,
        tif_subifds=tif_subifds,
        openslide_vendor=openslide_vendor,
        openslide_level_count=openslide_level_count,
        openslide_level_dimensions=openslide_level_dimensions,
        preview_shape=preview_shape,
        preview_mean=preview_mean,
        preview_std=preview_std,
        content_warning=content_warning,
        warnings=warnings,
    )


def load_preview(path: Path, max_size: int) -> np.ndarray:
    vendor = openslide.OpenSlide.detect_format(str(path))
    if vendor:
        with openslide.OpenSlide(str(path)) as slide:
            thumbnail = slide.get_thumbnail((max_size, max_size)).convert("RGB")
            return np.asarray(thumbnail, dtype=np.uint8)

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        array = page.asarray()
    return prepare_preview_array(array, max_size=max_size)


def prepare_preview_array(array: np.ndarray, max_size: int) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.transpose(array, (1, 2, 0))

    if array.ndim != 3:
        raise ValueError(f"shape {array.shape} 배열로 preview를 생성할 수 없습니다.")

    if array.shape[-1] == 4:
        array = array[..., :3]
    elif array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    image = Image.fromarray(_to_uint8(array), mode="RGB")
    image.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array

    array = np.asarray(array, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = (array - min_value) / (max_value - min_value)
    return np.rint(scaled * 255.0).astype(np.uint8)


def analyze_preview_content(preview: np.ndarray) -> str | None:
    preview = np.asarray(preview, dtype=np.float32)
    mean_value = float(preview.mean())
    std_value = float(preview.std())

    if std_value < 3.0:
        return "Preview has extremely low contrast; image may be blank or constant."
    if mean_value < 5.0:
        return "Preview is almost entirely black."
    if mean_value > 250.0:
        return "Preview is almost entirely white."
    return None


def print_report(title: str, report: WSIReport) -> None:
    print(f"\n===== {title} REPORT =====", flush=True)
    print(f"path: {report.path}", flush=True)
    print(f"exists: {report.exists}", flush=True)
    print(f"tifffile_ok: {report.tifffile_ok}", flush=True)
    print(f"openslide_ok: {report.openslide_ok}", flush=True)
    print(f"tif_shape: {report.tif_shape}", flush=True)
    print(f"tif_dtype: {report.tif_dtype}", flush=True)
    print(f"tif_pages: {report.tif_pages}", flush=True)
    print(f"tif_subifds: {report.tif_subifds}", flush=True)
    print(f"openslide_vendor: {report.openslide_vendor}", flush=True)
    print(f"openslide_level_count: {report.openslide_level_count}", flush=True)
    print(f"openslide_level_dimensions: {report.openslide_level_dimensions}", flush=True)
    print(f"preview_shape: {report.preview_shape}", flush=True)
    print(f"preview_mean: {report.preview_mean}", flush=True)
    print(f"preview_std: {report.preview_std}", flush=True)
    print(f"content_warning: {report.content_warning}", flush=True)
    if report.warnings:
        print("warnings:", flush=True)
        for warning in report.warnings:
            print(f"  - {warning}", flush=True)
    else:
        print("warnings: none", flush=True)


def print_comparison_summary(input_report: WSIReport, output_report: WSIReport) -> None:
    print("\n===== COMPARISON SUMMARY =====", flush=True)
    if not input_report.exists or not output_report.exists:
        print("Cannot compare because one or both files do not exist.", flush=True)
        return

    if input_report.openslide_level_dimensions and output_report.openslide_level_dimensions:
        print(
            "openslide level-0 dimensions: "
            f"input={input_report.openslide_level_dimensions[0]} "
            f"output={output_report.openslide_level_dimensions[0]}",
            flush=True,
        )
    elif input_report.tif_shape and output_report.tif_shape:
        print(
            f"tif page-0 shape: input={input_report.tif_shape} output={output_report.tif_shape}",
            flush=True,
        )

    if input_report.openslide_level_count is not None and output_report.openslide_level_count is not None:
        print(
            f"multilevel status: input_levels={input_report.openslide_level_count}, "
            f"output_levels={output_report.openslide_level_count}",
            flush=True,
        )

    if input_report.preview_mean is not None and output_report.preview_mean is not None:
        print(
            f"preview mean intensity: input={input_report.preview_mean:.2f}, "
            f"output={output_report.preview_mean:.2f}",
            flush=True,
        )
    if input_report.preview_std is not None and output_report.preview_std is not None:
        print(
            f"preview contrast(std): input={input_report.preview_std:.2f}, "
            f"output={output_report.preview_std:.2f}",
            flush=True,
        )


def save_side_by_side_preview(
    input_preview: np.ndarray,
    output_preview: np.ndarray,
    output_path: Path,
    input_path: Path,
    output_path_label: Path,
) -> None:
    input_image = Image.fromarray(input_preview, mode="RGB")
    output_image = Image.fromarray(output_preview, mode="RGB")

    target_height = max(input_image.height, output_image.height)
    input_image = pad_to_height(input_image, target_height)
    output_image = pad_to_height(output_image, target_height)

    header_height = 50
    canvas = Image.new(
        "RGB",
        (input_image.width + output_image.width, target_height + header_height),
        color=(255, 255, 255),
    )
    canvas.paste(input_image, (0, header_height))
    canvas.paste(output_image, (input_image.width, header_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((16, 16), f"INPUT: {input_path.name}", fill=(0, 0, 0))
    draw.text((input_image.width + 16, 16), f"OUTPUT: {output_path_label.name}", fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def pad_to_height(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image
    canvas = Image.new("RGB", (image.width, target_height), color=(255, 255, 255))
    y = (target_height - image.height) // 2
    canvas.paste(image, (0, y))
    return canvas


if __name__ == "__main__":
    main()
