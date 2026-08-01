import os
import sys
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
from dislo_detect import load_dm3


if __name__ == "__main__":
    dm3_path = Path(r"/mnt/c/Users/a.walrave/Documents/M2 Internship & PhD/DataTreatment/HRTEM Titan/2024-05-14/ABSF Filtered ZnO_0001Zn_P3_HRTEM13.dm3")
    out_ext = sys.argv[1] if len(sys.argv) > 1 else "tiff"
    tiff_rotation_deg = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0

    output_path = dm3_path.with_suffix(f".{out_ext}")

    # create out folder if doesnt exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img, pixel_size, unit_str = load_dm3(dm3_path)
    print(f"      Shape: {img.shape}  |  pixel size: {pixel_size:.4f} {unit_str}")

    # img_array = np.flipud(img.T)
    assert img.ndim == 2, f"Expected 2D image, got shape {img.shape}"
    img_norm = (img - img.min()) / max(img.max() - img.min(), 1e-8)

    if out_ext.lower() in ["tif", "tiff"]:
        img_for_tiff = img_norm

        # Rotate on float data to preserve dynamic range, padding uncovered areas with zeros.
        if tiff_rotation_deg % 360 != 0:
            img_f32_pil = Image.fromarray(img_for_tiff.astype(np.float32), mode="F")
            img_for_tiff = np.array(
                img_f32_pil.rotate(
                    tiff_rotation_deg,
                    resample=Image.Resampling.BILINEAR,
                    expand=True,
                    fillcolor=0.0,
                ),
                dtype=np.float32,
            )

        img_16 = np.clip(img_for_tiff, 0.0, 1.0)
        img_16 = (img_16 * 65535).astype(np.uint16)

        tifffile.imwrite(
            output_path, img_16,
            imagej=True,
            resolution=(1/pixel_size, 1/pixel_size),
            metadata={"unit": unit_str},
        )

    elif out_ext.lower() == "png":
        # increase contrast by percentiles
        p2, p98 = np.percentile(img_norm, (2, 98))
        img_norm = np.clip((img_norm - p2) / (p98 - p2), 0, 1)
        img_8 = (img_norm * 255).astype(np.uint8)
        img_8 = Image.fromarray(img_8)
        img_8.save(output_path.with_name(output_path.stem + "_preview.png"))

