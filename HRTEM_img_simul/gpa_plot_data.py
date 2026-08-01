# plot GPA data previously calculated and exported from Strain++ TIFF files

from matplotlib import pyplot as plt
plt.style.use('mypub')
import numpy as np
import tifffile
from pathlib import Path


def patch_from_mask_tiff(mask_tiff: Path, ax) -> tuple[float, float, float]:
    """Extract the center and radius of the circular mask from the masked FFT TIFF."""
    masked_fft = tifffile.imread(mask_tiff)
    y, x = np.where(np.abs(masked_fft) > 0)
    if len(x) == 0 or len(y) == 0:
        raise ValueError(f"No non-zero pixels found in {mask_tiff}")
    center_x, center_y = np.mean(x), np.mean(y)
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).max()
    circle = plt.Circle((center_x, center_y), radius, color='red', fill=False, linewidth=1)
    ax.add_patch(circle)
    return center_x, center_y, radius


gpa_folder = Path('/mnt/c/Users/a.walrave/Documents/M2 Internship & PhD/DataTreatment/HRTEM Titan/2024-05-14/ZnO_0001Zn_P3_HRTEM07_crop_311x1139_rot+162.0deg/')

# UNCOMMENT IF MULTIPLE FOLDERS    + 2tabs
# for dir in Path("/mnt/c/Users/a.walrave/Documents/M2 Internship & PhD/DataTreatment/HRTEM Titan/image_simulation").iterdir():
#     if dir.is_dir() and dir.name.startswith("corepos"):
#         print(dir)
#         print(f"Working on: {dir}")
#         gpa_folder = dir / "gpa"

if gpa_folder.exists():
    img = tifffile.imread(gpa_folder / "image.tif")
    fft = tifffile.imread(gpa_folder / "FFT.tif")

    gpa_files = list(gpa_folder.glob("e*.tif"))
    print(f"Found {len(gpa_files)} GPA files:")

    fig, axs = plt.subplots(1, 2+len(gpa_files), figsize=(2*(2+len(gpa_files)), 2.5))

    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(fft, cmap="gray")
    # plus, plot circles with radii corresponding to the masked FFTs
    for g in [1, 2]:
        masked_fft = gpa_folder / f"Phase {g} Masked FFT.tif"
        patch_from_mask_tiff(masked_fft, axs[1])
    axs[1].set_title("FFT")
    axs[1].axis("off")

    for i, f in enumerate(gpa_files):
        img = tifffile.imread(f)
        axs[2+i].imshow(img, cmap="plasma", vmin=np.percentile(img, 5), vmax=np.percentile(img, 95))        # cmap="RdBu"
        axs[2+i].set_title(f.stem)
        axs[2+i].axis("off")

    plt.tight_layout()
    plt.savefig(gpa_folder / "../gpa_distortion.png", dpi=300, facecolor="white")
