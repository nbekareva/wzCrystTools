import argparse
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")           # headless-safe
import matplotlib.patches as patches
from _hrtem_helpers import load_dm3, power_spectrum, auto_detect_g_vectors, manual_pick_g_vectors, \
                    mask_radius_from_g, make_mask, apply_mask_and_ifft, save_array_as_tiff

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # p.add_argument("dm3_file", help="Path to the .dm3 HRTEM image")
    p.add_argument("--mask-radius", type=float, default=0.25,
                   help="Band-pass mask radius in g fraction (default: 0.25 = 1/4)")
    # p.add_argument("--smooth-sigma", type=float, default=3.0,
    #                help="Gaussian σ (pixels) before thresholding (default: 3)")
    p.add_argument("--manual-g", action="store_true",
                   help="Interactively pick g-vectors instead of auto-detection")
    return p.parse_args()

def save_analysis_panneau(img, log_ps, peak_px, mask_multi, ifft1, ifft2, out_folder, pixel_size, unit_str, mask_radius):

    H, W = img.shape
    extent_img = [0, W * pixel_size, H * pixel_size, 0]   # physical coords

    # ── Layout ────────────────────────────────────────────────────────────────
    # Each cell in the mosaic is one equal "tile".
    # "hrtem" spans a 2×2 block  → 4× the area of one tile  (HRTEM large)
    # "ps" and "curl" each occupy 1×1                        (4× smaller)
    # "b1" / "b2"  each occupy 1×1  (unchanged relative size)
    # "cores" spans the full bottom row (4 tiles wide)
    layout = [
        ["hrtem", "g1_ifft"],
        ["hrtem", "g1_ifft"],
        ["ps",    "g2_ifft"],
        ["mask",  "g2_ifft"],
    ]
    fig, axd = plt.subplot_mosaic(
        layout,
        figsize=(10, 10),
        # gridspec_kw={"hspace": 0.38, "wspace": 0.30},
    )

    # ── 1. Raw HRTEM image (large) ────────────────────────────────────────────
    ax = axd["hrtem"]
    ax.imshow(img, cmap="gray",
              vmin=np.percentile(img, 3), vmax=np.percentile(img, 97),
              origin="upper", extent=extent_img)
    ax.set_title("HRTEM image", fontsize=11)
    ax.set_xlabel(f"x [{unit_str}]")
    ax.set_ylabel(f"y [{unit_str}]")

    # ── 2. Power spectrum (small) ─────────────────────────────────────────────
    ax = axd["ps"]
    ax.imshow(log_ps, cmap="inferno", origin="upper")
    colours = ["cyan", "lime"]
    labels  = ["g₁", "g₂"]
    for (r, c), col, lbl in zip(peak_px, colours, labels):
        circ = patches.Circle((c, r), radius=mask_radius, linewidth=1,
                               edgecolor=col, facecolor="none")
        ax.add_patch(circ)
        ax.text(c + 8, r, lbl, color=col, fontsize=8)
    ax.set_title("Power spectrum\n(log)", fontsize=9)
    ax.axis("off")

    # ── 3. Mask ─────────────────────────────────────────────
    ax = axd["mask"]
    ax.imshow(mask_multi, cmap="gray", origin="upper")
    ax.set_title("Mask", fontsize=9)
    ax.axis("off")

    # ── 4. IFFTs ─────────────────────────────────────────────
    ax = axd["g1_ifft"]
    ax.imshow(np.abs(ifft1), cmap="gray", origin="upper")
    ax.set_title("g₁ IFFT", fontsize=9)
    ax.axis("off")

    ax = axd["g2_ifft"]
    ax.imshow(np.abs(ifft2), cmap="gray", origin="upper")
    ax.set_title("g₂ IFFT", fontsize=9)
    ax.axis("off")

    plt.savefig(os.path.join(out_folder, "masking.png"), dpi=300, bbox_inches="tight")



if __name__ == "__main__":
    args = parse_args()

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    folder = '/mnt/c/Users/a.walrave/Documents/M2 Internship & PhD/DATA/HRTEM Titan/2024-05-14'
    dm3_file = 'ZnO_0001Zn_P3_HRTEM07.dm3'

    out_folder = '/mnt/c/Users/a.walrave/Documents/M2 Internship & PhD/DataTreatment/HRTEM Titan/2024-05-14/ZnO_0001Zn_P3_HRTEM07_py_ifft'
    sub_out_folder = os.path.join(out_folder, f"mask_{args.mask_radius:.2f}g")

    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(sub_out_folder, exist_ok=True)



    # ── 1. Load image ──────────────────────────────────────────────────────────
    dm3_path = f"{folder}/{dm3_file}"
    img, pixel_size, unit_str = load_dm3(dm3_path)
    logger.info("Loaded DM3 file: %s \n Shape: %s  |  pixel size: %.4f %s", dm3_path, img.shape, pixel_size, unit_str)

    # ── 2. Power spectrum ──────────────────────────────────────────────────────
    log_ps, F_shifted = power_spectrum(img, hann_window=False)
    logger.debug("Computed power spectrum.")

    # ── 3. G-vector detection ──────────────────────────────────────────────────
    if args.manual_g:
        matplotlib.use("TkAgg")    # need a display for interactive pick
        peak_px, g_pixels = manual_pick_g_vectors(np.exp(log_ps) - 1, pixel_size)
    else:
        peak_px, g_pixels = auto_detect_g_vectors(log_ps)
    

    for i, (pk, gv) in enumerate(zip(peak_px, g_pixels), 1):
        logger.info("      g%d: FFT pixel (%d, %d)  →  vector (%d, %d) px", 
                            i, pk[0], pk[1], gv[0], gv[1])

    # ── 3. Mask g's ──────────────────────────────────────────────────
    # r1 = mask_radius_from_g(g_pixels[0], args.mask_radius)
    # r2 = mask_radius_from_g(g_pixels[1], args.mask_radius)
    # mask_radius = min(r1, r2)
    mask_radius = min([mask_radius_from_g(g, args.mask_radius) for g in g_pixels])

    # ── 4. Calculate iFFT's ──────────────────────────────────────────────────
    # mask1 = make_mask(F_shifted, peak_px[0], mask_radius, include_friedel=True, mask_type="gaussian")
    # mask2 = make_mask(F_shifted, peak_px[1], mask_radius, include_friedel=True, mask_type="gaussian")
    masks = [make_mask(F_shifted, pk, mask_radius, include_friedel=True, mask_type="gaussian") for pk in peak_px]

    field1 = apply_mask_and_ifft(F_shifted, peak_px[0], masks[0])
    field2 = apply_mask_and_ifft(F_shifted, peak_px[1], masks[1])
    mask_multi = np.sum(masks, axis=0)
    field_multi = apply_mask_and_ifft(F_shifted, peak_px[1], mask_multi)

    def normalize_complex(field):
        bragg = np.real(field)
        bragg -= bragg.min()
        bragg /= bragg.max() + 1e-12
        return bragg

    ifft1 = normalize_complex(field1)
    ifft2 = normalize_complex(field2)
    ifft_multi = normalize_complex(field_multi)

    # ── 5. Save as tiff ──────────────────────────────────────────────────
    for array, filename in zip([ifft1, ifft2, ifft_multi, log_ps, masks[0], masks[1], mask_multi], 
                               ["bragg_g1.tif", "bragg_g2.tif", "bragg_multi.tif", "power_spectrum.tif", "mask1.tif", "mask2.tif", "mask_multi.tif"]):

        outfile = os.path.join(sub_out_folder, filename)

        save_array_as_tiff(
            array, outfile,
            pixel_size=pixel_size, unit=unit_str,
            description="Bragg filter, g/4, cosine mask",
            png_preview=True,
            tiff_with_scale_bar=False,
            scale_bar_kwargs={"location": "lower left", "bar_length_units": 2},
        )


    save_analysis_panneau(img, log_ps, peak_px, mask_multi, ifft1, ifft2, sub_out_folder, pixel_size, unit_str, mask_radius)