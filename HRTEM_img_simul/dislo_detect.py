"""
Dislocation Detection in HRTEM Images (.dm3) via Autocorrelation & Geometric Phase Analysis
==============================================================================================
Pipeline:
  1. Load .dm3 file (ncempy) and extract calibrated image data
  2. Compute 2D power spectrum (|FFT|²) to identify lattice vectors
  3. Pick two independent reciprocal-lattice vectors g1, g2 (auto or manual)
  4. Apply geometric phase analysis (GPA):
       - Band-pass filter around each g-vector
       - Unwrap the phase → P(r)
       - Displacement field u = -P / 2π (in units of the lattice spacing)
  5. Compute the curl of the displacement field → dislocation density map
  6. Threshold curl map to localise individual dislocation cores
  7. Export annotated figure + CSV of core positions

Dependencies (install once):
    pip install ncempy numpy scipy scikit-image matplotlib tifffile

Usage:
    python detect_dislocations.py image.dm3
    python detect_dislocations.py image.dm3 --mask-radius 15 --threshold 0.3 --manual-g
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")           # headless-safe
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, label, center_of_mass, rotate as ndimage_rotate
from _hrtem_helpers import load_dm3, power_spectrum, auto_detect_g_vectors, manual_pick_g_vectors, gpa_phase, bragg_filtered_image

# ── optional: ncempy (preferred) or hyperspy as fallback ──────────────────────
try:
    import ncempy.io.dm as dm_reader
    _READER = "ncempy"
except ImportError:
    try:
        import hyperspy.api as hs
        _READER = "hyperspy"
    except ImportError:
        sys.exit(
            "ERROR: Install ncempy  →  pip install ncempy\n"
            "       or hyperspy    →  pip install hyperspy"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Dislocation detection from displacement curl
# ═══════════════════════════════════════════════════════════════════════════════

def displacement_field(phi, g_pixel_vec, pixel_size):
    """
    Displacement component along g:  u = -phi / (2π |g|²) * g  (scalar projection).
    Returns the scalar field u_g in real-space units (same as pixel_size unit).
    """
    gy, gx = g_pixel_vec          # in pixels
    g_mag = np.hypot(gy, gx)      # pixels⁻¹ in pixel coords
    # Convert to physical: 1/unit
    g_phys = g_mag / (min(*phi.shape) * pixel_size)
    # Displacement projection (in units of the lattice spacing):
    u = -phi / (2 * np.pi * g_phys)
    return u


def curl_2d(ux, uy, pixel_size):
    """
    Numerical curl of 2D vector field (ux, uy):  ∂ux/∂y − ∂uy/∂x
    Returns curl array in 1/unit.
    """
    dux_dy = np.gradient(ux, pixel_size, axis=0)
    duy_dx = np.gradient(uy, pixel_size, axis=1)
    return dux_dy - duy_dx


def detect_cores(curl_map, threshold_factor, smooth_sigma=3.0):
    """
    Threshold the absolute curl map and find connected-component centroids.

    Returns list of (row, col) dislocation core positions.
    """
    curl_smooth = gaussian_filter(np.abs(curl_map), sigma=smooth_sigma)
    thresh = threshold_factor * curl_smooth.max()
    binary = curl_smooth > thresh
    labeled, n = label(binary)
    if n == 0:
        return []
    cores = center_of_mass(curl_smooth, labeled, range(1, n + 1))
    return [tuple(c) for c in cores]


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation & export
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(img, PS, log_ps, bragg1, bragg2, curl, cores,
                 peak_px, pixel_size, unit_str, out_stem):
    """Save a 6-panel diagnostic figure and a CSV of core positions."""

    H, W = img.shape
    extent_img = [0, W * pixel_size, H * pixel_size, 0]   # physical coords

    # ── Layout ────────────────────────────────────────────────────────────────
    # Each cell in the mosaic is one equal "tile".
    # "hrtem" spans a 2×2 block  → 4× the area of one tile  (HRTEM large)
    # "ps" and "curl" each occupy 1×1                        (4× smaller)
    # "b1" / "b2"  each occupy 1×1  (unchanged relative size)
    # "cores" spans the full bottom row (4 tiles wide)
    layout = [
        ["hrtem", "hrtem", "ps",   "curl"],
        ["hrtem", "hrtem", "b1",   "b2"  ],
        ["cores", "cores", "cores","cores"],
    ]
    fig, axd = plt.subplot_mosaic(
        layout,
        figsize=(18, 14),
        gridspec_kw={"hspace": 0.38, "wspace": 0.30},
    )
    fig.suptitle(f"HRTEM Dislocation Analysis — {out_stem}", fontsize=13)

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
        circ = patches.Circle((c, r), radius=6, linewidth=1.5,
                               edgecolor=col, facecolor="none")
        ax.add_patch(circ)
        ax.text(c + 8, r, lbl, color=col, fontsize=8)
    ax.set_title("Power spectrum\n(log)", fontsize=9)
    ax.axis("off")

    # ── 3. Bragg-filtered image g₁ (small) ───────────────────────────────────
    ax = axd["b1"]
    ax.imshow(bragg1, cmap="gray",
              vmin=np.percentile(bragg1, 3), vmax=np.percentile(bragg1, 97),
              origin="upper", extent=extent_img)
    ax.set_title("Bragg filter (g₁)\n[fringe terminations = dislocations]",
                 fontsize=8)
    ax.set_xlabel(f"x [{unit_str}]", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── 4. Bragg-filtered image g₂ (small) ───────────────────────────────────
    ax = axd["b2"]
    ax.imshow(bragg2, cmap="gray",
              vmin=np.percentile(bragg2, 3), vmax=np.percentile(bragg2, 97),
              origin="upper", extent=extent_img)
    ax.set_title("Bragg filter (g₂)\n[fringe terminations = dislocations]",
                 fontsize=8)
    ax.set_xlabel(f"x [{unit_str}]", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── 5. Curl / dislocation density (small) ────────────────────────────────
    ax = axd["curl"]
    vmax_c = np.percentile(np.abs(curl), 99)
    im = ax.imshow(curl, cmap="seismic", origin="upper",
                   vmin=-vmax_c, vmax=vmax_c, extent=extent_img)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=f"curl [{unit_str}⁻¹]")
    ax.set_title("Curl of\ndisp. field", fontsize=9)
    ax.set_xlabel(f"x [{unit_str}]", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── 6. Annotated image with detected cores (full-width bottom strip) ──────
    ax = axd["cores"]
    ax.imshow(img, cmap="gray",
              vmin=np.percentile(img, 3), vmax=np.percentile(img, 97),
              origin="upper", extent=extent_img)
    for (r, c) in cores:
        ax.plot(c * pixel_size, r * pixel_size,
                marker="o", ms=20, mew=1, mfc="none", color="red")
    ax.set_title(f"Detected dislocation cores  (n = {len(cores)})",
                 fontsize=11)
    ax.set_xlabel(f"x [{unit_str}]")
    ax.set_ylabel(f"y [{unit_str}]")

    plt.tight_layout()
    fig_path = out_stem + "_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {fig_path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = out_stem + "_cores.csv"
    with open(csv_path, "w") as f:
        f.write(f"core_id,row_px,col_px,x_{unit_str},y_{unit_str}\n")
        for i, (r, c) in enumerate(cores):
            f.write(f"{i+1},{r:.2f},{c:.2f},"
                    f"{c*pixel_size:.4f},{r*pixel_size:.4f}\n")
    print(f"  Core positions → {csv_path} ({len(cores)} dislocations found)")

    return fig_path, csv_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  512×512 crop export
# ═══════════════════════════════════════════════════════════════════════════════

CROP_SIZE = 512


def interactive_pick_crop_centre(img):
    """
    Show the HRTEM image and let the user click the crop centre.
    Zoom/pan freely; click registers only when the toolbar is in pointer mode.
    Returns (row, col) in pixel coordinates.
    """
    matplotlib.use("TkAgg")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 3), vmax=np.percentile(img, 97), origin="upper")
    ax.set_title(
        "Zoom/pan to your region of interest, then click the crop centre\n"
        "(pointer mode only — deactivate zoom/pan tool first)",
        fontsize=9
    )

    picked = []
    crosshair = []

    def _on_click(event):
        if fig.canvas.toolbar.mode != "":
            return
        if event.inaxes is not ax or event.button != 1:
            return
        if picked:          # only one point needed
            return

        x, y = event.xdata, event.ydata
        picked.append((y, x))       # (row, col)

        # Draw crosshair at chosen point
        h = ax.axhline(y, color="red", lw=0.8, ls="--")
        v = ax.axvline(x, color="red", lw=0.8, ls="--")
        half = CROP_SIZE // 2
        rect = patches.Rectangle(
            (x - half, y - half), CROP_SIZE, CROP_SIZE,
            linewidth=1.2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.set_title(
            f"Crop centre: ({int(round(x))}, {int(round(y))}) px  "
            f"— close window to continue",
            fontsize=9, color="tomato"
        )
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.tight_layout()
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if not picked:
        raise RuntimeError("No crop centre selected — close the window after clicking.")
    return picked[0]   # (row, col) floats


def save_crop_tiff(img, centre_rc, pixel_size, unit_str, out_stem,
                   rotate_deg=0.0):
    """
    Extract a CROP_SIZE × CROP_SIZE region from `img` centred on `centre_rc`.

    `img` is assumed to already be rotated if needed — rotation is applied
    upstream on the full image so the crop contains only real data.
    `rotate_deg` is used only for filename tagging and TIFF metadata.
    """
    half  = CROP_SIZE // 2
    r0, c0 = int(round(centre_rc[0])), int(round(centre_rc[1]))
    H, W   = img.shape

    # ── Clamp centre so the window fits inside the (possibly rotated) image ───
    r0_clamped = int(np.clip(r0, half, H - half))
    c0_clamped = int(np.clip(c0, half, W - half))
    if r0_clamped != r0 or c0_clamped != c0:
        print(f"  ⚠  Crop centre clamped from ({r0}, {c0}) → "
              f"({r0_clamped}, {c0_clamped}) to keep window inside image.")
        r0, c0 = r0_clamped, c0_clamped

    # Zero-pad only when the whole image is smaller than CROP_SIZE
    pad_r = max(0, half - min(r0, H - r0) + 1)
    pad_c = max(0, half - min(c0, W - c0) + 1)
    if pad_r > 0 or pad_c > 0:
        img = np.pad(img, ((pad_r, pad_r), (pad_c, pad_c)), mode="constant")
        r0 += pad_r;  c0 += pad_c

    crop = img[r0 - half : r0 + half, c0 - half : c0 + half].astype(np.float32)

    # ── Build output filename tag ──────────────────────────────────────────────
    rot_tag  = f"_rot{rotate_deg:+.1f}deg" if rotate_deg != 0.0 else ""
    file_tag = f"_crop_{r0}x{c0}{rot_tag}"

    # ── Save TIFF ──────────────────────────────────────────────────────────────
    res = 1.0 / pixel_size if pixel_size > 0 else 1.0
    imagej_meta = {
        "unit": unit_str,
        "physicalsizex": pixel_size,
        "physicalsizey": pixel_size,
    }
    if rotate_deg != 0.0:
        imagej_meta["rotation_deg_ccw"] = rotate_deg

    tiff_path = out_stem + file_tag + ".tif"
    tifffile.imwrite(
        tiff_path,
        crop,
        imagej=True,
        resolution=(res, res),
        metadata=imagej_meta,
    )
    print(f"  Crop TIFF saved  → {tiff_path}  ({CROP_SIZE}×{CROP_SIZE} px, "
          f"centre row={r0} col={c0})")

    # ── Quick-look PNG with scale bar ──────────────────────────────────────────
    title = f"Crop centre: ({c0}, {r0}) px"
    if rotate_deg != 0.0:
        title += f"  |  rotated {rotate_deg:+.1f}° CCW"

    fig, ax = plt.subplots(figsize=(5, 5))
    phys_width = CROP_SIZE * pixel_size
    extent = [0, phys_width, phys_width, 0]
    ax.imshow(crop, cmap="gray", vmin=np.percentile(crop, 3),
              vmax=np.percentile(crop, 97), origin="upper", extent=extent)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f"x [{unit_str}]"); ax.set_ylabel(f"y [{unit_str}]")

    raw_bar   = 0.2 * phys_width
    magnitude = 10 ** np.floor(np.log10(raw_bar))
    bar_len   = round(raw_bar / magnitude) * magnitude
    bar_x0    = 0.05 * phys_width
    bar_y     = 0.93 * phys_width
    ax.plot([bar_x0, bar_x0 + bar_len], [bar_y, bar_y],
            color="white", lw=3, solid_capstyle="butt")
    ax.text(bar_x0 + bar_len / 2, bar_y - 0.03 * phys_width,
            f"{bar_len:g} {unit_str}", color="white",
            ha="center", va="bottom", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    png_path = out_stem + file_tag + "_preview.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Crop preview PNG → {png_path}")

    return tiff_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dm3_file", help="Path to the .dm3 HRTEM image")
    p.add_argument("--mask-radius", type=float, default=20,
                   help="Band-pass mask radius in FFT pixels (default: 20)")
    p.add_argument("--threshold", type=float, default=0.25,
                   help="Fraction of max curl to threshold at (default: 0.25)")
    p.add_argument("--smooth-sigma", type=float, default=3.0,
                   help="Gaussian σ (pixels) before thresholding (default: 3)")
    p.add_argument("--manual-g", action="store_true",
                   help="Interactively pick g-vectors instead of auto-detection")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: same folder as input)")
    p.add_argument(
        "--crop", default=None, metavar="ROW,COL or 'interactive'",
        help=(
            "Export a 512×512 TIFF crop centred on this position.\n"
            "  --crop 312,256        pixel coordinates (row, col)\n"
            "  --crop interactive    click the centre in a pop-up window"
        ),
    )
    p.add_argument(
        "--rotate", type=float, default=0.0, metavar="DEGREES",
        help=(
            "Rotate the crop CCW by this many degrees before saving (default: 0).\n"
            "Use to align the crystal axes with the image x-y frame for GPA comparison.\n"
            "Example: --rotate 3.5"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    dm3_path = Path(args.dm3_file)
    if not dm3_path.exists():
        sys.exit(f"ERROR: File not found — {dm3_path}")

    if args.outdir:
        out_dir = Path(args.outdir)
    else:
        # Mirror input tree but swap the "DATA" segment for "DataTreatment"
        parts = list(dm3_path.parent.parts)
        parts = ["DataTreatment" if p == "DATA" else p for p in parts]
        out_dir = Path(*parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = str(out_dir / dm3_path.stem)

    # ── 1. Load image ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {dm3_path.name} …")
    img, pixel_size, unit_str = load_dm3(dm3_path)
    print(f"      Shape: {img.shape}  |  pixel size: {pixel_size:.4f} {unit_str}")

    # ── Optional full-image rotation (before any processing or cropping) ───────
    if args.rotate != 0.0:
        print(f"      Rotating full image by {args.rotate:+.2f}° CCW …")
        fill_val = float(np.mean(img))
        img = ndimage_rotate(
            img,
            angle=args.rotate,
            reshape=False,    # keep original dimensions; a few-degree rotation
                              # loses only the outermost corner pixels, which
                              # are never in the region of interest anyway
            order=3,          # bicubic
            mode="constant",
            cval=fill_val,
        ).astype(np.float32)

    # ── 2. Power spectrum ──────────────────────────────────────────────────────
    print("[2/5] Computing power spectrum …")
    log_ps, F_shifted = power_spectrum(img)

    # ── 3. G-vector detection ──────────────────────────────────────────────────
    print("[3/5] Detecting lattice g-vectors …")
    if args.manual_g:
        matplotlib.use("TkAgg")    # need a display for interactive pick
        peak_px, g_pixels = manual_pick_g_vectors(np.exp(log_ps) - 1, pixel_size)
    else:
        peak_px, g_pixels = auto_detect_g_vectors(log_ps)

    for i, (pk, gv) in enumerate(zip(peak_px, g_pixels), 1):
        print(f"      g{i}: FFT pixel ({pk[0]}, {pk[1]})  →  "
              f"vector ({gv[0]}, {gv[1]}) px")

    # ── 4. GPA + Bragg filtering ───────────────────────────────────────────────
    print("[4/5] Running GPA and computing Bragg-filtered images …")
    phi1 = gpa_phase(F_shifted, peak_px[0], args.mask_radius)
    phi2 = gpa_phase(F_shifted, peak_px[1], args.mask_radius)

    # Bragg-filtered lattice-fringe images (g + Friedel pair, soft aperture)
    bragg1 = bragg_filtered_image(F_shifted, peak_px[0], args.mask_radius, mask_type="gaussian")
    bragg2 = bragg_filtered_image(F_shifted, peak_px[1], args.mask_radius, mask_type="gaussian")

    # Displacement fields along each g
    ux = displacement_field(phi1, g_pixels[0], pixel_size)
    uy = displacement_field(phi2, g_pixels[1], pixel_size)

    # Curl → dislocation density
    curl = curl_2d(ux, uy, pixel_size)

    # ── 5. Detect cores ────────────────────────────────────────────────────────
    print("[5/5] Localising dislocation cores …")
    cores = detect_cores(curl, args.threshold, smooth_sigma=args.smooth_sigma)

    # ── Save ───────────────────────────────────────────────────────────────────
    fig_path, csv_path = save_results(
        img, np.exp(log_ps) - 1, log_ps, bragg1, bragg2, curl,
        cores, peak_px, pixel_size, unit_str, out_stem
    )

    print(f"\n✓ Done.  {len(cores)} dislocation core(s) detected.")

    # ── Optional 512×512 crop export ───────────────────────────────────────────
    if args.crop:
        if args.crop.lower() == "interactive":
            print("\n[crop] Opening interactive picker …")
            matplotlib.use("TkAgg")
            centre = interactive_pick_crop_centre(img)
        else:
            try:
                parts = args.crop.split(",")
                centre = (float(parts[0]), float(parts[1]))
            except (ValueError, IndexError):
                sys.exit(
                    "ERROR: --crop expects  ROW,COL  (e.g. --crop 312,256) "
                    "or  interactive"
                )
        save_crop_tiff(img, centre, pixel_size, unit_str, out_stem,
                       rotate_deg=args.rotate)

    print()
    return cores


if __name__ == "__main__":
    main()