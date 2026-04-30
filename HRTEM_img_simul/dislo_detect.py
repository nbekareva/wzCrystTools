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
from scipy.signal import find_peaks
from skimage.restoration import unwrap_phase

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
#  I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_dm3(path: str):
    """Return (image_2d_float32, pixel_size_nm, unit_string)."""
    path = str(path)
    if _READER == "ncempy":
        with dm_reader.fileDM(path) as f:
            d = f.getDataset(0)
        img = d["data"].astype(np.float32)
        # calibration lives in d["pixelSize"] / d["pixelUnit"]
        cal = d.get("pixelSize", [1.0, 1.0])
        unit = d.get("pixelUnit", ["px", "px"])
        pixel_size = float(cal[-1]) if hasattr(cal, "__len__") else float(cal)
        unit_str = unit[-1] if hasattr(unit, "__len__") else str(unit)
    else:                          # hyperspy fallback
        s = hs.load(path)
        img = s.data.astype(np.float32)
        ax = s.axes_manager.signal_axes[-1]
        pixel_size = ax.scale
        unit_str = ax.units

    # Collapse to 2-D if a stack was loaded
    while img.ndim > 2:
        img = img[0]

    return img, pixel_size, unit_str


# ═══════════════════════════════════════════════════════════════════════════════
#  FFT / power-spectrum helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _hann2d(shape):
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def power_spectrum(img):
    """Return (log-scaled power, raw FFT shift) with Hann windowing."""
    w = img * _hann2d(img.shape)
    F = np.fft.fftshift(np.fft.fft2(w))
    PS = np.abs(F) ** 2
    return np.log1p(PS), F


def freq_axes(shape, pixel_size):
    """Frequency axes in 1/unit for a centred FFT."""
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0], d=pixel_size))
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1], d=pixel_size))
    return fy, fx


# ═══════════════════════════════════════════════════════════════════════════════
#  Lattice-vector detection
# ═══════════════════════════════════════════════════════════════════════════════

def _radial_profile(PS):
    cy, cx = np.array(PS.shape) // 2
    Y, X = np.indices(PS.shape)
    R = np.hypot(Y - cy, X - cx).astype(int)
    radial = np.bincount(R.ravel(), PS.ravel()) / np.bincount(R.ravel())
    return radial


def auto_detect_g_vectors(PS, n_peaks=2, min_dist_frac=0.04, max_dist_frac=0.45):
    """
    Detect the two dominant lattice g-vectors from the power spectrum.

    Returns list of (row, col) pixel coordinates (in the shifted FFT).
    """
    H, W = PS.shape
    cy, cx = H // 2, W // 2

    # Suppress DC and very-low frequencies
    mask_dc = np.zeros_like(PS)
    rmin = int(min_dist_frac * min(H, W))
    rmax = int(max_dist_frac * min(H, W))
    Y, X = np.indices(PS.shape)
    R = np.hypot(Y - cy, X - cx)
    annulus = (R > rmin) & (R < rmax)

    PS_masked = PS * annulus

    # Find peaks in upper half-plane only (avoid Friedel pairs)
    PS_half = PS_masked.copy()
    PS_half[cy:] = 0          # keep top half
    PS_flat = PS_half.ravel()

    found = []
    suppression_r = int(0.04 * min(H, W))

    for _ in range(n_peaks * 4):           # gather more than needed, then filter
        idx = int(np.argmax(PS_flat))
        r, c = divmod(idx, W)
        if PS_flat[idx] <= 0:
            break
        found.append((r, c))
        # suppress neighbourhood
        rr, cc = np.ogrid[max(0, r - suppression_r):min(H, r + suppression_r),
                          max(0, c - suppression_r):min(W, c + suppression_r)]
        PS_flat = PS_half.ravel()
        PS_half[rr, cc] = 0
        PS_flat = PS_half.ravel()
        if len(found) >= n_peaks:
            break

    if len(found) < 2:
        raise RuntimeError(
            "Could not auto-detect 2 g-vectors. "
            "Try --manual-g or increase --mask-radius."
        )

    # Return the two strongest, expressed as vectors from centre
    g_pixels = [(r - cy, c - cx) for r, c in found[:2]]
    return found[:2], g_pixels


def manual_pick_g_vectors(PS, pixel_size):
    """
    Interactive g-vector picker with full zoom/pan support.

    Zoom and pan freely with the toolbar; clicks are only registered when
    the toolbar is in pointer mode (no active tool).  Right-click or closing
    the window finishes early if 2 points have been picked.
    """
    log_PS = np.log1p(PS)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(log_PS, cmap="inferno", origin="upper")
    ax.set_title(
        "Zoom/pan freely, then click 2 diffraction spots (g₁, g₂)\n"
        "— NOT the central DC spot —  right-click or close when done",
        fontsize=9
    )

    picked   = []          # list of (x_data, y_data) in FFT-pixel coords
    markers  = []          # scatter artists for visual feedback
    colours  = ["cyan", "lime"]
    labels   = ["g₁", "g₂"]

    def _on_click(event):
        # Ignore if a toolbar mode is active (zoom / pan)
        if fig.canvas.toolbar.mode != "":
            return
        # Only left-clicks inside the axes
        if event.inaxes is not ax or event.button != 1:
            return
        if len(picked) >= 2:
            return

        x, y = event.xdata, event.ydata
        picked.append((x, y))
        idx = len(picked) - 1

        # Draw a circle + label at the chosen spot
        circ = patches.Circle((x, y), radius=6, linewidth=1.5,
                               edgecolor=colours[idx], facecolor="none",
                               transform=ax.transData, zorder=5)
        ax.add_patch(circ)
        ax.text(x + 8, y, labels[idx], color=colours[idx],
                fontsize=10, zorder=5)
        fig.canvas.draw_idle()

        if len(picked) == 2:
            ax.set_title("Both g-vectors selected — close the window to continue",
                         fontsize=9, color="lime")
            fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.tight_layout()
    plt.show()                    # blocks until the window is closed
    fig.canvas.mpl_disconnect(cid)

    if len(picked) < 2:
        raise RuntimeError(
            f"Need 2 g-vector picks, got {len(picked)}. "
            "Re-run and click exactly 2 diffraction spots."
        )

    cy, cx = np.array(PS.shape) // 2
    peak_px = [(int(round(y)), int(round(x))) for x, y in picked]
    g_pixels = [(r - cy, c - cx) for r, c in peak_px]
    return peak_px, g_pixels


# ═══════════════════════════════════════════════════════════════════════════════
#  Geometric Phase Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def gpa_phase(F_shifted, peak_rc, mask_radius):
    """
    Band-pass filter the FFT around `peak_rc` (row, col in shifted array),
    shift to origin, IFFT → complex field, extract & unwrap phase.

    Returns unwrapped phase array (radians), same shape as image.
    """
    H, W = F_shifted.shape
    r0, c0 = peak_rc

    # Build circular mask centred on the g-spot
    Y, X = np.indices((H, W))
    dist = np.hypot(Y - r0, X - c0)
    band_mask = dist < mask_radius

    F_filtered = np.zeros_like(F_shifted)
    F_filtered[band_mask] = F_shifted[band_mask]

    # Shift g → DC so we get the slowly-varying phase modulation
    dy, dx = H // 2 - r0, W // 2 - c0
    F_shifted_to_dc = np.roll(np.roll(F_filtered, dy, axis=0), dx, axis=1)

    # Back to real space
    psi = np.fft.ifft2(np.fft.ifftshift(F_shifted_to_dc))

    # Wrapped phase
    phi_wrapped = np.angle(psi)

    # Unwrap
    phi = unwrap_phase(phi_wrapped)
    return phi


# ═══════════════════════════════════════════════════════════════════════════════
#  Bragg-filtered image
# ═══════════════════════════════════════════════════════════════════════════════

def bragg_filtered_image(F_shifted, peak_rc, mask_radius):
    """
    Reconstruct a Bragg-filtered (lattice-fringe) image for the g-vector at
    `peak_rc` by including BOTH the g and its Friedel pair (-g).

    Including -g ensures the result is real-valued, so fringes are directly
    visible as intensity modulations — terminating fringes mark dislocation
    cores.  A soft (cosine) edge on the circular mask suppresses Fourier
    ringing that would otherwise obscure the defects.

    Returns a real-valued 2D array normalised to [0, 1].
    """
    H, W = F_shifted.shape
    cy, cx = H // 2, W // 2
    r0, c0 = peak_rc

    # Friedel (conjugate) partner
    r1, c1 = 2 * cy - r0, 2 * cx - c0

    Y, X = np.indices((H, W))

    def soft_mask(rc, rr, cc):
        """Cosine-tapered circular aperture centred on (rc[0], rc[1])."""
        d = np.hypot(Y - rc[0], X - rc[1])
        m = np.zeros((H, W))
        inner = mask_radius * 0.75
        m[d <= inner] = 1.0
        taper = (d > inner) & (d < mask_radius)
        m[taper] = 0.5 * (1 + np.cos(np.pi * (d[taper] - inner) /
                                      (mask_radius - inner)))
        return m

    mask = soft_mask((r0, c0), r0, c0) + soft_mask((r1, c1), r1, c1)
    mask = np.clip(mask, 0, 1)

    F_bragg = F_shifted * mask

    # IFFT → real part (imaginary is numerical noise)
    bragg = np.real(np.fft.ifft2(np.fft.ifftshift(F_bragg)))

    # Normalise to [0, 1] for display
    bragg -= bragg.min()
    bragg /= bragg.max() + 1e-12
    return bragg


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

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"HRTEM Dislocation Analysis — {out_stem}", fontsize=13)

    # 1. Raw image
    ax = axes[0, 0]
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 3), vmax=np.percentile(img, 97), origin="upper", extent=extent_img)
    ax.set_title("HRTEM image")
    ax.set_xlabel(f"x [{unit_str}]"); ax.set_ylabel(f"y [{unit_str}]")

    # 2. Power spectrum with g-vector markers
    ax = axes[0, 1]
    ax.imshow(log_ps, cmap="inferno", origin="upper")
    colours = ["cyan", "lime"]
    labels = ["g₁", "g₂"]
    for (r, c), col, lbl in zip(peak_px, colours, labels):
        circ = patches.Circle((c, r), radius=6, linewidth=1.5,
                               edgecolor=col, facecolor="none")
        ax.add_patch(circ)
        ax.text(c + 8, r, lbl, color=col, fontsize=9)
    ax.set_title("Power spectrum (log)")
    ax.axis("off")

    # 3. Bragg-filtered image for g₁
    # Terminating fringes = dislocation cores visible directly in the image
    ax = axes[0, 2]
    ax.imshow(bragg1, cmap="gray", vmin=np.percentile(bragg1, 3), vmax=np.percentile(bragg1, 97), origin="upper", extent=extent_img)
    ax.set_title("Bragg-filtered image (g₁)\n[lattice fringes — terminations = dislocations]")
    ax.set_xlabel(f"x [{unit_str}]")

    # 4. Bragg-filtered image for g₂
    ax = axes[1, 0]
    ax.imshow(bragg2, cmap="gray", vmin=np.percentile(bragg2, 3), vmax=np.percentile(bragg2, 97), origin="upper", extent=extent_img)
    ax.set_title("Bragg-filtered image (g₂)\n[lattice fringes — terminations = dislocations]")
    ax.set_xlabel(f"x [{unit_str}]"); ax.set_ylabel(f"y [{unit_str}]")

    # 5. Curl (dislocation density)
    ax = axes[1, 1]
    vmax = np.percentile(np.abs(curl), 99)
    im = ax.imshow(curl, cmap="seismic", origin="upper",
                   vmin=-vmax, vmax=vmax, extent=extent_img)
    plt.colorbar(im, ax=ax, fraction=0.04, label=f"curl [{unit_str}⁻¹]")
    ax.set_title("Curl of displacement field")
    ax.set_xlabel(f"x [{unit_str}]")

    # 6. Annotated image with detected cores
    ax = axes[1, 2]
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 3), vmax=np.percentile(img, 97), origin="upper", extent=extent_img)
    for (r, c) in cores:
        ax.plot(c * pixel_size, r * pixel_size,
                marker="o", ms=20, mew=1, mfc="none", color="red")
    ax.set_title(f"Detected cores (n={len(cores)})")
    ax.set_xlabel(f"x [{unit_str}]")

    plt.tight_layout()
    fig_path = out_stem + "_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {fig_path}")

    # CSV
    csv_path = out_stem + "_cores.csv"
    with open(csv_path, "w") as f:
        f.write(f"core_id,row_px,col_px,x_{unit_str},y_{unit_str}\n")
        for i, (r, c) in enumerate(cores):
            f.write(f"{i+1},{r:.2f},{c:.2f},{c*pixel_size:.4f},{r*pixel_size:.4f}\n")
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
    bragg1 = bragg_filtered_image(F_shifted, peak_px[0], args.mask_radius)
    bragg2 = bragg_filtered_image(F_shifted, peak_px[1], args.mask_radius)

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