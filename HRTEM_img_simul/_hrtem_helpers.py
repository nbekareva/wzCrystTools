import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from skimage.restoration import unwrap_phase
import tifffile

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

def _read_tiff_pixel_size_and_unit(tif_path):
    """Read pixel size (unit/pixel) and unit string from TIFF metadata."""

    def _res_to_float(value):
        if value is None:
            return None
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                return None
            num, den = value
            if den == 0:
                return None
            return float(num) / float(den)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        xres = _res_to_float(page.tags.get("XResolution").value
                             if page.tags.get("XResolution") else None)
        yres = _res_to_float(page.tags.get("YResolution").value
                             if page.tags.get("YResolution") else None)
        px_per_unit = xres if (xres and xres > 0) else yres

        unit = None
        ij_meta = tif.imagej_metadata or {}
        if isinstance(ij_meta, dict):
            unit = ij_meta.get("unit")

        if not unit:
            res_unit_tag = page.tags.get("ResolutionUnit")
            res_unit_val = getattr(res_unit_tag, "value", None)
            if res_unit_val is not None and hasattr(res_unit_val, "name"):
                res_unit_val = res_unit_val.name
            if isinstance(res_unit_val, str):
                val = res_unit_val.lower()
                if "inch" in val:
                    unit = "in"
                elif "centimeter" in val or "cm" in val:
                    unit = "cm"
                else:
                    unit = "px"
            elif res_unit_val == 2:
                unit = "in"
            elif res_unit_val == 3:
                unit = "cm"
            else:
                unit = "px"

    pixel_size = (1.0 / px_per_unit) if (px_per_unit and px_per_unit > 0) else None
    return pixel_size, unit


def tif_to_png(tif_path, png_path=None, scale_bar=True,
               scale_bar_kwargs=None, pct_clip=(0.5, 99.5)):
    """
    Convert a TIFF file to a figure-ready PNG with an optional scale bar.

    Parameters
    ----------
    tif_path : str
        Path to the input TIFF file.
    png_path : str or None
        Path to the output PNG file. If None, the PNG will be saved in the
        same directory as the TIFF with the same base name.
    scale_bar : bool
        If True (default), read pixel calibration from TIFF metadata and
        burn a scale bar into the output PNG.
    scale_bar_kwargs : dict or None
        Extra keyword arguments forwarded to add_scale_bar().
    pct_clip : (float, float) or None
        Percentile clip used when converting non-uint8 arrays for display.

    Returns
    -------
    str
        Path to the saved PNG file.
    """

    if png_path is None:
        base, _ = os.path.splitext(tif_path)
        png_path = f"{base}.png"

    try:
        img = tifffile.imread(tif_path)
    except ValueError as exc:
        msg = str(exc).lower()
        # LZW-compressed TIFFs need imagecodecs for tifffile; Pillow can
        # usually decode them, so fall back instead of failing hard.
        if "requires the 'imagecodecs' package" not in msg:
            raise
        with Image.open(tif_path) as pil_img:
            img = np.array(pil_img)

    # If a stack is present, use the first frame by default.
    while img.ndim > 2 and img.shape[-1] not in (3, 4):
        img = img[0]

    if img.ndim == 2:
        display_img = _normalize_to_uint8(img, pct_clip=pct_clip)
    elif img.ndim == 3 and img.shape[-1] in (3, 4):
        rgb = img[..., :3]
        if rgb.dtype == np.uint8:
            display_img = rgb.copy()
        else:
            display_img = np.stack(
                [_normalize_to_uint8(rgb[..., i], pct_clip=pct_clip)
                 for i in range(3)],
                axis=-1,
            )
    else:
        raise ValueError(f"Unsupported TIFF shape for PNG export: {img.shape}")

    if scale_bar:
        pixel_size, unit = _read_tiff_pixel_size_and_unit(tif_path)
        if pixel_size is None or pixel_size <= 0:
            raise ValueError(
                "Could not read a valid pixel size from TIFF metadata; "
                "cannot draw scale bar."
            )
        out = add_scale_bar(
            display_img,
            pixel_size=pixel_size,
            unit=unit,
            **(scale_bar_kwargs or {}),
        )
    else:
        out = np.stack([display_img] * 3, axis=-1) if display_img.ndim == 2 else display_img

    Image.fromarray(out).save(png_path)

    return png_path


def _normalize_to_uint8(img, vmin=None, vmax=None, pct_clip=(0.5, 99.5)):
    """
    Rescale a float/int 2D array to uint8, for figure-ready export only
    (the calibrated TIFF written by save_array_as_tiff keeps full precision
    and is unaffected by this).

    Parameters
    ----------
    img : 2D array
    vmin, vmax : float or None
        Explicit display range. Overrides pct_clip if given.
    pct_clip : (low, high) or None
        Percentile clip applied before min-max scaling, so a few outlier
        pixels (e.g. a hot pixel or an unwrapping artefact) don't crush the
        contrast of everything else. Ignored if vmin/vmax are given.

    Returns
    -------
    2D uint8 array.
    """
    img = np.asarray(img, dtype=np.float64)
    if vmin is None or vmax is None:
        if pct_clip is not None:
            lo, hi = np.percentile(img, pct_clip)
        else:
            lo, hi = float(img.min()), float(img.max())
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1e-12
    out = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def _nice_scale_bar_length(fov_units, frac=0.2):
    """
    Pick a visually clean scale-bar length (1/2/5 x a power of ten) that's
    roughly `frac` of the field of view — the same convention microscopy
    software uses so bars read as round numbers (5 nm, 20 nm, ...) instead
    of odd values like 17.3 nm.
    """
    target = fov_units * frac
    if target <= 0:
        return fov_units * frac
    exponent = np.floor(np.log10(target))
    base = target / (10 ** exponent)
    if base < 1.5:
        nice = 1
    elif base < 3.5:
        nice = 2
    elif base < 7.5:
        nice = 5
    else:
        nice = 10
    return float(nice * (10 ** exponent))


def add_scale_bar(img_uint8_or_rgb, pixel_size, unit="nm",
                   bar_length_units=None, bar_frac=0.2,
                   location="lower right", color=(255, 255, 255),
                   bar_thickness_frac=0.012, margin_frac=0.04,
                   show_label=True, font_size=None, outline=True):
    """
    Burn a scale bar (and optional length label) into a displayable image.

    Parameters
    ----------
    img_uint8_or_rgb : 2D uint8 array (grayscale) or (H, W, 3) uint8 array
        Base image to draw on. Use _normalize_to_uint8() first if starting
        from raw float data (e.g. a GPA phase map or Bragg-filtered image).
    pixel_size : float
        Real-world size of one pixel, in `unit`.
    unit : str
        Physical unit label shown next to the bar, e.g. "nm", "Å", "µm".
    bar_length_units : float or None
        Explicit bar length in physical units (e.g. 5 for a "5 nm" bar). If
        None, a clean round length (~bar_frac of the image width) is chosen
        automatically via _nice_scale_bar_length().
    bar_frac : float
        Target fraction of the image width the bar should span; only used
        when bar_length_units is None.
    location : {"lower right", "lower left", "upper right", "upper left"}
    color : (int, int, int)
        RGB color of the bar and label.
    bar_thickness_frac : float
        Bar thickness as a fraction of the image's shorter dimension.
    margin_frac : float
        Margin from the image edge, as a fraction of the shorter dimension.
    show_label : bool
        If True, draw "<length> <unit>" above/below the bar.
    font_size : int or None
        Font size in points; auto-scaled from image size if None.
    outline : bool
        If True, draw a thin dark outline behind the bar/text so it stays
        legible against busy or bright backgrounds.

    Returns
    -------
    (H, W, 3) uint8 RGB array with the scale bar burned in.
    """
    img = np.asarray(img_uint8_or_rgb)
    if img.dtype != np.uint8:
        raise ValueError("img_uint8_or_rgb must be uint8 — run it through "
                          "_normalize_to_uint8() first.")
    rgb = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img.copy()

    H, W = rgb.shape[:2]
    fov_units = W * pixel_size

    if bar_length_units is None:
        bar_length_units = _nice_scale_bar_length(fov_units, bar_frac)

    bar_px = bar_length_units / pixel_size
    thickness_px = max(2, int(round(bar_thickness_frac * min(H, W))))
    margin_px = int(round(margin_frac * min(H, W)))

    pil_img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(pil_img)

    if font_size is None:
        font_size = max(10, int(round(0.035 * min(H, W))))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    if "right" in location:
        x1 = W - margin_px
        x0 = x1 - bar_px
    else:
        x0 = margin_px
        x1 = x0 + bar_px
    if "lower" in location:
        y1 = H - margin_px
        y0 = y1 - thickness_px
    else:
        y0 = margin_px
        y1 = y0 + thickness_px

    if outline:
        draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], fill=(0, 0, 0))
    draw.rectangle([x0, y0, x1, y1], fill=tuple(color))

    if show_label:
        label = f"{bar_length_units:g} {unit}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_x = x0 + (bar_px - text_w) / 2
        text_y = (y0 - text_h - 6) if "lower" in location else (y1 + 6)
        if outline:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        draw.text((text_x + dx, text_y + dy), label,
                                  font=font, fill=(0, 0, 0))
        draw.text((text_x, text_y), label, font=font, fill=tuple(color))

    return np.array(pil_img)


def save_array_as_tiff(array, path, pixel_size=None, unit="nm",
                        description=None, imagej=True,
                        png_preview=True,
                        tiff_with_scale_bar=False,
                        scale_bar_kwargs=None, pct_clip=(0.5, 99.5)):
    """
    Save any array produced by this module (a GPA phase map, a Bragg-
    filtered image, a power spectrum, a raw loaded image, ...) to a
    calibrated TIFF, with an optional second figure-ready copy that has a
    scale bar burned in.

    Two things are handled, kept deliberately separate:

    1. Calibration metadata (always, if pixel_size is given): pixel_size +
       unit are written into the TIFF resolution tags (and the ImageJ
       metadata block, if imagej=True), so Fiji/ImageJ opens the file
       already knowing the physical pixel size — no manual "Set Scale"
       needed. The array's actual values are saved unmodified, at full
       precision (e.g. float32 for a phase map) — this file stays suitable
       for further quantitative analysis.

    2. An optional visual scale bar (tiff_with_scale_bar=True): a
       *second*, contrast-stretched, 8-bit RGB TIFF is additionally saved
       (suffix "_scalebar") with the bar burned into the pixels. This is
       for figures/publication only — burning the bar in necessarily
       discards the original data range and precision, so don't use this
       file for further analysis.

    Parameters
    ----------
    array : 2D array (float, int, or bool)
        Data to save.
    path : str
        Output file path, e.g. "results/phase_map.tif". ".tif"/".tiff" is
        appended if missing.
    pixel_size : float or None
        Real-world size of one pixel, in `unit`. If None, the file is still
        saved, just without calibration.
    unit : str
        Physical unit for pixel_size, e.g. "nm", "um", "A". ImageJ natively
        recognises "nm", "um"/"µm", "mm", "cm", "in"; other units are still
        recorded correctly, just may not auto-display in Fiji's UI.
    description : str or None
        Free-text metadata to embed in the TIFF (e.g. processing
        parameters — g-vector, mask_type, frac — for provenance). Stored in
        the ImageDescription / ImageJ "Info" field.
    imagej : bool
        Write in ImageJ-compatible format (recommended for calibration to
        be auto-recognised by Fiji).
    tiff_with_scale_bar : bool
        If True, also save the second 8-bit RGB TIFF described above.
        Requires pixel_size.
    scale_bar_kwargs : dict or None
        Extra keyword arguments forwarded to add_scale_bar(), e.g.
        {"location": "lower left", "color": (255, 255, 0), "bar_length_units": 5}.
    pct_clip : (float, float) or None
        Percentile contrast clip used only for the scale-bar overlay's 8-bit
        stretch (see _normalize_to_uint8). Has no effect on the calibrated
        TIFF from point 1.

    Returns
    -------
    dict with key "data_path" (the calibrated TIFF) and, if requested,
    "scalebar_path" (the figure-ready overlay TIFF).
    """

    path = str(path)
    if not path.lower().endswith((".tif", ".tiff")):
        path += ".tif"

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    array = np.asarray(array)

    if imagej:
        # ImageJ's TIFF format only supports a limited set of dtypes
        # (8/16-bit int, float32, RGB uint8...). Cast losslessly-in-practice
        # types down so common outputs of this module (float64 phase maps,
        # boolean masks) save without the caller having to think about it.
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        elif array.dtype == np.bool_:
            array = array.astype(np.uint8) * 255

    resolution = None
    metadata = {}
    if pixel_size is not None and pixel_size > 0:
        # TIFF resolution tags are pixels-per-unit, i.e. the inverse of
        # pixel_size (which is unit-per-pixel).
        px_per_unit = 1.0 / pixel_size
        resolution = (px_per_unit, px_per_unit)
        unit_l = unit.lower()
        if unit_l in ("um", "µm", "micron", "microns"):
            metadata["unit"] = "um"
        elif unit_l in ("nm", "nanometer", "nanometers"):
            metadata["unit"] = "nm"
        else:
            # Still recorded, just outside ImageJ's small built-in unit set.
            metadata["unit"] = unit

    if description is not None:
        metadata["Info"] = description

    tifffile.imwrite(
        path,
        array,
        imagej=imagej,
        resolution=resolution,
        metadata=metadata if metadata else None,
    )

    result = {"data_path": path}

    if tiff_with_scale_bar:
        if not pixel_size:
            raise ValueError("pixel_size is required to draw a scale bar.")
        base = _normalize_to_uint8(array, pct_clip=pct_clip)
        rgb = add_scale_bar(base, pixel_size, unit=unit,
                             **(scale_bar_kwargs or {}))
        root, ext = os.path.splitext(path)
        sb_path = f"{root}_scalebar{ext}"
        tifffile.imwrite(sb_path, rgb)
        result["scalebar_path"] = sb_path

    if png_preview:
        base = _normalize_to_uint8(array, pct_clip=pct_clip)
        rgb = add_scale_bar(base, pixel_size, unit=unit,
                             **(scale_bar_kwargs or {}))
        root, _ = os.path.splitext(path)
        png_path = f"{root}.png"
        plt.imsave(png_path, rgb)
        result["png_preview"] = png_path

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  FFT / power-spectrum helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _hann2d(shape):
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def power_spectrum(img, hann_window=True):
    """Return (log-scaled power, raw FFT shift) with Hann windowing."""
    if hann_window:
        w = img * _hann2d(img.shape)
    else:
        w = img
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
    rmin = int(min_dist_frac * min(H, W))
    rmax = int(max_dist_frac * min(H, W))
    Y, X = np.indices(PS.shape)
    R = np.hypot(Y - cy, X - cx)
    # Keep only pixels in a ring outside the central DC component
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


def manual_pick_g_vectors(PS, pixel_size, n_points=None, min_points=1):
    """
    Interactive g-vector picker with full zoom/pan support, for an
    arbitrary number of g-vectors.
 
    Zoom and pan freely with the toolbar; clicks are only registered when
    the toolbar is in pointer mode (no active tool).
      - Left-click  : add a g-vector pick at that spot.
      - Right-click : undo the most recent pick.
      - Enter, or closing the window : finish picking.
 
    Parameters
    ----------
    PS : 2D array
        Power spectrum to display (e.g. power_spectrum(img)[0]).
    pixel_size : float
        Unused inside this function; kept for signature compatibility with
        other picker/calibration helpers.
    n_points : int or None
        If given, picking stops automatically (figure auto-closes) once
        this many points have been picked. If None (default), pick as many
        g-vectors as you like and finish manually (Enter or close window).
    min_points : int
        Minimum number of picks required; raises RuntimeError if fewer
        were made when picking ends. Default 1 (a single g-vector is valid
        for e.g. one-directional GPA/Bragg filtering); pass 2 if you
        specifically need a full 2D lattice basis.
 
    Returns
    -------
    (peak_px, g_pixels) : (list of (row, col), list of (row, col))
        Absolute pixel coordinates of each picked spot, and the
        corresponding g-vector offsets from the FFT centre, in pick order.
    """
    log_PS = np.log1p(PS)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(log_PS, cmap="inferno", origin="upper")
 
    cmap = plt.get_cmap("tab10")
    picked   = []   # list of (x_data, y_data) in FFT-pixel coords
    artists  = []   # list of (circle, text) per pick, for undo
 
    def _colour(idx):
        return cmap(idx % 10)
 
    def _label(idx):
        return f"g{idx + 1}"
 
    def _status_text():
        n_str = f"{len(picked)}/{n_points}" if n_points else f"{len(picked)}"
        return (
            f"Picked {n_str} — left-click to add, right-click to undo\n"
            "— NOT the central DC spot —  Enter or close window when done"
        )
 
    ax.set_title(_status_text(), fontsize=9)
 
    def _redraw_status(done=False):
        if done:
            ax.set_title("Done — figure will close", fontsize=9, color="lime")
        else:
            ax.set_title(_status_text(), fontsize=9)
        fig.canvas.draw_idle()
 
    def _add_point(x, y):
        picked.append((x, y))
        idx = len(picked) - 1
        colour = _colour(idx)
        circ = patches.Circle((x, y), radius=6, linewidth=1.5,
                               edgecolor=colour, facecolor="none",
                               transform=ax.transData, zorder=5)
        ax.add_patch(circ)
        txt = ax.text(x + 8, y, _label(idx), color=colour,
                       fontsize=10, zorder=5)
        artists.append((circ, txt))
 
    def _undo_last():
        if not picked:
            return
        picked.pop()
        circ, txt = artists.pop()
        circ.remove()
        txt.remove()
 
    def _on_click(event):
        # Ignore if a toolbar mode is active (zoom / pan)
        if fig.canvas.toolbar.mode != "":
            return
        if event.inaxes is not ax:
            return
 
        if event.button == 1:                      # left-click: add
            if n_points is not None and len(picked) >= n_points:
                return
            _add_point(event.xdata, event.ydata)
        elif event.button == 3:                     # right-click: undo
            _undo_last()
        else:
            return
 
        if n_points is not None and len(picked) >= n_points:
            _redraw_status(done=True)
            plt.close(fig)
        else:
            _redraw_status()
 
    def _on_key(event):
        if event.key == "enter":
            _redraw_status(done=True)
            plt.close(fig)
 
    cid_click = fig.canvas.mpl_connect("button_press_event", _on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.tight_layout()
    plt.show()                    # blocks until the window is closed
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)
 
    if len(picked) < min_points:
        raise RuntimeError(
            f"Need at least {min_points} g-vector pick(s), got {len(picked)}. "
            "Re-run and click at least that many diffraction spots."
        )
 
    cy, cx = np.array(PS.shape) // 2
    peak_px = [(int(round(y)), int(round(x))) for x, y in picked]
    g_pixels = [(r - cy, c - cx) for r, c in peak_px]
    return peak_px, g_pixels
 

# ═══════════════════════════════════════════════════════════════════════════════
#  Mask sizing as a fraction of |g|
# ═══════════════════════════════════════════════════════════════════════════════

def g_vector_length_px(g_px):
    """
    Magnitude of a g-vector in FFT pixels.

    Parameters
    ----------
    g_px : tuple(float, float)
        (row, col) offset from the FFT centre, as returned by
        auto_detect_g_vectors / manual_pick_g_vectors (their `g_pixels`
        output — NOT the raw peak_rc coordinates, which are absolute).

    Returns
    -------
    float
    """
    return float(np.hypot(g_px[0], g_px[1]))


def mask_radius_from_g(g_px, frac=0.5):
    """
    Define a Bragg/GPA mask radius as a fraction of |g|, rather than an
    arbitrary fixed pixel count. This is the standard way to size the
    band-pass aperture: it scales automatically with lattice spacing and
    image calibration.

    Common choices:
      frac = 0.5   → |g|/2, the widest mask that still avoids touching the
                     neighbouring spot or DC for a well-separated lattice.
                     Best real-space resolution, most sensitive to noise
                     from adjacent frequencies.
      frac = 0.25  → |g|/4, a tighter, more conservative mask. Cleaner
                     phase/fringe images at the cost of some real-space
                     (spatial) resolution.
      frac = 0.33  → |g|/3, a common middle-ground default.

    Parameters
    ----------
    g_px : tuple(float, float)
        (row, col) g-vector offset from the FFT centre, in pixels.
    frac : float
        Fraction of |g| to use as the mask radius. Must lie in (0, 0.5]:
        above 0.5 the mask would reach past the g-spot itself back toward
        the origin / the opposite spot.

    Returns
    -------
    float
        Mask radius, in FFT pixels.
    """
    if not (0 < frac <= 0.5):
        raise ValueError("frac must be in (0, 0.5] to avoid touching DC "
                          "or overlapping the neighbouring peak.")
    return frac * g_vector_length_px(g_px)


# ═══════════════════════════════════════════════════════════════════════════════
#  Mask construction + application (generic band-pass → IFFT)
# ═══════════════════════════════════════════════════════════════════════════════

def soft_circular_mask(shape, centre_rc, radius, taper_frac=0.25):
    """
    Cosine-tapered circular aperture, used to band-pass a single spot in a
    shifted FFT without introducing the ringing a hard-edged mask would
    cause in real space.

    Parameters
    ----------
    shape : (H, W)
        Shape of the array the mask will be applied to.
    centre_rc : (row, col)
        Centre of the aperture, in pixel coordinates of `shape`.
    radius : float
        Outer radius of the aperture, in pixels.
    taper_frac : float
        Fraction of `radius` given over to the cosine roll-off at the edge
        (0 → hard-edged disc, values approaching 1 → almost fully tapered).

    Returns
    -------
    2D float array, same shape as `shape`, values in [0, 1].
    """
    H, W = shape
    r0, c0 = centre_rc
    Y, X = np.indices((H, W))
    d = np.hypot(Y - r0, X - c0)

    inner = radius * (1 - taper_frac)
    m = np.zeros((H, W))
    m[d <= inner] = 1.0
    taper = (d > inner) & (d < radius)
    if radius > inner:
        m[taper] = 0.5 * (1 + np.cos(np.pi * (d[taper] - inner) / (radius - inner)))
    return m


def gaussian_circular_mask(shape, centre_rc, radius, n_sigma=3.0, hard_cutoff=False):
    """
    Gaussian band-pass aperture centred on `centre_rc`, the mask style used
    by default in several established GPA / Bragg-filtering packages
    (e.g. Hÿtch's original GPA formulation, HREM Research's GPA plugin,
    Strain++), as an alternative to the hard-ish cosine-tapered disc.

    IMPORTANT — `radius` here means the same thing it means for the cosine
    mask (soft_circular_mask): an aperture *radius* in pixels, typically
    from mask_radius_from_g(g_px, frac) — e.g. frac=0.25 → g/4. It is NOT
    the Gaussian's standard deviation. sigma is derived from it as
    sigma = radius / n_sigma, so that `radius` corresponds to the point
    n_sigma standard deviations from the centre (n_sigma=3, the default,
    means ~99.7% of the Gaussian's mass falls within `radius` — the usual
    "3-sigma" convention). This keeps "g/4" meaning the same physical
    aperture size regardless of which mask_type you pick.

    Unlike the cosine mask, the Gaussian still decays smoothly past
    `radius` unless `hard_cutoff=True` — which avoids the sidelobes/ringing
    a disc edge can leave in the reconstruction, at the cost of a fuzzier
    boundary between "kept" and "rejected" frequencies.

    Parameters
    ----------
    shape : (H, W)
        Shape of the array the mask will be applied to.
    centre_rc : (row, col)
        Centre of the aperture, in pixel coordinates of `shape`.
    radius : float
        Aperture radius in pixels (same meaning as mask_radius elsewhere —
        e.g. from mask_radius_from_g()), NOT sigma.
    n_sigma : float
        Number of standard deviations that `radius` corresponds to.
        sigma = radius / n_sigma. Larger n_sigma → narrower, more
        conservative Gaussian for the same radius; smaller n_sigma → wider,
        more permissive. 3.0 (99.7% within radius) is a sensible default.
    hard_cutoff : bool
        If True, zero the mask beyond `radius` (mirrors the cosine mask's
        true cutoff exactly). If False (default), leave the Gaussian's
        smooth tail beyond `radius` uncut.

    Returns
    -------
    2D float array, same shape as `shape`, values in [0, 1].
    """
    H, W = shape
    r0, c0 = centre_rc
    Y, X = np.indices((H, W))
    d2 = (Y - r0) ** 2 + (X - c0) ** 2

    sigma = radius / n_sigma
    m = np.exp(-d2 / (2.0 * sigma ** 2))
    if hard_cutoff:
        m[d2 > radius ** 2] = 0.0
    return m


def apply_mask_and_ifft(F_shifted, peak_rc, mask, shift_to_dc=False):
    """
    Band-pass filter a shifted FFT around `peak_rc` (optionally plus its
    Friedel pair -g) using a soft mask, then inverse-transform back to real
    space. This is the common core used by both GPA phase extraction and
    Bragg-filtered lattice-fringe imaging.

    Parameters
    ----------
    F_shifted : 2D complex array
        fftshift'ed FFT of the image (as returned by power_spectrum).
    peak_rc : (row, col)
        Pixel location of the g-spot in the shifted FFT (absolute array
        coordinates, e.g. from auto_detect_g_vectors' first return value).
    mask_radius : float
        Aperture radius in pixels, for BOTH mask types — e.g. from
        mask_radius_from_g(g_px, frac). For mask_type="gaussian" this is
        converted internally into sigma = mask_radius / n_sigma; it is
        never itself treated as sigma.
    include_friedel : bool
        If True, also keep the conjugate spot (-g) so the reconstruction is
        real-valued — used for Bragg-filtered lattice-fringe images. If
        False, only the single g-spot is kept — used for GPA, where the
        complex field's phase carries the displacement information.
    taper_frac : float
        Cosine-taper fraction, only used when mask_type="cosine" (suppresses
        Fourier ringing / edge artefacts from the disc's edge).
    shift_to_dc : bool
        If True, roll the filtered spectrum so peak_rc sits at the array
        centre before the IFFT — recovers the slowly-varying phase/
        amplitude modulation needed for GPA. If False, the IFFT is taken
        without shifting — appropriate for Bragg-filtered real-space
        fringe images.
    mask_type : {"cosine", "gaussian"}
        Aperture shape. "cosine" = flat centre with a cosine roll-off,
        hard cutoff at mask_radius (soft_circular_mask). "gaussian" =
        smooth Gaussian falloff, sigma = mask_radius / n_sigma
        (gaussian_circular_mask), the default in several established
        GPA/Bragg-filtering packages.
    n_sigma : float
        Only used when mask_type="gaussian". See gaussian_circular_mask.

    Returns
    -------
    complex 2D array
        Real-space reconstructed field. Use `.real` for Bragg-filtered
        images, or `np.angle(...)` (then `unwrap_phase`) for GPA phase maps.
    """
    F_filtered = F_shifted * mask

    H, W = F_shifted.shape
    cy, cx = H // 2, W // 2
    r0, c0 = peak_rc

    if shift_to_dc:
        dy, dx = cy - r0, cx - c0
        F_filtered = np.roll(np.roll(F_filtered, dy, axis=0), dx, axis=1)

    field = np.fft.ifft2(np.fft.ifftshift(F_filtered))
    return field

def make_mask(F_shifted, peak_rc, mask_radius,
              include_friedel=True, taper_frac=0.25,
              mask_type="gaussian", n_sigma=3.0):
    """
    Create a Fourier-space mask centered on `peak_rc`.

    Returns
    -------
    2D array
        The mask to be applied to the FFT of the image.
    """
    if mask_type not in ("cosine", "gaussian"):
        raise ValueError("mask_type must be 'cosine' or 'gaussian'")

    H, W = F_shifted.shape
    cy, cx = H // 2, W // 2
    r0, c0 = peak_rc

    def _make_mask(centre):
        if mask_type == "cosine":
            return soft_circular_mask((H, W), centre, mask_radius, taper_frac)
        return gaussian_circular_mask((H, W), centre, mask_radius, n_sigma)

    mask = _make_mask((r0, c0))

    if include_friedel:
        r1, c1 = 2 * cy - r0, 2 * cx - c0
        mask = mask + _make_mask((r1, c1))
        mask = np.clip(mask, 0, 1)

    return mask


# ═══════════════════════════════════════════════════════════════════════════════
#  Geometric Phase Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def gpa_phase(F_shifted, peak_rc, mask_radius, mask_type="cosine", n_sigma=3.0):
    """
    Band-pass filter the FFT around `peak_rc` (row, col in shifted array),
    shift to origin, IFFT → complex field, extract & unwrap phase.

    mask_type : {"cosine", "gaussian"} — see apply_mask_and_ifft(). In both
    cases `mask_radius` is an aperture radius in pixels; for "gaussian" it
    is converted to sigma = mask_radius / n_sigma internally.

    Returns unwrapped phase array (radians), same shape as image.
    """
    psi = apply_mask_and_ifft(F_shifted, peak_rc, mask_radius,
                               include_friedel=False, shift_to_dc=True,
                               mask_type=mask_type, n_sigma=n_sigma)
    phi_wrapped = np.angle(psi)
    phi = unwrap_phase(phi_wrapped)
    return phi


def gpa_phase_from_frac(F_shifted, peak_rc, g_px, frac=0.5, taper_frac=0.25,
                         mask_type="gaussian", n_sigma=3.0):
    """
    Convenience wrapper: run gpa_phase() with the mask radius defined as a
    fraction of |g| instead of a fixed pixel value.

    Parameters
    ----------
    F_shifted : 2D complex array
    peak_rc : (row, col)
        Absolute pixel location of the g-spot in the shifted FFT.
    g_px : (row, col)
        g-vector offset from the FFT centre (i.e. peak_rc - centre), used
        only to compute |g|.
    frac : float
        Mask (aperture) radius as a fraction of |g|; see
        mask_radius_from_g(). Same meaning for both mask_type values — for
        "gaussian" it is the point corresponding to n_sigma standard
        deviations, not sigma itself.
    taper_frac : float
        Cosine-taper fraction for the mask edge (ignored if
        mask_type="gaussian").
    mask_type : {"cosine", "gaussian"}
    n_sigma : float
        Only used when mask_type="gaussian"; sigma = mask_radius / n_sigma.

    Returns
    -------
    unwrapped phase array (radians), same shape as the image.
    """
    mask_radius = mask_radius_from_g(g_px, frac)
    psi = apply_mask_and_ifft(F_shifted, peak_rc, mask_radius,
                               include_friedel=False, taper_frac=taper_frac,
                               shift_to_dc=True, mask_type=mask_type,
                               n_sigma=n_sigma)
    return unwrap_phase(np.angle(psi))


# ═══════════════════════════════════════════════════════════════════════════════
#  Bragg-filtered image
# ═══════════════════════════════════════════════════════════════════════════════

def bragg_filtered_image(F_shifted, peak_rc, mask_radius, mask_type="gaussian",
                          n_sigma=3.0):
    """
    Reconstruct a Bragg-filtered (lattice-fringe) image for the g-vector at
    `peak_rc` by including BOTH the g and its Friedel pair (-g).

    Including -g ensures the result is real-valued, so fringes are directly
    visible as intensity modulations — terminating fringes mark dislocation
    cores. mask_type selects the aperture shape ("cosine": flat centre with
    a tapered edge and a hard cutoff at mask_radius; "gaussian": smooth
    falloff with sigma = mask_radius / n_sigma) — either suppresses the
    Fourier ringing a hard-edged disc would otherwise cause.

    Returns a real-valued 2D array normalised to [0, 1].
    """
    bragg = np.real(
        apply_mask_and_ifft(F_shifted, peak_rc, mask_radius,
                             include_friedel=True, shift_to_dc=False,
                             mask_type=mask_type, n_sigma=n_sigma)
    )

    # Normalise to [0, 1] for display
    bragg -= bragg.min()
    bragg /= bragg.max() + 1e-12
    return bragg


def bragg_filtered_image_from_frac(F_shifted, peak_rc, g_px, g_frac=0.5,
                                    taper_frac=0.25, mask_type="gaussian",
                                    n_sigma=3.0):
    """
    Convenience wrapper: run bragg_filtered_image() with the mask radius
    defined as a fraction of |g| instead of a fixed pixel value.

    Parameters
    ----------
    F_shifted : 2D complex array
    peak_rc : (row, col)
        Absolute pixel location of the g-spot in the shifted FFT.
    g_px : (row, col)
        g-vector offset from the FFT centre, used only to compute |g|.
    g_frac : float
        Mask (aperture) radius as a fraction of |g| (e.g. 0.5 → g/2,
        0.25 → g/4); see mask_radius_from_g(). Same meaning for both
        mask_type values.
    taper_frac : float
        Cosine-taper fraction for the mask edge (ignored if
        mask_type="gaussian").
    mask_type : {"cosine", "gaussian"}
    n_sigma : float
        Only used when mask_type="gaussian"; sigma = mask_radius / n_sigma.

    Returns
    -------
    Real-valued 2D array normalised to [0, 1].
    """
    mask_radius = mask_radius_from_g(g_px, g_frac)
    bragg = np.real(
        apply_mask_and_ifft(F_shifted, peak_rc, mask_radius,
                             include_friedel=True, taper_frac=taper_frac,
                             shift_to_dc=False, mask_type=mask_type,
                             n_sigma=n_sigma)
    )
    bragg -= bragg.min()
    bragg /= bragg.max() + 1e-12
    return bragg