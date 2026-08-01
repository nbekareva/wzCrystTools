import json
import argparse

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('mypub')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import numpy as np
import abtem
from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry import cellpar_to_cell
from ase.io import read
from math import ceil
from pathlib import Path
import re
import tifffile
from typing import Any
import shutil


# ═══════════════════════════════════════════════════════════════════════════════
#  Atomic model loaders  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_jems_txt(path: Path) -> Atoms:
    """Parse a JEMS text file into an ASE Atoms object.
    
	Expected records:
	- lattice|0..5|a,b,c,alpha,beta,gamma  (a,b,c in nm)
	- atom|idx|Element,Wyckoff,x,y,z,...   (x,y,z fractional)
	"""
    lines = path.read_text(encoding="utf-8").splitlines()
    lattice: dict[int, float] = {}
    symbols = []
    scaled_positions = []
    for raw in lines:
        line = raw.strip()
        if not line or "|" not in line:
            continue
        if line.startswith("lattice|"):
            parts = line.split("|")
            if len(parts) >= 3:
                lattice[int(parts[1])] = float(parts[2])
        elif line.startswith("atom|"):
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            fields = [x.strip() for x in parts[2].split(",")]
            if len(fields) < 5:
                continue
            symbols.append(fields[0].strip())
            scaled_positions.append([float(fields[2]), float(fields[3]), float(fields[4])])
    required = {0, 1, 2, 3, 4, 5}
    if set(lattice) != required:
        raise ValueError(f"Missing lattice entries: {sorted(required - set(lattice))}")
    if not symbols:
        raise ValueError("No atom entries found in JEMS file.")
	# JEMS stores a,b,c in nm; ASE uses Angstrom.
    cell = cellpar_to_cell([
        lattice[0]*10, lattice[1]*10, lattice[2]*10,
        lattice[3], lattice[4], lattice[5],
    ])
    atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)
    symbol_to_type: dict[str, int] = {}
    types = []
    for sym in atoms.get_chemical_symbols():
        if sym not in symbol_to_type:
            symbol_to_type[sym] = len(symbol_to_type) + 1
        types.append(symbol_to_type[sym])
    atoms.set_array("type", np.array(types, dtype=int))
    return atoms


def _parse_lammps_type_map(path: Path) -> dict[int, int]:
    """Read atomic numbers from the LAMMPS Masses section comments."""
    lines = path.read_text(encoding="utf-8").splitlines()
    in_masses = False
    z_of_type: dict[int, int] = {}
    for raw in lines:
        line = raw.strip()
        if line.startswith("Masses"):
            in_masses = True; continue
        if not in_masses or not line:
            continue
        if line.startswith("Atoms"):
            break
        match = re.match(r"^(\d+)\s+[\d.eE+-]+\s*(?:#\s*([A-Za-z][A-Za-z]?))?", line)
        if match:
            sym = match.group(2)
            if sym and sym in atomic_numbers:
                z_of_type[int(match.group(1))] = atomic_numbers[sym]
    return z_of_type


def _ensure_type_array(atoms: Atoms) -> Atoms:
    if "type" in atoms.arrays:
        atoms.set_array("type", np.asarray(atoms.arrays["type"], dtype=int))
        return atoms
    numbers = atoms.get_atomic_numbers().tolist()
    z_to_type: dict[int, int] = {}
    types = []
    for z in numbers:
        if z not in z_to_type:
            z_to_type[z] = len(z_to_type) + 1
        types.append(z_to_type[z])
    atoms.set_array("type", np.array(types, dtype=int))
    return atoms


def load_structure(path: str | Path) -> Atoms:
    """Load a structure from LAMMPS .lmp or JEMS .txt."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".lmp":
        z_of_type = _parse_lammps_type_map(path)
        kwargs: dict[str, Any] = {"format": "lammps-data", "atom_style": "atomic"}
        if z_of_type:
            kwargs["Z_of_type"] = z_of_type
        result = read(path, **kwargs)
        atoms = result[0] if isinstance(result, list) else result
        return _ensure_type_array(atoms)
    if suffix == ".txt":
        head = path.read_text(encoding="utf-8").splitlines()[:50]
        if any(l.startswith("atom|") for l in head) and any(l.startswith("lattice|") for l in head):
            return _ensure_type_array(_parse_jems_txt(path))
        raise ValueError("TXT file is not in JEMS format.")
    raise ValueError(f"Unsupported format: {path.suffix}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TIFF output
# ═══════════════════════════════════════════════════════════════════════════════

def save_grayscale_tiff(measurement: Any, output_path: Path) -> None:
    """Save an abTEM measurement as a 16-bit grayscale TIFF (for Strain++)."""
    img_array = np.flipud(measurement.array.T)
    assert img_array.ndim == 2, f"Expected 2D image, got shape {img_array.shape}"
    img_norm = (img_array - img_array.min()) / max(img_array.max() - img_array.min(), 1e-8)
    img_16 = (img_norm * 65535).astype(np.uint16)
    sampling = measurement.sampling[0]   # Å/pixel
    tifffile.imwrite(
        output_path, img_16,
        imagej=True,
        resolution=(1/sampling, 1/sampling),
        metadata={"unit": "Angstrom"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Series grid
# ═══════════════════════════════════════════════════════════════════════════════

def save_series_grid(
    images: list[list[Any]],        # [i_thickness][j_defocus]
    thicknesses: list[float],       # Å, row labels
    defoci: list[float],            # Å, column labels
    pxl_size: float,                # Å/px — experimental pixel size
    out_folder: Path,
    label: str,
) -> None:
    """
    Downsample, save individual TIFFs, and produce a thickness × defocus
    grid PNG.

    Layout:
        columns → defocus values  (one series per thickness)
        rows    → thickness values
    """
    n_t = len(thicknesses)
    n_d = len(defoci)

    fig, axes = plt.subplots(
        n_t, n_d,
        figsize=(3.5 * n_d, 3.5 * n_t),
        squeeze=False,
    )
    fig.suptitle(f"{label}  —  Thickness × Defocus series", fontsize=12, y=1.01)

    for i, t in enumerate(thicknesses):
        for j, df in enumerate(defoci):
            msmt = images[i][j]

            # Downsample to experimental pixel size
            msmt_ds = msmt.interpolate(sampling=pxl_size)

            # Individual TIFF
            tiff_name = f"{label}_t{t:.0f}A_df{df:+.0f}A.tif"
            save_grayscale_tiff(msmt_ds, out_folder / tiff_name)
            print(f"    saved {tiff_name}")

            # Panel
            ax = axes[i][j]
            arr = np.flipud(msmt_ds.array.T)
            ax.imshow(arr, cmap="gray",
                      vmin=np.percentile(arr, 2),
                      vmax=np.percentile(arr, 98),
                      origin="upper")
            ax.set_xticks([]); ax.set_yticks([])

            # Column header: defocus (top row only)
            if i == 0:
                ax.set_title(f"Δf = {df:+.0f} Å", fontsize=9)
            # Row header: thickness (left column only)
            if j == 0:
                ax.set_ylabel(f"t = {t:.0f} Å", fontsize=9)

    plt.tight_layout()
    grid_path = out_folder / f"{label}_series_grid.png"
    plt.savefig(grid_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Grid figure → {grid_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",")]


def build_ctf(energy: float, aberration_coefficients: dict, focal_spread: float, plot: bool = False, out_dir: Path = None) -> abtem.CTF:
    ctf = abtem.CTF(energy=energy, aberration_coefficients=aberration_coefficients, semiangle_cutoff=45)
    ctf0 = ctf.copy()
    ctf.focal_spread = focal_spread
    defocus = aberration_coefficients.get("C10", 0.0)
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        ctf0.profiles().show(ax=ax1, legend=True)
        ctf.profiles().show(ax=ax2, legend=True)
        ax1.set_title("Coherent CTF")
        ax2.set_title("With temporal coherence envelope")
        fig.tight_layout()
        if out_dir is not None:
            fig.savefig(out_dir / f"ctf_profiles_df{defocus:+.0f}A.png", facecolor="white", dpi=300)
        else:
            fig.savefig(f"ctf_profiles_df{defocus:+.0f}A.png", facecolor="white", dpi=300)
    return ctf


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="abTEM HRTEM simulation — single image or thickness×defocus series."
    )
    p.add_argument("structure",
                   help="Path to .lmp or JEMS .txt structure file")
    p.add_argument("pxl_size", type=float,
                   help="Experimental pixel size in Å/px (used for downsampling)")
    p.add_argument("--thickness", type=float, default=200.0,
                   help="Sample thickness in Å — single-image mode (default: 200)")

    # ── Series options ─────────────────────────────────────────────────────────
    p.add_argument(
        "--thickness-series", default=None, metavar="T1,T2,...",
        help=(
            "Comma-separated thicknesses in Å to include in the series.\n"
            "Example: --thickness-series 50,100,150,200\n"
            "The structure is replicated to cover the largest value."
        ),
    )
    p.add_argument(
        "--defocus-series", default=None, metavar="D1,D2,...",
        help=(
            "Comma-separated defocus values in Å (positive = underfocus).\n"
            "Example: --defocus-series -300,-150,0,150\n"
            "If omitted in series mode, Scherzer defocus is used for all thicknesses."
        ),
    )
    args = p.parse_args()

    filename   = Path(args.structure)
    pxl_size   = args.pxl_size
    series_mode = args.thickness_series is not None or args.defocus_series is not None

    # ── Resolve thickness and defocus lists ────────────────────────────────────
    if series_mode:
        thicknesses = (parse_float_list(args.thickness_series)
                       if args.thickness_series else [args.thickness])
        # defoci resolved after we know Scherzer defocus (below)
    else:
        thicknesses = [args.thickness]

    # ── Output folder ──────────────────────────────────────────────────────────
    out_folder = filename.parent / filename.stem
    out_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(filename, out_folder / filename.name)

    img_stem = filename.stem    # used in filenames throughout

    # ── Load & replicate structure ─────────────────────────────────────────────
    cryst = load_structure(filename)
    z_max = cryst.get_cell()[2, 2]          # unit cell extent along beam axis
    cryst_base = cryst                      # keep the unreplicated unit cell

    max_thickness = max(thicknesses)
    n_repeat = ceil(max_thickness / z_max)
    actual_max_t = n_repeat * z_max
    print(f"Unit cell z = {z_max:.3f} Å  |  "
          f"max requested = {max_thickness:.0f} Å  |  "
          f"replicated ×{n_repeat} → {actual_max_t:.2f} Å total")
    cryst = cryst_base * (1, 1, n_repeat)   # only for structure-view figure

    # ── Structure views ────────────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    abtem.show_atoms(cryst, ax=ax1, plane="xy", title="Beam view",  linewidths=0.1)
    abtem.show_atoms(cryst, ax=ax2, plane="yz", title="Side view",  linewidths=0.1)
    abtem.show_atoms(cryst, ax=ax3, plane="xz", title="Side view",  linewidths=0.1)
    plt.tight_layout()
    plt.savefig(out_folder / "structure_view.png", facecolor="white", dpi=300)
    plt.close()

    # ── Microscope / CTF parameters ────────────────────────────────────────────
    # Cs           = -8e-6 * 1e10    # spherical aberration in Å  (-8 µm)
    # C5		     = 6.8e-3 * 1e10   # fifth-order spherical aberration in Å  (6.8 mm)
    # Cc           = 1.7e-3 * 1e10   # chromatic aberration in Å  (1.7 mm)
    with open("2024-05-14_tem_params.json") as f:
        tem_params = json.load(f)
        
    aberration_coefficients = tem_params["aberration_coefficients"]
    Cs = aberration_coefficients['C3_Cs'] * 1e10
    C5 = aberration_coefficients['C5'] * 1e10

    Cc = tem_params['microscope_params']['Cc'] * 1e10
    voltage = tem_params['microscope_params']['acceleration_voltage_eV']
    energy_spread = tem_params['microscope_params']['energy_spread_eV']
    wave         = abtem.PlaneWave(energy=voltage)
    focal_spread = Cc * energy_spread / wave.energy
    print(f"Wavelength (relativistic): {wave.wavelength:.4f} Å")

    # Scherzer reference
    ctf_scherzer = abtem.CTF(Cs=Cs, energy=wave.energy,
                              defocus="scherzer", semiangle_cutoff=45)
    ctf_scherzer0 = ctf_scherzer.copy()
    ctf_scherzer.focal_spread = focal_spread
    scherzer_df = ctf_scherzer.defocus
    print(f"Scherzer defocus = {scherzer_df:.2f} Å")

    # Resolve defoci now that we have Scherzer value
    if series_mode:
        defoci = (parse_float_list(args.defocus_series)
                  if args.defocus_series else [scherzer_df])
    else:
        defoci = [scherzer_df]

    # ── CTF profiles figure ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    ctf_scherzer0.profiles().show(ax=ax1, legend=True)
    ctf_scherzer.profiles().show(ax=ax2, legend=True)
    ax1.set_title("Coherent CTF (Scherzer)")
    ax2.set_title("With temporal coherence envelope")
    plt.tight_layout()
    plt.savefig(out_folder / "ctf_scherzer.png", facecolor="white", dpi=300)
    plt.close()

    # ── Potential & multislice — one run per requested thickness ──────────────
    # Using exit_planes is fragile across abTEM versions; running multislice
    # independently per thickness is unambiguous and version-agnostic.
    SLICE_THICKNESS = 1.0   # Å

    # ── Snap each requested thickness to a whole number of unit cells ──────────
    # We replicate the base crystal to the minimum height that covers each T,
    # so the structure boundary coincides with a real crystallographic plane.
    actual_thicknesses: list[float] = []
    for t in thicknesses:
        n = ceil(t / z_max)
        actual_thicknesses.append(n * z_max)
        if abs(n * z_max - t) > 1.0:
            print(f"  ⚠  t={t:.0f} Å → nearest unit-cell boundary "
                  f"t={n*z_max:.1f} Å ({n} repeats)")

    print(f"\nSeries: {len(actual_thicknesses)} thickness(es) × {len(defoci)} defocus value(s)")
    print(f"  Thicknesses : {[f'{t:.1f}' for t in actual_thicknesses]} Å")
    print(f"  Defoci      : {[f'{d:+.0f}' for d in defoci]} Å\n")

    images: list[list[Any]] = []

    for i, (t_req, t_act) in enumerate(zip(thicknesses, actual_thicknesses)):
        n_rep = ceil(t_act / z_max)
        cryst_t = cryst_base * (1, 1, n_rep)
        print(f"  [{i+1}/{len(thicknesses)}] t = {t_act:.1f} Å  ({n_rep} unit cells) …")

        frozen_phonons_t = abtem.FrozenPhonons(cryst_t, 8, sigmas=0.1)
        potential_t = abtem.Potential(
            frozen_phonons_t,
            sampling=0.1,
            projection="infinite",
            slice_thickness=SLICE_THICKNESS,
        )

        exit_waves_t = wave.multislice(potential_t)
        exit_waves_t.compute()

        row: list[Any] = []
        for df in defoci:
            aberration_coefficients = {"C10": df, "C30": Cs, "C50": C5}
            ctf = build_ctf(wave.energy, aberration_coefficients, focal_spread, plot=True, out_dir=out_folder)
            msmt = exit_waves_t.apply_ctf(ctf).intensity().mean(0)
            row.append(msmt)
        images.append(row)

    if series_mode:
        save_series_grid(
            images,
            thicknesses=actual_thicknesses,
            defoci=defoci,
            pxl_size=pxl_size,
            out_folder=out_folder,
            label=img_stem,
        )
    else:
        # Single-image path — images[0][0] is the only entry
        img_path = out_folder / f"{img_stem}_t{actual_thicknesses[0]:.0f}A.tif"
        msmt = images[0][0].interpolate(sampling=pxl_size)
        save_grayscale_tiff(msmt, img_path)
        print(f"  Image saved → {img_path.name}")

    print("\n✓ Done.")