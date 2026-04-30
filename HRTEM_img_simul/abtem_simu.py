import sys

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('mypub')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # matplotlib's built-in default font
import numpy as np
import abtem
from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry import cellpar_to_cell
from ase.io import read
from PIL import Image
from math import ceil
from pathlib import Path
import re
import tifffile
from typing import Any
import shutil


# Atomic model


def _parse_jems_txt(path: Path) -> Atoms:
	"""Parse a JEMS text file into an ASE Atoms object.

	Expected records:
	- lattice|0..5|a,b,c,alpha,beta,gamma  (a,b,c in nm)
	- atom|idx|Element,Wyckoff,x,y,z,...   (x,y,z fractional)
	"""
	lines = path.read_text(encoding="utf-8").splitlines()

	lattice = {}
	symbols = []
	scaled_positions = []

	for raw in lines:
		line = raw.strip()
		if not line or "|" not in line:
			continue

		if line.startswith("lattice|"):
			parts = line.split("|")
			if len(parts) >= 3:
				lattice_idx = int(parts[1])
				lattice[lattice_idx] = float(parts[2])

		elif line.startswith("atom|"):
			parts = line.split("|", 2)
			if len(parts) < 3:
				continue
			fields = [x.strip() for x in parts[2].split(",")]
			if len(fields) < 5:
				continue

			symbol = fields[0].strip()
			fx, fy, fz = (float(fields[2]), float(fields[3]), float(fields[4]))
			symbols.append(symbol)
			scaled_positions.append([fx, fy, fz])

	required = {0, 1, 2, 3, 4, 5}
	if set(lattice) != required:
		missing = sorted(required - set(lattice))
		raise ValueError(f"Missing lattice entries in JEMS file: {missing}")

	if not symbols:
		raise ValueError("No atom entries were found in JEMS file.")

	# JEMS stores a,b,c in nm; ASE uses Angstrom.
	a = lattice[0] * 10.0
	b = lattice[1] * 10.0
	c = lattice[2] * 10.0
	alpha = lattice[3]
	beta = lattice[4]
	gamma = lattice[5]
	cell = cellpar_to_cell([a, b, c, alpha, beta, gamma])
	atoms = Atoms(
		symbols=symbols,
		scaled_positions=scaled_positions,
		cell=cell,
		pbc=True,
	)
	# Ensure a stable integer type array exists for downstream tools.
	symbol_to_type: dict[str, int] = {}
	types = []
	for symbol in atoms.get_chemical_symbols():
		if symbol not in symbol_to_type:
			symbol_to_type[symbol] = len(symbol_to_type) + 1
		types.append(symbol_to_type[symbol])
	atoms.set_array("type", np.array(types, dtype=int))
	return atoms


def _parse_lammps_type_map(path: Path) -> dict[int, int]:
	"""Read atomic numbers from the LAMMPS Masses section comments.

	Expected line format in Masses block, for example:
	1   65.38000000  # Zn
	"""
	lines = path.read_text(encoding="utf-8").splitlines()
	in_masses = False
	z_of_type: dict[int, int] = {}

	for raw in lines:
		line = raw.strip()

		if line.startswith("Masses"):
			in_masses = True
			continue

		if not in_masses:
			continue

		if not line:
			continue

		# Stop once the atom block starts.
		if line.startswith("Atoms"):
			break

		# type mass # Element
		match = re.match(r"^(\d+)\s+[\d.eE+-]+\s*(?:#\s*([A-Za-z][A-Za-z]?))?", line)
		if not match:
			continue

		atom_type = int(match.group(1))
		symbol = match.group(2)
		if symbol and symbol in atomic_numbers:
			z_of_type[atom_type] = atomic_numbers[symbol]

	return z_of_type


def save_grayscale_tiff(measurement: Any, output_path: Path) -> None:
	"""Save an abTEM measurement as an 16-bit grayscale TIFF --> use with Strain++."""
	img_array = np.flipud(measurement.array.T)
	assert img_array.ndim == 2, f"Expected 2D image, got shape {img_array.shape}"
	
	# Normalize to 16-bit for Strain++ (optional but conventional)
	img_norm = (img_array - img_array.min()) / (max(img_array.max() - img_array.min(), 1e-8))  # Avoid division by zero
	img_16 = (img_norm * 65535).astype(np.uint16)

	sampling = measurement.sampling[0]  # Å/pixel
	tifffile.imwrite(
        output_path,
        img_16,
        imagej=True,
        resolution=(1/sampling, 1/sampling),  # pixels/Å
        metadata={"unit": "Angstrom"}
    )


def _ensure_type_array(atoms: Atoms) -> Atoms:
	"""Guarantee integer `type` labels exist in atoms.arrays."""
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
	"""Load a structure from LAMMPS .lmp or JEMS .txt for abTEM use."""
	path = Path(path)
	suffix = path.suffix.lower()

	if suffix == ".lmp":
		z_of_type = _parse_lammps_type_map(path)
		kwargs: dict[str, Any] = {
			"format": "lammps-data",
			"atom_style": "atomic",
		}
		if z_of_type:
			kwargs["Z_of_type"] = z_of_type
		atoms_or_list = read(path, **kwargs)
		if isinstance(atoms_or_list, list):
			if not atoms_or_list:
				raise ValueError("LAMMPS file did not contain any structures.")
			return _ensure_type_array(atoms_or_list[0])
		return _ensure_type_array(atoms_or_list)

	if suffix == ".txt":
		# Only treat TXT as JEMS when expected tags are present.
		head = path.read_text(encoding="utf-8").splitlines()[:50]
		if any(line.startswith("atom|") for line in head) and any(
			line.startswith("lattice|") for line in head
		):
			return _ensure_type_array(_parse_jems_txt(path))
		raise ValueError("TXT file is not in the expected JEMS format.")

	raise ValueError(f"Unsupported structure format: {path.suffix}")


if __name__ == "__main__":
	try:
		filename = sys.argv[1]
		filename = Path(filename)
		sample_thickness = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
		pxl_size = float(sys.argv[3])
	except IndexError:
		print("Usage: python abtem_simu.py <structure_file> [sample_thickness (A)] [pixel_size (A/px)]")
		sys.exit(1)
	
	# outputs will be saved in a subfolder named after the input structure file
	out_folder = Path(filename.parent / filename.stem)
	out_folder.mkdir(parents=True, exist_ok=True)
	shutil.copy(filename, out_folder / filename.name)  # copy input structure to output folder for reference
	pot_path = out_folder / f"potential_t{sample_thickness:.2f}A.zarr"
	diffs_path = out_folder / f"diffraction_t{sample_thickness:.2f}A.png"
	img_path = out_folder / f"simulated_image_t{sample_thickness:.2f}A.tif"

	cryst = load_structure(filename)
	
	# Reduce atoms if calc too long
	# scaled_positions = cryst.get_scaled_positions()
	# mask = (scaled_positions[:, 0] >= 0.25) & (scaled_positions[:, 0] < 0.75) & \
	#        (scaled_positions[:, 1] >= 0.25) & (scaled_positions[:, 1] < 0.75)		# central quarter
	# cryst = cryst[mask]
	
	z_max = cryst.get_cell()[2, 2]
	n = ceil(sample_thickness / z_max)
	print(f"Sample thickness used: {n * z_max:.2f} Å")
	cryst = cryst * (1, 1, n)

	# rotate cell and atoms by 90 degrees around x
	# rotated_cryst = cryst.copy()
	# rotated_cryst.rotate("x", 90, rotate_cell=True)
	# rotated_cryst = abtem.standardize_cell(rotated_cryst)

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
	abtem.show_atoms(cryst, ax=ax1, plane="xy", title="Beam view", linewidths=0.1)
	abtem.show_atoms(cryst, ax=ax2, plane="yz", title="Side view", linewidths=0.1)
	abtem.show_atoms(cryst, ax=ax3, plane="xz", title="Side view", linewidths=0.1)
	plt.tight_layout()
	plt.savefig(out_folder / f"structure_view.png", facecolor='white', dpi=300)


	# 1. Potential: create a potential from the frozen phonons model
	frozen_phonons = abtem.FrozenPhonons(cryst, 8, sigmas=0.1)
	wave = abtem.PlaneWave(energy=300e3)
	print(f"Wavelength (relativistic): {wave.wavelength:.4f} Å")
	
	potential = abtem.Potential(
		frozen_phonons,
		sampling=0.1,				# finer, better physics
		projection="infinite",		# faster than finite
		slice_thickness=1,
		# exit_planes=10  	# output every 10th slice --> thickness series
	)

	# store the potential
	# potential_array = potential.build().compute()
	# potential_array.to_zarr(pot_path, overwrite=True)

	# 2. Multislice
	print("Running multislice algo...")
	exit_waves = wave.multislice(potential)		# DETECTOR TO ADD ?

	# 3. Diffraction patterns
	# msmts = exit_waves.diffraction_patterns(max_angle=10)		# up to a scattering angle of 10 mrad
	# print(f"Diffraction patterns shape: {msmts.shape}, extent: {msmts.extent}")

	# measurement = msmts.mean(0)		# average over frozen phonon configurations
	# measurement.compute()			# calc thickness series

	# visualization = measurement.block_direct().show(		# [::int(2//t)]
	# 	explode=True,
	# 	figsize=(18, 5),
	# 	cbar=True,
	# 	common_color_scale=True,
	# )
	# plt.savefig(diffs_path, facecolor='white', dpi=300)


	print("Calculating exit wave...")
	exit_waves.compute()


	# # 3. Imaging. CTF
	Cs = -8e-6 * 1e10  # spherical aberration (-8 um)
	ctf = abtem.CTF(Cs=Cs, energy=wave.energy, defocus="scherzer", semiangle_cutoff=45)
	print(f"defocus = {ctf.defocus:.2f} Å")

	# include partial temporal coherence effects using the chromatic aberration and energy spread
	Cc = 1.0e-3 * 1e10  # chromatic aberration (1.2 mm)
	energy_spread = 0.7  # standard deviation energy spread (0.35 eV)
	focal_spread = Cc * energy_spread / exit_waves.energy
	incoherent_ctf = ctf.copy()
	incoherent_ctf.focal_spread = focal_spread

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
	ctf.profiles().show(ax=ax1)
	incoherent_ctf.profiles().show(ax=ax2, legend=True)
	plt.tight_layout()
	plt.savefig(out_folder / f"ctf_profiles.png", facecolor='white', dpi=300)


	# 4. Apply CTF, calculate image intensity
	print("Applying CTF and calculating image intensity...")
	measurement_ensemble = exit_waves.apply_ctf(incoherent_ctf).intensity()
	print(f"Measurement ensemble shape: {measurement_ensemble.shape}")
	print(f"Measurement ensemble extent: {measurement_ensemble.extent}")
	# The result is an ensemble of images, one for each frozen phonon, 
	# we average the ensemble to obtain the final image
	measurement = measurement_ensemble.mean(0)
	
	# ── Downsample to experimental pixel size ──────────────────────────────────
	measurement = measurement.interpolate(sampling=pxl_size)   # Å/px from dm3 metadata
	# ──────────────────────────────────────────────────────────────────────────

	# fig, ax = plt.subplots(figsize=(5, 5))
	# measurement.show(ax=ax)
	# plt.tight_layout()
	# plt.savefig(filename.with_name(filename.stem + "_simulated_image.png"), facecolor='white', dpi=300)
	save_grayscale_tiff(measurement, img_path)