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
from pathlib import Path
import re
from typing import Any


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
	"""Save an abTEM measurement as an 8-bit grayscale TIFF."""
	data = np.asarray(measurement.array)
	if data.ndim != 2:
		data = np.squeeze(data)
	if data.ndim != 2:
		raise ValueError(f"Expected a 2D image, got shape {data.shape}.")

	# abTEM measurements are indexed as (x, y), while image writers expect
	# (row=y, col=x). Swap axes so TIFF orientation matches measurement.show().
	data = data.T
	# Match display origin used by plotting backends/viewers.
	data = np.flipud(data)

	data_min = float(np.min(data))
	data_max = float(np.max(data))
	if data_max > data_min:
		normalized = (data - data_min) / (data_max - data_min)
	else:
		normalized = np.zeros_like(data)

	image_u8 = (normalized * 255.0).astype(np.uint8)
	Image.fromarray(image_u8, mode="L").save(output_path)


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
		sample_thickness = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
	except IndexError:
		print("Usage: python abtem_simu.py <structure_file> [sample_thickness (A)]")
		sys.exit(1)

	filename = Path(filename)
	cryst = load_structure(filename)
	z_max = cryst.get_cell()[2, 2]
	n = int(sample_thickness / z_max)
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
	plt.savefig(filename.with_name(filename.stem + "_structure_view.png"), facecolor='white', dpi=300)


	# 1. Potential: create a potential from the frozen phonons model
	frozen_phonons = abtem.FrozenPhonons(cryst, 16, sigmas=0.1)
	wave = abtem.PlaneWave(energy=300e3)
	print(f"Wavelength (relativistic): {wave.wavelength:.4f} Å")
	
	# + find convergence slice thickness
	for t in (6, 4):
		pot_path = filename.with_name(filename.stem + f"_potential_t{t:.2f}A.zarr")
		diffs_path = filename.with_name(filename.stem + f"_diffraction_t{t:.2f}A.png")

		potential = abtem.Potential(
			frozen_phonons,
			sampling=0.05,
			projection="infinite",		# faster than finite
			slice_thickness=t,
			exit_planes=10  	# output every 10th slice --> thickness series
		)

		# store the potential
		potential_array = potential.build().compute()
		potential_array.to_zarr(pot_path, overwrite=True)

		# 2. Multislice
		exit_waves = wave.multislice(potential)		# DETECTOR TO ADD ?
		msmts = exit_waves.diffraction_patterns(max_angle=10)		# up to a scattering angle of 10 mrad
		print(f"Diffraction patterns shape: {msmts.shape}, extent: {msmts.extent}")

		measurement = msmts.mean(0)		# average over frozen phonon configurations
		measurement.compute()			# calc thickness series


		visualization = measurement.block_direct().show(		# [::int(2//t)]
			explode=True,
			figsize=(18, 5),
			cbar=True,
			common_color_scale=True,
		)
		plt.savefig(diffs_path, facecolor='white', dpi=300)
