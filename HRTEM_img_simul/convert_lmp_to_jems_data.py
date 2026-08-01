#!/usr/bin/env python3
"""
Convert LMP (LAMMPS) files to JEMS-compatible TXT format using Atomsk.

This script:
1. Runs: atomsk input.lmp -prop jems_data_Zhang.txt -unit A nm JEMS
2. Post-processes the output to replace Wyckoff positions (a) with (_)
3. Appends ,Def.,0 to every atom line
4. Optionally removes XX placeholder atoms (see --help)

XX ATOMS (Placeholders):
When Atomsk processes the property file, some atom indices may fall outside the
valid range for the property mapping. These atoms are substituted with XX placeholders
at 0,0,0 coordinates. By default, the script removes these (--keep-xx to retain them).
Example Atomsk warning: "WARNING: atom index #22281 is out of bounds, skipping"
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Convert an LMP file to a JEMS-compatible TXT using atomsk."
	)
	parser.add_argument("input_lmp", type=Path, help="Path to the input .lmp file")
	parser.add_argument(
		"--keep-xx",
		action="store_true",
		help="Keep Atomsk XX placeholder atoms. Default: remove them. "
		"XX atoms have coordinates (0,0,0) and appear when atom indices "
		"exceed property file bounds (see Atomsk warnings: 'atom index #... is out of bounds').",
	)
	return parser.parse_args()


def run_atomsk(input_lmp: Path, prop_file: Path) -> Path:
	output_txt = input_lmp.with_suffix(".txt")
	cmd = [
		"atomsk",
		input_lmp.name,
		"-prop",
		str(prop_file),
		"-unit",
		"A",
		"nm",
		"JEMS",
		"-v",
		"2",
		"-ow"
	]

	try:
		subprocess.run(
			cmd,
			check=True,
			cwd=input_lmp.parent,
			capture_output=True,
			text=True,
		)
	except FileNotFoundError:
		raise RuntimeError("atomsk executable was not found in PATH")
	except subprocess.CalledProcessError as exc:
		err_text = (exc.stderr or exc.stdout or "").strip()
		if err_text:
			raise RuntimeError(f"atomsk failed: {err_text}")
		raise RuntimeError("atomsk failed with a non-zero exit code")

	if not output_txt.exists():
		raise RuntimeError(f"Expected output file was not created: {output_txt}")

	return output_txt


def _is_zero_coord_triplet(values: list[str]) -> bool:
	"""Check if atom has coordinates at (0, 0, 0) — typical of XX placeholders."""
	if len(values) < 5:
		return False

	try:
		x = float(values[2])
		y = float(values[3])
		z = float(values[4])
	except ValueError:
		return False

	return x == 0.0 and y == 0.0 and z == 0.0


def patch_jems_atom_lines(output_txt: Path, keep_xx: bool) -> tuple[int, int, int]:
	lines = output_txt.read_text(encoding="utf-8").splitlines()
	patched_lines: list[str] = []
	replaced_wyckoff = 0
	appended_def = 0
	removed_xx = 0

	for line in lines:
		if line.startswith("atom|"):
			parts = line.split("|", 2)
			if len(parts) == 3:
				atom_data = [item.strip() for item in parts[2].split(",")]

				if (
					not keep_xx
					and len(atom_data) >= 1
					and atom_data[0] == "XX"
					and _is_zero_coord_triplet(atom_data)
				):
					removed_xx += 1
					continue

				if len(atom_data) >= 2 and atom_data[1] == "a":
					atom_data[1] = "_"
					replaced_wyckoff += 1

				if len(atom_data) < 2 or atom_data[-2:] != ["Def", "0"]:
					atom_data.extend(["Def", "0"])
					appended_def += 1

				line = f"{parts[0]}|{parts[1]}|{','.join(atom_data)}"

		patched_lines.append(line)

	output_txt.write_text("\n".join(patched_lines) + "\n", encoding="utf-8")
	return replaced_wyckoff, appended_def, removed_xx


def main() -> int:
	args = parse_args()
	input_lmp = args.input_lmp.expanduser().resolve()

	if not input_lmp.exists():
		print(f"Input file does not exist: {input_lmp}", file=sys.stderr)
		return 1

	if input_lmp.suffix.lower() != ".lmp":
		print(f"Input file is not an .lmp file: {input_lmp}", file=sys.stderr)
		return 1

	script_dir = Path(__file__).resolve().parent
	prop_file = script_dir / "jems_data_Zhang.txt"

	if not prop_file.exists():
		print(f"Property file not found: {prop_file}", file=sys.stderr)
		return 1

	try:
		output_txt = run_atomsk(input_lmp, prop_file)
		replaced_wyckoff, appended_def, removed_xx = patch_jems_atom_lines(
			output_txt, keep_xx=args.keep_xx
		)
	except RuntimeError as exc:
		print(str(exc), file=sys.stderr)
		return 1

	print(f"JEMS-compatible file created: {output_txt}")
	print(f"  Wyckoff a → _: {replaced_wyckoff}")
	print(f"  Appended ,Def.,0: {appended_def}")
	if removed_xx > 0:
		print(f"  Removed XX placeholders: {removed_xx} "
		      f"(atom indices out of bounds in property file)")
	if args.keep_xx and removed_xx == 0:
		print("  XX placeholders kept (--keep-xx enabled).")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
