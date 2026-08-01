#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def detect_image_color_type(image_path: str) -> tuple[str, str]:
	"""Return (mode, classification) for an image.

	classification is one of:
	- "black-and-white" for true grayscale content
	- "rgb" for color content
	"""
	path = Path(image_path)
	if not path.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")

	with Image.open(path) as img:
		mode = img.mode

		if mode in {"1", "L", "LA"}:
			return mode, "black-and-white"

		if mode in {"RGB", "RGBA"}:
			arr = np.array(img.convert("RGB"), dtype=np.uint8)
			r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

			# If all three channels are equal at each pixel, the content is grayscale.
			is_gray_content = np.array_equal(r, g) and np.array_equal(g, b)
			return mode, "black-and-white" if is_gray_content else "rgb"

		# Convert uncommon modes to RGB and inspect channel equality.
		arr = np.array(img.convert("RGB"), dtype=np.uint8)
		r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
		is_gray_content = np.array_equal(r, g) and np.array_equal(g, b)
		return mode, "black-and-white" if is_gray_content else "rgb"


def save_grayscale_tiff(image_path: str, output_path: str | None = None) -> Path:
	"""Convert image to 8-bit grayscale and save as TIFF."""
	input_path = Path(image_path)
	if not input_path.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")

	if output_path is None:
		out_path = input_path.with_suffix(".tiff")
	else:
		out_path = Path(output_path)

	with Image.open(input_path) as img:
		gray_img = img.convert("L")
		gray_img.save(out_path, format="TIFF")

	return out_path


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Check image color type, convert to grayscale, and save as TIFF."
	)
	parser.add_argument("image", help="Path to image file (e.g., .jpg)")
	parser.add_argument(
		"-o",
		"--output",
		help="Output TIFF path (default: same name as input with .tiff)",
	)
	args = parser.parse_args()

	mode, classification = detect_image_color_type(args.image)
	out_path = save_grayscale_tiff(args.image, args.output)
	print(f"Image: {args.image}")
	print(f"Pillow mode: {mode}")
	print(f"Detected: {classification}")
	print(f"Saved grayscale TIFF: {out_path}")


if __name__ == "__main__":
	main()
