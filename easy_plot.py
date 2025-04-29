#!/usr/bin/env python3
"""
Instant Plot - The simplest possible way to plot columns from a file
Usage: python instant_plot.py filename xcol ycol 

Super minimal one-liner plot script.
"""

import sys
import matplotlib
matplotlib.use('Qt5Agg')        # Use the Qt5Agg backend for interactive plotting
import matplotlib.pyplot as plt
import numpy as np
import re

if len(sys.argv) < 4:
    print("Usage: python instant_plot.py filename xcol ycol")
    sys.exit(1)

filename = sys.argv[1]
xcol = int(sys.argv[2])
ycol = int(sys.argv[3])

x_data = []
y_data = []

with open(filename, 'r') as f:
    for line in f:
        # Skip empty or comment lines
        if not line.strip() or line.strip().startswith('#'):
            continue
            
        # Try to match data lines starting with a number
        if re.match(r'^\s*-?\d+(\.\d*)?([eE][-+]?\d+)?\s', line):
            values = re.split(r'\s+', line.strip())
            values = [v for v in values if v]  # Remove empty strings
            
            # Only process lines with enough columns
            if len(values) > max(xcol, ycol):
                try:
                    x = float(values[xcol])
                    y = float(values[ycol])
                    # y = y * 1.602176565e-9 / 10e-20 / 10e6  # Convert force/surf (eV/Angstrom^3 --> Pa --> MPa)
                    x_data.append(x)
                    y_data.append(y)
                except (ValueError, IndexError):
                    # Skip lines that can't be converted to float
                    pass

if not x_data:
    print("No valid data found!")
    sys.exit(1)

plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'o-')
plt.grid(True)
plt.xlabel(f'Column {xcol}')
plt.ylabel(f'Column {ycol}')
plt.title(f'Plot of Column {ycol} vs Column {xcol} from {filename}')
plt.tight_layout()
plt.show()