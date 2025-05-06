#!/usr/bin/env python3
"""
Instant Multi Plot - Simple script to plot multiple columns from a file
Usage: python instant_multiplot.py filename xcol ycol1 [ycol2 ycol3 ...]

Examples:
  python instant_multiplot.py data.log 0 1 2 3    # Plot columns 1,2,3 against column 0
  python instant_multiplot.py data.log 0 6 7      # Plot columns 6,7 against column 0
"""

import sys
import matplotlib
matplotlib.use('Qt5Agg')        # Use the Qt5Agg backend for interactive plotting
import matplotlib.pyplot as plt
import numpy as np
import re

# First, try to set an interactive backend
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Try TkAgg first
except:
    try:
        matplotlib.use('Qt5Agg')  # Try Qt5Agg as fallback
    except:
        pass  # Let matplotlib choose its default

# Check command line arguments
if len(sys.argv) < 4:
    print("Usage: python instant_multiplot.py filename xcol ycol1 [ycol2 ycol3 ...]")
    print("Example: python instant_multiplot.py data.log 0 1 2 3")
    sys.exit(1)

filename = sys.argv[1]
xcol = int(sys.argv[2])
ycols = [int(arg) for arg in sys.argv[3:]]

# Define colors and markers for distinct lines
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'x']

# Read data from file
data = {xcol: []}
for ycol in ycols:
    data[ycol] = []

# Try to find column headers
headers = {}
header_line = None

with open(filename, 'r') as f:
    for line in f:
        line = line.strip()
        
        # Skip empty or comment lines
        if not line or line.startswith('#'):
            continue
        
        # Try to detect header line
        if "Step" in line or "Time" in line:
            header_line = line
            columns = re.split(r'\s+', line)
            columns = [col for col in columns if col]  # Remove empty strings
            for i, col in enumerate(columns):
                headers[i] = col
            continue
            
        # Try to match data lines starting with a number
        if re.match(r'^\s*-?\d+(\.\d*)?([eE][-+]?\d+)?\s', line):
            values = re.split(r'\s+', line)
            values = [v for v in values if v]  # Remove empty strings
            
            # Only process lines with enough columns
            if len(values) > max([xcol] + ycols):
                try:
                    x = float(values[xcol])
                    data[xcol].append(x)
                    
                    for ycol in ycols:
                        if ycol < len(values):
                            y = float(values[ycol])
                            data[ycol].append(y)
                        else:
                            data[ycol].append(float('nan'))
                except (ValueError, IndexError):
                    # Skip lines that can't be converted to float
                    pass

# Check if we got any data
if not data[xcol]:
    print("No valid data found!")
    sys.exit(1)

# Create the plot
plt.figure(figsize=(6, 4))

# Plot each y column against x
for i, ycol in enumerate(ycols):
    color_idx = i % len(colors)
    marker_idx = i % len(markers)
    
    # Get column labels if available
    x_label = headers.get(xcol, f"Column {xcol}")
    y_label = headers.get(ycol, f"Column {ycol}")
    
    # Filter out NaN values
    valid_indices = ~np.isnan(data[ycol])
    x_valid = [data[xcol][j] for j in range(len(data[xcol])) if valid_indices[j]]
    y_valid = [data[ycol][j] for j in range(len(data[ycol])) if valid_indices[j]]       # * 1.602176565e-9 / 10e-20 / 10e6  # Convert force/surf (eV/Angstrom^3 --> Pa --> MPa)
    
    if len(x_valid) > 0:
        plt.plot(x_valid, y_valid, 
                 marker=markers[marker_idx], 
                 markersize=1,
                 color=colors[color_idx], 
                 linestyle='-', 
                 label=y_label)

# Add labels and title
plt.xlabel(headers.get(xcol, f"Column {xcol}"))
plt.ylabel("Value")
plt.title(f"Multiple columns vs {headers.get(xcol, f'Column {xcol}')} from {filename}")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Adjust the plot for better visibility
if len(ycols) > 1:
    plt.legend(loc='best')

# Show the plot
try:
    plt.show()
except Exception as e:
    print(f"Error displaying plot: {e}")
    print("Saving to multiplot.png instead")
    plt.savefig("multiplot.png")
    print("Plot saved to multiplot.png")