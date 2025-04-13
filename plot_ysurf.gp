#!/usr/bin/gnuplot -c

# <datafile> = ARG1
# Check if a filename was provided
if (ARG1 eq '') {
    print "No file provided"
    print "Usage: plot_yline.gp <datafile> <plane_name> <unit_x> <unit_y> <xlabel> <ylabel> <output_png>"
    exit
}

# Get plot title
plane_name = ARG2
unit_x = ARG3 + 0  # Convert string to number
unit_y = ARG4 + 0
xdir = ARG5
ydir = ARG6

# Set terminal to X11 (opens in a window, stays open until closed)
# set terminal x11 persist
set terminal png font "Arial,18" size 800,600
set output ARG7

# Enable auto-scaling of axes
set autoscale fix
set cbrange [0:5000]

# Set a title for the plot
#set title "2D Surface Plot"

# Enable color mapping
set pm3d interpolate 2,2
set palette

# Set view for surface
set view map

# Remove margins
set lmargin 1
set rmargin 0
set tmargin 0
set bmargin 1
set format x "%.1f"
set format y "%.1f"

# Set labels for axes
set xlabel xdir offset 0
set ylabel ydir offset 0
set zlabel "GSFE, mJ/m2"
set title plane_name

# Use normalised columns 4, 5, and 8 from the data file
# Format: using 4:5:8 means x:y:z
splot ARG1 using ($5/unit_x):($4/unit_y):8 with pm3d notitle
