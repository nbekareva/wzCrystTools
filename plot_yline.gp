#!/usr/bin/gnuplot -c

# <datafile> = ARG1
# Check if a filename was provided
if (ARG1 eq '') {
    print "No file provided"
    print "Usage: plot_yline.gp <datafile>"
    exit
}

# Set terminal to X11 (opens in a window, stays open until closed)
set terminal x11 persist

# Enable auto-scaling of axes
set autoscale
set yrange [0:*]
# set y2range [0:*]
set logscale y2

# Add labels and title (using filename as title)
set xlabel 'Displacement, A'
set ylabel 'GSFE, mJ/m2'
set y2label 'fnorm, eV/Angstrom'
set title sprintf('Plot from %s', ARG1)

# Enable secondary y-axis
set ytics nomirror
set y2tics nomirror

# Enable key/legend
#set key outside right

# Plot command
    # fnorm plotting on y2 axis - convergence/minim-n check
plot ARG1 using 5:8 axes x1y1 with lines notitle, \
     ARG1 using 5:8 axes x1y1 with points pt 7 ps 0.7 notitle, \
     ARG1 using 5:6 axes x1y2 with lines notitle, \
     ARG1 using 5:6 axes x1y2 with points pt 7 ps 0.7 notitle
