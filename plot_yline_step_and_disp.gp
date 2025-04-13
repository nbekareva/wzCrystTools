#!/usr/bin/gnuplot -c

# Check if a filename was provided
if (ARG1 eq '') {
    print "Usage: gnuplot -c thisscript.gnu <datafile>"
    exit
}

# Set terminal to X11 (opens in a window, stays open until closed)
set terminal x11 persist

# Enable auto-scaling of axes
set autoscale

# Add labels and title
set xlabel 'Displacement, A'
set x2label 'Step'
set ylabel 'GSF, mJ/m2'
set title sprintf('Plot from %s', ARG1)

# Enable secondary x-axis
#set x2tics
set xtics nomirror tc linetype 1
set x2tics nomirror tc linetype 2

# Enable key/legend
set key outside right

# Plot both curves on same graph
plot ARG1 using 5:8 axes x1y1 with lines lt 1 title 'Col 1 vs 8', \
     ARG1 using 5:8 axes x1y1 with points pt 7 ps 0.7 notitle, \
     ARG1 using 3:8 axes x2y1 with lines lt 2 title 'Col 3 vs 8', \
     ARG1 using 3:8 axes x2y1 with points pt 7 ps 0.7 notitle
