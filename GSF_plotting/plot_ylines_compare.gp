#!/usr/bin/gnuplot -c

# <datafile> = ARG1
# Check if files were provided
if (ARGC < 3) {
    print "No files provided"
    print "Usage: plot_multi.gp <datafile1> [datafile2 ...] output_png_file"
    exit
}

output_png_file = ARGV[ARGC]
print ("# files passed: %d", ARGC)
print ("Output to: %s", output_png_file)

# Set terminal to X11 (opens in a window, stays open until closed)
# set terminal x11 persist
set terminal png font "Arial,18" size 800,600
set output output_png_file

# Enable auto-scaling of axes
set autoscale
set yrange [0:*]

# Add labels and title (using filename as title)
set xlabel 'Displacement, A'
set ylabel 'GSFE, mJ/m2'
#set y2label 'fnorm, eV/Angstrom'
# set title sprintf('Plot from %s', ARG1)

# Enable secondary y-axis
set ytics nomirror

# Enable key/legend
set key right

files = ""
do for [i=1:ARGC-1] {
    if (i > 1) { files = files . " " }
    files = files . ARGV[i]
}

plot for [file in files] file using 5:8 with lines title file[34:54], \
     for [file in files] file using 5:8 with points pt 7 ps 0.7 notitle