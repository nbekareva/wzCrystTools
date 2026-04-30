## Here you'll find:
./
JEMS simulation files and outputs

./*/ (subfolders)
abTEM simulation files and outputs

../*.py
Multislice simulation (abTEM) & GPA scripts. 
GPA in progress, currently done using GUI-based Strain++.

../atomsk_commands.txt
Helper commands to create cell from scratch or convert lmp --> jems txt.


## JEMS multislice simulation params:
1. Camera
   Real pixel size (old camera): 10.44 um
   y = Mag * [zone illuminée image] / #pxl

2. Cs = C3 = see session
   MT work with negative Cs
   Defocus > 0  ==  atoms white
      - Titan +upwards: current I ++ --> overfocus, I-- --> underfocus
      - JEMS defocus +downwards (e- path)
      - abTEM: same as JEMS

3. Cs & C5
   C5 = 6.8
   <!-- Mon Apr 27 17:24:11 CEST 2026   Prelim tests used C5=5.0 -->

4. Microscope (2nd button, 2nd top panel) --> Coherence
   - E spread = 0.7 eV
   - Lens stability = 0.5 ppm
   - Magn noise = 0.0
   - Voltage stability = 0.5 ppm
   - Beam half-convergence = 1.0 nm
   - Cc = 1.7 mm