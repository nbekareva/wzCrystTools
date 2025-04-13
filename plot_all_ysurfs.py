#!/usr/bin/python3
from math import sqrt
import os, re, subprocess, glob
import numpy as np

plot_ysurf_path = "/home/nbekareva/TOOLS/utils/plot_ysurf.gp"

def get_crystal_params(directory):
    rlx_path = os.path.join(directory, '../../RELAX/rlx.data')
    if not os.path.exists(rlx_path):
        return None, None
    with open(rlx_path, 'r') as f:
        f.readline()  # skip header
        data = f.readline().strip().split()
        lx, lz = float(data[4]), float(data[6])
        return lx/3, lz/3

def get_unit_vectors(directory):
    a, c = get_crystal_params(directory)
    if a is None or c is None:          # if rel path different, e.g. reaxc
        # Use default values reaxc
        a, c = 9.84951288433398/3, 15.909450849089/3
    b = sqrt(3)*a
    ac = sqrt(a**2 + c**2)
    
    axes_configs = {
        'basal': {'unit_x': lambda: a, 'unit_y': lambda: b, "xdir": "a", "ydir": "b"},
        'prismatic_m': {'unit_x': lambda: a, 'unit_y': lambda: c, "xdir": "a", "ydir": "c"},
        'prismatic_a': {'unit_x': lambda: c, 'unit_y': lambda: b, "xdir": "c", "ydir": "b"},
        'pyramidalIV': {'unit_x': lambda: ac, 'unit_y': lambda: b, "xdir": "a+c", "ydir": "b"},
        'pyramidalIII': {'unit_x': lambda: a, 'unit_y': lambda: c, "xdir": "a", "ydir": "c"},
        'pyramidalII': {'unit_x': lambda: a, 'unit_y': lambda: sqrt(b**2+c**2), "xdir": "a", "ydir": "b+c"},
        'pyramidalI': {'unit_x': lambda: a, 'unit_y': lambda: sqrt((b/2)**2+c**2), "xdir": "a", "ydir": "b/2+c"}
    }
    
    dir_name = os.path.basename(directory)
    plane_type = next((key for key in axes_configs if key in dir_name), "unknown")
    
    if plane_type != "unknown":
        conf = axes_configs.get(plane_type)
        unit_x = conf.get('unit_x')
        unit_y = conf.get('unit_y')
        xdir = conf.get('xdir')
        ydir = conf.get('ydir')

        if unit_x and unit_y:
            return plane_type, unit_x(), unit_y(), xdir, ydir

    return plane_type, None, None, None, None

def analyze_gamma_data(data_file):
    """Analyze gamma data file to find maximum GSF values along X and Y axes at zero coordinate"""
    data = np.loadtxt(data_file, skiprows=1, usecols=(3,4,7))  # Load only needed columns (Y,X,Z)
    y_data, x_data, z_data = data.T  # Transpose for better performance
    
    # Find maxima along X=0 and Y=0 lines
    x_zero_mask = np.isclose(x_data, 0, rtol=1e-10)
    y_zero_mask = np.isclose(y_data, 0, rtol=1e-10)
    
    # Along Y at X=0
    z_along_y = z_data[x_zero_mask]
    y_coords = y_data[x_zero_mask]
    max_idx_y = np.argmax(z_along_y)
    max_z_along_y = (y_coords[max_idx_y], z_along_y[max_idx_y])
    
    # Along X at Y=0
    z_along_x = z_data[y_zero_mask]
    x_coords = x_data[y_zero_mask]
    max_idx_x = np.argmax(z_along_x)
    max_z_along_x = (x_coords[max_idx_x], z_along_x[max_idx_x])
    
    return max_z_along_x, max_z_along_y

def process_directories(base_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for directory in glob.glob(os.path.join(base_path, '**', '*gsfkey3*'), recursive=True):
        if not os.path.isdir(directory):
            continue
            
        plane_type, unit_x, unit_y, xdir, ydir = get_unit_vectors(directory)

        if not (unit_x and unit_y):
            print(f"Couldn't get unit vectors for {directory}")
            continue
            
        data_file = os.path.join(directory, 'gamma.data.pt')
        if not os.path.exists(data_file):
            print(f"No gamma.data.pt in {directory}")
            continue
            
        output_file = os.path.join(output_dir, 
                                 f"{os.path.relpath(directory, base_path).replace(os.sep, '_')}.png")
        gsf_shift = re.search(r"(_shift[\d\-.]*?)_", directory)
        plot_title = plane_type + "\\" + (gsf_shift.group() if gsf_shift else "_nonpolar")

        try:
            # Analyze GSF maxima
            max_x, max_y = analyze_gamma_data(data_file)
            print(f"\nGSF maxima for {plot_title}:")
            print(f"Along X (Y=0): {max_x[1]:.1f} mJ/m² at X={max_x[0]/unit_x:.3f}{xdir}")
            print(f"Along Y (X=0): {max_y[1]:.1f} mJ/m² at Y={max_y[0]/unit_y:.3f}{ydir}")
            
            # Generate plot
            subprocess.run(['gnuplot', '-c', plot_ysurf_path, data_file, plot_title, 
                          str(unit_x), str(unit_y), str(xdir), str(ydir), output_file], 
                         check=True)
            print(f"Processed: {directory}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing {directory}: {e.stderr.decode()}")
        except Exception as e:
            print(f"Error analyzing {directory}: {str(e)}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', nargs='?', default=".")
    parser.add_argument('output_dir', nargs='?', default='y_surfs')
    args = parser.parse_args()
    process_directories(args.base_path, args.output_dir)

if __name__ == "__main__":
    main()