#!/usr/bin/python3
import os, re, subprocess, glob

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
    dir_name = os.path.basename(directory)
    matches = re.findall(r'\_([\w\d]+)', dir_name)
    plane_type = matches[-1] if matches else "unknown"
    
    a, c = get_crystal_params(directory)
    
    unit_configs = {
        'basal': {'unit_x': lambda a,c: a, 'unit_y': lambda a,c: a},
        'prismatic_m': {'unit_x': lambda a,c: c, 'unit_y': lambda a,c: a},
        'prismatic_a': {'unit_x': lambda a,c: c, 'unit_y': lambda a,c: a},
        'pyramidalI': {'unit_x': lambda a,c: a, 'unit_y': lambda a,c: c},
        'pyramidalII': {'unit_x': lambda a,c: 2*a, 'unit_y': lambda a,c: c},
        'pyramidalIII': {'unit_x': lambda a,c: 2*a, 'unit_y': lambda a,c: c},
        'pyramidalIV': {'unit_x': lambda a,c: 2*a, 'unit_y': lambda a,c: c}
    }
    
    if a and c:
        unit_x = unit_configs.get(plane_type, {}).get('unit_x')
        unit_y = unit_configs.get(plane_type, {}).get('unit_y')
        if unit_x and unit_y:
            return plane_type, unit_x(a, c), unit_y(a, c)
    return plane_type, None, None

def process_directories(base_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Config for different yline types
    yline_configs = {
        'gsfkey1': {'columns': (5, 8)},
        'gsfkey2': {'columns': (4, 8)},
        'gsfkey4': {'columns': (4, 8)}
    }
    
    patterns = [f'*{key}*' for key in yline_configs.keys()]
    
    for pattern in patterns:
        for directory in glob.glob(os.path.join(base_path, '**', pattern), recursive=True):
            if not os.path.isdir(directory):
                continue
                
            plane_type, unit_x, unit_y = get_unit_vectors(directory)
            if not (unit_x and unit_y):
                print(f"Couldn't get unit vectors for {directory}")
                continue
                
            data_file = os.path.join(directory, 'gamma.data.pt')
            if not os.path.exists(data_file):
                print(f"No gamma.data.pt in {directory}")
                continue
                
            for key, config in yline_configs.items():
                if key in directory:
                    x_col, y_col = config['columns']
                    break
                    
            output_file = os.path.join(output_dir, 
                                     f"{os.path.relpath(directory, base_path).replace(os.sep, '_')}.png")
                                     
            try:
                subprocess.run(['gnuplot', '-c', 'plot_yline.gp', data_file, plane_type, 
                              str(unit_x), str(unit_y), str(x_col), str(y_col)], check=True)
                print(f"Processed: {directory}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {directory}: {e.stderr.decode()}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', nargs='?', default=".")
    parser.add_argument('output_dir', nargs='?', default='y_lines')
    args = parser.parse_args()
    process_directories(args.base_path, args.output_dir)

if __name__ == "__main__":
    main()