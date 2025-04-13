#!/usr/bin/python3

import os
import re
import subprocess
import glob
from pathlib import Path
import shutil

def find_script_in_path(script_name):
    """
    Find a script in PATH environment.
    
    Args:
        script_name (str): Name of the script to find
        
    Returns:
        str: Full path to script if found, otherwise original name
    """
    # If it's already an absolute path or exists in current directory, return as is
    if os.path.isabs(script_name) or os.path.exists(script_name):
        return script_name
        
    # Look for the script in PATH
    return shutil.which(script_name)

def get_crystal_params(directory):
    """Read a, c parameters from rlx.data"""
    rlx_path = os.path.join(directory, '../../RELAX/rlx.data')      # TO DO: change relative path
    if not os.path.exists(rlx_path):
        return None, None
        
    with open(rlx_path, 'r') as f:
        f.readline()  # skip header
        data = f.readline().strip().split()
        lx, lz = float(data[4]), float(data[6])
        return lx/3, lz/3  # a, c parameters

def get_plot_details(directory):
    """
    Determine which plot script, which columns (and xy unit vectors for ysurf).
    
    Args:
        directory (str): Directory path to check
        
    Returns:
        dict: Plot configuration including script, columns, etc.
    """
    dir_name = os.path.basename(directory)
    matches = re.findall(r'\_([\w\d]+)', dir_name)
    plane_type = matches[-1] if matches else "unknown"

    # Get crystal parameters
    a, c = get_crystal_params(directory)
    
    # Unit formulae for different plane types
    unit_configs = {
        'basal': {'unit_x': lambda a,c: a, 'unit_y': lambda a,c: a},
        'prismatic_m': {'unit_x': lambda a,c: c, 'unit_y': lambda a,c: a},
        'prismatic_a': {'unit_x': lambda a,c: c, 'unit_y': lambda a,c: a},
        'pyramidalI': {'unit_x': lambda a,c: a, 'unit_y': lambda a,c: c},
        'pyramidalII': {'unit_x': lambda a,c: 2*a, 'unit_y': lambda a,c: c},
        'pyramidalIII': {'unit_x': lambda a,c: 2*a, 'unit_y': lambda a,c: c},
        'pyramidalIV': {'unit_x': lambda a,c: 2*a, 'unit_y': lambda a,c: c}
    }

    # Y-line plot configurations
    y_line_configs = {
        'gsfkey1': {'script': 'plot_yline.gp', 'columns': (5, 8)},
        'gsfkey2': {'script': 'plot_yline.gp', 'columns': (4, 8)},
        'gsfkey4': {'script': 'plot_yline.gp', 'columns': (4, 8)}
    }
    
    # Y-surface plot configuration
    y_surf_configs = {
        'gsfkey3': {
            'script': 'plot_ysurf.gp',
            'columns': None,
            'unit_x': unit_configs.get(plane_type, {}).get('unit_x'),
            'unit_y': unit_configs.get(plane_type, {}).get('unit_y')
        }
    }
    
    # Check y-line configs first
    for key, config in y_line_configs.items():
        if key in dir_name:
            return {'type': 'yline', 'plane_type': plane_type, **config}
    
    # Check y-surface configs
    for key, config in y_surf_configs.items():
        if key in dir_name:
            config = config.copy()
            if a and c and config['unit_x'] and config['unit_y']:
                config['unit_x'] = config['unit_x'](a, c)
                config['unit_y'] = config['unit_y'](a, c)
            return {'type': 'ysurf', 'plane_type': plane_type, **config}
    
    return None

def process_directories(base_path, y_line_output_dir, y_surf_output_dir):
    """
    Walk through directories matching patterns and run appropriate gnuplot scripts.
    
    Args:
        base_path (str): Base directory to start the search
        y_line_output_dir (str): Directory to save y-line PNG outputs
        y_surf_output_dir (str): Directory to save y-surface PNG outputs
    """
    # Create output directories if they don't exist
    os.makedirs(y_line_output_dir, exist_ok=True)
    os.makedirs(y_surf_output_dir, exist_ok=True)
    
    # Patterns to match
    patterns = ['*gsfkey4*', '*gsfkey1*', '*gsfkey2*', '*gsfkey3*']
    
    # Find all matching directories
    matching_dirs = set()
    for pattern in patterns:
        # Using glob to find matching directories
        matches = glob.glob(os.path.join(base_path, '**', pattern), recursive=True)
        matching_dirs.update([d for d in matches if os.path.isdir(d)])
    
    # Process each matching directory
    for directory in matching_dirs:
        # Determine plot configuration
        plot_config = get_plot_details(directory)
        if not plot_config:
            print(f"Skipping {directory}: No matching plot configuration")
            continue
        
        # Check for data file based on plot type
        data_file = os.path.join(directory, 'gamma.data.pt')
        if not os.path.exists(data_file):
            print(f"No gamma.data.pt found in {directory}")
            continue
        
        # Resolve plot script path
        plot_script_path = find_script_in_path(plot_config['script'])
        if not os.path.exists(plot_script_path):
            print(f"Could not find plot script: {plot_config['script']}")
            continue
        
        # Determine output directory and filename
        if plot_config['type'] == 'yline':
            output_dir = y_line_output_dir
        else:  # y-surface
            output_dir = y_surf_output_dir
        
        # Generate output filename
        rel_path = os.path.relpath(directory, base_path)
        output_file = os.path.join(output_dir, 
                                 f"{rel_path.replace(os.sep, '_')}.png")
        
        # Modify gnuplot script to output PNG
        temp_script = create_temp_script(
            plot_script_path, 
            base_path, 
            output_file, 
            plot_config.get('columns')
        )
        
        try:
            # Only get unit vectors for ysurfs              # TO DO: separate scripts
            args = [data_file, plot_config['plane_type']]
            if plot_config['type'] == 'ysurf':
                args.extend([str(plot_config.get('unit_x')), str(plot_config.get('unit_y'))])
            
            # Run the modified gnuplot script
            subprocess.run(['gnuplot', '-c', temp_script, data_file, plot_config['plane_type'], plot_config['unit_x'], plot_config['unit_y']], 
                         check=True,
                         stderr=subprocess.PIPE)
            print(f"Successfully processed: {directory}")
            print(f"Output saved to: {output_file}")
            if plot_config.get('columns'):
                print(f"Used columns {plot_config['columns'][0]}:{plot_config['columns'][1]}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {directory}: {e.stderr.decode()}")
        # finally:
            # Clean up temporary script
            # if os.path.exists(temp_script):
                # os.remove(temp_script)

def create_temp_script(original_script, base_path, output_png_file, columns=None, unit_x=None, unit_y=None):
    """
    Create a temporary gnuplot script with PNG output and optionally specified columns.
    
    Args:
        original_script (str): Path to original gnuplot script
        base_path (str): Base directory to start search
        output_png_file (str): Path for PNG output
        columns (tuple, optional): (x_column, y_column) to plot
        
    Returns:
        str: Path to temporary script
    """
    # Read original script
    with open(original_script, 'r') as f:
        script_content = f.read()
    
    # Replace terminal setting with PNG
    modified_content = script_content.replace(
        'set terminal x11 persist',
        f'set terminal png font "Arial,18" size 800,600\nset output "{output_png_file}"'
    )
    
    # Replace plotting columns if provided and applicable
    if columns:
        x_col, y_col = columns
        modified_content = modified_content.replace(
            'ARG1 using 5:8',
            f'ARG1 using {x_col}:{y_col}'
        )
        modified_content = modified_content.replace(
            'ARG1 using 5:6',
            f'ARG1 using {x_col}:6'
        )
    
    # Create temporary script
    temp_script = os.path.join(base_path, 'temp.gp')
    with open(temp_script, 'w') as f:
        f.write(modified_content)
    
    return temp_script

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process directories and run gnuplot scripts')
    parser.add_argument('base_path', help='Base directory to start search', default=".")
    parser.add_argument('y_line_output_dir', help='Directory to save y-line PNG outputs', default='y_lines')
    parser.add_argument('y_surf_output_dir', help='Directory to save y-surface PNG outputs', default='y_surfs')
    parser.add_argument('y_line_script', help='Path or name of y-line gnuplot script (can be in PATH)', 
                        default='plot_ysurf.gp')
    parser.add_argument('y_surf_script', help='Path or name of y-surface gnuplot script (can be in PATH)', 
                        nargs='?', default='plot_ysurf.gp')
    
    args = parser.parse_args()
    
    process_directories(
        args.base_path, 
        args.y_line_output_dir, 
        args.y_surf_output_dir
    )

if __name__ == "__main__":
    main()