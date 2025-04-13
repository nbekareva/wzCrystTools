#!/usr/bin/python3

import os
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

def get_plot_columns(directory):
    """
    Determine which columns to plot based on directory name.
    
    Args:
        directory (str): Directory path to check
        
    Returns:
        tuple: (x_column, y_column) to plot
    """
    dir_name = os.path.basename(directory)
    if  'gsfkey1' in dir_name:
        return (5, 8)
    elif 'gsfkey2' in dir_name or 'gsfkey4' in dir_name:
        return (4, 8)
    return None

def process_directories(base_path, output_dir, plot_script):
    """
    Walk through directories matching patterns and run gnuplot script on gamma.data.pt files.
    
    Args:
        base_path (str): Base directory to start the search
        output_dir (str): Directory to save PNG outputs
        plot_script (str): Path or name of the gnuplot script
    """
    # Resolve plot script path
    plot_script_path = find_script_in_path(plot_script)
    if not os.path.exists(plot_script_path):
        raise FileNotFoundError(f"Could not find plot script: {plot_script}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Patterns to match
    patterns = ['*gsfkey4*', '*gsfkey1*', '*gsfkey2*']
    
    # Find all matching directories
    matching_dirs = set()
    for pattern in patterns:
        # Using glob to find matching directories
        matches = glob.glob(os.path.join(base_path, '**', pattern), recursive=True)
        matching_dirs.update([d for d in matches if os.path.isdir(d)])
    
    # Process each matching directory
    for directory in matching_dirs:
        # Check if gamma.data.pt exists in the directory
        gamma_file = os.path.join(directory, 'gamma.data.pt')

        if os.path.exists(gamma_file):
            # Get columns to plot based on directory pattern
            columns = get_plot_columns(directory)
            if columns is None:
                print(f"Skipping {directory}: Unable to determine columns to plot")
                continue
                
            # Generate output filename based on directory structure
            rel_path = os.path.relpath(directory, base_path)
            output_file = os.path.join(output_dir, 
                                     f"{rel_path.replace(os.sep, '_')}.png")
            
            # Modify gnuplot script to output PNG and use correct columns
            temp_script = create_temp_script(plot_script_path, base_path, output_file, columns)
            
            try:
                # Run the modified gnuplot script
                subprocess.run(['gnuplot', '-c', temp_script, gamma_file], 
                             check=True,
                             stderr=subprocess.PIPE)
                print(f"Successfully processed: {directory}")
                print(f"Output saved to: {output_file}")
                print(f"Used columns {columns[0]}:{columns[1]}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {directory}: {e.stderr.decode()}")
            finally:
                # Clean up temporary script
                if os.path.exists(temp_script):
                    os.remove(temp_script)
        else:
            print(f"No gamma.data.pt found in {directory}")

def create_temp_script(original_script, base_path, output_png_file, columns):
    """
    Create a temporary gnuplot script with PNG output and specified columns.
    
    Args:
        original_script (str): Path to original gnuplot script
        base_path (str): Base directory to start search
        output_png_file (str): Path for PNG output
        columns (tuple): (x_column, y_column) to plot
        
    Returns:
        str: Path to temporary script
    """
    # Read original script
    with open(original_script, 'r') as f:
        script_content = f.read()
    
    # Replace terminal setting with PNG
    modified_content = script_content.replace(
        'set terminal x11 persist',
        f'set terminal png\nset output "{output_png_file}"'
    )
    
    # Replace plotting columns
    x_col, y_col = columns
    # GSF plotting on y axis
    modified_content = modified_content.replace(
        'using 5:8 with lines',
        f'using {x_col}:{y_col} with lines'
    )
    modified_content = modified_content.replace(
        'using 5:8 with points',
        f'using {x_col}:{y_col} with points'
    )
    # fnorm plotting on y2 axis - convergence/minim-n check
    modified_content = modified_content.replace(
        'using 5:6 with lines',
        f'using {x_col}:6 with lines'
    )
    modified_content = modified_content.replace(
        'using 5:6 with points',
        f'using {x_col}:6 with points'
    )

    # Create temporary script
    temp_script = os.path.join(base_path, 'temp.gp')
    with open(temp_script, 'w') as f:
        f.write(modified_content)
    
    return temp_script

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process directories and run gnuplot script')
    parser.add_argument('base_path', help='Base directory to start search')
    parser.add_argument('output_dir', help='Directory to save PNG outputs')
    parser.add_argument('plot_script', help='Path or name of gnuplot script (can be in PATH)')
    
    args = parser.parse_args()
    
    process_directories(args.base_path, args.output_dir, args.plot_script)

if __name__ == "__main__":
    main()