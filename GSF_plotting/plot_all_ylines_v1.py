#!/usr/bin/python3

import os
import subprocess
import glob
from pathlib import Path

def create_modified_gnuplot_script(original_script_path, output_dir):
    """
    Create a modified version of the gnuplot script that outputs to PNG
    """
    # Create the modified script directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    modified_script_path = os.path.join(output_dir, 'modified_plot_ysurf.gp')
    
    with open(original_script_path, 'r') as f:
        content = f.read()
    
    # Replace the X11 terminal line with PNG terminal
    modified_content = content.replace(
        'set terminal x11 persist',
        'set terminal png size 1024,768'
    )
    
    # Add output line after terminal setting
    modified_content = modified_content.replace(
        'set terminal png size 1024,768\n',
        'set terminal png size 1024,768\nset output ARG2\n'
    )
    
    with open(modified_script_path, 'w') as f:
        f.write(modified_content)
    
    # Make the script executable
    os.chmod(modified_script_path, 0o755)
    return modified_script_path

def process_directories(base_dir, output_dir, gnuplot_script_path):
    """
    Walk through directories and process matching ones
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create modified gnuplot script
    modified_script_path = create_modified_gnuplot_script(gnuplot_script_path, output_dir)
    
    # Pattern matching for directory names
    patterns = ['*gsfkey4*', '*gsfkey1*', '*gsfkey2*']
    
    # Find all matching directories
    matching_dirs = set()
    for pattern in patterns:
        matching_dirs.update(
            str(p) for p in Path(base_dir).rglob(pattern) if p.is_dir()
        )
    
    # Process each matching directory
    for dir_path in matching_dirs:
        # Construct the path to gamma.data.pt
        data_file = os.path.join(dir_path, 'gamma.data.pt')
        
        if os.path.exists(data_file):
            # Create output filename based on directory name
            dir_name = os.path.basename(dir_path)
            output_file = os.path.join(output_dir, f'{dir_name}.png')
            
            # Run the modified gnuplot script
            try:
                subprocess.run([
                    modified_script_path,
                    data_file,
                    output_file
                ], check=True)
                print(f"Successfully processed {dir_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {dir_path}: {e}")
        else:
            print(f"No gamma.data.pt found in {dir_path}")

def main():
    # Configuration
    base_dir = "."  # Current directory, modify as needed
    output_dir = "./yline_plot_outputs"  # Output directory for PNG files
    gnuplot_script_path = "/home/nbekareva/SIM/ZnO_interatomlic_models/pytools/plot_yline.gp"  # Path to original gnuplot script
    
    # Process the directories
    process_directories(base_dir, output_dir, gnuplot_script_path)

if __name__ == "__main__":
    main()