#!/usr/bin/env python3
"""
ZnO Dislocation Generator Script

Creates a ZnO wurtzite structure with dislocations using Atomsk with a single command.
Supports edge, screw, and mixed dislocations.

Examples:
1. Py IV a+c edge
    python dislocation_generator.py --cif ZnO.cif --box-size 10 0.5607256448574929 10 \
        --burgers 6.144132635909867 --orientation [1-213] [10-10] [-12-11] --output-dir pyIV_a+c_edge \
        --iterate-positions --x-range 490 510 --z-range 490 510 --position-scale 1000
    python dislocation_generator.py --cif ZnO.cif --box-size 10 0.5607256448574929 10 \
        --burgers 6.144132635909867 --orientation [-12-11] [1-213] [10-10] --output-dir pyIV_a+c_edge1 \
        --disloc-pos 50.7 48.0 --jems-props jems_data.txt
2. Pr m a edge
    python dislocation_generator.py --cif ZnO.cif --box-size 10 0.522206130 10 \
        --burgers 3.23735102 --orientation [1-210] [0001] [10-10] --output-dir prm_a_edge \
        --iterate-positions --x-range 490 510 --z-range 490 510 --position-scale 1000
"""

import os
import argparse
import subprocess
import numpy as np 


def parse_cif_file(cif_path):
    """Extract lattice parameters from a CIF file"""
    a_param = c_param = None
    
    with open(cif_path, 'r') as f:
        for line in f:
            if '_cell_length_a' in line:
                a_param = float(line.split()[1])
            elif '_cell_length_c' in line:
                c_param = float(line.split()[1])
                
            if a_param is not None and c_param is not None:
                break
                
    return a_param, c_param


def create_elastic_file(orientation, output_path="elastic.txt"):
    """
    Create the elastic tensor file with provided values
    
    Args:
        output_path (str): Path to save the elastic tensor file
        orientation (list): List of three orientation vectors
    """
    try:
        with open(output_path, 'w') as f:
            # Write elastic tensor values
            f.write("elastic\n")
            f.write("209.7    121.1  105.1  0  0  0\n")
            f.write("121.1    209.7  105.1  0  0  0\n")
            f.write("105.1    105.1  210.9  0  0  0\n")
            f.write("0        0      0      42.47  0      0\n")
            f.write("0        0      0      0      42.47  0\n")
            f.write("0        0      0      0      0  44.29\n\n")
            
            # Write orientation
            f.write("orientation\n")
            for vector in orientation:
                f.write(f"{vector}\n")
    
        return output_path
    
    except Exception as e:
        print("ERROR: No orientation provided")
        return None


def run_atomsk_command(command):
    """Execute an Atomsk command and handle errors"""
    print(f"Running command: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
        print(f"Command executed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.stderr}")
        return False


def convert_to_jems_atomsk(lmp_file, jems_prop_file):
    try:
        command = f"yes no | atomsk {lmp_file} -rotate x -90 -prop {jems_prop_file} -unit A nm JEMS"
        print(command)
        result = subprocess.run(command, 
                                shell=True, check=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        print(f"Converted to JEMS txt successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not convert to JEMS: {e.stderr}")
        return False


def get_unit_cell_dimensions(lmp_file):
    """
    Extract the unit cell dimensions from a LAMMPS data file
    
    Args:
        lmp_file (str): Path to the LAMMPS data file
        
    Returns:
        tuple: (x_size, y_size, z_size) dimensions in Å
    """
    try:
        with open(lmp_file, 'r') as f:
            lines = f.readlines()
            
        x_dim = y_dim = z_dim = None
        
        for i, line in enumerate(lines):
            if "xlo xhi" in line:
                parts = line.split()
                x_dim = float(parts[1]) - float(parts[0])
            elif "ylo yhi" in line:
                parts = line.split()
                y_dim = float(parts[1]) - float(parts[0])
            elif "zlo zhi" in line:
                parts = line.split()
                z_dim = float(parts[1]) - float(parts[0])
                
            if x_dim is not None and y_dim is not None and z_dim is not None:
                break
                
        if x_dim is None or y_dim is None or z_dim is None:
            print("Could not find all dimensions in LAMMPS file")
            return None
            
        return (x_dim, y_dim, z_dim)
            
    except Exception as e:
        print(f"Error reading LAMMPS file: {e}")
        return None


def get_supercell_multiplicators(params, elastic_file):
    # First create the orthogonal unit cell to get its dimensions
    temp_ucell_file = os.path.join(params['output_dir'], "temp_ucell.lmp")

    # Build the initial command (stopping at orthogonal-cell)
    init_cmd = f"yes yes | atomsk --create wz {params['a']} {params['c']} Zn O orient {" ".join(params['orientation'])} {temp_ucell_file}"

    # Run the initial command to create the unit cell
    if not run_atomsk_command(init_cmd):
        print("Failed to create temporary unit cell")
        return False
    
    ucell_dims = get_unit_cell_dimensions(temp_ucell_file)
    params['nx'] = max(1, int(params['box_x_nm'] * 10 / ucell_dims[0]))  # Convert nm to Å
    params['ny'] = max(1, int(params['box_y_nm'] * 10 / ucell_dims[1]))
    params['nz'] = max(1, int(params['box_z_nm'] * 10 / ucell_dims[2]))

    return temp_ucell_file, params


def create_dislocation(params, pos_x=None, pos_y=None, i=None):
    """
    Create a dislocation structure using Atomsk
    
    Args:
        params (dict): Parameters for the dislocation
        pos_x (str, optional): X position of the dislocation, overrides params['pos_x']
        pos_y (str, optional): Y position of the dislocation, overrides params['pos_y']
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Use provided position if given, otherwise use default from params
    if pos_x is not None:
        params['pos_x'] = pos_x
    if pos_y is not None:
        params['pos_y'] = pos_y
    
    # Create output directory
    os.makedirs(params['output_dir'], exist_ok=True)
    
    # Create elastic tensor file
    elastic_file = os.path.join(params['output_dir'], "elastic.txt")
    create_elastic_file(params['orientation'], elastic_file)

    temp_ucell_file, params = get_supercell_multiplicators(params, elastic_file)
    
    # Create a filename that includes the position
    pos_str = f"_pos_{str(params['pos_x']).replace('*BOX', '')}_{str(params['pos_y']).replace('*BOX', '')}"
    pos_str = pos_str.replace('.', 'p')  # Replace dots with p for filename safety
    output_file = os.path.join(params['output_dir'], f"ZnO_dislocation.{i}.lmp")
    
    # Build the command
    cmd_parts = [
        f"yes yes |",       # rewrite existing file
        f"atomsk {temp_ucell_file}",
        f"-duplicate {params['nx']} {params['ny']} {params['nz']}",
        f"-prop {elastic_file}"
    ]
    
    # Add dislocation part based on type
    if params['disloc_type'] == "edge":
        cmd_parts.append(
            f"-disloc {params['pos_x']} {params['pos_y']} {params['disloc_type']} "
            f"y z {params['burgers'][0]} {params['poisson']}"
        )
    elif params['disloc_type'] == "screw":
        cmd_parts.append(
            f"-disloc {params['pos_x']} {params['pos_y']} {params['disloc_type']} "
            f"y z {params['burgers'][0]}"
        )
    elif params['disloc_type'] == "mixed":
        b1, b2, b3 = params['burgers']
        cmd_parts.append(
            f"-disloc {params['pos_x']} {params['pos_y']} {params['disloc_type']} "
            f"y z {b1} {b2} {b3}"
        )
    
    # Add output file
    cmd_parts.append(output_file)
    
    # Join all parts with spaces
    command = " ".join(cmd_parts)
    
    # Run the command
    success = run_atomsk_command(command)
    if success:
        print(f"Dislocation structure created: {output_file}")
        # convert to JEMS format
        convert_to_jems_atomsk(output_file, params['jems_props'])
        # Remove temporary unit cell file
        try:
            os.remove(temp_ucell_file)
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {e}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Create ZnO wurtzite structure with dislocation")
    
    # Input parameters
    parser.add_argument("--cif", type=str, help="CIF file with lattice parameters")
    parser.add_argument("--a", type=float, help="a lattice parameter (Å)")
    parser.add_argument("--c", type=float, help="c lattice parameter (Å)")
    parser.add_argument("--box-size", nargs=3, type=float, default=[20, 20, 20], 
                        help="Box size (nm)")
    parser.add_argument("--orientation", nargs=3, type=str, 
                        default=["[1-213]", "[10-10]", "[-12-11]"],
                        help="Crystal orientation vectors for elastic file")
    parser.add_argument("--poisson", type=float, default=0.35655, help="Poisson ratio")
    parser.add_argument("--burgers", type=float, nargs='+', required=True,
                        help="Burgers vector magnitude(s). For mixed, provide 3 values.")
    parser.add_argument("--disloc-pos", nargs=2, type=str, default=["0.501*BOX", "0.501*BOX"],
                        help="Dislocation position (x y)")
    parser.add_argument("--disloc-type", choices=["edge", "screw", "mixed"],
                        default="edge", help="Dislocation type")
    parser.add_argument("--jems-props", type=str, default="jems_data.txt",
                        help="JEMS data: D-W factors, Absorption, ...")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    
    # Iteration parameters
    parser.add_argument("--iterate-positions", action="store_true", 
                        help="Enable position iteration mode")
    parser.add_argument("--x-range", nargs=2, type=float, 
                        help="Range for x position iteration (start, end)")
    parser.add_argument("--z-range", nargs=2, type=float, 
                        help="Range for z position iteration (start, end)")
    parser.add_argument("--position-scale", type=int, default=10,
                        help="Scale factor for positions (default: 100 means 0.44 for 44)")
    
    args = parser.parse_args()
    
    # Get lattice parameters
    a_param = args.a
    c_param = args.c
    
    if args.cif:
        cif_a, cif_c = parse_cif_file(args.cif)
        if cif_a and cif_c:
            a_param = cif_a
            c_param = cif_c
            print(f"Using parameters from CIF: a = {a_param}, c = {c_param}")
    
    # Validate parameters
    if not a_param or not c_param:
        parser.error("Lattice parameters required (--a and --c, or --cif)")
    
    # Validate Burgers vector based on dislocation type
    burgers = args.burgers
    if args.disloc_type == "mixed" and len(burgers) != 3:
        parser.error("Mixed dislocation requires 3 Burgers vector components")
    elif args.disloc_type != "mixed" and len(burgers) > 1:
        burgers = burgers[0]  # Use only the first value
        print(f"Using only first Burgers vector component: {burgers}")
    
    # Prepare parameters dictionary
    params = {
        'a': a_param,
        'c': c_param,
        'box_x_nm': args.box_size[0],
        'box_y_nm': args.box_size[1],
        'box_z_nm': args.box_size[2],
        'orientation': args.orientation,
        'pos_x': args.disloc_pos[0],
        'pos_y': args.disloc_pos[1],
        'disloc_type': args.disloc_type,
        'burgers': burgers,
        'poisson': args.poisson,
        'jems_props': args.jems_props,
        'output_dir': args.output_dir
    }
    
    # Check if position iteration is enabled
    if args.iterate_positions:
        if not args.x_range or not args.z_range:
            parser.error("Both --x-range and --z-range must be provided for position iteration")
        
        # Create x and z ranges
        scale = args.position_scale
        xrange = np.linspace(args.x_range[0], args.x_range[1], scale)
        zrange = np.linspace(args.z_range[0], args.z_range[1], scale)
        # Create meshgrid for all (x,z) combinations
        X, Z = np.meshgrid(xrange, zrange)
        positions = np.column_stack((X.flatten(), Z.flatten()))
            
        # Create dislocations with positions in range
        for i, (pos_x, pos_z) in enumerate(positions):
            print(f"\n=== Iteration {i+1}: Position ({pos_x}, {pos_z}) ===")
            create_dislocation(params, pos_x, pos_z, i)
    else:
        # Just create a single dislocation with the provided parameters
        create_dislocation(params)


if __name__ == "__main__":
    main()