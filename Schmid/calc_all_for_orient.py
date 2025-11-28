#!/home/pypotprop/bin/python3

import glob
import math as m
import re
import os
import sys
sys.path.append("..")  # Add parent directory to path
import numpy as np
from hexag_cristallo_tool_gamma import Wurtzite
from plot_all import plot_schmid_factors

def get_gsf_barrier(plane_type, dir_type, gsf_base_dir):
    """
    Get the minimum GSF barrier for a specific plane and direction from all available shifts.
    
    Args:
        plane_type: str, type of crystallographic plane
        dir_type: str, direction type ('a', 'b', 'c', etc.)
        gsf_base_dir: str, path to directory containing all GSF calculations
    
    Returns:
        tuple: (float: minimum GSF barrier value in mJ/m², str: shift value used)
    """
    # Define the axes configurations mapping
    axes_configs = {
        'basal': {'a': 'x', 'b': 'y'},
        'prismatic_m': {'a': 'x', 'c': 'y'},
        'prismatic_a': {'c': 'x', 'b': 'y'},
        'pyramidalIV': {'a+c': 'x', 'b': 'y'},
        'pyramidalIII': {'a': 'x', 'c': 'y'},
        'pyramidalII': {'a': 'x', 'b+c': 'y'},
        'pyramidalI': {'a': 'x', 'b/2+c': 'y'}
    }
    
    # Get the mapping for this plane type
    plane_config = axes_configs.get(plane_type)
    if not plane_config:
        return None, None
        
    # Determine if we should look along x or y axis
    axis = plane_config.get(dir_type)
    if not axis:
        return None, None

    # Find all relevant gamma.data.pt files for this plane type
    pattern = os.path.join(gsf_base_dir, f"*_gsfkey3_*_{plane_type}", "gamma.data.pt")
    gsf_files = glob.glob(pattern)
    
    if not gsf_files:
        return None, None
        
    min_barrier = float('inf')
    min_shift = None
    
    for gsf_file in gsf_files:
        try:
            # Extract shift value from directory name
            dir_name = os.path.basename(os.path.dirname(gsf_file))
            shift_match = re.search(r"shift([-\d.]+)", dir_name)
            shift_value = shift_match.group(1) if shift_match else "0.001"
            
            # Load gamma surface data
            data = np.loadtxt(gsf_file, skiprows=1, usecols=(3,4,7))
            y_data, x_data, z_data = data.T
            
            # Find maximum along the appropriate axis
            if axis == 'x':
                mask = np.isclose(y_data, 0, rtol=1e-10)
                relevant_data = z_data[mask]
            else:  # axis == 'y'
                mask = np.isclose(x_data, 0, rtol=1e-10)
                relevant_data = z_data[mask]
            
            if len(relevant_data) > 0:
                barrier = np.max(relevant_data)
                if barrier < min_barrier:
                    min_barrier = barrier
                    min_shift = shift_value
                    
        except Exception as e:
            print(f"Error reading GSF data from {gsf_file}: {str(e)}")
            continue
    
    if min_barrier == float('inf'):
        return None, None
        
    return min_barrier, min_shift



# Initialize crystal
a = 3.2494    # 3.238118214981954
c = 5.2054    # 5.176434187914933
crystal = Wurtzite(a, c)      # initialize a hexagonal crystalline system
# print(f"Wurtzite, Wang 2014 relaxed crystal: a {a} A, c {c} A")
print(crystal.generic_directions)

# print(crystal.generic_slip_modes.values())


# 1. BUILD A DICT OF SLIP SYSTEMS - LIST ALL
# SS_file = open("slip_system_dict.txt", "w")
# # crystal.slip_systems = {}

# for plane_conv_name, generic_plane in crystal.generic_slip_modes.items():
#     for i, plane in enumerate(crystal.list_equiv(generic_plane)):   # for every equiv plane
#         # crystal.slip_systems[plane] = []
#         n = crystal.n_to_hkil(plane)
        
#         for b_conv_name, generic_b in crystal.generic_perfect_Burgers.items():
#             for j, b in enumerate(crystal.list_equiv(generic_b)):   # for every equiv Burgers
#                 cos_phi, _ = crystal.angle_bw_directions(n, b)
#                 if cos_phi == 0:
#                     plane_str = " ".join(str(idx) for idx in plane)
#                     b_str = " ".join(str(idx) for idx in b)
#                     b_norm = crystal.norm(b)
#                     if b_conv_name == 'a' or b_conv_name == 'a+c' or b_conv_name == 'b2_PyIII':
#                         b_norm = b_norm / 3
#                     # print(f"{plane_conv_name} {i+1}\t {b_conv_name} {j+1}\t {plane_str}\t {b_str}")
#                     SS_file.write(f"{plane_conv_name} {i+1}\t {b_conv_name} {j+1}\t {plane_str}\t {b_str}\n")
#                     # crystal.slip_systems[slip_plane].append(b)
#                 else:
#                     continue

# SS_file.close()


# WORK IN PROGRESS: possibly error 
# def get_ST_angle(slip_plane, pillar_axis, observ_dir):
#     [slip_plane, pillar_axis, observ_dir] = [crystal.PlaneBravaisToMiller(vect) for vect in [slip_plane, pillar_axis, observ_dir]]
#     print(slip_plane, pillar_axis, observ_dir)
#     zone_axis = np.cross(slip_plane, observ_dir)
#     cos_xi = np.dot(pillar_axis, zone_axis) / (np.linalg.norm(pillar_axis) * np.linalg.norm(zone_axis))
#     cos_xi = round(cos_xi, 10)                  # angle bw slip trace & pillar axis
#     xi = round(m.degrees(m.acos(cos_xi)), 8)
#     print(crystal.VectorMillerToBravais(zone_axis), xi, "\n")
#     return xi


# 2. CALC SCHMID FACTORS FOR SPECIF ORIENTATIONS
sample_orienations = [
    {'sample': '0 0 0 1', 'face_polished': None, 'offset_angle': 0.0, 'orient': [0, 0, 0, 1]},
    {'sample': '0 0 0 1', 'face_polished': '1 0 -1 0', 'offset_angle': 20.0, 'orient': [1, 0, -1, 2.969744897959184]},
    {'sample': '0 0 0 1', 'face_polished': '1 0 -1 0', 'offset_angle': 45.0, 'orient': [1, 0, -1, 1.0808928571428573]},
    {'sample': '0 0 0 1', 'face_polished': '1 0 -1 0', 'offset_angle': 70.0, 'orient': [1, 0, -1, 0.39385474860335207]},
    {'sample': '0 0 0 1', 'face_polished': '1 -2 1 0', 'offset_angle': 20.0, 'orient': [1, 1, -2, 5.143714285714286]},
    {'sample': '0 0 0 1', 'face_polished': '1 -2 1 0', 'offset_angle': 45.0, 'orient': [1, 1, -2, 1.8721632653061226]},
    {'sample': '0 0 0 1', 'face_polished': '1 -2 1 0', 'offset_angle': 70.0, 'orient': [1, -2, 1, 0.6801346801346804]},

    {'sample': '1 0 -1 0', 'face_polished': None, 'offset_angle': 0.0, 'orient': [1, 0, -1, 0]},
    {'sample': '1 0 -1 0', 'face_polished': '0 0 0 1', 'offset_angle': 20.0, 'orient': [1, 0, -1, 0.39385474860335185]},
    {'sample': '1 0 -1 0', 'face_polished': '0 0 0 1', 'offset_angle': 45.0, 'orient': [1, 0, -1, 1.0808928571428573]},
    {'sample': '1 0 -1 0', 'face_polished': '0 0 0 1', 'offset_angle': 70.0, 'orient': [1, 0, -1, 2.9697346938775513]},
    {'sample': '1 0 -1 0', 'face_polished': '1 -2 1 0', 'offset_angle': 20.0, 'orient': [1, -0.3486973947895791, -0.6513026052104209, 0]},
    {'sample': '1 0 -1 0', 'face_polished': '1 -2 1 0', 'offset_angle': 45.0, 'orient':  [1, -0.7334669338677354, -0.2665330661322645, 0]},
    {'sample': '1 0 -1 0', 'face_polished': '1 -2 1 0', 'offset_angle': 70.0, 'orient': [1, -1.2264529058116231, 0.22645290581162314, 0]},
    
    {'sample': '1 1 -2 0', 'face_polished': None, 'offset_angle': 0.0, 'orient': [1, 1, -2, 0]},
    {'sample': '1 1 -2 0', 'face_polished': '0 0 0 1', 'offset_angle': 20.0, 'orient': [1.0, 1.0, -2.0, 0.6864406779661016]},   # 0.1358°
    {'sample': '1 1 -2 0', 'face_polished': '0 0 0 1', 'offset_angle': 45.0, 'orient': [1, 1, -2, 1.8721632653061226]},
    {'sample': '1 1 -2 0', 'face_polished': '0 0 0 1', 'offset_angle': 70.0, 'orient': [1, 1, -2, 5.143722448979592]},
    {'sample': '1 1 -2 0', 'face_polished': '1 -1 0 0', 'offset_angle': 20.0, 'orient': [1.0, 0.22645290581162336, -1.2264529058116234, 0.0]},
    # {'sample': '1 1 -2 0', 'face_polished': '1 -1 0 0', 'offset_angle': 30.0, 'orient': [1, 0, -1, 0]},    # equiv to sample '1 0 -1 0'
    {'sample': '1 1 -2 0', 'face_polished': '1 -1 0 0', 'offset_angle': 45.0, 'orient': [1, -0.26530612244897944, -0.7346938775510206, 0]},
    {'sample': '1 1 -2 0', 'face_polished': '1 -1 0 0', 'offset_angle': 70.0, 'orient': [1, -0.6513026052104207, -0.3486973947895793, 0]}
]

for item in sample_orienations:
    sample = item['sample']
    face = item['face_polished'] if item['face_polished'] is not None else "no_face"
    offset = item['offset_angle']
    orient = item['orient']
    
    print(f"\nCalculating Schmid factors for sample {sample}, face polished {face}, offset {offset}°, orient {orient}...")

    csv_file = f'Schmid_{sample}_{offset}deg_off_along_{face}.csv'

    file = open(csv_file, 'w')
    file.write('orient\tplane_conv_name\ti\tb_conv_name\tj\t' + 
            'plane\tb\tphi(orient_vs_plane)\tlambda(orient_vs_b)\tSchmid' + 
            '\tb_norm\tGSF_barrier\tGSF_shift\tabs(Schmid)\tm/b2\tm/(b2*GSF)\n')

    for plane_conv_name, generic_plane in crystal.generic_planes.items():
        for i, plane in enumerate(crystal.equivalent_directions(generic_plane, drop_inverse=True)):   # for every equiv plane
            n = crystal.plane_normal(plane)
            cos_phi, phi = crystal.angle_between_directions(orient, n)       # Schmid for plane

            if cos_phi != 0:
                for b_conv_name, generic_b in crystal.generic_directions.items():
                    for j, b in enumerate(crystal.equivalent_directions(generic_b, drop_inverse=True)):   # for every equiv Burgers
                        b_in_plane, _ = crystal.angle_between_directions(n, b)    # Burgers in plane
                        cos_lambda, lambdaa = crystal.angle_between_directions(orient, b)    # Schmid for Burgers
                        
                        if b_in_plane == 0 and cos_lambda != 0:
                            plane_str = " ".join(str(idx) for idx in plane)
                            b_str = " ".join(str(idx) for idx in b)
                            Schmid = cos_phi * cos_lambda               # only Schmid factor

                            if "part" in b_conv_name:       # partial dislocations, norms hardcoded
                                if generic_b == '1 0 -1 0':
                                    if plane_conv_name != 'basal':
                                        continue
                                    b_norm = crystal.physical_norm(b) / 3
                                elif generic_b == '1 0 -1 1':
                                    if plane_conv_name not in ['prismatic_a', 'pyramidalI']:
                                        continue
                                    elif plane_conv_name == 'prismatic_a':
                                        b_norm = crystal.physical_norm(b) / 2
                                    elif  plane_conv_name == 'pyramidalI':
                                        b_norm = m.sqrt((a/2)**2 + (crystal.physical_norm('1 0 -1 2')/2)**2 )
                                elif generic_b == '2 -1 -1 3':
                                    if plane_conv_name != 'pyramidalIV':
                                        continue
                                    b_norm = crystal.physical_norm(b) / 2
                                else:
                                    print(f"WARNING: Unhandled partial dislocation: {plane_conv_name} {generic_b}")
                                
                            
                            else:
                                b_norm = crystal.physical_norm(b)       # perfect diislo
                                # print(f"DEBUG: {plane_conv_name} {b_conv_name} norm = {b_norm}")
                            

                            factor = abs(Schmid) / b_norm**2            # first metric
                            
                            # Get GSF barrier for this system
                            gsf_barrier, gsf_shift = get_gsf_barrier(plane_conv_name, b_conv_name, "/home/nbekareva/TOOLS/potprop/DATABASE/ZnO/Zhang2024_NNP/STACKING_FAULTS.data")  # Replace with your actual GSF directory path
                            gsf_str = f"{gsf_barrier:.1f}" if gsf_barrier is not None else "N/A"
                            shift_str = gsf_shift if gsf_shift is not None else "N/A"
                            
                            combined_metric = "N/A"                     # second metric
                            if gsf_barrier is not None and gsf_barrier != 0:
                                combined_metric = f"{factor/gsf_barrier:.6f}"
                                
                            file.write(f"{orient}\t{plane_conv_name}\t{i+1}\t{b_conv_name}\t{j+1}\t" +
                                    f"{plane_str}\t{b_str}\t{phi:.1f}\t{lambdaa:.1f}\t{Schmid}\t" +
                                    f"{b_norm}\t{gsf_str}\t{shift_str}\t{abs(Schmid):.3f}\t{factor:.4f}\t{combined_metric}\n")

    file.close()
    
    plot_schmid_factors(infile=csv_file, orient=orient, outfile=f'Schmid_{sample}_{offset}deg_off_along_{face}.png')
