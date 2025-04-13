import pandas as pd
import numpy as np
from hexag_cristallo_tool import *


def ST_geometry(observ_dir, pillar_orient, slip_n, burgers, wz: Wurtzite):
    upright = wz.vector_in_plane(slip_n, observ_dir)
    flat_projected = wz.vector_in_plane(slip_n, wz.physical_crossprod(pillar_orient, observ_dir))
    
    ST_line = wz.physical_crossprod(observ_dir, slip_n)
    _, ST_inclination = wz.angle_bw_directions(pillar_orient, ST_line)      # ST angle with pillar orient vector
    _, plane_inclination = wz.angle_bw_directions(observ_dir, slip_n)
    return upright, flat_projected, ST_inclination, plane_inclination      
    
def analyze_schmid_factors(file_path, observ_dir):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Split the single column into multiple columns
    df = pd.DataFrame([x.split('\t') for x in df.iloc[:, 0]], 
                     columns=['orient', 'plane_conv_name', 'i', 'b_conv_name', 'j', 
                             'plane', 'b', 'phi', 'lambda', 'Schmid', 
                             'b_norm', 'GSF_barrier', 'GSF_shift', 'abs_Schmid', 
                             'm_b2', 'm_b2_GSF'])
    
    # Convert numeric columns
    numeric_columns = ['abs_Schmid', 'm_b2', 'm_b2_GSF']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate threshold for m/b2 (0.5 * max value)
    mb2_threshold = 0.5 * df['m_b2'].max()
    filtered_df = df[df['m_b2'] > mb2_threshold]

    # Add ST geometry cols
    # Define crystal structure
    cryst = Wurtzite(3.25, 5.2)
    
    # Define parameters needed for ST_geometry
    pillar_orient = filtered_df['orient'].iloc[0]  # Assuming all rows have the same orientation
    
    # Initialize columns for ST geometry
    filtered_df['upright'] = False
    filtered_df['flat_projected'] = False
    filtered_df['ST_inclination'] = 0.0
    filtered_df['plane_inclination'] = 0.0
    
    # Apply ST_geometry to each row
    for idx, row in filtered_df.iterrows():
        # Extract slip plane normal and burgers vector
        slip_n = row['plane']
        burgers = row['b']
        
        # Apply ST_geometry function
        upright, flat_projected, ST_inclination, plane_inclination = ST_geometry(
            observ_dir, pillar_orient, slip_n, burgers, cryst)
        
        # Update the dataframe with results
        filtered_df.at[idx, 'upright'] = upright
        filtered_df.at[idx, 'flat_projected'] = flat_projected
        filtered_df.at[idx, 'ST_inclination'] = ST_inclination
        filtered_df.at[idx, 'plane_inclination'] = plane_inclination

    result = filtered_df[['plane_conv_name', 'b_conv_name', 'abs_Schmid', 'm_b2', 'm_b2_GSF', 
                          'upright', 'flat_projected', 'ST_inclination', 'plane_inclination']].drop_duplicates()
    result = result.sort_values('m_b2', ascending=False)
    
    return result

# Usage
if __name__ == "__main__":
    orient = '1 1 -2 0'
    B0 = '0 0 0 1'          # zero_ZoneAxis, observ dir
    
    file_path = f"Schmid_factors_{orient}.csv"
    result = analyze_schmid_factors(file_path, observ_dir=B0)
    
    # Print results in a formatted way
    print("\nUnique combinations where m/b2 > 0.5*max(m/b2):")
    print("="*105)
    print(f"{'Plane':<15} {'Burgers Vector':<15} {'|Schmid|':<15} {'m/b2':<10} {'m_b2_GSF':<10} {'Upright':<8} {'Flat':<8} {'ST_incl°':<10} {'Plane_incl°':<10}")
    print("-"*105)
    
    for _, row in result.iterrows():
        print(f"{row['plane_conv_name']:<15} {row['b_conv_name']:<15} "
              f"{row['abs_Schmid']:<15.3f} {row['m_b2']:<10.4f} {row['m_b2_GSF']:<10.6f} "
              f"{str(row['upright']):<8} {str(row['flat_projected']):<8} "
              f"{row['ST_inclination']:<10.2f} {row['plane_inclination']:<10.2f}")