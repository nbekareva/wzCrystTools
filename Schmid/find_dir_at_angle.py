import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")  # Add parent directory to path
from hexag_cristallo_tool_gamma import Wurtzite as Wurtzite

# class Wurtzite:
#     """
#     Wurtzite crystal structure class for hexagonal crystallographic calculations.
#     """
#     def __init__(self, a, c):
#         """
#         Initialize Wurtzite structure with lattice parameters.
        
#         Parameters:
#         -----------
#         a : float
#             Lattice parameter a (in Angstroms)
#         c : float
#             Lattice parameter c (in Angstroms)
#         """
#         self.a = a
#         self.c = c
    
#     def parse_direction(self, direction):
#         """
#         Parse direction from string or list format.
        
#         Parameters:
#         -----------
#         direction : str or list
#             Direction in Miller-Bravais notation
            
#         Returns:
#         --------
#         list : [h, k, i, l] or [u, v, t, w]
#         """
#         if isinstance(direction, str):
#             # Parse string like '0 0 0 1'
#             return [float(x) for x in direction.split()]
#         else:
#             # Already a list
#             return direction
    
#     def direction_to_cartesian(self, direction):
#         """
#         Convert Miller-Bravais direction [uvtw] to Cartesian coordinates.
        
#         Parameters:
#         -----------
#         direction : list or str
#             Direction in Miller-Bravais notation [u, v, t, w]
            
#         Returns:
#         --------
#         np.array : Cartesian coordinates [x, y, z]
#         """
#         dir_list = self.parse_direction(direction)
#         u, v, t, w = dir_list
        
#         # Conversion from 4-index to Cartesian for hexagonal system
#         # x = a * (2*u + v) / 2
#         # y = a * v * sqrt(3) / 2
#         # z = c * w
        
#         x = self.a * (2 * u + v) / 2
#         y = self.a * v * np.sqrt(3) / 2
#         z = self.c * w
        
#         return np.array([x, y, z])
    
#     def angle_between_directions(self, dir1, dir2):
#         """
#         Calculate the angle between two crystallographic directions.
        
#         Parameters:
#         -----------
#         dir1, dir2 : str or list
#             Directions in Miller-Bravais notation
            
#         Returns:
#         --------
#         float : Angle in degrees
#         """
#         # Convert to Cartesian coordinates
#         cart1 = self.direction_to_cartesian(dir1)
#         cart2 = self.direction_to_cartesian(dir2)
        
#         # Calculate angle using dot product
#         cos_angle = np.dot(cart1, cart2) / (np.linalg.norm(cart1) * np.linalg.norm(cart2))
        
#         # Clamp to [-1, 1] to avoid numerical errors
#         cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
#         # Convert to degrees
#         angle_rad = np.arccos(cos_angle)
#         angle_deg = np.degrees(angle_rad)
        
#         return angle_deg


def search_crystallographic_direction(crystal, reference_direction, target_angle, test_directions=None, 
                                      base_direction=None, l_search_range=(-10, 10), a=3.2488, c=5.20596):
    """
    Search for crystallographic directions by varying the last index.
    
    Parameters:
    -----------
    target_angle : float
        The target angle to search for (in degrees)
    base_direction : list
        The first three indices of the direction [h, k, i]
    l_search_range : tuple
        Range of values to search for the last index (min, max)
    a, c : float
        Lattice parameters for the Wurtzite structure
    
    Returns:
    --------
    results : dict
        Dictionary containing angles, directions, and differences from target
    """
    last_indices = []
    directions = []
    angles = []
    differences = []
    

    if base_direction is not None and directions is None:
        # Vary the last index
        for l in np.linspace(l_search_range[0], l_search_range[1], 50):
            direction = base_direction + [l]
            
            try:
                # Calculate angle between reference and current direction
                _, angle = crystal.angle_between_directions(reference_direction, direction)
                
                directions.append(direction)
                angles.append(angle)
                differences.append(abs(angle - target_angle))
                
            except Exception as e:
                print(f"Error calculating angle for direction {direction}: {e}")
                continue

    else:
        for direction in test_directions:
            try:
                # Calculate angle between reference and current direction
                _, angle = crystal.angle_between_directions(reference_direction, direction)
                
                directions.append(direction)
                angles.append(angle)
                differences.append(abs(angle - target_angle))
                
            except Exception as e:
                print(f"Error calculating angle for direction {direction}: {e}")
                continue
    
    # Find the closest match
    if differences:
        min_diff_idx = np.argmin(differences)
        best_match = {
            'direction': directions[min_diff_idx],
            'angle': angles[min_diff_idx],
            'difference': differences[min_diff_idx]
        }
    else:
        best_match = None
    
    results = {
        'directions': directions,
        'angles': angles,
        'differences': differences,
        'best_match': best_match,
        'target_angle': target_angle
    }
    
    return results


def plot_results(results, base_direction, reference_direction):
    """
    Plot the angle vs last index relationship.
    
    Parameters:
    -----------
    results : dict
        Results from search_crystallographic_direction function
    base_direction : list
        The first three indices for labeling
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Angle vs Last Index
    ax1.plot(results['last_indices'], results['angles'], 'b.-', linewidth=2, markersize=8)
    ax1.axhline(y=results['target_angle'], color='r', linestyle='--', 
                label=f"Target angle: {results['target_angle']:.2f}°")
    
    if results['best_match']:
        ax1.plot(results['best_match']['direction'][-1], 
                results['best_match']['angle'], 
                'go', markersize=15, label='Best match')
        ax1.annotate(f"[{', '.join(map(str, results['best_match']['direction']))}]\n" + 
                    f"Angle: {results['best_match']['angle']:.3f}°",
                    xy=(results['best_match']['direction'][-1], results['best_match']['angle']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax1.set_xlabel('Last Index (l)', fontsize=12)
    ax1.set_ylabel(f'Angle from {reference_direction} (degrees)', fontsize=12)
    ax1.set_title(f'Crystallographic Direction Search: [{base_direction[0]} {base_direction[1]} {base_direction[2]} l]', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Difference from Target Angle
    ax2.plot(results['last_indices'], results['differences'], 'r.-', linewidth=2, markersize=8)
    
    if results['best_match']:
        ax2.plot(results['best_match']['direction'][-1], 
                results['best_match']['difference'], 
                'go', markersize=15, label='Best match')
    
    ax2.set_xlabel('Last Index (l)', fontsize=12)
    ax2.set_ylabel('|Angle - Target| (degrees)', fontsize=12)
    ax2.set_title('Difference from Target Angle', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('crystallographic_direction_search.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: crystallographic_direction_search.png")
    
    return fig

def navigate_between_directions(crystal, dir1, dir2, reference_dir, n_steps=50):
    """
    Navigate between two crystallographic directions by interpolating indices.
    
    Parameters:
    -----------
    dir1, dir2 : list
        Starting and ending directions [u, v, t, w]
    n_steps : int
        Number of interpolation steps
    a, c : float
        Lattice parameters
    reference_dir : str or list
        Reference direction to measure angles against (default: c-axis [0001])
    
    Returns:
    --------
    results : dict
        Dictionary containing interpolated directions and angles
    """
    # Convert directions to numpy arrays
    dir1 = np.array(dir1, dtype=float)
    dir2 = np.array(dir2, dtype=float)
    
    # Storage for results
    interpolation_params = []
    directions = []
    angles_from_reference = []
    angles_between_neighbors = []
    
    # Interpolate between directions
    for i, t in enumerate(np.linspace(0, 1, n_steps)):
        # Linear interpolation: dir(t) = (1-t)*dir1 + t*dir2
        current_dir = (1 - t) * dir1 + t * dir2
        
        interpolation_params.append(t)
        directions.append(current_dir)
        
        # Calculate angle from reference direction
        angle_ref = crystal.angle_between_directions(reference_dir, current_dir)
        angles_from_reference.append(angle_ref)
        
        # Calculate angle between consecutive directions (except for first)
        if i > 0:
            _, angle_neighbor = crystal.angle_between_directions(directions[i-1], current_dir)
            angles_between_neighbors.append(angle_neighbor)
        else:
            angles_between_neighbors.append(0)
    
    results = {
        'interpolation_params': np.array(interpolation_params),
        'directions': np.array(directions),
        'angles_from_reference': np.array(angles_from_reference),
        'angles_between_neighbors': np.array(angles_between_neighbors),
        'dir1': dir1,
        'dir2': dir2,
        'reference_dir': reference_dir
    }
    
    return results


def main():
    """
    Main function to demonstrate the search.
    """
    # Example: Search for a direction with a specific angle
    # First, let's calculate the angle for [1, 0, -1, 2] as reference
    c = Wurtzite(a=3.2488, c=5.20596)
    target_angle = 70
    reference_direction = [1, 1, -2, 0]
    # base_direction = [1, 1, -2]
    
    print(f"Target angle vs [0001]: {target_angle:.4f}°")
    # print(f"\nSearching for crystallographic directions {base_direction}, l]...")
    print("=" * 60)
    
    test = navigate_between_directions(crystal=c, dir1=reference_direction, dir2=[1, -1, 0, 0], 
                                       reference_dir=reference_direction, n_steps=500)
    test_dirs = test['directions']

    # Search by varying the last index
    results = search_crystallographic_direction(
        crystal=c,
        reference_direction=reference_direction,
        target_angle=target_angle,
        # base_direction=base_direction,
        # search_range=(5.1436, 5.1438), 
        test_directions=test_dirs.tolist()
    )
    
    # Display results
    print(f"\nResults:")
    print("-" * 60)
    for dir, angle, diff in zip(results['directions'], results['angles'], results['differences']):
        direction_str = f"{dir}"        # f"{dir[0]} {dir[1]} {dir[2]} {dir[3]:.4f}"
        print(f"{direction_str:20s} → Angle: {angle:8.4f}° (Δ = {diff:7.4f}°)")
    
    if results['best_match']:
        print("\n" + "=" * 60)
        print("BEST MATCH:")
        best_dir = results['best_match']['direction']
        print(f"Direction: {[x / best_dir[0] for x in best_dir]}")  # Normalize by first index
        print(f"Angle: {results['best_match']['angle']:.4f}°")
        print(f"Difference from target: {results['best_match']['difference']:.4f}°")
        print("=" * 60)
    
    # Plot the results
    plot_results(results, base_direction=base_direction, reference_direction=reference_direction)
    plt.show()


if __name__ == "__main__":
    main()