import numpy as np
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("..")


def metric_inner_product(v1, v2, metric):
    """Compute inner product using metric tensor: v1^T * metric * v2"""
    return np.dot(v1, np.dot(metric, v2))

def metric_norm(v, metric):
    """Compute norm using metric tensor: sqrt(v^T * metric * v)"""
    return np.sqrt(metric_inner_product(v, v, metric))

def angle_between_vectors(v1, v2, metric):
    """Compute angle between vectors using metric tensor"""
    dot_product = metric_inner_product(v1, v2, metric)
    norm1 = metric_norm(v1, metric)
    norm2 = metric_norm(v2, metric)
    cos_angle = dot_product / (norm1 * norm2)
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)

def find_vector_from_angles_v3_reference(v1, v2, v3, theta1, theta2, metric, normalize=True):
    """
    Find vector u such that:
    - u forms angle theta1 with v3 in plane (v1, v3) 
    - u forms angle theta2 with v3 in plane (v2, v3)
    
    Parameters:
    - v1, v2, v3: basis vectors (1D numpy arrays)
    - theta1: angle between u and v3 in plane (v1, v3) [radians]
    - theta2: angle between u and v3 in plane (v2, v3) [radians]
    - metric: metric tensor (2D numpy array)
    - normalize: whether to normalize the result vector
    
    Returns:
    - u: the unknown vector
    - coefficients: [alpha1, alpha2, alpha3] coefficients
    """
    
    def equations(coeffs):
        alpha1, alpha2, alpha3 = coeffs
        u = alpha1 * v1 + alpha2 * v2 + alpha3 * v3
        
        # Project u onto plane (v1, v3)
        u_13 = alpha1 * v1 + alpha3 * v3
        
        # Project u onto plane (v2, v3)  
        u_23 = alpha2 * v2 + alpha3 * v3
        
        # Constraint 1: angle between u and v3 in plane (v1, v3)
        if metric_norm(u_13, metric) < 1e-12:
            eq1 = float('inf')  # Degenerate case
        else:
            cos_theta1_actual = metric_inner_product(u_13, v3, metric) / (
                metric_norm(u_13, metric) * metric_norm(v3, metric))
            cos_theta1_actual = np.clip(cos_theta1_actual, -1.0, 1.0)
            eq1 = cos_theta1_actual - np.cos(theta1)
        
        # Constraint 2: angle between u and v3 in plane (v2, v3)
        if metric_norm(u_23, metric) < 1e-12:
            eq2 = float('inf')  # Degenerate case
        else:
            cos_theta2_actual = metric_inner_product(u_23, v3, metric) / (
                metric_norm(u_23, metric) * metric_norm(v3, metric))
            cos_theta2_actual = np.clip(cos_theta2_actual, -1.0, 1.0)
            eq2 = cos_theta2_actual - np.cos(theta2)
        
        # Constraint 3: normalization (if requested)
        if normalize:
            eq3 = metric_norm(u, metric) - 1.0
        else:
            eq3 = alpha1 - 1.0  # Arbitrary constraint to fix scale
            
        return [eq1, eq2, eq3]
    
    # Try multiple initial guesses to find solutions
    initial_guesses = [
        [1.0, 1.0, 1.0],
        [0.1, 0.1, 1.0],
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ]
    
    best_solution = None
    best_residual = float('inf')
    
    for guess in initial_guesses:
        try:
            solution = fsolve(equations, guess, xtol=1e-12)
            residual = equations(solution)
            max_residual = max(abs(r) for r in residual if not np.isinf(r))
            
            if max_residual < best_residual:
                best_residual = max_residual
                best_solution = solution
                
        except Exception:
            continue
    
    if best_solution is not None and best_residual < 1e-8:
        alpha1, alpha2, alpha3 = best_solution
        u = alpha1 * v1 + alpha2 * v2 + alpha3 * v3
        return u, [alpha1, alpha2, alpha3]
    else:
        print(f"Warning: Could not find accurate solution. Best residual: {best_residual:.2e}")
        if best_solution is not None:
            alpha1, alpha2, alpha3 = best_solution
            u = alpha1 * v1 + alpha2 * v2 + alpha3 * v3
            return u, [alpha1, alpha2, alpha3]
        return None, None

def verify_solution(u, v1, v2, v3, theta1, theta2, metric, coeffs):
    """Verify that the solution satisfies the angle constraints"""
    print("Verification:")
    print(f"Vector u = {u}")
    print(f"Coefficients [α₁, α₂, α₃] = {coeffs}")
    print(f"Norm of u = {metric_norm(u, metric):.6f}")
    
    # Check angle between u and v3 in plane (v1, v3)
    alpha1, alpha2, alpha3 = coeffs
    u_13 = alpha1 * v1 + alpha3 * v3
    actual_theta1 = angle_between_vectors(u_13, v3, metric)
    print(f"Angle between u and v3 in plane (v1, v3): target = {theta1:.4f} rad ({np.degrees(theta1):.1f}°), actual = {actual_theta1:.4f} rad ({np.degrees(actual_theta1):.1f}°)")
    
    # Check angle between u and v3 in plane (v2, v3)
    u_23 = alpha2 * v2 + alpha3 * v3
    actual_theta2 = angle_between_vectors(u_23, v3, metric)
    print(f"Angle between u and v3 in plane (v2, v3): target = {theta2:.4f} rad ({np.degrees(theta2):.1f}°), actual = {actual_theta2:.4f} rad ({np.degrees(actual_theta2):.1f}°)")
    
    # Additional verification: check that u actually lies in the specified planes for the angle measurements
    print(f"Component of u in v2 direction (should be zero for plane (v1,v3)): α₂ = {coeffs[1]:.6f}")
    print(f"Component of u in v1 direction (should be zero for plane (v2,v3)): α₁ = {coeffs[0]:.6f}")

def cross_product_metric(a, b, metric, normalize=True):
    """
    Compute generalized cross product for 3D vectors using metric tensor.
    
    Returns vector n orthogonal to both a and b in the metric space.
    Uses formula: n^k = ε^{ijk} * a_i * b_j / sqrt(|g|)
    
    Parameters:
    - a, b: 3D vectors (numpy arrays)
    - metric: 3x3 metric tensor (numpy array) 
    - normalize: if True, return unit vector
    
    Returns:
    - n: orthogonal vector
    """
    # Validate inputs
    if len(a) != 3 or len(b) != 3 or metric.shape != (3, 3):
        raise ValueError("Requires 3D vectors and 3x3 metric tensor")
    
    # Lower indices: a_i = g_ij * a^j
    a_lower = metric @ a
    b_lower = metric @ b
    
    # Compute metric determinant
    det_g = np.linalg.det(metric)
    if det_g <= 0:
        raise ValueError("Metric must be positive definite")
    
    # Generalized cross product: n^k = ε^{ijk} * a_i * b_j / sqrt(|g|)
    sqrt_det_g = np.sqrt(det_g)
    n = np.array([
        (a_lower[1] * b_lower[2] - a_lower[2] * b_lower[1]) / sqrt_det_g,
        (a_lower[2] * b_lower[0] - a_lower[0] * b_lower[2]) / sqrt_det_g,
        (a_lower[0] * b_lower[1] - a_lower[1] * b_lower[0]) / sqrt_det_g
    ])
    
    # Normalize if requested
    if normalize:
        norm_n = np.sqrt(n @ metric @ n)
        if norm_n > 1e-12:
            n = n / norm_n
    
    return n

def vector_equivalent(v):
    uu = np.round(v, 4)
    return [i/np.min(np.abs(uu[np.nonzero(uu)])) for i in uu]


# Example usage
if __name__ == "__main__":
    # Example 1: Euclidean space (identity metric)
    # print("Example 1: Euclidean 3D space")
    # print("=" * 50)
    
    # # Define metric tensor (identity for Euclidean space)
    # metric = np.eye(3)
    
    # # Define basis vectors
    # v1 = np.array([1.0, 0.0, 0.0])  # x-axis
    # v2 = np.array([0.0, 1.0, 0.0])  # y-axis
    # v3 = np.array([0.0, 0.0, 1.0])  # z-axis
    
    # # Define angles with respect to v3
    # theta1 = np.pi/4  # 45° between u and v3 in plane (v1, v3) = (x, z) plane
    # theta2 = np.pi/3  # 60° between u and v3 in plane (v2, v3) = (y, z) plane
    
    # print(f"θ₁ = {theta1:.4f} rad ({np.degrees(theta1):.1f}°) - angle with v3 in (v1,v3) plane")
    # print(f"θ₂ = {theta2:.4f} rad ({np.degrees(theta2):.1f}°) - angle with v3 in (v2,v3) plane")
    
    # # Solve
    # u, coeffs = find_vector_from_angles_v3_reference(v1, v2, v3, theta1, theta2, metric)
    
    # if u is not None:
    #     verify_solution(u, v1, v2, v3, theta1, theta2, metric, coeffs)
    
    # print("\n" + "="*50)
    
    # Example 2: Non-Euclidean space
    print("Example 2: Non-Euclidean space")
    print("=" * 50)
    
    # Define a non-trivial metric tensor
    from hexag_cristallo_tool_gamma import Wurtzite
    c = Wurtzite(a=3.2494, c=5.2054)
    metric = c.G
    # metric = np.array([[2.0, 0.5, 0.0],
    #                    [0.5, 1.0, 0.3],
    #                    [0.0, 0.3, 3.0]])
    
    # Define basis vectors
    v1 = c.vector_4ind_to_3ind('1 0 -1 0')
    v2 = c.vector_4ind_to_3ind('0 0 0 1')
    v3 = c.vector_4ind_to_3ind('1 -2 1 0')
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v3 = {v3}")
    
    # Define angles with respect to v3
    theta1 = np.radians(30)  # 30° with v3 in plane (v1, v3)
    theta2 = np.radians(24.81795638)  # 45° with v3 in plane (v2, v3)
    
    print(f"θ₁ = {theta1:.4f} rad ({np.degrees(theta1):.1f}°) - angle with v3 in (v1,v3) plane")
    print(f"θ₂ = {theta2:.4f} rad ({np.degrees(theta2):.1f}°) - angle with v3 in (v2,v3) plane")
    
    # Solve
    u, coeffs = find_vector_from_angles_v3_reference(v1, v2, v3, theta1, theta2, metric)
    
    if u is not None:
        verify_solution(u, v1, v2, v3, theta1, theta2, metric, coeffs)
    
    print("\n" + "="*50)

    print("Conversions:")
    print(f"v1 = {c.vector_3ind_to_4ind(v1)}")
    print(f"v2 = {c.vector_3ind_to_4ind(v2)}")
    print(f"v3 = {c.vector_3ind_to_4ind(v3)}")
    print(f"u = {c.vector_3ind_to_4ind(vector_equivalent(u))}")

    print("\n" + "="*50)

    print("2. u1 (e-beam) x line1 plane")
    b = c.vector_4ind_to_3ind('1 -1 0 4')  # dislo line direction
    n = cross_product_metric(u, b, metric)
    print(f"u = {u}")
    print(f"b = {b}")
    print(f"n = {n} = {c.vector_3ind_to_4ind(vector_equivalent(n))}")
    print(f"u·n = {u @ metric @ n:.2e}")
    print(f"b·n = {b @ metric @ n:.2e}")

    print("3. u2 (e-beam) x line2 plane")

    print("4. Line direction")
    
    # Example 3: Test with specific geometry
    # print("Example 3: More complex basis vectors")
    # print("=" * 50)
    
    # # Define metric tensor (identity)
    # metric = np.eye(3)
    
    # # Define non-orthogonal basis vectors
    # v1 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)  # normalized
    # v2 = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)  # normalized
    # v3 = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)  # normalized
    
    # print(f"v1 = {v1}")
    # print(f"v2 = {v2}")
    # print(f"v3 = {v3}")
    
    # # Define angles with respect to v3
    # theta1 = np.pi/3  # 60° with v3 in plane (v1, v3)
    # theta2 = np.pi/4  # 45° with v3 in plane (v2, v3)
    
    # print(f"θ₁ = {theta1:.4f} rad ({np.degrees(theta1):.1f}°) - angle with v3 in (v1,v3) plane")
    # print(f"θ₂ = {theta2:.4f} rad ({np.degrees(theta2):.1f}°) - angle with v3 in (v2,v3) plane")
    
    # # Solve
    # u, coeffs = find_vector_from_angles_v3_reference(v1, v2, v3, theta1, theta2, metric)
    
    # if u is not None:
    #     verify_solution(u, v1, v2, v3, theta1, theta2, metric, coeffs)