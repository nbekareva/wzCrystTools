# Requirements: numpy
# References:
# [1] https://ssd.phys.strath.ac.uk/resources/crystallography/crystallographic-direction-calculator/
# [2] https://youtu.be/vIomv4fFTHw?si=_MYn0UngxuZ6QRaB
import numpy as np
from copy import deepcopy
import math as m
from itertools import permutations, chain
from typing import List, Union, Tuple, Dict, Set, Optional


class Wurtzite:
    """
    Wurtzite crystal structure handler with Miller and Miller-Bravais notations.
    
    Provides tools for working with hexagonal crystal structures including:
    - Conversions between 3-index (Miller) and 4-index (Miller-Bravais) notations
    - Vector and plane normal calculations
    - Crystallographic calculations: angles, dot products, cross products
    - Conversion to Cartesian coordinates
    - Elastic property calculations
    
    Note on vector normalization:
    In hexagonal systems, vectors that represent primitive translations in the basal plane 
    require a 1/3 correction factor. These can be identified by converting to Miller indices -
    if both U and V components are divisible by 3, the vector needs the 1/3 correction.
    Examples:
    - [2,-1,-1,0] → [3,0,0] (requires 1/3 correction)
    - [1,0,-1,0] → [1,1,0] (no correction needed)
    """
    def __init__(self, a: float, c: float) -> None:
        """
        Initialize the Wurtzite crystal structure with lattice parameters.
        
        Args:
            a: basal plane lattice parameter
            c: c-axis lattice parameter
        """
        self.a = a
        self.c = c
        
        # Initialize metric tensors
        self._init_metric_tensors()
        
        # Initialize elastic constants
        self._init_elastic_constants()
        
        # Define common crystallographic vectors and planes
        self._define_standard_directions()
    
    def _init_metric_tensors(self) -> None:
        """Initialize the metric tensors for both 3-index and 4-index systems."""
        # Direct metric tensors
        self.g_hex = np.array([
            [self.a**2, -self.a**2/2, 0],
            [-self.a**2/2, self.a**2, 0],
            [0, 0, self.c**2]
        ])
        
        self.G_hex = (self.a**2/2) * np.array([
            [2, -1, -1, 0],
            [-1, 2, -1, 0],
            [-1, -1, 2, 0],
            [0, 0, 0, 2*self.c**2/self.a**2]
        ])
        
        # Reciprocal metric tensors
        self.g_recip_hex = 2/(3*self.a**2) * np.array([
            [2, 1, 0],
            [1, 2, 0],
            [0, 0, (3*self.a**2)/(2*self.c**2)]
        ])
        self.g_recip_hex = np.linalg.inv(self.g_hex)
        
        self.G_recip_hex = (2/(9*self.a**2)) * np.array([
            [2, -1, -1, 0],
            [-1, 2, -1, 0],
            [-1, -1, 2, 0],
            [0, 0, 0, (9*self.a**2)/(2*self.c**2)]
        ])

    def _init_elastic_constants(self) -> None:
        """Initialize elastic constants for ZnO (in GPa)."""
        # Elastic stiffness tensor [ref 86 from Ozgur Gen props of ZnO]
        self.Cij = np.array([
            [209.7,    121.1,  105.1,  0,  0,  0],
            [121.1,    209.7,  105.1,  0,  0,  0],
            [105.1,    105.1,  210.9,  0,  0,  0],
            [0,        0,      0,      42.47,  0,      0],
            [0,        0,      0,      0,      42.47,  0],
            [0,        0,      0,      0,      0,  44.29]
        ])
        
        # Elastic compliance tensor
        self.Sij = np.linalg.inv(self.Cij)
        
        # Transverse Poisson ratio
        self.nu_transverse = -self.Sij[0, 2] / self.Sij[2, 2]
    
    def _define_standard_directions(self) -> None:
        """Define standard Burgers vectors and slip systems."""
        # Common Burgers vectors
        self.generic_directions = {
            'a': '2 -1 -1 0',
            'b': '1 0 -1 0',
            'c': '0 0 0 1',
            'b+c': '1 0 -1 1',
            'b/2+c': '1 0 -1 2',
            'a/3+c': '2 -1 -1 1',
            'a+c': '2 -1 -1 3',
            '3 0 -3 2': '3 0 -3 2',
            '5 -1 -4 3': '5 -1 -4 3'
        }
        
        # Common slip planes
        self.generic_planes = {
            'basal': '0 0 0 1',
            'prismatic_m': '-1 0 1 0',
            'prismatic_a': '-2 1 1 0',
            'pyramidalIV': '-2 1 1 2', 
            'pyramidalIII': '-1 0 1 3', 
            'pyramidalII': '-1 0 1 2', 
            'pyramidalI': '-1 0 1 1'
        }
    
    # ==================== INPUT PROCESSING ====================
    
    @staticmethod
    def _to_list(item: Union[str, List]) -> List:
        """
        Convert string representation of indices to list of integers.
        
        Args:
            item: String with space-separated indices or a list
            
        Returns:
            List of integers
        """
        if isinstance(item, str):
            return list(map(int, item.split()))
        return item
    
    @staticmethod
    def _to_np_array(item: Union[List, np.ndarray]) -> np.ndarray:
        """Convert list to numpy array if needed."""
        if isinstance(item, list):
            return np.array(item)
        return item
    
    # ==================== INDEX CONVERSIONS ====================
    
    def miller_bravais_to_miller(self, hkil: Union[str, List]) -> List:
        """
        Convert plane indices from 4-index Miller-Bravais to 3-index Miller.
        
        Args:
            hkil: Miller-Bravais indices [h,k,i,l]
            
        Returns:
            Miller indices [h,k,l]
        """
        indices = self._to_list(deepcopy(hkil))
        indices.pop(2)  # Remove the i-index
        return indices
    
    def miller_to_miller_bravais(self, hkl: Union[str, List]) -> List:
        """
        Convert plane indices from 3-index Miller to 4-index Miller-Bravais.
        
        Args:
            hkl: Miller indices [h,k,l]
            
        Returns:
            Miller-Bravais indices [h,k,i,l]
        """
        indices = self._to_list(deepcopy(hkl))
        h, k, l = indices
        i = -(h + k)
        return [h, k, i, l]
    
    def vector_4ind_to_3ind(self, uvtw: Union[str, List]) -> np.ndarray:
        """
        Convert direction vector from 4-index to 3-index notation.
        
        Args:
            uvtw: Direction vector in 4-index notation [u,v,t,w]
            
        Returns:
            Direction vector in 3-index notation [U,V,W]
        """
        indices = self._to_list(deepcopy(uvtw))
        u, v, _, w = indices
        return np.array([2*u+v, 2*v+u, w])
    
    def vector_3ind_to_4ind(self, uvw: Union[str, List]) -> np.ndarray:
        """
        Convert direction vector from 3-index to 4-index notation.
        
        Args:
            uvw: Direction vector in 3-index notation [U,V,W]
            
        Returns:
            Direction vector in 4-index notation [u,v,t,w]
        """
        indices = self._to_list(deepcopy(uvw))
        u, v, w = indices
        u1 = (2*u-v)/3
        v1 = (2*v-u)/3
        t = -(u1+v1)
        return np.array([u1, v1, t, w])
    
    # ==================== VECTOR OPERATIONS ====================
    
    def dot_product(self, vec1: Union[str, List], vec2: Union[str, List]) -> float:
        """
        Calculate dot product between two vectors in 4-index notation.
        
        Args:
            vec1: First vector [u,v,t,w]
            vec2: Second vector [u,v,t,w]
            
        Returns:
            Dot product value
        """
        v1 = self._to_np_array(self._to_list(vec1))
        v2 = self._to_np_array(self._to_list(vec2))
        return np.dot(v1, np.dot(self.G_hex, v2))
    
    def cross_product(self, vec1: Union[str, List], vec2: Union[str, List]) -> np.ndarray:
        """
        Calculate cross product between two vectors in 4-index notation.
        
        Args:
            vec1: First vector [u,v,t,w]
            vec2: Second vector [u,v,t,w]
            
        Returns:
            Cross product vector in 4-index notation
        """
        # Convert to 3-index
        v1_3ind = self.vector_4ind_to_3ind(vec1)
        v2_3ind = self.vector_4ind_to_3ind(vec2)
        
        # Convert second vector to covariant form
        v2_cov = np.dot(self.g_hex, v2_3ind)
        
        # Cross product in 3-index
        cross_3ind = np.cross(v1_3ind, v2_cov)
        
        # Convert back to 4-index
        cross = self.vector_3ind_to_4ind(cross_3ind)
        return cross
    
    def norm(self, vec: Union[str, List]) -> float:
        """
        Calculate the standard metric tensor norm of a vector.
        
        Args:
            vec: Vector in 4-index notation [u,v,t,w]
            
        Returns:
            Vector magnitude
        """
        indices = self._to_list(vec)
        mag2 = self.dot_product(indices, indices)
        return m.sqrt(mag2)
    
    def needs_triple_correction(self, vec: Union[str, List]) -> bool:
        """
        Check if a vector needs the 1/3 normalization factor.
        
        In hexagonal systems, vectors that represent primitive translations in the basal plane 
        require a 1/3 correction factor. These can be identified by converting to Miller indices -
        if both U and V components are divisible by 3, the vector needs the 1/3 correction.
        
        Args:
            vec: Vector in 4-index notation [u,v,t,w]
            
        Returns:
            Boolean indicating if correction is needed
        """
        indices = self._to_list(vec)
        w = indices[-1]
        miller = self.vector_4ind_to_3ind(indices)
        
        # Check if basal plane components are divisible by 3
        basal_divisible = all(x % 3 == 0 for x in miller[:2])
        
        if basal_divisible and w == 0:
            return True     # Pure basal plane vectors always get correction
        if basal_divisible and w % 3 == 0:
            return True     # vectors with c-component --> only when equivalent atom pos is reached
        
        return False
    
    def physical_norm(self, vec: Union[str, List]) -> float:
        """
        Calculate the physically correct length of a crystallographic vector.
        
        This applies the 1/3 correction factor when needed.
        
        Args:
            vec: Vector in 4-index notation [u,v,t,w]
            
        Returns:
            Physical vector magnitude
        """
        indices = self._to_list(vec)
        standard_norm = self.norm(indices)
        return standard_norm / 3 if self.needs_triple_correction(indices) else standard_norm
    
    def normalized_cross_product(self, vec1: Union[str, List], vec2: Union[str, List]) -> np.ndarray:
        """
        Calculate the normalized cross product direction.
        
        Args:
            vec1: First vector [u,v,t,w]
            vec2: Second vector [u,v,t,w]
            
        Returns:
            Normalized cross product vector
        """
        cross = self.cross_product(vec1, vec2)
        max_component = np.max(np.abs(cross))
        if max_component > 0:
            cross = cross / max_component
        return np.round(cross, 10)  # Round to handle floating-point precision
    
    # ==================== PLANE NORMAL OPERATIONS ====================
    
    def plane_normal(self, hkil: Union[str, List]) -> List:
        """
        Calculate the normal vector to a plane in a hexagonal system using 4x4 metric tensor.
        
        Args:
            hkil: Miller-Bravais indices of the plane [h,k,i,l]
            
        Returns:
            Normal vector to the plane in 4-index notation
        """
        indices = self._to_list(deepcopy(hkil))
        indices = self._to_np_array(indices)
        normal = np.dot(indices, self.G_recip_hex)

        return normal
    
    def plane_normal_DEPRECATED(self, hkil: Union[str, List]) -> List:
        """
        Calculate the normal vector to a plane in a hexagonal system USING LITERAL EXPRESSIONS.
        
        Args:
            hkil: Miller-Bravais indices of the plane [h,k,i,l]
            
        Returns:
            Normal vector to the plane in 4-index notation
        """
        indices = self._to_list(deepcopy(hkil))
        [h, k, i, l] = indices
        
        # Special case for basal plane
        if indices == [0, 0, 0, 1]:
            return indices
        
        # For non-basal planes with l component, adjust l according to hexagonal metrics
        if l != 0:
            l = 3 * self.a**2 * l / (2 * self.c**2)
        
        return [h, k, i, l]
    
    def vector_in_plane(self, vec: Union[str, List], plane: Union[str, List]) -> bool:
        """
        Check if a vector lies in a plane.
        
        Args:
            vec: Vector in 4-index notation [u,v,t,w]
            plane: Plane in Miller-Bravais notation [h,k,i,l]
            
        Returns:
            Boolean indicating if the vector is in the plane
        """
        plane_normal = self.plane_normal(plane)
        cosa, _ = self.angle_between_directions(vec, plane_normal)
        return abs(cosa) < 1e-10  # Near-zero check for perpendicularity
    
    # ==================== ANGLE CALCULATIONS ====================
    
    def angle_between_directions(self, vec1: Union[str, List], vec2: Union[str, List]) -> Tuple[float, float]:
        """
        Calculate the angle between two directions.
        
        Args:
            vec1: First vector in 4-index notation [u,v,t,w]
            vec2: Second vector in 4-index notation [u,v,t,w]
            
        Returns:
            Tuple containing (cosine of angle, angle in degrees)
        """
        v1 = self._to_list(vec1)
        v2 = self._to_list(vec2)
        
        dot = self.dot_product(v1, v2)
        norm1 = self.norm(v1)
        norm2 = self.norm(v2)
        
        cos_phi = dot / (norm1 * norm2)
        cos_phi = round(cos_phi, 10)  # Handle floating point
        
        # Clamp to valid range for acos
        cos_phi = max(-1.0, min(1.0, cos_phi))
        
        phi = round(m.degrees(m.acos(cos_phi)), 8)
        return cos_phi, phi
    
    def angle_between_planes(self, plane1: Union[str, List], plane2: Union[str, List]) -> Tuple[float, float]:
        """
        Calculate the angle between two planes.
        
        Args:
            plane1: First plane in Miller-Bravais notation [h,k,i,l]
            plane2: Second plane in Miller-Bravais notation [h,k,i,l]
            
        Returns:
            Tuple containing (cosine of angle, angle in degrees)
        """
        try:
            int(str(plane1[-1]))        # check if int
            int(str(plane2[-1]))
        except:
            raise ValueError("Plane indices are not in Miller-Bravais notation [h,k,i,l]")
        
        n1 = self.plane_normal(plane1)
        n2 = self.plane_normal(plane2)
        return self.angle_between_directions(n1, n2)
    
    # ==================== SYMMETRY OPERATIONS ====================
    
    @staticmethod
    def _remove_inverse_duplicates(tuples_set: Set[Tuple]) -> Set[Tuple]:
        """
        Remove inverse duplicates from a set of direction index tuples.
        
        Args:
            tuples_set: Set of tuples representing crystallographic directions
            
        Returns:
            Set with inverse duplicates removed
        """
        unique_tuples = set()
        for tup in tuples_set:
            inverse_tup = tuple(-x for x in tup)
            if tup not in unique_tuples and inverse_tup not in unique_tuples:
                unique_tuples.add(tup)
        return unique_tuples
    
    def equivalent_directions(self, vec: Union[str, List], drop_inverse=True) -> List[List]:
        """
        Generate all crystallographically equivalent directions in hexagonal system.
        
        Args:
            vec: Vector in 4-index notation [u,v,t,w]
            
        Returns:
            List of equivalent directions
        """
        indices = self._to_list(vec)
        
        # Special case for [0001]
        if indices == [0, 0, 0, 1]:
            return [indices]
        
        # Generate permutations of the first three indices
        uvt_permutations = set(permutations(indices[:-1]))
        
        # Remove inverse duplicates
        if drop_inverse:
            uvt_permutations = self._remove_inverse_duplicates(uvt_permutations)
        
        w = indices[-1]
        equiv_list = []
        
        # For directions with c-component, include both +w and -w versions
        if w != 0:
            for perm in uvt_permutations:
                equiv_list.append([*perm, w])
                equiv_list.append([*perm, -w])
        else:
            for perm in uvt_permutations:
                equiv_list.append([*perm, w])
                
        return equiv_list
    
    # ==================== SLIP SYSTEMS ====================
    
    def get_slip_systems(self) -> Dict[str, List[List]]:
        """
        Generate all slip systems of the crystal.
        
        Returns:
            Dictionary mapping slip planes to possible Burgers vectors
        """
        slip_systems = {}
        
        # For each slip plane type
        for slip_mode in self.generic_slip_modes.values():
            # Get all equivalent planes
            for plane in self.equivalent_directions(slip_mode):
                plane_str = ' '.join(str(i) for i in plane)
                slip_systems[plane_str] = []
                
                # Get the plane normal
                normal = self.plane_normal(plane)
                
                # For each Burgers vector type
                for burgers_vector in self.generic_perfect_Burgers.values():
                    # Get all equivalent Burgers vectors
                    for b_vec in self.equivalent_directions(burgers_vector):
                        # Check if Burgers vector is in the slip plane
                        if self.vector_in_plane(b_vec, plane):
                            slip_systems[plane_str].append(b_vec)
        
        return slip_systems
    
    # ==================== CARTESIAN CONVERSIONS ====================
    
    def plane_normal_to_cartesian(self, hkil: Union[str, List]) -> np.ndarray:
        """
        Convert a plane normal from Miller-Bravais indices to Cartesian coordinates.
        
        Args:
            hkil: Miller-Bravais indices of the plane [h,k,i,l]
            
        Returns:
            Normalized Cartesian coordinates [x,y,z] of the plane normal
        """
        indices = self._to_list(hkil)
        
        # Get the reciprocal lattice vector (normal to the plane)
        normal = self.plane_normal(indices)
        
        # Convert to Miller indices (3-index system)
        hkl = self.miller_bravais_to_miller(normal)
        
        # Calculate reciprocal lattice constants
        a_star = 2 / (self.a * np.sqrt(3))
        c_star = 1 / self.c
        
        # Calculate Cartesian coordinates
        h, k, l = hkl
        x = (2 * h + k) * a_star / 3
        y = k * a_star * np.sqrt(3) / 3
        z = l * c_star
        
        # Create and normalize the Cartesian vector
        cartesian = np.array([x, y, z])
        norm = np.linalg.norm(cartesian)
        
        if norm > 0:
            cartesian = cartesian / norm
            
        return cartesian
    
    def vector_to_cartesian(self, uvtw: Union[str, List]) -> np.ndarray:
        """
        Convert a direction vector from Miller-Bravais indices to Cartesian coordinates.
        
        Args:
            uvtw: Miller-Bravais indices of the direction [u,v,t,w]
            
        Returns:
            Normalized Cartesian coordinates [x,y,z] of the direction
        """
        indices = self._to_list(uvtw)
        
        # Convert to Miller indices (3-index system)
        uvw = self.vector_4ind_to_3ind(indices)
        
        # Extract components
        u, v, w = uvw
        
        # Calculate Cartesian coordinates
        x = u * self.a * np.sqrt(3) / 2
        y = (2 * v + u) * self.a / 2
        z = w * self.c
        
        # Create the Cartesian vector
        cartesian = np.array([x, y, z])
        
        # Apply normalization if physical correction is needed
        if self.needs_triple_correction(indices):
            cartesian = cartesian / 3
            
        # Return normalized vector
        norm = np.linalg.norm(cartesian)
        if norm > 0:
            cartesian = cartesian / norm
            
        return cartesian
    
    # ==================== ELASTIC PROPERTIES ====================
    
    def young_modulus_for_direction(self, uvtw: Union[str, List]) -> float:
        """
        Calculate Young's modulus in the given direction.
        
        Args:
            uvtw: Direction in 4-index notation [u,v,t,w]
            
        Returns:
            Young's modulus value (GPa)
        """
        # Convert to 3-index and normalize
        uvw = self.vector_4ind_to_3ind(uvtw)
        uvw = uvw / np.linalg.norm(uvw)
        
        # For hexagonal, the formula depends on the angle with the c-axis
        _, _, w = uvw
        S = self.Sij
        
        # Calculate reciprocal of Young's modulus
        E_recip = (1 - w**2)**2 * S[0,0] + w**4 * S[2,2] + \
                  w**2 * (1 - w**2) * (2 * S[0,2] + S[3,3])
        
        return 1 / E_recip
    
    def bulk_elastic_properties(self) -> Tuple[float, float]:
        """
        Calculate bulk elastic properties using the Voigt-Reuss-Hill approximation.
        
        Returns:
            Tuple of (Young's modulus, Poisson's ratio)
        """
        C = self.Cij
        C11, C12, C13 = C[0,0], C[0,1], C[0,2]
        C33, C55, C66 = C[2,2], C[4,4], C[5,5]

        # Voigt approximation (upper bound)
        BV = 2/9 * (C11 + C12 + 2*C13 + 0.5*C33)
        GV = 1/30 * (C11 + C12 + 2*C33 - 4*C13 + 12*C55 + 12*C66)
        
        # Reuss approximation (lower bound)
        repeated_term = (C11 + C12)*C33 - 2*C13**2
        BR = repeated_term / (C11 + C12 + 2*C33 - 4*C13)
        GR = 5/2 * repeated_term * C55 * C66 / (3*BV*C55*C66 + repeated_term*(C55+C66))
        
        # Voigt-Reuss-Hill average
        B = 0.5 * (BV + BR)
        G = 0.5 * (GV + GR)
        
        # Calculate Young's modulus and Poisson's ratio
        E = 9*B*G / (3*B + G)
        nu = (3*B - 2*G) / (2*(3*B + G))
        
        return E, nu