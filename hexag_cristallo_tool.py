# Requirements: numpy
# References:
# [1] https://ssd.phys.strath.ac.uk/resources/crystallography/crystallographic-direction-calculator/
# [2] https://youtu.be/vIomv4fFTHw?si=_MYn0UngxuZ6QRaB
from copy import deepcopy
import math as m
from itertools import permutations, combinations, combinations_with_replacement, chain
import numpy as np


class Wurtzite:
    """
    Wurtzite crystal structure handling with 3-index (Miller) and 4-index (Miller-Bravais) notations.
    
    Note on vector normalization:
    In hexagonal systems, vectors that represent primitive translations in the basal plane 
    require a 1/3 correction factor. These can be identified by converting to Miller indices -
    if both U and V components are divisible by 3, the vector needs the 1/3 correction.
    Examples:
    - [2,-1,-1,0] → [3,0,0] (requires 1/3 correction)
    - [1,0,-1,0] → [1,1,0] (no correction needed)
    
    This normalization is automatically applied in the norm() method.
    """
    def __init__(self, a: float, c: float) -> None:
        self.a = a
        self.c = c
        # direct metric tensor for Miller idxs
        self.g_hex = np.array([[self.a**2, -self.a**2 / 2, 0],
                               [-self.a**2 / 2, self.a**2, 0],
                               [0, 0, self.c**2]])
        # direct metric tensor for Miller-Bravais idxs
        self.G_hex = (self.a**2 / 2) * np.array([[2, -1, -1, 0],
                                                [-1, 2, -1, 0],
                                                [-1, -1, 2, 0],
                                                [0, 0, 0, 2 * self.c**2 / self.a**2]])
        # reciprocal metric tensor for Miller idxs
        self.g_recip_hex = 2 / (3 * self.a**2) * np.array([[2, 1, 0],
                                                         [1, 2, 0],
                                                         [0, 0, (3 * self.a**2) / (2 * self.c**2)]])
        # reciprocal metric tensor for Miller-Bravais idxs
        # self.G_recip_hex = np.linalg.inv(self.G_hex)
        self.G_recip_hex = (2 / (9*self.a**2)) * np.array([[2, -1, -1, 0],
                                                      [-1, 2, -1, 0],
                                                      [-1, -1, 2, 0],
                                                      [0, 0, 0, (9*self.a**2) / (2*self.c**2)]])
        # elastic stiffness tensor
        self.Cij = np.array([[209.7,    121.1,  105.1,  0,  0,  0],     # GPa [ref 86 from Ozgur Gen props of ZnO]
                             [121.1,    209.7,  105.1,  0,  0,  0],
                             [105.1,    105.1,  210.9,  0,  0,  0],
                             [0,        0,      0,      42.47,  0,      0],
                             [0,        0,      0,      0,      42.47,  0],
                             [0,        0,      0,      0,      0,  44.29]])
        # self.Cij = np.array([[195.2,    111.2,  92.5,  0,  0,  0],     # GPa
        #                      [111.2,    195.2,  92.5,  0,  0,  0],
        #                      [92.5,    92.5,  199.8,  0,  0,  0],
        #                      [0,        0,      0,      39.6,  0,      0],
        #                      [0,        0,      0,      0,      39.6,  0],
        #                      [0,        0,      0,      0,      0,  42.1]])
        # elastic compliance tensor
        self.Sij = np.linalg.inv(self.Cij)

        self.nu_transverse = - self.Sij[0,2] / self.Sij[2,2]

        self.generic_perfect_Burgers = {
            'a': '2 -1 -1 0',
            'b': '1 0 -1 0',
            'c': '0 0 0 1',
            'b+c': '1 0 -1 1',
            'a+c': '2 -1 -1 3',
            'b/2+c': '1 0 -1 2',
            '3 0 -3 2': '3 0 -3 2',
            '5 -1 -4 3': '5 -1 -4 3'
        }
            # # Process Burgers vectors once during initialization
            # for name, uvtw_str in self.generic_perfect_Burgers.items():
            #     uvtw = list(map(int, uvtw_str.split()))
            #     b_norm = self.norm(uvtw)  # Now this automatically applies 1/3 correction when needed
            #     self.processed_burgers[name] = {'indices': uvtw, 'norm': b_norm}
            
        self.generic_slip_modes = {
            'basal': '0 0 0 1',
            'prismatic_m': '-1 0 1 0',
            'prismatic_a': '-2 1 1 0',
            'pyramidalIV': '-2 1 1 2', 
            'pyramidalIII': '-1 0 1 3', 
            'pyramidalII': '-1 0 1 2', 
            'pyramidalI': '-1 0 1 1'
        }
            # for name, hkil in self.generic_slip_modes.items():
            #     self.generic_slip_modes[name] = self.list_equiv(hkil)

    @staticmethod
    def __to_list(*args) -> list:
        """Convert str vectors to lists."""
        return [list(map(int, item.split())) if type(item) is str else item for item in args]
    
    @staticmethod
    def __to_np_array(*args):
        """Convert list vectors to np.arrays."""
        return [np.array(item) if type(item) is list else item for item in args]
    
    # INDEX CONVERSIONS
    def PlaneBravaisToMiller(self, hkil):
        """Convert plane 4-ind to 3-ind."""
        [hkil] = self.__to_list(deepcopy(hkil))
        hkil.pop(2)
        return hkil

    # def PlaneMillerToBravais(self, hkl):
    #     """Convert plane 4-ind to 3-ind."""
    #     [hkl] = self.__to_list(hkl)
    #     i = -(h+k)
    #     hkl.insert(2, i)
    #     return hkl

    def Vector4to3(self, uvtw):
        """From 4-ind vector, return normalized 3-ind vector."""
        [uvtw] = self.__to_list(deepcopy(uvtw))
        [u, v, _, w] = uvtw
        u1, v1, w1 = 2*u+v, 2*v+u, w
        return np.array([u1, v1, w1])

    def Vector3to4(self, uvw):
        """3-ind vector to 4-ind."""
        [uvw] = self.__to_list(deepcopy(uvw))
        [u, v, w] = uvw
        u1, v1, t, w1 = (2*u-v)/3, (2*v-u)/3, -(u+v), w
        # norm = max([u1, v1, w1]) 
        # [u1, v1, t, w1] = [i/norm for i in [u1, v1, t, w1]]
        return np.array([u1, v1, t, w1])

    def plane_normal_to_cartesian(self, hkil):
        """
        Convert a plane normal from Miller-Bravais indices to Cartesian coordinates.
        
        Arguments:
            hkil: str or list, Miller-Bravais indices of the plane (hkil)
            
        Returns:
            numpy.ndarray: Cartesian coordinates [x, y, z] of the normalized plane normal vector
        """
        [hkil] = self.__to_list(hkil)
        # Get the reciprocal lattice vector (normal to the plane)
        normal = self.n_to_hkil(hkil)
        
        # Convert to Miller indices (3-index system)
        hkl = self.PlaneBravaisToMiller(normal)
        
        # Calculate Cartesian coordinates for the normal vector
        # For hexagonal system, the transformation from reciprocal space to Cartesian is:
        # x = (2h + k) * a*/3
        # y = k * a* * sqrt(3)/3
        # z = l * c*
        # where a* = 2/(a*sqrt(3)) and c* = 1/c are reciprocal lattice constants
        
        h, k, l = hkl
        
        # Calculate reciprocal lattice constants
        a_star = 2 / (self.a * np.sqrt(3))
        c_star = 1 / self.c
        
        # Calculate Cartesian coordinates
        x = (2 * h + k) * a_star / 3
        y = k * a_star * np.sqrt(3) / 3
        z = l * c_star
        
        # Create the Cartesian vector
        cartesian = np.array([x, y, z])
        
        # Normalize the vector
        norm = np.linalg.norm(cartesian)
        if norm > 0:
            cartesian = cartesian / norm
            
        return cartesian
    
    def vector_to_cartesian(self, uvtw):
        """
        Convert a crystallographic direction (vector) from Miller-Bravais indices to Cartesian coordinates.
        
        Arguments:
            uvtw: str or list, Miller-Bravais indices of the direction [uvtw]
            
        Returns:
            numpy.ndarray: Cartesian coordinates [x, y, z] of the direction vector
                          (normalized unless a physical correction is needed)
        """
        [uvtw] = self.__to_list(uvtw)
        
        # Convert to Miller indices (3-index system)
        uvw = self.Vector4to3(uvtw)
        
        # For hexagonal systems, the transformation from direct space to Cartesian is:
        # x = u * a * sqrt(3)/2
        # y = (2v + u) * a / 2
        # z = w * c
        
        u, v, w = uvw
        
        # Calculate Cartesian coordinates
        x = u * self.a * np.sqrt(3) / 2
        y = (2 * v + u) * self.a / 2
        z = w * self.c
        
        # Create the Cartesian vector
        cartesian = np.array([x, y, z])
        
        # Apply normalization if physical correction is needed
        if self.__needs_triple_correction(uvtw):
            cartesian = cartesian / 3
            
        # Return normalized vector
        norm = np.linalg.norm(cartesian)
        if norm > 0:
            cartesian = cartesian / norm
            
        return cartesian
    
    def __dotprod(self, uvtw1, uvtw2) -> float:
        # explicit formulas - see [1, 2].
        [uvtw1, uvtw2] = self.__to_np_array(uvtw1, uvtw2)
        dot = np.dot(uvtw1, np.dot(self.G_hex, uvtw2))
        return dot

    def __crossprod(self, uvtw1, uvtw2) -> list:
        [uvt1, uvt2] = [self.Vector4to3(vect) for vect in [uvtw1, uvtw2]]
        uvt2_cov = np.dot(self.g_hex, uvt2)     # uvt2 covariant using metric tensor
        cross_Miller = np.cross(uvt1, uvt2_cov)
        cross = self.Vector3to4(cross_Miller)
        return cross
    
    def norm(self, uvtw) -> float:
        """Returns the standard metric tensor norm of a vector uvtw. Explicit formula - see [1]."""
        [uvtw] = self.__to_list(uvtw)
        mag2 = self.__dotprod(uvtw, uvtw)
        mag = m.sqrt(mag2)
        return mag

    def __needs_triple_correction(self, uvtw) -> bool:
        """Check if a vector needs the 1/3 normalization factor."""
        w = uvtw[-1]
        miller = self.Vector4to3(uvtw)
        basal_divisible = all(x % 3 == 0 for x in miller[:2])       # Check basal plane projection (U and V divisible by 3)
        
        if basal_divisible and w == 0:
            return True     # Pure basal plane vectors always get correction
        if basal_divisible and w % 3 == 0: # or w % 2 == 0:
            return True     # vectors with c-component --> only when equivalent atom pos is reached
                
        return False
    
    def physical_norm(self, uvtw):
        """Returns the physically correct length of a crystallographic vector."""
        return self.norm(uvtw) / 3 if self.__needs_triple_correction(uvtw) else self.norm(uvtw)

    def physical_crossprod(self, uvtw1, uvtw2) -> list:
        """
        Calculate the cross product direction (NOT normalized, NOT conventional/fractional indices).
        Example: zone axis calculation.
        TO DO: (Calculate the cross product with physical normalization applied, output as conventional 4-ind vector, e.g. [11-20].)
        """
        cross = self.__crossprod(uvtw1, uvtw2)
        cross = cross / np.max(np.abs(cross))
        cross = np.round(cross, 10)         # Round to handle floating-point precision
        return cross

    def n_to_hkil(self, hkil) -> list:
        """
        Calculate the normal vector to a plane (hkil) in a hexagonal system.
        NB: NOT mathematically consistent.
        Arguments:  hkil: str (or list), Miller-Bravais indices separated by spaces.
        Returns:    uvtw: list, indices of normal to the plane (hkil). ! Not Miller-Bravais if l!=0 !
        """
        [uvtw] = self.__to_list(deepcopy(hkil))
        [h, k, i, l] = uvtw
        if (l != 0) & ([h, k, i, l] != [0, 0, 0, 1]):
            l = 3 * self.a**2 * l / (2 * self.c**2)
        return [h, k, i, l]

    def vector_in_plane(self, vector, plane_n):
        cosa, _ = self.angle_bw_directions(vector, plane_n)
        if abs(cosa) == 0:
            return True
        return False

    def angle_bw_directions(self, uvtw1, uvtw2) -> float:
        """
        Returns the math angle between directions.
        """
        [uvtw1, uvtw2] = self.__to_list(uvtw1, uvtw2)
        # [uvt1, uvt2] = [self.Vector4to3(vect) for vect in [uvtw1, uvtw2]]
        # cos_phi = np.dot(uvt1, uvt2) / np.linalg.norm(uvt1) / np.linalg.norm(uvt2)
        # First, determine if either vector needs the 1/3 correction
        dot = self.__dotprod(uvtw1, uvtw2)
        norm1 = self.norm(uvtw1)
        norm2 = self.norm(uvtw2)
        
        cos_phi = dot / (norm1 * norm2)
        cos_phi = round(cos_phi, 10)
        phi = round(m.degrees(m.acos(cos_phi)), 8)
        return cos_phi, phi

    def angle_bw_planes(self, hkil1, hkil2):
        """
        Equivalent to angle between normals to planes.
        Arguments:
            hkil1, hkil2: str, two planes. Miller-Bravais indices separated by spaces.
        Returns:
            cos(phi): float
            phi:      float, angle between planes hkil1 and hkil2, in degrees.
        """
        n1, n2 = self.n_to_hkil(hkil1), self.n_to_hkil(hkil2)
        cos_phi, phi = self.angle_bw_directions(n1, n2)
        return cos_phi, phi
    
    @staticmethod
    def __remove_inverse_uvt(tuples_set):
        """Apply 'set' to inverse indices. To be used in list_equiv"""
        unique_tuples = set()
        for uvt in tuples_set:
            inverse_uvt = tuple(-x for x in uvt)
            if uvt not in unique_tuples and inverse_uvt not in unique_tuples:
                unique_tuples.add(uvt)
        return unique_tuples
    
    def list_equiv(self, uvtw):
        """
        See symmetry operations in hexagonal lattices (3 rotations around c, 1 inversion across planes passing c).
        Returns: list, all equivalent uvtw directions.
        """
        [uvtw] = self.__to_list(uvtw)
        if uvtw == [0,0,0,1]:   # special case of '0001', it's unique
            return [uvtw]
        uvt_superset = set(permutations(uvtw[:-1]))
        uvt_trueset = self.__remove_inverse_uvt(uvt_superset)
        w = uvtw[-1]
        if w != 0:
            equiv_list = list(chain.from_iterable([[[*perm, w], [*perm, -w]] for perm in uvt_trueset]))
        else:
            equiv_list = [[*perm, w] for perm in uvt_trueset]
        return equiv_list
    
    def list_slip_systems(self) -> None:
        """
        Initialize slip systems of the crystal.
        To access the result after function execution: call self.slip_systems
        Returns: dict, {'slip plane': [Burger's vectors]} 
        """
        self.slip_systems = {}
        for generic_slip_mode in self.generic_slip_modes.values():
            for plane in self.list_equiv(generic_slip_mode):
                slip_plane = ' '.join(str(i) for i in plane)
                self.slip_systems[slip_plane] = []
                n = self.n_to_hkil(plane)
                for generic_b in self.generic_perfect_Burgers.values():
                    for b in self.list_equiv(generic_b):
                        cos_phi, _ = self.angle_bw_directions(n, b)
                        if cos_phi == 0:
                            self.slip_systems[slip_plane].append(b)

    # 4. ELASTICITY
    def YoungForDir(self, uvtw):
        """Return Young's modulus in the given 4-ind dir."""
        [_, _, w] = self.Vector4to3(uvtw)
        S = self.Sij
        E_recip = (1 - w**2)**2 * S[0,0] + (w**2)**4 * S[2,2] + \
                    w**2 * (1 - w**2) * (2 * S[0,2] + S[3,3])
        return 1/E_recip

    # def __YoungForDirMiller(self, uvw):
    #     """Return Young's modulus in the given 4-ind dir."""
    #     S = self.Sij
    #     
    #     return 1/E_recip

    def E_nu_bulk(self):    # for indentation Sneddon correction
        C = self.Cij
        C11, C12, C13 = C[0,0], C[0,1], C[0,2]
        C33, C55, C66 = C[2,2], C[4,4], C[5,5]

        # [Shein et al. 2007 doi.org/10.1134/S106378340706008X]
        BV = 2/9 * (C11 + C12 + 2*C13 + 0.5*C33)
        repeated_term = ((C11 + C12)*C33 - 2*C12**2)
        BR = repeated_term / (C11 + C12 + 2*C33 - 4*C13)
        GV = 1/30 * (C11 + C12 + 2*C33 - \
                     4*C13 + 12*C55 + 12*C66)
        GR = 5/2 * repeated_term *C55*C66 / \
            (3*BV*C55*C66 + repeated_term*(C55+C66))        # or repeated_term**2 ?
        
        print("Bulk moduli for substrate Sneddon:")
        print(f"Voigt: BV {BV:.2f}, GV {GV:.2f}")
        print(f"Reuss: BR {BR:.2f}, GR {GR:.2f}")

        # [Ding et al. 2020 arXiv preprint arXiv:2003.07546]
        # S = self.Sij
        # S11, S22, S33 = S[0,0], S[1,1], S[2,2]
        # S12, S13, S23 = S[0,1], S[0,2], S[1,2]
        # S44, S55, S66 = S[3,3], S[4,4], S[5,5]
        # BR = ((S11+S22+S33) + 2*(S12+S13+S23))**(-1)
        # GR = 15 * (4*(S11+S22+S33) - (S12+S13+S23) + 3*(S44+S55+S66))**(-1)
        # print(f"S Reuss: BR {BR}, GR {GR}")

        B = 0.5 * (BV + BR)     # Voigt–Reuss–Hill (VRH) procedure
        G = 0.5 * (GV + GR)

        nu = (3*B-2*G) / (2*(3*B+G))
        E = (9*B*G) / (3*B+G)

        return E, nu