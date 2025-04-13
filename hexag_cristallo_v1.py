# Requirements: numpy
# References:
# [1] https://ssd.phys.strath.ac.uk/resources/crystallography/crystallographic-direction-calculator/
# [2] https://youtu.be/vIomv4fFTHw?si=_MYn0UngxuZ6QRaB
from copy import deepcopy
import math as m
from itertools import permutations
import numpy as np


class Wurtzite:
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

    @staticmethod
    def __to_list(*args) -> list:
        """Convert str vectors to lists."""
        return [list(map(int, item.split())) if type(item) is str else item for item in args]
    
    @staticmethod
    def __to_np_array(*args):
        """Convert list vectors to np.arrays."""
        return [np.array(item) if type(item) is list else item for item in args]
    
    def __dotprod(self, uvtw1, uvtw2) -> float:
        # explicit formulas - see [1, 2].
        [uvtw1, uvtw2] = self.__to_np_array(uvtw1, uvtw2)
        dot = np.dot(uvtw1, np.dot(self.G_hex, uvtw2))
        return dot

    def norm(self, uvtw) -> float:
        # explicit formula - see [1].
        [uvtw] = self.__to_list(uvtw)
        mag2 = self.__dotprod(uvtw, uvtw)
        mag = m.sqrt(mag2)
        return mag

    def n_to_hkil(self, hkil) -> list:
        """
        Arguments:  hkil: str (or list), Miller-Bravais indices separated by spaces.
        Returns:    uvtw: list, indices of normal to the plane (hkil). ! Not Miller-Bravais if l!=0 !
        """
        [uvtw] = self.__to_list(deepcopy(hkil))
        if (uvtw[-1] != 0) & (uvtw != [0, 0, 0, 1]):
            uvtw[-1] = 3 * self.a**2 * uvtw[-1] / (2 * self.c**2)
        return uvtw

    def angle_bw_directions(self, uvtw1, uvtw2) -> float:
        [uvtw1, uvtw2] = self.__to_list(uvtw1, uvtw2)
        cos_theta = self.__dotprod(uvtw1, uvtw2) / self.norm(uvtw1) / self.norm(uvtw2)
        cos_theta = round(cos_theta, 10)
        theta = round(m.degrees(m.acos(cos_theta)), 8)
        return cos_theta, theta

    def angle_bw_planes(self, hkil1, hkil2):
        """
        Equivalent to angle between normals to planes.
        Arguments:
            hkil1, hkil2: str, two planes. Miller-Bravais indices separated by spaces.
        Returns:
            cos(theta): float
            theta:      float, angle between planes hkil1 and hkil2, in degrees.
        """
        n1, n2 = self.n_to_hkil(hkil1), self.n_to_hkil(hkil2)
        cos_theta, theta = self.angle_bw_directions(n1, n2)
        return cos_theta, theta
    
    def list_equiv(self, uvtw):
        """
        Returns: list, all equivalent uvtw directions for w=const.
        """
        # ? not sure about keeping w=const if listing equiv Burger vectors
        # ? not sure about keeping inverse normals, f.e. (1 0 -1 0), (-1 0 1 0)
        [uvtw] = self.__to_list(uvtw)
        w = uvtw[-1]
        return [[*perm, w] for perm in set(permutations(uvtw[:-1]))]
    
    def list_slip_systems(self) -> None:
        """
        Initialize slip systems of the crystal.
        Access the result after function execution: call self.slip_systems
        Returns: dict, {'slip plane': [Burger's vectors]} 
        """
        self.slip_systems = {}
        for generic_plane in ['0 0 0 1', '1 0 -1 0', '2 -1 -1 0', '1 0 -1 1', '2 -1 -1 1']:
            for plane in self.list_equiv(generic_plane):
                slip_plane = ' '.join(str(i) for i in plane)
                self.slip_systems[slip_plane] = []
                n = self.n_to_hkil(plane)
                for generic_b in ['0 0 0 1', '2 -1 -1 0', '2 -1 -1 3', '-2 1 1 3']:
                    for b in self.list_equiv(generic_b):
                        cos_theta, _ = crystal.angle_bw_directions(n, b)
                        if cos_theta == 0:
                            self.slip_systems[slip_plane].append(b)


## EXAMPLES ##
# ZnO
# a = 3.238118
# c = 5.176434
# Al2O3
a = 3.238118214981954
c = 5.176434187914933
crystal = Wurtzite(a, c)      # initialize a hexagonal crystalline system
print("Wurtzite", a, c)

# 1. Normal to plane, norm of a vector
plane = '1 0 -1 1'
print(f"Normal to {plane}", crystal.n_to_hkil(plane))
# print(crystal.norm('0 0 0 1'))

# 2. Angle bw planes, bw directions in the system
_, phi = crystal.angle_bw_planes(plane, '0 0 0 1')
print(phi)
print(m.degrees(m.atan(c/a/m.sin(m.pi/3))))
print("angle bw dirs", crystal.angle_bw_directions('-1 0 1 2', crystal.n_to_hkil('1 0 -1 1')))

# 3. List equivalent planes / directions
# print(crystal.list_equiv([1, 0, -1, 0]))

## 4. Schmid factor: simple example, tryout random vectors
# z = '-1 0 1 0'
# n = '0 1 -1 0'
# b = '2 -1 -1 0'
# cos_theta, theta = crystal.angle_bw_planes(z, n)
# cos_lambda, lambdaa = crystal.angle_bw_directions(z, b)
# print('b in plane?', crystal.angle_bw_directions(n, b))
# print(theta, lambdaa)
# print('Schmid factor', cos_theta, cos_lambda, cos_theta*cos_lambda)

# 5. Schmid factor: complete example
# crystal.list_slip_systems()
# orient = '1 1 -2 0'

# file = open(f'Schmid_factors_{orient}.csv', 'w')
# file.write('orient\t plane\t b\t theta\t lambda\t Schmid\n')

# for plane, Burgers in crystal.slip_systems.items():
#     n = crystal.n_to_hkil(plane)
#     cos_theta, theta = crystal.angle_bw_planes(orient, plane)
#     for b in Burgers:
#         cos_lambda, lambdaa = crystal.angle_bw_directions(orient, b)
#         Schmid = cos_theta * cos_lambda
#         if Schmid != 0:
#             file.write(f'{orient}\t {plane}\t {" ".join(str(i) for i in b)}\t {theta}\t {lambdaa}\t {Schmid}\n')

# file.close()
