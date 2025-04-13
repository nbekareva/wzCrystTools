import itertools
from math import isclose
import sys
sys.path.append("/home/nbekareva/TOOLS/utils")
from typing import List, Union, Tuple, Dict, Set, Optional, Literal
import numpy as np
from hexag_cristallo_tool_beta import Wurtzite


def combinations_from_two_lists(list1, list2):
    """
    Generate all possible combinations of 2 elements, one from each list,
    where order in the pair doesn't matter.
    
    Args:
        list1: First list of elements
        list2: Second list of elements
        
    Returns:
        A list of unique combinations (as frozensets)
    """
    all_pairs = itertools.product(list1, list2)
    # Only keep pairs where the elements are different
    unique_combinations = set()
    for a, b in all_pairs:
        if a != b:  # Only include pairs with different elements
            # Sort to ignore order (a,b) == (b,a)
            pair = frozenset((a, b))
            unique_combinations.add(pair)
    
    return unique_combinations


    # def dirs_from_2families(self, pole_family1, pole_family2):          # old code, TO EDIT
    #     list_a = self.crystal.equivalent_directions(pole_family1)
    #     list_b = self.crystal.equivalent_directions(pole_family2)
    #     list_a = [tuple(a) for a in list_a]
    #     list_b = [tuple(b) for b in list_b]

    #     combinations = combinations_from_two_lists(list_a, list_b)
    #     angles = set()

    #     for combo in combinations:
    #         if not all(self.in_zone(c, approx_zone_axis, pole) for pole in combo):
    #             continue

    #         (n1, n2) = (self.crystal.plane_normal(list(dir)) for dir in combo)
    #         cos, angle_bw_planes = self.crystal.angle_between_directions(n1, n2)
    #         angles.add(angle_bw_planes)

    #         if isclose(angle_bw_planes, 72.6, abs_tol=0.5):
    #             print(f"{tuple(combo)}\t{(n1*3, n2*3)}\t{cos:.2f}\t{angle_bw_planes:.2f}")

    #     # print(f"Total number of combinations: {len(combinations)}")
    #     # print(f"Unique angles: {angles}")
    
class WzStereogram:
    """
    Requires Wurtzite from hexag_cristallo_tool_beta, where hkil, uvtw are lists.
    Here they become tuples, mutability is no more allowed for stereogram calculations.
    """
    def __init__(self, a=3.2494, c=5.2054):
        self.crystal = Wurtzite(a, c)
        
        for pole_type in ["planes", "directions"]:
            setattr(self, f"generic_{pole_type}_poles", self._init_generic_poles(pole_type=pole_type))
            setattr(self, f"all_{pole_type}_poles", self._init_all_poles(pole_type=pole_type))

    def _init_generic_poles(self, pole_type: Literal["planes", "directions"], dhkl_min=0.8):
        generic_poles = set()
        source_dict = getattr(self.crystal, f"generic_{pole_type}")
        
        for pole_str in source_dict.values():
            generic_poles.add(pole_str)

        for i in range(1, 6):
            generic_poles.add(f'2 -1 -1 {i}')
            # if dhkl_min < 0.9:
                # generic_poles.add(f'4 -2 -2 {i}')
            generic_poles.add(f'1 0 -1 {i}')
            if i < 4:
                generic_poles.add(f'1 2 -3 {i}')
        generic_poles.add(f'2 0 -2 1')
        generic_poles.add(f'2 0 -2 3')
        generic_poles.add(f'2 0 -2 5')
        generic_poles.add(f'3 0 -3 1')

        return generic_poles
    
    def _init_all_poles(self, pole_type: Literal["planes", "directions"], pole_set=None, dhkl_min=0.8):
        unique_poles = set()

        if pole_set is None:
            pole_set = getattr(self, f"generic_{pole_type}_poles")

        for pole_family in pole_set:
            equiv_poles = self.crystal.equivalent_directions(pole_family, drop_inverse=False)

            for equiv_pole in equiv_poles:
                pole_tuple = tuple(equiv_pole)      # Convert to tuple for hashability in the set
                unique_poles.add(pole_tuple)
    
        return unique_poles

    def in_zone(self, B, pole, pole_type: Literal["planes", "directions"], abs_tol=5):

        if pole_type == "planes":
            vector = self.crystal.plane_normal(pole)
        else:
            vector = pole

        _, angle = self.crystal.angle_between_directions(B, vector)

        if isclose(angle, 90, abs_tol=abs_tol):
            return True
        
        return False

    def find_poles_in_zone(self, zone_axis, pole_type: Literal["planes", "directions"], abs_tol=5):
        all_poles = getattr(self, f"all_{pole_type}_poles")
        in_zone_mask = (self.in_zone(B=zone_axis, pole=pole, pole_type=pole_type, abs_tol=abs_tol) for pole in all_poles)
        poles_in_zone = list(itertools.compress(all_poles, in_zone_mask))

        return poles_in_zone

    def index_DP_spots(self, angle_bw_spots, plane_poles_in_zone):
        plane_pole_combos = combinations_from_two_lists(plane_poles_in_zone, plane_poles_in_zone)
        all_angles = set()

        for combo in plane_pole_combos:
            (n1, n2) = (self.crystal.plane_normal(list(pole)) for pole in combo)
            cos, angle_bw_planes = proj.crystal.angle_between_directions(n1, n2)
            all_angles.add(angle_bw_planes)

            if isclose(angle_bw_planes, angle_bw_spots, abs_tol=0.5):
                combo = tuple(combo)
                pole1_str = " ".join(map(str, combo[0]))
                pole2_str = " ".join(map(str, combo[1]))
                print(f"{pole1_str}\t{pole2_str}\t{cos:.2f}\t{angle_bw_planes:.2f}")

    def b_by_extictions(self, *kwargs, abs_tol=5):
        common_dirs = None

        for g_ext in kwargs:
            n_g_ext = self.crystal.plane_normal(g_ext)
            dirs_in_zone = self.find_poles_in_zone(zone_axis=n_g_ext, pole_type="directions", abs_tol=abs_tol)

            dirs_set = set(map(tuple, dirs_in_zone))  # Convert inner lists/arrays to tuples

            if common_dirs is None:
                common_dirs = dirs_set
            else:
                common_dirs &= dirs_set  # Intersect with previous set

        return list(common_dirs) if common_dirs is not None else []

if __name__ == "__main__":
    
    # DATA TO INPUT:
    approx_zone_axis = '7 -5 -2 -3'       # impossible to index
    # approx_zone_axis = '8 -10 2 -3'
    # approx_zone_axis = '1 0 -1 2'
    some_DP_angle = 30.7

    proj = WzStereogram(a=3.2494, c=5.2054)

    for pole_type in ["planes", "directions"]:
        in_zone = proj.find_poles_in_zone(approx_zone_axis, pole_type=pole_type, abs_tol=7)
        print(f" *** {pole_type} in zone with B=[{approx_zone_axis}]:\n {in_zone}\n")

        if pole_type == "planes":
            plane_poles_in_zone = in_zone
    
    print("Identified poles:")
    proj.index_DP_spots(some_DP_angle, plane_poles_in_zone=plane_poles_in_zone)
    
    print(proj.b_by_extictions('1 0 -1 0', '1 0 -1 1', '1 0 -1 2', '2 -1 -1 -2', abs_tol=20))

    # print(f"Total number of combinations: {len(combinations)}")
    # print(f"Unique angles: {sorted(list(angles))}")
    