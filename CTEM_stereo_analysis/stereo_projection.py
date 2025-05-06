import itertools
from math import isclose
import re
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
    def __init__(self, a=3.2494, c=5.2054, dhkl_list_file=None):
        self.crystal = Wurtzite(a, c)
        self.dhkl_list = self._load_dhkl_list(dhkl_list_file) if dhkl_list_file else None
        self.dhkl_min = 0.8
        
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

    def _load_dhkl_list(self, filename: str) -> Dict[str, List[Union[float, float]]]:
        """
        Read dhkl list from a file.
        Returns: dictionary as follows: 
            "h k i l": [dhkl, Intensity]
            where "h k i l" is a string and [dhkl, Intensity] is a list of floats.
        """
        data = {}
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                hkil = re.sub(r"[()']", "", " ".join(parts[:4]))
                dhkl = float(parts[5]) if len(parts) > 4 else None
                intensity = float(parts[9]) if len(parts) > 5 else None
                data[hkil] = [dhkl, intensity]
        return data
    
    def get_dhkl_int(self, plane_pole: str) -> Optional[Tuple[float, float]]:
        """
        Get dhkl and intensity values for a given pole from the dhkl list.
        Args:
            pole: The pole for which to get the dhkl and intensity values.
        Returns:
            The dhkl values if found, otherwise None.
        """
        if self.dhkl_list is None:
            raise ValueError("dhkl_list is not initialized. Please provide a valid dhkl_list_file during initialization.")
        
        for equiv_pole_list in self.crystal.equivalent_directions(plane_pole, drop_inverse=False):  # Get equivalent poles (format is list)
            equiv_pole = " ".join(map(str, equiv_pole_list))        # Convert back to string
            if equiv_pole in self.dhkl_list.keys():
                dhkl, intensity = self.dhkl_list[equiv_pole][0], self.dhkl_list[equiv_pole][1]
                return dhkl, intensity
        return None, None
    
    def is_spot_off(self, plane_pole: str, min_intensity=7) -> bool:
        """
        Check if a spot is off by checking its intensity value from dhkl list provided.
        Tolerance (minimal spot intensity) is set to 7 by default.

        Raises:
            ValueError: If `self.dhkl_list` is not initialized.
        """
        _, intensity = self.get_dhkl_int(plane_pole)
        if intensity is not None and intensity > min_intensity:
            return False
        return True

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
        """
        Find all poles of a given type that are in the zone defined by the zone axis.
        Args:
            zone_axis: The zone axis to check against.
            pole_type: The type of poles to find ("planes" or "directions").
            abs_tol: Absolute tolerance for angle to determine if the pole is in zone.
        Returns:
            A list of poles of the specified type that are in the zone defined by the zone axis.
        """
        all_poles = getattr(self, f"all_{pole_type}_poles")
        in_zone_mask = (self.in_zone(B=zone_axis, pole=pole, pole_type=pole_type, abs_tol=abs_tol) for pole in all_poles)
        poles_in_zone = list(itertools.compress(all_poles, in_zone_mask))

        return poles_in_zone

    def index_DP_spots(self, angle_bw_spots, plane_poles_in_zone, drop_weak_spots=True, min_intensity=7):
        plane_pole_combos = combinations_from_two_lists(plane_poles_in_zone, plane_poles_in_zone)
        all_angles = set()

        for combo in plane_pole_combos:
            if drop_weak_spots and any(self.is_spot_off(pole, min_intensity=min_intensity) for pole in combo):
                continue
            (n1, n2) = (self.crystal.plane_normal(list(pole)) for pole in combo)
            cos, angle_bw_planes = proj.crystal.angle_between_directions(n1, n2)
            all_angles.add(angle_bw_planes)

            if isclose(angle_bw_planes, angle_bw_spots, abs_tol=0.5):
                combo = tuple(combo)
                pole1_str = " ".join(map(str, combo[0]))
                pole2_str = " ".join(map(str, combo[1]))
                dhkl1, int1 = self.get_dhkl_int(pole1_str)
                dhkl2, int2 = self.get_dhkl_int(pole2_str)
                print(f"{pole1_str}\t{pole2_str}\t{angle_bw_planes:.2f}\t{dhkl1}\t{int1}\t{dhkl2}\t{int2}")

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
    dhkl_list_file = "dhkl_list.txt"
    approx_zone_axis = '7 -5 -2 -3'       # impossible to index
    # approx_zone_axis = '8 -10 2 -3'
    # approx_zone_axis = '1 0 -1 2'
    abs_tol_in_zone = 7
    weak_none, min_intensity = False, 7
    some_DP_angle = 72.6

    proj = WzStereogram(a=3.2494, c=5.2054, dhkl_list_file=dhkl_list_file)
    print(f"Loaded dhkl list from {dhkl_list_file}")
    print(f"Approximate zone axis: {approx_zone_axis}")
    print(f"Approximate angle between searched diffraction spots: {some_DP_angle} degrees\n")

    for pole_type in ["planes", "directions"]:
        in_zone = proj.find_poles_in_zone(approx_zone_axis, pole_type=pole_type, abs_tol=abs_tol_in_zone) 
        print(f" *** {pole_type} in zone with B=[{approx_zone_axis}]:\n {in_zone}\n")

        if pole_type == "planes":
            plane_poles_in_zone = in_zone
    
    print("Identified poles:")
    print(f"Pole 1\t\tPole 2\t\tAngle\tdhkl1\tint1\tdhkl2\tint2")
    proj.index_DP_spots(some_DP_angle, plane_poles_in_zone=plane_poles_in_zone, drop_weak_spots=weak_none, min_intensity=min_intensity)
    
    print(proj.b_by_extictions('1 0 -1 0', '1 0 -1 1', '1 0 -1 2', '2 -1 -1 -2', abs_tol=5))

    # print(f"Total number of combinations: {len(combinations)}")
    # print(f"Unique angles: {sorted(list(angles))}")
    