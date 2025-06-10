import sys
sys.path.append("..")  # Adjust the path to import hexag_cristallo_tool_beta
from itertools import permutations
import numpy as np
from hexag_cristallo_tool_beta import Wurtzite

#  Sample (0001)Zn P5
#  Date of LACBED session: 2025-06-05

#  Upper dislo - in 3-ind notation (drop i)
hkl1, n1 = np.array([-3, 0, 2]), 6
hkl2, n2 = np.array([2, 0, -1]), -3
hkl3, n3 = np.array([2, 0, 7]), -5
hkl4, n4 = np.array([-1, 0, -5]), 4
hkl5, n5 = np.array([-1, 0, -7]), 6

#  Lower dislo - in 3-ind notation (drop i)
# hkl1, n1 = np.array([-1, 0, -5]), 4
# hkl2, n2 = np.array([2, 0, -1]), -3

n = np.array([n1, n2, n3, n4, n5])  # Cherns factors
n = n.reshape(-1, 1)  # Reshape to a column vector

# -------------------- Solve approximately the system --------------------
# g = np.array([hkl1, hkl2, hkl3, hkl4, hkl5])  # Reciprocal lattice vectors
# # g = g.T  # Transpose to match the expected shape
# print("g matrix:\n", g, "\n", n)

# b, residuals, rank, s = np.linalg.lstsq(g, n, rcond=None)
# print("Shape of g:", g.shape)
# print("Rank of g:", rank)
# print("Singular values of g:", s)
# print("Residuals:", residuals)
# print("Least-squares solution b:\n", b.T)


# -------------------- Or check the supposed Burgers vector --------------------
supposed_b = np.array([1, 0, -1]).reshape(-1, 1)  # Example Burgers vector in 3-ind notation

hkl_dict = {    # Dictionary of hkl and their corresponding n values
    str(hkl1): n1,
    str(hkl2): n2,
    str(hkl3): n3,
    str(hkl4): n4,
    str(hkl5): n5
}

# Check if the supposed Burgers vector satisfies g*b = n for any hkl in g
for comb in permutations(hkl_dict.items(), 3):
    g = np.array([hkl[1:-1].split() for hkl, _ in comb], dtype=int)  # Extract hkl values and convert to int
    n = np.array([int(i) for _, i in comb]).reshape(-1, 1)
    # print(n.T)
    # print(np.matmul(g, supposed_b).T, "\n")
    try:
        nn = np.matmul(g, supposed_b)
        if np.allclose(n, nn):
            print(f"g satisfied")  #: {nn.T} = {n.T}")
        else:
            print(f"g not satisfied") #:\n {g}:\n {nn.T} != {n.T}")
    except np.linalg.LinAlgError:
        print(f"Combination {g} is singular or not solvable.")