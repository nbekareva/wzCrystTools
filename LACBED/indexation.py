import json
import ast
import sys
sys.path.append("..")  # Adjust the path to import hexag_cristallo_tool_beta
import numpy as np
from hexag_cristallo_tool_gamma import Wurtzite

c = Wurtzite(a=3.2494, c=5.2054)

#  Sample (0001)Zn P5   -   Upper dislo - in 3-ind notation (drop i)
#  Date of LACBED session: 2025-06-05, 2025-06-16
datafile = "2025-06-05.16_0001Zn_P5_2ndseries.json"
datafile = "2025-07-17_11-2_P4.json"

with open(datafile) as f:
    data = json.load(f)
print(f"Dislos in file: {data.keys()}\n")

# -------------------- Solve approximately the system --------------------
for dislo, splittings in data.items():
    print(f"Dislocation: {dislo}")
    g = np.array([ast.literal_eval(k) for k in splittings.keys()])  # Reciprocal lattice vectors
    # g = g.T
    n = np.array(list(splittings.values())).reshape(-1, 1)  # Cherns factors
    print("g matrix:\n", g, "\n", n)

    b, residuals, rank, s = np.linalg.lstsq(g, n, rcond=None)
    print("Shape of g:", g.shape)
    print("Rank of g:", rank)
    print("Singular values of g:", s)
    print("Residuals:", residuals)
    print(f"Least-squares solution b:\n {b.T} == {c.vector_3ind_to_4ind(b).T.round(3)}\n")


# -------------------- Or check the family of supposed Burgers vectors --------------------

# pyI_supposed_b_list = ['-1 0 1 2', '-1 -1 2 3', '0 -1 1 1', '1 -5 4 3', '1 -2 1 0', \
#        '4 -5 1 -3', '1 -1 0 -1', '2 -1 -1 -3', '1 0 -1 -2']
# pyIV_supposed_b_list = ['-2 -2 4 3', '0 -1 1 0', '2 -4 2 -3', '2 -1 -1 -3']

# Check if the supposed Burgers vector satisfies g*b = n for any hkl in g
# supposed_b_list = c.equivalent_directions('1 1 -2 3', drop_inverse=False)

# for supposed_b in supposed_b_list:
#     supposed_b = c.vector_4ind_to_3ind(supposed_b).reshape(-1, 1)
#     counter, bad_ns = 0, []
#     print(f"Checking Burgers vector: {c.vector_3ind_to_4ind(supposed_b).T} == {supposed_b.T}")

#     for g, n in data.items():
#         g = np.array(ast.literal_eval(g))
#         [nn] = np.dot(g, supposed_b)  # Calculate g*b
#         if nn == n:
#             # print(f"    {g} satisfied,\t{n} = {nn}")
#             counter += 1
#         else:
#             bad_ns.append((n, int(nn)))
#         #     print(f"    {g} not satisfied,\t{n} != {nn}")

#     print(f"    {counter} out of {len(data.items())} satisfied, n expected/got: {bad_ns}\n")


# -------------------- Check only one supposed Burgers vector --------------------
supposed_b = c.vector_4ind_to_3ind('-1 -1 2 0').reshape(-1, 1)  # Example Burgers vector in 3-ind notation

for g, n in data["dislo 2"].items():
    g = np.array(ast.literal_eval(g))
    [nn] = np.dot(g, supposed_b)  # Calculate g*b
    if nn == n:
        print(f"    {g} satisfied,\t{n} = {nn}")
    else:
        print(f"    {g} not satisfied,\t{n} != {nn}")