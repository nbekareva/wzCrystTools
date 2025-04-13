## EXAMPLES ##
from hexag_cristallo_tool import *

# ZnO
a = 3.238118
c = 5.176434
# Al2O3
# a = 3.238118214981954
# c = 5.176434187914933
crystal = Wurtzite(a, c)      # initialize a hexagonal crystalline system

# 1. Normal to plane, norm of a vector
# plane = '-1 2 -1 2'
# print(f"Normal to {plane}", crystal.n_to_hkil(plane))
# print(crystal.norm('0 0 0 1'))

# 2. Angle bw planes, bw directions in the system
# _, phi = crystal.angle_bw_planes(plane, '0 0 0 1')
# print(phi)
# print(m.degrees(m.atan(c/a/m.sin(m.pi/3))))

plane, dir = '1 -2 1 2', '-1 -1 2 3'    # sic!: [5 -4 -1 3] ~ [2-11]
print(f"plane {crystal.PlaneBravaisToMiller(plane)}, dir {crystal.VectorBravaisToMiller(dir)}")
# print(f"dir back to Bravais: {crystal.VectorMillerToBravais(crystal.VectorBravaisToMiller(dir))}")      # CHECK IF CORRECT
print("angle bw dirs", crystal.angle_bw_directions(crystal.n_to_hkil(plane), dir))
# dir2 = '2 -2 0 3'
# print([i*3 for i in crystal.VectorBravaisToMiller(dir2)])
# print(crystal.angle_bw_directions(dir, dir2))

# print(crystal.norm('5 -1 -4 3') / 2)

# 3. List equivalent planes / directions
# print(crystal.list_equiv([1, 0, -1, 0]))
# print(crystal.list_equiv('2 -1 -1 3'))
# print(crystal.list_equiv('-2 1 1 3'))

## 4. Schmid factor: simple example, tryout random vectors
# z = '-1 0 1 0'
# n = '0 1 -1 0'
# b = '2 -1 -1 0'
# cos_phi, phi = crystal.angle_bw_planes(z, n)
# cos_lambda, lambdaa = crystal.angle_bw_directions(z, b)
# print('b in plane?', crystal.angle_bw_directions(n, b))
# print(phi, lambdaa)
# print('Schmid factor', cos_phi, cos_lambda, cos_phi*cos_lambda)

# 5. Schmid factor: complete example
# crystal.list_slip_systems()
# # print(crystal.slip_systems)
# orient = '1 1 -2 0'

# file = open(f'Schmid_factors_{orient}.csv', 'w')
# file.write('orient\t b\t plane\t phi\t lambda\t Schmid\n')

# for plane, Burgers in crystal.slip_systems.items():
#     n = crystal.n_to_hkil(plane)
#     cos_phi, phi = crystal.angle_bw_planes(orient, plane)
#     for b in Burgers:
#         cos_lambda, lambdaa = crystal.angle_bw_directions(orient, b)
#         Schmid = cos_phi * cos_lambda
#         if Schmid != 0:
#             # print(f'{orient}\t {" ".join(str(i) for i in b)}\t {plane}\t {phi}\t {lambdaa}\t {Schmid}')
#             file.write(f'{orient}\t {" ".join(str(i) for i in b)}\t {plane}\t {phi}\t {lambdaa}\t {Schmid}\n')

# file.close()


# ELASTICITY CALC
# a = 3.25
# c = 5.20
# crystal = Wurtzite(a, c)      # initialize a hexagonal crystalline system
# print(f"Wurtzite crystal: a {a} A, c {c} A")

# print(crystal.Sij)
# print(crystal.YoungForDir('1 1 -2 0'), "GPa")
# S = crystal.Sij
# print(S[0,0], S[2,2], S[0,2], S[3,3])

# Draw the surface representation of E - to do later
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# x = np.cos(u)*np.sin(v)
# y = np.sin(u)*np.sin(v)
# z = np.cos(v)
