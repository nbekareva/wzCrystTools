import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": False,
    # "font.family": "helvetica",
    "font.size": 28,
    "mathtext.default": "regular"
})


def plot_yline(filename, label, col):
    data = np.loadtxt(filename, skiprows=1)
    x = data[:, col] / max(data[:, col])
    y = data[:, 7]
    plt.plot(x, y, '-o', markersize=5, label=label, c='purple')

plt.figure(figsize=(8, 6))

for (file, label) in [
             ('/home/nbekareva/TOOLS/potprop/DATABASE/ZnO/Zhang2024_NNP/STACKING_FAULTS.data/wz_x100_y010_z001_1_2_10_gsfkey1_shift0.6_pyramidalIV/gamma.data.pt', 'far set'),]:
    plot_yline(file, label, 4)     # 4: x, 3: y
plt.xlabel(r'$\langle a+c \rangle = \frac{1}{3} [1\bar{2}13]$')
plt.ylabel(r'$GSFE\ (mJ/m^2)$')
# plt.title(r'y-lines for {PyIV}')
plt.xlim(0, 1)
plt.ylim(bottom=0)
# plt.legend()
plt.grid()
plt.savefig('ylines/y_lines_pyIV_ac.png', dpi=300, bbox_inches='tight')
plt.show()