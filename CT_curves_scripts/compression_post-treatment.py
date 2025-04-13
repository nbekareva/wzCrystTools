import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import medfilt
# import re


def strip_off_non_contact(data):
    print(f"Data array shape: {data.shape}")

    # take the start of the curve d < 0.3 um
    cond = data[:, 0] < 0.3
    noise_data = data[cond, 1]

    # calc noise value s
    s = np.abs(noise_data).max()
    print(f"Noise abs value: {s} uN")

    # strip off F < s
    cond = np.abs(data[:, 1]) > s
    data = data[cond, :]

    # shift zero d
    data[:, 0] = data[:, 0] - data[0, 0]

    print(f"New data array shape: {data.shape}")
    return data


def find_lift_off(d, F, sampling_freq, slopes_medfilt_window):
    """Calculate the slope between consecutive data points and identify regions 
    where the slope remains relatively constant (indicating a linear trend). 
    You can then set thresholds for what constitutes a "relatively constant" slope."""
    slopes_raw = np.diff(F[::sampling_freq]) / np.diff(d[::sampling_freq])
    slopes = medfilt(slopes_raw, slopes_medfilt_window)

    slope_threshold = 1000
    change_indices = np.where(np.abs(np.diff(slopes)) > slope_threshold)[0]
    idx_widely_spaced = np.where(np.diff(change_indices) > 10)[0]     # keep only where slope constant at dd = 10*sampling_dist = 0.04 um
    idx_widely_spaced = np.sort(np.concatenate([idx_widely_spaced, idx_widely_spaced+1]))       # add neighbors
    change_indices = change_indices[idx_widely_spaced]
    start = change_indices[-2] * sampling_freq
    end = change_indices[-1] * sampling_freq

    plt.figure()
    plt.plot(slopes_raw, "o-", alpha=0.3, label=f"Slopes, n={sampling_freq}")
    plt.plot(slopes, "o-", alpha=0.3, label="filtered")
    plt.scatter(change_indices, slopes[change_indices], c='g', label='Flexion pts')
    plt.legend()
    plt.savefig(f"Slopes_{id}.png")

    return start, end


# *** Import data ***
# test_n = re.find()
id = 6
sampling_freq = 500
data = np.loadtxt(f"2024-03-28_15-16-08_{id}_0_Aborted.txt", delimiter="\t", skiprows=4)
N = data.shape[0]

plt.figure()
plt.scatter(data[:, 0], data[:, 1], marker="+", s=4, label="Raw data")
plt.xlabel("Displacement, um")
plt.ylabel("Force, uN")
plt.grid()
plt.savefig(f"0001O_{id}_raw.png")

# *** Noise removal ***
data = strip_off_non_contact(data)
d = data[:, 0]
F = data[:, 1]

# *** Smoothen data ***
F = savgol_filter(F, window_length=sampling_freq, polyorder=5)

# *** Find lift-off part ***
start, end = find_lift_off(d, F, sampling_freq*2, 9)

# *** Linear fit of the lift-off ***
coefficients = np.polyfit(d[start:end], F[start:end], 1)
E, f0 = coefficients
print(f"Young's module for the lift-off part: {E} uN/um")

# Generate the fitted line
fitted_F = E * d[start:end] + f0

# *** Plot results ***
plt.figure()
plt.scatter(d, F, marker="+", s=4, label="Raw data")
plt.plot(d[start:end], fitted_F, c="g", label="Linear fit")
plt.scatter(d[[start, end]], F[[start, end]], c='r')
plt.xlabel("Displacement, um")
plt.ylabel("Force, uN")
plt.legend()
plt.grid()
plt.savefig(f"0001O_{id}.png")
# plt.savefig(f"0001O_{test_n}")
