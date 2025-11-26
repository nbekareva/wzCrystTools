import re
from adjustText import adjust_text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    # "font.family": "helvetica",
    "font.size": 14,
    "mathtext.default": "regular"
})

plane_names = {'basal': 'Bas',
               'pyramidalIV': 'PyIV',
               'pyramidalIII': 'PyIII',
               'pyramidalII': 'PyII',
               'pyramidalI': 'PyI',
               'prismatic_m': 'Prm',
               'prismatic_a': 'Pra'}

def format_to_latex(plane, b):
    fplane = r'\{' + plane_names.get(plane, plane) + r'\}'
    # Replace -2 with \bar{2}, -1 with \bar{1}, etc.
    fb = re.sub(r'-(\d)', r'\\bar{\1}', b)
    fb = re.sub(r' ', r'', fb)
    label = r"${} \langle {} \rangle$".format(fplane, fb)
    return label


# orient = [1, 1, -2, 1.8726]         # ~45 deg with [0001]
# orient = [1, 0, -1, 1.0809]
orient = "1 1 -2 0"

# dtype = [
#     ('orient', 'U10'), ('plane_conv_name', 'U6'), ('i', 'i1'),
#     ('b_conv_name', 'U6'), ('j', 'i1'), ('plane', 'U6'), ('b', 'U6'),
#     ('phi', 'f12'), ('lambda', 'f12'), 
#     ('Schmid', 'f12'), ('b_norm', 'f12'), ('m/b2', 'f12')
#          ]
# data = np.genfromtxt(f'Schmid_factors_{orient}.csv', delimiter='\t', \
                    # names=True, dtype=None, encoding='utf-8')
df = pd.read_csv(f'Schmid_factors_{orient}.csv', delimiter='\t')

# # Column index to check for duplicates
# column_name = 'm/b2'

# # Drop duplicates based on 'Name' column
# df_no_duplicates = df.drop_duplicates(subset=column_name)

# print("\nDataFrame after dropping duplicates based on 'Name' column:")
# print(df_no_duplicates)


# Plotting
# Filter unique values of 'm/b2' and corresponding labels
unique_values = df['m/b2'].drop_duplicates()
thresh = unique_values.max() * 0.25
print(f"Max m/b2: {unique_values.max()}, annotate values > thresh 30%: > {thresh}")

# assign unique colors to same labels=same SS
df["label"] = df.apply(
    lambda row: format_to_latex(row.plane_conv_name, row.b_conv_name), axis=1       # row = slip system
)
unique_labels = df["label"].unique()
cmap = plt.get_cmap("Set1", len(unique_labels))

label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}


texts = []

# Plot scatterplot
# cm = 1/2.54
# plt.figure(figsize=(16*cm, 6.8*cm))
fig, ax = plt.subplots(figsize=(5.5, 4.4))
for i, value in enumerate(unique_values):
    # Filter data for current 'm/b2' value
    mask = df['m/b2'] == value
    subset = df[mask]
    x = subset['abs(Schmid)']
    y = subset['m/b2']
    # # Plot scatter points with corresponding labels
    color = label_to_color[subset["label"].values[0]]
    ax.scatter(x, y, color=color, alpha=0.8)

    if abs(y.values[0]) > thresh:
        plane, b = subset[['plane_conv_name', 'b_conv_name']].values[0]
        # print(plane, b)
        # render plane name & b vector to LaTeX format
        label = format_to_latex(plane, b)
        
        t = ax.annotate(
            label,
            (x.values[0], y.values[0]),
            fontstyle='italic',
            fontsize='x-small'
        )
        texts.append(t)

# automatic collision resolution!
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.5))

# plt.yticks(range(len(unique_values)), unique_values)
# plt.xticks(np.arange(-5,6,1)*0.1)
# plt.locator_params(axis='y', nbins=6)       # Set number of y-ticks
# plt.xlim(0, 0.55)
# plt.ylim(0, None)  # Start y-axis from 0
ax.set_xlabel(r'Schmid factor $|m|$')
ax.set_ylabel(r'$\frac{|m|}{b^2}$')
ax.set_title(f'<{orient}>-oriented ZnO micropillars', fontsize=12)
# plt.legend()
ax.grid(color='gray', linestyle='dotted', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(f'Schmid_factors_{orient}.png', dpi=300, bbox_inches='tight')
# save as svg
# plt.gca().set_position([0, 0, 1, 1])
# plt.savefig(f'Schmid_factors_{orient}.svg')