import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

orient = '0 0 0 1'

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
thresh = unique_values.max() * 0.2
print(unique_values.max(), thresh)

# Plot scatterplot
cm = 1/2.54
plt.figure(figsize=(16*cm, 6.8*cm))
for i, value in enumerate(unique_values):
    # Filter data for current 'm/b2' value
    mask = df['m/b2'] == value
    subset = df[mask]
    x = subset['Schmid']
    y = subset['m/b2']
    # # Plot scatter points with corresponding labels
    plt.scatter(x, y, alpha=0.8)
    if abs(y.values[0]) > thresh:
        labels = subset[['plane_conv_name', 'i', 'b_conv_name', 'j']].values[0]
        label = f"{labels[0]}{labels[1]}{labels[2]}{labels[3]}"
        plt.annotate(label, (x.values[0], y.values[0]), fontsize='small', fontstyle='italic')

# plt.yticks(range(len(unique_values)), unique_values)
plt.xticks(np.arange(-5,6,1)*0.1)
plt.xlim(-0.55, 0.55)
plt.ylim(0, None)  # Start y-axis from 0
plt.xlabel('Schmid factor m')
plt.ylabel('m/b2')
# plt.title(f'<{orient}>-oriented ZnO micropillars')
# plt.legend()
plt.grid(color='gray', linestyle='dotted', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(f'Schmid_factors_{orient}.png')
# save as svg
plt.gca().set_position([0, 0, 1, 1])
plt.savefig(f'Schmid_factors_{orient}.svg')