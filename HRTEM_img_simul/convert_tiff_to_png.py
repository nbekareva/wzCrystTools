import os
from hrtem_helpers import tif_to_png

folder = '/mnt/c/Users/a.walrave/Documents/M2 Internship & PhD/DataTreatment/HRTEM Titan/2024-05-14/ZnO_0001Zn_P3_HRTEM07_py_ifft'
file = 'ABSF Filtered ZnO_0001Zn_P3_HRTEM07.tif'

tif_path = os.path.join(folder, file)

tif_to_png(tif_path, scale_bar=False)