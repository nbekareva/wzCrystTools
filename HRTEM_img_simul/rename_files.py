from pathlib import Path


directory = Path("/mnt/c/Users/a.walrave/Documents/M2 Internship & PhD/DataTreatment/HRTEM Titan/2024-05-14")
for file in directory.iterdir():
    if file.is_file() and file.suffix == ".png" and file.name.endswith("analysis.png"):
        new_name = file.name.replace("analysis", "manual")
        new_path = file.parent / new_name
        print(f"Renaming {file} to {new_path}")
        file.rename(new_path)