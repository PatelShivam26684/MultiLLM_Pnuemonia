import os
import pandas as pd
import shutil
import random

# === Paths ===
csv_path = "/Users/shivampatel/Research/Chest_X_Ray/chest_xray/chest_xray_predictions.csv"
original_data_dir = '/Users/shivampatel/Research/Chest_X_Ray/chest_xray 2' # path with train/test/val
output_normal_dir = "/Users/shivampatel/Research/Chest_X_Ray/5_from_NORMAL"
output_pneumonia_dir = "/Users/shivampatel/Research/Chest_X_Ray/5_from_PNEUMONIA"

# === Step 1: Read CSV and get already processed filenames ===
if os.path.exists(csv_path):
    processed = set(pd.read_csv(csv_path)["filename"].tolist())
else:
    processed = set()

print("Processed: ",len(processed))

df = pd.read_csv(csv_path)
print("Numbers: ", df[''].value_counts())

# === Step 2: Collect new candidate images ===
normal_candidates = []
pneumonia_candidates = []

for root, _, files in os.walk(original_data_dir):
    for file in files:
        #print(file)
        if not file.lower().endswith(".jpeg"):
            continue
        if file in processed:
            continue

        full_path = os.path.join(root, file)
        if "NORMAL" in root.upper():
            normal_candidates.append(full_path)
        elif "PNEUMONIA" in root.upper():
            pneumonia_candidates.append(full_path)

# Shuffle for randomness
random.shuffle(normal_candidates)
random.shuffle(pneumonia_candidates)

print("Normal: ", len(normal_candidates))
print("Pnuemonia: ", len(pneumonia_candidates))

# === Step 3: Select 5 from each and copy ===
os.makedirs(output_normal_dir, exist_ok=True)
os.makedirs(output_pneumonia_dir, exist_ok=True)

selected_normal = normal_candidates[:10]
selected_pneumonia = pneumonia_candidates[:10]

for path in selected_normal:
    shutil.copy(path, os.path.join(output_normal_dir, os.path.basename(path)))

for path in selected_pneumonia:
    shutil.copy(path, os.path.join(output_pneumonia_dir, os.path.basename(path)))

print(f"✅ Copied {len(selected_normal)} NORMAL images to {output_normal_dir}")
print(f"✅ Copied {len(selected_pneumonia)} PNEUMONIA images to {output_pneumonia_dir}")
