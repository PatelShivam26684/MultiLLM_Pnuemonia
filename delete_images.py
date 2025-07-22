import os
import random
from pathlib import Path

# Define the base directory
base_dir = '/Users/shivampatel/Research/Chest_X_Ray/chest_xray'

# Define paths to the two target directories
normal_target = os.path.join(base_dir, 'NORMAL')
pneumonia_target = os.path.join(base_dir, 'PNEUMONIA')

directories = [normal_target, pneumonia_target]

KEEP_LIMIT = 500  # how many to keep in each folder
RNG_SEED = 42  # set to None for a different random pick each run

random.seed(RNG_SEED)

for directory in directories:
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) <= KEEP_LIMIT:
        print(f"{directory}: already has {len(image_files)} files or fewer, skipping.")
        continue

    keep_files = set(random.sample(image_files, KEEP_LIMIT))

    for f in image_files:
        if f not in keep_files:
            try:
                os.remove(os.path.join(directory, f))
            except Exception as e:
                print(f"Failed to delete {f} in {directory}: {e}")

    print(f"{directory}: kept {KEEP_LIMIT}, deleted {len(image_files) - KEEP_LIMIT}")

