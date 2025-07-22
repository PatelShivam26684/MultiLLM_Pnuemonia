import os
import shutil

# Define the base directory
base_dir = '/Users/shivampatel/Research/Chest_X_Ray/chest_xray'

# Create two new directories: NORMAL and PNEUMONIA in chest_xray
normal_target = os.path.join(base_dir, 'NORMAL')
pneumonia_target = os.path.join(base_dir, 'PNEUMONIA')

os.makedirs(normal_target, exist_ok=True)
os.makedirs(pneumonia_target, exist_ok=True)

# Subdirectories to loop through
sub_dirs = ['train', 'test', 'val']

for sub_dir in sub_dirs:
    for label in ['NORMAL', 'PNEUMONIA']:
        source_dir = os.path.join(base_dir, sub_dir, label)
        target_dir = normal_target if label == 'NORMAL' else pneumonia_target

        if os.path.exists(source_dir):
            for filename in os.listdir(source_dir):
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(target_dir, filename)
                # Avoid overwriting files with the same name
                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
                else:
                    print(f"File already exists and will not be moved: {dst_path}")
