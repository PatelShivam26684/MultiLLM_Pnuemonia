import os

def collect_image_names(root_dir):
    image_names = set()
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Adjust if you want only .jpeg
                image_names.add(file)
    return image_names

# Paths to your directories
dir1 = '/Users/shivampatel/Research/Chest_X_Ray/chest_xray'
dir2 = '/Users/shivampatel/Research/Chest_X_Ray/chest_xray 2'

# Collect image names
images_dir1 = collect_image_names(dir1)
images_dir2 = collect_image_names(dir2)
print(len(images_dir1))
print(len(images_dir2))
# Compare sets
if images_dir1 == images_dir2:
    print("Both directories contain the same images by name.")
else:
    print("The directories contain different images.")
    only_in_dir1 = images_dir1 - images_dir2
    only_in_dir2 = images_dir2 - images_dir1

    if only_in_dir1:
        print(f"Images only in {dir1}:")
        for img in sorted(only_in_dir1):
            print(f"  {img}")
    if only_in_dir2:
        print(f"Images only in {dir2}:")
        for img in sorted(only_in_dir2):
            print(f"  {img}")
