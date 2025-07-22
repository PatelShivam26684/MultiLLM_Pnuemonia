import os
from PIL import Image

# Create the directory if it doesn't exist
output_dir = "/Users/shivampatel/Research/Chest_X_Ray/blank_images_2"
os.makedirs(output_dir, exist_ok=True)

# Generate 10 blank white images
for i in range(10):
    img = Image.new('L', (1024, 1024), color=255)  # 'L' for grayscale, 255 = white
    filename = f"blank_white_{i+1:02}.jpeg"
    img.save(os.path.join(output_dir, filename))

print(f"✅ Created 10 white images in '{output_dir}'")