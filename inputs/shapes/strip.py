from PIL import Image
import os

def remove_icc_profile(image_path):
    with Image.open(image_path) as img:
        # Copy image data to remove metadata
        img_data = list(img.getdata())
        img_without_icc = Image.new(img.mode, img.size)
        img_without_icc.putdata(img_data)
        img_without_icc.save(image_path)

# Process all PNG files in the current directory
for file_name in os.listdir('.'):
    if file_name.lower().endswith('.png'):
        remove_icc_profile(file_name)