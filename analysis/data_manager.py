import os
import shutil

# Set the parent directory that contains subdirectories "1" to "24".
# Make sure to update this path as needed.
parent_directory = r"C:\Users\Beno\Documents\CEU\continuous_psychophysics\vsl_with_tracking\outputs\1"

# Define the items to keep in each numbered subdirectory
items_to_keep = {'11', '12', 'tracking.txt'}

# Iterate over directories "1" to "24"
for i in range(1, 25):
    subdir_path = os.path.join(parent_directory, str(i))
    if os.path.isdir(subdir_path):
        # List all items (files and directories) in the current subdirectory
        for item in os.listdir(subdir_path):
            if item not in items_to_keep:
                item_path = os.path.join(subdir_path, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        print(f"Deleted directory: {item_path}")
                    else:
                        os.remove(item_path)
                        print(f"Deleted file: {item_path}")
                except Exception as e:
                    print(f"Error deleting {item_path}: {e}")
    else:
        print(f"{subdir_path} is not a valid directory")