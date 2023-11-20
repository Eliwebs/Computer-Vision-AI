import os
from PIL import Image

# Path to the folder containing images
folder_path = 'path_to_your_folder'

# Initialize a list to store images
image_list = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other extensions if needed
        file_path = os.path.join(folder_path, filename)
        
        # Open the image file and append it to the list
        img = Image.open(file_path)
        image_list.append(img)
