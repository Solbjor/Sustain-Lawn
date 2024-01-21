import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths
json_file_path1 = r'C:\Users\codyk\OneDrive\Desktop\Sustain-Lawn\train\newdata.json'
json_file_path2 = r'C:\Users\codyk\OneDrive\Desktop\Sustain-Lawn\train\traindata.json'
image_dir = r'C:\Users\codyk\OneDrive\Desktop\homeImages'

# Load annotations
with open(json_file_path1, 'r') as file:
    data1 = json.load(file)
    annotations1 = data1 if '_via_img_metadata' not in data1 else data1['_via_img_metadata']

with open(json_file_path2, 'r') as file:
    annotations2 = json.load(file)['_via_img_metadata']

# Combine annotations
annotations = {**annotations1, **annotations2} 

# Prepare dataset
images = []
masks = []

desired_size = (300, 300)  # Define the desired size (width, height)

for key, value in annotations.items():
    filename = value['filename']
    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is not None:
        # Resize image to the desired size
        resized_image = cv2.resize(image, desired_size)
        images.append(resized_image)

        # Create a blank mask with the desired size
        mask = np.zeros(desired_size, dtype=np.uint8)
        for region in value['regions']:
            # Scale the annotation points to the new size
            scaled_points_x = [int(x * desired_size[0] / image.shape[1]) for x in region['shape_attributes']['all_points_x']]
            scaled_points_y = [int(y * desired_size[1] / image.shape[0]) for y in region['shape_attributes']['all_points_y']]
            points = np.array(list(zip(scaled_points_x, scaled_points_y)), dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        masks.append(mask)
    else:
        print(f"Failed to load image: {image_path}")

# Convert lists to NumPy arrays
images = np.array(images)
masks = np.array(masks)

# Save the NumPy arrays as files
np.save('X_train.npy', images)
np.save('y_train.npy', masks)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Print a few images for visualization
num_images_to_display = 1  # You can adjust this based on how many images you want to display

for i in range(num_images_to_display):
    plt.figure(figsize=(8, 8))
   
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(X_train[i], cv2.COLOR_BGR2RGB)

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title(f'Training Image {i+1}')
   
    plt.subplot(1, 2, 2)
    plt.imshow(y_train[i], cmap='gray')
    plt.title(f'Training Mask {i+1}')

    plt.show()

# Now, you can proceed with the rest of your code
print("Arrays saved successfully!")
print("Total images in training set:", len(X_train))
print("Total masks in training set:", len(y_train))
