import os
import json
import cv2
import tensorflow as tf

# Path to the JSON file containing annotations
json_file_path = r'C:\Users\s0lbj\OneDrive\Desktop\sustainApp/test.json'

# Directory containing images
image_dir = r'C:\Users\s0lbj\OneDrive\Desktop\homeImages'

# Check if the JSON file exists
if not os.path.exists(json_file_path):
    print("JSON file not found:", json_file_path)
else:
    print("JSON file found. Reading data...")

    # Load annotations from JSON file
    with open(json_file_path, 'r') as json_file:
        annotations = json.load(json_file)

    print("JSON data loaded successfully.")

# Prepare your data structure for TensorFlow
data_for_tf = []

for key, item in annotations.items():
    filename = item['filename']
    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path)

    # Check if the image is loaded properly
    if image is not None:
        # Assuming 'regions' is a list of dictionaries containing 'shape_attributes'
        for region in item['regions']:
            points_x = region['shape_attributes']['all_points_x']
            points_y = region['shape_attributes']['all_points_y']

            for i in range(len(points_x)):
                start_point = (points_x[i], points_y[i])
                end_point = (points_x[(i + 1) % len(points_x)], points_y[(i + 1) % len(points_y)])
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow('Annotated Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Failed to load image: {image_path}")