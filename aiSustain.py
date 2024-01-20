import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model_path = '' 
model = tf.saved_model.load(model_path)

def detect_lawn(image):
    # Preprocess the image
    input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.uint8)
    detections = model(input_tensor)

    # Extract bounding box coordinates
    boxes = detections['detection_boxes'].numpy()[0]
    height, width, _ = image.shape

    # Draw bounding boxes on the image
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return image

# Read image data from CSV file
csv_file_path = 'C:\Users\s0lbj\OneDrive\Desktop\sustainApp\images_in_csv\via_project_20Jan2024_11h52m_csv.csv'
image_data = np.genfromtxt(csv_file_path, delimiter=',')

# Reshape image data if necessary
image_shape = (100, 100, 2)  # Replace with the dimensions
image_data = image_data.reshape(image_shape)

# Perform object detection
image_with_boxes = detect_lawn(image_data.astype(np.uint8))

# Display the result
cv2.imshow('Lawn Detection', image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
