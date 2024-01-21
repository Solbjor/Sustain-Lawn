import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\codyk\OneDrive\Desktop\Sustain-Lawn\TeamsAI.h5')

# Function to preprocess the image
def preprocess_image(image_path, desired_size=(260, 260)):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    image = cv2.resize(image, desired_size)
    image = image / 255.0  # Normalize
    return image

# Function to postprocess the mask and draw an outline
def postprocess_and_draw_outline(image, mask):
    # Convert mask to binary
    _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(np.uint8(binary_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw an outline around the detected regions
    outline_color = (0, 255, 0)  # Green color for outline
    outline_thickness = 1
    for contour in contours:
        cv2.drawContours(image, [contour], 0, outline_color, outline_thickness)
    
    return image

# Path to the new image
new_image_path = r'C:\Users\codyk\OneDrive\Desktop\well-tended_lawns_part_of_american_dream.jpg'

# Preprocess the image
new_image = preprocess_image(new_image_path)

if new_image is not None:
    # Predict the mask
    predicted_mask = model.predict(np.array([new_image]))[0]

    # Postprocess the prediction and draw an outline
    final_image = postprocess_and_draw_outline(new_image, predicted_mask[:,:,0])

    # Display the image
    plt.imshow(final_image)
    plt.show()
else:
    print("Image processing failed.")
