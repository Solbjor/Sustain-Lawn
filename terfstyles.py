import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\codyk\OneDrive\Desktop\Sustain-Lawn\TeamsAI.h5')

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


def postprocess_and_fill_lawn(image, mask):
    _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    lawn_mask = np.zeros_like(image)
    contours, _ = cv2.findContours(np.uint8(binary_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill_color = (255, 255, 255)
    for contour in contours:
        cv2.fillPoly(lawn_mask, [contour], fill_color)
    return lawn_mask


def overlay_images_with_mask(background, overlay, lawn_mask, alpha=0.7):
    # Ensure both images are of the same data type
    background = background.astype(np.uint8)
    overlay = overlay.astype(np.uint8)

    # Blend the images
    overlay_blend = cv2.addWeighted(background, 1 - alpha, overlay, alpha, 0)

    # Apply the lawn mask
    return np.where(lawn_mask == 1, overlay_blend, background)

new_image_path = r'C:\Users\codyk\OneDrive\Desktop\well-tended_lawns_part_of_american_dream.jpg'
overlay_image_path = r'C:\Users\codyk\OneDrive\Desktop\gravel6.jpg'
new_image = preprocess_image(new_image_path)

if new_image is not None:
    # Predict the mask and process it
    predicted_mask = model.predict(np.array([new_image]))[0]
    lawn_mask = postprocess_and_fill_lawn(new_image * 255, predicted_mask[:,:,0])  # Multiplying by 255

    # Ensure lawn mask is binary (0 or 1)
    lawn_mask = (lawn_mask > 0).astype(np.uint8)

    # Load and resize the overlay image
    overlay_image = cv2.imread(overlay_image_path)
    overlay_image = cv2.resize(overlay_image, (lawn_mask.shape[1], lawn_mask.shape[0]))

    # Generate the final image
    final_image = overlay_images_with_mask(new_image * 255, overlay_image, lawn_mask)

    plt.imshow(final_image)
    plt.show()
else:
    print("Image processing failed.")