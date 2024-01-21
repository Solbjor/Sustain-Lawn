import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

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

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Now, reshape data for model training
X_train_reshaped = X_train.reshape(-1, 300, 300, 3)
y_train_reshaped = y_train.reshape(-1, 300, 300, 1)
X_val_reshaped = X_val.reshape(-1, 300, 300, 3)
y_val_reshaped = y_val.reshape(-1, 300, 300, 1)

# Define the U-Net model
def unet_model(input_size=(300, 300, 3)):
    inputs = Input(input_size)
    
    # Down-sampling
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # More down-sampling layers can be added here

    # Up-sampling
    up1 = UpSampling2D(size=(2, 2))(pool1)
    conv2 = Conv2D(64, 2, activation='relu', padding='same')(up1)
    merged = concatenate([conv1, conv2], axis=3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(merged)

    # Output layer
    conv4 = Conv2D(1, 1, activation='sigmoid')(conv3)

    model = Model(inputs=inputs, outputs=conv4)

    return model

# Instantiate and compile the model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train_reshaped, validation_data=(X_val_reshaped, y_val_reshaped), epochs=10, batch_size=32)

model.save('ACroppingAI.h5')

print("Done!")
