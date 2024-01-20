import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

# Load image data from CSV file
csv_file_path = r'C:\Users\s0lbj\OneDrive\Desktop\sustainApp\images_in_csv\via_project_20Jan2024_11h52m_csv.csv'
image_data = np.genfromtxt(csv_file_path, delimiter=',') 

# Reshape image data if necessary
image_shape = (15173, 1) #placeholder nums to convert (cant multiply due to test file's columns being prime number)
image_data = image_data.reshape(image_shape)

# Display the image
plt.imshow(image_data, cmap='gray')
plt.title('Lawn Image')
plt.show()

