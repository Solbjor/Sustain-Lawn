import cv2
from aiSustain import image_with_boxes, boxes

def superimpose_turf(image, turf_image_path, bounding_box):
    turf_image = cv2.imread(turf_image_path)

    # Extract the bounded region
    ymin, xmin, ymax, xmax = bounding_box
    region = image[ymin:ymax, xmin:xmax]

    # Resize turf image to match the bounded region
    turf_image = cv2.resize(turf_image, (xmax - xmin, ymax - ymin))

    # Overlay the turf image on the bounded region
    result = image.copy()
    result[ymin:ymax, xmin:xmax] = cv2.addWeighted(region, 0.7, turf_image, 0.3, 0)

    return result

# Example usage
turf_image_path = ''
image_with_turf = superimpose_turf(image_with_boxes, turf_image_path, boxes[0])

# Display the result
cv2.imshow('Image with Turf Style', image_with_turf)
cv2.waitKey(0)
cv2.destroyAllWindows()
