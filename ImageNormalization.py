import numpy as np
import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found. Please check the file path.")
    return image

def save_image(image, image_path):
    cv2.imwrite(image_path, image)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def rotate_image(image, angle):
    # Convert angle to radians
    theta = np.radians(angle)
    
    # Define the rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Get image dimensions
    height, width = image.shape[:2]

    # Compute the center of the image
    center_x, center_y = width / 2, height / 2

    # Prepare an empty array for the rotated image
    rotated_image = np.zeros_like(image)

    # Iterate over every pixel in the image
    for i in range(height):
        for j in range(width):
            # Calculate the coordinates relative to the center
            relative_coords = np.array([j - center_x, i - center_y, 1])

            # Apply the rotation matrix
            new_coords = rotation_matrix @ relative_coords

            # Translate back to image coordinates
            new_x, new_y = int(new_coords[0] + center_x), int(new_coords[1] + center_y)

            # Assign the pixel value if within bounds
            if 0 <= new_x < width and 0 <= new_y < height:
                rotated_image[i, j] = image[new_y, new_x]

    return rotated_image

image_path = "image.png"
index_no = "200522F"
image = load_image(image_path)

# Convert to grayscale
gray_image = convert_to_grayscale(image)
save_image(gray_image, f"{index_no}_org.png")

# Rotate the grayscale image by 45 degrees
rotated_image = rotate_image(gray_image, 45)
save_image(rotated_image, f"{index_no}_rotated.png")
