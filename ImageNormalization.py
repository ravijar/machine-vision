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

def downsample_image(image):
    height, width = image.shape[:2]
    new_height, new_width = height // 2, width // 2

    # Reduce the image size by a factor of 2
    downsampled_image = image[:new_height*2:2, :new_width*2:2] // 4
    downsampled_image += image[1:new_height*2:2, :new_width*2:2] // 4
    downsampled_image += image[:new_height*2:2, 1:new_width*2:2] // 4
    downsampled_image += image[1:new_height*2:2, 1:new_width*2:2] // 4

    return downsampled_image

def generate_image_pyramid(image, levels=5):
    pyramid_images = [image]
    for i in range(1, levels):
        image = downsample_image(image)
        pyramid_images.append(image)
    return pyramid_images

def magnify_image(image, factor):
    height, width = image.shape[:2]
    new_height = int(height * factor)
    new_width = int(width * factor)
    
    # Resize the image using bilinear interpolation
    magnified_image = np.zeros((new_height, new_width), dtype=np.float32)
    
    for i in range(new_height):
        for j in range(new_width):
            # Map coordinates to original image
            x = j / factor
            y = i / factor

            # Get the coordinates of the four surrounding pixels
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1, width - 1)
            y2 = min(y1 + 1, height - 1)

            # Calculate the fractional parts
            fx = x - x1
            fy = y - y1

            # Perform bilinear interpolation
            top_left = float(image[y1, x1])
            top_right = float(image[y1, x2])
            bottom_left = float(image[y2, x1])
            bottom_right = float(image[y2, x2])

            top = top_left + (top_right - top_left) * fx
            bottom = bottom_left + (bottom_right - bottom_left) * fx

            magnified_image[i, j] = top + (bottom - top) * fy

    # Clip values to range [0, 255] and convert back to uint8
    magnified_image = np.clip(magnified_image, 0, 255).astype(np.uint8)

    return magnified_image

def generate_difference_image(image1, image2):
    # Ensure that the two images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same dimensions.")
    
    # Calculate the absolute difference between the two images
    difference_image = np.abs(image1.astype(np.int16) - image2.astype(np.int16))

    # Clip the values to the valid range [0, 255] and convert to uint8
    difference_image = np.clip(difference_image, 0, 255).astype(np.uint8)

    return difference_image


image_path = "image.png"
index_no = "200522F"
image = load_image(image_path)

# Convert to grayscale
gray_image = convert_to_grayscale(image)
save_image(gray_image, f"{index_no}_org.png")

# Rotate the grayscale image by 45 degrees
rotated_image = rotate_image(gray_image, 45)
save_image(rotated_image, f"{index_no}_rotated.png")

# Generate image pyramid with 5 levels
pyramid_images = generate_image_pyramid(gray_image)
for i, level_image in enumerate(pyramid_images):
    save_image(level_image, f"{index_no}_pyramid_{i+1}.png")

# Magnify the third level of the pyramid (25% scale) by a factor of 4
magnified_image = magnify_image(pyramid_images[2], 4)
save_image(magnified_image, f"{index_no}_mag.png")

# Genarate difference image of the original image and the magnified image
difference_image = generate_difference_image(gray_image, magnified_image)
save_image(difference_image, f"{index_no}_diff.png")