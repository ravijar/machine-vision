import numpy as np
import cv2
import sys
import os

def apply_filter(img, kernel):
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the input image with reflect mode
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')

    filtered_img = np.zeros_like(img, dtype=np.float32)

    # Perform convolution
    for i in range(img_height):
        for j in range(img_width):
            # Extract the region of interest (ROI) from the padded image
            roi = padded_img[i:i+kernel_height, j:j+kernel_width]
            
            # Perform element-wise multiplication and sum the result
            filtered_value = np.sum(roi * kernel)

            filtered_img[i, j] = filtered_value

    return filtered_img


def gaussian_filter(img, kernel_size=3, sigma=1.4):
    k = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))

    # Calculate the Gaussian kernel
    for x in range(-k, k + 1):
        for y in range(-k, k + 1):
            kernel[x + k, y + k] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)

    blurred_img = apply_filter(img, kernel)
    return np.clip(blurred_img, 0, 255).astype(np.uint8)


def gradient_intensity_direction(img):
    # sobel kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = apply_filter(img, Kx)
    Iy = apply_filter(img, Ky)

    # gradient magnitude
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255

    # gradient direction
    theta = np.arctan2(Iy, Ix)
    
    return G, theta


def non_max_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi
    # Ensure angles are between 0 and 180 degrees
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255
                
                # Angle 0 degrees (horizontal)
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = G[i, j+1]
                    r = G[i, j-1]
                # Angle 45 degrees (diagonal: bottom-left to top-right)
                elif (22.5 <= angle[i,j] < 67.5):
                    q = G[i+1, j-1]
                    r = G[i-1, j+1]
                # Angle 90 degrees (vertical)
                elif (67.5 <= angle[i,j] < 112.5):
                    q = G[i+1, j]
                    r = G[i-1, j]
                # Angle 135 degrees (diagonal: top-left to bottom-right)
                elif (112.5 <= angle[i,j] < 157.5):
                    q = G[i-1, j-1]
                    r = G[i+1, j+1]

                # Suppress non-maxima
                if (G[i,j] >= q) and (G[i,j] >= r):
                    Z[i,j] = G[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z


def double_threshold(img, lowThreshold, highThreshold):
    strong = 255
    weak = 50

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    # Set pixel values: strong edges, weak edges, and non-edges
    img[strong_i, strong_j] = strong
    img[weak_i, weak_j] = weak
    img[zeros_i, zeros_j] = 0

    return img, weak, strong


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                # Check if any of the 8 neighboring pixels are strong
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img


def scale_gradient_magnitudes(G, gamma=0.5):
    # Apply power law transformation to scale the gradient magnitudes
    G_scaled = G ** gamma
    
    # Normalize the scaled gradients to fit within the range [0, 255]
    G_scaled = (G_scaled / G_scaled.max()) * 255
    
    return G_scaled.astype(np.uint8)


def canny_edge_detection(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 1: Gaussian Filter
    blurred_img = gaussian_filter(img, kernel_size=5, sigma=1.4)
    
    # Step 2: Gradient calculation (Sobel Filters)
    G, theta = gradient_intensity_direction(blurred_img)
    
    # Step 3: Non-Maximum Suppression
    non_max_img = non_max_suppression(scale_gradient_magnitudes(G), theta) # gradient scaling is done to emphasize weak edges
    
    # Step 4: Double Thresholding
    threshold_img, weak, strong = double_threshold(non_max_img, lowThreshold=50, highThreshold=150)
    
    # Step 5: Hysteresis
    edge_img = hysteresis(threshold_img, weak, strong)
    
    # Saving the edge-detected image
    output_filename = os.path.splitext(image_path)[0] + '_edge.png'
    cv2.imwrite(output_filename, edge_img)

    print(f"Edge-detected image saved as: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Enter the image path!")
        print("Example: python CannyEdgeDetection.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    canny_edge_detection(image_path)
