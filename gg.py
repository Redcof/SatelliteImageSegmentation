import numpy as np
import cv2
from scipy import ndimage


# Step 3.2: Image Pre-Processing
def image_preprocessing(image):
    # Implement masking of non-agricultural areas
    # Create an RGB composite using bands 4, 3, and 2
    # composite = image[:, :, [3, 2, 1]]
    # Apply bilateral filtering for noise reduction and edge preservation
    # Implement bilateral filtering for noise reduction and smoothing
    bilateral_filtered = cv2.bilateralFilter(image, d=15, sigmaColor=75, sigmaSpace=75)
    # bilateral_filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    # Transform the image to the YUV color space
    yuv_image = cv2.cvtColor(bilateral_filtered, cv2.COLOR_RGB2YUV)
    
    # Apply sigmoid transform to increase contrast
    def sigmoid(x, alpha, x0):
        return 1 / (1 + np.exp(-alpha * (x - x0)))
    
    alpha = 0.5  # Adjust the slope parameter as needed
    x0 = 128  # Adjust the centering parameter as needed
    yuv_image[:, :, 0] = sigmoid(yuv_image[:, :, 0], alpha, x0)
    
    # Convert YUV data back to RGB color space
    preprocessed_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
    return preprocessed_image


# Step 3.3: Edge Detection and Enhancement
def meijering_filter(image, sigma=1):
    # Calculate the second-order derivatives using Gaussian derivatives
    Ix = ndimage.gaussian_filter(image, sigma, order=(1, 0))
    Iy = ndimage.gaussian_filter(image, sigma, order=(0, 1))
    
    # Calculate the elements of the Hessian matrix
    Ixx = ndimage.gaussian_filter(Ix ** 2, sigma)
    Iyy = ndimage.gaussian_filter(Iy ** 2, sigma)
    Ixy = ndimage.gaussian_filter(Ix * Iy, sigma)
    
    # Compute the vesselness measure using Meijering's formula
    R = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            Hessian = np.array([[Ixx[i, j], Ixy[i, j]], [Ixy[i, j], Iyy[i, j]]])
            eigenvalues = np.linalg.eigvals(Hessian)
            eigenvalues.sort()  # Sort eigenvalues in ascending order
            lambda1, lambda2 = eigenvalues
            R[i, j] = abs(lambda1) * np.exp(-(lambda1 ** 2) / (2 * sigma ** 2))
    # Normalize the output to the range [0, 255]
    R = ((R - np.min(R)) / (np.max(R) - np.min(R)) * 255).astype(np.uint8)
    return R


def edge_detection_and_enhancement(preprocessed_image):
    # Convolve with Sobel operator for gradient calculation
    gray_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2GRAY)
    # Apply Sobel operator for gradient calculation
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    
    # Apply the Meijering filter to enhance field boundaries
    filtered_img = meijering_filter(gradient_magnitude)
    
    return filtered_img, gradient_direction


def graph_grwoing(image):
    # Define the size of the image tiles (50x50 pixels)
    tile_size = (50, 50)
    
    # Create a mask for the segmented region
    mask = np.zeros_like(image)
    
    # Define the threshold for region growing
    threshold = 30
    
    # Define the step size for moving the seed points
    step_x = tile_size[0]
    step_y = tile_size[1]
    
    # Iterate over the image in tiles
    for y in range(0, image.shape[0], step_y):
        for x in range(0, image.shape[1], step_x):
            # Seed point selection for each tile (center of the tile)
            seed_x = x + tile_size[0] // 2
            seed_y = y + tile_size[1] // 2
            
            # Initialize the stack for graph-based growing
            stack = [(seed_x, seed_y)]
            
            while stack:
                x, y = stack.pop()
                
                # Check if the pixel is within the image boundaries
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    # Check if the pixel is not already part of the segmented region
                    if mask[y, x].all() == 0:
                        # Check the intensity difference between the seed pixel and the current pixel
                        diff = np.abs(image[y, x].astype(int) - image[seed_y, seed_x].astype(int))
                        
                        # If the difference is below the threshold, add the pixel to the region
                        if np.all(diff < threshold):
                            mask[y, x] = [255, 255, 255]  # Mark the pixel as part of the region
                            stack.append((x + 1, y))
                            stack.append((x - 1, y))
                            stack.append((x, y + 1))
                            stack.append((x, y - 1))
    
    # Find contours in the mask
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the contours on the original image
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 1)  # Green contours
    return output, contours


# Step 3.4: Graph-Based Growing Contours
def graph_based_growing_contours(enhanced_image):
    # Implement seed point selection
    # Create a local graph around each seed point
    # Define movement and termination criteria
    # Extract contours using the local graph
    _, extracted_contours = graph_grwoing(enhanced_image)
    return extracted_contours


# Step 3.5: Polygon Creation and Post-Processing
import cv2
import numpy as np
import networkx as nx
from scipy.spatial import distance
from skimage.segmentation import flood


# Step 1: Convert Extracted Contours to Binary Image
def contours_to_binary_image(contours, img_shape):
    binary_image = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(binary_image, contours, -1, 1, thickness=cv2.FILLED)
    return binary_image


# Step 2: Flood Fill Algorithm
def apply_flood_fill(binary_image, seed_point):
    labeled_image = flood(binary_image, seed_point, connectivity=1)
    return labeled_image


# Step 3: Extract Nodes Within Field Boundary Pixels
def extract_nodes_within_boundary(labeled_image, contour, distance_threshold=10):
    nodes = []
    for point in contour:
        x, y = point[0]
        if labeled_image[y, x] > 0:
            nodes.append((x, y))
    return nodes


# Step 4: Create a Local Boundary Graph
def create_local_boundary_graph(nodes):
    local_boundary_graph = nx.Graph()
    for node in nodes:
        local_boundary_graph.add_node(node)
        for neighbor in nodes:
            if node != neighbor and distance.euclidean(node, neighbor) <= 1.5:
                local_boundary_graph.add_edge(node, neighbor)
    return local_boundary_graph


# Step 5: Find the Longest Cycle
def find_longest_cycle(local_boundary_graph):
    cycles = nx.cycle_basis(local_boundary_graph)
    longest_cycle = max(cycles, key=len)
    return longest_cycle


# Step 6: Refine the Field Polygon (Optional)
# You can implement this step as needed for polygon refinement.

# Step 7: Repeat for All Fields
def extract_field_polygons(contours, img):
    field_polygons = []
    for contour in contours:
        binary_image = contours_to_binary_image([contour], img.shape)
        seed_point = tuple(contour[0][0])
        labeled_image = apply_flood_fill(binary_image, seed_point)
        nodes = extract_nodes_within_boundary(labeled_image, contour)
        local_boundary_graph = create_local_boundary_graph(nodes)
        longest_cycle = find_longest_cycle(local_boundary_graph)
        field_polygons.append(longest_cycle)
    return field_polygons


def create_and_post_process_polygons(extracted_contours, img):
    # Create a binary contour image from extracted contours
    # Apply flood fill algorithm to segment fields
    # Extract nodes within a certain distance from boundaries
    # Create a local graph for each field segment
    # Find the longest cycle in the local graph to form polygons
    field_polygons = extract_field_polygons(extracted_contours, img)
    return field_polygons


# Step 3.6: Selecting Optimal Parameters (Optional)
def select_optimal_parameters(optimal_parameters):
    # You can implement a parameter optimization algorithm here
    return optimal_parameters


if __name__ == '__main__':
    # Load your remote sensing image
    input_image = cv2.imread('images/field-pic-1.png')
    
    # Main Workflow
    preprocessed_image = image_preprocessing(input_image)
    enhanced_image, _ = edge_detection_and_enhancement(preprocessed_image)
    enhanced_rgbimage = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    extracted_contours = graph_based_growing_contours(enhanced_rgbimage)
    field_polygons = create_and_post_process_polygons(extracted_contours, enhanced_image)
    # optimal_parameters = select_optimal_parameters()
    
    # Print or save your field polygons and optimal parameters
    print("Field Polygons:", field_polygons)
    # print("Optimal Parameters:", optimal_parameters)
