import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

from skimage.feature import corner_harris, peak_local_max
from scipy.spatial import KDTree

# ##############################################################################
# ############### CONFIG STUFF #################################################

# Number of correspondence points between images [Part A]
nppp = 8  

# Configuration Modes
FAST_MODE_CONFIG = {
    'num_corners': 200,            # Fewer corners for quick processing
    'anms_points': 100,            # Fewer points retained
    'harris_threshold': 0.05,      # Higher threshold for fewer detections
    'edge_discard': 15,            # Maintain edge discard for quality
    'descriptor_window_size': 10,  # Smaller window size
    'descriptor_size': 4,          # Small patch size
    'feature_ratio_thresh': 0.9,   # Slightly relaxed ratio threshold
    'ransac_max_iters': 2000,      # Fewer iterations for faster computation
    'ransac_inlier_thresh': 1.5,   # Relaxed inlier threshold for more flexibility
}

BALANCED_MODE_CONFIG = {
    'num_corners': 500,            # Moderate number of corners
    'anms_points': 200,            # Balanced points retained
    'harris_threshold': 0.01,      # Moderate threshold
    'edge_discard': 10,            # Maintain edge discard
    'descriptor_window_size': 20,  # Moderate window size
    'descriptor_size': 6,          # Medium patch size
    'feature_ratio_thresh': 0.85,  # Balanced ratio threshold
    'ransac_max_iters': 4000,      # Sufficient iterations for quality
    'ransac_inlier_thresh': 1.0,   # Reasonable inlier threshold
}

HIGH_QUALITY_MODE_CONFIG = {
    'num_corners': 2000,           # More corners for detailed features
    'anms_points': 300,            # More points retained for accuracy
    'harris_threshold': 0.005,     # Lower threshold for maximum detection
    'edge_discard': 15,            # Fewer edge corners discarded
    'descriptor_window_size': 80,  # Larger window for context
    'descriptor_size': 5,          # Larger patch size for detail
    'feature_ratio_thresh': 0.55,   # Stricter ratio threshold for better matching
    'ransac_max_iters': 6000,      # More iterations for robust homography
    'ransac_inlier_thresh': 0.5,   # Tighter inlier threshold for precision
}

def select_part_b_config():
    print("\nSelect the mode for Part B configuration:")
    print("1. Fast Mode")
    print("2. Balanced Mode")
    print("3. High-Quality Mode")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == '1':
        return FAST_MODE_CONFIG
    elif choice == '2':
        return BALANCED_MODE_CONFIG
    elif choice == '3':
        return HIGH_QUALITY_MODE_CONFIG
    else:
        print("Invalid choice. Defaulting to Balanced Mode.")
        return BALANCED_MODE_CONFIG

PART_B_CONFIG = None

# ##############################################################################
# ############### HELPER FUNCTIONS #############################################

def dirs(num_images):
    # Setting up directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, 'data')
    output_dir = os.path.join(script_dir, 'output')
    points_file = os.path.join(script_dir, 'points.json')

    os.makedirs(output_dir, exist_ok=True)

    # Load all images dynamically
    image_paths = [os.path.join(images_dir, f'image{i+1}.jpg') for i in range(num_images)]

    return script_dir, images_dir, output_dir, points_file, image_paths

def load_imgs(image_paths):
    """
    Loads images and converts them to RGB.
    """
    print("\nLoading images...")
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    print("Images loaded and converted to RGB.")
    return images

def display_imgs(images, titles, delay=0):
    print("Displaying images...")
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        #plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    if delay > 0:
        plt.show(block=False)
        plt.pause(delay)
        plt.close()
    else:
        plt.show()
    print("Images displayed.")

def correspondence(images, num_points):
    """
    Collects correspondence points between consecutive images.
    """
    all_pts = []
    for i in range(len(images) - 1):
        pts1, pts2 = get_correspondence(images[i], images[i + 1], num_points, f"Image {i+1} and Image {i+2}")
        all_pts.append((pts1, pts2))
    return all_pts

def get_correspondence(image1, image2, num_points, description):
    """
    Helper function to get correspondence points between two images.
    """
    print(f"\nGetting {num_points} corresponding points for {description}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image1)
    ax1.set_title('Image 1')
    ax1.axis('off')
    ax2.imshow(image2)
    ax2.set_title('Image 2')
    ax2.axis('off')

    print(f"Click on a point in Image 1, then the corresponding point in Image 2. Repeat for {num_points} points.")

    class CorrSelect:
        def __init__(self, ax1, ax2, num_points):
            self.ax1 = ax1
            self.ax2 = ax2
            self.num_points = num_points
            self.current_image = 'Image1'
            self.pts1 = []
            self.pts2 = []
            self.cid = fig.canvas.mpl_connect('button_press_event', self.onclick)

        def onclick(self, event):
            if event.inaxes == self.ax1 and self.current_image == 'Image1':
                x, y = event.xdata, event.ydata
                self.pts1.append([x, y])
                self.ax1.scatter(x, y, c='r', marker='o')
                print(f"Image1 point {len(self.pts1)}: ({x:.2f}, {y:.2f})")
                self.current_image = 'Image2'
                fig.canvas.draw()
            elif event.inaxes == self.ax2 and self.current_image == 'Image2':
                x, y = event.xdata, event.ydata
                self.pts2.append([x, y])
                self.ax2.scatter(x, y, c='b', marker='o')
                x1, y1 = self.pts1[-1]
                x2, y2 = self.pts2[-1]
                self.ax1.plot([x1, x2], [y1, y2], 'g--', linewidth=1)
                print(f"Image2 point {len(self.pts2)}: ({x:.2f}, {y:.2f})")
                self.current_image = 'Image1'
                fig.canvas.draw()
            else:
                print("Click on the correct image in the correct order.")

            if len(self.pts1) == self.num_points and len(self.pts2) == self.num_points:
                fig.canvas.mpl_disconnect(self.cid)
                plt.close()

    selector = CorrSelect(ax1, ax2, num_points)
    plt.show()

    if len(selector.pts1) != num_points or len(selector.pts2) != num_points:
        raise ValueError("Not enough points selected.")

    pts1 = np.array(selector.pts1, dtype=np.float32)
    pts2 = np.array(selector.pts2, dtype=np.float32)

    print(f"Selected points:\npts1: {pts1}\npts2: {pts2}")
    return pts1, pts2

def visualize1(image1, image2, pts1, pts2, title, save_path):
    """
    Visualizes correspondences between two images.
    """
    print(f"\nVisualizing correspondences: {title}")

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.scatter(pts1[:, 0], pts1[:, 1], c='r', marker='o')
    #plt.title('Image 1 Points')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.scatter(pts2[:, 0], pts2[:, 1], c='b', marker='o')
    #plt.title('Image 2 Points')
    plt.axis('off')

    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        plt.plot([x1, x2], [y1, y2], 'g--', linewidth=1)

    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Correspondences visualization saved to {save_path}.")
    except Exception as e:
        print(f"Failed to save correspondences visualization to {save_path}: {e}")

    plt.close()
    print(f"Correspondences visualized and saved: {title}")

def computeH(im1_pts, im2_pts):
    """
    Computes homography matrix using Direct Linear Transformation (DLT) algorithm.
    """
    # DLT algorithm
    N = im1_pts.shape[0]
    if N < 4:
        raise ValueError("At least 4 points are required for computeH")
    A = []
    for i in range(N):
        x, y = im1_pts[i][0], im1_pts[i][1]
        x_prime, y_prime = im2_pts[i][0], im2_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # Last row of Vt -> smallest singular value
    H = h.reshape((3, 3))
    H /= H[2, 2]
    return H

def visualize2(image, H, pts_source, pts_target, title, save_path):
    """
    Visualizes transformation using homography H.
    """
    print(f"\nVisualizing transformation: {title}")
    print(f"H shape: {H.shape}")
    print(f"H contents:\n{H}")

    pts_source_homogeneous = np.hstack([pts_source, np.ones((pts_source.shape[0], 1))])  # Shape: (N, 3)
    transformed_pts_homogeneous = np.dot(H, pts_source_homogeneous.T).T  # shape: (N, 3)

    transformed_pts_homogeneous /= transformed_pts_homogeneous[:, [2]] + 1e-8  # Avoid division by zero
    transformed_pts = transformed_pts_homogeneous[:, :2]

    print(f"Transformed points:\n{transformed_pts}")

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.scatter(pts_target[:, 0], pts_target[:, 1], c='r', marker='o', label='Target Points')
    plt.scatter(transformed_pts[:, 0], transformed_pts[:, 1], c='b', marker='x', label='Transformed Source Points')
    #plt.title(title)
    plt.legend()
    plt.axis('off')

    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Transformation visualization saved to {save_path}.")
    except Exception as e:
        print(f"Failed to save transformation visualization to {save_path}: {e}")

    plt.close()
    print(f"Transformation visualized and saved: {title}")

def pano_size(images, homographies):
    """
    Computes the size of the panorama based on transformed image corners.
    """
    print("\nComputing panorama size...")
    all_corners = []

    for i, (image, H) in enumerate(zip(images, homographies)):
        h, w = image.shape[:2]
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ])  # shape: (4, 3)

        # Transform corners using homography
        transformed_corners = np.dot(H, corners.T).T

        # Normalize
        transformed_corners /= transformed_corners[:, [2]] + 1e-8  
        transformed_corners = transformed_corners[:, :2]
        print(f"Transformed corners for Image {i+1}:\n{transformed_corners}")
        all_corners.append(transformed_corners)

    all_corners = np.vstack(all_corners)
    print(f"All transformed corners:\n{all_corners}")
    x_min, y_min = np.floor(np.min(all_corners, axis=0)).astype(int)
    x_max, y_max = np.ceil(np.max(all_corners, axis=0)).astype(int)

    panorama_width = x_max - x_min
    panorama_height = y_max - y_min
    print(f"Panorama width: {panorama_width}, Panorama height: {panorama_height}")

    offset_x = -x_min
    offset_y = -y_min
    print(f"Computed offsets: x_offset = {offset_x}, y_offset = {offset_y}")

    return (panorama_height, panorama_width), (offset_x, offset_y)

def warpImage(image, H, panorama_size, offset):
    """
    Warps the image using the homography H and applies the given offset.
    """
    print("Warping image...")

    panorama_height, panorama_width = panorama_size
    offset_x, offset_y = offset

    warped_image = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    H_inv = np.linalg.inv(H)

    y_indices, x_indices = np.indices((panorama_height, panorama_width))
    x_indices_flat = x_indices.flatten()
    y_indices_flat = y_indices.flatten()

    x_panorama = x_indices_flat - offset_x
    y_panorama = y_indices_flat - offset_y

    # Homogeneous coordinates
    ones = np.ones_like(x_panorama)
    output_coords = np.stack((x_panorama, y_panorama, ones), axis=1)  # shape: (N, 3)

    # Transform with H_inv
    source_coords = np.dot(H_inv, output_coords.T).T  # shape: (N, 3)
    source_coords /= source_coords[:, [2]] + 1e-8  # Normalize
    x_src = source_coords[:, 0]
    y_src = source_coords[:, 1]

    x_src_int = np.floor(x_src).astype(np.int32)
    y_src_int = np.floor(y_src).astype(np.int32)

    # Mask of valid coordinates inside bounds
    valid_mask = (
        (x_src_int >= 0) & (x_src_int < image.shape[1]) &
        (y_src_int >= 0) & (y_src_int < image.shape[0])
    )

    x_dst = x_indices_flat[valid_mask]
    y_dst = y_indices_flat[valid_mask]
    x_src_valid = x_src_int[valid_mask]
    y_src_valid = y_src_int[valid_mask]

    # FINAL WARP
    warped_image[y_dst, x_dst] = image[y_src_valid, x_src_valid]
    print("Image warped successfully.")
    return warped_image

def feather_mask(image):
    """
    Creates a feather blending mask based on distance from image center.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.indices((h, w))
    center_x = w / 2
    center_y = h / 2

    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    mask = 1 - (distance / max_distance)

    mask = np.power(mask, 2)
    return mask

def blend_imgs(warped_images, panorama_size, offset):
    """
    Blends warped images into a single panorama using feather blending.
    """
    print("\nBlending images into mosaic...")
    panorama = np.zeros((panorama_size[0], panorama_size[1], 3), dtype=np.float32)
    weight_sum = np.zeros((panorama_size[0], panorama_size[1]), dtype=np.float32)

    for i, warped_image in enumerate(warped_images):
        print(f"Processing warped image {i + 1}...")
        mask = (cv2.cvtColor(warped_image, cv2.COLOR_RGB2GRAY) > 0).astype(np.float32)

        feat_mask = feather_mask(warped_image) * mask
        feat_mask_3ch = cv2.merge([feat_mask, feat_mask, feat_mask])

        # Adding to panorama
        panorama += warped_image.astype(np.float32) * feat_mask_3ch

        weight_sum += feat_mask

    weight_sum[weight_sum == 0] = 1.0

    # Normalizing
    panorama /= weight_sum[..., np.newaxis]
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)

    print("Mosaic finished successfully (with feather blending)!")
    return panorama

def visualize_inliers_outliers(image1, image2, pts1, pts2, inliers, title, save_path):
    """
    Visualizes inlier and outlier matches after RANSAC and displays them before saving.

    Args:
        image1 (np.ndarray): Source image.
        image2 (np.ndarray): Destination image.
        pts1 (np.ndarray): Inlier points from image1.
        pts2 (np.ndarray): Corresponding points from image2.
        inliers (np.ndarray): Boolean mask indicating inliers.
        title (str): Title for the visualization.
        save_path (str): Path to save the visualization image.
    """
    print(f"\nVisualizing inliers and outliers: {title}")
    plt.figure(figsize=(20, 10))
    # Combine images side by side
    combined_image = np.hstack((image1, image2))
    plt.imshow(combined_image)
    # Adjust points for combined image
    offset = image1.shape[1]
    pts2_adj = pts2.copy()
    pts2_adj[:, 0] += offset

    # Plot inliers in green
    inlier_pts1 = pts1[inliers]
    inlier_pts2 = pts2_adj[inliers]
    plt.scatter(inlier_pts1[:, 0], inlier_pts1[:, 1], c='g', marker='o', label='Inliers')
    plt.scatter(inlier_pts2[:, 0], inlier_pts2[:, 1], c='g', marker='o')

    # Draw lines between inliers
    for i in range(len(inlier_pts1)):
        plt.plot([inlier_pts1[i, 0], inlier_pts2[i, 0]], [inlier_pts1[i, 1], inlier_pts2[i, 1]], 'g-', linewidth=0.5)

    # Plot outliers in red
    outlier_pts1 = pts1[~inliers]
    outlier_pts2 = pts2_adj[~inliers]
    plt.scatter(outlier_pts1[:, 0], outlier_pts1[:, 1], c='r', marker='x', label='Outliers')
    plt.scatter(outlier_pts2[:, 0], outlier_pts2[:, 1], c='r', marker='x')

    # Optionally, draw lines for outliers
    for i in range(len(outlier_pts1)):
        plt.plot([outlier_pts1[i, 0], outlier_pts2[i, 0]], [outlier_pts1[i, 1], outlier_pts2[i, 1]], 'r--', linewidth=0.5)

    #plt.title(title)
    plt.legend()
    plt.axis('off')

    # Display the plot
    plt.show()

    # Save the plot after displaying
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Inliers and outliers visualization saved to {save_path}.")
    except Exception as e:
        print(f"Failed to save inliers/outliers visualization to {save_path}: {e}")

    plt.close()
    print(f"Inliers and outliers visualized and saved: {title}")

def visualize_homography_mapping(imageA, imageB, H, ptsA, title, save_path):
    """
    Visualizes how points from imageA are mapped to imageB using homography H,
    displays the plot, and then saves it.
    
    Args:
        imageA (np.ndarray): Source image.
        imageB (np.ndarray): Destination image.
        H (np.ndarray): Homography matrix.
        ptsA (np.ndarray): Points from imageA.
        title (str): Title for the plot.
        save_path (str): Path to save the visualization.
    """
    print(f"\nVisualizing homography mapping: {title}")
    plt.figure(figsize=(20, 10))
    combined_image = np.hstack((imageA, imageB))
    plt.imshow(combined_image)

    # Project points from imageA to imageB
    ptsA_homogeneous = np.hstack([ptsA, np.ones((ptsA.shape[0], 1))])
    projected_pts2 = (H @ ptsA_homogeneous.T).T
    projected_pts2 /= projected_pts2[:, [2]] + 1e-8  # Normalize
    projected_pts2 = projected_pts2[:, :2]

    offset = imageA.shape[1]

    # Plot original points
    plt.scatter(ptsA[:, 0], ptsA[:, 1], c='r', s=40, label='Original Points (Image A)')
    # Plot projected points
    plt.scatter(projected_pts2[:, 0] + offset, projected_pts2[:, 1], c='b', s=40, label='Projected Points (Image B)')

    # Draw lines between original and projected points
    for (x1, y1), (x2, y2) in zip(ptsA, projected_pts2):
        plt.plot([x1, x2 + offset], [y1, y2], 'k--', linewidth=1)

    #plt.title(title)
    plt.legend()
    plt.axis('off')

    # Display the plot
    plt.show()

    # Save the plot after displaying
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Homography mapping visualization saved to {save_path}.")
    except Exception as e:
        print(f"Failed to save homography mapping visualization to {save_path}: {e}")

    plt.close()
    print(f"Homography mapping visualized and saved: {title}")

def visualize_corners(image, keypoints, title, save_path):
    """
    Visualizes detected corners on the image and displays them before saving.
    
    Args:
        image (np.ndarray): Input RGB image.
        keypoints (np.ndarray): Array of (x, y) coordinates of detected corners.
        title (str): Title for the visualization.
        save_path (str): Path to save the visualization image.
    """
    print(f"\nVisualizing corners: {title}")
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=10)
    #plt.title(title)
    plt.axis('off')

    # Display the plot
    plt.show()

    # Save the plot after displaying
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Corners visualization saved to {save_path}.")
    except Exception as e:
        print(f"Failed to save corners visualization to {save_path}: {e}")

    plt.close()
    print(f"Corners visualized and saved: {title}")

def get_harris_corners(im, edge_discard=20, threshold_factor=0.0005, sigma=1, min_distance=0):
    """
    Detects Harris corners in a grayscale image.

    Args:
        im (np.ndarray): Grayscale image.
        edge_discard (int): Number of pixels to discard from the image edges.
        threshold_factor (float): Factor to adjust threshold for corner strength.
        sigma (float): Standard deviation for Gaussian smoothing in Harris detection.
        min_distance (int): Minimum distance between detected corners.

    Returns:
        np.ndarray: Coordinates of detected corners (row, column) format.
    """
    # Compute Harris corner response
    harris_response = corner_harris(im, sigma=sigma)
    threshold = threshold_factor * harris_response.max()
    coords = peak_local_max(harris_response, min_distance=min_distance, threshold_abs=threshold)

    # Discard points near edges
    mask = (
        (coords[:, 0] >= edge_discard) & (coords[:, 0] < im.shape[0] - edge_discard) &
        (coords[:, 1] >= edge_discard) & (coords[:, 1] < im.shape[1] - edge_discard)
    )
    coords = coords[mask]

    return coords, harris_response

def get_harris_corners_wrapper(image, config):
    """
    Wrapper function to integrate get_harris_corners with improved ANMS and selection.

    Args:
        image (np.ndarray): Input RGB image.
        config (dict): Configuration parameters.

    Returns:
        tuple: (keypoints_limited, corner_response)
            - keypoints_limited (np.ndarray): Array of (x, y) coordinates of detected corners.
            - corner_response (np.ndarray): Harris corner response image.
    """
    num_corners = config['num_corners']
    threshold = config['harris_threshold']
    edge_discard = config['edge_discard']

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Get Harris corners
    corner_coords, corner_response = get_harris_corners(
        gray, edge_discard=edge_discard, threshold_factor=threshold
    )

    # Check if we have any valid coordinates left
    if corner_coords.shape[0] == 0:
        print("No valid coordinates left after masking.")
        return np.array([]), corner_response

    # Extract strengths
    strengths = corner_response[corner_coords[:, 0], corner_coords[:, 1]]
    sorted_indices = np.argsort(-strengths)
    keypoints_sorted = corner_coords[sorted_indices]
    keypoints_limited = keypoints_sorted[:num_corners]

    # Convert from (y, x) to (x, y)
    keypoints_limited = keypoints_limited[:, ::-1]  # Now (x, y)

    print(f"Number of corners detected: {len(keypoints_limited)}")

    return keypoints_limited, corner_response

def anms(keypoints, corner_response, num_points, img_shape):
    """
    Performs Adaptive Non-Maximal Suppression on detected corners with boundary checks.

    Args:
        keypoints (np.ndarray): Array of (x, y) coordinates of detected corners.
        corner_response (np.ndarray): Harris corner response image.
        num_points (int): Number of keypoints to retain after ANMS.
        img_shape (tuple): Shape of the image (height, width).

    Returns:
        np.ndarray: Selected keypoints after ANMS.
    """
    print("Performing Adaptive Non-Maximal Suppression (ANMS)...")
    num_keypoints = keypoints.shape[0]
    print(f"Initial number of keypoints: {num_keypoints}")

    if num_keypoints == 0:
        print("No keypoints to process.")
        return np.array([])

    # Extract strengths
    strengths = corner_response[keypoints[:, 1], keypoints[:, 0]]
    sorted_indices = np.argsort(-strengths)
    keypoints = keypoints[sorted_indices]
    strengths = strengths[sorted_indices]

    # Initialize suppression radii to infinity
    radii = np.full(len(keypoints), np.inf)

    for i in range(len(keypoints)):
        for j in range(i):
            if strengths[j] > strengths[i]:
                dist = np.linalg.norm(keypoints[i] - keypoints[j])
                if dist < radii[i]:
                    radii[i] = dist

    # Select keypoints with the largest radii
    selected_indices = np.argsort(-radii)[:num_points]
    selected_keypoints = keypoints[selected_indices]

    # Additional boundary check
    valid_mask = (
        (selected_keypoints[:, 0] >= 0) & (selected_keypoints[:, 0] < img_shape[1]) &
        (selected_keypoints[:, 1] >= 0) & (selected_keypoints[:, 1] < img_shape[0])
    )
    selected_keypoints = selected_keypoints[valid_mask]

    print(f"Number of keypoints after ANMS and boundary checks: {len(selected_keypoints)}")
    return selected_keypoints

def visualize_anms_corners(image, keypoints, title, save_path):
    """
    Visualizes ANMS-selected corners on the image.
    
    Args:
        image (np.ndarray): Input RGB image.
        keypoints (np.ndarray): Array of (x, y) coordinates of ANMS-selected corners.
        title (str): Title for the visualization.
        save_path (str): Path to save the visualization image.
    """
    print(f"\nVisualizing ANMS corners: {title}")
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='b', s=10)
    #plt.title(title)
    plt.axis('off')
    
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"ANMS corners visualization saved to {save_path}.")
    except Exception as e:
        print(f"Failed to save ANMS corners visualization to {save_path}: {e}")
    
    plt.close()
    print(f"ANMS corners visualized and saved: {title}")

def extract_features(image, keypoints, config):
    """
    Extracts rotation-invariant feature descriptors based on MOPS with boundary checks.

    Args:
        image (np.ndarray): Input RGB image.
        keypoints (np.ndarray): Array of (x, y) coordinates of keypoints.
        config (dict): Configuration parameters.

    Returns:
        tuple: (valid_keypoints, descriptors)
            - valid_keypoints (np.ndarray): Keypoints with valid descriptors.
            - descriptors (np.ndarray): Array of normalized descriptors.
    """
    print("Extracting rotation-invariant feature descriptors...")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    descriptors = []
    valid_keypoints = []

    # Compute image gradients
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)

    half_window = config['descriptor_window_size'] // 2
    descriptor_size = config['descriptor_size']
    img_height, img_width = gray.shape[:2]

    for point in keypoints:
        x, y = int(point[0]), int(point[1])

        # Define window bounds
        if (y - half_window < 0 or y + half_window >= img_height or
            x - half_window < 0 or x + half_window >= img_width):
            continue  # Skip keypoints too close to the edges

        # Compute orientation
        window_Ix = Ix[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
        window_Iy = Iy[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
        orientation = np.arctan2(window_Iy, window_Ix)
        magnitude = np.sqrt(window_Ix**2 + window_Iy**2)

        # Weighted average orientation
        weighted_orientation = np.arctan2(
            np.sum(magnitude * np.sin(orientation)),
            np.sum(magnitude * np.cos(orientation))
        )
        angle = np.degrees(weighted_orientation)

        # Rotate the patch to align with the dominant orientation
        M = cv2.getRotationMatrix2D((half_window, half_window), angle, 1)
        patch = gray[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
        rotated_patch = cv2.warpAffine(patch, M, (2 * half_window + 1, 2 * half_window + 1))

        # Resize to descriptor size
        small_patch = cv2.resize(rotated_patch, (descriptor_size, descriptor_size), interpolation=cv2.INTER_AREA).astype(np.float32)

        # Bias/Gain normalization
        small_patch -= np.mean(small_patch)
        norm = np.linalg.norm(small_patch)
        if norm > 1e-5:
            small_patch /= norm

        descriptors.append(small_patch.flatten())
        valid_keypoints.append([x, y])  # Store as (x, y)

    descriptors = np.array(descriptors)
    valid_keypoints = np.array(valid_keypoints)
    print(f"Number of descriptors extracted: {len(descriptors)}")

    return valid_keypoints, descriptors

def visualize_descriptors(image, keypoints, descriptors, num_descriptors=5, save_path=None):
    """
    Visualizes a subset of feature descriptors as images.
    
    Args:
        image (np.ndarray): Input RGB image.
        keypoints (np.ndarray): Array of (x, y) coordinates of keypoints.
        descriptors (np.ndarray): Array of descriptors.
        num_descriptors (int): Number of descriptors to visualize.
        save_path (str): Path to save the visualization image.
    """
    print("Visualizing feature descriptors...")
    plt.figure(figsize=(num_descriptors * 2, 2))
    for i in range(min(num_descriptors, len(descriptors))):
        patch = descriptors[i].reshape((config['descriptor_size'], config['descriptor_size']))
        # Re-scale for visualization
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
        plt.subplot(1, num_descriptors, i + 1)
        plt.imshow(patch, cmap='gray')
        #plt.title(f"Desc {i+1}")
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Descriptors visualization saved to {save_path}.")
        except Exception as e:
            print(f"Failed to save descriptors visualization to {save_path}: {e}")
    plt.show()

def match_features_ratio_test(descriptors1, descriptors2, ratio_thresh=0.75):
    """
    Matches features using Lowe's ratio test.

    Args:
        descriptors1 (np.ndarray): Descriptors from image 1.
        descriptors2 (np.ndarray): Descriptors from image 2.
        ratio_thresh (float): Threshold for Lowe's ratio test.

    Returns:
        list: List of matched index pairs (index_in_desc1, index_in_desc2).
    """
    print("Matching features using Lowe's ratio test...")
    matches = []
    tree = KDTree(descriptors2)
    for i, desc1 in enumerate(descriptors1):
        distances, indices = tree.query(desc1, k=2)
        if len(distances) < 2:
            continue
        if distances[0] < ratio_thresh * distances[1]:
            matches.append((i, indices[0]))
        if (i + 1) % 500 == 0 or i == len(descriptors1) - 1:
            print(f"Processed {i + 1}/{len(descriptors1)} descriptors.")
    print(f"Number of matches after ratio test: {len(matches)}")
    return matches

def interactive_match_verification(image1, image2, keypoints1, keypoints2, matches, title, save_path):
    """
    Visualizes matches between two images and allows the user to interactively remove incorrect matches.
    
    Args:
        image1 (np.ndarray): Source image.
        image2 (np.ndarray): Destination image.
        keypoints1 (np.ndarray): Keypoints from image1.
        keypoints2 (np.ndarray): Keypoints from image2.
        matches (list): List of matched index pairs (index_in_desc1, index_in_desc2).
        title (str): Title for the visualization.
        save_path (str): Path to save the visualization image.
    
    Returns:
        list: Filtered matches after user removal.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    print(f"\nInteractive match verification: {title}")

    fig, ax = plt.subplots(figsize=(20,10))
    combined_image = np.hstack((image1, image2))
    ax.imshow(combined_image)
    ax.set_title(title)

    # Plot matches
    lines = []
    scatters = []
    for idx, (i, j) in enumerate(matches):
        x1, y1 = keypoints1[i]
        x2, y2 = keypoints2[j]
        line, = ax.plot([x1, x2 + image1.shape[1]], [y1, y2], 'g-', linewidth=0.5)
        scatter = ax.scatter([x1, x2 + image1.shape[1]], [y1, y2], c='r', s=10)
        lines.append(line)
        scatters.append(scatter)

    ax.axis('off')
    plt.show(block=False)

    # List to store matches to remove
    matches_to_remove = []

    def on_click(event):
        if event.inaxes != ax:
            return
        x_click, y_click = event.xdata, event.ydata
        # Find the closest match
        min_dist = float('inf')
        closest_match = None
        closest_idx = -1
        for idx, (line, scatter) in enumerate(zip(lines, scatters)):
            x1, y1 = keypoints1[matches[idx][0]]
            x2, y2 = keypoints2[matches[idx][1]]
            # Compute distance from click to line segment
            px, py = x_click, y_click
            norm = np.hypot(x2 - x1, y2 - y1)
            if norm == 0:
                continue
            u = ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / (norm**2)
            u = max(0, min(1, u))
            closest_x = x1 + u * (x2 - x1)
            closest_y = y1 + u * (y2 - y1)
            dist = np.hypot(px - closest_x, py - closest_y)
            if dist < min_dist:
                min_dist = dist
                closest_match = (matches[idx][0], matches[idx][1])
                closest_idx = idx
        # Define a threshold for selecting a match
        threshold = 10  # pixels
        if min_dist < threshold:
            print(f"Match {closest_idx+1} selected for removal: {closest_match}")
            matches_to_remove.append(closest_idx)
            # Remove the line and scatter
            lines[closest_idx].remove()
            scatters[closest_idx].remove()
            fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    print("Click on matches to remove them. Close the window when done.")
    plt.show()

    # Disconnect the event
    fig.canvas.mpl_disconnect(cid)

    # Remove matches in reverse order to avoid index issues
    for idx in sorted(matches_to_remove, reverse=True):
        del matches[idx]

    # Save the final visualization
    plt.figure(figsize=(20,10))
    combined_image = np.hstack((image1, image2))
    plt.imshow(combined_image)
    #plt.title(f"{title} - Final Matches")
    for (i, j) in matches:
        x1, y1 = keypoints1[i]
        x2, y2 = keypoints2[j]
        plt.plot([x1, x2 + image1.shape[1]], [y1, y2], 'g-', linewidth=0.5)
        plt.scatter([x1, x2 + image1.shape[1]], [y1, y2], c='r', s=10)
    plt.axis('off')
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Final matches visualization saved to {save_path}.")
    except Exception as e:
        print(f"Failed to save matches visualization to {save_path}: {e}")
    plt.close()

    print(f"Interactive match verification completed: {title}")
    print(f"Total matches removed: {len(matches_to_remove)}")

    return matches

def grid_verification(img, matches, keypoints1, keypoints2, grid_size=10, min_matches_per_grid=5):
    """
    Ensures matches are evenly distributed across the image by dividing the image into grids.

    Args:
        img (np.ndarray): Image to determine grid size.
        matches (list): List of matched index pairs.
        keypoints1 (np.ndarray): Keypoints from image1.
        keypoints2 (np.ndarray): Keypoints from image2.
        grid_size (int): Number of grids along each axis.
        min_matches_per_grid (int): Minimum matches required per grid.

    Returns:
        list: Verified matches that meet the spatial distribution criteria.
    """

    height, width = img.shape[:2]
    grid_h = height // grid_size
    grid_w = width // grid_size
    
    grid_matches = {}
    for i, j in matches:
        x, y = keypoints1[i]
        grid_x = int(x) // grid_w
        grid_y = int(y) // grid_h
        grid_idx = (grid_x, grid_y)
        grid_matches.setdefault(grid_idx, []).append((i, j))
    
    verified_matches = []
    for grid, grid_m in grid_matches.items():
        if len(grid_m) >= min_matches_per_grid:
            verified_matches.extend(grid_m)
    
    print(f"Number of matches after grid verification: {len(verified_matches)}")
    return verified_matches

def computeH_ransac(pts1, pts2, config):
    """
    Computes homography using RANSAC.

    Args:
        pts1 (np.ndarray): Points from image 1 (Nx2).
        pts2 (np.ndarray): Corresponding points from image 2 (Nx2).
        config (dict): Configuration parameters.

    Returns:
        tuple: Best homography matrix (3x3) and inlier mask (bool array).
    """
    max_iters = config['ransac_max_iters']
    inlier_thresh = config['ransac_inlier_thresh']
    print("Computing homography using RANSAC...")
    num_points = pts1.shape[0]
    best_inliers = np.zeros(num_points, dtype=bool)
    best_H = None

    for iteration in range(max_iters):
        indices = np.random.choice(num_points, 4, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        try:
            H = computeH(sample_pts1, sample_pts2)
        except np.linalg.LinAlgError:
            continue  # Skip degenerate configurations

        # Project all pts1 to image 2
        pts1_homogeneous = np.hstack([pts1, np.ones((num_points, 1))])
        projected_pts2 = (H @ pts1_homogeneous.T).T
        projected_pts2 /= projected_pts2[:, [2]] + 1e-8
        projected_pts2 = projected_pts2[:, :2]

        distances = np.linalg.norm(pts2 - projected_pts2, axis=1)
        inliers = distances < inlier_thresh

        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_H = H
            if np.sum(inliers) > 0.85 * num_points:
                break

    if best_H is None or np.sum(best_inliers) < 4:
        print("Failed to compute a valid homography.")
        return None, None

    # Recompute homography using all inliers
    best_H = computeH(pts1[best_inliers], pts2[best_inliers])

    return best_H, best_inliers

def visualize_matches(image1, image2, keypoints1, keypoints2, matches, title, save_path):
    """
    Visualizes matches between two images and displays them before saving.

    Args:
        image1 (np.ndarray): Source image.
        image2 (np.ndarray): Destination image.
        keypoints1 (np.ndarray): Keypoints from image1.
        keypoints2 (np.ndarray): Keypoints from image2.
        matches (list): List of matched index pairs.
        title (str): Title for the visualization.
        save_path (str): Path to save the visualization image.
    """
    print(f"\nVisualizing matches: {title}")
    combined_image = np.hstack((image1, image2))
    plt.figure(figsize=(20, 10))
    plt.imshow(combined_image)
    #plt.title(title)

    for match in matches:
        idx1, idx2 = match
        x1, y1 = keypoints1[idx1]
        x2, y2 = keypoints2[idx2]
        plt.plot([x1, x2 + image1.shape[1]], [y1, y2], 'g-', linewidth=0.5)
        plt.scatter([x1, x2 + image1.shape[1]], [y1, y2], c='r', s=10)

    plt.axis('off')

    # Display the plot
    plt.show()

    # Save the plot after displaying
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Matches visualization saved to {save_path}.")
    except Exception as e:
        print(f"Failed to save matches visualization to {save_path}: {e}")

    plt.close()
    print(f"Matches visualized and saved: {title}")

def visualize_descriptors_with_orientation(image, keypoints, descriptors, num_descriptors=5, save_path=None):
    """
    Visualizes a subset of feature descriptors with orientation.

    Args:
        image (np.ndarray): Input RGB image.
        keypoints (np.ndarray): Array of (x, y) coordinates of keypoints.
        descriptors (np.ndarray): Array of descriptors.
        num_descriptors (int): Number of descriptors to visualize.
        save_path (str): Path to save the visualization image.
    """
    print("Visualizing descriptors with orientation...")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i in range(min(num_descriptors, len(keypoints))):
        x, y = keypoints[i]
        plt.scatter(x, y, c='r')
        plt.annotate(str(i+1), (x, y), color='yellow')
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Descriptors with orientation visualization saved to {save_path}.")
        except Exception as e:
            print(f"Failed to save descriptors with orientation visualization to {save_path}: {e}")
    plt.show()

def compute_average_2nn(descriptors):
    """
    Computes the average 2-NN distance across all descriptors for outlier rejection.

    Args:
        descriptors (np.ndarray): Array of feature descriptors.

    Returns:
        float: Average 2-NN distance.
    """
    print("Computing average 2-NN distance for outlier rejection...")
    two_nn_distances = []
    
    for i, desc in enumerate(descriptors):
        distances = np.linalg.norm(descriptors - desc, axis=1)
        sorted_distances = np.sort(distances[distances > 0])  # Exclude self-match
        if len(sorted_distances) >= 2:
            two_nn_distances.append(sorted_distances[:2].mean())
    
    average_2nn = np.mean(two_nn_distances)
    print(f"Average 2-NN distance: {average_2nn}")
    return average_2nn

# ##############################################################################
# ############### MAIN FUNCTIONS ################################################

def partA(num_images):
    print("\n=== Program started - Part A ===")

    # Set up directories
    script_dir, images_dir, output_dir, points_file, image_paths = dirs(num_images)
    images = load_imgs(image_paths)

    display_imgs(images, [f"Image {i+1}" for i in range(num_images)])

    num_points = nppp 

    # Collect correspondence points between all images
    print("\nCollecting correspondence points.")
    correspondence_points = correspondence(images, num_points)

    # Save correspondence points
    points_data = {}
    for i, (pts1, pts2) in enumerate(correspondence_points):
        points_data[f'pts{i+1}'] = pts1.tolist()
        points_data[f'pts{i+2}'] = pts2.tolist()

    with open(points_file, 'w') as f:
        json.dump(points_data, f)
    print(f"Saved correspondence points to {points_file}.")

    # Verify point arrays have the same shape and visualize correspondences
    for i, (pts1, pts2) in enumerate(correspondence_points):
        if not (pts1.shape == pts2.shape):
            print("Point arrays do not have the same shape. Exiting Part A.")
            return

        # Visualize correspondences
        visualize1(
            images[i], images[i + 1], pts1, pts2,
            f"Correspondences between Image {i + 1} and Image {i + 2}",
            os.path.join(output_dir, f"correspondences_image{i + 1}_image{i + 2}.png")
        )

    # Compute homographies
    print("\nComputing homographies...")
    homographies = []
    for i, (pts1, pts2) in enumerate(correspondence_points):
        H = computeH(pts1, pts2)
        homographies.append(H)

    # Visualize transformations
    for i, H in enumerate(homographies):
        visualize2(
            images[i + 1], H, correspondence_points[i][0], correspondence_points[i][1],
            f"Point Transformation for Image {i +1}",
            os.path.join(output_dir, f"transformation_image{i +1}.png")
        )

    # Compute panorama size
    print("\nComputing panorama size...")
    # Assume the middle image is the reference
    reference_idx = num_images // 2
    homographies_ref = []
    for i in range(num_images):
        if i == reference_idx:
            homographies_ref.append(np.eye(3))
        elif i < reference_idx:
            H = np.eye(3)
            for j in range(i, reference_idx):
                H = homographies[j] @ H
            homographies_ref.append(H)
        else:
            H = np.eye(3)
            for j in range(reference_idx, i):
                H = homographies[j] @ H
            homographies_ref.append(H)

    panorama_size, offset = pano_size(images, homographies_ref)

    # Warp images onto panorama canvas
    print("\nWarping images onto panorama canvas...")
    warped_images = []
    for i, H in enumerate(homographies_ref):
        warped_image = warpImage(images[i], H, panorama_size, offset)
        warped_images.append(warped_image)

    # Display warped images
    display_imgs(warped_images, [f"Warped Image {i+1}" for i in range(num_images)])

    # Blend images into mosaic
    print("\nCreating mosaic with feather blending...")
    panorama = blend_imgs(warped_images, panorama_size, offset)

    # Save and display mosaic
    mosaic_path = os.path.join(output_dir, 'mosaic_partA.jpg')
    cv2.imwrite(mosaic_path, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
    print(f"Mosaic saved to {mosaic_path}.")
    display_imgs([panorama], ["Mosaic - Part A"])
    print("=== Program completed successfully - Part A ===\n")

def partB(configuration):
    """
    Part B: Automatic Feature Matching and Image Stitching
    """
    print("\n=== Starting Part B: Automatic Feature Matching and Image Stitching ===")

    # Set up directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, 'data')
    output_dir = os.path.join(script_dir, 'output')

    os.makedirs(output_dir, exist_ok=True)

    # Find all image files in the data directory
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    num_images = len(image_files)
    if num_images < 2:
        print("Need at least two images for stitching. Exiting Part B.")
        return

    image_paths = [os.path.join(images_dir, f) for f in image_files]
    images = load_imgs(image_paths)

    # Detect Harris Corners
    print("\n--- Step 1: Detecting Harris Corners ---")
    keypoints_all = []
    descriptors_all = []
    corner_responses_all = []

    for i, image in enumerate(images):
        print(f"\nProcessing Image {i+1}")
        keypoints, corner_response = get_harris_corners_wrapper(image, configuration)
        anms_points = anms(keypoints, corner_response, configuration['anms_points'], image.shape)
        visualize_anms_corners(image, anms_points, f"ANMS Corners in Image {i+1}", os.path.join(output_dir, f"anms_corners_image{i+1}.png"))
        valid_keypoints, descriptors = extract_features(image, anms_points, configuration)
        keypoints_all.append(valid_keypoints)
        descriptors_all.append(descriptors)
        corner_responses_all.append(corner_response)

    # Step 2: Feature Matching
    print("\n--- Step 2: Matching Features ---")
    matches_all = []
    for i in range(num_images -1):
        print(f"\nMatching Image {i+1} with Image {i+2}")
        matches = match_features_ratio_test(descriptors_all[i], descriptors_all[i +1], configuration['feature_ratio_thresh'])
        matches_all.append(matches)

    # Step 3: Grid-Based Verification to Remove Incorrect Pairings
    print("\n--- Step 3: Grid-Based Verification of Matches ---")
    verified_matches_all = []
    for i in range(num_images -1):
        print(f"\nVerifying matches between Image {i+1} and Image {i+2}")
        img = images[i]
        matches = matches_all[i]
        keypoints1 = keypoints_all[i]
        keypoints2 = keypoints_all[i +1]
        verified_matches = grid_verification(img, matches, keypoints1, keypoints2, grid_size=10, min_matches_per_grid=5)
        verified_matches_all.append(verified_matches)

    # Step 4: Interactive Match Verification
    print("\n--- Step 4: Interactive Match Verification ---")
    for i in range(num_images -1):
        print(f"\nVerifying matches between Image {i+1} and Image {i+2}")
        image1 = images[i]
        image2 = images[i +1]
        keypoints1 = keypoints_all[i]
        keypoints2 = keypoints_all[i +1]
        matches = verified_matches_all[i]
        title = f"Interactive Match Verification between Image {i+1} and Image {i+2}"
        save_path = os.path.join(output_dir, f"interactive_matches_image{i+1}_image{i+2}.png")
        matches = interactive_match_verification(image1, image2, keypoints1, keypoints2, matches, title, save_path)
        verified_matches_all[i] = matches

    # Step 5: Compute Homographies
    print("\n--- Step 5: Computing Homographies ---")
    homographies = []
    for i in range(num_images):
        if i == num_images //2:
            homographies.append(np.eye(3))
        elif i < num_images //2:
            H = np.eye(3)
            for j in range(i, num_images//2):
                if homographies[j] is not None:
                    H = homographies[j] @ H
                else:
                    print(f"Homography for Image {j+1} is None. Skipping.")
            homographies.append(H)
        else:
            H = np.eye(3)
            for j in range(num_images//2, i):
                if homographies[j] is not None:
                    H = homographies[j] @ H
                else:
                    print(f"Homography for Image {j+1} is None. Skipping.")
            homographies.append(H)

    # Compute homographies using RANSAC
    for i in range(num_images -1):
        print(f"\nComputing homography between Image {i+1} and Image {i+2}")
        keypoints1 = keypoints_all[i]
        keypoints2 = keypoints_all[i +1]
        matches = verified_matches_all[i]
        if len(matches) < 4:
            print(f"Not enough matches between Image {i+1} and Image {i+2} to compute homography.")
            homographies[i +1] = homographies[i]  # Assign previous homography
            continue
        pts1 = keypoints1[[m[0] for m in matches]]
        pts2 = keypoints2[[m[1] for m in matches]]
        H, inliers = computeH_ransac(pts1, pts2, configuration)
        if H is not None:
            homographies[i +1] = H
            # Optionally visualize inliers/outliers
            visualize_inliers_outliers(
                image1, image2, pts1, pts2, inliers,
                f"Inliers and Outliers between Image {i+1} and Image {i+2}",
                os.path.join(output_dir, f"inliers_outliers_image{i+1}_image{i+2}.png")
            )
        else:
            print(f"Failed to compute homography between Image {i+1} and Image {i+2}. Using identity.")
            homographies[i +1] = np.eye(3)

    # Step 6: Compute Panorama Size and Warp Images
    print("\n--- Step 6: Computing Panorama Size and Warping Images ---")
    panorama_size, offset = pano_size(images, homographies)

    warped_images = []
    for i, H in enumerate(homographies):
        print(f"\nWarping Image {i+1}...")
        warped_image = warpImage(images[i], H, panorama_size, offset)
        warped_images.append(warped_image)

    # Step 7: Blend Images to Create Panorama
    print("\n--- Step 7: Blending Images to Create Panorama ---")
    panorama = blend_imgs(warped_images, panorama_size, offset)

    # Save and Display the Final Panorama
    mosaic_path = os.path.join(output_dir, 'mosaic_partB.jpg')
    cv2.imwrite(mosaic_path, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
    print(f"\nFinal mosaic saved to {mosaic_path}.")
    display_imgs([panorama], ["Mosaic - Part B"])
    print("=== Program completed successfully - Part B ===\n")

if __name__ == '__main__':

    print("=== Image Stitching and Panorama Creation ===\n")
    num_images = int(input("Enter the number of images to process: ").strip())

    if num_images < 2:
        print("At least two images are required for stitching. Exiting.")
        exit()

    print("\nSelect which part to execute:")
    print("1. Part A (Manual Correspondence Points)")
    print("2. Part B (Automatic Feature Matching and Image Stitching)")
    print("3. Both Part A and Part B")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == '1':
        partA(num_images)
    elif choice == '2':
        part_b_config = select_part_b_config()
        partB(part_b_config)
    elif choice == '3':
        partA(num_images)
        part_b_config = select_part_b_config()
        partB(part_b_config)
    else:
        print("Invalid choice. Exiting.")
