import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

from skimage.feature import corner_harris, peak_local_max
from scipy.spatial import KDTree

# ##############################################################################
# ############### CONFIG STUFF #################################################

# TODO: add more prints for debugging
# TODO: add more mosaic examples [part A]
# TODO: finish [part B]

nppp = 8  # Number of correspondence points between images [Part A]

# Configuration Modes
FAST_MODE_CONFIG = {
    'num_corners': 200,             # fewer corners, speed
    'anms_points': 100,             # fewer points
    'harris_threshold': 0.05,       # higher threshold
    'edge_discard': 15,             # edge discard
    'descriptor_window_size': 10,   # smaller window
    'descriptor_size': 4,           # small patch
    'feature_ratio_thresh': 0.9,    # relaxed ratio
    'ransac_max_iters': 2000,       # fewer iterations
    'ransac_inlier_thresh': 1.5,    # relaxed inlier thresh
}

BALANCED_MODE_CONFIG = {
    'num_corners': 500,             # moderate corners
    'anms_points': 200,             # balanced points
    'harris_threshold': 0.01,       # moderate threshold
    'edge_discard': 10,             # edge discard
    'descriptor_window_size': 20,   # moderate window
    'descriptor_size': 6,           # medium patch
    'feature_ratio_thresh': 0.85,   # balanced ratio
    'ransac_max_iters': 4000,       # sufficient iterations
    'ransac_inlier_thresh': 1.0,    # reasonable inlier thresh
}

HIGH_QUALITY_MODE_CONFIG = {
    'num_corners': 2000,            # more corners, detail
    'anms_points': 300,             # more points for accuracy
    'harris_threshold': 0.005,      # lower threshold
    'edge_discard': 15,             # fewer edge discards
    'descriptor_window_size': 80,   # larger window
    'descriptor_size': 5,           # larger patch
    'feature_ratio_thresh': 0.55,   # stricter ratio
    'ransac_max_iters': 6000,       # more iterations
    'ransac_inlier_thresh': 0.5,    # tighter inlier thresh
}


def select_part_b_config():
    print("select the mode for part b's config:")
    print("1. fast mode")
    print("2. balanced mode")
    print("3. high-quality mode")
    choice = input("enter 1, 2, or 3: ").strip()

    if choice == '1':
        return FAST_MODE_CONFIG
    elif choice == '2':
        return BALANCED_MODE_CONFIG
    elif choice == '3':
        return HIGH_QUALITY_MODE_CONFIG
    else:
        print("ERR. defaulting to balanced mode.")
        return BALANCED_MODE_CONFIG

PART_B_CONFIG = None

# ##############################################################################
# ############### [A] HELPER FUNCTIONS #########################################

def dirs():
    # Setting up directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, '../data')
    output_dir = os.path.join(script_dir, 'output')
    points_file = os.path.join(script_dir, 'points.json')

    os.makedirs(output_dir, exist_ok=True)

    image1_path = os.path.join(images_dir, 'image1d.jpg')
    image2_path = os.path.join(images_dir, 'image2d.jpg')
    image3_path = os.path.join(images_dir, 'image3d.jpg')

    return script_dir, images_dir, output_dir, points_file, image1_path, image2_path, image3_path

def load_imgs(image1_path, image2_path, image3_path):
    print("loading images.")
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    image3 = cv2.imread(image3_path)

    if image1 is None or image2 is None or image3 is None:
        raise ValueError("failed to load images. check paths.")

    # Convert to RGB
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image3_rgb = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

    print("images loaded and converted.")
    return image1_rgb, image2_rgb, image3_rgb

def display_imgs(images, titles, delay=0):
    print("displaying images.")
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    if delay > 0:
        plt.show(block=False)
        plt.pause(delay)
        plt.close()
    else:
        plt.show()
    print("images displayed.")

# vis correspondences
def visualize1(image1, image2, pts1, pts2, title, save_path):

    print(f"visualizing correspondences: {title}")

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.scatter(pts1[:, 0], pts1[:, 1], c='r', marker='o')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.scatter(pts2[:, 0], pts2[:, 1], c='b', marker='o')
    plt.axis('off')

    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        plt.plot([x1, x2], [y1, y2], 'g--', linewidth=1)

    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"correspondences saved to {save_path}.")
    except Exception as e:
        print(f"failed to save correspondences to {save_path}: {e}")

    plt.close()
    print(f"correspondences visualized and saved: {title}")

# vis homography transforms
def visualize2(image, H, pts_source, pts_target, title, save_path):

    print(f"visualizing transformation: {title}")
    print(f"h shape: {H.shape}")
    print(f"h contents:\n{H}")

    pts_source_homogeneous = np.hstack([pts_source, np.ones((pts_source.shape[0], 1))])  # Shape: (N, 3)
    transformed_pts_homogeneous = np.dot(H, pts_source_homogeneous.T).T  # shape: (N, 3)

    transformed_pts_homogeneous /= transformed_pts_homogeneous[:, [2]] + 1e-8  # Avoid division by zero
    transformed_pts = transformed_pts_homogeneous[:, :2]

    print(f"transformed points:\n{transformed_pts}")

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.scatter(pts_target[:, 0], pts_target[:, 1], c='r', marker='o', label='target points')
    plt.scatter(transformed_pts[:, 0], transformed_pts[:, 1], c='b', marker='x', label='transformed points')
    plt.legend()
    plt.axis('off')

    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"transformation saved to {save_path}.")
    except Exception as e:
        print(f"failed to save transformation to {save_path}: {e}")

    plt.close()
    print(f"transformation visualized and saved: {title}")

# ##############################################################################
# ############### PART A FUNCTIONS #############################################

def correspondence(image1, image2, num_points, description):
    print(f"getting {num_points} corresponding points for {description}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image1)
    ax1.set_title('Image 1')
    ax1.axis('off')
    ax2.imshow(image2)
    ax2.set_title('Image 2')
    ax2.axis('off')

    print(f"click on a point in image1 then image2 for {num_points} points.")

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
                print(f"image1 point {len(self.pts1)}: ({x:.2f}, {y:.2f})")
                self.current_image = 'Image2'
                fig.canvas.draw()
            elif event.inaxes == self.ax2 and self.current_image == 'Image2':
                x, y = event.xdata, event.ydata
                self.pts2.append([x, y])
                self.ax2.scatter(x, y, c='b', marker='o')
                x1, y1 = self.pts1[-1]
                x2, y2 = self.pts2[-1]
                self.ax1.plot([x1, x2], [y1, y2], 'g--', linewidth=1)
                print(f"image2 point {len(self.pts2)}: ({x:.2f}, {y:.2f})")
                self.current_image = 'Image1'
                fig.canvas.draw()
            else:
                print("click on the correct image in order.")

            if len(self.pts1) == self.num_points and len(self.pts2) == self.num_points:
                fig.canvas.mpl_disconnect(self.cid)
                plt.close()

    selector = CorrSelect(ax1, ax2, num_points)
    plt.show()

    if len(selector.pts1) != num_points or len(selector.pts2) != num_points:
        raise ValueError("not enough points selected.")

    pts1 = np.array(selector.pts1, dtype=np.float32)
    pts2 = np.array(selector.pts2, dtype=np.float32)

    print(f"selected points:\npts1: {pts1}\npts2: {pts2}")
    return pts1, pts2

# use dlt, compute homographies
def computeH(im1_pts, im2_pts):
    # DLT algorithm
    N = im1_pts.shape[0]
    if N < 4:
        raise ValueError("need at least 4 points for computeH")
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

# based on transformed img corners
def pano_size(images, homographies):
    print("computing panorama size.")
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
        print(f"transformed corners for image {i+1}:\n{transformed_corners}")
        all_corners.append(transformed_corners)

    all_corners = np.vstack(all_corners)
    print(f"all transformed corners:\n{all_corners}")
    x_min, y_min = np.floor(np.min(all_corners, axis=0)).astype(int)
    x_max, y_max = np.ceil(np.max(all_corners, axis=0)).astype(int)

    panorama_width = x_max - x_min
    panorama_height = y_max - y_min
    print(f"panorama width: {panorama_width}, height: {panorama_height}")

    offset_x = -x_min
    offset_y = -y_min
    print(f"offsets: x = {offset_x}, y = {offset_y}")

    return (panorama_height, panorama_width), (offset_x, offset_y)

def warpImage(image, H, panorama_size, offset):
    print("warping image.")

    panorama_height, panorama_width = panorama_size
    offset_x, offset_y = offset

    warped_image = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    H_inv = np.linalg.inv(H)

    y_indices, x_indices = np.indices((panorama_height, panorama_width))
    x_indices_flat = x_indices.flatten()
    y_indices_flat = y_indices.flatten()

    x_panorama = x_indices_flat - offset_x
    y_panorama = y_indices_flat - offset_y

    ones = np.ones_like(x_panorama)
    output_coords = np.stack((x_panorama, y_panorama, ones), axis=1)  # shape: (N, 3)

    #  H_inv
    source_coords = np.dot(H_inv, output_coords.T).T  # shape: (N, 3)
    source_coords /= source_coords[:, [2]] + 1e-8  # Normalize
    x_src = source_coords[:, 0]
    y_src = source_coords[:, 1]

    x_src_int = np.floor(x_src).astype(np.int32)
    y_src_int = np.floor(y_src).astype(np.int32)

    # valid coord mask inside bounds
    valid_mask = (
        (x_src_int >= 0) & (x_src_int < image.shape[1]) &
        (y_src_int >= 0) & (y_src_int < image.shape[0])
    )

    x_dst = x_indices_flat[valid_mask]
    y_dst = y_indices_flat[valid_mask]
    x_src_valid = x_src_int[valid_mask]
    y_src_valid = y_src_int[valid_mask]

    # FINAL WARPPP
    warped_image[y_dst, x_dst] = image[y_src_valid, x_src_valid]
    print("image warped.")
    return warped_image

def feather_mask(image):
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

# Uses feather blend and adds all of it up
def blend_imgs(warped_images, panorama_size, offset):
    print("blending images.")
    panorama = np.zeros((panorama_size[0], panorama_size[1], 3), dtype=np.float32)
    weight_sum = np.zeros((panorama_size[0], panorama_size[1]), dtype=np.float32)

    for i, warped_image in enumerate(warped_images):
        print(f"processing warped image {i + 1}.")
        mask = (cv2.cvtColor(warped_image, cv2.COLOR_RGB2GRAY) > 0).astype(np.float32)

        feat_mask = feather_mask(warped_image) * mask
        feat_mask_3ch = cv2.merge([feat_mask, feat_mask, feat_mask])

        panorama += warped_image.astype(np.float32) * feat_mask_3ch

        weight_sum += feat_mask

    weight_sum[weight_sum == 0] = 1.0

    panorama /= weight_sum[..., np.newaxis]
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)

    print("mosaic finished (w/ feather blending!!).")
    return panorama

# ###########################################################################
# ############### [B] HELPER FUNCTIONS ######################################

# inlier and outleir matches after RANSAC
def visualize_inliers_outliers(image1, image2, pts1, pts2, inliers, title, save_path):
    print(f"visualizing inliers and outliers: {title}")
    plt.figure(figsize=(20, 10))

    combined_image = np.hstack((image1, image2))
    plt.imshow(combined_image)

    offset = image1.shape[1]
    pts2_adj = pts2.copy()
    pts2_adj[:, 0] += offset

    # inliers green
    inlier_pts1 = pts1[inliers]
    inlier_pts2 = pts2_adj[inliers]
    plt.scatter(inlier_pts1[:, 0], inlier_pts1[:, 1], c='g', marker='o', label='inliers')
    plt.scatter(inlier_pts2[:, 0], inlier_pts2[:, 1], c='g', marker='o')

    for i in range(len(inlier_pts1)):
        plt.plot([inlier_pts1[i, 0], inlier_pts2[i, 0]], [inlier_pts1[i, 1], inlier_pts2[i, 1]], 'g-', linewidth=0.5)

    # outliers red
    outlier_pts1 = pts1[~inliers]
    outlier_pts2 = pts2_adj[~inliers]
    plt.scatter(outlier_pts1[:, 0], outlier_pts1[:, 1], c='r', marker='x', label='outliers')
    plt.scatter(outlier_pts2[:, 0], outlier_pts2[:, 1], c='r', marker='x')

    for i in range(len(outlier_pts1)):
        plt.plot([outlier_pts1[i, 0], outlier_pts2[i, 0]], [outlier_pts1[i, 1], outlier_pts2[i, 1]], 'r--', linewidth=0.5)

    plt.legend()
    plt.axis('off')
    plt.show()

    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"inliers and outliers saved to {save_path}.")
    except Exception as e:
        print(f"failed to save inliers/outliers to {save_path}: {e}")

    plt.close()
    print(f"inliers and outliers visualized and saved: {title}")

# imageA and B map using homography
def visualize_homography_mapping(imageA, imageB, H, ptsA, title, save_path):
    print(f"visualizing homography mapping: {title}")
    plt.figure(figsize=(20, 10))
    combined_image = np.hstack((imageA, imageB))
    plt.imshow(combined_image)

    # points from imageA to imageB
    ptsA_homogeneous = np.hstack([ptsA, np.ones((ptsA.shape[0], 1))])
    projected_pts2 = (H @ ptsA_homogeneous.T).T
    projected_pts2 /= projected_pts2[:, [2]] + 1e-8  # Normalize
    projected_pts2 = projected_pts2[:, :2]

    offset = imageA.shape[1]


    plt.scatter(ptsA[:, 0], ptsA[:, 1], c='r', s=40, label='original points (img a)')
    plt.scatter(projected_pts2[:, 0] + offset, projected_pts2[:, 1], c='b', s=40, label='projected points (img b)')

    for (x1, y1), (x2, y2) in zip(ptsA, projected_pts2):
        plt.plot([x1, x2 + offset], [y1, y2], 'k--', linewidth=1)

    plt.legend()
    plt.axis('off')
    plt.show()

    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"homography mapping saved to {save_path}.")
    except Exception as e:
        print(f"failed to save homography mapping to {save_path}: {e}")

    plt.close()
    print(f"homography mapping visualized and saved: {title}")

def visualize_corners(image, keypoints, title, save_path):
    print(f"visualizing corners: {title}")
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=10)
    plt.axis('off')
    plt.show()

    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"corners saved to {save_path}.")
    except Exception as e:
        print(f"failed to save corners to {save_path}: {e}")

    plt.close()
    print(f"corners visualized and saved: {title}")

def visualize_anms_corners(image, keypoints, title, save_path):
    print(f"visualizing anms corners: {title}")
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='b', s=10)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"anms corners saved to {save_path}.")

# def visualize_descriptors(image, keypoints, descriptors, num_descriptors=5, save_path=None):
#     print("visualizing feature descriptors.")
#     plt.figure(figsize=(num_descriptors * 2, 2))
#     for i in range(min(num_descriptors, len(descriptors))):
#         patch = descriptors[i].reshape((config['descriptor_size'], config['descriptor_size']))
#         patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
#         plt.subplot(1, num_descriptors, i + 1)
#         plt.imshow(patch, cmap='gray')
#         plt.axis('off')
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
#         print(f"descriptors saved to {save_path}.")
#     plt.show()

def visualize_matches(image1, image2, keypoints1, keypoints2, matches, title, save_path):
    print(f"visualizing matches: {title}")
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
    plt.show()
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"matches saved to {save_path}.")
    except Exception as e:
        print(f"failed to save matches to {save_path}: {e}")

    plt.close()
    print(f"matches visualized and saved: {title}")

# ##############################################################################
# ############### PART B FUNCTIONS #############################################

# HARRIS CORNERS
def harris_corners1(im, edge_discard=20, threshold_factor=0.0005, sigma=1, min_distance=0):
    h = corner_harris(im, sigma=sigma)
    threshold = threshold_factor * h.max()
    coords = peak_local_max(h, min_distance=min_distance, threshold_abs=threshold)

    mask = (
        (coords[:, 0] >= edge_discard) & (coords[:, 0] < im.shape[0] - edge_discard) &
        (coords[:, 1] >= edge_discard) & (coords[:, 1] < im.shape[1] - edge_discard)
    )
    coords = coords[mask]

    return coords, h

# wrapper func
def harris_corners2(image, config):
    num_corners = config['num_corners']
    threshold = config['harris_threshold']
    edge_discard = config['edge_discard']

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    corner_coords, corner_response = harris_corners1(
        gray, edge_discard=edge_discard, threshold_factor=threshold
    )

    if corner_coords.shape[0] == 0:
        print("no valid coords after masking.")
        return np.array([]), corner_response
    strengths = corner_response[corner_coords[:, 0], corner_coords[:, 1]]
    sorted = np.argsort(-strengths)
    keyp_sorted = corner_coords[sorted]
    keyp_limited = keyp_sorted[:num_corners]

    # convert (y, x) to (x, y)
    keyp_limited = keyp_limited[:, ::-1]  # Now (x, y)

    print(f"corners detected: {len(keyp_limited)}")

    return keyp_limited, corner_response

# ANMS
def anms(keypoints, corner_response, num_points, img_shape):
    # adaptive non-maximal suppression LMAO
    print("performing anms.")
    num_keypoints = keypoints.shape[0]
    print(f"initial keypoints: {num_keypoints}")

    if num_keypoints == 0:
        print("no keypoints to process.")
        return np.array([])

    strengths = corner_response[keypoints[:, 1], keypoints[:, 0]]
    sorted_indices = np.argsort(-strengths)
    keypoints = keypoints[sorted_indices]
    strengths = strengths[sorted_indices]

    radii = np.full(len(keypoints), np.inf)

    for i in range(len(keypoints)):
        for j in range(i):
            if strengths[j] > strengths[i]:
                dist = np.linalg.norm(keypoints[i] - keypoints[j])
                if dist < radii[i]:
                    radii[i] = dist

    selected_indices = np.argsort(-radii)[:num_points]
    selected_keypoints = keypoints[selected_indices]

    valid_mask = (
        (selected_keypoints[:, 0] >= 0) & (selected_keypoints[:, 0] < img_shape[1]) &
        (selected_keypoints[:, 1] >= 0) & (selected_keypoints[:, 1] < img_shape[0])
    )
    selected_keypoints = selected_keypoints[valid_mask]

    print(f"keypoints after anms: {len(selected_keypoints)}")
    return selected_keypoints

# FEATURES
def extract_features(image, keypoints, config):
    # extract descriptors
    print("extracting descriptors.")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    descrp = []
    valid_keyp = []

    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)

    half_window = config['descriptor_window_size'] // 2
    descriptor_size = config['descriptor_size']
    img_height, img_width = gray.shape[:2]

    for point in keypoints:
        x, y = int(point[0]), int(point[1])

        if (y - half_window < 0 or y + half_window >= img_height or
            x - half_window < 0 or x + half_window >= img_width):
            continue 

        #  orientation
        window_Ix = Ix[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
        window_Iy = Iy[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
        orientation = np.arctan2(window_Iy, window_Ix)
        magnitude = np.sqrt(window_Ix**2 + window_Iy**2)

        # weighed average orientation
        weighed = np.arctan2(
            np.sum(magnitude * np.sin(orientation)),
            np.sum(magnitude * np.cos(orientation))
        )
        angle = np.degrees(weighed)

        # rotate to align w  dominant orientation
        M = cv2.getRotationMatrix2D((half_window, half_window), angle, 1)
        patch = gray[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
        rotated_patch = cv2.warpAffine(patch, M, (2 * half_window + 1, 2 * half_window + 1))
        small = cv2.resize(rotated_patch, (descriptor_size, descriptor_size), interpolation=cv2.INTER_AREA).astype(np.float32)

        # normalization
        small -= np.mean(small)
        norm = np.linalg.norm(small)
        if norm > 1e-5:
            small /= norm

        descrp.append(small.flatten())
        valid_keyp.append([x, y])  # store as (x, y)

    descrp = np.array(descrp)
    valid_keyp = np.array(valid_keyp)
    print(f"descriptors extracted: {len(descrp)}")

    return valid_keyp, descrp

# MATCHING FEATURES
def lowes(descriptors1, descriptors2, ratio_thresh=0.75):

    # Lowe's ratio test
    print("matching features with lowe's ratio.")
    matches = []
    tree = KDTree(descriptors2)
    for i, desc1 in enumerate(descriptors1):
        distances, indices = tree.query(desc1, k=2)
        if len(distances) < 2:
            continue
        if distances[0] < ratio_thresh * distances[1]:
            matches.append((i, indices[0]))
        if (i + 1) % 500 == 0 or i == len(descriptors1) - 1:
            print(f"processed {i + 1}/{len(descriptors1)} descriptors.")
    print(f"matches after ratio test: {len(matches)}")
    return matches

# HOMOGRAPHY AGAIN
def computeH_ransac(pts1, pts2, config):
    # compute homography w RANSAC
    max_iters = config['ransac_max_iters']
    inlier_thresh = config['ransac_inlier_thresh']
    print("computing homography with ransac.")
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
            continue 

        # proj  pts1 to image2
        pts1_homogeneous = np.hstack([pts1, np.ones((num_points, 1))])
        projected_pts2 = (H @ pts1_homogeneous.T).T
        projected_pts2 /= projected_pts2[:, [2]]
        projected_pts2 = projected_pts2[:, :2]

        distances = np.linalg.norm(pts2 - projected_pts2, axis=1)
        inliers = distances < inlier_thresh

        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_H = H
            if np.sum(inliers) > 0.85 * num_points:
                break

    if best_H is None or np.sum(best_inliers) < 4:
        print("failed to compute a valid homography.")
        return None, None

    # do homog again using all inliers
    best_H = computeH(pts1[best_inliers], pts2[best_inliers])

    return best_H, best_inliers

# ##############################################################################
# ############### MAIN FUNCTIONS ################################################

def partA():

    # do 
    print("part a.")
    script_dir, images_dir, output_dir, points_file, image1_path, image2_path, image3_path = dirs()
    image1_rgb, image2_rgb, image3_rgb = load_imgs(image1_path, image2_path, image3_path)
    display_imgs([image1_rgb, image2_rgb, image3_rgb], ["Image 1", "Image 2", "Image 3"])

    num_points = nppp 

    print("collecting correspondence points.")

    use_saved = input("use saved points from json file? (y/n): ").strip().lower()

    if use_saved == 'y' and os.path.exists(points_file):
        with open(points_file, 'r') as f:
            points_data = json.load(f)
        pts1 = np.array(points_data['pts1'], dtype=np.float32)
        pts2 = np.array(points_data['pts2'], dtype=np.float32)
        pts3 = np.array(points_data['pts3'], dtype=np.float32)
        pts2_3 = np.array(points_data['pts2_3'], dtype=np.float32)
        print("loaded points from json file.")
    else:
        print("collecting new correspondence points.")
        print("select points between image1 and image2.")
        pts1, pts2 = correspondence(image1_rgb, image2_rgb, num_points, "Image 1 and Image 2")

        print("select points between image3 and image2.")
        pts3, pts2_3 = correspondence(image3_rgb, image2_rgb, num_points, "Image 3 and Image 2")


        with open(points_file, 'w') as f:
            json.dump({
                'pts1': pts1.tolist(), 
                'pts2': pts2.tolist(), 
                'pts3': pts3.tolist(), 
                'pts2_3': pts2_3.tolist()
            }, f)
        print(f"saved points to {points_file}.")

    if not (pts1.shape == pts2.shape == pts3.shape == pts2_3.shape):
        print("point shapes mismatch. exiting part a.")
        return

    # correspondences
    visualize1(
        image1_rgb, image2_rgb, pts1, pts2,
        "correspondences between image1 and image2",
        os.path.join(images_dir, "correspondences_image1_image2.png")
    )

    visualize1(
        image3_rgb, image2_rgb, pts3, pts2_3,
        "correspondences between image3 and image2",
        os.path.join(images_dir, "correspondences_image3_image2.png")
    )

    #  homographies
    print("computing homographies.")
    # Note: computeH only takes two arguments, remove PART_B_CONFIG
    H1to2 = computeH(pts1, pts2)  # img1 to img2
    H3to2 = computeH(pts3, pts2_3)  # img3 to img2

    # transformations (visualize)
    visualize2(
        image2_rgb, H1to2, pts1, pts2,
        "point transformation for image1",
        os.path.join(images_dir, "transformation_image1.png")
    )

    visualize2(
        image2_rgb, H3to2, pts3, pts2_3,
        "point transformation for image3",
        os.path.join(images_dir, "transformation_image3.png")
    )

    # panorama size
    print("computing panorama size.")
    homographies = [H1to2, np.eye(3), H3to2]  # The reference is Image 2
    images = [image1_rgb, image2_rgb, image3_rgb]
    panorama_size, offset = pano_size(images, homographies)

    print("warping images.")
    warped_image1 = warpImage(image1_rgb, H1to2, panorama_size, offset)  # img1 to img2
    warped_image2 = warpImage(image2_rgb, np.eye(3), panorama_size, offset)  # ref (img2)
    warped_image3 = warpImage(image3_rgb, H3to2, panorama_size, offset)  # img3 to img2
    display_imgs([image1_rgb, image2_rgb, image3_rgb], ["Image 1", "Image 2 (Reference)", "Image 3"])
    display_imgs([warped_image1, warped_image2, warped_image3], ["Warped Image 1", "Warped Image 2 (Reference)", "Warped Image 3"])


    print("creating mosaic with feather.")
    warped_images = [warped_image1, warped_image2, warped_image3]
    panorama = blend_imgs(warped_images, panorama_size, offset)
    mosaic_path = os.path.join(output_dir, 'mosaic.jpg')
    cv2.imwrite(mosaic_path, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
    print(f"mosaic saved to {mosaic_path}.")
    display_imgs([panorama], ["Mosaic"])
    print("program completed - part a.")

def partB(configuration):
    print("part b: automatic feature matching.")

    script_dir, images_dir, output_dir, points_file, image1_path, image2_path, image3_path = dirs()
    image1_rgb, image2_rgb, image3_rgb = load_imgs(image1_path, image2_path, image3_path)

    config = configuration

    # step 1: Detect Harris Corners
    print("\n---> STEP 1: detect corners")
    keypoints1, corner_response1 = harris_corners2(image1_rgb, config)
    keypoints2, corner_response2 = harris_corners2(image2_rgb, config)
    keypoints3, corner_response3 = harris_corners2(image3_rgb, config)

    visualize_corners(image1_rgb, keypoints1, "harris corners image1", os.path.join(output_dir, "harris_corners_image1.png"))
    visualize_corners(image2_rgb, keypoints2, "harris corners image2", os.path.join(output_dir, "harris_corners_image2.png"))
    visualize_corners(image3_rgb, keypoints3, "harris corners image3", os.path.join(output_dir, "harris_corners_image3.png"))

    # step 2: Apply ANMS
    print("\n---> STEP 2: apply anms")
    anms_points1 = anms(keypoints1, corner_response1, config['anms_points'], image1_rgb.shape)
    anms_points2 = anms(keypoints2, corner_response2, config['anms_points'], image2_rgb.shape)
    anms_points3 = anms(keypoints3, corner_response3, config['anms_points'], image3_rgb.shape)

    visualize_anms_corners(image1_rgb, anms_points1, "anms corners image1", os.path.join(output_dir, "anms_corners_image1.png"))
    visualize_anms_corners(image2_rgb, anms_points2, "anms corners image2", os.path.join(output_dir, "anms_corners_image2.png"))
    visualize_anms_corners(image3_rgb, anms_points3, "anms corners image3", os.path.join(output_dir, "anms_corners_image3.png"))

    # step 3: Extract Feature Descriptors
    print("\n---> STEP 3: extract descriptors")
    valid_keyp1, descriptors1 = extract_features(image1_rgb, anms_points1, config)
    valid_keyp2, descriptors2 = extract_features(image2_rgb, anms_points2, config)
    valid_keyp3, descriptors3 = extract_features(image3_rgb, anms_points3, config)

    # step 4: Feature Matching
    print("\n---> STEP 4: match features")
    matches12 = lowes(descriptors1, descriptors2, config['feature_ratio_thresh'])
    matches32 = lowes(descriptors3, descriptors2, config['feature_ratio_thresh'])

    visualize_matches(
        image1_rgb, image2_rgb, valid_keyp1, valid_keyp2, matches12,
        "final matches image1 image2",
        os.path.join(output_dir, "final_matches_image1_image2.png")
    )
    visualize_matches(
        image3_rgb, image2_rgb, valid_keyp3, valid_keyp2, matches32,
        "final matches image3 image2",
        os.path.join(output_dir, "final_matches_image3_image2.png")
    )

    # step 5: Compute Homographies
    print("\n---> STEP 5: compute homographies")
    H1to2, inliers12 = None, None
    H3to2, inliers32 = None, None

    if len(matches12) >= 4:
        pts1 = valid_keyp1[[m[0] for m in matches12]]
        pts2 = valid_keyp2[[m[1] for m in matches12]]
        H1to2, inliers12 = computeH_ransac(pts1, pts2, config)
    else:
        print("not enough matches btwn image1 image2 to compute homography.")

    if len(matches32) >= 4:
        pts3 = valid_keyp3[[m[0] for m in matches32]]
        pts2_3 = valid_keyp2[[m[1] for m in matches32]]
        H3to2, inliers32 = computeH_ransac(pts3, pts2_3, config)
    else:
        print("not enough matches btwn image3 image2 to compute homography.")

    if H1to2 is None or H3to2 is None:
        print("failed to compute homographies. exiting part b.")
        return

    # step 6: Compute Panorama Size and Warp Images
    print("\n---> STEP 6: compute size and warp")
    homographies = [H1to2, np.eye(3), H3to2]
    images = [image1_rgb, image2_rgb, image3_rgb]
    panorama_size, offset = pano_size(images, homographies)

    warped_image1 = warpImage(image1_rgb, H1to2, panorama_size, offset)
    warped_image2 = warpImage(image2_rgb, np.eye(3), panorama_size, offset)
    warped_image3 = warpImage(image3_rgb, H3to2, panorama_size, offset)

    # step 7: Blend Images to Create Panorama
    print("\n---> STEP 7: blend images")
    warped_images = [warped_image1, warped_image2, warped_image3]
    panorama = blend_imgs(warped_images, panorama_size, offset)

    # Save and Display the Final Panorama
    mosaic_path = os.path.join(output_dir, 'mosaic_auto.jpg')
    cv2.imwrite(mosaic_path, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
    print(f"mosaic saved to {mosaic_path}.")
    display_imgs([panorama], ["mosaic - automatic stitching"])
    print("program completed - part b.")

if __name__ == '__main__':

    print("select which part to execute:")
    print("1. part a")
    print("2. part b")
    print("3. both part a and part b")
    choice = input("enter 1, 2, or 3: ").strip()

    if choice == '1':
        partA()
    elif choice == '2':
        part_b_config = select_part_b_config()
        partB(part_b_config)
    elif choice == '3':
        partA()
        part_b_config = select_part_b_config()
        partB(part_b_config)
    else:
        print("ERR. exiting.")
 