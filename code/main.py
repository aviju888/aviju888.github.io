import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

# ##############################################################################
# ############### CONFIG STUFF #################################################

# ##TODO: add more prints for debugging

nppp = 8 # num correspondence points between imgs

def dirs():
    # setting up directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, '../data')
    output_dir = os.path.join(script_dir, 'output')
    points_file = os.path.join(script_dir, 'points.json')

    os.makedirs(output_dir, exist_ok=True)

    image1_path = os.path.join(images_dir, 'image1.jpg')
    image2_path = os.path.join(images_dir, 'image2.jpg')
    image3_path = os.path.join(images_dir, 'image3.jpg')

    return script_dir, images_dir, output_dir, points_file, image1_path, image2_path, image3_path

def load_imgs(image1_path, image2_path, image3_path):
    """
    Loads three images and converts them to RGB.
    """
    print("loading images...")
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    image3 = cv2.imread(image3_path)

    if image1 is None or image2 is None or image3 is None:
        raise ValueError("failed to load images. check file paths.")

    # Convert to RGB
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image3_rgb = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

    print("images loaded and converted to rgb.")
    return image1_rgb, image2_rgb, image3_rgb

# ##############################################################################
# ############### MAIN CHUNK OF STUFF AND HELPERS ##############################

def display_imgs(images, titles, delay=0):
    print("displaying images...")
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    if delay > 0:
        plt.show(block=False)
        plt.pause(delay)
        plt.close()
    else:
        plt.show()
    print("images displayed.")

def correspondence(image1, image2, num_points, description):
    # user input. get correspondence points for all the imgs

    print(f"getting {num_points} corresponding points for {description}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image1)
    ax1.set_title('Image 1')
    ax1.axis('off')
    ax2.imshow(image2)
    ax2.set_title('Image 2')
    ax2.axis('off')

    print(f"click on a point in image 1, then the corresponding point in image 2. repeat for {num_points} points.")

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
                print("click on the correct image in the correct order.")

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

# 1: visualize correspondences
def visualize1(image1, image2, pts1, pts2, title, save_path):
    print(f"visualizing correspondences: {title}")

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.scatter(pts1[:, 0], pts1[:, 1], c='r', marker='o')
    plt.title('Image 1 Points')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.scatter(pts2[:, 0], pts2[:, 1], c='b', marker='o')
    plt.title('Image 2 Points')
    plt.axis('off')

    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        plt.plot([x1, x2], [y1, y2], 'g--', linewidth=1)

    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"correspondences visualization saved to {save_path}.")
    except Exception as e:
        print(f"failed to save correspondences visualization to {save_path}: {e}")

    plt.close()
    print(f"correspondences visualized and saved: {title}")

# 2: visualize transformations
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
    plt.scatter(pts_target[:, 0], pts_target[:, 1], c='r', marker='o', label='Target Points')
    plt.scatter(transformed_pts[:, 0], transformed_pts[:, 1], c='b', marker='x', label='Transformed Source Points')
    plt.title(title)
    plt.legend()
    plt.axis('off')

    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"transformation visualization saved to {save_path}.")
    except Exception as e:
        print(f"failed to save transformation visualization to {save_path}: {e}")

    plt.close()
    print(f"transformation visualized and saved: {title}")

# compute homogrpahy matrix
def computeH(im1_pts, im2_pts):
    print("computing computeH...")
    # DLT algorithm
    N = im1_pts.shape[0]
    if N < 4:
        raise ValueError("at least 4 points are required for computeH")
    A = []
    for i in range(N):
        x, y = im1_pts[i][0], im1_pts[i][1]
        x_prime, y_prime = im2_pts[i][0], im2_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # last row of Vt -> smallest singular value
    H = h.reshape((3, 3))
    H /= H[2, 2]
    print(f"computed computeH h:\n{H}")
    return H

#figure out panorama size bc offsets
def pano_size(images, homographies):

    print("computing panorama size...")
    all_corners = []

    for i, (image, H) in enumerate(zip(images, homographies)):
        h, w = image.shape[:2]
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ])  # shape: (4, 3)

        # use computeH
        transformed_corners = np.dot(H, corners.T).T

        # for normalizing
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
    print(f"panorama width: {panorama_width}, panorama height: {panorama_height}")

    offset_x = -x_min
    offset_y = -y_min
    print(f"computed offsets: x_offset = {offset_x}, y_offset = {offset_y}")

    return (panorama_height, panorama_width), (offset_x, offset_y)

# warp using computeH matrix -- applies offset
def warpImage(image, H, panorama_size, offset):

    print("warping image...")

    panorama_height, panorama_width = panorama_size
    offset_x, offset_y = offset

    warped_image = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    H_inv = np.linalg.inv(H)

    y_indices, x_indices = np.indices((panorama_height, panorama_width))
    x_indices_flat = x_indices.flatten()
    y_indices_flat = y_indices.flatten()

    x_panorama = x_indices_flat - offset_x # offsetttt
    y_panorama = y_indices_flat - offset_y

    # homog coords
    ones = np.ones_like(x_panorama)
    output_coords = np.stack((x_panorama, y_panorama, ones), axis=1)  # shape: (N, 3)

    # transform with H inv
    source_coords = np.dot(H_inv, output_coords.T).T  # shape: (N, 3)
    source_coords /= source_coords[:, [2]] + 1e-8  # Normalize
    x_src = source_coords[:, 0]
    y_src = source_coords[:, 1]

    x_src_int = np.floor(x_src).astype(np.int32)
    y_src_int = np.floor(y_src).astype(np.int32)

    # mask of valid coordinates inside bounds
    valid_mask = (
        (x_src_int >= 0) & (x_src_int < image.shape[1]) &
        (y_src_int >= 0) & (y_src_int < image.shape[0])
    )

    x_dst = x_indices_flat[valid_mask]
    y_dst = y_indices_flat[valid_mask]
    x_src_valid = x_src_int[valid_mask]
    y_src_valid = y_src_int[valid_mask]

    # FINAL WARPPPPPP
    warped_image[y_dst, x_dst] = image[y_src_valid, x_src_valid]
    print("image warped successfully.")
    return warped_image

# feather blending mask -- accounts for dist from img borders
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

# uses feather mask, blends imgs
def blend_imgs(warped_images, panorama_size, offset):
    print("blending images into mosaic...")
    panorama = np.zeros((panorama_size[0], panorama_size[1], 3), dtype=np.float32)
    weight_sum = np.zeros((panorama_size[0], panorama_size[1]), dtype=np.float32)

    for i, warped_image in enumerate(warped_images):
        print(f"processing warped image {i + 1}...")
        mask = (cv2.cvtColor(warped_image, cv2.COLOR_RGB2GRAY) > 0).astype(np.float32)

        feat_mask = feather_mask(warped_image) * mask
        feat_mask_3ch = cv2.merge([feat_mask, feat_mask, feat_mask])

        # adding to panorama
        panorama += warped_image.astype(np.float32) * feat_mask_3ch

        weight_sum += feat_mask

    weight_sum[weight_sum == 0] = 1.0

    # normalizing
    panorama /= weight_sum[..., np.newaxis]
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)

    print("mosaic finished successfully (with feather blending)!")
    return panorama

# ##############################################################################
# ############### MAIN FUNCTION ################################################

def main():
    print("program started.")

    # pt1 fix dirs
    # ----------------------------------------
    # Set up directories
    script_dir, images_dir, output_dir, points_file, image1_path, image2_path, image3_path = dirs()
    image1_rgb, image2_rgb, image3_rgb = load_imgs(image1_path, image2_path, image3_path)
    display_imgs([image1_rgb, image2_rgb, image3_rgb], ["Image 1", "Image 2", "Image 3"])

    num_points = nppp 

    # pt2 correspondence pts
    # --------------------------------------
    # SAVED OR NEW?
    use_saved = input("use saved points from json file? (y/n): ").strip().lower()

    # old points
    if use_saved == 'y' and os.path.exists(points_file):
        with open(points_file, 'r') as f:
            points_data = json.load(f)
        pts1 = np.array(points_data['pts1'], dtype=np.float32)
        pts2 = np.array(points_data['pts2'], dtype=np.float32)
        pts3 = np.array(points_data['pts3'], dtype=np.float32)
        pts2_3 = np.array(points_data['pts2_3'], dtype=np.float32)
    else:
    # else, new points
        print("collecting new points.")
        print("select points between image 1 and image 2.")
        pts1, pts2 = correspondence(image1_rgb, image2_rgb, num_points, "Image 1 and Image 2")

        print("select points between image 3 and image 2.")
        pts3, pts2_3 = correspondence(image3_rgb, image2_rgb, num_points, "Image 3 and Image 2")

        with open(points_file, 'w') as f:
            json.dump({
                'pts1': pts1.tolist(), 'pts2': pts2.tolist(), 'pts3': pts3.tolist(), 'pts2_3': pts2_3.tolist()
            })

    # just in case
    if not (pts1.shape == pts2.shape == pts3.shape == pts2_3.shape):
        print("point arrays do not have the same shape.")
        return

    # pt3 homographies
    # -----------------------------------------------------------
    # visualize pt2 stuff
    visualize1(
        image1_rgb, image2_rgb, pts1, pts2,
        "Correspondences between Image 1 and Image 2",
        os.path.join(images_dir, "correspondences_image1_image2.png")
    )

    visualize1(
        image3_rgb, image2_rgb, pts3, pts2_3,
        "Correspondences between Image 3 and Image 2",
        os.path.join(images_dir, "correspondences_image3_image2.png")
    )

    # compute h
    print("computing homographies...")
    H1to2 = computeH(pts1, pts2)  # img1 to 2
    H3to2 = computeH(pts3, pts2_3)  # img3 to 2

    visualize2(
        image2_rgb, H1to2, pts1, pts2,
        "Point Transformation for Image 1",
        os.path.join(images_dir, "transformation_image1.png")
    )

    visualize2(
        image2_rgb, H3to2, pts3, pts2_3,
        "Point Transformation for Image 3",
        os.path.join(images_dir, "transformation_image3.png")
    )

    # pt4: do panorama size and warp imgs
    # ----------------------------------------------

    print("computing panorama size...")
    homographies = [H1to2, np.eye(3), H3to2]  # The reference is Image 2
    images = [image1_rgb, image2_rgb, image3_rgb]
    panorama_size, offset = pano_size(images, homographies)

    # Warp images to align them onto the panorama canvas
    print("warping images onto panorama canvas...")
    warped_image1 = warpImage(image1_rgb, H1to2, panorama_size, offset)  # img1 to img2
    warped_image2 = warpImage(image2_rgb, np.eye(3), panorama_size, offset)  # ref (img2)
    warped_image3 = warpImage(image3_rgb, H3to2, panorama_size, offset)  # img3 to img2

    # Display original and warped images
    display_imgs([image1_rgb, image2_rgb, image3_rgb], ["img1", "img2 (reference)", "img3"])
    display_imgs([warped_image1, warped_image2, warped_image3], ["warped img1", "warped img2 (reference)", "warped img3"])

    # pt5: mosaic -- feather blending
    # -----------------------------------------------------
    print("creating mosaic with feather blending...")
    warped_images = [warped_image1, warped_image2, warped_image3]
    panorama = blend_imgs(warped_images, panorama_size, offset)

    # FINAL MOSAAICCCCC
    mosaic_path = os.path.join(output_dir, 'mosaic.jpg')
    cv2.imwrite(mosaic_path, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
    print(f"mosaic saved to {mosaic_path}.")
    display_imgs([panorama], ["Mosaic"])
    print("program completed successfully.")

if __name__ == '__main__':
    main()
