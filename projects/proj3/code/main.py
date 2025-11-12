import os
import numpy as np
import cv2
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import json

# ##############################################################################
# ############### CONFIG AND IMAGE LOADING #####################################

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')
image_A_path = os.path.join(images_dir, '../../web/images/imageMe.jpg')
image_B_path = os.path.join(images_dir, '../../web/images/imageObama.jpg')
jsonn = os.path.join(script_dir, 'imageMe_imageObama.json')

web_dir = os.path.join(script_dir, '..', 'web')
images_dir = os.path.join(web_dir, 'images', 'population')
population_images = [os.path.join(images_dir, f) for f 
                     in os.listdir(images_dir) if 'a' in f and 
                     f.endswith(('.jpg', '.png'))]

image_A = cv2.imread(image_A_path)
image_B = cv2.imread(image_B_path)
if image_A is None or image_B is None:
    raise ValueError("Failed to load images.")

# resizing if necessary

hA_orig, wA_orig = image_A.shape[:2]
hB_orig, wB_orig = image_B.shape[:2]
scale_A_x, scale_A_y = 1.0, 1.0
scale_B_x, scale_B_y = 1.0, 1.0
resized_A = False
resized_B = False

if image_A.shape != image_B.shape:
    if image_A.shape[0] < image_B.shape[0] or image_A.shape[1] < image_B.shape[1]:
        hB, wB = image_B.shape[:2]
        scale_A_x = wB / wA_orig
        scale_A_y = hB / hA_orig
        image_A = cv2.resize(image_A, (wB, hB))
        resized_A = True
    else:
        hA, wA = image_A.shape[:2]
        scale_B_x = wA / wB_orig
        scale_B_y = hA / hB_orig
        image_B = cv2.resize(image_B, (wA, hA))
        resized_B = True

# need to convert images to rgb
image_A_rgb = cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB)
image_B_rgb = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)

# ##############################################################################
# ############### HELPERS ######################################################

# lots of helper stuff for visualising 

def get_points(image, num_points):
    plt.imshow(image)
    plt.title(f"Select {num_points} points on this image")
    points = plt.ginput(num_points, timeout=0)
    plt.close()
    return np.array(points, dtype=np.float32)

def show_img(image, title=''):
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def vis_triangulation(image, points, tri, title=''):
    image_copy = image.copy()
    for t in tri.simplices:
        pts = points[t].astype(int)
        cv2.polylines(image_copy, [pts], isClosed=True, 
                      color=(0, 255, 0), thickness=1)
    for pt in points:
        cv2.circle(image_copy, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
    show_img(image_copy, title)

# stuff for displaying (progression and also images)
def display_imgs(images, titles):
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_progression(seq, num_steps=5):
    indices = np.linspace(0, len(seq)-1, num_steps, dtype=int)
    images = [seq[i] for i in indices]
    titles = [f"Frame {i}" for i in indices]
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ##############################################################################
# ############### AFFINE FUNCS ################################################

#  affine transformation matrix 
def compute_affine(tri1_pts, tri2_pts):
    ones = np.ones((3, 1))
    X = np.hstack([tri1_pts, ones])
    Y = tri2_pts
    A_matrix = np.linalg.lstsq(X, Y, rcond=None)[0].T
    return A_matrix

# affine transformation for triangular region
def apply_affine(src, src_tri, dst_tri, size):
    affine_mat = compute_affine(src_tri, dst_tri)
    affine_mat_inv = np.linalg.inv(np.vstack([affine_mat, [0, 0, 1]]))[:2, :]
    x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    coords = np.stack([x.flatten(), y.flatten(), np.ones(x.size)])
    src_coords = affine_mat_inv @ coords
    src_coords = src_coords[:2, :].reshape(2, size[1], size[0])
    warped_img = cv2.remap(src, src_coords[0].astype(np.float32), 
                           src_coords[1].astype(np.float32), cv2.INTER_LINEAR, 
                           borderMode=cv2.BORDER_REFLECT_101)
    return warped_img

# ##############################################################################
# ############### MORPH FUNCS ##################################################

def morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac):
    intermediate_pts = (1 - warp_frac) * im1_pts + warp_frac * im2_pts
    img_morphed = np.zeros_like(im1, dtype=np.float32)

    for t in tri.simplices:
        tri_pts1 = im1_pts[t]
        tri_pts2 = im2_pts[t]
        tri_intermediate = intermediate_pts[t]

        # bounding rectangle
        r = cv2.boundingRect(np.float32([tri_intermediate]))
        x_start, y_start, w, h = r

        tri_intermediate_rect = tri_intermediate - np.array([x_start, y_start])
        tri_pts1_rect = tri_pts1 - np.array([x_start, y_start])
        tri_pts2_rect = tri_pts2 - np.array([x_start, y_start])

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri_intermediate_rect), 1.0)

        # cropping
        img1_cropped = im1[y_start:y_start+h, x_start:x_start+w]
        img2_cropped = im2[y_start:y_start+h, x_start:x_start+w]

        size = (w, h)

        # warp and blend
        im1_warped = apply_affine(img1_cropped, tri_pts1_rect, 
                                  tri_intermediate_rect, size)
        im2_warped = apply_affine(img2_cropped, tri_pts2_rect, 
                                  tri_intermediate_rect, size)
        img_rect = (1 - dissolve_frac) * im1_warped + dissolve_frac * im2_warped

        # mask  blended triangle onto morphed image 
        img_morphed[y_start:y_start+h, x_start:x_start+w][mask > 0] = img_rect[mask > 0]

    return img_morphed.astype(np.uint8)

def gen_morph_seq(im1, im2, im1_pts, im2_pts, tri, num_frames=45):
    seq = []
    for i in range(num_frames + 1):
        frac = i / num_frames
        warp_frac = frac
        dissolve_frac = frac
        morphed_img = morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac)
        seq.append(morphed_img)
    return seq

def save_morph_seq(seq, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, frame in enumerate(seq):
        output_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# ##############################################################################
# ############### PART 4: MEAN FACE ############################################

def save_points(points, file_path):
    with open(file_path, 'w') as file:
        json.dump({'points': points.tolist()}, file)
    print(f"[INFO] Points saved to {file_path}")

def load_points(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return np.array(data['points'], dtype=np.float32)

def mean_face(images, points):
    num_images = len(images)
    h, w = images[0].shape[:2]
    mean_image = np.zeros((h, w, 3), dtype=np.float32)
    mean_points = np.zeros((points[0].shape[0], 2), dtype=np.float32)

    for img, pts in zip(images, points):
        mean_image += img
        mean_points += pts

    mean_image /= num_images
    mean_points /= num_images

    return mean_image.astype(np.uint8), mean_points

def morph2(source_img, source_pts, target_img, target_pts, tri, num_frames=30):
    seq = []
    for i in range(num_frames + 1):
        alpha = i / num_frames
        intermediate_pts = (1 - alpha) * source_pts + alpha * target_pts
        morphed_img = morph(source_img, target_img, source_pts, target_pts, tri, warp_frac=alpha, dissolve_frac=alpha)
        seq.append(morphed_img)
    return seq

# ##############################################################################
# ############### MAIN TESTING  ################################################

if __name__ == "__main__":

    ######### pts 1-3
    use_json = input("load correspondence points from JSON file? (y/n): ").strip().lower()
    if use_json == 'y':
        json_file_path = jsonn
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        pts1 = np.array(data['im1Points'], dtype=np.float32)
        pts2 = np.array(data['im2Points'], dtype=np.float32)

        # adjust
        if resized_A:
            pts1[:, 0] *= scale_A_x
            pts1[:, 1] *= scale_A_y
        if resized_B:
            pts2[:, 0] *= scale_B_x
            pts2[:, 1] *= scale_B_y

        # add corners
        print("adding corner points to the correspondence points.")
        h, w = image_A_rgb.shape[:2]
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        pts1 = np.vstack([pts1, corners])
        pts2 = np.vstack([pts2, corners])

        np.save('pts1.npy', pts1)
        np.save('pts2.npy', pts2)
    else:
        # case if manual
        num_points = int(input("enter the # of correspondence points: "))
        print("Select points for Image A.")
        pts1 = get_points(image_A_rgb, num_points)
        print("Select corresponding points for Image B.")
        pts2 = get_points(image_B_rgb, num_points)
        # add corner points #corners
        print("adding corner points to the correspondence points.")

        h, w = image_A_rgb.shape[:2]
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], 
                           dtype=np.float32)
        pts1 = np.vstack([pts1, corners])
        pts2 = np.vstack([pts2, corners])

        np.save('pts1.npy', pts1)
        np.save('pts2.npy', pts2)


    print("displaying original images...")
    display_imgs([image_A_rgb, image_B_rgb], ["Image A", "Image B"])

    average_pts = (pts1 + pts2) / 2.0
    tri = Delaunay(average_pts)
    vis_triangulation(image_A_rgb, pts1, tri) # triangulation on A
    vis_triangulation(image_B_rgb, pts2, tri) # triangulation on B
    mid_face = morph(image_A_rgb, image_B_rgb, pts1, pts2, tri, 0.5, 0.5)
    show_img(mid_face) # midface

    morph_seq = gen_morph_seq(image_A_rgb, image_B_rgb, pts1, pts2, tri, num_frames=45)
    display_progression(morph_seq, num_steps=6)

    output_dir = os.path.join(script_dir, 'morph_frames')
    save_morph_seq(morph_seq, output_dir)

    ######### Part 4: Mean Face

    # check f a points file already exists for this pop
    json_file_path = os.path.join(script_dir, 'population_points.json')
    if os.path.exists(json_file_path):
        overwrite = input("points file for population already exists. overwrite? (y/n): ").strip().lower()
        if overwrite == 'y':
            first_image = cv2.imread(population_images[0])
            first_image_rgb = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
            num_points = int(input("enter the # of correspondence points: "))
            pts = get_points(first_image_rgb, num_points)
            h, w = first_image_rgb.shape[:2]
            corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], 
                               dtype=np.float32)
            pts = np.vstack([pts, corners])
            save_points(pts, json_file_path)
        else:
            print("loading points from existing.")
            pts = load_points(json_file_path)
    else:
        first_image = cv2.imread(population_images[0])
        first_image_rgb = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
        num_points = int(input("enter the # of correspondence points: "))
        pts = get_points(first_image_rgb, num_points)
        h, w = first_image_rgb.shape[:2]
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], 
                           dtype=np.float32)
        pts = np.vstack([pts, corners])
        save_points(pts, json_file_path)

    population_points = [pts for _ in population_images]
    population_images_rgb = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
                             for img_path in population_images]

    # Compute the mean face
    mean_image, mean_pts = mean_face(population_images_rgb, population_points)
    show_img(mean_image, "Mean Face")

    # Perform Delaunay triangulation on the mean points
    tri = Delaunay(mean_pts)
    vis_triangulation(mean_image, mean_pts, tri, "Triangulation on Mean Face")

    print("Mean face computation complete.")

    ######### Morphing Image A to Mean Face
    print("Morphing Image A to Mean Face...")
    morph_seq_A_to_mean = morph2(image_A_rgb, pts, mean_image, mean_pts, tri)
    display_progression(morph_seq_A_to_mean, num_steps=6)

    ######### Morphing Mean Face to Image B
    print("Morphing Mean Face to Image B...")
    morph_seq_mean_to_B = morph2(mean_image, mean_pts, image_B_rgb, pts, tri)
    display_progression(morph_seq_mean_to_B, num_steps=6)



    print("completed.")
