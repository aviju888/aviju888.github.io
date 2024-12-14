# main.py

# ##############################################################################
# ############### IMPORT LIBRARIES ##############################################
# ##############################################################################
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ##############################################################################
# ############### CONFIGURATION #################################################
# ##############################################################################

# Define constants and configurations for the project
MAX_SHIFT = 10  # Maximum pixel shift for depth refocusing
# FOCUS_DEPTHS = np.linspace(0.5, 1.5, num=5)  # Depth values to refocus on
# FOCUS_DEPTHS = [-10, -5, 2.5, -1, -0.5 -0.25, 0.0, 0.25, 0.5, 1, 2.5, 5, 10]
FOCUS_DEPTHS = list(range(-15, 16))
APERTURE_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Aperture sizes to simulate (must be odd integers)

# ##############################################################################
# ############### HELPER FUNCTIONS ##############################################
# ##############################################################################

def load_imgs(folder_path):
    img_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    imgs = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in img_paths]
    print(f"Loaded {len(imgs)} imgs from {folder_path}.")
    return imgs

def get_grid_size(imgs):
    num_imgs = len(imgs)
    grid_size = int(np.sqrt(num_imgs))
    if grid_size * grid_size == num_imgs:
        print(f"Determined grid size: {grid_size}x{grid_size}.")
        return (grid_size, grid_size)
    else:
        print(f"({num_imgs}) is not a perfect square.")

def refocus_depth(imgs, grid_size, focus_depth, max_shift):
    shifted_imgs = []
    rows, cols = grid_size
    center_row = rows // 2
    center_col = cols // 2
    print(f"refocusing at depth {focus_depth:.2f} with max shift {max_shift}.")

    for idx, img in enumerate(imgs):
        u = idx % cols
        v = idx // cols 

        shift_x = (u - center_col) * focus_depth * max_shift / cols
        shift_y = (v - center_row) * focus_depth * max_shift / rows

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        shifted_imgs.append(shifted)
        print(f"Shifted img {idx+1}/{len(imgs)} by ({shift_x:.2f}, {shift_y:.2f}) pixels.")

    refocused_img = np.mean(shifted_imgs, axis=0).astype(np.uint8)
    print("Averaged shifted imgs to obtain refocused img.")
    return refocused_img

def adjust_aperture(imgs, grid_size, aperture_size):
    radius = aperture_size // 2
    rows, cols = grid_size
    center_row = rows // 2
    center_col = cols // 2
    selected_imgs = []
    print(f"Adjusting aperture to size {aperture_size}x{aperture_size}.")
    # print(f"Adjusting aperture to size {aperture_size}x{aperture_ize}.")

    for idx, img in enumerate(imgs):
        u = idx % cols
        v = idx // cols 
        if (center_col - radius <= u <= center_col + radius) and (center_row - radius <= v <= center_row + radius):
            selected_imgs.append(img)
            print(f"Selected img {idx+1}/{len(imgs)} for aperture.")

    if not selected_imgs:
        raise ValueError("No imgs selected for the given aperture size.")

    aperture_img = np.mean(selected_imgs, axis=0).astype(np.uint8)
    print("Averaged selected imgs to obtain aperture-adjusted img.")
    return aperture_img

def save_img(img, filename):
    plt.imsave(filename, img)
    print(f"Saved img to {filename}.")

def show_imgs(imgs, titles, delay=0):
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
    plt.tight_layout()
    if delay > 0:
        plt.show(block=False)
        plt.pause(delay)
        plt.close()
    else:
        plt.show()
    print("Displayed imgs.")

# ##############################################################################
# ############### PART 1: DEPTH REFOCUSING ######################################
# ##############################################################################

def part1_depth_refocusing():

    # Part 1: brefocus the lightfield imgs at different depths

    print("\n=== Part 1: Depth Refocusing ===")
    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(script_dir, 'images/proj6ba')
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Imgs directory: {imgs_dir}")
    print(f"Output directory: {output_dir}")


    imgs = load_imgs(imgs_dir)
    grid_size = get_grid_size(imgs)

    # its time
    for depth in FOCUS_DEPTHS:
        refocused_img = refocus_depth(imgs, grid_size, depth, MAX_SHIFT)
        filename = os.path.join(output_dir, f'refocused_depth_{depth:.2f}.png')
        save_img(refocused_img, filename)
        print(f"Refocused img at depth {depth:.2f} saved to {filename}.")

    # Display
    print("\nDisplaying sample refocused imgs:")
    sample_imgs = []
    sample_titles = []
    for depth in [FOCUS_DEPTHS[0], FOCUS_DEPTHS[-1]]:  # Display first and last samples
        refocused_img = refocus_depth(imgs, grid_size, depth, MAX_SHIFT)
        sample_imgs.append(refocused_img)
        sample_titles.append(f'Focus Depth {depth:.2f}')
    show_imgs(sample_imgs, sample_titles)

    print("=== Part 1 Done ===\n")

# ##############################################################################
# ############### PART 2: APERTURE ADJUSTMENT ###################################
# ##############################################################################

def part2_aperture_adjustment():
    # Part 2: adjust the aperture size by avg different subsets of imgs and save result
    print("\n=== Part 2: Aperture Adjustment ===")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(script_dir, 'images/proj6ba')
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Imgs directory: {imgs_dir}")
    print(f"Output directory: {output_dir}")
    imgs = load_imgs(imgs_dir)
    grid_size = get_grid_size(imgs)

    for aperture_size in APERTURE_SIZES:
        aperture_img = adjust_aperture(imgs, grid_size, aperture_size)
        filename = os.path.join(output_dir, f'aperture_size_{aperture_size}.png')
        save_img(aperture_img, filename)
        print(f"Aperture-adjusted img with size {aperture_size}x{aperture_size} saved to {filename}.")

    print("\nDisplaying sample aperture-adjusted imgs:")
    sample_imgs = []
    sample_titles = []
    for aperture_size in APERTURE_SIZES[:2]: 
        aperture_img = adjust_aperture(imgs, grid_size, aperture_size)
        sample_imgs.append(aperture_img)
        sample_titles.append(f'Aperture Size {aperture_size}x{aperture_size}')
    show_imgs(sample_imgs, sample_titles)

    print("=== Part 2 Done ===\n")

# ##############################################################################
# ############### MAIN EXECUTION #################################################
# ##############################################################################

def main():
    print("Select which part to execute:")
    print("1. Part 1 (Depth Refocusing)")
    print("2. Part 2 (Aperture Adjustment)")
    print("3. Both Part 1 and Part 2")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == '1':
        part1_depth_refocusing()
    elif choice == '2':
        part2_aperture_adjustment()
    elif choice == '3':
        part1_depth_refocusing()
        part2_aperture_adjustment()
    else:
        print("Invalid. Exiting.")

if __name__ == "__main__":
    main()
