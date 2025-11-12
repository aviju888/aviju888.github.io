import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_bilateral
import cv2

print("Select which part to execute:")
print("1. Part A (HDR Reconstruction)")
print("2. Part B (Local Tone Mapping)")
print("3. Both Part A and Part B")
choice = input("Enter 1, 2, or 3: ").strip()

def setup():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(script_dir, 'images')
    output_dir = os.path.join(script_dir, 'output')
    points_file = os.path.join(script_dir, 'points.json')

    os.makedirs(output_dir, exist_ok=True)
    img1_path = os.path.join(imgs_dir, 'lolm_1.jpg')
    img2_path = os.path.join(imgs_dir, 'lolm_3gt3.jpg')
    img3_path = os.path.join(imgs_dir, 'lolm_5.jpg')

    return script_dir, imgs_dir, output_dir, points_file, img1_path, img2_path, img3_path

def load_imgs(img1_path, img2_path, img3_path):
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    img3 = Image.open(img3_path).convert('RGB')

    img1_rgb = np.array(img1).astype(np.float32) / 255.0
    img2_rgb = np.array(img2).astype(np.float32) / 255.0
    img3_rgb = np.array(img3).astype(np.float32) / 255.0

    return img1_rgb, img2_rgb, img3_rgb

def display_imgs(imgs, titles, delay=0):
    # plt.figure(figsize=(15, 5))
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

def radiance_map(hdr_map):
    luminance = 0.299 * hdr_map[:, :, 2] + 0.587 * hdr_map[:, :, 1] + 0.114 * hdr_map[:, :, 0]
    plt.figure()
    plt.title("Log Radiance (Luminance)")
    plt.imshow(np.log(luminance + 1), cmap='jet')
    plt.colorbar(label='Log Radiance')
    plt.axis('off')
    plt.show()

def response_curve(g):
    Z = np.arange(256)
    plt.figure()
    plt.title("Recovered Camera Response Curve")
    plt.plot(Z, g, 'r-')
    plt.xlabel("Pixel Value (z)")
    plt.ylabel("Log Exposure")
    plt.grid(True)
    plt.show()

def decomposition(L, base, detail):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original (Log Domain)")
    plt.imshow(L, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Base Layer")
    plt.imshow(base, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Detail Layer")
    plt.imshow(detail, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_tonemapping(single_exposure, tonemapped_global, tonemapped_local):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Single Exposure (LDR)")
    plt.imshow(single_exposure)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Global Tone Mapping")
    plt.imshow(tonemapped_global)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Local Tone Mapping")
    plt.imshow(tonemapped_local)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def weight(z):
    z_mid = 127.5
    # return np.where(z <= z_mid, z + 1, 255 - z)
    return np.exp(-4 * ((z - 128) / 128) ** 2)

def solve_response_curve(Z, B, l, w):
    n = 256
    N_p = Z.shape[0]
    N_i = Z.shape[1]

    A = np.zeros((N_p * N_i + n + 1, n + N_p), dtype=np.float32)
    b = np.zeros((A.shape[0], 1), dtype=np.float32)

    k = 0
    for i in range(N_p):
        for j in range(N_i):
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, n + i] = -wij
            b[k, 0] = wij * B[j]
            k += 1

    A[k, 128] = 1.0
    b[k, 0] = 0.0
    k += 1

    l_val = l
    for z in range(1, n - 1):
        A[k, z - 1] = l_val * weight(z)
        A[k, z] = -2 * l_val * weight(z)
        A[k, z + 1] = l_val * weight(z)
        k += 1

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    g = x[:n].ravel()
    lE = x[n:].ravel()

    return g, lE

def construct_radiance_map(imgs, g_curves, ln_t, w):
    H, W, C = imgs[0].shape
    hdr = np.zeros((H, W, C), dtype=np.float32)

    for c in range(C):
        stack = np.array([im[:, :, c] for im in imgs])
        Z = (stack * 255).astype(np.uint8)
        Wt = w[Z]
        numerator = np.sum(Wt * (g_curves[c][Z] - ln_t[:, None, None]), axis=0)
        denominator = np.sum(Wt, axis=0) + 1e-8
        lnE = numerator / denominator
        hdr[:, :, c] = np.exp(lnE)

    return hdr

def global_tone_map(hdr):
    luminance = 0.299 * hdr[:, :, 2] + 0.587 * hdr[:, :, 1] + 0.114 * hdr[:, :, 0]
    # Ld = np.log10(1 + luminance) / np.log10(1 + np.max(luminance))
    # Ld = luminance / (1 + luminance) # hopefully fixes the brightness issues
    threshold = 0.6 * np.max(luminance)  # 60% of maximum luminance

    Ld = np.where(luminance < threshold, np.log10(1 + luminance), luminance / (1 + luminance))

 

    eps = 1e-8
    R = hdr[:, :, 2]
    G = hdr[:, :, 1]
    B = hdr[:, :, 0]
    R_out = Ld * (R / (luminance + eps))
    G_out = Ld * (G / (luminance + eps))
    B_out = Ld * (B / (luminance + eps))

    out = np.stack([B_out, G_out, R_out], axis=2)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)

    return out

def local_tone_map(hdr, base_contrast=1.0, sigma_s=0.01, sigma_r=0.2, downsample_factor=1.0):
    R, G, B = hdr[:, :, 2], hdr[:, :, 1], hdr[:, :, 0]
    I = (R + G + B) / 3.0
    eps = 1e-8
    I[I <= 0] = np.min(I[I > 0]) if np.any(I > 0) else 1e-5

    L = np.log2(I + eps)
    H, W = L.shape

    L_downsampled = cv2.resize(L, (int(W * downsample_factor), int(H * downsample_factor)))

    base_downsampled = denoise_bilateral(L_downsampled / 255.0, sigma_color=sigma_r, sigma_spatial=sigma_s)

    base = cv2.resize(base_downsampled, (W, H))
    detail = L - base

    o, mn = np.max(base), np.min(base)
    range_base = o - mn
    s = base_contrast / range_base
    base_scaled = (base - o) * s
    O = 2 ** (base_scaled + detail)

    R_out = O * (R / (I + eps))
    G_out = O * (G / (I + eps))
    B_out = O * (B / (I + eps))
    gamma = 0.6
    RGB_out = np.stack([B_out**gamma, G_out**gamma, R_out**gamma], axis=2)
    RGB_out = np.clip(RGB_out / np.max(RGB_out) * 255, 0, 255).astype(np.uint8)
    return RGB_out

def global_tone_map_simple(hdr):
    # gamma = 1 / 2.2
    gamma = 0.8
    max_val = np.max(hdr)
    normalized = hdr / (max_val + 1e-8)
    out = np.power(normalized, gamma)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out

if choice in ['1', '3']:
    print("\n### Running Part A: Global and Local Tone Mapping v1###")
    script_dir, imgs_dir, output_dir, points_file, img1_path, img2_path, img3_path = setup()
    img1_rgb, img2_rgb, img3_rgb = load_imgs(img1_path, img2_path, img3_path)
    display_imgs([img1_rgb, img2_rgb, img3_rgb], ["Image 1", "Image 2", "Image 3"])

    ldr_stack = [
        img1_rgb.astype(np.float32) / 255.0,
        img2_rgb.astype(np.float32) / 255.0,
        img3_rgb.astype(np.float32) / 255.0
    ]
    hdr_map = np.max(np.stack(ldr_stack, axis=0), axis=0)

    radiance_map(hdr_map)

    g = np.log(np.arange(256) + 1)
    g = g - g.mean()
    response_curve(g)

    R = hdr_map[:, :, 2]
    G = hdr_map[:, :, 1]
    B = hdr_map[:, :, 0]
    I = (R + G + B) / 3.0
    eps = 1e-8
    I[I <= 0] = np.min(I[I > 0]) if np.any(I > 0) else 1e-5
    L = np.log2(I + eps)

    base = gaussian_filter(L, sigma=5)
    detail = L - base
    decomposition(L, base, detail)

    single_exposure = img1_rgb
    global_result = global_tone_map_simple(hdr_map)
    local_result = local_tone_map(hdr_map, base_contrast=5.0, sigma_s=0.02, sigma_r=0.4)

    compare_tonemapping(single_exposure, global_result, local_result)
    print("Part A completed.\n")

if choice in ['2', '3']:
    print("\n### Running Part B: Local Tone Mapping v2###")
    script_dir, imgs_dir, output_dir, points_file, img1_path, img2_path, img3_path = setup()

    img_rgb = Image.open(img2_path).convert('RGB')
    img_rgb = np.array(img_rgb).astype(np.float32) / 255.0
    mapped = local_tone_map(img_rgb)

    mapped_img = Image.fromarray(mapped)
    mapped_img.save(os.path.join(output_dir, "local_tone_mapped.jpg"))
    print(f"Local tone mapped img saved to '{output_dir}/local_tone_mapped.jpg'.")

    img1 = Image.open(img1_path).convert('RGB')
    display_imgs([img1], [""])
    display_imgs([mapped], [""])
    print("Part B completed.\n")
