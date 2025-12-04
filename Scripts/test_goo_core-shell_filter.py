import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
datasets_path = os.path.join(SCRIPT_DIR, os.pardir, "an2dl2526c2")
train_data_path = os.path.join(datasets_path, "train_data")
SOURCE_FOLDER = train_data_path

TARGET_FILES = [
    "img_0229.png", 
    "img_0223.png",
    "img_0239.png",
    "img_0308.png"
]

USE_EXTERNAL_MASKS = False

CORE_LOWER = np.array([35, 100, 50]) 
CORE_UPPER = np.array([85, 255, 255])
SHELL_LOWER = np.array([30, 30, 30]) 
SHELL_UPPER = np.array([95, 255, 255])
MIN_GOO_AREA = 200

def get_smart_goo_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask_core = cv2.inRange(hsv, CORE_LOWER, CORE_UPPER)
    mask_shell = cv2.inRange(hsv, SHELL_LOWER, SHELL_UPPER)

    # Combine shell blobs that overlap with core
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_shell, connectivity=8)
    smart_mask = np.zeros_like(mask_core)
    
    for label_id in range(1, num_labels):
        blob_mask = (labels == label_id).astype(np.uint8) * 255
        overlap = cv2.bitwise_and(blob_mask, mask_core)
        if cv2.countNonZero(overlap) > 0:
            smart_mask = cv2.bitwise_or(smart_mask, blob_mask)
            
    # Fill holes and filter by area
    contours, _ = cv2.findContours(smart_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_filled_mask = np.zeros_like(smart_mask)
    for contour in contours:
        if cv2.contourArea(contour) > MIN_GOO_AREA:
            cv2.drawContours(final_filled_mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Dilate by 1 pixel
    kernel = np.ones((3, 3), np.uint8)
    final_expanded_mask = cv2.dilate(final_filled_mask, kernel, iterations=1)
    
    return final_expanded_mask

def show_final_goo_test(filename, folder, num_images, current_index):
    img_path = os.path.join(folder, filename)
    if not os.path.exists(img_path): 
        print(f"Image not found: {filename}")
        return

    # Load and convert image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    # Load external mask if enabled
    mask_found = False
    mask_status = "Disabled"

    if USE_EXTERNAL_MASKS:
        mask_name = filename.replace('img_', 'mask_', 1)
        mask_path = os.path.join(folder, mask_name)
        
        if not os.path.exists(mask_path):
            mask_stem = os.path.splitext(filename)[0].replace('img_', 'mask_', 1)
            mask_path = os.path.join(folder, mask_stem + ".png")

        if os.path.exists(mask_path):
            orig_mask = cv2.imread(mask_path, 0)
            if img.shape[:2] != orig_mask.shape[:2]:
                orig_mask = cv2.resize(orig_mask, (img.shape[1], img.shape[0]))
            
            _, binary_orig_mask = cv2.threshold(orig_mask, 127, 255, cv2.THRESH_BINARY)
            mask_status = "Mask Loaded"
            mask_found = True
        else:
            mask_status = "File Not Found (Fallback)"

    # Fallback: use full image if no mask
    if not mask_found:
        binary_orig_mask = np.full(img.shape[:2], 255, dtype=np.uint8)

    # Detect goo
    smart_goo_mask = get_smart_goo_mask(img)

    # Create overlay visualization
    overlay = img.copy()
    red_color = np.array([255, 0, 0], dtype=np.uint8)
    mask_indices = smart_goo_mask > 0
    overlay[mask_indices] = red_color
    
    alpha = 0.4 
    comparison_view = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Calculate final result
    final_clean_mask = cv2.bitwise_and(binary_orig_mask, cv2.bitwise_not(smart_goo_mask))
    final_clean_img = cv2.bitwise_and(img, img, mask=final_clean_mask)

    # Setup plots
    plot_base_index = current_index * 4 + 1
    
    # Plot original
    plt.subplot(num_images, 4, plot_base_index)
    plt.imshow(img)
    plt.title(f"Original: {filename}\n({mask_status})", fontsize=9)
    plt.axis('off')

    # Plot overlay
    plt.subplot(num_images, 4, plot_base_index + 1)
    plt.imshow(comparison_view)
    plt.title("Red = Removed", fontsize=9, color='red')
    plt.axis('off')

    # Plot zoomed view
    plt.subplot(num_images, 4, plot_base_index + 2)
    if np.any(mask_indices):
        y_indices, x_indices = np.where(mask_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        pad = 60 
        y_min = max(0, y_min - pad); y_max = min(img.shape[0], y_max + pad)
        x_min = max(0, x_min - pad); x_max = min(img.shape[1], x_max + pad)
        
        plt.imshow(comparison_view[y_min:y_max, x_min:x_max])
        plt.title("Zoom Check", fontsize=9)
    else:
        plt.imshow(np.zeros((100,100,3)))
        plt.title("No Goo Detected", fontsize=9)
    plt.axis('off')

    # Plot final result
    plt.subplot(num_images, 4, plot_base_index + 3)
    plt.imshow(final_clean_img)
    plt.title("Final Cleaned Result", fontsize=9)
    plt.axis('off')

if __name__ == "__main__":
    num_files = len(TARGET_FILES)
    print(f"Running Smart Goo Removal on {num_files} images...")
    print(f"External Mask Usage: {USE_EXTERNAL_MASKS}")
    
    plt.figure(figsize=(20, 5 * num_files))
    for i, f in enumerate(TARGET_FILES):
        show_final_goo_test(f, SOURCE_FOLDER, num_files, i)
    
    plt.tight_layout()
    plt.show()