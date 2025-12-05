# Image Preprocessing Pipeline

This document describes the comprehensive image preprocessing pipeline implemented in `v1.ipynb` for the ANN Challenge 2 dataset.

## Overview

The preprocessing pipeline transforms raw training images with masks into clean, normalized images ready for deep learning model training. The process involves masking, green artifact removal, quality filtering, and data preparation.

---

## Pipeline Stages

### 1. **Initial Masking & Resizing**

**Functions:** `apply_mask()`, `process_batch()`

- Loads raw images (`img_xxxx`) and corresponding binary masks (`mask_xxxx`)
- Resizes both to **224×224** pixels using appropriate interpolation:
  - Images: Linear interpolation (better for photos)
  - Masks: Nearest neighbor (preserves sharp edges)
- Applies binary threshold to masks (values >127 → white, ≤127 → black)
- Uses `cv2.bitwise_and()` to apply mask and remove background
- Saves processed images to `train_masked/` folder

**Technical Details:**
```python
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
masked_img = cv2.bitwise_and(img, img, mask=binary_mask)
```

**Output:** ~1,412 masked images normalized to [0, 1] range in RGB format

---

### 2. **Green Artifact Analysis**

**Function:** `analyze_green_frequency()`

Before filtering, we analyzed green channel distribution across all images to identify those with significant green screen artifacts.

**Metrics Calculated:**
- **High Green Percentage:** % of pixels with green intensity >0.5
- **Dominant Green Percentage:** % of pixels where G > R and G > B

**Analysis Results:**
- Identified 171 images with >20% high green intensity
- Used histogram analysis to compare pixel distributions
- Examined images 304, 306, 308 for reference cases

**Key Insight:** Images with bright green backgrounds showed strong green channel peaks in 0.6-1.0 intensity range.

---

### 3. **Green Screen Removal Filter**

**Function:** `filter_bright_green_areas()`

Removes bright green screen artifacts using HSV color space filtering with morphological operations.

#### Algorithm Steps:

1. **Color Space Conversion**
   - Convert RGB → BGR → HSV
   - HSV separates color (Hue) from brightness (Value)

2. **Primary Green Detection**
   - Define HSV range for bright green:
     - Hue: 10-110° (green spectrum)
     - Saturation: 10-255
     - Value: 50-255
   - Create binary mask using `cv2.inRange()`

3. **Morphological Cleaning**
   - Apply MORPH_OPEN with 5×5 elliptical kernel (2 iterations)
   - Removes small noise and isolated pixels

4. **Mask Expansion (Dilation)**
   - Dilate mask by 2 iterations (configurable)
   - Catches edge artifacts and boundary pixels

5. **Subtle Green Detection**
   - Create secondary mask with expanded range (±10 HSV tolerance)
   - Detects faint green reflections near main green areas
   - Combines with primary mask using `cv2.bitwise_or()`

6. **Mask Application**
   - Invert combined mask to preserve non-green areas
   - Apply to original image, replacing green with black

#### Parameters:
```python
filter_bright_green_areas(
    image,
    lg_H=10,          # Lower Hue bound
    lg_S=10,          # Lower Saturation bound
    lg_V=50,          # Lower Value bound
    ug_H=110,         # Upper Hue bound
    ug_S=255,         # Upper Saturation bound
    ug_V=255,         # Upper Value bound
    dilate_iterations=2  # Edge expansion
)
```

**Applied to:** All 1,412 training images

---

### 4. **Quality Filtering**

**Threshold:** Mean intensity ≤ 0.001

After green removal, some images became nearly black (failed mask or complete green screen). These low-quality images were filtered out.

**Process:**
```python
MEAN_LIMIT = 0.001
for idx in range(len(train_images)):
    if train_images[idx].mean() <= MEAN_LIMIT:
        # Eliminate image
```

**Results:**
- Eliminated images stored in `tot_eliminated_imgs` array
- Remaining high-quality images kept in `train_images`
- Visualization of eliminated images confirms correct filtering

---

### 5. **Label Alignment & Data Splitting**

**Label Synchronization:**
- Loaded labels from `train_labels.csv`
- Matched filenames to image indices
- Removed images without corresponding labels
- Encoded string labels to integers using `LabelEncoder`

**Train/Val/Test Split:**
- **Train:** 80% of data
- **Validation:** 10% of data (50% of remaining 20%)
- **Test:** 10% of data (50% of remaining 20%)
- Used **stratified splitting** to maintain class distribution

**Final Shapes:**
- Training: 1,129 images
- Validation: 141 images
- Test: 142 images

---

### 6. **PyTorch Dataset Preparation**

**Transformations Applied:**
- Convert NumPy arrays to PyTorch tensors
- Permute dimensions: (N, H, W, C) → (N, C, H, W)
- Create `TensorDataset` objects for train/val/test
- Wrap in `DataLoader` with optimizations:
  - Batch size: 32
  - Pin memory for faster GPU transfer
  - 2-4 worker processes for parallel loading
  - Prefetch factor: 4 batches ahead

**Output Format:**
- Input shape: (3, 224, 224) - RGB channels first
- Labels: Integer class indices
- Ready for CNN model training

---

## Key Design Decisions

### Why HSV for Green Removal?
- HSV separates color from brightness, making it easier to target specific hues
- More robust to lighting variations than RGB thresholding
- Standard approach in chroma key/green screen removal

### Why Morphological Operations?
- **Opening (erosion→dilation):** Removes small noise while preserving larger structures
- **Dilation:** Expands mask to catch subtle green pixels at edges
- **Elliptical kernel:** Better follows curved object boundaries than rectangular

### Why Dual-Mask Strategy?
- Primary mask catches obvious bright green areas
- Subtle mask catches faint green reflections and edge artifacts
- Combined approach removes residuals missed by single-pass filtering

### Why Filter by Mean Intensity?
- Images with mean ≤0.001 are essentially black (no useful information)
- Likely result of complete green screen removal or failed masking
- Removing them prevents training on corrupted data

---

## Validation & Quality Checks

Throughout the pipeline, we performed visual inspections:

1. **Histogram Analysis:** Compared pixel intensity distributions before/after filtering
2. **Grid Visualization:** Displayed sample images at each stage (10-20 images)
3. **Channel Statistics:** Monitored R/G/B channel means and standard deviations
4. **Eliminated Images:** Verified filtered-out images were indeed low quality

**Channel Statistics After Filtering (Example):**
- Image 304: Green mean reduced from 0.6+ to <0.1
- Image 308: Maintained balance across R/G/B channels
- Image 306: Already clean, minimal change

---

## Pipeline Summary

```
Raw Images (train_data/)
    ↓
[1] Apply Binary Masks + Resize to 224×224
    ↓
Masked Images (train_masked/) - 1,412 images
    ↓
[2] Analyze Green Channel Distribution
    ↓
[3] HSV-Based Green Artifact Removal
    ↓
Filtered Images - Green artifacts removed
    ↓
[4] Quality Filter (mean > 0.001)
    ↓
Clean Images - High quality subset
    ↓
[5] Label Alignment + Stratified Split
    ↓
Train/Val/Test Sets
    ↓
[6] PyTorch DataLoaders (batch=32)
    ↓
Ready for Model Training
```

---

## Files & Functions Reference

| Function | Purpose | Key Parameters |
|----------|---------|----------------|
| `apply_mask()` | Apply binary mask to single image | `target_size=(224,224)` |
| `process_batch()` | Batch process all images with masks | `input_dir`, `output_dir` |
| `load_images_from_folder()` | Load and normalize images | Returns RGB, [0,1] range |
| `analyze_green_frequency()` | Compute green channel metrics | Returns 2 percentages |
| `filter_bright_green_areas()` | Remove green screen artifacts | HSV bounds, `dilate_iterations` |
| `make_loader()` | Create optimized DataLoader | `batch_size`, `shuffle`, `drop_last` |

---

## Dependencies

- **OpenCV (cv2):** Image processing, color space conversion, morphological operations
- **NumPy:** Array operations, numerical computations
- **Matplotlib:** Visualization, histogram plotting
- **Pandas:** Label management, DataFrame operations
- **PyTorch:** Dataset creation, DataLoader optimization
- **scikit-learn:** Label encoding, train/test splitting
- **tqdm:** Progress bars for long operations

---

## Results

**Final Dataset:**
- **Input:** 1,412 raw masked images
- **After Green Removal:** All images processed
- **After Quality Filter:** ~1,400+ high-quality images (exact count varies)
- **Splits:** 80/10/10 train/val/test with stratification
- **Format:** (3, 224, 224) tensors, normalized [0, 1], integer labels

**Image Quality:**
- Green screen artifacts successfully removed
- Object boundaries preserved
- Black background maintained for masked regions
- Consistent size and normalization across dataset

---

## Future Improvements

Potential enhancements to consider:

1. **Adaptive Green Filtering:** Adjust HSV ranges per image based on histogram analysis
2. **Data Augmentation:** Add rotation, flip, color jitter to training set
3. **Edge Smoothing:** Apply Gaussian blur to mask boundaries to reduce harsh edges
4. **Advanced Filtering:** Use contour analysis to preserve only largest objects
5. **Metadata Tracking:** Log which images were filtered and why for reproducibility

---

## Conclusion

This preprocessing pipeline successfully transforms raw masked images into a clean, standardized dataset ready for deep learning. The multi-stage approach (masking → green removal → quality filtering → data preparation) ensures high-quality inputs while maintaining reproducibility and allowing for visual validation at each step.
