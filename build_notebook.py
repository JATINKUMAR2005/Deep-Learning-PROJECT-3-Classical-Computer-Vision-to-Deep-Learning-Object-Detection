import nbformat as nbf
import json
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.0"
        }
    }

    cells = []

    # Title cell
    cells.append(nbf.v4.new_markdown_cell("""# Red & White Skill Education
## Deep Learning — PR 3: Classical Computer Vision to Deep Learning Object Detection
**Student Name:** Jatin Kumar  
**Subject:** Deep Learning  
**Project:** PR 3  
**Topics Covered:**
1. Morphological Operations (Erosion, Dilation, Opening, Closing, Kernel Variations)
2. Bitwise Operations (AND, OR, XOR, NOT, Masked Overlay) & Image Histograms (Color & Gray, Brightness/Contrast)
3. Face Detection with YuNet (OpenCV 5.0, Landmarks, Live Webcam, Privacy Blurring)
4. Object Detection with YOLOv8 (COCO Classes, Confidence/IoU Tuning, Live Webcam, Per-Class Summary)
5. Integrated Real-Time Live Webcam Pipeline & FPS Benchmarking
"""))

    # -------------------------------------------------------------
    # TASK 1
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Task 1: Environment Setup, Image I/O & Morphological Operations

> **Objective:** Set up required libraries, load images, perform thresholding, and understand how morphological filters (erosion, dilation, opening, closing) clean noise and alter binary shapes.  
> **Expected Output:** Library versions printed, grayscale and color images displayed with shapes, and visual grids comparing the 4 morphological operations and various kernel sizes/shapes.
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.1 Environment Setup & Image Loading
We will import OpenCV (`cv2`), NumPy, and Matplotlib. We check the OpenCV version and define a simple helper function `show()` that correctly displays color (RGB) and grayscale images inline in Jupyter.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.1 Imports and Environment Verification
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

print(f"OpenCV Version : {cv2.__version__}")
print(f"NumPy Version  : {np.__version__}")

# Helper function to display images inline using matplotlib
def show(img, title="Image", cmap=None, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    if len(img.shape) == 2:  # Grayscale image
        plt.imshow(img, cmap=cmap if cmap else 'gray')
    else:  # Color image: convert BGR (OpenCV format) to RGB (Matplotlib format)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.show()

# Ensure directories exist
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load a grayscale and a color image
gray_img = cv2.imread("data/images/lena.jpg", cv2.IMREAD_GRAYSCALE)
color_img = cv2.imread("data/images/lena.jpg", cv2.IMREAD_COLOR)

print(f"Grayscale Image -> Shape: {gray_img.shape}, Data Type: {gray_img.dtype}")
print(f"Color Image     -> Shape: {color_img.shape}, Data Type: {color_img.dtype}")

# Display both images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_img, cmap='gray')
axes[0].set_title("Grayscale Image (lena.jpg)", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
axes[1].set_title("Color Image (RGB)", fontweight='bold')
axes[1].axis('off')
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.2 Binarize a Test Image
Morphological operations work on binary images (black and white pixels only). Here, we convert an image to grayscale and apply binary thresholding using `cv2.threshold()`. Pixels brighter than 127 become 255 (white), and others become 0 (black).
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.2 Binarization with cv2.threshold
# Convert image to binary using a threshold of 127
threshold_value = 127
max_value = 255
ret, binary_img = cv2.threshold(gray_img, threshold_value, max_value, cv2.THRESH_BINARY)

# Display Original Grayscale vs Binary
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_img, cmap='gray')
axes[0].set_title("Original Grayscale", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(binary_img, cmap='gray')
axes[1].set_title("Binary Thresholded (Threshold = 127)", fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.3 Apply the Four Morphological Operations
We define a $5 \\times 5$ rectangular kernel of ones. We apply:
1. **Erosion (`cv2.erode`)**: Shrinks white regions.
2. **Dilation (`cv2.dilate`)**: Expands white regions.
3. **Opening (`MORPH_OPEN`)**: Erosion followed by Dilation (removes small bright noise).
4. **Closing (`MORPH_CLOSE`)**: Dilation followed by Erosion (fills small dark holes).
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.3 The Four Fundamental Morphological Operations
# Define a 5x5 rectangular structuring element (kernel)
kernel_5x5 = np.ones((5, 5), np.uint8)

# Apply operations
eroded = cv2.erode(binary_img, kernel_5x5, iterations=1)
dilated = cv2.dilate(binary_img, kernel_5x5, iterations=1)
opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_5x5)
closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_5x5)

# Plot 1x5 grid
titles = ["Original Binary", "Erosion (Shrink)", "Dilation (Grow)", "Opening (Erode->Dilate)", "Closing (Dilate->Erode)"]
images = [binary_img, eroded, dilated, opened, closed]

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.4 Kernel Shape & Size Comparison
We create a noisy binary image with artificial salt-and-pepper noise. Then, we test 3 different kernel shapes (`MORPH_RECT`, `MORPH_ELLIPSE`, `MORPH_CROSS`) across 3 different kernel sizes ($3\\times3$, $5\\times5$, $9\\times9$).
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.4 Kernel Shape and Size Grid on a Noisy Image
# Create artificial salt-and-pepper noise on the binary image
np.random.seed(42)
noisy_bin = binary_img.copy()
salt = np.random.rand(*noisy_bin.shape) < 0.04
pepper = np.random.rand(*noisy_bin.shape) < 0.04
noisy_bin[salt] = 255
noisy_bin[pepper] = 0

shapes = [
    ("Rectangular (MORPH_RECT)", cv2.MORPH_RECT),
    ("Elliptical (MORPH_ELLIPSE)", cv2.MORPH_ELLIPSE),
    ("Cross (MORPH_CROSS)", cv2.MORPH_CROSS)
]
sizes = [(3, 3), (5, 5), (9, 9)]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
plt.suptitle("Morphological Opening: Kernel Shape vs. Size Comparison", fontsize=15, fontweight='bold', y=0.95)

for i, (shape_name, shape_code) in enumerate(shapes):
    for j, (kw, kh) in enumerate(sizes):
        k = cv2.getStructuringElement(shape_code, (kw, kh))
        cleaned = cv2.morphologyEx(noisy_bin, cv2.MORPH_OPEN, k)
        axes[i, j].imshow(cleaned, cmap='gray')
        axes[i, j].set_title(f"{shape_name.split()[0]} ({kw}x{kh})", fontsize=11, fontweight='bold')
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig("plots/morphology_grid.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved comparison plot to plots/morphology_grid.png")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.5 WHY Morphology Works (Simple Explanation)

* **Erosion (Local Minimum Filter):**  
  A pixel is kept white (1) only if **all** pixels under the kernel are white. If any pixel under the kernel is black, the output pixel becomes black. This *shrinks* white objects and completely eliminates small isolated bright noise spots.
  
* **Dilation (Local Maximum Filter):**  
  A pixel turns white (1) if **at least one** pixel under the kernel is white. This *expands* white objects and fills in small dark cracks and holes.

* **Opening (Erosion followed by Dilation):**  
  First erodes away small bright noise dots, then dilates back the remaining larger structures so their original shape and size are preserved.
  
* **Closing (Dilation followed by Erosion):**  
  First expands structures to bridge narrow dark gaps and fill internal holes, then erodes back to maintain the original outer boundaries.

* **Best Combination for Noise Removal:**  
  A **$3\\times3$ or $5\\times5$ Elliptical or Rectangular kernel with Morphological Opening** removes salt-and-pepper noise effectively without rounding corners or eating away real object contours. A $9\\times9$ kernel is too aggressive and destroys fine details.

* **Real-World Scenario:**  
  In an automated license plate recognition or document OCR system, raw binarization often contains speckle noise from dirt or uneven camera lighting. Applying **Morphological Opening** cleans up isolated speckles before passing the mask to character segmentation.
"""))

    # -------------------------------------------------------------
    # TASK 2
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Task 2: Bitwise Operations & Image Histograms

> **Objective:** Master bitwise logic (AND, OR, XOR, NOT) for masking and compositing, and use image histograms to diagnose and adjust image brightness and contrast.  
> **Expected Output:** Bitwise logic operations grid, a masked watermark overlay demo, color & grayscale histograms, and stacked before/after contrast adjustment comparisons.
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.1 Bitwise AND / OR / XOR / NOT
Bitwise operations work pixel-by-pixel using boolean logic on binary masks:
- **AND**: Output is white only where **both** masks are white.
- **OR**: Output is white where **either** mask is white.
- **XOR**: Output is white where **exactly one** mask is white (not both).
- **NOT**: Inverts the image (white becomes black, black becomes white).
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.1 Bitwise AND, OR, XOR, NOT Demonstration
# Create two blank black canvases
mask_rect = np.zeros((300, 300), dtype=np.uint8)
mask_circle = np.zeros((300, 300), dtype=np.uint8)

# Draw white rectangle and white circle
cv2.rectangle(mask_rect, (40, 40), (220, 220), 255, -1)
cv2.circle(mask_circle, (180, 180), 90, 255, -1)

# Apply bitwise operations
bitwise_and = cv2.bitwise_and(mask_rect, mask_circle)
bitwise_or = cv2.bitwise_or(mask_rect, mask_circle)
bitwise_xor = cv2.bitwise_xor(mask_rect, mask_circle)
bitwise_not = cv2.bitwise_not(mask_rect)

# Plot a 2x3 grid
bitwise_imgs = [mask_rect, mask_circle, bitwise_and, bitwise_or, bitwise_xor, bitwise_not]
bitwise_titles = ["Mask 1: Rectangle", "Mask 2: Circle", "Bitwise AND", "Bitwise OR", "Bitwise XOR", "Bitwise NOT (Rect)"]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, img, title in zip(axes.flatten(), bitwise_imgs, bitwise_titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.2 Applied Use-Case: Masked Region Extraction & Logo Overlay
We demonstrate two practical industry use cases:
1. **Region of Interest (ROI) Extraction:** Using a circular mask with `bitwise_and` to crop a specific area.
2. **Logo Watermark Overlay:** Placing an OpenCV logo seamlessly onto a background photo using bitwise masking.
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.2 Masked ROI Extraction & Watermark Overlay
photo = cv2.imread("data/images/messi5.jpg")
h, w, _ = photo.shape

# 1. Circular Mask ROI Extraction
circular_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(circular_mask, (w // 2, h // 2), 100, 255, -1)
extracted_roi = cv2.bitwise_and(photo, photo, mask=circular_mask)

# 2. Logo Overlay on Background Photo
logo = cv2.imread("data/images/opencv-logo.png")
logo = cv2.resize(logo, (120, 120))
logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

# Create mask of logo (non-black pixels)
_, mask = cv2.threshold(logo_gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Place logo at top-left of photo
roi = photo[10:130, 10:130]
# Black-out the area of logo in ROI
bg_cleared = cv2.bitwise_and(roi, roi, mask=mask_inv)
# Take only region of logo from logo image
fg_logo = cv2.bitwise_and(logo, logo, mask=mask)
# Combine ROI and logo
composite_roi = cv2.add(bg_cleared, fg_logo)

# Put back into main image
overlay_result = photo.copy()
overlay_result[10:130, 10:130] = composite_roi

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Photo", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(extracted_roi, cv2.COLOR_BGR2RGB))
axes[1].set_title("Circular Mask Extraction (bitwise_and)", fontweight='bold')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(overlay_result, cv2.COLOR_BGR2RGB))
axes[2].set_title("Seamless Logo Overlay (bitwise masking)", fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.3 Grayscale & Per-Channel Colour Histograms
An image histogram counts how many pixels exist at each intensity level from 0 (pure black) to 255 (pure white).
We compute the grayscale histogram and overlay all 3 color channels (Blue, Green, Red) on a single plot.
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.3 Grayscale and Per-Channel Color Histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Grayscale Histogram
gray_sample = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
gray_hist = cv2.calcHist([gray_sample], [0], None, [256], [0, 256])

axes[0].plot(gray_hist, color='black', linewidth=2)
axes[0].set_title("Grayscale Histogram", fontweight='bold', fontsize=12)
axes[0].set_xlabel("Pixel Intensity (0 - 255)")
axes[0].set_ylabel("Pixel Count")
axes[0].set_xlim([0, 256])
axes[0].grid(True, linestyle='--', alpha=0.6)

# 2. Overlaid BGR Color Histograms
colors = ('b', 'g', 'r')
channel_names = ('Blue Channel', 'Green Channel', 'Red Channel')

for i, col in enumerate(colors):
    hist_channel = cv2.calcHist([photo], [i], None, [256], [0, 256])
    axes[1].plot(hist_channel, color=col, linewidth=2, label=channel_names[i])

axes[1].set_title("Per-Channel Color Histograms (B, G, R)", fontweight='bold', fontsize=12)
axes[1].set_xlabel("Pixel Intensity (0 - 255)")
axes[1].set_ylabel("Pixel Count")
axes[1].set_xlim([0, 256])
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.4 Brightness & Contrast Adjustment
Using OpenCV's `cv2.convertScaleAbs(img, alpha, beta)`:
$$\\text{new\\_pixel} = \\alpha \\cdot \\text{pixel} + \\beta$$
* $\\alpha > 1.0$: Increases contrast (widens histogram).
* $\\alpha < 1.0$: Decreases contrast (narrows histogram).
* $\\beta > 0$: Increases brightness (shifts histogram right).
* $\\beta < 0$: Decreases brightness (shifts histogram left).
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.4 Brightness and Contrast Adjustments with Histograms
# 1. Higher Contrast & Bright: alpha=1.5, beta=30
bright_img = cv2.convertScaleAbs(photo, alpha=1.5, beta=30)
bright_gray = cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY)

# 2. Lower Contrast & Dark: alpha=0.7, beta=-30
dark_img = cv2.convertScaleAbs(photo, alpha=0.7, beta=-30)
dark_gray = cv2.cvtColor(dark_img, cv2.COLOR_BGR2GRAY)

# Plot 2x3 Grid: Images on top, Histograms on bottom
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
plt.suptitle("Brightness & Contrast Adjustments with Grayscale Histograms", fontsize=15, fontweight='bold')

# Images
axes[0, 0].imshow(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original Image (alpha=1.0, beta=0)", fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("High Contrast & Bright (alpha=1.5, beta=30)", fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(dark_img, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("Low Contrast & Dark (alpha=0.7, beta=-30)", fontweight='bold')
axes[0, 2].axis('off')

# Histograms
hist_orig = cv2.calcHist([gray_sample], [0], None, [256], [0, 256])
hist_bright = cv2.calcHist([bright_gray], [0], None, [256], [0, 256])
hist_dark = cv2.calcHist([dark_gray], [0], None, [256], [0, 256])

axes[1, 0].plot(hist_orig, color='blue', linewidth=2)
axes[1, 0].set_title("Original Histogram", fontweight='bold')
axes[1, 0].set_xlim([0, 256])
axes[1, 0].set_xlabel("Pixel Intensity")
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

axes[1, 1].plot(hist_bright, color='green', linewidth=2)
axes[1, 1].set_title("Stretched & Shifted Right (Bright)", fontweight='bold')
axes[1, 1].set_xlim([0, 256])
axes[1, 1].set_xlabel("Pixel Intensity")
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

axes[1, 2].plot(hist_dark, color='purple', linewidth=2)
axes[1, 2].set_title("Compressed & Shifted Left (Dark)", fontweight='bold')
axes[1, 2].set_xlim([0, 256])
axes[1, 2].set_xlabel("Pixel Intensity")
axes[1, 2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("plots/histogram_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved histogram comparison to plots/histogram_comparison.png")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.5 WHY Alpha/Beta and Histograms Matter (Simple Explanation)

* **Understanding the Math:**  
  $$\\text{Output Pixel} = \\alpha \\cdot \\text{Input Pixel} + \\beta$$
  - **$\\alpha$ (Contrast Gain):** Multiplies pixel differences from the mean. Values $> 1$ widen the histogram spread (creating punchier, clearer contrast). Values $< 1$ squash the histogram into a narrow band (making the image look flat and gray).
  - **$\\beta$ (Brightness Bias):** Adds a constant offset to every pixel. Positive values slide the entire histogram curve to the right (brighter), while negative values slide it to the left (darker).

* **Diagnosing Poor Images with Histograms:**  
  1. **Narrow Histogram:** Indicates low contrast (hazy or foggy scene).  
  2. **Histogram Clipped at 0:** Indicates underexposure (lost shadow details).  
  3. **Histogram Clipped at 255:** Indicates overexposure (blown-out highlights).  

* **Why It Matters for Deep Learning:**  
  Deep neural networks (like face detectors and YOLO) rely on clear edge gradients and textures. If an input frame from a surveillance camera is underexposed or low contrast, normalizing and adjusting $\\alpha$ and $\\beta$ before feeding the frame into the detector drastically boosts detection accuracy and reduces missed detections.
"""))

    # -------------------------------------------------------------
    # TASK 3
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Task 3: Face Detection with YuNet (OpenCV 5.0)

> **Objective:** Load the modern CNN-based YuNet face detector in OpenCV 5.0, detect faces and 5 facial landmarks on static images, stream live detection from your webcam with live FPS, and build a privacy face-blurring filter.  
> **Expected Output:** Bounding boxes and landmarks drawn on images, live real-time webcam face tracking window (press 'q' to exit), threshold comparisons (0.5 vs 0.7 vs 0.9), and privacy face-blurring before/after.
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.1 Load the YuNet Face Detector
YuNet is OpenCV's official, lightweight deep-learning face detector (`cv2.FaceDetectorYN`). It requires an ONNX weight file (`face_detection_yunet_2023mar.onnx`) and predicts bounding boxes alongside 5 facial landmarks in one single forward pass.
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.1 Initialize the YuNet Face Detector
yunet_model_path = "data/models/face_detection_yunet_2023mar.onnx"

# Verify model file exists
assert os.path.exists(yunet_model_path), f"YuNet model not found at {yunet_model_path}"

# Create the YuNet detector
# Parameters: model_path, config, input_size=(w, h), score_threshold, nms_threshold, top_k
dummy_w, dummy_h = 300, 300
yunet_detector = cv2.FaceDetectorYN.create(
    yunet_model_path,
    "",
    (dummy_w, dummy_h),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

print("YuNet Face Detector successfully loaded and initialized!")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.2 Detect Faces on Static Images
We test YuNet on 3 diverse static images (`lena.jpg`, `messi5.jpg`, `baboon.jpg`). For each face found, we parse:
- Bounding Box: $(x, y, \\text{width}, \\text{height})$
- 5 Facial Landmarks: Right Eye, Left Eye, Nose Tip, Right Mouth Corner, Left Mouth Corner
- Detection Confidence Score
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.2 Face Detection on Static Images
test_face_images = ["data/images/lena.jpg", "data/images/messi5.jpg", "data/images/baboon.jpg"]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
plt.suptitle("YuNet Face Detection & 5 Facial Landmarks (OpenCV 5.0)", fontsize=15, fontweight='bold')

for idx, img_path in enumerate(test_face_images):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # Set input size to match image dimensions
    yunet_detector.setInputSize((w, h))
    _, faces = yunet_detector.detect(img)
    
    annotated = img.copy()
    face_count = 0
    if faces is not None:
        face_count = len(faces)
        for face in faces:
            # Bounding box
            fx, fy, fw, fh = map(int, face[:4])
            score = face[-1]
            
            # Draw green rectangle for face box
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            cv2.putText(annotated, f"Face: {score*100:.1f}%", (fx, max(20, fy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            
            # Draw red landmark circles: Right Eye, Left Eye, Nose, Right Mouth, Left Mouth
            landmarks = face[4:14].reshape((5, 2))
            for lx, ly in landmarks:
                cv2.circle(annotated, (int(lx), int(ly)), 4, (0, 0, 255), -1)
                
    axes[idx].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(f"{os.path.basename(img_path)} ({face_count} face(s))", fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig("plots/face_detection_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved face detection results to plots/face_detection_comparison.png")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.3 Real-Time Face Detection from Live Webcam (`cv2.VideoCapture(0)`)
We connect directly to your laptop's **live web camera** (`cv2.VideoCapture(0)`). A real-time window will open showing live face detection, 5 facial landmarks, and live FPS counter.  
👉 **Press `q` on your keyboard inside the video window to stop the webcam stream anytime.**
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.3 Real-Time Face Detection from Live Webcam (cv2.VideoCapture(0))
cap = cv2.VideoCapture(0)

# If webcam is not attached, fallback to video file so execution never crashes
if not cap.isOpened():
    print("Webcam (index 0) not detected, falling back to sample video.")
    cap = cv2.VideoCapture("data/images/vtest.avi")
else:
    print("Live Webcam opened successfully! A live window has started. Press 'q' to stop.")

# Read initial frame to set detector input dimensions
ret, frame = cap.read()
if ret:
    h, w, _ = frame.shape
    yunet_detector.setInputSize((w, h))

frame_count = 0
prev_time = time.time()
fps_history = []
sample_captured_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    fps_history.append(fps)
    frame_count += 1
    
    # Run YuNet Face Detection on the live webcam frame
    _, faces = yunet_detector.detect(frame)
    
    # Draw green bounding boxes and red landmark points
    if faces is not None:
        for face in faces:
            fx, fy, fw, fh = map(int, face[:4])
            score = face[-1]
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {score*100:.1f}%", (fx, max(20, fy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            
            landmarks = face[4:14].reshape((5, 2))
            for lx, ly in landmarks:
                cv2.circle(frame, (int(lx), int(ly)), 4, (0, 0, 255), -1)
                
    # Draw Live FPS & Exit hint on frame
    cv2.putText(frame, f"Live FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'q' in this window to Stop", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if frame_count == 15 or sample_captured_frame is None:
        sample_captured_frame = frame.copy()
        
    # Display live interactive OpenCV window
    cv2.imshow("YuNet Real-Time Face Detection (Press 'q' to exit)", frame)
    
    # Press 'q' or 'ESC' inside the video window to stop the webcam
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("Webcam stream stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()

avg_fps = np.mean(fps_history) if fps_history else 0
print(f"Total Webcam Frames Processed : {frame_count}")
print(f"Average YuNet Detection Speed : {avg_fps:.2f} FPS")

if sample_captured_frame is not None:
    show(sample_captured_frame, f"Captured Live Webcam Frame Annotated with YuNet ({avg_fps:.1f} FPS)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.4 Score-Threshold Experiment & Applied Privacy Blur Use-Case
1. **Threshold Comparison (0.5 vs 0.7 vs 0.9):** Demonstrates how changing confidence threshold affects true detections and false positives.
2. **Applied Privacy Blur:** Automatically applies a strong Gaussian Blur (`cv2.GaussianBlur`) exclusively inside detected face bounding boxes to anonymize individuals in surveillance footage.
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.4 Score-Threshold Experiment and Face Anonymization (Privacy Blur)
test_img = cv2.imread("data/images/lena.jpg")
h, w, _ = test_img.shape

thresholds = [0.5, 0.7, 0.9]
detected_counts = []

print("Score Threshold Experiment Results:")
for thresh in thresholds:
    det = cv2.FaceDetectorYN.create(yunet_model_path, "", (w, h), score_threshold=thresh)
    det.setInputSize((w, h))
    _, f_res = det.detect(test_img)
    count = 0 if f_res is None else len(f_res)
    detected_counts.append(count)
    print(f" - Score Threshold = {thresh:.1f} -> Detected Faces: {count}")

# Privacy Blur Implementation
blur_detector = cv2.FaceDetectorYN.create(yunet_model_path, "", (w, h), score_threshold=0.6)
blur_detector.setInputSize((w, h))
_, faces_to_blur = blur_detector.detect(test_img)

blurred_privacy_img = test_img.copy()

if faces_to_blur is not None:
    for face in faces_to_blur:
        fx, fy, fw, fh = map(int, face[:4])
        # Ensure within bounds
        fx, fy = max(0, fx), max(0, fy)
        fw, fh = min(fw, w - fx), min(fh, h - fy)
        
        # Extract face region of interest
        face_roi = blurred_privacy_img[fy:fy+fh, fx:fx+fw]
        # Apply heavy Gaussian Blur for anonymization
        blurred_roi = cv2.GaussianBlur(face_roi, (51, 51), 30)
        # Put back
        blurred_privacy_img[fy:fy+fh, fx:fx+fw] = blurred_roi
        # Draw red boundary to indicate anonymized area
        cv2.rectangle(blurred_privacy_img, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
        cv2.putText(blurred_privacy_img, "ANONYMIZED", (fx, max(20, fy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

# Display Before and After Privacy Blur
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image (Unblurred)", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(blurred_privacy_img, cv2.COLOR_BGR2RGB))
axes[1].set_title("Privacy Anonymized (Selective Face Gaussian Blur)", fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.5 WHY YuNet (Simple Explanation)

* **Modern Lightweight CNN Architecture:**  
  Unlike legacy Haar Cascades (which are slow, sensitive to lighting, and produce many false positives), YuNet is a lightweight Convolutional Neural Network trained specifically on large, diverse "in-the-wild" datasets (WIDER Face). It handles rotated heads, side profiles, sunglasses, and varied lighting conditions effortlessly.
  
* **One-Shot Multi-Task Output:**  
  In a single forward pass, YuNet outputs both the face bounding box $(x, y, w, h)$ and 5 essential facial landmarks (right eye, left eye, nose tip, mouth corners), enabling face alignment and emotion analysis downstream.

* **Selecting the Right Score Threshold:**  
  - **Real-Time Security Surveillance (Threshold $\\approx 0.5 - 0.6$):** Prioritize *high recall*. We would rather have an occasional false alarm than miss an actual human intruder entering a restricted area.  
  - **Automated Photo Album Tagging (Threshold $\\approx 0.85 - 0.9$):** Prioritize *high precision*. We want to be certain that every detected region is truly a human face so random background patterns are not mislabeled as family members.
"""))

    # -------------------------------------------------------------
    # TASK 4
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Task 4: Object Detection with YOLOv8

> **Objective:** Load the pretrained Ultralytics YOLOv8 nano model, detect multi-class everyday objects on static images, stream live YOLO detection from webcam, tune confidence & IoU thresholds, and summarize per-class detections.  
> **Expected Output:** Bounding boxes with COCO class names and confidence scores, live real-time webcam YOLO detection window (press 'q' to exit), IoU threshold tuning graph, and a per-class detection frequency bar chart.
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4.1 Load Pretrained YOLOv8 & Run on Static Images
YOLOv8-Nano (`yolov8n.pt`) is a state-of-the-art object detection model pretrained on the COCO dataset (80 object classes such as person, car, bottle, chair, sports ball, apple, etc.).
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.1 Load YOLOv8n and Run on Static Images
from ultralytics import YOLO

yolo_model_path = "data/models/yolov8n.pt"
yolo_model = YOLO(yolo_model_path)

test_yolo_images = ["data/images/messi5.jpg", "data/images/fruits.jpg", "data/images/lena.jpg"]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
plt.suptitle("Ultralytics YOLOv8n Object Detection (COCO Pretrained)", fontsize=15, fontweight='bold')

for idx, img_path in enumerate(test_yolo_images):
    # Run YOLO inference
    results = yolo_model(img_path, conf=0.25, verbose=False)
    
    # results[0].plot() draws boxes, labels, and confidences automatically
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    box_count = len(results[0].boxes)
    axes[idx].imshow(annotated_rgb)
    axes[idx].set_title(f"{os.path.basename(img_path)} ({box_count} objects detected)", fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig("plots/yolo_detections.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved YOLO detections plot to plots/yolo_detections.png")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4.2 Real-Time YOLO Detection from Live Webcam (`cv2.VideoCapture(0)`)
We run YOLOv8 frame-by-frame on your **live laptop webcam**. An interactive popup window displays all recognized objects in real-time with class names, confidence percentages, and live FPS!  
👉 **Press `q` on your keyboard inside the video window to stop the webcam stream.**
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.2 Real-Time YOLOv8 Object Detection from Live Webcam (cv2.VideoCapture(0))
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam (index 0) not detected, falling back to sample video.")
    cap = cv2.VideoCapture("data/images/vtest.avi")
else:
    print("Live Webcam opened successfully! A live window has started. Press 'q' to stop.")

frame_count = 0
prev_time = time.time()
fps_history_yolo = []
sample_captured_yolo = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    fps_history_yolo.append(fps)
    frame_count += 1
    
    # Run YOLOv8 inference on live webcam frame
    res = yolo_model(frame, conf=0.25, verbose=False)
    annotated_frame = res[0].plot()
    
    # Draw Live FPS & Exit hint on frame
    cv2.putText(annotated_frame, f"YOLOv8 Live FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, "Press 'q' in this window to Stop", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if frame_count == 15 or sample_captured_yolo is None:
        sample_captured_yolo = annotated_frame.copy()
        
    # Display live interactive OpenCV window
    cv2.imshow("YOLOv8 Real-Time Object Detection (Press 'q' to exit)", annotated_frame)
    
    # Press 'q' or 'ESC' inside the video window to stop the webcam
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("Webcam stream stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()

avg_fps_yolo = np.mean(fps_history_yolo) if fps_history_yolo else 0
print(f"Total Webcam Frames Processed : {frame_count}")
print(f"Average YOLOv8n Speed         : {avg_fps_yolo:.2f} FPS")

if sample_captured_yolo is not None:
    show(sample_captured_yolo, f"Captured Live Webcam Frame Annotated with YOLOv8n ({avg_fps_yolo:.1f} FPS)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4.3 Confidence & IoU Threshold Tuning
* **Confidence (`conf`):** Minimum prediction probability required to keep a detected box.
* **Intersection over Union (`iou`):** Overlap threshold used during Non-Maximum Suppression (NMS) to eliminate duplicate boxes on the same object.
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.3 Confidence and IoU Threshold Tuning Experiment
conf_values = [0.25, 0.50, 0.75]
iou_values = [0.30, 0.50, 0.70]

tuning_img_path = "data/images/fruits.jpg"
counts_matrix = np.zeros((len(conf_values), len(iou_values)))

for i, c in enumerate(conf_values):
    for j, iou in enumerate(iou_values):
        res = yolo_model(tuning_img_path, conf=c, iou=iou, verbose=False)
        counts_matrix[i, j] = len(res[0].boxes)

# Plotting the threshold sensitivity
fig, ax = plt.subplots(figsize=(8, 5))
for i, c in enumerate(conf_values):
    ax.plot(iou_values, counts_matrix[i, :], marker='o', linewidth=2, label=f'Conf Threshold = {c}')

ax.set_title("Retained Detections vs. IoU and Confidence Thresholds", fontsize=13, fontweight='bold')
ax.set_xlabel("NMS IoU Threshold", fontsize=11)
ax.set_ylabel("Number of Detected Objects", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.show()

print("Retained Boxes Matrix (Rows: Conf 0.25, 0.50, 0.75 | Cols: IoU 0.3, 0.5, 0.7):")
print(counts_matrix)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4.4 Per-Class Detection Summary
We tally the detections across all test images and summarize them in a clean bar chart.
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.4 Per-Class Detection Summary
class_counts = {}

all_test_imgs = ["data/images/messi5.jpg", "data/images/fruits.jpg", "data/images/lena.jpg"]

for p in all_test_imgs:
    res = yolo_model(p, conf=0.25, verbose=False)
    for box in res[0].boxes:
        cls_id = int(box.cls[0].item())
        cls_name = yolo_model.names[cls_id]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

# Plot bar chart
fig, ax = plt.subplots(figsize=(9, 5))
classes = list(class_counts.keys())
frequencies = list(class_counts.values())

bars = ax.bar(classes, frequencies, color='#2b5c8f', width=0.5, edgecolor='black')
ax.set_title("Total Objects Detected Across Test Images by COCO Class", fontsize=13, fontweight='bold')
ax.set_xlabel("Object Class Name", fontsize=11)
ax.set_ylabel("Total Detections", fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, str(yval), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("Detected Object Breakdown:", class_counts)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4.5 WHY YOLO Generalises (Simple Explanation)

* **Single-Stage Grid Architecture:**  
  Traditional two-stage detectors (like Faster R-CNN) first generate region proposals and then classify them separately, making them slow. YOLO ("You Only Look Once") passes the entire image through a convolutional backbone once, predicting bounding boxes and class probabilities directly across a multi-scale grid.
  
* **Non-Maximum Suppression (NMS):**  
  Multiple grid cells around an object might all predict a bounding box for it. NMS calculates the overlap (IoU) between all candidate boxes and keeps only the highest-scoring box while eliminating redundant overlapping duplicates.

* **Model Variant Trade-offs ($n, s, m, l, x$):**  
  - **YOLOv8-Nano (`yolov8n`):** ~3.2M parameters. Extremely fast and lightweight. Ideal for edge devices like Raspberry Pi, drones, and smart cameras.
  - **YOLOv8-Small / Medium (`yolov8s`, `yolov8m`):** Balanced trade-off for real-time edge servers.
  - **YOLOv8-XLarge (`yolov8x`):** Highest accuracy (mAP) for demanding cloud server analysis where powerful GPUs are available.

* **Generalization Power:**  
  Because YOLO was trained on the rich 80-class COCO dataset containing hundreds of thousands of varied everyday scenes, it generalizes seamlessly to diverse lighting, backgrounds, and object sizes without requiring custom hand-tuning for each individual class.
"""))

    # -------------------------------------------------------------
    # TASK 5
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Task 5: Integrated Pipeline & Final Comparison

> **Objective:** Build a unified real-time computer vision pipeline combining YuNet (Face) and YOLOv8 (Objects) on your live webcam, evaluate morphological pre-cleaning, benchmark speed (FPS), and construct a final comparative analysis table.  
> **Expected Output:** Unified live webcam stream (green face boxes + red/blue object boxes), FPS bar chart across all 3 modes, and a complete techniques comparison table.
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5.1 Build the Unified Real-Time Live Webcam Pipeline
The unified pipeline runs **both** detectors concurrently on every live camera frame:
1. **Layer 1 (YuNet):** Green bounding boxes + 5 facial landmarks for human faces.
2. **Layer 2 (YOLOv8):** Red/Blue bounding boxes with COCO class labels and confidence percentages for general objects.  
👉 **Press `q` in the live video window to stop.**
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.1 Unified Dual-Detector Pipeline on Live Webcam (YuNet Faces + YOLOv8 Objects)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam (index 0) not detected, falling back to sample video.")
    cap = cv2.VideoCapture("data/images/vtest.avi")
else:
    print("Live Webcam opened successfully! A live dual-detector window has started. Press 'q' to stop.")

ret, frame = cap.read()
if ret:
    h, w, _ = frame.shape
    yunet_detector.setInputSize((w, h))

frame_count = 0
prev_time = time.time()
fps_history_unified = []
sample_captured_unified = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    fps_history_unified.append(fps)
    frame_count += 1
    
    # 1. Run YuNet Face Detection
    _, faces = yunet_detector.detect(frame)
    
    # 2. Run YOLOv8 Object Detection
    yolo_results = yolo_model(frame, conf=0.25, verbose=False)
    
    unified_frame = frame.copy()
    
    # Draw YOLO general objects (Blue/Red boxes)
    for box in yolo_results[0].boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = f"{yolo_model.names[cls_id]}: {conf*100:.1f}%"
        cv2.rectangle(unified_frame, (bx1, by1), (bx2, by2), (255, 50, 50), 2)
        cv2.putText(unified_frame, label, (bx1, max(18, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)
        
    # Draw YuNet Faces (Green boxes + Red landmarks)
    if faces is not None:
        for face in faces:
            fx, fy, fw_box, fh_box = map(int, face[:4])
            score = face[-1]
            cv2.rectangle(unified_frame, (fx, fy), (fx + fw_box, fy + fh_box), (0, 255, 0), 2)
            cv2.putText(unified_frame, f"FACE: {score*100:.1f}%", (fx, max(35, fy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            
            landmarks = face[4:14].reshape((5, 2))
            for lx, ly in landmarks:
                cv2.circle(unified_frame, (int(lx), int(ly)), 4, (0, 0, 255), -1)
                
    # Draw Live Unified FPS & Exit hint
    cv2.putText(unified_frame, f"Unified Pipeline Live FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(unified_frame, "Press 'q' in this window to Stop", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if frame_count == 15 or sample_captured_unified is None:
        sample_captured_unified = unified_frame.copy()
        
    cv2.imshow("Unified Pipeline: YuNet (Green) + YOLOv8 (Blue) - Press 'q' to exit", unified_frame)
    
    # Press 'q' or 'ESC' inside the video window to stop the webcam
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("Webcam stream stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()

avg_fps_unified = np.mean(fps_history_unified) if fps_history_unified else 0
print(f"Total Webcam Frames Processed : {frame_count}")
print(f"Average Unified Pipeline Speed: {avg_fps_unified:.2f} FPS")

if sample_captured_unified is not None:
    show(sample_captured_unified, f"Captured Live Webcam Frame - Unified Dual Detection ({avg_fps_unified:.1f} FPS)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5.2 Morphological Pre-Cleaning Experiment
We evaluate whether classical morphological denoising helps deep learning detectors when dealing with noisy camera feeds.
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.2 Morphological Pre-Cleaning on a Noisy Input Frame
frame_to_process = cv2.imread("data/images/messi5.jpg")
noisy_frame = frame_to_process.copy()
# Add noise
noise = np.random.normal(0, 25, noisy_frame.shape).astype(np.int16)
noisy_frame = np.clip(noisy_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Morphological pre-cleaning (OpenCV Morphological Opening on color channels)
clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned_frame = cv2.morphologyEx(noisy_frame, cv2.MORPH_OPEN, clean_kernel)

# Run detections on both
res_noisy = yolo_model(noisy_frame, conf=0.25, verbose=False)
res_cleaned = yolo_model(cleaned_frame, conf=0.25, verbose=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(cv2.cvtColor(res_noisy[0].plot(), cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Noisy Frame ({len(res_noisy[0].boxes)} objects detected)", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(res_cleaned[0].plot(), cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Morphological Pre-Cleaned Frame ({len(res_cleaned[0].boxes)} objects detected)", fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5.3 FPS Benchmarking
We benchmark execution speed over a sample of 100 iterations for:
1. **YuNet Face Detector alone**
2. **YOLOv8n Object Detector alone**
3. **Combined Unified Pipeline**
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.3 FPS Benchmarking over 100 Sample Frames
benchmark_img = cv2.imread("data/images/messi5.jpg")
bm_h, bm_w, _ = benchmark_img.shape
yunet_detector.setInputSize((bm_w, bm_h))

num_runs = 100
print(f"Running {num_runs}-frame FPS Benchmark...")

# 1. Face Alone
t0 = time.time()
for _ in range(num_runs):
    yunet_detector.detect(benchmark_img)
fps_face_val = num_runs / (time.time() - t0)

# 2. YOLO Alone
t0 = time.time()
for _ in range(num_runs):
    yolo_model(benchmark_img, conf=0.25, verbose=False)
fps_yolo_val = num_runs / (time.time() - t0)

# 3. Combined Pipeline
t0 = time.time()
for _ in range(num_runs):
    yunet_detector.detect(benchmark_img)
    yolo_model(benchmark_img, conf=0.25, verbose=False)
fps_combined_val = num_runs / (time.time() - t0)

print(f"Results -> YuNet Face: {fps_face_val:.1f} FPS | YOLOv8n: {fps_yolo_val:.1f} FPS | Combined: {fps_combined_val:.1f} FPS")

# Plot FPS Comparison
fig, ax = plt.subplots(figsize=(8, 5))
configs = ['YuNet Face Alone', 'YOLOv8n Alone', 'Combined Pipeline']
fps_vals = [fps_face_val, fps_yolo_val, fps_combined_val]
colors = ['#2ca02c', '#1f77b4', '#d62728']

bars = ax.bar(configs, fps_vals, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Frames Per Second (FPS)', fontsize=12, fontweight='bold')
ax.set_title('Inference Speed Comparison (100-Frame Benchmark)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(fps_vals) * 1.25)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f} FPS',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("plots/fps_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved FPS comparison bar chart to plots/fps_comparison.png")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5.4 Final Results Comparison Table
Here is the complete comparison across all classical and deep learning computer vision techniques explored in this project:
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.4 Final Results Comparison Table Generation
table_data = [
    ["Morphological Operations", "Classical CV", "Binary shapes, Noise, Boundaries", "Low (relies on binary threshold)", "~500+ FPS", "Mask denoising & OCR cleanup"],
    ["Bitwise & Histograms", "Classical CV", "Color masks, Contrast/Brightness", "Medium (diagnostic analysis)", "~400+ FPS", "ROI extraction, watermarks, lighting fix"],
    ["YuNet Face Detector", "Deep Learning (CNN)", "Faces & 5 Key Facial Landmarks", "High (in-the-wild robustness)", f"{fps_face_val:.1f} FPS", "Attendance, surveillance, privacy blur"],
    ["YOLOv8n Object Detector", "Deep Learning (CNN)", "80 General COCO Object Classes", "High (multi-scale deep features)", f"{fps_yolo_val:.1f} FPS", "Real-time smart retail, traffic, robots"]
]
col_labels = ["Technique", "Type", "What It Detects", "Lighting Robustness", "Approx. Speed", "Best Use Case"]

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
ax.axis('tight')

tbl = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.1, 2.0)

for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#1a365d')
    else:
        if row % 2 == 1:
            cell.set_facecolor('#f8fafc')
        else:
            cell.set_facecolor('#ffffff')

plt.title("Summary & Comparison of Computer Vision Techniques", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("plots/results_table.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved final results table to plots/results_table.png")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5.5 Reflection & Architecture Decisions (Simple Explanation)

* **Best Combination for Real-Time Monitoring:**  
  The most effective architecture for a real-world smart monitoring pipeline combines **fast classical pre-processing with modern lightweight deep detectors**:
  1. Use **Histogram Equalization / Brightness Adjustment (Task 2)** when camera lighting is dim or uneven.
  2. Use **Morphological Opening (Task 1)** to suppress camera sensor noise.
  3. Run **YuNet Face Detection (Task 3)** for ultra-fast, high-precision human biometric monitoring (~60+ FPS).
  4. Run **YOLOv8n Object Detection (Task 4)** to track surrounding vehicles, luggage, bottles, and animals.

* **Deploying on Low-Power Edge Devices (e.g., Raspberry Pi, Jetson Nano):**  
  - Keep **YOLOv8-Nano (`yolov8n`)** or export to **ONNX / TensorRT (FP16/INT8)** for maximum acceleration.
  - Skip computationally expensive full-frame morphology if noise is low; apply it only on small bounding-box regions.
  - Set `score_threshold = 0.6` on YuNet to avoid unneeded post-processing on faint false positives.

* **Deploying on Cloud Servers (e.g., NVIDIA A100 / RTX 4090):**  
  - Upgrade from `yolov8n` to **`yolov8s` or `yolov8m`** for higher detection accuracy (mAP).
  - Process high-resolution 1080p/4K streams in batches.
  - Apply advanced super-resolution or HDR tone-mapping pre-processing pipelines.
"""))

    nb.cells = cells
    return nb

if __name__ == "__main__":
    nb = create_notebook()
    with open("project 3.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    with open("CV_PR3.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebooks project 3.ipynb and CV_PR3.ipynb successfully created!")
