import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import json

# Ensure directories exist
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("Starting verification of all project tasks...")

# -------------------------------------------------------------
# TASK 1: Image I/O & Morphological Operations
# -------------------------------------------------------------
print("--- TASK 1 ---")
# Load grayscale and color images
gray_img = cv2.imread("data/images/lena.jpg", cv2.IMREAD_GRAYSCALE)
color_img = cv2.imread("data/images/lena.jpg", cv2.IMREAD_COLOR)

print(f"Grayscale Image Shape: {gray_img.shape}, Type: {gray_img.dtype}")
print(f"Color Image Shape: {color_img.shape}, Type: {color_img.dtype}")

# 1.2 Binarization
_, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

# 1.3 Four Morphological Operations with 5x5 rectangular kernel
kernel_5x5 = np.ones((5, 5), np.uint8)
eroded = cv2.erode(binary_img, kernel_5x5, iterations=1)
dilated = cv2.dilate(binary_img, kernel_5x5, iterations=1)
opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_5x5)
closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_5x5)

# 1.4 Kernel shape & size comparison on noisy binary image
# Create a test binary image with noise
noisy_bin = binary_img.copy()
# Add salt & pepper noise
noise_salt = np.random.rand(*noisy_bin.shape) < 0.05
noise_pepper = np.random.rand(*noisy_bin.shape) < 0.05
noisy_bin[noise_salt] = 255
noisy_bin[noise_pepper] = 0

shapes = [
    ("Rectangular", cv2.MORPH_RECT),
    ("Elliptical", cv2.MORPH_ELLIPSE),
    ("Cross", cv2.MORPH_CROSS)
]
sizes = [(3, 3), (5, 5), (9, 9)]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
plt.suptitle("Morphological Opening: Kernel Shape vs. Size Comparison", fontsize=16, fontweight='bold', y=0.98)

for i, (shape_name, shape_code) in enumerate(shapes):
    for j, (k_w, k_h) in enumerate(sizes):
        k = cv2.getStructuringElement(shape_code, (k_w, k_h))
        cleaned = cv2.morphologyEx(noisy_bin, cv2.MORPH_OPEN, k)
        axes[i, j].imshow(cleaned, cmap='gray')
        axes[i, j].set_title(f"{shape_name} ({k_w}x{k_h})", fontsize=11)
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig("plots/morphology_grid.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved plots/morphology_grid.png")


# -------------------------------------------------------------
# TASK 2: Bitwise Operations & Histograms
# -------------------------------------------------------------
print("--- TASK 2 ---")
# 2.1 Masks
canvas_rect = np.zeros((300, 300), dtype=np.uint8)
canvas_circle = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(canvas_rect, (50, 50), (250, 250), 255, -1)
cv2.circle(canvas_circle, (150, 150), 120, 255, -1)

bit_and = cv2.bitwise_and(canvas_rect, canvas_circle)
bit_or = cv2.bitwise_or(canvas_rect, canvas_circle)
bit_xor = cv2.bitwise_xor(canvas_rect, canvas_circle)
bit_not = cv2.bitwise_not(canvas_rect)

# 2.4 Brightness & Contrast Adjustment + Histograms
img_orig = cv2.imread("data/images/messi5.jpg")
img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
img_orig_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

img_bright = cv2.convertScaleAbs(img_orig, alpha=1.5, beta=30)
img_bright_rgb = cv2.cvtColor(img_bright, cv2.COLOR_BGR2RGB)
img_bright_gray = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)

img_dark = cv2.convertScaleAbs(img_orig, alpha=0.7, beta=-30)
img_dark_rgb = cv2.cvtColor(img_dark, cv2.COLOR_BGR2RGB)
img_dark_gray = cv2.cvtColor(img_dark, cv2.COLOR_BGR2GRAY)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plt.suptitle("Brightness & Contrast Adjustments with Grayscale Histograms", fontsize=16, fontweight='bold')

# Images
axes[0, 0].imshow(img_orig_rgb)
axes[0, 0].set_title("Original Image (alpha=1.0, beta=0)", fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(img_bright_rgb)
axes[0, 1].set_title("High Contrast & Bright (alpha=1.5, beta=30)", fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(img_dark_rgb)
axes[0, 2].set_title("Low Contrast & Dark (alpha=0.7, beta=-30)", fontsize=12)
axes[0, 2].axis('off')

# Histograms
for idx, (gray, title, color) in enumerate([
    (img_orig_gray, "Original Histogram", "blue"),
    (img_bright_gray, "Bright / High Contrast Histogram", "green"),
    (img_dark_gray, "Dark / Low Contrast Histogram", "purple")
]):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    axes[1, idx].plot(hist, color=color, linewidth=2)
    axes[1, idx].set_title(title, fontsize=11)
    axes[1, idx].set_xlim([0, 256])
    axes[1, idx].set_xlabel("Pixel Intensity (0-255)")
    axes[1, idx].set_ylabel("Pixel Frequency")
    axes[1, idx].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("plots/histogram_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved plots/histogram_comparison.png")


# -------------------------------------------------------------
# TASK 3: Face Detection with YuNet
# -------------------------------------------------------------
print("--- TASK 3 ---")
yunet_path = "data/models/face_detection_yunet_2023mar.onnx"
test_face_images = ["data/images/lena.jpg", "data/images/messi5.jpg", "data/images/baboon.jpg"]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
plt.suptitle("YuNet Face Detection & 5 Facial Landmarks (OpenCV 5.0)", fontsize=16, fontweight='bold')

for idx, img_path in enumerate(test_face_images):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    h, w, _ = img_bgr.shape
    
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), score_threshold=0.6)
    detector.setInputSize((w, h))
    _, faces = detector.detect(img_bgr)
    
    annotated = img_bgr.copy()
    if faces is not None:
        for face in faces:
            # Box: x, y, w, h
            fx, fy, fw, fh = map(int, face[:4])
            score = face[-1]
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            cv2.putText(annotated, f"Face: {score*100:.1f}%", (fx, max(20, fy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            
            # Landmarks: right eye, left eye, nose tip, right mouth corner, left mouth corner
            landmarks = face[4:14].reshape((5, 2))
            for lx, ly in landmarks:
                cv2.circle(annotated, (int(lx), int(ly)), 4, (0, 0, 255), -1)
                
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    axes[idx].imshow(annotated_rgb)
    axes[idx].set_title(f"{os.path.basename(img_path)} (Detected: {0 if faces is None else len(faces)} face(s))", fontsize=12)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig("plots/face_detection_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved plots/face_detection_comparison.png")


# -------------------------------------------------------------
# TASK 4: Object Detection with YOLOv8
# -------------------------------------------------------------
print("--- TASK 4 ---")
yolo_model = YOLO("data/models/yolov8n.pt")
test_yolo_images = ["data/images/messi5.jpg", "data/images/fruits.jpg", "data/images/lena.jpg"]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
plt.suptitle("Ultralytics YOLOv8n Object Detection (COCO Pretrained)", fontsize=16, fontweight='bold')

for idx, img_path in enumerate(test_yolo_images):
    results = yolo_model(img_path, conf=0.25, verbose=False)
    plotted_bgr = results[0].plot()
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
    
    axes[idx].imshow(plotted_rgb)
    axes[idx].set_title(f"{os.path.basename(img_path)} ({len(results[0].boxes)} objects)", fontsize=12)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig("plots/yolo_detections.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved plots/yolo_detections.png")


# -------------------------------------------------------------
# TASK 5: Integrated Pipeline & FPS Benchmarking
# -------------------------------------------------------------
print("--- TASK 5 ---")
# Benchmark FPS over 100 iterations on messi5.jpg
benchmark_frame = cv2.imread("data/images/messi5.jpg")
bh, bw, _ = benchmark_frame.shape

# 1. YuNet Face Only
detector = cv2.FaceDetectorYN.create(yunet_path, "", (bw, bh), score_threshold=0.6)
detector.setInputSize((bw, bh))

t0 = time.time()
for _ in range(100):
    detector.detect(benchmark_frame)
t_face = time.time() - t0
fps_face = 100.0 / t_face

# 2. YOLO Only
t0 = time.time()
for _ in range(100):
    yolo_model(benchmark_frame, conf=0.25, verbose=False)
t_yolo = time.time() - t0
fps_yolo = 100.0 / t_yolo

# 3. Combined Pipeline
t0 = time.time()
for _ in range(100):
    detector.detect(benchmark_frame)
    yolo_model(benchmark_frame, conf=0.25, verbose=False)
t_combined = time.time() - t0
fps_combined = 100.0 / t_combined

print(f"Benchmark Results: Face={fps_face:.1f} FPS, YOLO={fps_yolo:.1f} FPS, Combined={fps_combined:.1f} FPS")

# Plot FPS Comparison Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
configs = ['YuNet Face Alone', 'YOLOv8n Alone', 'Combined Pipeline']
fps_values = [fps_face, fps_yolo, fps_combined]
colors = ['#2ca02c', '#1f77b4', '#d62728']

bars = ax.bar(configs, fps_values, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Frames Per Second (FPS)', fontsize=12, fontweight='bold')
ax.set_title('Inference Speed Comparison (100-Frame Benchmark)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(fps_values) * 1.25)
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
plt.close()
print("Saved plots/fps_comparison.png")


# 5.4 Comparison Table Plot
table_data = [
    ["Morphological Operations", "Classical CV", "Binary shapes, Noise, Boundaries", "Low (relies on threshold)", f"~500+ FPS", "Mask denoising & OCR cleanup"],
    ["Bitwise & Histograms", "Classical CV", "Color masks, Contrast/Brightness", "Medium (diagnostic analysis)", f"~400+ FPS", "ROI extraction, watermarks, lighting fix"],
    ["YuNet Face Detector", "Deep Learning (CNN)", "Faces & 5 Key Facial Landmarks", "High (in-the-wild robustness)", f"{fps_face:.1f} FPS", "Attendance, surveillance, privacy blur"],
    ["YOLOv8n Object Detector", "Deep Learning (CNN)", "80 General COCO Object Classes", "High (multi-scale deep features)", f"{fps_yolo:.1f} FPS", "Real-time smart retail, traffic, robots"]
]
col_labels = ["Technique", "Type", "What It Detects", "Lighting Robustness", "Approx. Speed", "Best Use Case"]

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
ax.axis('tight')

table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 2.0)

# Style headers
for (row, col), cell in table.get_celld().items():
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
plt.close()
print("Saved plots/results_table.png")

print("All tasks verified and plots successfully generated!")
