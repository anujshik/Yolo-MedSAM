import os
import glob
from ultralytics import YOLO
from subprocess import run

test_folder = "test2/" 
medsam_script_path = "/Users/anuj/MedSAM/MedSAM_Inference.py" 
output_folder = "/Users/anuj/yolo_sam/segmented_ct_slices5/"  # Segmentation output path
checkpoint_path = "/Users/anuj/MedSAM/work_dir/MedSAM/medsam_vit_b.pth"  # Path to MedSAM checkpoint

left_knee_folder = os.path.join(output_folder, "left_knee")
right_knee_folder = os.path.join(output_folder, "right_knee")
os.makedirs(left_knee_folder, exist_ok=True)
os.makedirs(right_knee_folder, exist_ok=True)

# Load trained YOLO model
model = YOLO("/Users/anuj/yolo_sam/runs/detect/train/weights/best.pt")

# Loop through all images in the test folder
image_paths = sorted(glob.glob(os.path.join(test_folder, "*.png")))
for image_path in image_paths:
    print(f"Processing: {image_path}")

    # Get image name without extension
    image_name = os.path.basename(image_path).replace(".png", "")
    
    # Load image dimensions
    img_width = 512  # Adjust based on actual image width

    # Run YOLO on each image
    results = model.predict(image_path, conf=0.1)

    # Extract bounding boxes and classify left/right knee
    left_knee_boxes = []
    right_knee_boxes = []

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract YOLO bounding box
        conf = float(box.conf[0])  # Confidence score
        area = (x2 - x1) * (y2 - y1)

        # Compute x-midpoint to classify left/right knee
        x_mid = (x1 + x2) / 2

        if x_mid < img_width / 2:  # Left knee (appears on left side of image)
            left_knee_boxes.append((area, [x1, y1, x2, y2]))
            print(f" Left Knee Detected: ({x1}, {y1}) - ({x2}, {y2}), Confidence: {conf}")
        else:  # Right knee (appears on right side of image)
            right_knee_boxes.append((area, [x1, y1, x2, y2]))
            print(f" Right Knee Detected: ({x1}, {y1}) - ({x2}, {y2}), Confidence: {conf}")


    left_knee_boxes.sort(reverse=True, key=lambda x: x[0])  # Sort by area
    right_knee_boxes.sort(reverse=True, key=lambda x: x[0])

    # Extract only the sorted bounding boxes for MedSAM
    left_knee_boxes = [box for _, box in left_knee_boxes]
    right_knee_boxes = [box for _, box in right_knee_boxes]

    # Process left knee with MedSAM if detected
    if left_knee_boxes:
        left_output_folder = os.path.join(left_knee_folder, image_name)
        os.makedirs(left_output_folder, exist_ok=True)
        run(["python", medsam_script_path, "-i", image_path, "-o", left_output_folder, "--box", str(left_knee_boxes), "-chk", checkpoint_path])

    # Process right knee with MedSAM if detected
    if right_knee_boxes:
        right_output_folder = os.path.join(right_knee_folder, image_name)
        os.makedirs(right_output_folder, exist_ok=True)
        run(["python", medsam_script_path, "-i", image_path, "-o", right_output_folder, "--box", str(right_knee_boxes), "-chk", checkpoint_path])

print("\n All images have been processed. Segmentation results saved separately for left and right knees.")
