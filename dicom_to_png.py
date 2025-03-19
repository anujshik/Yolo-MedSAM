import os
import glob
import pydicom
import numpy as np
import cv2

dicom_folder = "/Users/anuj/Dataset/CT scan files(11,12)/file 11 lata chavan (lakshmi)/LAXMI/DICOM/20240503/17040000"
output_folder = "converted_pngs_p11/"  
os.makedirs(output_folder, exist_ok=True)

def is_valid_dicom(file_path):
    try:
        pydicom.dcmread(file_path)
        return True
    except Exception:
        return False

def convert_dicom_to_png(dicom_path, output_folder):
    """Convert a single DICOM file to PNG format and save it."""
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        image_array = dicom_data.pixel_array  # Extract pixel data

        # Normalize pixel values to 0-255
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
        image_array = image_array.astype(np.uint8)

        # Generate PNG filename (same as DICOM filename)
        png_filename = os.path.basename(dicom_path) + ".png"
        output_path = os.path.join(output_folder, png_filename)

        # Save the PNG image
        success = cv2.imwrite(output_path, image_array)
        if success:
            print(f"Saved: {output_path}")
        else:
            print(f"Failed to save: {output_path}")

    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")

# Detect all files in the folder (even without .dcm extensions)
dicom_files = glob.glob(os.path.join(dicom_folder, "*"))  # Get all files

# Filter valid DICOM files by actually trying to read them
valid_dicom_files = [f for f in dicom_files if is_valid_dicom(f)]

# Check if files are detected
if len(valid_dicom_files) == 0:
    print("No valid DICOM files found in the folder. Check the path or file extensions!")
else:
    print(f"Found {len(valid_dicom_files)} DICOM files. Converting to PNG...")

# Convert each valid DICOM file to PNG
for dicom_file in valid_dicom_files:
    convert_dicom_to_png(dicom_file, output_folder)

print("\n All valid DICOM files have been converted to PNG!")
