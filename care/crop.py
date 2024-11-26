import nibabel as nib
import numpy as np
import os
import json
import argparse
import glob
from tqdm import tqdm

def load_nii(nii_path):
    """Load a NIfTI file."""
    return nib.load(nii_path)

def save_nii(img_data, img_affine, save_path):
    """Save data to a NIfTI file."""
    nii_img = nib.Nifti1Image(img_data, img_affine)
    nib.save(nii_img, save_path)

def crop_roi(scan_data, label_data, padding=0):
    """Crop the scan and label to the region of interest (ROI) in the first and second dimensions."""
    # Find non-zero elements in the label data to determine the bounding box
    non_zero = np.nonzero(label_data)
    
    # Determine the cropping bounds in the first and second dimensions
    min_x = np.min(non_zero[0]) - padding
    max_x = np.max(non_zero[0]) + padding
    min_y = np.min(non_zero[1]) - padding
    max_y = np.max(non_zero[1]) + padding
    
    # Ensure the coordinates are within the bounds of the image
    min_x = max(min_x, 0)
    max_x = min(max_x, scan_data.shape[0])
    min_y = max(min_y, 0)
    max_y = min(max_y, scan_data.shape[1])
    
    # Crop the scan and label using the bounding box
    cropped_scan = scan_data[min_x:max_x, min_y:max_y, :]
    cropped_label = label_data[min_x:max_x, min_y:max_y, :]
    
    return cropped_scan, cropped_label, (min_x, min_y), (max_x, max_y)

def crop_and_save(scan_path, save_dir, min_coords, max_coords, casename, padding=0):
    """Crop the scan using the provided ROI coordinates and save it."""
    # Load the scan NIfTI file
    scan_img = load_nii(scan_path)
    scan_data = scan_img.get_fdata()

    # Crop the scan using the provided coordinates
    cropped_scan = scan_data[min_coords[0]:max_coords[0], 
                             min_coords[1]:max_coords[1], 
                             :]  # Keep the third dimension unchanged

    # Extract the scan's suffix (e.g., "_0000.nii.gz") to preserve the naming convention
    scan_suffix = '_' + os.path.basename(scan_path).split('_')[-1]
    
    # Construct the save path for the cropped scan
    scan_save_path = os.path.join(save_dir, f'{casename}{scan_suffix}')
    
    # Save the cropped scan
    save_nii(cropped_scan, scan_img.affine, scan_save_path)
    
    #print(f"Cropped scan saved to: {scan_save_path}")

def process_folder(scan_dir, label_dir, save_dir, padding=0):
    """
    Process all scan and label files in the provided directories.

    Args:
        scan_dir (str): Directory containing the scan NIfTI files.
        label_dir (str): Directory containing the label NIfTI files.
        save_dir (str): Directory to save the cropped scans, labels, and cropping information.
        padding (int): Padding around the ROI.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subdirectories for labels and images
    #labels_dir = os.path.join(save_dir, 'labelsTs')
    images_dir = os.path.join(save_dir, 'cropped_images')
    cropping_info_dir = os.path.join(save_dir, 'cropping_info')
    #os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(cropping_info_dir, exist_ok=True)
    
    # Find all label files
    label_files = glob.glob(os.path.join(label_dir, "*.nii.gz"))
    for label_path in tqdm(label_files):
        # Extract the casename (without extension)
        casename = os.path.basename(label_path).replace(".nii.gz", "")
        # Find all corresponding scan files
        scan_files = glob.glob(os.path.join(scan_dir, f"{casename}_*.nii.gz"))
        
        if not scan_files:
            print(f"No scans found for label {casename}")
            continue
        
        # Load the label NIfTI file and determine the cropping coordinates
        label_img = load_nii(label_path)
        label_data = label_img.get_fdata()
        cropped_scan, cropped_label, min_coords, max_coords = crop_roi(label_data, label_data, padding)

        # Save the cropped label only once
        #label_save_path = os.path.join(labels_dir, f'{casename}.nii.gz')
        #save_nii(cropped_label, label_img.affine, label_save_path)
        
        # Convert coordinates and shapes to Python integers for JSON serialization
        cropping_info = {
            "min_coords": list(map(int, min_coords)),
            "max_coords": list(map(int, max_coords)),
            "original_shape": list(map(int, label_data.shape))
        }
        #print(os.path.join(cropping_info_dir, f'{casename}_cropping_info.json'))
        with open(os.path.join(cropping_info_dir, f'{casename}_cropping_info.json'), 'w') as f:
            json.dump(cropping_info, f)
        
        #print(f"Cropped label saved to: {label_save_path}")
        #print(f"Cropping info saved to: {os.path.join(cropping_info_dir, f'{casename}_cropping_info.json')}")
        
        # Process each corresponding scan file
        for scan_path in scan_files:
            # Crop and save the scan using the same coordinates as the label
            crop_and_save(scan_path, images_dir, min_coords, max_coords, casename, padding=padding)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process NIfTI scans and labels, crop ROIs, and save them.")
    
    parser.add_argument('-s', '--scan_dir', type=str, required=True, 
                        help="Directory containing the scan NIfTI files.")
    
    parser.add_argument('-l', '--label_dir', type=str, required=True, 
                        help="Directory containing the label NIfTI files.")
    
    parser.add_argument('-o', '--output_dir', type=str, required=True, 
                        help="Directory where the cropped images and cropping info will be saved.")
    
    parser.add_argument('-p', '--padding', type=int, default=0, 
                        help="Padding around the ROI. Default is 0.")
    
    args = parser.parse_args()
    
    # Process the folder with the given arguments
    process_folder(scan_dir=args.scan_dir, 
                   label_dir=args.label_dir, 
                   save_dir=args.output_dir, 
                   padding=args.padding)

if __name__ == "__main__":
    main()
