import nibabel as nib
import numpy as np
import os
import json
import glob
import argparse

def load_nii(nii_path):
    """Load a NIfTI file."""
    return nib.load(nii_path)

def save_nii(img_data, img_affine, save_path):
    """Save data to a NIfTI file."""
    nii_img = nib.Nifti1Image(img_data, img_affine)
    nib.save(nii_img, save_path)

def restore_to_original(pred_label, cropping_info, original_shape):
    """Restore the cropped label to its original size."""
    restored_label = np.zeros(original_shape)
    
    min_coords = cropping_info["min_coords"]
    max_coords = cropping_info["max_coords"]
    
    restored_label[min_coords[0]:max_coords[0], 
                   min_coords[1]:max_coords[1], 
                   :] = pred_label
    
    return restored_label

def restore_and_save_all_labels(label_dir, cropping_info_dir, original_scan_dir, save_dir):
    """Restore and save all labels from the given directories."""
    # Ensure the save directory exists
    #os.makedirs(save_dir, exist_ok=True)
    
    # Find all label files in the directory
    label_files = glob.glob(os.path.join(label_dir, "*.nii.gz"))
    
    for label_path in label_files:
        # Extract the casename (without extension)
        casename = os.path.basename(label_path).replace(".nii.gz", "")
        
        # Load the cropping info corresponding to the label
        cropping_info_path = os.path.join(cropping_info_dir, f'{casename}_cropping_info.json')
        
        if not os.path.exists(cropping_info_path):
            print(f"No cropping info found for label {casename}. Skipping.")
            continue
        
        with open(cropping_info_path, 'r') as f:
            cropping_info = json.load(f)
        
        # Load the predicted label from the NIfTI file
        label_img = load_nii(label_path)
        pred_label = label_img.get_fdata()

        # Restore the predicted label to the original size
        original_shape = cropping_info['original_shape']
        restored_pred_label = restore_to_original(pred_label, cropping_info, original_shape)

        restored_pred_label[restored_pred_label == 1] = 0
        restored_pred_label[restored_pred_label == 2] = 0

        # make sure the scar value is 2221 and the edema value is 1220
        restored_pred_label[restored_pred_label == 3] = 2221  # scar
        restored_pred_label[restored_pred_label == 4] = 1220  # edema

        restored_pred_label = restored_pred_label.astype(np.int32)

        # Load the original scan to get the affine matrix
        original_scan_path = os.path.join(original_scan_dir, f'{casename}_0000.nii.gz')
        original_scan_img = load_nii(original_scan_path)
        
        # Save the restored label as a NIfTI file

        # if not os.path.exists(os.path.join(save_dir, casename)):
        #     os.makedirs(os.path.join(save_dir, casename))
        

        restored_save_path = os.path.join(save_dir, f'{casename}_pred.nii.gz')
        save_nii(restored_pred_label, original_scan_img.affine, restored_save_path)
        #print(f"Restored label saved to: {restored_save_path}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Restore cropped NIfTI labels to their original size using cropping info.")
    
    parser.add_argument('-l', '--label_dir', type=str, required=True, 
                        help="Directory containing the cropped label NIfTI files.")
    
    parser.add_argument('-c', '--cropping_info_dir', type=str, required=True, 
                        help="Directory containing the cropping info JSON files.")
    
    parser.add_argument('-o', '--original_scan_dir', type=str, required=True, 
                        help="Directory containing the original scan NIfTI files for retrieving affine matrices.")
    
    parser.add_argument('-s', '--save_dir', type=str, required=True, 
                        help="Directory where the restored labels will be saved.")
    
    args = parser.parse_args()
    
    # Restore and save all labels with the given arguments
    restore_and_save_all_labels(
        label_dir=args.label_dir, 
        cropping_info_dir=args.cropping_info_dir, 
        original_scan_dir=args.original_scan_dir, 
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()
