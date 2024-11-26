import os
import nibabel as nib
import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops
import argparse

def remove_small_segments_3d(label_img, min_size, target_labels):
    """
    Remove isolated small segments for specified target labels from a 3D labeled image.

    Parameters:
    - label_img: 3D array with labels.
    - min_size: Minimum size (number of pixels) for a segment to be retained.
    - target_labels: List of label values to process (e.g., [2221, 1220]).

    Returns:
    - cleaned_label_img: 3D label image with small segments removed for target labels.
    """
    cleaned_label_img = np.copy(label_img)

    for label_value in target_labels:
        binary_img = (label_img == label_value)
        labeled_array, num_features = label(binary_img)

        for region in regionprops(labeled_array):
            if region.area < min_size:
                cleaned_label_img[labeled_array == region.label] = 0  # Set small segments to background (0)

    return cleaned_label_img

def remove_small_segments_2d(label_img, min_size, target_labels):
    """
    Remove isolated small segments for specified target labels from a 3D labeled image slice-by-slice (2D).

    Parameters:
    - label_img: 3D array with labels.
    - min_size: Minimum size (number of pixels) for a segment to be retained in each 2D slice.
    - target_labels: List of label values to process (e.g., [2221, 1220]).

    Returns:
    - cleaned_label_img: 3D label image with small segments removed for target labels in 2D slices.
    """
    cleaned_label_img = np.copy(label_img)

    for z in range(label_img.shape[2]):  # Assuming Z-dimension is the third axis (slices)
        slice_img = label_img[:, :, z]

        for label_value in target_labels:
            binary_img = (slice_img == label_value)
            labeled_array, num_features = label(binary_img)

            for region in regionprops(labeled_array):
                if region.area < min_size:
                    cleaned_label_img[:, :, z][labeled_array == region.label] = 0  # Set small segments to background (0)

    return cleaned_label_img

def process_segmentation_files(input_folder, output_folder, min_size, approach):
    """
    Process all .nii.gz segmentation files in a folder and remove small segments.

    Parameters:
    - input_folder: Path to the folder containing input segmentation files.
    - output_folder: Path to save the cleaned segmentation files.
    - min_size: Minimum size (number of pixels) for a segment to be retained.
    - approach: String '2D' or '3D' to select the processing method.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    target_labels = [2221, 1220]  # Labels to process

    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")

            # Load the segmentation file
            seg_img = nib.load(file_path)
            seg_data = seg_img.get_fdata()

            # Remove small segments based on the selected approach (2D or 3D)
            if approach == '3D':
                cleaned_data = remove_small_segments_3d(seg_data, min_size, target_labels)
            else:
                cleaned_data = remove_small_segments_2d(seg_data, min_size, target_labels)

            # Save the cleaned segmentation
            cleaned_img = nib.Nifti1Image(cleaned_data, seg_img.affine, seg_img.header)
            output_path = os.path.join(output_folder, filename)
            nib.save(cleaned_img, output_path)
            print(f"Saved cleaned file to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove isolated small segments from segmentation files.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing input segmentation files (.nii.gz).")
    parser.add_argument("output_folder", type=str, help="Path to save the cleaned segmentation files.")
    parser.add_argument("--min_size", type=int, default=100, help="Minimum size (in pixels) for retaining segments.")
    parser.add_argument("--approach", choices=["2D", "3D"], default="3D", help="Choose between 2D slice-by-slice or 3D processing.")

    args = parser.parse_args()

    process_segmentation_files(args.input_folder, args.output_folder, args.min_size, args.approach)
