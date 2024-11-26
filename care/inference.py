from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import os
from crop import process_folder
from reverse import restore_and_save_all_labels
import shutil

# Set the device to use for model inference
# device = torch.device('cuda', 0)
device = torch.device('cpu')
# Define base paths
homepath = '/workspace/' #"/home/hln0895/care/" ##"/home/hln0895/care/"
original_scan_dir = '/input/'# "/home/hln0895/care/test/"#'/input/' #os.path.join(homepath,  '/input/') #"/home/hln0895/care_validate/"
final_output_dir = '/output/'#os.path.join(homepath, 'output/') #
destination_input_dir = os.path.join(homepath, 'medium_input')
# final_output_dir1 = os.path.join(homepath, '3final_output')

# Define folder paths
model_folder_coarse = os.path.join(homepath, "model_folder_coarse/")
coarse_output_dir = os.path.join(homepath, '0coarse_output/')
cropped_output_dir = os.path.join(homepath, '1cropped_output/')
cropped_images_dir = os.path.join(cropped_output_dir, 'cropped_images')
cropping_info_dir = os.path.join(cropped_output_dir, 'cropping_info')
finer_output_dir = os.path.join(homepath, '2finer_output')


# Create necessary directories if they don't exist
os.makedirs(coarse_output_dir, exist_ok=True)
os.makedirs(cropped_images_dir, exist_ok=True)
os.makedirs(cropping_info_dir, exist_ok=True)
os.makedirs(finer_output_dir, exist_ok=True)
os.makedirs(destination_input_dir, exist_ok=True)
# os.makedirs(final_output_dir1, exist_ok=True)

def main():




    for case in os.listdir(original_scan_dir):
        file_path = os.path.join(original_scan_dir, case, case + '_LGE.nii.gz')
        shutil.copy(file_path, os.path.join(destination_input_dir, case + '_0000.nii.gz'))

        file_path = os.path.join(original_scan_dir, case, case + '_T2.nii.gz')
        shutil.copy(file_path, os.path.join(destination_input_dir, case + '_0001.nii.gz'))

        file_path = os.path.join(original_scan_dir, case, case + '_C0.nii.gz')
        shutil.copy(file_path, os.path.join(destination_input_dir, case + '_0002.nii.gz'))

    # Step 1: Initial (coarse) segmentation
    print("=== Coarse Segmentation ===")
    print(device)

    predictor = nnUNetPredictor(device=device, allow_tqdm=False,perform_everything_on_device=False)
    predictor.initialize_from_trained_model_folder(
        model_folder_coarse,
        (0,),  # GPU index
        checkpoint_name=os.path.join(model_folder_coarse, 'checkpoint_best.pth')
    )
    predictor.predict_from_files(
        list_of_lists_or_source_folder = destination_input_dir,
        output_folder_or_list_of_truncated_output_files=coarse_output_dir,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1
    )

    #Step 2: Cropping coarse output
    print("=== Cropping ===")

    process_folder(destination_input_dir, coarse_output_dir, cropped_output_dir, padding=20)

    # Step 3: Finer segmentation on cropped images
    print("=== Finer Segmentation ===")
    finer_segmentation_model_folder = os.path.join(homepath, "model_folder_fine/")
    finer_predictor = nnUNetPredictor(device=device, allow_tqdm=False)
    finer_predictor.initialize_from_trained_model_folder(
        finer_segmentation_model_folder,
        (0,),  # GPU index
        checkpoint_name=os.path.join(finer_segmentation_model_folder, 'checkpoint_best.pth')
    )
    finer_predictor.predict_from_files(
        list_of_lists_or_source_folder=cropped_images_dir,
        output_folder_or_list_of_truncated_output_files=finer_output_dir,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1
    )

    # Step 4: Restore original label size
    print("=== Restoring Segmentation Labels ===")

    restore_and_save_all_labels(
        label_dir=finer_output_dir,
        cropping_info_dir=cropping_info_dir,
        original_scan_dir=destination_input_dir,
        save_dir=final_output_dir
    )
    print('Done')


if __name__ == '__main__':
    main()
