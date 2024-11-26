from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import time
import subprocess  # For running crop.py and reverse.py
import os
from crop import process_folder
import shutil
from reverse import restore_and_save_all_labels
device = torch.device('cuda', 0)
#device = torch.device('cpu')
model_folder_coarse = "/home/hln0895/care/model_folder_coarse/"

os.makedirs('/home/hln0895/care/0coarse_output/', exist_ok=True)
os.makedirs('/home/hln0895/care/1cropped_output/', exist_ok=True)
os.makedirs('/home/hln0895/care/1cropped_output/cropped_images', exist_ok=True)
os.makedirs('/home/hln0895/care/1cropped_output/cropping_info', exist_ok=True)
os.makedirs('/home/hln0895/care/2finer_output', exist_ok=True)
os.makedirs('/home/hln0895/care/3final_output', exist_ok=True)

destination_input_dir="/home/hln0895/care/input/"
def run_command(command):
    """Helper function to run shell commands."""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(result.stdout)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        return e.returncode

if __name__ == '__main__':
    start_time = time.time()

    original_scan_dir = '/home/hln0895/care/test/' #"/home/hln0895/care_validate2/"

    print("===coarse segmentation=====")
    # Step 1: Initial segmentation
    predictor = nnUNetPredictor(device=device, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(
        model_folder_coarse,
        (0,),
        checkpoint_name=os.path.join(model_folder_coarse, 'checkpoint_best.pth')
    )

    for case in os.listdir(original_scan_dir):
        file_path = os.path.join(original_scan_dir, case, case + '_LGE.nii.gz')
        shutil.copy(file_path, os.path.join(destination_input_dir, case + '_0000.nii.gz'))

        file_path = os.path.join(original_scan_dir, case, case + '_T2.nii.gz')
        shutil.copy(file_path, os.path.join(destination_input_dir, case + '_0001.nii.gz'))

        file_path = os.path.join(original_scan_dir, case, case + '_C0.nii.gz')
        shutil.copy(file_path, os.path.join(destination_input_dir, case + '_0002.nii.gz'))


    predictor.predict_from_files(
        list_of_lists_or_source_folder= destination_input_dir,#'/home/hln0895/care/test/',
        output_folder_or_list_of_truncated_output_files='/home/hln0895/care/0coarse_output/',
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1
    )
    end_time = time.time()
    print((end_time-start_time)/25)
    print("===croppping=====")
    process_folder(destination_input_dir, "/home/hln0895/care/0coarse_output/", "/home/hln0895/care/1cropped_output/", padding=20)
    # Step 2: Finer segmentation on cropped images
    finer_segmentation_model_folder = "/home/hln0895/care/model_folder_fine/"
    finer_predictor = nnUNetPredictor(device=device, allow_tqdm=False)
    finer_predictor.initialize_from_trained_model_folder(
        finer_segmentation_model_folder,
        (0,),
        checkpoint_name=os.path.join(finer_segmentation_model_folder, 'checkpoint_best.pth')
    )
    print("===fine segmentation=====")
    finer_predictor.predict_from_files(
        list_of_lists_or_source_folder='/home/hln0895/care/1cropped_output/cropped_images',
        output_folder_or_list_of_truncated_output_files='/home/hln0895/care/2finer_output/',
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1
    )

    print("===reverse segmentation=====")
    label_dir = "/home/hln0895/care/2finer_output/"      # Directory containing the cropped label NIfTI files
    cropping_info_dir = "/home/hln0895/care/1cropped_output/cropping_info/"  # Directory containing cropping info JSON files
    #original_scan_dir = "/home/hln0895/care_validate/"    # Directory containing the original scan NIfTI files
    save_dir = "/home/hln0895/care/3final_output/"      # Directory where the restored labels will be saved
    restore_and_save_all_labels(label_dir, cropping_info_dir, destination_input_dir, save_dir)



    end_time = time.time()
    print('Total time cost:', (end_time - start_time)/25, 's')
