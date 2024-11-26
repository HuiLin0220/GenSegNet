import os
import torch
from ptflops import get_model_complexity_info
import argparse

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate MACs and parameters for a model.')

    parser.add_argument('--plan_path', type=str, required=True, help='Path to the nnUNet plans.json file')
    parser.add_argument('--dataset_json_path', type=str, required=True, help='Path to the dataset.json file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file (.pth)')
    parser.add_argument('--trainer_root', type=str, required=True, help='Root path to the nnUNetTrainer folder')
    parser.add_argument('--input_shape', type=int, nargs='+', required=True,
                        help='Input shape for the model (C, D, H, W)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load Plans and Dataset JSON
    plans_manager = PlansManager(args.plan_path)
    configuration_manager = plans_manager.get_configuration('3d_fullres')
    dataset_json = load_json(args.dataset_json_path)

    # Load Checkpoint and find the trainer
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    trainer_name = checkpoint['trainer_name']

    # Find the trainer class by name
    trainer_class = recursive_find_python_class(args.trainer_root, trainer_name, 'nnunetv2.training.nnUNetTrainer')

    # Determine number of input channels
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)

    # Build the model
    model = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False
    )

    # Calculate MACs and parameters
    input_shape = tuple(args.input_shape)  # Convert list of integers to tuple
    macs, params = get_model_complexity_info(
        model,
        input_shape,  # Pass the input shape provided in the command line
        as_strings=True,
        print_per_layer_stat=False,
        verbose=True
    )

    # Print the results
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    main()
