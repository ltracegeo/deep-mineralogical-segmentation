#!/usr/bin/env python-real
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.join("..", ".."))

import traceback

from tqdm import tqdm

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
import numpy as np
import torch
from monai.transforms import MapLabelValued
import xarray as xr

import inference_utils.inference_utils as inf

from mineralogia.inference.MonaiModelsCLILib.models.unet import UNetAct, UNetActWithBoundarySupervision

from mineralogia.inference.MonaiModelsCLILib.transforms import (
    ComposedTransform,
    IdentityTransform,
    ReadNetCDFTransform,
    ReadFirstNetCDFVariableTransform,
    ToTensorTransform,
    QuantileTransform,
    QuantileDeformationTransform,
    ApplyPickledTransform,
    MultiChannelTransform,
    AddAxisTransform,
    SwapAxesTransform,
    PermuteTransform,
    ConcatenateTransform,
    RenameTransform,
    AddBinaryBoundaryMaskTransform,
    BinarizerTransform,
    ArgmaxTransform,
    MinMaxTransform,
    AddConstantTransform,
    TakeChannelsTransform,
)

def make_get_object_by_name(*objects):
    def get_object_by_name(name):
        for obj in objects:
            if obj.__name__.lower() == name.lower():
                return obj
        raise Exception(f"{name} is not defined")

    return get_object_by_name


get_model = make_get_object_by_name(
    UNet,
    UNetAct,
    UNetActWithBoundarySupervision,
    # UNestPretrained,
)

get_transform = make_get_object_by_name(
    ComposedTransform,
    IdentityTransform,
    ReadNetCDFTransform,
    ReadFirstNetCDFVariableTransform,
    ToTensorTransform,
    QuantileTransform,
    QuantileDeformationTransform,
    ApplyPickledTransform,
    MultiChannelTransform,
    AddAxisTransform,
    SwapAxesTransform,
    PermuteTransform,
    ConcatenateTransform,
    RenameTransform,
    BinarizerTransform,
    ArgmaxTransform,
    MinMaxTransform,
    AddConstantTransform,
    AddBinaryBoundaryMaskTransform,
    TakeChannelsTransform,
    MapLabelValued,
)

def run_inference_on_images(images, saved_model, device):
    """run inferece for a model"""

    config = saved_model["config"]
    meta = config["meta"]

    model_name = config["model"]["name"]
    model_params = config["model"].get("params", {})
    model_state_dict = saved_model["model_state_dict"]
    model = get_model(model_name)(**model_params)
    model.load_state_dict(model_state_dict)

    transforms = config.get("transforms", {})
    pre_processing_transforms = transforms.get("pre_processing_transforms", [])
    post_processing_transforms = transforms.get("post_processing_transforms", [])

    pre_processing_transforms = ComposedTransform(
        [get_transform(t["name"])(**t["params"]) for t in pre_processing_transforms]
    )

    post_processing_transforms = ComposedTransform(
        [get_transform(t["name"])(**t["params"]) for t in post_processing_transforms]
    )
    
    input_roi_shape = meta["input_roi_shape"]

    inputs = meta["inputs"]
    pre_processed_inputs = meta.get("pre_processed_inputs", inputs)
    outputs = meta["outputs"]

    required_input_types = list(inputs.keys()) # PP, PX, ...
    pre_processed_input_names = list(pre_processed_inputs.keys())
    output_names = list(outputs.keys())

    # temporary limitation: only volume is fed to the model, only output is accepted
    pre_processed_input_name = pre_processed_input_names[0]
    output_name = output_names[0]

    sample = {}
    for input_type in required_input_types:
        description = inputs[input_type]
        n_channels = description.get("n_channels", 1)

        image = images[input_type]
        
        image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])
        image = image[:n_channels]

        sample[input_type] = torch.as_tensor(image.astype(np.float32))

    sample = pre_processing_transforms(sample)
    # create batch dimension for prediction
    batch = {name: tensor[None, ...] for name, tensor in sample.items()}

    for i in range(2):
        try:
            model.to(device)
            model.eval()
            with torch.no_grad():
                batched_output = {
                    output_name: sliding_window_inference(
                        inputs=batch[pre_processed_input_name],
                        roi_size=input_roi_shape,
                        predictor=model,
                        sw_batch_size=1,
                        sw_device=device,
                    ),
                }
                batched_output = post_processing_transforms(batched_output)
                batched_inference = batched_output[output_name]
            break
        except RuntimeError as e:
            print(e)
            print("PyTorch is not able to use GPU: falling back to CPU.")
            device = "cpu"

    # remove batch and channel dimensions
    output = batched_inference.detach().numpy()[0, 0]

    return output

def run_inference_on_input(input_data, images_keys, is_nc, output_image_path, device, saved_model):
    print("** Running mineralogic segmentation...")
    images = {input_type: input_data[image_key].data[0].astype(np.uint8) for input_type, image_key in images_keys.items()}
            
    # Composed model
    if "models_to_compose" in saved_model.keys():
        inferences = []
        models = saved_model["models_to_compose"]

        for model_key in models:
            inference = run_inference_on_images(images, models[model_key], device)
            inferences.append(inference)

        output = np.zeros_like(inferences[0])

        for combination_rule in saved_model["config"]["inference_combination"]:
            model_index = combination_rule["model_index"]
            values_to_take = combination_rule["take"]

            for take_class_value in values_to_take:
                output[np.where(inferences[model_index] == take_class_value)] = take_class_value

    # Single model
    else:
        output = run_inference_on_images(images, saved_model, device)

    return {"Mineralogy":
                xr.DataArray(
                    np.expand_dims(output, axis=0),
                    dims=("z", "y", "x"),
                    attrs={"type": "segmentation", "labels": inf.get_labels_from_model(saved_model)},
                )
    }

def runcli(args):
    input_image_paths, input_is_dir = inf.check_and_get_input_paths(args.input_path, [".png", ".jpg", ".jpeg", ".nc"])
    output_dir_files, output_is_dir = inf.check_and_get_output_paths(args.output_path, ".nc", input_is_dir, args.overwrite)
    processed = [] # to skip PP/PX already processed when iterating over its counterpart

    device = "cuda" if torch.cuda.is_available() else "cpu"

    saved_model = torch.load(args.input_model, map_location = device)
    input_types = list(saved_model["config"]["meta"]["inputs"].keys())

    """Read input volumes"""
    for input_image_path in tqdm(input_image_paths):
        if input_image_path in processed:
            continue

        print("\n\nRunning inference on", input_image_path)

        input_data = None
        try:
            patterns2search = {"PP": args.pp_pattern, "PX": args.px_pattern}

            input_data, images_keys, images_paths, is_nc, input_image_pattern = inf.load_input(
                input_image_path, input_types, patterns2search, args.mm_per_pixel)
            processed += images_paths

            output_image_path = inf.get_output_path(
                args.output_path,
                output_is_dir,
                args.overwrite,
                ".nc",
                output_dir_files,
                input_image_path,
                input_image_pattern,
                patterns2search.values()
            )

            output_segments = run_inference_on_input(input_data, images_keys, is_nc, output_image_path, device, saved_model)
        
            inf.write_output_and_close_input(output_image_path, output_segments, input_data, is_nc)
        except Exception as e:
            print(f"\nERROR: Could not run inference sucessfully on this image:")
            traceback.print_exc()
            if input_data is not None and is_nc:
                input_data.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mineralogic inference in thin section.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter # para mostrar os valores default na mensagem de ajuda (--help)
    )
    parser.add_argument(
        "--input_model",
        type=argparse.FileType("rb"),
        dest="input_model",
        required=True,
        help="Input model file",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        dest="input_path",
        required=True,
        help="Input image or directory of images. Accepted image formats are PNG, JPEG or netCDF. \
            Images\' types (PP or PX) are inferred based on pp_pattern and px_pattern arguments, if any of them are provided. \
            These patterns are searched in the images\' names, for PNG/JPEG images, or in the nodes\' names, for \
            netCDF. If both PP and PX patterns are requested, the input\'s type is automatically inferred from the \
            patterns and its counterpart is automatically obtained. For example, if input_path=ABC_3200.5_pp.jpg, \
            pp_pattern=pp and px_pattern=c2, then input_path is recognized as PP and, if PX is also required, it \
            is expected that it exists as ABC_3200.5_c2.jpg.",
    )
    parser.add_argument(
        "-pp",
        "--pp_pattern",
        type=str,
        dest="pp_pattern",
        default='PP',
        help="Pattern to be identified in the input image or inside the input netCDF to obtain a PP image. \
            It has a default value since all models require PP. Refer to input_path for knowing more.",
    )
    parser.add_argument(
        "-px",
        "--px_pattern",
        type=str,
        dest="px_pattern",
        default=None,
        help="Pattern to be identified in the input image or inside the input netCDF to obtain a PX image. \
            Must be provided for models that require PX image. Refer to input_path for knowing more.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        dest="output_path",
        default="results",
        help="Output netCDF image or directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If specified, allows the output files to overwrite files with the same name in the output directory. \
                        Otherwise, eventual overwriting is avoided by automatically adding a suffix to the output file.",
    )
    parser.add_argument(
        "--mm_per_pixel",
        type=float,
        default=0.00132,
        help="Pixel size (mm). Used for PNG/JPEG files and ignored for netCDF files \
                        (in this case, the pixel size is obtained from the files themselves).",
    )

    args = parser.parse_args()

    runcli(args)

