import folder_paths
from PIL import Image
import numpy as np
import cv2
import torch
import logging # Added for logging warnings

def add_folder_path_and_extensions(folder_name, full_folder_paths, extensions):
    # Iterate over the list of full folder paths
    for full_folder_path in full_folder_paths:
        # Use the provided function to add each model folder path
        folder_paths.add_model_folder_path(folder_name, full_folder_path)

    # Now handle the extensions. If the folder name already exists, update the extensions
    if folder_name in folder_paths.folder_names_and_paths:
        # Unpack the current paths and extensions
        current_paths, current_extensions = folder_paths.folder_names_and_paths[folder_name]
        # Update the extensions set with the new extensions
        updated_extensions = current_extensions | extensions
        # Reassign the updated tuple back to the dictionary
        folder_paths.folder_names_and_paths[folder_name] = (current_paths, updated_extensions)
    else:
        # If the folder name was not present, add_model_folder_path would have added it with the last path
        # Now we just need to update the set of extensions as it would be an empty set
        # Also ensure that all paths are included (since add_model_folder_path adds only one path at a time)
        folder_paths.folder_names_and_paths[folder_name] = (full_folder_paths, extensions)


def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp+size)

    return int(new_startp), int(new_endp)


def tensor2pil(image_tensor: torch.Tensor) -> Image.Image:
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Input must be a PyTorch tensor. Got {type(image_tensor)}")

    processed_tensor = image_tensor.detach().clone() # Work on a detached clone

    # Handle NCHW to NHWC if an NCHW tensor is passed (heuristic based on channel dim)
    # Common ComfyUI IMAGE type is NHWC (batch, height, width, channel)
    # Or HWC for single images not in a batch.
    if processed_tensor.ndim == 4 and processed_tensor.shape[1] in [1, 3, 4] and processed_tensor.shape[3] not in [1,3,4]: # Likely NCHW
        logging.debug(f"tensor2pil received likely NCHW tensor with shape {processed_tensor.shape}, permuting to NHWC.")
        processed_tensor = processed_tensor.permute(0, 2, 3, 1)
    elif processed_tensor.ndim == 3 and processed_tensor.shape[0] in [1,3,4] and processed_tensor.shape[2] not in [1,3,4]: # Likely CHW
        logging.debug(f"tensor2pil received likely CHW tensor with shape {processed_tensor.shape}, permuting to HWC.")
        processed_tensor = processed_tensor.permute(1, 2, 0)


    # Ensure tensor is in NHWC or HWC format before processing
    if processed_tensor.ndim == 4:
        if processed_tensor.shape[0] == 1:
            processed_tensor = processed_tensor.squeeze(0)  # NHWC (N=1) -> HWC
        else:
            # If it's a batch with N > 1, take the first image as per original intent
            logging.warning(f"tensor2pil received a batch of {processed_tensor.shape[0]} images. Processing only the first one.")
            processed_tensor = processed_tensor[0]
            # Now processed_tensor should be H
