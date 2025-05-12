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
            # Now processed_tensor should be HWC

    if processed_tensor.ndim != 3:
        raise ValueError(f"Expected 3D (HWC) or 4D (1HWC preferred) tensor after initial processing, but got shape {image_tensor.shape} (processed to {processed_tensor.shape})")

    # Check channel dimension for HWC
    num_channels = processed_tensor.shape[-1]
    if num_channels not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3, or 4 channels (HWC format), but found {num_channels} channels. Processed shape was {processed_tensor.shape}")

    img_np = processed_tensor.cpu().numpy()

    # Convert data type and range if necessary
    # Common ComfyUI tensors are float32 in range [0.0, 1.0]
    if np.issubdtype(img_np.dtype, np.floating):
        if img_np.min() < -0.001 or img_np.max() > 1.001 : # Check if values are outside typical 0-1 range for floats
            logging.debug(f"Floating point tensor values are outside the expected [0,1] range (min: {img_np.min()}, max: {img_np.max()}). Clamping and scaling.")
        img_np = np.clip(255. * img_np, 0, 255).astype(np.uint8)
    elif img_np.dtype == np.uint8:
        img_np = np.clip(img_np, 0, 255) # Ensure uint8 is also within bounds
    else:
        raise ValueError(f"Unsupported tensor dtype for conversion to PIL: {img_np.dtype}. Expected float or uint8.")

    # Create PIL Image based on channels
    if num_channels == 1:
        # If HWC and C=1, then img_np is (H, W, 1). Squeeze last dim for mode 'L'.
        return Image.fromarray(img_np.squeeze(axis=-1), mode='L')
    elif num_channels == 4:
        return Image.fromarray(img_np, mode='RGBA')
    elif num_channels == 3:
        return Image.fromarray(img_np, mode='RGB')
    else:
        # This case should ideally be caught by the channel check above, but as a fallback:
        raise ValueError(f"Cannot create PIL Image. Unexpected number of channels: {num_channels}")


def dilate_masks(segmasks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return segmasks

    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    for i in range(len(segmasks)):
        # Assuming segmasks[i] is a tuple where segmasks[i][1] is the mask array
        cv2_mask_float = segmasks[i][1] # This was float32 from create_segmasks
        cv2_mask_uint8 = (cv2_mask_float * 255).astype(np.uint8) # Convert to uint8 for OpenCV processing

        if dilation_factor > 0:
            dilated_mask_uint8 = cv2.dilate(cv2_mask_uint8, kernel, iter)
        else:
            dilated_mask_uint8 = cv2.erode(cv2_mask_uint8, kernel, iter)
        
        dilated_mask_float = dilated_mask_uint8.astype(np.float32) / 255.0 # Convert back to float [0,1]

        item = (segmasks[i][0], dilated_mask_float, segmasks[i][2])
        dilated_masks.append(item)

    return dilated_masks


def combine_masks(masks): # Expects masks as list of tuples (bbox, mask_array, conf)
    if not masks: # Check if the list is empty
        return None
    
    # Find a valid mask to initialize combined_mask and get its shape
    initial_mask_array = None
    for m_tuple in masks:
        if isinstance(m_tuple, tuple) and len(m_tuple) > 1 and isinstance(m_tuple[1], np.ndarray):
            initial_mask_array = m_tuple[1]
            break
    
    if initial_mask_array is None: # No valid masks found
        return None

    # Ensure initial_mask_array is boolean or can be converted
    if initial_mask_array.dtype != bool:
        combined_mask_np = initial_mask_array > 0.5 # Example threshold to boolean
    else:
        combined_mask_np = initial_mask_array.copy()


    for i in range(len(masks)):
        if isinstance(masks[i], tuple) and len(masks[i]) > 1 and isinstance(masks[i][1], np.ndarray):
            current_mask_np = masks[i][1]
            if current_mask_np.dtype != bool:
                 current_mask_np = current_mask_np > 0.5 # Threshold to boolean

            if combined_mask_np.shape == current_mask_np.shape:
                combined_mask_np = np.logical_or(combined_mask_np, current_mask_np)
            else:
                logging.warning(f"Skipping mask combination due to shape mismatch: {combined_mask_np.shape} vs {current_mask_np.shape}")
        else:
            logging.warning(f"Skipping invalid mask entry in combine_masks: {masks[i]}")


    return torch.from_numpy(combined_mask_np.astype(np.float32)) # Return as float tensor as often expected by ComfyUI


def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]


def crop_ndarray2(npimg, crop_region): # Expects HWC or HW
    x1, y1, x2, y2 = crop_region
    return npimg[y1:y2, x1:x2, ...] # Ellipsis handles both HW and HWC


def crop_ndarray4(npimg_batch, crop_region): # Expects NHWC
    x1, y1, x2, y2 = crop_region
    return npimg_batch[:, y1:y2, x1:x2, :]


crop_tensor4 = crop_ndarray4


def crop_image(image_tensor, crop_region): # Expects NHWC typically
    if image_tensor.ndim == 3: # HWC
        # Unsqueeze to NHWC, crop, then squeeze back if needed, or handle as HWC directly
        # For simplicity, if this function is only called with NHWC, this branch isn't needed.
        # If called with HWC, crop_ndarray2 might be more appropriate if it's a numpy array.
        # Assuming crop_tensor4 expects batch.
        temp_batch = image_tensor.unsqueeze(0)
        cropped_batch = crop_tensor4(temp_batch, crop_region)
        return cropped_batch.squeeze(0)
    elif image_tensor.ndim == 4: # NHWC
        return crop_tensor4(image_tensor, crop_region)
    else:
        raise ValueError(f"crop_image expects a 3D (HWC) or 4D (NHWC) tensor, got {image_tensor.ndim}D")
