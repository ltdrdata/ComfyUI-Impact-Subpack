import folder_paths
from PIL import Image
import numpy as np
import cv2
import torch


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


def _tensor_check_image(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")
    return


def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))


def dilate_masks(segmasks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return segmasks

    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]

        if dilation_factor > 0:
            dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        else:
            dilated_mask = cv2.erode(cv2_mask, kernel, iter)

        item = (segmasks[i][0], dilated_mask, segmasks[i][2])
        dilated_masks.append(item)

    return dilated_masks


def combine_masks(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0][1])
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i][1])

            if combined_cv2_mask.shape == cv2_mask.shape:
                combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
            else:
                # do nothing - incompatible mask
                pass

        mask = torch.from_numpy(combined_cv2_mask)
        return mask


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

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]


def crop_ndarray2(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[y1:y2, x1:x2]

    return cropped

    
def crop_ndarray4(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped


crop_tensor4 = crop_ndarray4


def crop_image(image, crop_region):
    return crop_tensor4(image, crop_region)
