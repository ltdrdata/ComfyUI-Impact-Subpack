from PIL import Image

import cv2
import numpy as np
import torch
from collections import namedtuple
from . import utils
import inspect
import logging
import os

import pickle # Correctly imported
import folder_paths


orig_torch_load = torch.load


SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])


# --- Whitelist Configuration ---
WHITELIST_DIR = None
WHITELIST_FILE_PATH = None

try:
    # --- Attempting: Use ComfyUI's folder_paths (Preferred Method) ---
    user_dir = folder_paths.get_user_directory()
    if user_dir and os.path.isdir(user_dir):
        WHITELIST_DIR = os.path.join(user_dir, "default", "ComfyUI-Impact-Subpack")
        WHITELIST_FILE_PATH = os.path.join(WHITELIST_DIR, "model-whitelist.txt")
        logging.info(f"[Impact Pack/Subpack] Using folder_paths to determine whitelist path: {WHITELIST_FILE_PATH}")
    else:
        logging.warning(f"[Impact Pack/Subpack] folder_paths.get_user_directory() returned invalid path: {user_dir}.")
        # Fallback: Use a path relative to this script file if user_dir is not valid
        script_dir = os.path.dirname(os.path.abspath(__file__))
        WHITELIST_DIR = os.path.join(script_dir, "ComfyUI-Impact-Subpack-Data") # Fallback directory
        WHITELIST_FILE_PATH = os.path.join(WHITELIST_DIR, "model-whitelist.txt")
        logging.warning(f"[Impact Pack/Subpack] Fallback: Using script-relative whitelist path: {WHITELIST_FILE_PATH}")


    # --- Ensure directory exists---
    if WHITELIST_FILE_PATH: # Check if any method succeeded in setting the path
        try:
            os.makedirs(WHITELIST_DIR, exist_ok=True)
            logging.info(f"[Impact Pack/Subpack] Ensured whitelist directory exists: {WHITELIST_DIR}")
        except OSError as e:
            logging.error(f"[Impact Pack/Subpack] Failed to create whitelist directory {WHITELIST_DIR}: {e}. Whitelisting may not function.")
            WHITELIST_FILE_PATH = None # Indicate failure / disable whitelisting
        except Exception as e:
            logging.error(f"[Impact Pack/Subpack] Unexpected error creating whitelist directory: {e}", exc_info=True)
            WHITELIST_FILE_PATH = None # Indicate failure / disable whitelisting
    else:
         # Handle case where path determination failed via all methods
         logging.error("[Impact Pack/Subpack] Whitelist path determination failed using all methods. Whitelisting disabled.")
         # WHITELIST_FILE_PATH is already None


except Exception as e:
    # Catch errors during the whole setup process (e.g., inspect failing)
    logging.error(f"[Impact Pack/Subpack] Critical error during whitelist path setup: {e}", exc_info=True)
    WHITELIST_FILE_PATH = None # Disable whitelisting on critical setup error
    logging.error("[Impact Pack/Subpack] Whitelisting disabled due to critical setup error.")


def load_whitelist(filepath):
    """
    Loads filenames from the whitelist file.
    Attempts to create the file with instructions if it doesn't exist.
    Returns a set of approved base filenames.
    """
    approved_files = set()
    if filepath is None or not isinstance(filepath, str):
        return approved_files

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    approved_files.add(os.path.basename(line))
        logging.info(f"[Impact Pack/Subpack] Loaded {len(approved_files)} model(s) from whitelist: {filepath}")

    except FileNotFoundError:
        logging.warning(f"[Impact Pack/Subpack] Model whitelist file not found at: {filepath}. ")
        logging.warning(f" >> An empty whitelist file will be created with instructions.")
        try:
            with open(filepath, 'w') as f:
                f.write("# Add base filenames of trusted models (e.g., my_old_yolo.pt) here, one per line.\n")
                f.write("# This allows loading them with `weights_only=False` if they fail safe loading\n")
                f.write("# due to errors like 'restricted getattr' in newer PyTorch versions.\n")
                f.write("# WARNING: Only add files you absolutely trust, as this bypasses a security feature.\n")
                f.write("# Files automatically added by the script due to safe load failures will also appear here.\n")
                f.write("# Prefer using .safetensors files whenever possible.\n")
            logging.info(f"[Impact Pack/Subpack] Created empty whitelist file: {filepath}")
        except Exception as create_e:
            logging.error(f"[Impact Pack/Subpack] Failed to create empty whitelist file at {filepath}: {create_e}", exc_info=True)

    except Exception as e:
        logging.error(f"[Impact Pack/Subpack] Error loading model whitelist from {filepath}: {e}", exc_info=True)

    return approved_files

def _add_to_whitelist_file_and_reload(filename_to_add, whitelist_filepath):
    """
    Adds a filename to the whitelist file and reloads the in-memory whitelist.
    Returns the updated set of approved files.
    """
    global _MODEL_WHITELIST
    if whitelist_filepath is None or not isinstance(whitelist_filepath, str) or filename_to_add is None:
        logging.error("[Impact Pack/Subpack] Cannot add to whitelist: Invalid filepath or filename.")
        return _MODEL_WHITELIST # Return current whitelist

    try:
        # Ensure the directory exists (it should by now, but double-check)
        os.makedirs(os.path.dirname(whitelist_filepath), exist_ok=True)

        # Check if file already contains the entry to avoid duplicates (optional but good practice)
        current_entries = set()
        try:
            with open(whitelist_filepath, 'r') as f_read:
                for line in f_read:
                    current_entries.add(line.strip())
        except FileNotFoundError:
            pass # File will be created by load_whitelist or append mode

        if os.path.basename(filename_to_add) not in current_entries:
            with open(whitelist_filepath, 'a') as f_append: # Append mode
                f_append.write(f"\n{os.path.basename(filename_to_add)}") # Add on a new line
            logging.info(f"[Impact Pack/Subpack] Automatically added '{os.path.basename(filename_to_add)}' to whitelist file: {whitelist_filepath}")
        else:
            logging.info(f"[Impact Pack/Subpack] Filename '{os.path.basename(filename_to_add)}' already in whitelist file: {whitelist_filepath}")

        # Reload the whitelist to update the in-memory set
        _MODEL_WHITELIST = load_whitelist(whitelist_filepath)
        return _MODEL_WHITELIST

    except Exception as e:
        logging.error(f"[Impact Pack/Subpack] Failed to automatically add '{filename_to_add}' to whitelist file {whitelist_filepath}: {e}", exc_info=True)
        return _MODEL_WHITELIST # Return current (potentially unchanged) whitelist


_MODEL_WHITELIST = load_whitelist(WHITELIST_FILE_PATH)
# ---------- End of Whitelist Management ----------

class NO_BBOX_DETECTOR:
    pass

class NO_SEGM_DETECTOR:
    pass

def create_segmasks(results):
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    output_results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        output_results.append(item)
    return output_results


def restricted_getattr(obj, name, *args):
    if name != "forward":
        logging.error(f"Access to potentially dangerous attribute '{obj.__module__}.{obj.__name__}.{name}' is blocked.\nIf you believe the use of this code is genuinely safe, please report it.\nhttps://github.com/ltdrdata/ComfyUI-Impact-Subpack/issues")
        raise RuntimeError(f"Access to potentially dangerous attribute '{obj.__module__}.{obj.__name__}.{name}' is blocked.")
    return getattr(obj, name, *args)

restricted_getattr.__module__ = 'builtins'
restricted_getattr.__name__ = 'getattr'

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel
    from ultralytics.utils import IterableSimpleNamespace
    from ultralytics.utils.tal import TaskAlignedAssigner
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block_modules
    import torch.nn.modules as torch_modules
    import ultralytics.utils.loss as loss_modules
    import dill._dill
    from numpy.core.multiarray import scalar

    try:
        from numpy import dtype
        from numpy.dtypes import Float64DType
    except ImportError:
        logging.warning("[Impact Subpack] Could not import Float64DType from numpy.dtypes. This might affect loading models saved with newer numpy versions if you have an older numpy.")
        from numpy import dtype
        Float64DType = type(np.float64(0))
        logging.warning("[Impact Subpack] Falling back to using type(np.float64(0)) for Float64DType. Please consider updating numpy to >=1.24 for full compatibility.")


    torch_whitelist = []

    def build_torch_whitelist():
        global torch_whitelist
        # ... (build_torch_whitelist content remains the same as your last provided version) ...
        for name, obj in inspect.getmembers(modules):
            if inspect.isclass(obj) and obj.__module__.startswith("ultralytics.nn.modules"):
                aliasObj = type(name, (obj,), {})
                aliasObj.__module__ = "ultralytics.nn.modules"
                torch_whitelist.append(obj)
                torch_whitelist.append(aliasObj)

        for name, obj in inspect.getmembers(block_modules):
            if inspect.isclass(obj) and obj.__module__.startswith("ultralytics.nn.modules"):
                aliasObj = type(name, (obj,), {})
                aliasObj.__module__ = "ultralytics.nn.modules.block"
                torch_whitelist.append(obj)
                torch_whitelist.append(aliasObj)

        for name, obj in inspect.getmembers(loss_modules):
            if inspect.isclass(obj) and obj.__module__.startswith("ultralytics.utils.loss"):
                aliasObj = type(name, (obj,), {})
                aliasObj.__module__ = "ultralytics.yolo.utils.loss"
                torch_whitelist.append(obj)
                torch_whitelist.append(aliasObj)

        for name, obj in inspect.getmembers(torch_modules):
            if inspect.isclass(obj) and obj.__module__.startswith("torch.nn.modules"):
                torch_whitelist.append(obj)

        aliasIterableSimpleNamespace = type("IterableSimpleNamespace", (IterableSimpleNamespace,), {})
        aliasIterableSimpleNamespace.__module__ = "ultralytics.yolo.utils"

        aliasTaskAlignedAssigner = type("TaskAlignedAssigner", (TaskAlignedAssigner,), {})
        aliasTaskAlignedAssigner.__module__ = "ultralytics.yolo.utils.tal"

        aliasYOLOv10DetectionModel = type("YOLOv10DetectionModel", (DetectionModel,), {})
        aliasYOLOv10DetectionModel.__module__ = "ultralytics.nn.tasks"
        aliasYOLOv10DetectionModel.__name__ = "YOLOv10DetectionModel"
        
        if hasattr(loss_modules, 'E2EDetectLoss'):
            aliasv10DetectLoss = type("v10DetectLoss", (loss_modules.E2EDetectLoss,), {})
            aliasv10DetectLoss.__name__ = "v10DetectLoss"
            aliasv10DetectLoss.__module__ = "ultralytics.utils.loss"
        else:
            class DummyLoss: pass
            aliasv10DetectLoss = type("v10DetectLoss", (DummyLoss,), {}) 
            logging.warning("[Impact Pack/Subpack] loss_modules.E2EDetectLoss not found for aliasv10DetectLoss.")


        torch_whitelist += [
            DetectionModel, aliasYOLOv10DetectionModel, SegmentationModel,
            IterableSimpleNamespace, aliasIterableSimpleNamespace,
            TaskAlignedAssigner, aliasTaskAlignedAssigner, aliasv10DetectLoss,
            restricted_getattr, dill._dill._load_type, scalar, dtype, Float64DType
        ]
    build_torch_whitelist()

except ImportError as e:
    logging.error(f"[Impact Pack/Subpack] Failed to import ultralytics or its dependencies: {e}")
    logging.error("\n!!!!!\n\n[ComfyUI-Impact-Subpack] If this error occurs, please check the following link:\n\thttps://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/Main/troubleshooting/TROUBLESHOOTING.md\n\n!!!!!\n")
    YOLO = None
    class DetectionModel: pass
    class SegmentationModel: pass
except Exception as e:
    logging.error(f"[Impact Pack/Subpack] General error during ultralytics setup: {e}", exc_info=True)
    logging.error("\n!!!!!\n\n[ComfyUI-Impact-Subpack] If this error occurs, please check the following link:\n\thttps://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/Main/troubleshooting/TROUBLESHOOTING.md\n\n!!!!!\n")
    raise e

def torch_wrapper(*args, **kwargs):
    global _MODEL_WHITELIST

    filename = None
    filename_arg_source = "[unknown source]"
    if args and isinstance(args[0], str):
        filename = os.path.basename(args[0])
        filename_arg_source = args[0]
    elif 'f' in kwargs and isinstance(kwargs['f'], str):
        filename = os.path.basename(kwargs['f'])
        filename_arg_source = kwargs['f']

    if hasattr(torch.serialization, 'safe_globals'):
        load_kwargs_attempt1 = kwargs.copy()
        effective_wo_attempt1 = load_kwargs_attempt1.get('weights_only', True)

        logging.debug(f"[Impact Pack/Subpack] Attempting first load for: {filename_arg_source}. "
                      f"Effective 'weights_only' for this attempt (True if unspecified on PT>=2.6): {effective_wo_attempt1}")
        try:
            return orig_torch_load(*args, **load_kwargs_attempt1)

        except pickle.UnpicklingError as e:
            is_disallowed_global_error = 'getattr' in str(e) or "Unsupported global" in str(e)

            if is_disallowed_global_error and effective_wo_attempt1:
                # Check current whitelist (reloaded one if previously reloaded in this call)
                if filename and filename in _MODEL_WHITELIST:
                    logging.warning("##############################################################################")
                    logging.warning(f"[Impact Pack/Subpack] WARNING: Safe load failed for '{filename}' (Reason: {e}).")
                    logging.warning(f" >> FILE IS ALREADY IN THE WHITELIST: {WHITELIST_FILE_PATH}")
                    logging.warning(" >> This model likely uses legacy Python features blocked by default for security.")
                    logging.warning(" >> RETRYING WITH 'weights_only=False' as it's whitelisted.")
                    logging.warning("##############################################################################")
                    retry_kwargs = kwargs.copy()
                    retry_kwargs['weights_only'] = False
                    return orig_torch_load(*args, **retry_kwargs)
                else:
                    # File not in whitelist. AUTOMATICALLY ADD AND RETRY.
                    logging.warning("##############################################################################")
                    logging.warning(f"[Impact Pack/Subpack] CAUTION: Safe load failed for '{filename_arg_source}' (Reason: {e}).")
                    logging.warning(f" >> File '{filename}' was NOT found in the whitelist: {WHITELIST_FILE_PATH}")
                    logging.warning(f" >> AUTOMATICALLY ADDING '{filename}' TO WHITELIST and retrying with 'weights_only=False'.")
                    logging.warning(" >> SECURITY IMPLICATION: This model will now be trusted for unsafe loading.")
                    logging.warning(" >> If you did not intend this, remove it from the whitelist file and restart.")
                    logging.warning(" >> Prefer using .safetensors files whenever available for better security.")
                    logging.warning("##############################################################################")

                    if filename and WHITELIST_FILE_PATH:
                        _MODEL_WHITELIST = _add_to_whitelist_file_and_reload(filename, WHITELIST_FILE_PATH)
                        # After adding and reloading, check again (it should be there now)
                        if filename in _MODEL_WHITELIST:
                            logging.info(f"[Impact Pack/Subpack] Successfully added '{filename}' to in-memory whitelist. Proceeding with unsafe load.")
                            retry_kwargs = kwargs.copy()
                            retry_kwargs['weights_only'] = False
                            return orig_torch_load(*args, **retry_kwargs)
                        else:
                            logging.error(f"[Impact Pack/Subpack] Failed to verify '{filename}' in whitelist after automatic add attempt. Blocking load.")
                            # Fall through to raise the original error if auto-add logic failed verification
                    else:
                        logging.error("[Impact Pack/Subpack] Cannot automatically add to whitelist: Filename or whitelist path is missing. Blocking load.")
                    
                    # If auto-add failed or filename/path was missing, raise original error
                    logging.error("##############################################################################")
                    logging.error(f"[Impact Pack/Subpack] BLOCKED (after auto-add attempt failed or was not possible): Safe load failed for '{filename_arg_source}' (Reason: {e}).")
                    logging.error(f" >> Whitelist path: {WHITELIST_FILE_PATH if WHITELIST_FILE_PATH else '[Path not determined]'}")
                    logging.error("##############################################################################")
                    raise e # Re-raise the original security-related error
            else: # Not the specific error we handle for whitelisting, or first attempt was not safe
                logging.error(f"[Impact Pack/Subpack] UnpicklingError during load for '{filename_arg_source}'. Error: {e}. "
                              f"First attempt 'weights_only' was {effective_wo_attempt1}. "
                              "Not an auto-whitelisting scenario. Re-raising.")
                raise e
        except Exception as general_e: # Catch other errors like RuntimeError, etc.
            logging.error(f"[Impact Pack/Subpack] A general exception occurred during torch.load for {filename_arg_source}: {general_e}", exc_info=True)
            raise general_e
    else: # Older PyTorch (no safe_globals)
        load_kwargs_old_torch = kwargs.copy()
        effective_wo_old_torch = load_kwargs_old_torch.get('weights_only', False)
        if not effective_wo_old_torch:
            logging.warning(f"[Impact Pack/Subpack] Older PyTorch version detected. Proceeding with potentially unsafe load "
                            f"(effective weights_only=False) for: {filename_arg_source}")
        else:
            logging.debug(f"[Impact Pack/Subpack] Older PyTorch version detected. Proceeding with explicit weights_only=True "
                          f"for: {filename_arg_source}")
        return orig_torch_load(*args, **load_kwargs_old_torch)

torch.load = torch_wrapper

def load_yolo(model_path: str):
    if YOLO is None:
        logging.error("[Impact Pack/Subpack] YOLO class not available. Cannot load YOLO model.")
        raise RuntimeError("[Impact Pack/Subpack] YOLO could not be imported. Please check ultralytics installation.")

    current_safe_globals = list(getattr(torch.serialization, 'get_safe_globals', lambda: [])())
    for item in torch_whitelist:
        if inspect.isclass(item) and item not in current_safe_globals:
            current_safe_globals.append(item)

    if hasattr(torch.serialization, 'safe_globals'):
        with torch.serialization.safe_globals(current_safe_globals):
            try:
                return YOLO(model_path)
            except ModuleNotFoundError:
                logging.warning("[Impact Pack/Subpack] ModuleNotFoundError during YOLO load, attempting fallback with yolov8n.pt")
                YOLO("yolov8n.pt") # Attempt to initialize with a default model
                return YOLO(model_path) # Retry loading the target model
            except pickle.UnpicklingError as e:
                # This catch is for UnpicklingErrors raised directly by YOLO() constructor,
                # potentially not via the torch.load wrapper (e.g. if YOLO unpickles other config files)
                # The torch.load of weights is handled by our torch_wrapper.
                # If the error is the same "Weights only load failed..." it means it still went through torch.load
                # which should have been handled by torch_wrapper. This path indicates either
                # A) YOLO does its own unpickling apart from weights, or B) an unexpected error flow.
                logging.error(f"[Impact Pack/Subpack] UnpicklingError in YOLO constructor for {model_path} (within safe_globals): {e}", exc_info=True)
                logging.error(f" >> This error might indicate an issue with the model's pickled structure itself, beyond just weights, or an unhandled path for torch.load.")
                raise e
            except Exception as e_other:
                logging.error(f"[Impact Pack/Subpack] Other exception in YOLO constructor for {model_path} (within safe_globals): {e_other}", exc_info=True)
                raise e_other
    else: # Older PyTorch or safe_globals not available
        try:
            return YOLO(model_path)
        except ModuleNotFoundError:
            logging.warning("[Impact Pack/Subpack] ModuleNotFoundError during YOLO load (older PyTorch), attempting fallback with yolov8n.pt")
            YOLO("yolov8n.pt")
            return YOLO(model_path)
        except pickle.UnpicklingError as e:
            logging.error(f"[Impact Pack/Subpack] UnpicklingError in YOLO constructor for {model_path} (older PyTorch): {e}", exc_info=True)
            raise e
        except Exception as e_other:
            logging.error(f"[Impact Pack/Subpack] Other exception in YOLO constructor for {model_path} (older PyTorch): {e_other}", exc_info=True)
            raise e_other

# --- inference_bbox, inference_segm, UltraBBoxDetector, UltraSegmDetector classes remain unchanged ---
# (Assuming their last provided version was correct and complete)

def inference_bbox(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
):
    pred = model(image, conf=confidence, device=device if device else None)

    if not pred or not pred[0].boxes:
        return [[], [], [], []]
        
    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if not bboxes.any(): 
        return [[], [], [], []]

    cv2_image = np.array(image.convert("RGB"))
    cv2_image = cv2_image[:, :, ::-1].copy()

    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    output_results = [[], [], [], []]
    for i in range(len(bboxes)):
        output_results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        output_results[1].append(bboxes[i])
        output_results[2].append(segms[i])
        conf_value = pred[0].boxes[i].conf
        output_results[3].append(conf_value.cpu().numpy() if hasattr(conf_value, 'cpu') else np.array([conf_value.item()]))
    return output_results


def inference_segm(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
):
    pred = model(image, conf=confidence, device=device if device else None)

    if not pred or not pred[0].boxes:
         return [[], [], [], []]

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if not bboxes.any():
        return [[], [], [], []]

    h_orig, w_orig = image.size[1], image.size[0]
    scaled_masks_np = []

    if pred[0].masks is None:
        logging.warning("[Impact Pack/Subpack] Segmentation model called, but no masks found in prediction.")
        scaled_masks_np = [np.zeros((h_orig, w_orig), dtype=bool) for _ in bboxes]
    else:
        segms_data = pred[0].masks.data.cpu().numpy()
        if segms_data.ndim < 3 or segms_data.shape[0] == 0:
             scaled_masks_np = [np.zeros((h_orig, w_orig), dtype=bool) for _ in bboxes]
        else:
            for i in range(segms_data.shape[0]):
                mask_tensor = torch.from_numpy(segms_data[i])
                scaled_mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                    size=(h_orig, w_orig),
                    mode='bilinear',
                    align_corners=False
                )
                scaled_mask = scaled_mask_tensor.squeeze().squeeze() > 0.5
                scaled_masks_np.append(scaled_mask.numpy())
    
    output_results = [[], [], [], []]
    for i in range(len(bboxes)):
        output_results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        output_results[1].append(bboxes[i])
        output_results[2].append(scaled_masks_np[i] if i < len(scaled_masks_np) else np.zeros((h_orig, w_orig), dtype=bool) )
        conf_value = pred[0].boxes[i].conf
        output_results[3].append(conf_value.cpu().numpy() if hasattr(conf_value, 'cpu') else np.array([conf_value.item()]))
    return output_results


class UltraBBoxDetector:
    bbox_model = None
    def __init__(self, bbox_model):
        self.bbox_model = bbox_model
    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        pil_image = utils.tensor2pil(image[0] if image.ndim == 4 and image.shape[0] == 1 else image)
        detected_results = inference_bbox(self.bbox_model, pil_image, threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)
        items = []
        img_h = image.shape[1 if image.ndim == 4 else 0]
        img_w = image.shape[2 if image.ndim == 4 else 1]
        for i, item_tuple in enumerate(segmasks):
            item_bbox_coords, item_mask_array, confidence_val = item_tuple
            label = detected_results[0][i]
            x1, y1, x2, y2 = item_bbox_coords
            if (x2 - x1) >= drop_size and (y2 - y1) >= drop_size:
                crop_region_coords = utils.make_crop_region(img_w, img_h, item_bbox_coords, crop_factor)
                if detailer_hook is not None and hasattr(detailer_hook, 'post_crop_region'):
                    crop_region_coords = detailer_hook.post_crop_region(img_w, img_h, item_bbox_coords, crop_region_coords)
                cropped_image_tensor = utils.crop_image(image, crop_region_coords)
                cropped_mask_array = utils.crop_ndarray2(item_mask_array, crop_region_coords)
                item = SEG(cropped_image_tensor, cropped_mask_array, confidence_val, crop_region_coords, item_bbox_coords, label, None)
                items.append(item)
        output_shape = (img_h, img_w)
        segs = output_shape, items
        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)
        return segs
    def detect_combined(self, image, threshold, dilation):
        pil_image = utils.tensor2pil(image[0] if image.ndim == 4 and image.shape[0] == 1 else image)
        detected_results = inference_bbox(self.bbox_model, pil_image, threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)
        return utils.combine_masks(segmasks)
    def setAux(self, x):
        pass

class UltraSegmDetector:
    segm_model = None
    def __init__(self, segm_model):
        self.segm_model = segm_model
    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        pil_image = utils.tensor2pil(image[0] if image.ndim == 4 and image.shape[0] == 1 else image)
        detected_results = inference_segm(self.segm_model, pil_image, threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)
        items = []
        img_h = image.shape[1 if image.ndim == 4 else 0]
        img_w = image.shape[2 if image.ndim == 4 else 1]
        for i, item_tuple in enumerate(segmasks):
            item_bbox_coords, item_mask_array, confidence_val = item_tuple
            label = detected_results[0][i]
            x1, y1, x2, y2 = item_bbox_coords
            if (x2 - x1) >= drop_size and (y2 - y1) >= drop_size:
                crop_region_coords = utils.make_crop_region(img_w, img_h, item_bbox_coords, crop_factor)
                if detailer_hook is not None and hasattr(detailer_hook, 'post_crop_region'):
                    crop_region_coords = detailer_hook.post_crop_region(img_w, img_h, item_bbox_coords, crop_region_coords)
                cropped_image_tensor = utils.crop_image(image, crop_region_coords)
                cropped_mask_array = utils.crop_ndarray2(item_mask_array, crop_region_coords)
                item = SEG(cropped_image_tensor, cropped_mask_array, confidence_val, crop_region_coords, item_bbox_coords, label, None)
                items.append(item)
        output_shape = (img_h, img_w)
        segs = output_shape, items
        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)
        return segs
    def detect_combined(self, image, threshold, dilation):
        pil_image = utils.tensor2pil(image[0] if image.ndim == 4 and image.shape[0] == 1 else image)
        detected_results = inference_segm(self.segm_model, pil_image, threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)
        return utils.combine_masks(segmasks)
    def setAux(self, x):
        pass
