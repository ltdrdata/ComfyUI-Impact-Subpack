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
        logging.warning(f" >> An empty whitelist file will be created.")
        logging.warning(f" >> To allow unsafe loading for specific trusted legacy models (e.g., older .pt),")
        logging.warning(f" >> add their base filenames (one per line) to this file.")
        try:
            with open(filepath, 'w') as f:
                f.write("# Add base filenames of trusted models (e.g., my_old_yolo.pt) here, one per line.\n")
                f.write("# This allows loading them with `weights_only=False` if they fail safe loading\n")
                f.write("# due to errors like 'restricted getattr' in newer PyTorch versions.\n")
                f.write("# WARNING: Only add files you absolutely trust, as this bypasses a security feature.\n")
                f.write("# Prefer using .safetensors files whenever possible.\n")
            logging.info(f"[Impact Pack/Subpack] Created empty whitelist file: {filepath}")
        except Exception as create_e:
            logging.error(f"[Impact Pack/Subpack] Failed to create empty whitelist file at {filepath}: {create_e}", exc_info=True)

    except Exception as e:
        logging.error(f"[Impact Pack/Subpack] Error loading model whitelist from {filepath}: {e}", exc_info=True)

    return approved_files

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

    output_results = [] # Renamed to avoid conflict with the outer scope 'results' name
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

# Attempt to import ultralytics and related components
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
        from numpy.dtypes import Float64DType # For numpy 1.24+
    except ImportError:
        logging.warning("[Impact Subpack] Could not import Float64DType from numpy.dtypes. This might affect loading models saved with newer numpy versions if you have an older numpy.")
        from numpy import dtype
        Float64DType = type(np.float64(0))
        logging.warning("[Impact Subpack] Falling back to using type(np.float64(0)) for Float64DType. Please consider updating numpy to >=1.24 for full compatibility.")


    torch_whitelist = []

    def build_torch_whitelist():
        global torch_whitelist

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

        # Assuming E2EDetectLoss exists in loss_modules
        if hasattr(loss_modules, 'E2EDetectLoss'):
            aliasv10DetectLoss = type("v10DetectLoss", (loss_modules.E2EDetectLoss,), {})
            aliasv10DetectLoss.__name__ = "v10DetectLoss"
            aliasv10DetectLoss.__module__ = "ultralytics.utils.loss"
        else:
            # Fallback if E2EDetectLoss is not found, to prevent error during build_torch_whitelist
            class DummyLoss: pass
            aliasv10DetectLoss = type("v10DetectLoss", (DummyLoss,), {}) # Create a dummy type
            logging.warning("[Impact Pack/Subpack] loss_modules.E2EDetectLoss not found. aliasv10DetectLoss created as a dummy type.")


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


# --- Start: REPLACE the existing torch_wrapper function ---
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

        except pickle.UnpicklingError as e: # Catch standard pickle.UnpicklingError
            is_disallowed_global_error = 'getattr' in str(e) or "Unsupported global" in str(e)

            if is_disallowed_global_error and effective_wo_attempt1:
                if filename and filename in _MODEL_WHITELIST:
                    logging.warning("##############################################################################")
                    logging.warning(f"[Impact Pack/Subpack] WARNING: Safe load failed for '{filename}' (Reason: {e}).")
                    logging.warning(f" >> FILE IS IN THE WHITELIST: {WHITELIST_FILE_PATH}")
                    logging.warning(" >> This model likely uses legacy Python features blocked by default for security.")
                    logging.warning(" >> RETRYING WITH 'weights_only=False' because it's whitelisted.")
                    logging.warning(" >> SECURITY RISK: Ensure you added this file to the whitelist consciously")
                    logging.warning(f" >> and trust its source: {filename_arg_source}")
                    logging.warning(" >> Prefer using .safetensors files whenever available.")
                    logging.warning("##############################################################################")

                    retry_kwargs = kwargs.copy()
                    retry_kwargs['weights_only'] = False
                    return orig_torch_load(*args, **retry_kwargs)
                else:
                    logging.warning(f"[Impact Pack/Subpack] File '{filename}' not found in current whitelist cache.")
                    whitelist_path_msg = WHITELIST_FILE_PATH if WHITELIST_FILE_PATH else "[Path not determined]"
                    logging.info(f"[Impact Pack/Subpack] Attempting to reload whitelist from: {whitelist_path_msg}")
                    try:
                        _MODEL_WHITELIST = load_whitelist(WHITELIST_FILE_PATH)
                        logging.info(f"[Impact Pack/Subpack] Whitelist reloaded. Now contains {len(_MODEL_WHITELIST)} entries.")

                        if filename and filename in _MODEL_WHITELIST:
                            logging.warning("##############################################################################")
                            logging.warning(f"[Impact Pack/Subpack] SUCCESS: File '{filename}' FOUND in reloaded whitelist.")
                            logging.warning(f" >> Proceeding with whitelisted unsafe load (weights_only=False).")
                            logging.warning(f" >> Ensure you recently added this file to: {whitelist_path_msg}")
                            logging.warning(" >> SECURITY RISK: Ensure you trust its source.")
                            logging.warning("##############################################################################")
                            retry_kwargs = kwargs.copy()
                            retry_kwargs['weights_only'] = False
                            return orig_torch_load(*args, **retry_kwargs)
                        else:
                            logging.error("[Impact Pack/Subpack] File still not found in whitelist after reload.")
                    except Exception as reload_e:
                        logging.error(f"[Impact Pack/Subpack] Error occurred during whitelist reload attempt: {reload_e}", exc_info=True)

                    logging.error("##############################################################################")
                    logging.error(f"[Impact Pack/Subpack] ERROR: Safe load failed for '{filename_arg_source}' (Reason: {e}).")
                    logging.error(f" >> This model likely uses legacy Python features blocked by default for security.")
                    logging.error(f" >> UNSAFE LOAD BLOCKED because the file ('{filename or 'unknown'}') is NOT in the whitelist (even after reload attempt).")
                    logging.error(f" >> Whitelist path: {whitelist_path_msg}")
                    if filename:
                        logging.error(f" >> To allow loading this specific file (IF YOU TRUST IT), ensure its base name")
                        logging.error(f" >> ('{filename}') is correctly added to the whitelist file (one name per line) and saved.")
                    else:
                        logging.error(f" >> Cannot determine filename to check against whitelist.")
                    logging.error(" >> SECURITY RISK: Only whitelist files from sources you absolutely trust.")
                    logging.error(" >> Prefer using .safetensors files whenever available.")
                    logging.error("##############################################################################")
                    raise e
            else:
                logging.error(f"[Impact Pack/Subpack] UnpicklingError during load for '{filename_arg_source}'. Error: {e}. "
                              f"First attempt 'weights_only' was {effective_wo_attempt1}. "
                              "Not a whitelisting scenario or already tried unsafe. Re-raising.")
                raise e
        except Exception as general_e:
            logging.error(f"[Impact Pack/Subpack] A general exception occurred during torch.load for {filename_arg_source}: {general_e}", exc_info=True)
            raise general_e
    else:
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
# --- End: Replacement block for the torch_wrapper function ---


def load_yolo(model_path: str):
    if YOLO is None:
        logging.error("[Impact Pack/Subpack] YOLO class not available. Cannot load YOLO model.")
        raise RuntimeError("[Impact Pack/Subpack] YOLO could not be imported. Please check ultralytics installation.")

    current_safe_globals = list(getattr(torch.serialization, 'get_safe_globals', lambda: [])()) # Handle if get_safe_globals doesn't exist

    # Add types from torch_whitelist that are classes
    for item in torch_whitelist: # torch_whitelist should be defined
        if inspect.isclass(item) and item not in current_safe_globals:
            current_safe_globals.append(item)

    if hasattr(torch.serialization, 'safe_globals'):
        with torch.serialization.safe_globals(current_safe_globals):
            try:
                return YOLO(model_path)
            except ModuleNotFoundError:
                logging.warning("[Impact Pack/Subpack] ModuleNotFoundError during YOLO load, attempting fallback with yolov8n.pt")
                YOLO("yolov8n.pt")
                return YOLO(model_path)
            except pickle.UnpicklingError as e: # MODIFIED: Changed from _pickle.UnpicklingError
                logging.error(f"[Impact Pack/Subpack] UnpicklingError directly in YOLO constructor for {model_path}: {e}", exc_info=True)
                logging.error(f" >> This could be due to the model structure itself, not just weights.")
                logging.error(f" >> Ensure all necessary classes are in torch_whitelist and potentially safe_globals if this persists.")
                raise e
            except Exception as e_other: # Catch any other exceptions
                logging.error(f"[Impact Pack/Subpack] Other exception in YOLO constructor for {model_path} within safe_globals context: {e_other}", exc_info=True)
                raise e_other
    else: # Older PyTorch or safe_globals not available
        try:
            return YOLO(model_path)
        except ModuleNotFoundError:
            logging.warning("[Impact Pack/Subpack] ModuleNotFoundError during YOLO load (older PyTorch), attempting fallback with yolov8n.pt")
            YOLO("yolov8n.pt")
            return YOLO(model_path)
        except pickle.UnpicklingError as e: # MODIFIED: Changed from _pickle.UnpicklingError
            logging.error(f"[Impact Pack/Subpack] UnpicklingError directly in YOLO constructor for {model_path} (older PyTorch): {e}", exc_info=True)
            raise e
        except Exception as e_other: # Catch any other exceptions
            logging.error(f"[Impact Pack/Subpack] Other exception in YOLO constructor for {model_path} (older PyTorch): {e_other}", exc_info=True)
            raise e_other


def inference_bbox(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
):
    pred = model(image, conf=confidence, device=device if device else None)

    # Check if pred[0].boxes is None or empty before accessing attributes
    if not pred or not pred[0].boxes:
        return [[], [], [], []]
        
    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if not bboxes.any(): # Check if bboxes numpy array is empty
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
        # Ensure conf is correctly accessed and converted
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

    if pred[0].masks is None:
        logging.warning("[Impact Pack/Subpack] Segmentation model called, but no masks found in prediction.")
        # Create empty boolean masks matching image size for each bounding box
        h_orig, w_orig = image.size[1], image.size[0]
        scaled_masks_np = [np.zeros((h_orig, w_orig), dtype=bool) for _ in bboxes]
    else:
        segms_data = pred[0].masks.data.cpu().numpy()
        h_orig, w_orig = image.size[1], image.size[0]
        scaled_masks_np = []

        if segms_data.ndim < 3 or segms_data.shape[0] == 0: # No masks to process
             scaled_masks_np = [np.zeros((h_orig, w_orig), dtype=bool) for _ in bboxes]
        else:
            for i in range(segms_data.shape[0]):
                mask_tensor = torch.from_numpy(segms_data[i])
                # Direct resize, assuming masks are for the full image and need scaling
                scaled_mask = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                    size=(h_orig, w_orig),
                    mode='bilinear',
                    align_corners=False
                )
                scaled_mask = scaled_mask.squeeze().squeeze() > 0.5 # Threshold
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
            item_bbox_coords = item_tuple[0]
            item_mask_array = item_tuple[1]
            confidence_val = item_tuple[2]
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
            item_bbox_coords = item_tuple[0]
            item_mask_array = item_tuple[1]
            confidence_val = item_tuple[2]
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
