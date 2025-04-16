from PIL import Image

import cv2
import numpy as np
import torch
from collections import namedtuple
from . import utils
import inspect
import logging
import os

import pickle



orig_torch_load = torch.load


SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

# --- Whitelist Configuration ---
# Determine the base directory for ComfyUI
try:
    # Start from the directory of the current file (subcore.py)
    module_path = os.path.dirname(inspect.getfile(lambda: None))
    # Navigate up from the current module's directory to find ComfyUI base
    current_path = module_path
    comfyui_base_path = None
    max_levels_up = 5 # Safety limit search depth
    for _ in range(max_levels_up):
        # Look for characteristic files/folders of the ComfyUI base
        # Adjust these checks if your ComfyUI layout is non-standard
        if os.path.exists(os.path.join(current_path, 'main.py')) and \
           os.path.exists(os.path.join(current_path, 'web')) and \
           os.path.exists(os.path.join(current_path, 'custom_nodes')):
           comfyui_base_path = current_path
           logging.info(f"[Impact Pack/Subpack] Found ComfyUI base path: {comfyui_base_path}")
           break
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path: # Reached filesystem root
            logging.warning(f"[Impact Pack/Subpack] Reached filesystem root without finding ComfyUI base path.")
            break
        current_path = parent_path

    if comfyui_base_path:
        # Define the target directory and file path within user/default
        WHITELIST_DIR = os.path.join(comfyui_base_path, "user", "default", "ComfyUI-Impact-Pack")
        WHITELIST_FILE_PATH = os.path.join(WHITELIST_DIR, "model-whitelist.txt")
        logging.info(f"[Impact Pack/Subpack] Determined whitelist path: {WHITELIST_FILE_PATH}")
    else:
        # Fallback strategy if base path detection fails
        logging.error("[Impact Pack/Subpack] Failed to automatically determine ComfyUI base path.")
        # Attempt to use a path relative to potential base if stopped early
        if current_path and os.path.exists(os.path.join(current_path, 'main.py')): # Simple check if we stopped at base
             comfyui_base_path = current_path # Assume current_path is base
             WHITELIST_DIR = os.path.join(comfyui_base_path, "user", "default", "ComfyUI-Impact-Pack")
             WHITELIST_FILE_PATH = os.path.join(WHITELIST_DIR, "model-whitelist.txt")
             logging.warning(f"[Impact Pack/Subpack] Using assumed base path for whitelist: {WHITELIST_FILE_PATH}")
        else:
             # Last resort: place near the module code (warn user it might be lost)
             WHITELIST_DIR = os.path.join(os.path.dirname(inspect.getfile(lambda: None)), ".impact_config") # Hidden dir
             WHITELIST_FILE_PATH = os.path.join(WHITELIST_DIR, "model-whitelist.txt")
             logging.error(f"[Impact Pack/Subpack] Using fallback whitelist location (may be lost on update): {WHITELIST_FILE_PATH}")


    # --- Ensure directory exists ---
    # Check if WHITELIST_FILE_PATH was successfully determined before trying to create dirs
    if 'WHITELIST_FILE_PATH' in locals() and WHITELIST_FILE_PATH:
        try:
            # Crucially, create the DIRECTORY first
            os.makedirs(WHITELIST_DIR, exist_ok=True)
            logging.info(f"[Impact Pack/Subpack] Ensured whitelist directory exists: {WHITELIST_DIR}")
        except OSError as e:
            logging.error(f"[Impact Pack/Subpack] Failed to create whitelist directory {WHITELIST_DIR}: {e}. Whitelisting may not function.")
            WHITELIST_FILE_PATH = None # Indicate failure / disable whitelisting
        except NameError:
             logging.error(f"[Impact Pack/Subpack] WHITELIST_DIR not defined, cannot create directory. Whitelisting disabled.")
             WHITELIST_FILE_PATH = None # Indicate failure / disable whitelisting
        except Exception as e:
            logging.error(f"[Impact Pack/Subpack] Unexpected error creating whitelist directory: {e}", exc_info=True)
            WHITELIST_FILE_PATH = None # Indicate failure / disable whitelisting
    else:
         # Handle case where path determination failed earlier
         logging.error("[Impact Pack/Subpack] Whitelist path determination failed. Whitelisting disabled.")
         # Explicitly set to None if it wasn't defined or set to None already
         if 'WHITELIST_FILE_PATH' not in locals():
              WHITELIST_FILE_PATH = None


except Exception as e:
    # Catch errors during the whole setup process
    logging.error(f"[Impact Pack/Subpack] Critical error during whitelist path setup: {e}", exc_info=True)
    # Define a fallback path as absolute last resort, but log error
    WHITELIST_FILE_PATH = None # Disable whitelisting on critical setup failure
    logging.error("[Impact Pack/Subpack] Whitelisting disabled due to critical setup error.")


def load_whitelist(filepath):
    """
    Loads filenames from the whitelist file.
    Attempts to create the file with instructions if it doesn't exist.
    Returns a set of approved base filenames.
    """
    approved_files = set()
    # Check again if filepath is valid before proceeding
    if filepath is None or not isinstance(filepath, str):
        # Log was already done if None during setup, avoid duplicate messages
        # logging.error("[Impact Pack/Subpack] Whitelist file path is invalid. Whitelisting disabled.")
        return approved_files # Return empty set

    try:
        # Try reading the existing file
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Store only the base filename for easier matching
                if line and not line.startswith('#'):
                    approved_files.add(os.path.basename(line))
        logging.info(f"[Impact Pack/Subpack] Loaded {len(approved_files)} model(s) from whitelist: {filepath}")

    except FileNotFoundError:
        # This block now runs only if the directory was created successfully but the file is missing
        logging.warning(f"[Impact Pack/Subpack] Model whitelist file not found at: {filepath}. ")
        logging.warning(f" >> An empty whitelist file will be created.")
        logging.warning(f" >> To allow unsafe loading for specific trusted legacy models (e.g., older .pt),")
        logging.warning(f" >> add their base filenames (one per line) to this file.")
        try:
            # Attempt to create the file with comments since it wasn't found
            # This should now succeed because os.makedirs created the directory
            with open(filepath, 'w') as f:
                f.write("# Add base filenames of trusted models (e.g., my_old_yolo.pt) here, one per line.\n")
                f.write("# This allows loading them with `weights_only=False` if they fail safe loading\n")
                f.write("# due to errors like 'restricted getattr' in newer PyTorch versions.\n")
                f.write("# WARNING: Only add files you absolutely trust, as this bypasses a security feature.\n")
                f.write("# Prefer using .safetensors files whenever possible.\n")
            logging.info(f"[Impact Pack/Subpack] Created empty whitelist file: {filepath}")
        except Exception as create_e:
             # Log error if creating the file fails even after creating the directory
             logging.error(f"[Impact Pack/Subpack] Failed to create empty whitelist file at {filepath}: {create_e}", exc_info=True)

    except Exception as e:
        logging.error(f"[Impact Pack/Subpack] Error loading model whitelist from {filepath}: {e}", exc_info=True)

    return approved_files

# Now call the function using the dynamically determined (or None) path
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

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results


# Limit the commands that can be executed through `getattr` to `ultralytics.nn.modules.head.Detect.forward`.
def restricted_getattr(obj, name, *args):
    if name != "forward":
        logging.error(f"Access to potentially dangerous attribute '{obj.__module__}.{obj.__name__}.{name}' is blocked.\nIf you believe the use of this code is genuinely safe, please report it.\nhttps://github.com/ltdrdata/ComfyUI-Impact-Subpack/issues")
        raise RuntimeError(f"Access to potentially dangerous attribute '{obj.__module__}.{obj.__name__}.{name}' is blocked.")
        
    return getattr(obj, name, *args)

restricted_getattr.__module__ = 'builtins'
restricted_getattr.__name__ = 'getattr'


try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.tasks import SegmentationModel
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
    except:
        logging.error("[Impact Subpack] installed 'numpy' is outdated. Please update 'numpy' to 1.26.4")
        raise Exception("[Impact Subpack] installed 'numpy' is outdated. Please update 'numpy' to 1.26.4")


    torch_whitelist = []

    # https://github.com/comfyanonymous/ComfyUI/issues/5516#issuecomment-2466152838
    def build_torch_whitelist():
        """
        For security, only a limited set of namespaces is allowed during loading.

        Since the same module may be identified by different namespaces depending on the model,
        some modules are additionally registered with aliases to ensure backward compatibility.
        """
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

        aliasv10DetectLoss = type("v10DetectLoss", (loss_modules.E2EDetectLoss,), {})
        aliasv10DetectLoss.__name__ = "v10DetectLoss"
        aliasv10DetectLoss.__module__ = "ultralytics.utils.loss"

        torch_whitelist += [DetectionModel, aliasYOLOv10DetectionModel, SegmentationModel, IterableSimpleNamespace,
                            aliasIterableSimpleNamespace, TaskAlignedAssigner, aliasTaskAlignedAssigner, aliasv10DetectLoss,
                            restricted_getattr, dill._dill._load_type, scalar, dtype, Float64DType]

    build_torch_whitelist()

except Exception as e:
    logging.error(e)
    logging.error("\n!!!!!\n\n[ComfyUI-Impact-Subpack] If this error occurs, please check the following link:\n\thttps://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/Main/troubleshooting/TROUBLESHOOTING.md\n\n!!!!!\n")
    raise e

# HOTFIX: https://github.com/ltdrdata/ComfyUI-Impact-Pack/issues/754
# importing YOLO breaking original torch.load capabilities

# --- Start: REPLACE the existing torch_wrapper function ---

def torch_wrapper(*args, **kwargs):
    """
    Wrapper for torch.load that attempts safe loading (weights_only=True) first.
    If a specific UnpicklingError related to disallowed globals (like 'getattr')
    occurs, it checks a user-defined whitelist (_MODEL_WHITELIST). If the file
    is whitelisted, it retries with weights_only=False. Otherwise, it blocks
    the unsafe load and raises the error.
    """
    # Use the globally saved original torch.load reference from the top of the file
    # Check if weights_only was explicitly passed by the caller
    weights_only_explicit = kwargs.get('weights_only', None) # Read value without popping yet

    # Try to get the filename being loaded (usually the first arg if it's a path)
    filename = None
    filename_arg_source = "[unknown source]"
    if args and isinstance(args[0], str):
        filename = os.path.basename(args[0]) # Get just the filename part
        filename_arg_source = args[0]
    elif 'f' in kwargs and isinstance(kwargs['f'], str):
        filename = os.path.basename(kwargs['f']) # Get just the filename part
        filename_arg_source = kwargs['f']
    # Note: filename might remain None if loading from a file-like object

    # Check if using newer PyTorch with safe_globals attribute (indicates >= 2.6 behavior likely)
    if hasattr(torch.serialization, 'safe_globals'):

        # Determine the effective weights_only setting for the FIRST attempt
        load_kwargs = kwargs.copy()
        load_kwargs['weights_only'] = True # ALWAYS attempt safe load first on newer PyTorch

        try:
            # --- Attempt 1: Safe Load ---
            # Try loading with the determined weights_only setting (usually True)
            logging.debug(f"[Impact Pack/Subpack] Attempting safe load (weights_only=True) for: {filename_arg_source}")
            return orig_torch_load(*args, **load_kwargs)

        except pickle.UnpicklingError as e:
            # --- Handle Specific Load Failure ---
            # Check if the error is the specific one caused by disallowed globals
            # like 'getattr' AND we were attempting a safe load (weights_only=True)
            # Using 'getattr' because it was the specific error reported.
            is_disallowed_global_error = 'getattr' in str(e)

            if is_disallowed_global_error:
                # Check the whitelist
                if filename and filename in _MODEL_WHITELIST:
                    # --- Fallback: Whitelisted Unsafe Load ---
                    logging.warning("##############################################################################")
                    logging.warning(f"[Impact Pack/Subpack] WARNING: Safe load failed for '{filename}' (Reason: {e}).")
                    logging.warning(f" >> FILE IS IN THE WHITELIST: {WHITELIST_FILE_PATH}")
                    logging.warning(" >> This model likely uses legacy Python features blocked by default for security.")
                    logging.warning(" >> RETRYING WITH 'weights_only=False' because it's whitelisted.")
                    logging.warning(" >> SECURITY RISK: Ensure you added this file to the whitelist consciously")
                    logging.warning(f" >> and trust its source: {filename_arg_source}")
                    logging.warning(" >> Prefer using .safetensors files whenever available.")
                    logging.warning("##############################################################################")

                    # Modify kwargs ONLY for the retry
                    retry_kwargs = kwargs.copy()
                    retry_kwargs['weights_only'] = False
                    # Call the original function again, now unsafely (because whitelisted)
                    return orig_torch_load(*args, **retry_kwargs)

                else:
                    # --- Blocked: Not Whitelisted ---
                    logging.error("##############################################################################")
                    logging.error(f"[Impact Pack/Subpack] ERROR: Safe load failed for '{filename_arg_source}' (Reason: {e}).")
                    logging.error(f" >> This model likely uses legacy Python features blocked by default for security.")
                    logging.error(f" >> UNSAFE LOAD BLOCKED because the file ('{filename or 'unknown'}') is NOT in the whitelist.")
                    logging.error(f" >> Whitelist path: {WHITELIST_FILE_PATH}")
                    logging.error(f" >> To allow loading this specific file (IF YOU TRUST IT), add its base name")
                    logging.error(f" >> ('{filename}') to the whitelist file, one name per line.")
                    logging.error(" >> SECURITY RISK: Only whitelist files from sources you absolutely trust.")
                    logging.error(" >> Prefer using .safetensors files whenever available.")
                    logging.error("##############################################################################")
                    # Re-raise the original security-related error because it's not whitelisted
                    raise e

            else:
                # If it's a different UnpicklingError, re-raise it. Don't attempt unsafe load.
                logging.error(f"[Impact Pack/Subpack] UnpicklingError during safe load (not 'getattr' related): {e}. Re-raising.")
                raise e # Re-raise other UnpicklingErrors

    else:
        # --- Handle Older PyTorch Versions (no safe_globals) ---
        # Behavior here respects the caller's explicit request or defaults to False
        load_kwargs = kwargs.copy()
        effective_weights_only = weights_only_explicit if weights_only_explicit is not None else False # Default False for old torch
        load_kwargs['weights_only'] = effective_weights_only

        if not effective_weights_only:
            logging.warning(f"[Impact Pack/Subpack] Older PyTorch version detected. Proceeding with potentially unsafe load (weights_only=False) for: {filename_arg_source}")
        else:
             logging.debug(f"[Impact Pack/Subpack] Older PyTorch version detected. Proceeding with explicit weights_only=True for: {filename_arg_source}")

        # Call the original torch.load directly with the determined settings for older PyTorch
        return orig_torch_load(*args, **load_kwargs)

# --- End: Replacement block for the torch_wrapper function ---

torch.load = torch_wrapper


def load_yolo(model_path: str):
    # https://github.com/comfyanonymous/ComfyUI/issues/5516#issuecomment-2466152838
    if hasattr(torch.serialization, 'safe_globals'):
        with torch.serialization.safe_globals(torch_whitelist):
            try:
                return YOLO(model_path)
            except ModuleNotFoundError:
                # https://github.com/ultralytics/ultralytics/issues/3856
                YOLO("yolov8n.pt")
                return YOLO(model_path)
    else:
        try:
            return YOLO(model_path)
        except ModuleNotFoundError:
            YOLO("yolov8n.pt")
            return YOLO(model_path)


def inference_bbox(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
):
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    cv2_image = np.array(image)
    if len(cv2_image.shape) == 3:
        cv2_image = cv2_image[:, :, ::-1].copy()  # Convert RGB to BGR for cv2 processing
    else:
        # Handle the grayscale image here
        # For example, you might want to convert it to a 3-channel grayscale image for consistency:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results


def inference_segm(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
):
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    # NOTE: masks.data will be None when n == 0
    segms = pred[0].masks.data.cpu().numpy()

    h_segms = segms.shape[1]
    w_segms = segms.shape[2]
    h_orig = image.size[1]
    w_orig = image.size[0]
    ratio_segms = h_segms / w_segms
    ratio_orig = h_orig / w_orig

    if ratio_segms == ratio_orig:
        h_gap = 0
        w_gap = 0
    elif ratio_segms > ratio_orig:
        h_gap = int((ratio_segms - ratio_orig) * h_segms)
        w_gap = 0
    else:
        h_gap = 0
        ratio_segms = w_segms / h_segms
        ratio_orig = w_orig / h_orig
        w_gap = int((ratio_segms - ratio_orig) * w_segms)

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])

        mask = torch.from_numpy(segms[i])
        mask = mask[h_gap:mask.shape[0] - h_gap, w_gap:mask.shape[1] - w_gap]

        scaled_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(image.size[1], image.size[0]),
                                                      mode='bilinear', align_corners=False)
        scaled_mask = scaled_mask.squeeze().squeeze()

        results[2].append(scaled_mask.numpy())
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results


class UltraBBoxDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        detected_results = inference_bbox(self.bbox_model, utils.tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)

        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = utils.make_crop_region(w, h, item_bbox, crop_factor)

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = utils.crop_image(image, crop_region)
                cropped_mask = utils.crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)

                items.append(item)

        shape = image.shape[1], image.shape[2]
        segs = shape, items

        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)

        return segs

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_bbox(self.bbox_model, utils.tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)

        return utils.combine_masks(segmasks)

    def setAux(self, x):
        pass


class UltraSegmDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        detected_results = inference_segm(self.bbox_model, utils.tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)

        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = utils.make_crop_region(w, h, item_bbox, crop_factor)

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = utils.crop_image(image, crop_region)
                cropped_mask = utils.crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)

                items.append(item)

        shape = image.shape[1], image.shape[2]
        segs = shape, items

        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)

        return segs

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_segm(self.bbox_model, utils.tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)

        return utils.combine_masks(segmasks)

    def setAux(self, x):
        pass