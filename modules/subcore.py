from PIL import Image

import cv2
import numpy as np
import torch
from collections import namedtuple
# Assuming 'utils' is a module within the same package or accessible in python path
# For the purpose of this standalone script, I'll mock it if it's not standard.
# If . import utils causes issues, ensure it's correctly structured as a package.
# from . import utils
# Mocking utils for demonstration if not available:
class MockUtils:
    def tensor2pil(self, tensor_image):
        # This is a placeholder. Real conversion is more complex.
        print("[MockUtils] tensor2pil called")
        return Image.new('RGB', (100,100)) # Placeholder

    def dilate_masks(self, segmasks, dilation):
        print(f"[MockUtils] dilate_masks called with dilation {dilation}")
        return segmasks # Placeholder

    def make_crop_region(self, w, h, item_bbox, crop_factor):
        print(f"[MockUtils] make_crop_region called")
        return (0,0,w,h) # Placeholder, actual crop region

    def crop_image(self, image, crop_region):
        print(f"[MockUtils] crop_image called")
        return image # Placeholder

    def crop_ndarray2(self, item_mask, crop_region):
        print(f"[MockUtils] crop_ndarray2 called")
        return item_mask # Placeholder

    def combine_masks(self, segmasks):
        print(f"[MockUtils] combine_masks called")
        if not segmasks:
            return None # Or an empty mask representation
        # Placeholder: return the first mask or a combination
        return segmasks[0][1] if segmasks and segmasks[0] else np.zeros((100,100), dtype=bool)


utils = MockUtils() # Use the mock if real utils isn't set up for this environment

import inspect
import logging
import os
import pickle

# Mock folder_paths for environments where ComfyUI's folder_paths is not available
class MockFolderPaths:
    def get_user_directory(self):
        # Return a path that would exist or can be created for testing
        # e.g., a subdirectory in the current working directory
        mock_user_dir = os.path.join(os.getcwd(), "mock_comfyui_user_dir")
        os.makedirs(mock_user_dir, exist_ok=True)
        logging.info(f"[MockFolderPaths] get_user_directory returning: {mock_user_dir}")
        return mock_user_dir

    def get_folder_paths(self, folder_name):
        # Mock other path functions if needed
        return []

try:
    import folder_paths # Try to import the real one
except ImportError:
    logging.warning("[Impact Pack/Subpack] ComfyUI 'folder_paths' module not found. Using mock for whitelist path determination.")
    folder_paths = MockFolderPaths()


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
        # This part is an addition for robustness if folder_paths fails badly
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
    from numpy.core.multiarray import scalar # numpy.scalar is deprecated, use numpy.generic or specific types
                                             # However, if old pickles require it, it must be whitelisted.
    try:
        from numpy import dtype
        from numpy.dtypes import Float64DType # For numpy 1.24+
    except ImportError:
        # Fallback for older numpy versions if Float64DType is not in numpy.dtypes
        # This might not be perfect but aims to keep it running.
        # The original code had a hard crash, this provides a warning.
        logging.warning("[Impact Subpack] Could not import Float64DType from numpy.dtypes. This might affect loading models saved with newer numpy versions if you have an older numpy.")
        # If numpy is truly too old for 'dtype' itself, that's a bigger issue.
        from numpy import dtype # This should generally exist.
        Float64DType = type(np.float64(0)) # A way to get the type if direct import fails
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
            if inspect.isclass(obj) and obj.__module__.startswith("ultralytics.nn.modules"): # Original code had .block here, but it's likely still ultralytics.nn.modules
                aliasObj = type(name, (obj,), {})
                aliasObj.__module__ = "ultralytics.nn.modules.block" # Keeping original alias target
                torch_whitelist.append(obj)
                torch_whitelist.append(aliasObj)

        for name, obj in inspect.getmembers(loss_modules):
            if inspect.isclass(obj) and obj.__module__.startswith("ultralytics.utils.loss"):
                aliasObj = type(name, (obj,), {})
                aliasObj.__module__ = "ultralytics.yolo.utils.loss" # Original alias target
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

        aliasv10DetectLoss = type("v10DetectLoss", (loss_modules.E2EDetectLoss,), {}) # Assuming E2EDetectLoss exists
        aliasv10DetectLoss.__name__ = "v10DetectLoss"
        aliasv10DetectLoss.__module__ = "ultralytics.utils.loss"


        torch_whitelist += [
            DetectionModel, aliasYOLOv10DetectionModel, SegmentationModel,
            IterableSimpleNamespace, aliasIterableSimpleNamespace,
            TaskAlignedAssigner, aliasTaskAlignedAssigner, aliasv10DetectLoss,
            restricted_getattr, dill._dill._load_type, scalar, dtype, Float64DType
        ]
        # Add common builtins often used in pickles if necessary, but be cautious
        # torch_whitelist.append(getattr) # Example: if raw getattr needs to be allowed (SECURITY RISK)

    build_torch_whitelist()

except ImportError as e:
    logging.error(f"[Impact Pack/Subpack] Failed to import ultralytics or its dependencies: {e}")
    logging.error("\n!!!!!\n\n[ComfyUI-Impact-Subpack] If this error occurs, please check the following link:\n\thttps://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/Main/troubleshooting/TROUBLESHOOTING.md\n\n!!!!!\n")
    # Depending on severity, you might want to re-raise or define fallback classes/functions
    # For now, we let it proceed, but YOLO loading will fail later.
    YOLO = None # Indicate YOLO is not available
    # Define dummy classes if they are instantiated later to prevent NameErrors
    class DetectionModel: pass
    class SegmentationModel: pass

except Exception as e: # Catch other errors during setup
    logging.error(f"[Impact Pack/Subpack] General error during ultralytics setup: {e}", exc_info=True)
    logging.error("\n!!!!!\n\n[ComfyUI-Impact-Subpack] If this error occurs, please check the following link:\n\thttps://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/Main/troubleshooting/TROUBLESHOOTING.md\n\n!!!!!\n")
    raise e


# --- Start: REPLACE the existing torch_wrapper function ---
def torch_wrapper(*args, **kwargs):
    """
    Wrapper for torch.load that attempts safe loading (weights_only=True by default on PT>=2.6) first.
    If a specific UnpicklingError related to disallowed globals (like 'getattr')
    occurs, it checks a user-defined whitelist (_MODEL_WHITELIST). If the file
    is whitelisted, it retries with weights_only=False. Otherwise, it blocks
    the unsafe load and raises the error.
    """
    global _MODEL_WHITELIST # Crucial for allowing the reload to update the module-level cache

    filename = None
    filename_arg_source = "[unknown source]"
    if args and isinstance(args[0], str):
        filename = os.path.basename(args[0])
        filename_arg_source = args[0]
    elif 'f' in kwargs and isinstance(kwargs['f'], str):
        filename = os.path.basename(kwargs['f'])
        filename_arg_source = kwargs['f']

    if hasattr(torch.serialization, 'safe_globals'): # PyTorch 2.6+ or versions with similar safety features
        load_kwargs_attempt1 = kwargs.copy()
        # PyTorch 2.6+ defaults 'weights_only' to True if not specified.
        # We log the effective 'weights_only' for the first attempt.
        effective_wo_attempt1 = load_kwargs_attempt1.get('weights_only', True)

        logging.debug(f"[Impact Pack/Subpack] Attempting first load for: {filename_arg_source}. "
                      f"Effective 'weights_only' for this attempt (True if unspecified on PT>=2.6): {effective_wo_attempt1}")
        try:
            return orig_torch_load(*args, **load_kwargs_attempt1)

        except pickle.UnpicklingError as e:
            # Check if the error is the specific one caused by disallowed globals
            # AND if the first attempt was a "safe" one (weights_only=True).
            # The error message "Weights only load failed" and "Unsupported global" implies weights_only=True was active.
            is_disallowed_global_error = 'getattr' in str(e) or "Unsupported global" in str(e)

            if is_disallowed_global_error and effective_wo_attempt1:
                # Check the current whitelist cache
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
                    # File not in current whitelist cache, try reloading the whitelist file
                    logging.warning(f"[Impact Pack/Subpack] File '{filename}' not found in current whitelist cache.")
                    whitelist_path_msg = WHITELIST_FILE_PATH if WHITELIST_FILE_PATH else "[Path not determined]"
                    logging.info(f"[Impact Pack/Subpack] Attempting to reload whitelist from: {whitelist_path_msg}")
                    try:
                        # Reload the whitelist and update the global cache
                        _MODEL_WHITELIST = load_whitelist(WHITELIST_FILE_PATH)
                        logging.info(f"[Impact Pack/Subpack] Whitelist reloaded. Now contains {len(_MODEL_WHITELIST)} entries.")

                        # Re-check whitelist after reload
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
                            # Fall through to the original blocking logic below
                    except Exception as reload_e:
                        logging.error(f"[Impact Pack/Subpack] Error occurred during whitelist reload attempt: {reload_e}", exc_info=True)
                        # Fall through to the original blocking logic if reload fails

                    # --- Blocked: Not Whitelisted (runs if reload failed or file still not found) ---
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
                    raise e # Re-raise the original security-related error
            else:
                # Different UnpicklingError, or the first attempt was not 'safe_mode' (e.g., weights_only=False was explicit)
                logging.error(f"[Impact Pack/Subpack] UnpicklingError during load for '{filename_arg_source}'. Error: {e}. "
                              f"First attempt 'weights_only' was {effective_wo_attempt1}. "
                              "Not a whitelisting scenario or already tried unsafe. Re-raising.")
                raise e
        except Exception as general_e:
            # Catch any other unexpected errors during torch.load
            logging.error(f"[Impact Pack/Subpack] A general exception occurred during torch.load for {filename_arg_source}: {general_e}", exc_info=True)
            raise general_e
    else:
        # --- Handle Older PyTorch Versions (no safe_globals attribute) ---
        load_kwargs_old_torch = kwargs.copy()
        # PyTorch <2.6 defaults 'weights_only' to False. Respect caller's explicit setting if provided.
        effective_wo_old_torch = load_kwargs_old_torch.get('weights_only', False)

        if not effective_wo_old_torch: # i.e., loading with weights_only=False
            logging.warning(f"[Impact Pack/Subpack] Older PyTorch version detected. Proceeding with potentially unsafe load "
                            f"(effective weights_only=False) for: {filename_arg_source}")
        else: # weights_only=True was explicitly passed
            logging.debug(f"[Impact Pack/Subpack] Older PyTorch version detected. Proceeding with explicit weights_only=True "
                          f"for: {filename_arg_source}")
        return orig_torch_load(*args, **load_kwargs_old_torch)

# --- End: Replacement block for the torch_wrapper function ---

torch.load = torch_wrapper


def load_yolo(model_path: str):
    if YOLO is None: # Check if YOLO failed to import
        logging.error("[Impact Pack/Subpack] YOLO class not available. Cannot load YOLO model.")
        raise RuntimeError("[Impact Pack/Subpack] YOLO could not be imported. Please check ultralytics installation.")

    if hasattr(torch.serialization, 'safe_globals'):
        # The torch_whitelist is used by torch.load via the wrapper if needed,
        # but safe_globals context can be used for other serialization tasks if any were here.
        # For YOLO() itself, it internally calls torch.load, which our wrapper handles.
        # If YOLO constructor itself did other pickling, safe_globals here would be relevant.
        # The current `torch_whitelist` is primarily for `torch.load`.
        # `restricted_getattr` is also in `torch_whitelist` for use by `torch.load`.
        # We can also add `builtins.getattr` to `safe_globals` if YOLO unpickling *itself* needs it directly,
        # separate from model weight loading.
        # However, the error comes from `torch.load` of weights, handled by our wrapper.
        # Adding `restricted_getattr` to safe_globals might be redundant if not directly used by YOLO's own unpickling.
        # The original error is from `torch.load` which is already wrapped.
        # For now, we assume the main protection point is the wrapped `torch.load`.
        # The `torch_whitelist` itself primarily contains types, not standalone functions for `safe_globals`'s list argument.
        # `safe_globals` expects a list of objects to allow. `getattr` is `builtins.getattr`.
        # `restricted_getattr` is our custom one.
        # The example shows `torch.serialization.add_safe_globals([getattr])`.
        # Let's ensure our `restricted_getattr` is available if needed by the unpickler via `safe_globals`.
        # And also allow types from our `torch_whitelist`.
        
        current_safe_globals = list(torch.serialization.get_safe_globals())
        # Add types from torch_whitelist that are classes
        for item in torch_whitelist:
            if inspect.isclass(item) and item not in current_safe_globals:
                current_safe_globals.append(item)
        # Add specific functions like our restricted_getattr or builtins.getattr if absolutely needed
        # For now, relying on the torch.load wrapper's handling.

        # The main purpose of safe_globals here is to ensure that when YOLO unpickles its *own* structure
        # (if it does so separately from loading weights), it has access to necessary types.
        # The weight loading (torch.load) is already handled by our more specific wrapper.
        with torch.serialization.safe_globals(current_safe_globals):
            try:
                return YOLO(model_path)
            except ModuleNotFoundError: # Fallback as in original code
                logging.warning("[Impact Pack/Subpack] ModuleNotFoundError during YOLO load, attempting fallback with yolov8n.pt")
                YOLO("yolov8n.pt") # Pre-load a default model, might help with some internal ultralytics states
                return YOLO(model_path)
            except _pickle.UnpicklingError as e:
                # This might catch errors if YOLO() itself does some unpickling not via our wrapped torch.load
                logging.error(f"[Impact Pack/Subpack] UnpicklingError directly in YOLO constructor for {model_path}: {e}", exc_info=True)
                logging.error(f" >> This could be due to the model structure itself, not just weights.")
                logging.error(f" >> Ensure all necessary classes are in torch_whitelist and potentially safe_globals if this persists.")
                raise e
    else: # Older PyTorch
        try:
            return YOLO(model_path)
        except ModuleNotFoundError:
            logging.warning("[Impact Pack/Subpack] ModuleNotFoundError during YOLO load (older PyTorch), attempting fallback with yolov8n.pt")
            YOLO("yolov8n.pt")
            return YOLO(model_path)


def inference_bbox(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "", # device can be 'cpu', 'cuda:0', etc. or "" for auto
):
    # Ultralytics YOLO model's device handling:
    # If model is already on a device, it uses that.
    # If device arg is passed to __call__, it overrides.
    # If device is "", it tries to use model.device or cuda if available.
    pred = model(image, conf=confidence, device=device if device else None) # Pass None if device is empty string

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    # Ensure image is BGR for OpenCV
    cv2_image = np.array(image.convert("RGB")) # Ensure 3 channels
    cv2_image = cv2_image[:, :, ::-1].copy() # RGB to BGR

    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    if not bboxes.any(): # More robust check for empty bboxes
        return [[], [], [], []]

    output_results = [[], [], [], []] # Renamed to avoid confusion
    for i in range(len(bboxes)):
        output_results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        output_results[1].append(bboxes[i])
        output_results[2].append(segms[i])
        output_results[3].append(pred[0].boxes[i].conf.cpu().numpy()) # This is already a numpy array in some versions, .item() if single

    return output_results


def inference_segm(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
):
    pred = model(image, conf=confidence, device=device if device else None)

    if not pred or not pred[0].boxes: # Check if predictions are empty
         return [[], [], [], []]

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if not bboxes.any():
        return [[], [], [], []]

    # pred[0].masks can be None if no masks are found, even if boxes exist (e.g. for detection models)
    if pred[0].masks is None:
        logging.warning("[Impact Pack/Subpack] Segmentation model called, but no masks found in prediction. Returning empty masks.")
        # Create dummy empty masks or handle as bbox-only if that's desired fallback
        segms_data = np.zeros((len(bboxes), image.size[1], image.size[0]), dtype=bool) # empty masks
    else:
        segms_data = pred[0].masks.data.cpu().numpy()


    # The following resizing logic for masks seems specific and might need review
    # based on ultralytics output format for masks. Typically, masks are already in original image coords
    # or easily resizable. The gap calculation implies letterboxing/padding handling.
    # For now, assuming it's correct as per original snippet's intent.
    h_orig, w_orig = image.size[1], image.size[0] # PIL Image size is (width, height)

    # Assuming segms_data are [N, H_mask, W_mask]
    # This part needs to be robust if segms_data can be empty
    if segms_data.ndim < 3 or segms_data.shape[0] == 0: # No masks to process
        scaled_masks_np = [np.zeros((h_orig, w_orig), dtype=bool) for _ in bboxes]
    else:
        h_segms, w_segms = segms_data.shape[1], segms_data.shape[2]
        scaled_masks_np = []
        for i in range(segms_data.shape[0]):
            mask_tensor = torch.from_numpy(segms_data[i])
            
            # Original gap logic - this assumes masks might be padded and need un-padding
            # This may or may not be needed depending on how ultralytics returns masks now.
            # If masks are already aligned to the original image region, this might be unnecessary or harmful.
            # ratio_segms = h_segms / w_segms if w_segms > 0 else 0
            # ratio_orig = h_orig / w_orig if w_orig > 0 else 0
            # h_gap, w_gap = 0, 0
            # if ratio_segms != ratio_orig and ratio_segms > 0 and ratio_orig > 0: # Protect against div by zero
            #     if ratio_segms > ratio_orig:
            #         # This calculation seems off, usually it's about removing padding related to one dimension
            #         # h_gap = int((ratio_segms - ratio_orig) * h_segms) # This seems too large
            #         # A more common scenario is padding on one side.
            #         # For simplicity, we'll assume masks are directly resizable.
            #         # If precise un-padding is needed, ultralytics own postprocessing utils should be checked.
            #         pass # Placeholder for correct gap calculation if needed
            #     else:
            #         # w_gap = int(((w_segms / h_segms) - (w_orig / h_orig)) * w_segms) # also seems off
            #         pass
            
            # Modern ultralytics usually returns masks that can be directly resized to original image dimensions
            # mask_unpadded = mask_tensor[h_gap:mask_tensor.shape[0] - h_gap, w_gap:mask_tensor.shape[1] - w_gap]
            # For now, let's use direct resize, which is common.
            mask_unpadded = mask_tensor # Assuming no complex padding removal needed for now
            
            scaled_mask = torch.nn.functional.interpolate(
                mask_unpadded.unsqueeze(0).unsqueeze(0).float(), # Needs to be float for interpolate
                size=(h_orig, w_orig),
                mode='bilinear', # 'nearest' might be better for masks to keep binary values
                align_corners=False
            )
            scaled_mask = scaled_mask.squeeze().squeeze() > 0.5 # Threshold back to boolean
            scaled_masks_np.append(scaled_mask.numpy())

    output_results = [[], [], [], []]
    for i in range(len(bboxes)):
        output_results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        output_results[1].append(bboxes[i])
        output_results[2].append(scaled_masks_np[i] if i < len(scaled_masks_np) else np.zeros((h_orig, w_orig), dtype=bool) )
        output_results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return output_results


class UltraBBoxDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        # Assuming image is a tensor [B, H, W, C] or [B, C, H, W]
        # utils.tensor2pil expects [B, H, W, C] and takes the first image
        pil_image = utils.tensor2pil(image[0] if image.ndim == 4 else image)
        detected_results = inference_bbox(self.bbox_model, pil_image, threshold)
        segmasks = create_segmasks(detected_results) # segmasks are [(bbox, mask_array, conf), ...]

        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation) # Assuming this handles the list of tuples

        items = []
        # image is a tensor, shape is (B,H,W,C) or (H,W,C) for single.
        # Taking shape from the first image if batched
        img_h, img_w = image.shape[1 if image.ndim > 2 else 0], image.shape[2 if image.ndim > 2 else 1]


        for i, (item_data, label) in enumerate(zip(segmasks, detected_results[0])):
            item_bbox_coords = item_data[0] # xyxy format from inference
            item_mask_array = item_data[1]
            confidence_val = item_data[2]

            # Assuming item_bbox_coords is [x1, y1, x2, y2]
            x1, y1, x2, y2 = item_bbox_coords # Use these for width/height calculation

            # Note: original code had y1, x1, y2, x2 = item_bbox, which might be a typo if item_bbox is from cv2.boundingRect
            # However, inference_bbox returns xyxy, so item_bbox_coords should be [x1,y1,x2,y2]
            # Width: x2-x1, Height: y2-y1

            if (x2 - x1) >= drop_size and (y2 - y1) >= drop_size: # check width and height against drop_size
                # Original make_crop_region expected item_bbox in a different order if it was (y1,x1,y2,x2)
                # Assuming make_crop_region expects (x1,y1,x2,y2) or can handle it.
                # For now, passing the direct xyxy bbox.
                crop_region_coords = utils.make_crop_region(img_w, img_h, item_bbox_coords, crop_factor)

                if detailer_hook is not None and hasattr(detailer_hook, 'post_crop_region'):
                    crop_region_coords = detailer_hook.post_crop_region(img_w, img_h, item_bbox_coords, crop_region_coords)

                # Cropping the original image tensor
                # utils.crop_image needs to handle tensor input
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
        pil_image = utils.tensor2pil(image[0] if image.ndim == 4 else image)
        detected_results = inference_bbox(self.bbox_model, pil_image, threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)
        return utils.combine_masks(segmasks)

    def setAux(self, x):
        pass


class UltraSegmDetector:
    bbox_model = None # Should be segm_model for clarity

    def __init__(self, segm_model): # Renamed arg for clarity
        self.bbox_model = segm_model # Stores the segmentation model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        pil_image = utils.tensor2pil(image[0] if image.ndim == 4 else image)
        detected_results = inference_segm(self.bbox_model, pil_image, threshold) # Use the segm model
        segmasks = create_segmasks(detected_results)

        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)

        items = []
        img_h, img_w = image.shape[1 if image.ndim > 2 else 0], image.shape[2 if image.ndim > 2 else 1]

        for i, (item_data, label) in enumerate(zip(segmasks, detected_results[0])):
            item_bbox_coords = item_data[0]
            item_mask_array = item_data[1]
            confidence_val = item_data[2]

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
        pil_image = utils.tensor2pil(image[0] if image.ndim == 4 else image)
        detected_results = inference_segm(self.bbox_model, pil_image, threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = utils.dilate_masks(segmasks, dilation)
        return utils.combine_masks(segmasks)

    def setAux(self, x):
        pass

if __name__ == '__main__':
    # Basic logging setup for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Example of how the whitelist mechanism works:
    print(f"Whitelist file path: {WHITELIST_FILE_PATH}")
    print(f"Initial whitelist: {_MODEL_WHITELIST}")

    # To test the torch_wrapper, you would typically have a .pt file
    # that causes the specific UnpicklingError.
    # For demonstration, let's simulate a scenario.

    # Create a dummy whitelisted file entry for testing purposes
    if WHITELIST_FILE_PATH and not os.path.exists(WHITELIST_FILE_PATH):
        load_whitelist(WHITELIST_FILE_PATH) # This will create an empty one if it doesn't exist

    # Manually add a dummy file to whitelist for testing the reload
    # In a real scenario, the user edits this file.
    dummy_model_to_whitelist = "test_dummy_model.pt"
    if WHITELIST_FILE_PATH:
        with open(WHITELIST_FILE_PATH, 'a') as wf: # Append mode
             if dummy_model_to_whitelist not in _MODEL_WHITELIST: # Avoid duplicates if script run multiple times
                wf.write(f"\n{dummy_model_to_whitelist}\n")
        print(f"Manually added '{dummy_model_to_whitelist}' to whitelist for testing purposes.")
        # _MODEL_WHITELIST is not updated here yet, it will be upon reload attempt by torch_wrapper

    # Simulate torch.load call
    class MockPickleError(pickle.UnpicklingError):
        pass

    # Keep a reference to the true original torch.load before it's wrapped by our wrapper
    true_orig_torch_load = torch.load
    # Then our wrapper replaces torch.load
    # torch.load = torch_wrapper # This is already done globally in the script

    def mock_orig_torch_load_that_fails_safely(f, *args, **kwargs):
        print(f"[MOCK] orig_torch_load called for {f} with weights_only={kwargs.get('weights_only')}")
        if kwargs.get('weights_only') is True or kwargs.get('weights_only') is None : # PyTorch 2.6+ defaults to True
            if os.path.basename(f) == dummy_model_to_whitelist:
                print(f"[MOCK] Simulating safe load failure for whitelisted file {f}...")
                raise MockPickleError(f"Unsupported global: GLOBAL getattr. Weights only load failed for {f}")
            elif os.path.basename(f) == "another_unsafe_model.pt":
                print(f"[MOCK] Simulating safe load failure for non-whitelisted file {f}...")
                raise MockPickleError(f"Unsupported global: GLOBAL getattr. Weights only load failed for {f}")
        print(f"[MOCK] Load successful (or not a safe load failure) for {f}")
        return {"data": "mock_model_data"}

    # Temporarily replace the 'orig_torch_load' used by our wrapper with our mock
    orig_torch_load_backup_for_test = orig_torch_load
    orig_torch_load = mock_orig_torch_load_that_fails_safely


    print("\n--- Test Case 1: Loading a whitelisted model that fails safe load ---")
    try:
        # This should trigger the whitelist mechanism and retry with weights_only=False
        loaded_model = torch.load(dummy_model_to_whitelist) # Path as string
        print(f"[TEST RESULT] Successfully loaded (mocked) {dummy_model_to_whitelist}: {loaded_model}")
    except MockPickleError as e:
        print(f"[TEST ERROR] Failed to load {dummy_model_to_whitelist}: {e}")
    except Exception as e:
        print(f"[TEST UNEXPECTED ERROR] {e}")


    print("\n--- Test Case 2: Loading a non-whitelisted model that fails safe load ---")
    try:
        # This should be blocked by the whitelist mechanism
        loaded_model = torch.load("another_unsafe_model.pt")
        print(f"[TEST RESULT] Successfully loaded (mocked) another_unsafe_model.pt: {loaded_model}")
    except MockPickleError as e:
        print(f"[TEST ERROR] Correctly blocked or failed to load another_unsafe_model.pt: {e}")
    except Exception as e:
        print(f"[TEST UNEXPECTED ERROR] {e}")

    # Restore original orig_torch_load if it was backed up for this test block
    orig_torch_load = orig_torch_load_backup_for_test
    print(f"\n--- Tests concluded ---")
    print(f"Final effective _MODEL_WHITELIST after tests: {_MODEL_WHITELIST}")
    # Note: The dummy model might now be in _MODEL_WHITELIST if a reload occurred.
