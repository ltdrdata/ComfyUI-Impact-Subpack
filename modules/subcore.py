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

    # --- Ensure directory exists---
    if WHITELIST_FILE_PATH:
        try:
            os.makedirs(WHITELIST_DIR, exist_ok=True)
            logging.info(f"[Impact Pack/Subpack] Ensured whitelist directory exists: {WHITELIST_DIR}")
        except OSError as e:
            logging.error(f"[Impact Pack/Subpack] Failed to create whitelist directory {WHITELIST_DIR}: {e}. Whitelisting may not function.")
            WHITELIST_FILE_PATH = None
        except Exception as e:
            logging.error(f"[Impact Pack/Subpack] Unexpected error creating whitelist directory: {e}", exc_info=True)
            WHITELIST_FILE_PATH = None
    else:
         logging.error("[Impact Pack/Subpack] Whitelist path determination failed using all methods. Whitelisting disabled.")

except Exception as e:
    logging.error(f"[Impact Pack/Subpack] Critical error during whitelist path setup: {e}", exc_info=True)
    WHITELIST_FILE_PATH = None
    logging.error("[Impact Pack/Subpack] Whitelisting disabled due to critical setup error.")

def load_whitelist(filepath):
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

def torch_wrapper(*args, **kwargs):
    global _MODEL_WHITELIST
    weights_only_explicit = kwargs.get('weights_only', None)
    filename = None
    filename_arg_source = "[unknown source]"
    
    if args and isinstance(args[0], str):
        filename = os.path.basename(args[0])
        filename_arg_source = args[0]
    elif 'f' in kwargs and isinstance(kwargs['f'], str):
        filename = os.path.basename(kwargs['f'])
        filename_arg_source = kwargs['f']

    if hasattr(torch.serialization, 'safe_globals'):
        # Add getattr to safe globals if it's not already there
        if not hasattr(torch.serialization, '_getattr_added'):
            torch.serialization.add_safe_globals([getattr])
            torch.serialization._getattr_added = True

        load_kwargs = kwargs.copy()
        if weights_only_explicit is None:
            load_kwargs['weights_only'] = True

        try:
            logging.debug(f"[Impact Pack/Subpack] Attempting safe load (weights_only=True) for: {filename_arg_source}")
            return orig_torch_load(*args, **load_kwargs)
        except pickle.UnpicklingError as e:
            is_disallowed_global_error = 'getattr' in str(e)
            
            if is_disallowed_global_error:
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
                logging.error(f"[Impact Pack/Subpack] UnpicklingError during safe load (not 'getattr' related): {e}. Re-raising.")
                raise e
    else:
        load_kwargs = kwargs.copy()
        effective_weights_only = weights_only_explicit if weights_only_explicit is not None else False
        load_kwargs['weights_only'] = effective_weights_only

        if not effective_weights_only:
            logging.warning(f"[Impact Pack/Subpack] Older PyTorch version detected. Proceeding with potentially unsafe load (weights_only=False) for: {filename_arg_source}")
        else:
            logging.debug(f"[Impact Pack/Subpack] Older PyTorch version detected. Proceeding with explicit weights_only=True for: {filename_arg_source}")

        return orig_torch_load(*args, **load_kwargs)

torch.load = torch_wrapper

def load_yolo(model_path: str):
    if hasattr(torch.serialization, 'safe_globals'):
        with torch.serialization.safe_globals(torch_whitelist):
            try:
                return YOLO(model_path)
            except ModuleNotFoundError:
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
        cv2_image = cv2_image[:, :, ::-1].copy()
    else:
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
    image: Image
