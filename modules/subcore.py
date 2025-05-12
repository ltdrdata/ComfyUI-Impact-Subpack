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
    user_dir = folder_paths.get_user_directory()
    if user_dir and os.path.isdir(user_dir):
        WHITELIST_DIR = os.path.join(user_dir, "default", "ComfyUI-Impact-Subpack")
        WHITELIST_FILE_PATH = os.path.join(WHITELIST_DIR, "model-whitelist.txt")
        logging.info(f"[Impact Pack/Subpack] Using folder_paths to determine whitelist path: {WHITELIST_FILE_PATH}")
    else:
        logging.warning(f"[Impact Pack/Subpack] folder_paths.get_user_directory() returned invalid path: {user_dir}.")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        WHITELIST_DIR = os.path.join(script_dir, "ComfyUI-Impact-Subpack-Data")
        WHITELIST_FILE_PATH = os.path.join(WHITELIST_DIR, "model-whitelist.txt")
        logging.warning(f"[Impact Pack/Subpack] Fallback: Using script-relative whitelist path: {WHITELIST_FILE_PATH}")

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
        logging.info(f"[Impact Pack/Subpack] Model whitelist file not found at: {filepath}. Creating a new one.")
        try:
            with open(filepath, 'w') as f:
                f.write("# Add base filenames of trusted models (e.g., my_old_yolo.pt) here, one per line.\n")
                f.write("# This allows loading them with `weights_only=False` if they fail safe loading.\n")
                f.write("# Files automatically added by the script due to safe load failures will also appear here.\n")
                f.write("# WARNING: Only add files you explicitly trust or understand the implications of auto-adding.\n")
                f.write("# Prefer using .safetensors files whenever possible for better security.\n")
            logging.info(f"[Impact Pack/Subpack] Created empty whitelist file: {filepath}")
        except Exception as create_e:
            logging.error(f"[Impact Pack/Subpack] Failed to create empty whitelist file at {filepath}: {create_e}", exc_info=True)
    except Exception as e:
        logging.error(f"[Impact Pack/Subpack] Error loading model whitelist from {filepath}: {e}", exc_info=True)
    return approved_files

def _add_to_whitelist_file_and_reload(filename_to_add, whitelist_filepath):
    global _MODEL_WHITELIST
    if whitelist_filepath is None or not isinstance(whitelist_filepath, str) or filename_to_add is None:
        logging.error("[Impact Pack/Subpack] Cannot add to whitelist: Invalid filepath or filename.")
        return _MODEL_WHITELIST
    try:
        os.makedirs(os.path.dirname(whitelist_filepath), exist_ok=True)
        current_entries = set()
        try:
            with open(whitelist_filepath, 'r') as f_read:
                for line in f_read:
                    current_entries.add(line.strip())
        except FileNotFoundError:
            pass
        if os.path.basename(filename_to_add) not in current_entries:
            with open(whitelist_filepath, 'a') as f_append:
                f_append.write(f"\n{os.path.basename(filename_to_add)}")
            logging.debug(f"[Impact Pack/Subpack] Added '{os.path.basename(filename_to_add)}' to whitelist file: {whitelist_filepath}") # Changed to DEBUG
        else:
            logging.debug(f"[Impact Pack/Subpack] Filename '{os.path.basename(filename_to_add)}' already in whitelist file: {whitelist_filepath}") # Changed to DEBUG
        _MODEL_WHITELIST = load_whitelist(whitelist_filepath) # This will log "Loaded X model(s)" at INFO
        return _MODEL_WHITELIST
    except Exception as e:
        logging.error(f"[Impact Pack/Subpack] Failed to automatically add '{filename_to_add}' to whitelist file {whitelist_filepath}: {e}", exc_info=True)
        return _MODEL_WHITELIST

_MODEL_WHITELIST = load_whitelist(WHITELIST_FILE_PATH)

class NO_BBOX_DETECTOR: pass
class NO_SEGM_DETECTOR: pass

def create_segmasks(results_list): # Renamed 'results' to 'results_list' to avoid confusion
    bboxs = results_list[1]
    segms = results_list[2]
    confidence = results_list[3]
    output_results = []
    for i in range(len(segms)): # Make sure segms is not empty and i is valid
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        output_results.append(item)
    return output_results

def restricted_getattr(obj, name, *args):
    if name != "forward":
        logging.error(f"Access to potentially dangerous attribute '{obj.__module__}.{obj.__name__}.{name}' is blocked.")
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
        logging.warning("[Impact Subpack] Could not import Float64DType from numpy.dtypes.")
        from numpy import dtype
        Float64DType = type(np.float64(0))
        logging.warning("[Impact Subpack] Using fallback for Float64DType.")
    torch_whitelist = []
    def build_torch_whitelist():
        global torch_whitelist
        for name, obj in inspect.getmembers(modules):
            if inspect.isclass(obj) and obj.__module__.startswith("ultralytics.nn.modules"):
                aliasObj = type(name, (obj,), {})
                aliasObj.__module__ = "ultralytics.nn.modules"
                torch_whitelist.extend([obj, aliasObj])
        for name, obj in inspect.getmembers(block_modules):
            if inspect.isclass(obj) and obj.__module__.startswith("ultralytics.nn.modules"):
                aliasObj = type(name, (obj,), {})
                aliasObj.__module__ = "ultralytics.nn.modules.block"
                torch_whitelist.extend([obj, aliasObj])
        for name, obj in inspect.getmembers(loss_modules):
            if inspect.isclass(obj) and obj.__module__.startswith("ultralytics.utils.loss"):
                aliasObj = type(name, (obj,), {})
                aliasObj.__module__ = "ultralytics.yolo.utils.loss"
                torch_whitelist.extend([obj, aliasObj])
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
        E2EDetectLoss_attr = getattr(loss_modules, 'E2EDetectLoss', None)
        if E2EDetectLoss_attr:
            aliasv10DetectLoss = type("v10DetectLoss", (E2EDetectLoss_attr,), {})
        else:
            class DummyLoss: pass
            aliasv10DetectLoss = type("v10DetectLoss", (DummyLoss,), {})
            logging.warning("[Impact Pack/Subpack] loss_modules.E2EDetectLoss not found for aliasv10DetectLoss.")
        aliasv10DetectLoss.__name__ = "v10DetectLoss"
        aliasv10DetectLoss.__module__ = "ultralytics.utils.loss"
        torch_whitelist.extend([
            DetectionModel, aliasYOLOv10DetectionModel, SegmentationModel,
            IterableSimpleNamespace, aliasIterableSimpleNamespace,
            TaskAlignedAssigner, aliasTaskAlignedAssigner, aliasv10DetectLoss,
            restricted_getattr, dill._dill._load_type, scalar, dtype, Float64DType
        ])
    build_torch_whitelist()
except ImportError as e:
    logging.error(f"[Impact Pack/Subpack] Failed to import ultralytics or its dependencies: {e}")
    YOLO = None; DetectionModel = type("DetectionModel", (object,), {}); SegmentationModel = type("SegmentationModel", (object,), {}) # Dummy classes
except Exception as e:
    logging.error(f"[Impact Pack/Subpack] General error during ultralytics setup: {e}", exc_info=True)
    raise e

def torch_wrapper(*args, **kwargs):
    global _MODEL_WHITELIST
    filename, filename_arg_source = None, "[unknown source]"
    if args and isinstance(args[0], str):
        filename, filename_arg_source = os.path.basename(args[0]), args[0]
    elif 'f' in kwargs and isinstance(kwargs['f'], str):
        filename, filename_arg_source = os.path.basename(kwargs['f']), kwargs['f']

    if hasattr(torch.serialization, 'safe_globals'):
        load_kwargs_attempt1 = kwargs.copy()
        effective_wo_attempt1 = load_kwargs_attempt1.get('weights_only', True)
        logging.debug(f"[Impact Pack/Subpack] Attempting load for: {filename_arg_source}. Effective 'weights_only': {effective_wo_attempt1}")
        try:
            return orig_torch_load(*args, **load_kwargs_attempt1)
        except pickle.UnpicklingError as e:
            is_disallowed_global_error = 'getattr' in str(e) or "Unsupported global" in str(e)
            if is_disallowed_global_error and effective_wo_attempt1:
                if filename and filename in _MODEL_WHITELIST:
                    logging.info(f"[Impact Pack/Subpack] Whitelisted file '{filename}' failed safe load. Retrying with unsafe load.")
                    retry_kwargs = kwargs.copy(); retry_kwargs['weights_only'] = False
                    return orig_torch_load(*args, **retry_kwargs)
                else:
                    # Auto-add to whitelist and retry
                    if filename and WHITELIST_FILE_PATH:
                        logging.info(f"[Impact Pack/Subpack] Auto-adding '{filename}' to whitelist and retrying unsafe load due to: {str(e).splitlines()[0]}")
                        _MODEL_WHITELIST = _add_to_whitelist_file_and_reload(filename, WHITELIST_FILE_PATH)
                        if filename in _MODEL_WHITELIST:
                            retry_kwargs = kwargs.copy(); retry_kwargs['weights_only'] = False
                            return orig_torch_load(*args, **retry_kwargs)
                        else:
                            logging.error(f"[Impact Pack/Subpack] Failed to verify '{filename}' in whitelist after auto-add. Blocking load.")
                    else:
                        logging.error("[Impact Pack/Subpack] Cannot auto-add to whitelist (filename or path missing). Blocking load.")
                    # If auto-add failed or was not possible, log and raise
                    logging.error(f"[Impact Pack/Subpack] BLOCKED (after auto-add attempt failed/not possible): Load failed for '{filename_arg_source}'. Reason: {e}")
                    raise e
            else:
                logging.error(f"[Impact Pack/Subpack] Load failed for '{filename_arg_source}'. Error: {e}. Not an auto-whitelisting scenario. Re-raising.")
                raise e
        except Exception as general_e:
            logging.error(f"[Impact Pack/Subpack] General exception during torch.load for {filename_arg_source}: {general_e}", exc_info=True)
            raise general_e
    else: # Older PyTorch
        load_kwargs_old_torch = kwargs.copy()
        effective_wo_old_torch = load_kwargs_old_torch.get('weights_only', False)
        log_level = logging.WARNING if not effective_wo_old_torch else logging.DEBUG
        logging.log(log_level, f"[Impact Pack/Subpack] Older PyTorch. Load for: {filename_arg_source} with weights_only={effective_wo_old_torch}")
        return orig_torch_load(*args, **load_kwargs_old_torch)
torch.load = torch_wrapper

def load_yolo(model_path: str):
    if YOLO is None:
        logging.error("[Impact Pack/Subpack] YOLO class not available. Cannot load YOLO model.")
        raise RuntimeError("YOLO could not be imported.")
    current_safe_globals = list(getattr(torch.serialization, 'get_safe_globals', lambda: [])())
    for item in torch_whitelist:
        if inspect.isclass(item) and item not in current_safe_globals:
            current_safe_globals.append(item)
    context_manager = torch.serialization.safe_globals(current_safe_globals) if hasattr(torch.serialization, 'safe_globals') else lambda: type('dummy_context_mgr', (), {'__enter__': lambda self: None, '__exit__': lambda *a: None})()
    with context_manager:
        try:
            return YOLO(model_path)
        except (ModuleNotFoundError, pickle.UnpicklingError, Exception) as e: # Broader catch for YOLO init issues
            if isinstance(e, ModuleNotFoundError):
                logging.warning(f"[Impact Pack/Subpack] ModuleNotFoundError during YOLO load for '{model_path}', attempting fallback. Error: {e}")
                YOLO("yolov8n.pt") # Attempt to initialize with a default model
                return YOLO(model_path) # Retry
            elif isinstance(e, pickle.UnpicklingError): # If it's an UnpicklingError, it should have been handled by torch_wrapper
                logging.error(f"[Impact Pack/Subpack] UnpicklingError in YOLO constructor for {model_path} (should be rare if torch.load is wrapped): {e}", exc_info=True)
            else: # Other exceptions during YOLO init
                logging.error(f"[Impact Pack/Subpack] Exception in YOLO constructor for {model_path}: {e}", exc_info=True)
            raise e


# --- inference_bbox, inference_segm, UltraBBoxDetector, UltraSegmDetector classes ---
# (Assuming their last provided version was correct and complete, no changes here)

def inference_bbox(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
):
    pred = model(image, conf=confidence, device=device if device else None)
    if not pred or not pred[0].boxes: return [[], [], [], []]
    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if not bboxes.any(): return [[], [], [], []]
    cv2_image = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        segms.append(cv2_mask.astype(bool))
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
    if not pred or not pred[0].boxes: return [[], [], [], []]
    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if not bboxes.any(): return [[], [], [], []]
    h_orig, w_orig = image.size[1], image.size[0]
    scaled_masks_np = []
    if pred[0].masks is None:
        logging.warning("[Impact Pack/Subpack] Segm model: no masks found in prediction.")
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
                    size=(h_orig, w_orig), mode='bilinear', align_corners=False
                )
                scaled_masks_np.append((scaled_mask_tensor.squeeze() > 0.5).numpy())
    output_results = [[], [], [], []]
    for i in range(len(bboxes)):
        output_results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        output_results[1].append(bboxes[i])
        output_results[2].append(scaled_masks_np[i] if i < len(scaled_masks_np) else np.zeros((h_orig, w_orig), dtype=bool))
        conf_value = pred[0].boxes[i].conf
        output_results[3].append(conf_value.cpu().numpy() if hasattr(conf_value, 'cpu') else np.array([conf_value.item()]))
    return output_results

class UltraBBoxDetector:
    bbox_model = None
    def __init__(self, bbox_model): self.bbox_model = bbox_model
    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        pil_image = utils.tensor2pil(image[0] if image.ndim == 4 and image.shape[0] == 1 else image)
        detected_results = inference_bbox(self.bbox_model, pil_image, threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0: segmasks = utils.dilate_masks(segmasks, dilation)
        items = []
        img_h, img_w = (image.shape[1], image.shape[2]) if image.ndim == 4 else (image.shape[0], image.shape[1]) # Simpler H,W access
        for i, item_tuple in enumerate(segmasks):
            item_bbox_coords, item_mask_array, confidence_val = item_tuple
            label = detected_results[0][i]
            x1, y1, x2, y2 = item_bbox_coords
            if (x2 - x1) >= drop_size and (y2 - y1) >= drop_size:
                crop_region_coords = utils.make_crop_region(img_w, img_h, item_bbox_coords, crop_factor)
                if detailer_hook and hasattr(detailer_hook, 'post_crop_region'):
                    crop_region_coords = detailer_hook.post_crop_region(img_w, img_h, item_bbox_coords, crop_region_coords)
                cropped_image_tensor = utils.crop_image(image, crop_region_coords)
                cropped_mask_array = utils.crop_ndarray2(item_mask_array, crop_region_coords)
                items.append(SEG(cropped_image_tensor, cropped_mask_array, confidence_val, crop_region_coords, item_bbox_coords, label, None))
        segs = ((img_h, img_w), items)
        if detailer_hook and hasattr(detailer_hook, "post_detection"): segs = detailer_hook.post_detection(segs)
        return segs
    def detect_combined(self, image, threshold, dilation):
        pil_image = utils.tensor2pil(image[0] if image.ndim == 4 and image.shape[0] == 1 else image)
        detected_results = inference_bbox(self.bbox_model, pil_image, threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0: segmasks = utils.dilate_masks(segmasks, dilation)
        return utils.combine_masks(segmasks)
    def setAux(self, x): pass

class UltraSegmDetector:
    segm_model = None
    def __init__(self, segm_model): self.segm_model = segm_model
    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        pil_image = utils.tensor2pil(image[0] if image.ndim == 4 and image.shape[0] == 1 else image)
        detected_results = inference_segm(self.segm_model, pil_image, threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0: segmasks = utils.dilate_masks(segmasks, dilation)
        items = []
        img_h, img_w = (image.shape[1], image.shape[2]) if image.ndim == 4 else (image.shape[0], image.shape[1])
        for i, item_tuple in enumerate(segmasks):
            item_bbox_coords, item_mask_array, confidence_val = item_tuple
            label = detected_results[0][i]
            x1, y1, x2, y2 = item_bbox_coords
            if (x2 - x1) >= drop_size and (y2 - y1) >= drop_size:
                crop_region_coords = utils.make_crop_region(img_w, img_h, item_bbox_coords, crop_factor)
                if detailer_hook and hasattr(detailer_hook, 'post_crop_region'):
                    crop_region_coords = detailer_hook.post_crop_region(img_w, img_h, item_bbox_coords, crop_region_coords)
                cropped_image_tensor = utils.crop_image(image, crop_region_coords)
                cropped_mask_array = utils.crop_ndarray2(item_mask_array, crop_region_coords)
                items.append(SEG(cropped_image_tensor, cropped_mask_array, confidence_val, crop_region_coords, item_bbox_coords, label, None))
        segs = ((img_h, img_w), items)
        if detailer_hook and hasattr(detailer_hook, "post_detection"): segs = detailer_hook.post_detection(segs)
        return segs
    def detect_combined(self, image, threshold, dilation):
        pil_image = utils.tensor2pil(image[0] if image.ndim == 4 and image.shape[0] == 1 else image)
        detected_results = inference_segm(self.segm_model, pil_image, threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0: segmasks = utils.dilate_masks(segmasks, dilation)
        return utils.combine_masks(segmasks)
    def setAux(self, x): pass
