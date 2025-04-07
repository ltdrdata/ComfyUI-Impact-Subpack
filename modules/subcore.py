from PIL import Image

import cv2
import numpy as np
import torch
from collections import namedtuple
from . import utils
import inspect
import logging

import pickle


orig_torch_load = torch.load


SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])


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
    occurs with newer PyTorch versions, it logs a warning and retries with
    weights_only=False for compatibility with older formats.
    Requires trusting the source file if the fallback occurs.
    """
    # Use the globally saved original torch.load reference from the top of the file
    # Check if weights_only was explicitly passed by the caller
    weights_only_explicit = kwargs.get('weights_only', None) # Read value without popping yet

    # Check if using newer PyTorch with safe_globals attribute (indicates >= 2.6 behavior likely)
    if hasattr(torch.serialization, 'safe_globals'):

        # Determine the effective weights_only setting for the FIRST attempt
        # Prioritize explicit setting, otherwise default to True (safe)
        effective_weights_only = weights_only_explicit if weights_only_explicit is not None else True
        
        load_kwargs = kwargs.copy()
        load_kwargs['weights_only'] = effective_weights_only # Set for the first attempt

        try:
            # --- Attempt 1: Safe Load ---
            # Try loading with the determined weights_only setting (usually True)
            if effective_weights_only:
                print(f"[Impact Pack/Subpack] Attempting safe load (weights_only=True)...")
            else:
                 print(f"[Impact Pack/Subpack] Explicit weights_only=False requested. Attempting unsafe load...")

            # Call original torch.load using the determined kwargs
            return orig_torch_load(*args, **load_kwargs)

        except pickle.UnpicklingError as e:
            # --- Handle Specific Load Failure ---
            # Check if the error is the specific one caused by disallowed globals
            # like 'getattr' AND we were attempting a safe load (weights_only=True)
            # Using 'getattr' because it was the specific error reported.
            is_disallowed_global_error = 'getattr' in str(e)

            if is_disallowed_global_error and effective_weights_only:
                # --- Fallback: Unsafe Load ---
                print("##############################################################################")
                print(f"[Impact Pack/Subpack] WARNING: Safe load failed (Reason: {e}).")
                print(" >> This often happens with older .pt/.pth formats using specific")
                print(" >> Python features (like 'getattr') blocked by default for security")
                print(" >> in newer PyTorch versions (>= 2.6).")
                print(" >> RETRYING WITH 'weights_only=False'.")
                print(" >> SECURITY RISK: This bypasses a PyTorch security feature.")
                print(" >> ENSURE YOU TRUST THE SOURCE of the file being loaded:")
                # Try to display the filename if available in args
                filename_arg = args[0] if args else load_kwargs.get('f', '[unknown file]')
                print(f" >> File: {filename_arg}")
                print(" >> Prefer using .safetensors files whenever available.")
                print("##############################################################################")

                # Modify the local kwargs copy for the retry attempt
                load_kwargs['weights_only'] = False
                # Call the original function again, now unsafely
                return orig_torch_load(*args, **load_kwargs)
            else:
                # If it's a different error OR weights_only was already False,
                # re-raise the original error. Don't bypass other issues.
                print(f"[Impact Pack/Subpack] Load failed with an unrelated error or weights_only=False was already set. Re-raising original error.")
                raise e # Re-raise the original exception
        except Exception as e:
             # Catch any other unexpected loading errors
             print(f"[Impact Pack/Subpack] Error: An unexpected error occurred during torch load wrapper: {e}")
             raise e # Re-raise the error

    else:
        # --- Handle Older PyTorch Versions ---
        # Use a local copy of kwargs here too for consistency
        load_kwargs = kwargs.copy()
        # If weights_only was explicitly passed, use that value
        if weights_only_explicit is not None:
            load_kwargs['weights_only'] = weights_only_explicit
        else:
            # Otherwise, default to False for old versions and maybe warn
            load_kwargs['weights_only'] = False

        # Call the original torch.load for older versions
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