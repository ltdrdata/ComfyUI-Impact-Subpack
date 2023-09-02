from pathlib import Path
from PIL import Image

import impact.core as core
import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image
import torch

from ultralytics import YOLO

def load_yolo(model_path: str):
    try:
        return YOLO(model_path)
    except ModuleNotFoundError:
        # https://github.com/ultralytics/ultralytics/issues/3856
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
    cv2_image = cv2_image[:, :, ::-1].copy()
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
    segms = pred[0].masks.data.cpu().numpy()

    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])

        mask = torch.from_numpy(segms[i])
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

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1):
        drop_size = max(drop_size, 1)
        detected_results = inference_bbox(self.bbox_model, core.tensor2pil(image), threshold)
        segmasks = core.create_segmasks(detected_results)

        if dilation > 0:
            segmasks = core.dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = core.make_crop_region(w, h, item_bbox, crop_factor)
                cropped_image = core.crop_image(image, crop_region)
                cropped_mask = core.crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                item = core.SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)

                items.append(item)

        shape = image.shape[1], image.shape[2]
        return shape, items

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_bbox(self.bbox_model, image, threshold)
        segmasks = core.create_segmasks(detected_results)
        if dilation > 0:
            segmasks = core.dilate_masks(segmasks, dilation)

        return core.combine_masks(segmasks)

    def setAux(self, x):
        pass


class UltraSegmDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1):
        drop_size = max(drop_size, 1)
        detected_results = inference_segm(self.bbox_model, core.tensor2pil(image), threshold)
        segmasks = core.create_segmasks(detected_results)

        if dilation > 0:
            segmasks = core.dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = core.make_crop_region(w, h, item_bbox, crop_factor)
                cropped_image = core.crop_image(image, crop_region)
                cropped_mask = core.crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                item = core.SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)

                items.append(item)

        shape = image.shape[1], image.shape[2]
        return shape, items

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_bbox(self.bbox_model, image, threshold)
        segmasks = core.create_segmasks(detected_results)
        if dilation > 0:
            segmasks = core.dilate_masks(segmasks, dilation)

        return core.combine_masks(segmasks)

    def setAux(self, x):
        pass