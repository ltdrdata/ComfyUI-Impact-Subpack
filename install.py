import os
import sys
from torchvision.datasets.utils import download_url

subpack_path = os.path.join(os.path.dirname(__file__))
comfy_path = os.path.join(subpack_path, '..', '..', '..')

sys.path.append(comfy_path)

import folder_paths
model_path = folder_paths.models_dir
ultralytics_bbox_path = os.path.join(model_path, "ultralytics", "bbox")
ultralytics_segm_path = os.path.join(model_path, "ultralytics", "segm")

if not os.path.exists(os.path.join(subpack_path, '..', '..', 'skip_download_model')):
    if not os.path.exists(ultralytics_bbox_path):
        os.makedirs(ultralytics_bbox_path)

    if not os.path.exists(ultralytics_segm_path):
        os.makedirs(ultralytics_segm_path)

    if not os.path.exists(os.path.join(ultralytics_bbox_path, "face_yolov8m.pt")):
        download_url("https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt",
                     ultralytics_bbox_path)

    if not os.path.exists(os.path.join(ultralytics_bbox_path, "hand_yolov8s.pt")):
        download_url("https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt",
                     ultralytics_bbox_path)

    if not os.path.exists(os.path.join(ultralytics_segm_path, "person_yolov8m-seg.pt")):
        download_url("https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt",
                     ultralytics_segm_path)
