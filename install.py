import os
import sys
from torchvision.datasets.utils import download_url

subpack_path = os.path.join(os.path.dirname(__file__))

comfy_path = os.environ.get('COMFYUI_PATH')
if comfy_path is None:
    print(f"\n[bold yellow]WARN: The `COMFYUI_PATH` environment variable is not set. Assuming `{os.path.dirname(__file__)}/../../../` as the ComfyUI path.[/bold yellow]", file=sys.stderr)
    comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

sys.path.append(comfy_path)

model_path = os.environ.get('COMFYUI_MODEL_PATH')
if model_path is None:
    print(f"\n[bold yellow]WARN: The `COMFYUI_MODEL_PATH` environment variable is not set. Assuming `{model_path}` as the ComfyUI path.[/bold yellow]", file=sys.stderr)
    model_path = os.path.abspath(os.path.join(comfy_path, 'models'))

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
