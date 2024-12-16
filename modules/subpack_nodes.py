import logging
import os
import folder_paths
from . import subcore
from . import utils
import logging

model_path = folder_paths.models_dir
utils.add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(model_path, "ultralytics", "bbox")], folder_paths.supported_pt_extensions)
utils.add_folder_path_and_extensions("ultralytics_segm", [os.path.join(model_path, "ultralytics", "segm")], folder_paths.supported_pt_extensions)
utils.add_folder_path_and_extensions("ultralytics", [os.path.join(model_path, "ultralytics")], folder_paths.supported_pt_extensions)
logging.info(f'[Impact Subpack] ultralytics_bbox: {os.path.join(model_path, "ultralytics", "bbox")}')
logging.info(f'[Impact Subpack] ultralytics_segm: {os.path.join(model_path, "ultralytics", "segm")}')


class UltralyticsDetectorProvider:
    @classmethod
    def INPUT_TYPES(s):
        bboxs = ["bbox/"+x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/"+x for x in folder_paths.get_filename_list("ultralytics_segm")]
        return {"required": {"model_name": (bboxs + segms, )}}
    RETURN_TYPES = ("BBOX_DETECTOR", "SEGM_DETECTOR")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

    def doit(self, model_name):
        model_path = folder_paths.get_full_path("ultralytics", model_name)
        model = subcore.load_yolo(model_path)

        if model_name.startswith("bbox"):
            return subcore.UltraBBoxDetector(model), subcore.NO_SEGM_DETECTOR()
        else:
            return subcore.UltraBBoxDetector(model), subcore.UltraSegmDetector(model)


NODE_CLASS_MAPPINGS = {
    "UltralyticsDetectorProvider": UltralyticsDetectorProvider
}


NODE_DISPLAY_NAME_MAPPINGS = {

}
