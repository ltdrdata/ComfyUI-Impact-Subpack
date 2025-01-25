import os
import folder_paths
from . import subcore
from . import utils
import logging


def update_model_paths(model_path):
    utils.add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(model_path, "ultralytics", "bbox")], folder_paths.supported_pt_extensions)
    utils.add_folder_path_and_extensions("ultralytics_segm", [os.path.join(model_path, "ultralytics", "segm")], folder_paths.supported_pt_extensions)
    utils.add_folder_path_and_extensions("ultralytics", [os.path.join(model_path, "ultralytics")], folder_paths.supported_pt_extensions)
    logging.info(f'[Impact Subpack] ultralytics_bbox: {os.path.join(model_path, "ultralytics", "bbox")}')
    logging.info(f'[Impact Subpack] ultralytics_segm: {os.path.join(model_path, "ultralytics", "segm")}')

update_model_paths(folder_paths.models_dir)
if 'download_model_base' in folder_paths.folder_names_and_paths:
    update_model_paths(folder_paths.get_folder_paths('download_model_base')[0])


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

        if model_path is None:
            print(f"[Impact Subpack] model file '{model_name}' is not found in one of the following directories:")

            cands = []
            cands.extend(folder_paths.get_folder_paths("ultralytics"))
            if model_name.startswith('bbox/'):
                cands.extend(folder_paths.get_folder_paths("ultralytics_bbox"))
            elif model_name.startswith('segm/'):
                cands.extend(folder_paths.get_folder_paths("ultralytics_segm"))

            formatted_cands = "\n\t".join(cands)
            print(f'\t{formatted_cands}\n')

            raise ValueError(f"[Impact Subpack] model file '{model_name}' is not found.")

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
