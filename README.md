# ComfyUI-Impact-Subpack
This node pack provides nodes that complement the [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack), such as the UltralyticsDetectorProvider.


## Nodes
* `UltralyticsDetectorProvider` - Loads the Ultralystics model to provide SEGM_DETECTOR, BBOX_DETECTOR.
- Unlike `MMDetDetectorProvider`, for segm models, `BBOX_DETECTOR` is also provided.
- The various models available in UltralyticsDetectorProvider can be downloaded through **ComfyUI-Manager**.


## Ultralytics models
* When using ultralytics models, save them separately in `models/ultralytics/bbox` and `models/ultralytics/segm` depending on the type of model. Many models can be downloaded by searching for `ultralytics` in the Model Manager of ComfyUI-Manager.
* huggingface.co/Bingsu/[adetailer](https://huggingface.co/Bingsu/adetailer/tree/main) - You can download face, people detection models, and clothing detection models.
* ultralytics/[assets](https://github.com/ultralytics/assets/releases/) - You can download various types of detection models other than faces or people.
* civitai/[adetailer](https://civitai.com/search/models?sortBy=models_v5&query=adetailer) - You can download various types detection models....Many models are associated with NSFW content.


## Paths
* In `extra_model_paths.yaml`, you can add the following entries:
- `ultralytics_bbox` - Specifies the paths for bbox YOLO models.
- `ultralytics_segm` - Specifies the paths for segm YOLO models.
- `ultralytics` - Allows the presence of `bbox/` and `segm/` subdirectories.


## How To Install?

### Install via ComfyUI-Manager (Recommended)
* Search `ComfyUI Impact Subpack` in ComfyUI-Manager and click `Install` button.

### Manual Install (Not Recommended)
1. `cd custom_nodes`
2. `git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack`
3. `cd ComfyUI-Impact-Subpack`
4. `pip install -r requirements.txt`
    * **IMPORTANT**:
        * You must install it within the Python environment where ComfyUI is running.
        * For the portable version, use `<installed path>\python_embeded\python.exe -m pip` instead of `pip`. For a `venv`, activate the `venv` first and then use `pip`.
5. Restart ComfyUI


## Credits

ComfyUI/[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - A powerful and modular stable diffusion GUI.

Bing-su/[adetailer](https://github.com/Bing-su/adetailer/) - This repository provides an object detection model and features based on Ultralystics.

huggingface/Bingsu/[adetailer](https://huggingface.co/Bingsu/adetailer/tree/main) - This repository offers various models based on Ultralystics.
* You can download other models supported by the UltralyticsDetectorProvider from here.
