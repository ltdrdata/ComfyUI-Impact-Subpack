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


## Credits

ComfyUI/[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - A powerful and modular stable diffusion GUI.

Bing-su/[adetailer](https://github.com/Bing-su/adetailer/) - This repository provides an object detection model and features based on Ultralystics.

huggingface/Bingsu/[adetailer](https://huggingface.co/Bingsu/adetailer/tree/main) - This repository offers various models based on Ultralystics.
* You can download other models supported by the UltralyticsDetectorProvider from here.
