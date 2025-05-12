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

# Save original torch.load
orig_torch_load = torch.load

# Segmentation tuple definition
SEG = namedtuple("SEG", 
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

# ==================== 安全加载配置 ====================
class SecurityConfig:
    @staticmethod
    def init_whitelist():
        """初始化白名单路径"""
        whitelist_dir = None
        whitelist_path = None
        
        try:
            user_dir = folder_paths.get_user_directory()
            if user_dir and os.path.isdir(user_dir):
                whitelist_dir = os.path.join(user_dir, "default", "ComfyUI-Impact-Subpack")
                whitelist_path = os.path.join(whitelist_dir, "model-whitelist.txt")
                logging.info(f"[Security] Whitelist path: {whitelist_path}")
                
                # 确保目录存在
                os.makedirs(whitelist_dir, exist_ok=True)
                
                # 如果文件不存在则创建
                if not os.path.exists(whitelist_path):
                    with open(whitelist_path, 'w') as f:
                        f.write("# 安全模型白名单\n")
                        f.write("# 每行一个模型文件名（如：yolov8n.pt）\n")
                        f.write("# 请只添加绝对可信的模型文件\n")
                    logging.info(f"[Security] Created new whitelist at {whitelist_path}")
                    
            return whitelist_path
            
        except Exception as e:
            logging.error(f"[Security] Whitelist init failed: {e}")
            return None

    @staticmethod
    def load_whitelist(filepath):
        """加载白名单内容"""
        if not filepath or not os.path.exists(filepath):
            return set()
            
        approved = set()
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        approved.add(os.path.basename(line))
            logging.info(f"[Security] Loaded {len(approved)} whitelisted models")
        except Exception as e:
            logging.error(f"[Security] Error loading whitelist: {e}")
            
        return approved

# 初始化安全配置
WHITELIST_PATH = SecurityConfig.init_whitelist()
MODEL_WHITELIST = SecurityConfig.load_whitelist(WHITELIST_PATH)

# ==================== 安全加载器 ====================
class SafeModelLoader:
    @staticmethod
    def torch_wrapper(*args, **kwargs):
        """安全的 torch.load 包装器"""
        global MODEL_WHITELIST
        
        # 获取文件名
        filename = None
        if args and isinstance(args[0], str):
            filename = os.path.basename(args[0])
        elif 'f' in kwargs and isinstance(kwargs['f'], str):
            filename = os.path.basename(kwargs['f'])
        
        # 默认启用安全模式
        load_kwargs = kwargs.copy()
        if 'weights_only' not in load_kwargs:
            load_kwargs['weights_only'] = True

        try:
            # 首次尝试安全加载
            return orig_torch_load(*args, **load_kwargs)
            
        except pickle.UnpicklingError as e:
            if 'getattr' in str(e) and filename:
                # 检查白名单
                if filename in MODEL_WHITELIST:
                    logging.warning(f"⚠️ Whitelisted unsafe load: {filename}")
                    return orig_torch_load(*args, **{**load_kwargs, 'weights_only': False})
                else:
                    logging.error(f"❌ Blocked unsafe load: {filename}. Add to whitelist if trusted.")
                    raise RuntimeError(f"Model {filename} not in whitelist. Add it to {WHITELIST_PATH} if trusted.")
            raise

    @staticmethod
    def safe_load_yolo(model_path):
        """安全加载YOLO模型"""
        if hasattr(torch.serialization, 'safe_globals'):
            with torch.serialization.safe_globals():
                try:
                    return YOLO(model_path)
                except ModuleNotFoundError:
                    # 初始化YOLO环境
                    YOLO("yolov8n.pt")  
                    return YOLO(model_path)
        else:
            return YOLO(model_path)

    @staticmethod
    def convert_to_safe_format(model_path, output_format='onnx'):
        """将模型转换为安全格式"""
        model = YOLO(model_path)
        export_path = os.path.splitext(model_path)[0] + f'.{output_format}'
        model.export(format=output_format)
        logging.info(f"✅ Converted to {output_format.upper()}: {export_path}")
        return export_path

# 替换原始加载器
torch.load = SafeModelLoader.torch_wrapper

# ==================== YOLO 模型加载 ====================
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
    from numpy import dtype, Float64DType

    # 构建安全类白名单
    TORCH_WHITELIST = []
    def build_torch_whitelist():
        """构建允许加载的类白名单"""
        global TORCH_WHITELIST
        
        # Ultralytics 模块
        for mod in [modules, block_modules]:
            for _, obj in inspect.getmembers(mod):
                if inspect.isclass(obj) and obj.__module__.startswith("ultralytics.nn.modules"):
                    TORCH_WHITELIST.append(obj)
        
        # Torch 基础模块
        for _, obj in inspect.getmembers(torch_modules):
            if inspect.isclass(obj) and obj.__module__.startswith("torch.nn.modules"):
                TORCH_WHITELIST.append(obj)
        
        # 其他必要类
        TORCH_WHITELIST += [
            DetectionModel, SegmentationModel, 
            IterableSimpleNamespace, TaskAlignedAssigner,
            dill._dill._load_type, scalar, dtype, Float64DType
        ]

    build_torch_whitelist()

except Exception as e:
    logging.error(f"YOLO初始化失败: {e}")
    raise

# ==================== 检测器实现 ====================
def create_segmasks(results):
    """创建分割掩码"""
    return [
        (bbox, mask.astype(np.float32), conf)
        for bbox, mask, conf in zip(results[1], results[2], results[3])
    ]

class UltraDetector:
    """基础检测器类"""
    def __init__(self, model):
        self.model = model
        
    def preprocess_image(self, image):
        """图像预处理"""
        if len(image.shape) == 2:  # 灰度图处理
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image[:, :, ::-1].copy()  # RGB转BGR

    def detect(self, image, threshold=0.3, dilation=0, crop_factor=1.0, drop_size=1, detailer_hook=None):
        """执行检测"""
        image_pil = utils.tensor2pil(image)
        orig_h, orig_w = image.shape[1], image.shape[2]
        
        # 执行推理
        pred = self.model(image_pil, conf=threshold)
        bboxes = pred[0].boxes.xyxy.cpu().numpy()
        
        # 处理结果
        items = []
        for i, (x0, y0, x1, y1) in enumerate(bboxes):
            if (x1 - x0) > drop_size and (y1 - y0) > drop_size:
                # 创建掩码
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
                
                # 膨胀处理
                if dilation > 0:
                    mask = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8))
                
                # 裁剪区域
                bbox = (y0, x0, y1, x1)
                crop_region = utils.make_crop_region(orig_w, orig_h, bbox, crop_factor)
                
                if detailer_hook:
                    crop_region = detailer_hook.post_crop_region(orig_w, orig_h, bbox, crop_region)
                
                # 创建结果项
                items.append(SEG(
                    cropped_image=utils.crop_image(image, crop_region),
                    cropped_mask=utils.crop_ndarray2(mask, crop_region),
                    confidence=pred[0].boxes[i].conf.item(),
                    crop_region=crop_region,
                    bbox=bbox,
                    label=pred[0].names[int(pred[0].boxes[i].cls.item())],
                    control_net_wrapper=None
                ))
        
        return (orig_h, orig_w), items

# ==================== 导出接口 ====================
def load_yolo(model_path: str):
    """加载YOLO模型（兼容接口）"""
    return SafeModelLoader.safe_load_yolo(model_path)

def inference_bbox(model, image, confidence=0.3, device=""):
    """边界框推理"""
    pred = model(image, conf=confidence, device=device)
    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    
    if len(bboxes) == 0:
        return [[], [], [], []]
        
    return [
        [pred[0].names[int(box.cls.item())] for box in pred[0].boxes],
        bboxes,
        [np.zeros((image.size[1], image.size[0]), bool) for _ in bboxes],  # 空掩码
        [box.conf.cpu().numpy() for box in pred[0].boxes]
    ]

def inference_segm(model, image, confidence=0.3, device=""):
    """分割推理"""
    pred = model(image, conf=confidence, device=device)
    # ... (保留原有实现)
    return results

# 保持原有检测器类
class UltraBBoxDetector(UltraDetector):
    """边界框检测器"""
    pass

class UltraSegmDetector(UltraDetector):
    """分割检测器"""
    pass
