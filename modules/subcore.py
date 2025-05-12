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

# ========== NumPy 兼容性补丁 ==========
try:
    from numpy import Float64DType  # 尝试原生导入
except ImportError:
    # 为 NumPy 1.20+ 提供向后兼容
    np.Float64DType = np.dtype('float64')
    Float64DType = np.Float64DType
# =====================================

# 保存原始 torch.load 方法
orig_torch_load = torch.load

# 定义 SEG 结构
SEG = namedtuple("SEG", 
                ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                defaults=[None])

# ==================== 安全配置 ====================
class SecurityConfig:
    @staticmethod
    def init_whitelist():
        """初始化白名单路径"""
        try:
            user_dir = folder_paths.get_user_directory()
            if user_dir and os.path.isdir(user_dir):
                whitelist_dir = os.path.join(user_dir, "default", "ComfyUI-Impact-Subpack")
                whitelist_path = os.path.join(whitelist_dir, "model-whitelist.txt")
                
                os.makedirs(whitelist_dir, exist_ok=True)
                
                if not os.path.exists(whitelist_path):
                    with open(whitelist_path, 'w') as f:
                        f.write("# 安全模型白名单\n# 每行一个模型文件名\n")
                
                logging.info(f"[Security] Whitelist initialized at {whitelist_path}")
                return whitelist_path
                
        except Exception as e:
            logging.error(f"[Security] Whitelist init error: {e}")
            return None

    @staticmethod
    def load_whitelist(whitelist_path):
        """加载白名单内容""" 
        whitelist = set()
        try:
            if whitelist_path and os.path.exists(whitelist_path):
                with open(whitelist_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            whitelist.add(os.path.basename(line))
        except Exception as e:
            logging.error(f"[Security] Whitelist load error: {e}")
        return whitelist

# 初始化安全配置
WHITELIST_PATH = SecurityConfig.init_whitelist()
MODEL_WHITELIST = SecurityConfig.load_whitelist(WHITELIST_PATH)

# ==================== 安全加载器 ====================
class SafeModelLoader:
    @staticmethod
    def torch_wrapper(*args, **kwargs):
        global MODEL_WHITELIST
        
        # 获取文件名
        filename = None
        if args and isinstance(args[0], str):
            filename = os.path.basename(args[0])
        
        # 默认启用安全模式
        load_kwargs = kwargs.copy()
        if 'weights_only' not in load_kwargs:
            load_kwargs['weights_only'] = True

        try:
            return orig_torch_load(*args, **load_kwargs)
            
        except pickle.UnpicklingError as e:
            if 'getattr' in str(e) and filename:
                if filename in MODEL_WHITELIST:
                    logging.warning(f"⚠️ Whitelisted model: {filename}")
                    return orig_torch_load(*args, **{**load_kwargs, 'weights_only': False})
                else:
                    logging.error(f"❌ Blocked: {filename} not in whitelist")
                    raise RuntimeError(f"Model not in whitelist: {filename}")
            raise

# 替换原始加载器
torch.load = SafeModelLoader.torch_wrapper

# ==================== YOLO 初始化 ==================== 
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
    from numpy import dtype
    
    # 构建类白名单
    TORCH_WHITELIST = [
        DetectionModel, SegmentationModel,
        IterableSimpleNamespace, TaskAlignedAssigner,
        dill._dill._load_type, scalar, dtype, Float64DType
    ]
    
    # 添加模块类
    for mod in [modules, block_modules]:
        for _, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and 'ultralytics' in obj.__module__:
                TORCH_WHITELIST.append(obj)
    
    # 添加Torch基础类            
    for _, obj in inspect.getmembers(torch_modules):
        if inspect.isclass(obj) and 'torch.nn' in obj.__module__:
            TORCH_WHITELIST.append(obj)

    logging.info("YOLO dependencies initialized successfully")

except Exception as e:
    logging.error(f"YOLO init failed: {e}")
    raise

# ==================== 检测器实现 ====================
class UltraDetector:
    def __init__(self, model):
        self.model = model
        
    def detect(self, image, threshold=0.3, **kwargs):
        """基础检测方法"""
        image_pil = utils.tensor2pil(image)
        pred = self.model(image_pil, conf=threshold)
        
        # 处理结果
        results = [[], [], [], []]  # labels, bboxes, masks, confidences
        for box in pred[0].boxes:
            results[0].append(pred[0].names[int(box.cls.item())])
            results[1].append(box.xyxy.cpu().numpy()[0])
            results[3].append(box.conf.cpu().numpy())
        
        return results

# ==================== 导出接口 ====================
def load_yolo(model_path):
    """安全加载YOLO模型"""
    if hasattr(torch.serialization, 'safe_globals'):
        with torch.serialization.safe_globals(TORCH_WHITELIST):
            return YOLO(model_path)
    return YOLO(model_path)

# 保持原有类定义
class UltraBBoxDetector(UltraDetector):
    pass

class UltraSegmDetector(UltraDetector):
    pass

# ==================== 工具函数 ====================
def create_segmasks(results):
    return [(bbox, mask.astype(np.float32), conf) 
           for bbox, mask, conf in zip(results[1], results[2], results[3])]

def inference_bbox(model, image, confidence=0.3, device=""):
    """兼容性接口"""
    return UltraBBoxDetector(model).detect(image, confidence)

def inference_segm(model, image, confidence=0.3, device=""):
    """兼容性接口""" 
    return UltraSegmDetector(model).detect(image, confidence)
