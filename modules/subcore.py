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
            return orig_torch_load(*args, **load_kwargs)
            
        except pickle.UnpicklingError as e:
            if 'getattr' in str(e) and filename:
                if filename in MODEL_WHITELIST:
                    logging.warning(f"⚠️ Whitelisted unsafe load: {filename}")
                    return orig_torch_load(*args, **{**load_kwargs, 'weights_only': False})
                else:
                    logging.error(f"❌ Blocked unsafe load: {filename}. Add to whitelist if trusted.")
                    raise RuntimeError(f"Model not in whitelist: {filename}")
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
    logging.info("YOLO dependencies initialized successfully")

except Exception as e:
    logging.error(f"YOLO初始化失败: {e}")
    raise

# ==================== 检测器实现 ====================
class UltraDetector:
    """基础检测器类（兼容新旧参数格式）"""
    def __init__(self, model):
        self.model = model
        
    def detect(self, image, threshold=0.3, dilation=0, crop_factor=1.0, drop_size=1, detailer_hook=None):
        """
        执行检测（兼容新旧参数格式）
        参数:
            image: 输入图像张量
            threshold: 置信度阈值
            dilation: 掩码膨胀像素
            crop_factor: 裁剪区域扩展系数
            drop_size: 最小检测尺寸
            detailer_hook: 后处理钩子
        """
        # 转换输入图像
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

    def detect_combined(self, image, threshold=0.3, dilation=0):
        """组合掩码检测"""
        results = self.detect(image, threshold, dilation)
        return utils.combine_masks([(seg.cropped_mask, seg.bbox) for seg in results[1]])

# ==================== 兼容性接口 ====================
def create_segmasks(results):
    """创建分割掩码（兼容旧版）"""
    return [(bbox, mask.astype(np.float32), conf) 
           for bbox, mask, conf in zip(results[1], results[2], results[3])]

class UltraBBoxDetector(UltraDetector):
    """边界框检测器（兼容旧版接口）"""
    def detect(self, image, confidence=0.3, device=""):
        """兼容旧版接口"""
        return super().detect(image, threshold=confidence)

class UltraSegmDetector(UltraDetector):
    """分割检测器（兼容旧版接口）"""
    def detect(self, image, confidence=0.3, device=""):
        """兼容旧版接口"""
        return super().detect(image, threshold=confidence)

# ==================== 导出函数 ====================
def load_yolo(model_path: str):
    """加载YOLO模型（兼容接口）"""
    return SafeModelLoader.safe_load_yolo(model_path)

def inference_bbox(model, image, confidence=0.3, device=""):
    """边界框推理（兼容旧版）"""
    return UltraBBoxDetector(model).detect(image, confidence)

def inference_segm(model, image, confidence=0.3, device=""):
    """分割推理（兼容旧版）"""
    return UltraSegmDetector(model).detect(image, confidence)
