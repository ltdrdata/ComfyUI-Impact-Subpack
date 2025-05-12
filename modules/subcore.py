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

# ========== 基础配置 ==========
orig_torch_load = torch.load

# 定义SEG数据结构
SEG = namedtuple("SEG",
                ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                defaults=[None])

# ========== 白名单管理 ==========
class WhitelistManager:
    @staticmethod
    def init():
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
                logging.info(f"[Whitelist] Initialized at {whitelist_path}")
                return whitelist_path
        except Exception as e:
            logging.error(f"[Whitelist] Init failed: {e}")
        return None

    @staticmethod
    def load(whitelist_path):
        """加载白名单内容"""
        if not whitelist_path:
            return set()
            
        approved = set()
        try:
            with open(whitelist_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        approved.add(os.path.basename(line))
            logging.info(f"[Whitelist] Loaded {len(approved)} models")
        except Exception as e:
            logging.error(f"[Whitelist] Load error: {e}")
        return approved

# 初始化白名单
WHITELIST_PATH = WhitelistManager.init()
MODEL_WHITELIST = WhitelistManager.load(WHITELIST_PATH)

# ========== 安全加载器 ==========
class SafeLoader:
    @staticmethod
    def torch_wrapper(*args, **kwargs):
        """安全加载包装器"""
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
                    logging.warning(f"⚠️ Whitelisted load: {filename}")
                    return orig_torch_load(*args, **{**load_kwargs, 'weights_only': False})
                else:
                    logging.error(f"❌ Blocked: {filename} not in whitelist")
                    raise RuntimeError(f"Model not in whitelist: {filename}")
            raise

# 替换原始加载器
torch.load = SafeLoader.torch_wrapper

# ========== YOLO 初始化 ==========
try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel
    from ultralytics.utils import IterableSimpleNamespace, TaskAlignedAssigner
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
    logging.info("YOLO dependencies initialized")

except Exception as e:
    logging.error(f"YOLO init failed: {e}")
    raise

# ========== 检测器基类 ==========
class BaseDetector:
    """基础检测器类"""
    def __init__(self, model):
        self.model = model
        
    def preprocess_image(self, image):
        """图像预处理"""
        if len(image.shape) == 2:  # 灰度图处理
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image[:, :, ::-1].copy()  # RGB转BGR

    def _create_mask(self, image_shape, bbox):
        """创建基础掩码"""
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.rectangle(mask, (int(bbox[0]), int(bbox[1])), 
                      (int(bbox[2]), int(bbox[3])), 255, -1)
        return mask

# ========== 具体检测器实现 ==========
class BBoxDetector(BaseDetector):
    """边界框检测器"""
    def detect(self, image, threshold=0.3, dilation=0, crop_factor=1.0, drop_size=1, detailer_hook=None):
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
                mask = self._create_mask((orig_h, orig_w), (x0, y0, x1, y1))
                
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

class SegmDetector(BaseDetector):
    """分割检测器"""
    def detect(self, image, threshold=0.3, dilation=0, crop_factor=1.0, drop_size=1, detailer_hook=None):
        image_pil = utils.tensor2pil(image)
        orig_h, orig_w = image.shape[1], image.shape[2]
        
        # 执行推理
        pred = self.model(image_pil, conf=threshold)
        bboxes = pred[0].boxes.xyxy.cpu().numpy()
        segms = pred[0].masks.data.cpu().numpy()
        
        # 处理结果
        items = []
        for i, (x0, y0, x1, y1) in enumerate(bboxes):
            if (x1 - x0) > drop_size and (y1 - y0) > drop_size:
                # 获取分割掩码
                mask = segms[i]
                
                # 调整掩码尺寸
                mask = torch.from_numpy(mask)
                scaled_mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0), 
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                
                # 膨胀处理
                if dilation > 0:
                    scaled_mask = cv2.dilate(scaled_mask, np.ones((dilation, dilation), np.uint8))
                
                # 裁剪区域
                bbox = (y0, x0, y1, x1)
                crop_region = utils.make_crop_region(orig_w, orig_h, bbox, crop_factor)
                
                if detailer_hook:
                    crop_region = detailer_hook.post_crop_region(orig_w, orig_h, bbox, crop_region)
                
                # 创建结果项
                items.append(SEG(
                    cropped_image=utils.crop_image(image, crop_region),
                    cropped_mask=utils.crop_ndarray2(scaled_mask, crop_region),
                    confidence=pred[0].boxes[i].conf.item(),
                    crop_region=crop_region,
                    bbox=bbox,
                    label=pred[0].names[int(pred[0].boxes[i].cls.item())],
                    control_net_wrapper=None
                ))
        
        return (orig_h, orig_w), items

# ========== 兼容性接口 ==========
def create_segmasks(results):
    """创建分割掩码（兼容旧版）"""
    return [(bbox, mask.astype(np.float32), conf) 
           for bbox, mask, conf in zip(results[1], results[2], results[3])]

def load_yolo(model_path: str):
    """安全加载YOLO模型"""
    if hasattr(torch.serialization, 'safe_globals'):
        with torch.serialization.safe_globals(TORCH_WHITELIST):
            return YOLO(model_path)
    return YOLO(model_path)

# ========== 导出接口 ==========
def UltraBBoxDetector(model):
    """边界框检测器工厂函数"""
    return BBoxDetector(model)

def UltraSegmDetector(model):
    """分割检测器工厂函数"""
    return SegmDetector(model)

def inference_bbox(model, image, confidence=0.3, device=""):
    """边界框推理（兼容旧版）"""
    return BBoxDetector(model).detect(image, threshold=confidence)

def inference_segm(model, image, confidence=0.3, device=""):
    """分割推理（兼容旧版）"""
    return SegmDetector(model).detect(image, threshold=confidence)
