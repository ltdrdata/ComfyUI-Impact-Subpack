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

# ========== 安全加载修复 ==========
class SafeGlobalsWrapper:
    def __init__(self, allowed_list=None):
        self.allowed_list = allowed_list or []
        
    def __enter__(self):
        if hasattr(torch.serialization, '_safe_globals'):
            torch.serialization._safe_globals = self.allowed_list
        return self
        
    def __exit__(self, *args):
        if hasattr(torch.serialization, '_safe_globals'):
            torch.serialization._safe_globals = None

# 保存原始 torch.load
orig_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    """修复后的安全加载函数"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = True
        
    try:
        # 尝试普通加载
        return orig_torch_load(*args, **kwargs)
    except (pickle.UnpicklingError, TypeError) as e:
        if "missing 1 required positional argument: 'safe_globals'" in str(e):
            # 修复 safe_globals 参数问题
            with SafeGlobalsWrapper():
                return orig_torch_load(*args, **kwargs)
        raise

# 替换原始加载器
torch.load = safe_torch_load

# ========== YOLO 加载器 ==========
def load_yolo(model_path):
    """兼容新旧版 PyTorch 的安全加载"""
    try:
        from ultralytics import YOLO
        
        # 构建允许的类列表
        allowed_classes = []
        try:
            from ultralytics.nn.tasks import DetectionModel, SegmentationModel
            allowed_classes.extend([DetectionModel, SegmentationModel])
        except ImportError:
            pass
            
        # 使用修复后的安全上下文
        with SafeGlobalsWrapper(allowed_classes):
            return YOLO(model_path)
            
    except Exception as e:
        logging.error(f"YOLO加载失败: {e}")
        raise

# ========== 其他兼容性修复 ==========
# 确保 numpy 兼容性
try:
    from numpy import Float64DType
except ImportError:
    Float64DType = type(np.float64())

# 保持原有 SEG 定义
SEG = namedtuple("SEG", 
                ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                defaults=[None])
