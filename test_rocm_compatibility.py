#!/usr/bin/env python3
"""
Test script for ComfyUI Impact Subpack ROCm/PyTorch 2.7+ compatibility
"""

import sys
import os
import traceback
from pathlib import Path

# Add ComfyUI path
COMFYUI_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(COMFYUI_PATH))

def test_imports():
    """Test basic imports"""
    print("🔍 Testing imports...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        from modules import subcore
        print("✅ Impact Subpack modules imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_pytorch_rocm():
    """Test PyTorch ROCm functionality"""
    print("\n🔍 Testing PyTorch ROCm support...")
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
            
            # Test tensor operations on ROCm
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.matmul(x, y)
            print("✅ ROCm tensor operations working")
        else:
            print("⚠️  CUDA/ROCm not available, using CPU")
            
        return True
    except Exception as e:
        print(f"❌ ROCm test failed: {e}")
        return False

def test_model_loading():
    """Test YOLO model loading"""
    print("\n🔍 Testing YOLO model loading...")
    try:
        from modules import subcore
        
        # Find available models
        model_dirs = [
            "/home/armdian/github/ComfyUI/models/ultralytics/segm",
            "/home/armdian/github/ComfyUI/models/ultralytics/bbox"
        ]
        
        models_found = []
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith('.pt'):
                        models_found.append(os.path.join(model_dir, file))
        
        if not models_found:
            print("⚠️  No YOLO models found for testing")
            return True
        
        # Test loading first model
        test_model = models_found[0]
        print(f"Testing model: {os.path.basename(test_model)}")
        
        model = subcore.load_yolo(test_model)
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        
        # Test moving to device if available
        import torch
        if torch.cuda.is_available():
            model.to('cuda')
            print("✅ Model moved to CUDA/ROCm device")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_inference():
    """Test inference functionality"""
    print("\n🔍 Testing inference functionality...")
    try:
        from modules import subcore
        from PIL import Image
        import numpy as np
        import torch
        
        # Create a test image
        test_image = Image.fromarray(np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8))
        
        # Find a model to test
        model_dirs = [
            "/home/armdian/github/ComfyUI/models/ultralytics/segm",
            "/home/armdian/github/ComfyUI/models/ultralytics/bbox"
        ]
        
        test_model = None
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith('.pt'):
                        test_model = os.path.join(model_dir, file)
                        break
            if test_model:
                break
        
        if not test_model:
            print("⚠️  No models available for inference testing")
            return True
        
        print(f"Testing inference with: {os.path.basename(test_model)}")
        model = subcore.load_yolo(test_model)
        
        # Test device detection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Test inference
        if "segm" in test_model:
            results = subcore.inference_segm(model, test_image, confidence=0.5, device=device)
        else:
            results = subcore.inference_bbox(model, test_image, confidence=0.5, device=device)
        
        print(f"✅ Inference completed successfully")
        print(f"Results: {len(results[0]) if results and len(results) > 0 else 0} detections")
        
        return True
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 ComfyUI Impact Subpack ROCm/PyTorch 2.7+ Compatibility Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_pytorch_rocm,
        test_model_loading,
        test_inference
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ComfyUI Impact Subpack is ready for ROCm/AMD GPUs")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
