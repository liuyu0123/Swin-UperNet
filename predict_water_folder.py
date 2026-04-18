# encoding = utf-8

import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

from predict_water_img import get_model, preprocess_image, postprocess_prediction


def create_overlay(image_path, pred, alpha=0.4):
    """创建透明红色叠加图"""
    # 读取原图
    image = Image.open(image_path).convert('RGB')
    image = np.array(image).astype(np.float32)
    
    # 水面区域叠加红色
    water_mask = pred == 1
    overlay = image.copy()
    overlay[water_mask] = overlay[water_mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    
    return overlay.astype(np.uint8)


def batch_inference():
    parser = argparse.ArgumentParser(description="Batch Inference")
    parser.add_argument('--INPUT_DIR', type=str, required=True, help='输入图片目录')
    parser.add_argument('--OUTPUT_DIR', type=str, required=True, help='输出结果目录')
    parser.add_argument('--MODEL_TYPE', type=str, default='upernet')
    parser.add_argument('--BACKBONE_TYPE', type=str, default='swin_t')
    parser.add_argument('--BANDS', type=int, default=3)
    parser.add_argument('--NUM_CLASS', type=int, default=2)
    parser.add_argument('--IMG_SIZE', type=int, default=256)
    parser.add_argument('--GPU_ID', type=int, default=0)
    parser.add_argument('--MODEL_PATH', type=str, required=True)
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = get_model(args)
    model = model.to(device)
    
    # 加载权重
    if os.path.isfile(args.MODEL_PATH):
        print(f"=> Loading model from {args.MODEL_PATH}")
        checkpoint = torch.load(args.MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("=> Model loaded successfully")
    else:
        raise FileNotFoundError(f"Model not found: {args.MODEL_PATH}")
    
    model.eval()
    
    # 创建两个输出目录
    pred_dir = os.path.join(args.OUTPUT_DIR, 'predictions')
    mask_dir = os.path.join(args.OUTPUT_DIR, 'predictions_mask')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # 获取所有图片
    image_paths = glob.glob(os.path.join(args.INPUT_DIR, '*.jpg')) + \
                  glob.glob(os.path.join(args.INPUT_DIR, '*.png')) + \
                  glob.glob(os.path.join(args.INPUT_DIR, '*.jpeg')) + \
                  glob.glob(os.path.join(args.INPUT_DIR, '*.bmp'))
    
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print(f"No images found in {args.INPUT_DIR}")
        return
    
    # 批量推理
    for img_path in tqdm(image_paths):
        # 预处理
        image_tensor, original_size = preprocess_image(img_path, args.BANDS, args.IMG_SIZE)
        image_tensor = image_tensor.to(device).float()
        
        # 推理
        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1)
        
        # 后处理
        pred = postprocess_prediction(pred, original_size, args.NUM_CLASS)
        
        # 获取文件名
        basename = os.path.basename(img_path)
        name_wo_ext = os.path.splitext(basename)[0]
        
        # 1. 保存预测结果 (predictions) - 灰度图 0/127
        pred_img = (pred * 127).astype(np.uint8)
        Image.fromarray(pred_img).save(
            os.path.join(pred_dir, f"{name_wo_ext}_pred.png")
        )
        
        # 2. 保存叠加图 (predictions_mask) - 透明红色
        overlay = create_overlay(img_path, pred, alpha=0.4)
        Image.fromarray(overlay).save(
            os.path.join(mask_dir, f"{name_wo_ext}_overlay.png")
        )
    
    print(f"\nAll results saved:")
    print(f"  - Predictions (gray): {pred_dir}")
    print(f"  - Overlay (red mask): {mask_dir}")
    print(f"Total processed: {len(image_paths)} images")


if __name__ == '__main__':
    batch_inference()