# encoding = utf-8

import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

from predict_water_img import get_model, preprocess_image, postprocess_prediction


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
    
    # 创建模型
    model = get_model(args)
    model = model.to(device)
    
    # 加载权重
    checkpoint = torch.load(args.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    # 创建输出目录
    os.makedirs(args.OUTPUT_DIR, exist_ok=True)
    
    # 获取所有图片
    image_paths = glob.glob(os.path.join(args.INPUT_DIR, '*.jpg')) + \
                  glob.glob(os.path.join(args.INPUT_DIR, '*.png')) + \
                  glob.glob(os.path.join(args.INPUT_DIR, '*.jpeg'))
    
    print(f"Found {len(image_paths)} images")
    
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
        
        # 保存结果
        basename = os.path.basename(img_path)
        name_wo_ext = os.path.splitext(basename)[0]
        
        # 保存为PNG
        pred_img = (pred * 127).astype(np.uint8)
        Image.fromarray(pred_img).save(
            os.path.join(args.OUTPUT_DIR, f"{name_wo_ext}_pred.png")
        )
    
    print(f"All results saved to: {args.OUTPUT_DIR}")


if __name__ == '__main__':
    batch_inference()