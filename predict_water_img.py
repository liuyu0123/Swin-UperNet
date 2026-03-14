# encoding = utf-8

import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from nets.unet import Unet
# from nets.deeplabv3_plus import DeepLab
from nets.ENet import ENet
from nets.fcn import FCN16s
from nets.hrnet import hrnet
from nets.pspnet import PSPNet
from nets.segformer import Segformer
from nets.segnet import SegNet
from nets.SETR import SETR
from nets.refinenet import rf50
from nets.UperNet import UperNet
from nets.segnext import SegNeXt
from nets.hrnet_ocr import hrnetocr
from nets.mask2former import Mask2Former


def get_model(args):
    """根据参数创建模型"""
    if args.MODEL_TYPE == 'unet':
        model = Unet(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=args.BACKBONE_TYPE)
    elif args.MODEL_TYPE == 'deeplab':
        model = DeepLab(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=args.BACKBONE_TYPE)
    elif args.MODEL_TYPE == 'fcn':
        model = FCN16s(bands=args.BANDS, num_classes=args.NUM_CLASS)
    elif args.MODEL_TYPE == 'hrnet':
        model = hrnet(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=18, version='v2')
    elif args.MODEL_TYPE == 'pspnet':
        model = PSPNet(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=args.BACKBONE_TYPE)
    elif args.MODEL_TYPE == 'refinenet':
        model = rf50(bands=args.BANDS, num_classes=args.NUM_CLASS)
    elif args.MODEL_TYPE == 'enet':
        model = ENet(bands=args.BANDS, num_classes=args.NUM_CLASS)
    elif args.MODEL_TYPE == 'segformer':
        model = Segformer(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone='b0')
    elif args.MODEL_TYPE == 'setr':
        model = SETR(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone='Base', img_size=256)
    elif args.MODEL_TYPE == 'segnet':
        model = SegNet(bands=args.BANDS, num_classes=args.NUM_CLASS)
    elif args.MODEL_TYPE == 'upernet':
        model = UperNet(bands=args.BANDS, num_classes=args.NUM_CLASS)
    elif args.MODEL_TYPE == 'segnext':
        model = SegNeXt(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone='T')
    elif args.MODEL_TYPE == 'ocrnet':
        model = hrnetocr(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=48)
    elif args.MODEL_TYPE == 'mask2former':
        model = Mask2Former(bands=args.BANDS, num_classes=args.NUM_CLASS)
    else:
        raise NotImplementedError(f'Model type {args.MODEL_TYPE} not supported')
    return model


def preprocess_image(image_path, bands=3, img_size=256):
    """预处理单张图像"""
    image = Image.open(image_path)
    original_size = image.size
    
    image = image.resize((img_size, img_size), Image.BILINEAR)
    image = np.array(image).astype(np.float32)
    
    if len(image.shape) == 2:
        image = np.stack([image] * bands, axis=0)
    else:
        image = image.transpose(2, 0, 1)
        if image.shape[0] < bands:
            repeat_times = bands // image.shape[0] + 1
            image = np.tile(image, (repeat_times, 1, 1))[:bands]
        elif image.shape[0] > bands:
            image = image[:bands]
    
    image = image / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    
    return image, original_size


def postprocess_prediction(pred, original_size, num_classes=2):
    """后处理预测结果"""
    pred = pred.squeeze(0).cpu().numpy()
    pred_img = Image.fromarray((pred * 127).astype(np.uint8))
    pred_img = pred_img.resize(original_size, Image.NEAREST)
    pred = np.array(pred_img) // 127
    
    return pred


def visualize_result(image_path, pred, save_path=None, show=True):
    """可视化推理结果 - 水面用透明红色表示"""
    # 读取原图
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # 创建颜色映射 - 水面用红色
    colors = {
        0: [0, 0, 0],       # 背景 - 黑色
        1: [255, 0, 0],     # 水 - 红色
    }
    
    # 创建彩色预测图
    pred_color = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for cls, color in colors.items():
        pred_color[pred == cls] = color
    
    # 创建叠加图 - 水面透明红色
    overlay = image.copy().astype(np.float32)
    # 水面区域 (pred == 1) 叠加红色，透明度 40%
    water_mask = pred == 1
    overlay[water_mask] = overlay[water_mask] * 0.6 + np.array([255, 0, 0]) * 0.4
    overlay = overlay.astype(np.uint8)
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Prediction (0:BG, 1:Water)')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red = Water)')
    axes[2].axis('off')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color=[0, 0, 0], label='Background'),
        mpatches.Patch(color=[1, 0, 0], label='Water')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Result saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def inference():
    parser = argparse.ArgumentParser(description="Single Image Inference")
    parser.add_argument('--IMAGE_PATH', type=str, required=True, help='单张图片路径')
    parser.add_argument('--MODEL_TYPE', type=str, default='upernet')
    parser.add_argument('--BACKBONE_TYPE', type=str, default='swin_t')
    parser.add_argument('--BANDS', type=int, default=3)
    parser.add_argument('--NUM_CLASS', type=int, default=2)
    parser.add_argument('--IMG_SIZE', type=int, default=256)
    parser.add_argument('--GPU_ID', type=int, default=0)
    parser.add_argument('--MODEL_PATH', type=str, required=True, help='模型权重路径')
    parser.add_argument('--SAVE_PATH', type=str, default=None, help='保存结果路径')
    parser.add_argument('--SHOW', type=bool, default=True, help='是否显示结果')
    
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
    
    # 预处理
    print(f"Processing image: {args.IMAGE_PATH}")
    image_tensor, original_size = preprocess_image(args.IMAGE_PATH, args.BANDS, args.IMG_SIZE)
    image_tensor = image_tensor.to(device).float()
    
    # 推理
    print("Running inference...")
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1)
    
    # 后处理
    pred = postprocess_prediction(pred, original_size, args.NUM_CLASS)
    
    # 统计信息
    water_pixels = np.sum(pred == 1)
    total_pixels = pred.size
    water_ratio = water_pixels / total_pixels * 100
    
    print(f"\nInference Results:")
    print(f"Image size: {original_size}")
    print(f"Water pixels: {water_pixels} / {total_pixels} ({water_ratio:.2f}%)")
    
    # 可视化
    if args.SAVE_PATH is None:
        args.SAVE_PATH = os.path.join(
            os.path.dirname(args.MODEL_PATH),
            f"pred_{os.path.basename(args.IMAGE_PATH)}"
        )
    
    visualize_result(args.IMAGE_PATH, pred, args.SAVE_PATH, args.SHOW)
    
    print(f"\nDone!")


if __name__ == '__main__':
    inference()