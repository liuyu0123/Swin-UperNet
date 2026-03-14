# encoding = utf-8

import os
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

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

from utils.dataset import Labeled_Model_Dataset
from utils.metrics import Evaluator


parser = argparse.ArgumentParser(description="Model Testing")
parser.add_argument('--TEST_IMAGE_DIR', type=str, required=True, help='测试集图像目录')
parser.add_argument('--TEST_MASK_DIR', type=str, required=True, help='测试集标签目录')
parser.add_argument('--MODEL_TYPE', type=str, default='upernet')
parser.add_argument('--BACKBONE_TYPE', type=str, default='swin_t')
parser.add_argument('--BANDS', type=int, default=3)
parser.add_argument('--NUM_CLASS', type=int, default=2)
parser.add_argument('--IMG_SIZE', type=int, default=256)
parser.add_argument('--BATCH_SIZE', type=int, default=16)
parser.add_argument('--GPU_ID', type=int, default=0)
parser.add_argument('--MODEL_PATH', type=str, required=True, help='模型权重路径')
parser.add_argument('--SAVE_PRED', type=bool, default=True, help='是否保存预测结果')


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


def save_prediction(pred, save_path, img_name):
    """保存预测结果为图像"""
    # pred: (H, W) 类别索引
    pred_img = (pred * 127).astype(np.uint8)  # 0->0, 1->127
    Image.fromarray(pred_img).save(os.path.join(save_path, f"{img_name}_pred.png"))


def test():
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    
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
    
    # 创建测试数据集
    test_dataset = Labeled_Model_Dataset(
        args.TEST_IMAGE_DIR, 
        args.TEST_MASK_DIR, 
        bands=args.BANDS, 
        img_size=args.IMG_SIZE
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    # 创建保存目录
    if args.SAVE_PRED:
        save_dir = os.path.join(os.path.dirname(args.MODEL_PATH), 'predictions')
        os.makedirs(save_dir, exist_ok=True)
        print(f"Predictions will be saved to: {save_dir}")
    
    # 评估器
    evaluator = Evaluator(args.NUM_CLASS)
    evaluator.reset()
    
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTesting on {len(test_dataset)} images...")
    
    with torch.no_grad():
        test_loader = tqdm(test_loader)
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device).float()
            labels = labels.to(device).long()
            
            # 推理
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 获取预测
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            # 添加到评估器
            evaluator.add_batch(labels, preds)
            
            # 保存预测结果（可选）
            if args.SAVE_PRED and i == 0:  # 只保存第一批作为示例
                for j in range(min(5, len(preds))):
                    save_prediction(preds[j], save_dir, f"sample_{j}")
            
            test_loader.set_description(f'Test Loss: {total_loss/(i+1):.3f}')
    
    # 计算指标
    Acc = evaluator.OverAll_Accuracy()
    Kappa = evaluator.Kappa()
    mIoU, IoU = evaluator.mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    mPrecision, Precision = evaluator.Precision()
    mRecall, Recall = evaluator.Recall()
    mF1, F1 = evaluator.F1_Score()
    
    # 打印结果
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Loss: {total_loss/len(test_loader):.4f}")
    print(f"Accuracy:  {Acc:.4f}")
    print(f"Kappa:     {Kappa:.4f}")
    print(f"mIoU:      {mIoU:.4f}")
    print(f"FWIoU:     {FWIoU:.4f}")
    print(f"Precision: {mPrecision:.4f}")
    print(f"Recall:    {mRecall:.4f}")
    print(f"F1-Score:  {mF1:.4f}")
    print("="*50)
    print(f"\nPer-class IoU: {IoU}")
    print(f"Per-class Precision: {Precision}")
    print(f"Per-class Recall: {Recall}")
    print(f"Per-class F1: {F1}")
    
    return Acc, mIoU, mF1


if __name__ == '__main__':
    test()