# encoding = utf-8

import os
import csv
import time
import argparse
from pathlib import Path

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
parser.add_argument('--OUTPUT', type=str, default=None, help='结果保存路径（可选）')


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


def get_model_info(model):
    """获取模型静态信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


def save_prediction(pred, save_path, img_name):
    """保存预测结果为图像"""
    pred_img = (pred * 127).astype(np.uint8)
    Image.fromarray(pred_img).save(os.path.join(save_path, f"{img_name}_pred.png"))


def save_results_to_csv(metrics, model_info, args, save_path):
    """保存测试结果到 CSV（标准格式）"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建结果字典（与之前格式一致）
    result = {
        'model_path': args.MODEL_PATH,
        'model_type': args.MODEL_TYPE,
        'backbone': args.BACKBONE_TYPE,
        'test_images': args.TEST_IMAGE_DIR,
        'test_masks': args.TEST_MASK_DIR,
        'input_size': args.IMG_SIZE,
        'num_classes': args.NUM_CLASS,
        'total_params': model_info['total_params'],
        'model_size_mb': f"{model_info['model_size_mb']:.2f}",
        'test_loss': f"{metrics['loss']:.6f}",
        'test_precision': f"{metrics['precision']:.6f}",
        'test_recall': f"{metrics['recall']:.6f}",
        'test_f1': f"{metrics['f1']:.6f}",
        'test_miou': f"{metrics['miou']:.6f}",
        'test_acc': f"{metrics['acc']:.6f}",
        'test_kappa': f"{metrics['kappa']:.6f}",
        'inference_time_ms': f"{metrics['inference_time_ms']:.4f}",
        'fps': f"{metrics['fps']:.2f}",
        'total_images': metrics['total_images'],
    }
    
    # 写入 CSV（追加模式，方便对比多个模型）
    header = list(result.keys())
    file_exists = save_path.exists()
    
    with open(save_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
    
    print(f"Results saved to {save_path}")
    
    # 同时保存详细文本报告
    report_path = save_path.parent / f"{save_path.stem}_report.txt"
    with open(report_path, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.MODEL_PATH}\n")
        f.write(f"Model Type: {args.MODEL_TYPE}\n")
        f.write(f"Backbone: {args.BACKBONE_TYPE}\n")
        f.write(f"Test Images: {args.TEST_IMAGE_DIR}\n")
        f.write(f"Test Masks: {args.TEST_MASK_DIR}\n\n")
        f.write(f"Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
        f.write(f"Input Size: {args.IMG_SIZE}\n")
        f.write(f"Number of Classes: {args.NUM_CLASS}\n")
        f.write(f"Batch Size: {args.BATCH_SIZE}\n\n")
        f.write(f"Test Set Size: {metrics['total_images']} images\n\n")
        f.write(f"Loss:           {metrics['loss']:.6f}\n")
        f.write(f"Accuracy:       {metrics['acc']:.6f}\n")
        f.write(f"Kappa:          {metrics['kappa']:.6f}\n")
        f.write(f"mIoU:           {metrics['miou']:.6f}\n")
        f.write(f"Precision:      {metrics['precision']:.6f}\n")
        f.write(f"Recall:         {metrics['recall']:.6f}\n")
        f.write(f"F1-Score:       {metrics['f1']:.6f}\n")
        f.write(f"Inference Time: {metrics['inference_time_ms']:.4f} ms\n")
        f.write(f"FPS:            {metrics['fps']:.2f}\n")
        f.write(f"{'='*50}\n")
    
    print(f"Text report appended to {report_path}")


def test():
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = get_model(args)
    model = model.to(device)
    
    # 获取模型信息
    model_info = get_model_info(model)
    print(f"Model: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")
    
    # 加载权重
    if os.path.isfile(args.MODEL_PATH):
        print(f"=> Loading model from {args.MODEL_PATH}")
        checkpoint = torch.load(args.MODEL_PATH, map_location=device, weights_only=False)
        
        # 过滤非模型参数
        for key in ['epoch', 'optimizer', 'miou', 'metrics']:
            checkpoint.pop(key, None)
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
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
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
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
    
    # 测量推理时间
    inference_times = []
    total_images = 0
    
    print(f"\nTesting on {len(test_dataset)} images...")
    
    with torch.no_grad():
        test_loader = tqdm(test_loader)
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device).float()
            labels = labels.to(device).long()
            
            # 测量推理时间
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            
            # 推理
            outputs = model(images)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_time = time.time() - start
            inference_times.append(batch_time)
            
            # 计算 loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            # 获取预测
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            # 添加到评估器
            evaluator.add_batch(labels, preds)
            total_images += images.size(0)
            
            # 保存预测结果（可选，只保存第一批前5个）
            if args.SAVE_PRED and i == 0:
                for j in range(min(5, len(preds))):
                    save_prediction(preds[j], save_dir, f"sample_{j}")
            
            test_loader.set_description(f'Test Loss: {total_loss/total_images:.3f}')
    
    # 计算指标
    Acc = evaluator.OverAll_Accuracy()
    Kappa = evaluator.Kappa()
    mIoU, IoU = evaluator.mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    mPrecision, Precision = evaluator.Precision()
    mRecall, Recall = evaluator.Recall()
    mF1, F1 = evaluator.F1_Score()
    
    # 计算推理时间
    avg_inference_time_ms = np.mean(inference_times) * 1000
    fps = args.BATCH_SIZE / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
    
    # 构建指标字典
    metrics = {
        'loss': total_loss / total_images,
        'acc': Acc,
        'kappa': Kappa,
        'miou': mIoU,
        'precision': mPrecision,
        'recall': mRecall,
        'f1': mF1,
        'inference_time_ms': avg_inference_time_ms,
        'fps': fps,
        'total_images': total_images,
    }
    
    # 打印结果
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Loss:      {metrics['loss']:.4f}")
    print(f"Accuracy:       {metrics['acc']:.4f}")
    print(f"Kappa:          {metrics['kappa']:.4f}")
    print(f"mIoU:           {metrics['miou']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1-Score:       {metrics['f1']:.4f}")
    print(f"Inference Time: {metrics['inference_time_ms']:.4f} ms")
    print(f"FPS:            {metrics['fps']:.2f}")
    print("="*50)
    print(f"\nPer-class IoU: {IoU}")
    print(f"Per-class Precision: {Precision}")
    print(f"Per-class Recall: {Recall}")
    print(f"Per-class F1: {F1}")
    
    # 保存结果
    if args.OUTPUT:
        output_path = args.OUTPUT
    else:
        # 默认保存到模型目录
        model_dir = Path(args.MODEL_PATH).parent
        output_path = model_dir / f"{args.MODEL_TYPE}_test_results.csv"
    
    save_results_to_csv(metrics, model_info, args, output_path)
    
    return metrics


if __name__ == '__main__':
    test()