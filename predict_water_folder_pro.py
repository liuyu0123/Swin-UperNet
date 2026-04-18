# encoding = utf-8

import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import csv
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import Counter

from predict_water_img import get_model, preprocess_image, postprocess_prediction


def calculate_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int = 2) -> Dict[str, float]:
    """计算分割指标：Precision, Recall, F1, mIoU"""
    pred_flat = pred.astype(np.int64).flatten()
    target_flat = target.astype(np.int64).flatten()
    
    # 计算混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum((pred_flat == i) & (target_flat == j))
    
    # 计算指标
    ious, precisions, recalls = [], [], []
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        ious.append(iou)
    
    water_class = 1
    f1 = 2 * precisions[water_class] * recalls[water_class] / (precisions[water_class] + recalls[water_class]) \
         if (precisions[water_class] + recalls[water_class]) > 0 else 0.0
    
    return {
        'precision': precisions[water_class],
        'recall': recalls[water_class],
        'f1_score': f1,
        'miou': np.mean(ious),
        'water_iou': ious[water_class],
        'non_water_iou': ious[0]
    }


def create_overlay(image_path: str, pred: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """创建透明红色叠加图（水为红色，非水为透明）"""
    image = Image.open(image_path).convert('RGB')
    image = np.array(image).astype(np.float32)
    
    water_mask = pred == 1
    overlay = image.copy()
    overlay[water_mask] = overlay[water_mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    
    return overlay.astype(np.uint8)


def find_ground_truth(image_path: str, gt_dir: str) -> Optional[str]:
    """根据文件名查找真值mask"""
    basename = Path(image_path).stem
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        candidate = os.path.join(gt_dir, basename + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def load_ground_truth(mask_path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """加载真值mask并resize"""
    try:
        mask = Image.open(mask_path).convert('L').resize(target_size, Image.NEAREST)
        mask = np.array(mask)
        # 二值化：支持0/1或0/255
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        return mask
    except Exception as e:
        print(f"警告: 无法加载真值 {mask_path}: {e}")
        return None


def process_single_image(image_path: str, model: torch.nn.Module, device: torch.device,
                        args, output_dir: Optional[str] = None,
                        gt_dir: Optional[str] = None,
                        measure_time: bool = True) -> Optional[Dict]:
    """处理单张图像"""
    try:
        basename = Path(image_path).name

        # 预处理
        image_tensor, original_size = preprocess_image(image_path, args.BANDS, args.IMG_SIZE)
        image_tensor = image_tensor.to(device).float()

        # 推理（带计时）
        with torch.no_grad():
            if measure_time:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

            output = model(image_tensor)
            pred = torch.argmax(output, dim=1)

            if measure_time:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                inference_time = end_time - start_time
                fps = 1.0 / inference_time if inference_time > 0 else 0.0
            else:
                inference_time = None
                fps = None

        # 后处理恢复尺寸
        pred = postprocess_prediction(pred, original_size, args.NUM_CLASS)

        # 保存叠加图（如果提供output_dir）
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            overlay = create_overlay(image_path, pred, alpha=0.4)
            save_path = os.path.join(output_dir, f"{Path(image_path).stem}_overlay.png")
            Image.fromarray(overlay).save(save_path)

        # 评估（如果提供gt_dir）
        metrics = None
        if gt_dir:
            gt_path = find_ground_truth(image_path, gt_dir)
            if gt_path:
                gt_mask = load_ground_truth(gt_path, original_size)
                if gt_mask is not None:
                    metrics = calculate_metrics(pred, gt_mask, args.NUM_CLASS)
                    metrics['image_name'] = basename
            else:
                print(f"警告: 未找到 {basename} 的真值mask")

        # 附加推理时间信息
        if metrics is not None and inference_time is not None:
            metrics['inference_time'] = inference_time
            metrics['fps'] = fps

        return metrics
    
    except Exception as e:
        print(f"错误处理 {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_metrics_to_csv(metrics_list: List[Dict], output_path: str):
    """保存指标到CSV，包含每图结果和平均值"""
    if not metrics_list:
        return

    # 计算平均值
    avg = {
        'image_name': 'AVERAGE',
        'precision': np.mean([m['precision'] for m in metrics_list]),
        'recall': np.mean([m['recall'] for m in metrics_list]),
        'f1_score': np.mean([m['f1_score'] for m in metrics_list]),
        'miou': np.mean([m['miou'] for m in metrics_list]),
        'water_iou': np.mean([m['water_iou'] for m in metrics_list]),
        'non_water_iou': np.mean([m['non_water_iou'] for m in metrics_list]),
        'inference_time': np.mean([m['inference_time'] for m in metrics_list]),
        'fps': np.mean([m['fps'] for m in metrics_list])
    }

    # 写入CSV
    fieldnames = ['image_name', 'precision', 'recall', 'f1_score', 'miou',
                  'water_iou', 'non_water_iou', 'inference_time', 'fps']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list + [avg]:
            writer.writerow({k: m.get(k, '') for k in fieldnames})

    print(f"\n评估结果已保存: {output_path}")


def get_image_paths(input_path: str) -> List[str]:
    """自适应获取图像路径（文件或文件夹）"""
    input_path = os.path.abspath(input_path)
    
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        paths = []
        for p in patterns:
            paths.extend(glob.glob(os.path.join(input_path, p)))
            paths.extend(glob.glob(os.path.join(input_path, p.upper())))
        return sorted(list(set(paths)))
    else:
        raise ValueError(f"路径不存在: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Water Segmentation Inference")
    
    # 主要参数
    parser.add_argument('INPUT', type=str, help='输入图像或文件夹路径')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                       help='输出目录（可选，不提供则不保存结果图）')
    parser.add_argument('-g', '--ground_truth_dir', type=str, default=None,
                       help='真值mask目录（可选，提供则生成评估CSV）')
    
    # 模型参数（与原脚本一致）
    parser.add_argument('--MODEL_TYPE', type=str, default='upernet')
    parser.add_argument('--BACKBONE_TYPE', type=str, default='swin_t')
    parser.add_argument('--BANDS', type=int, default=3)
    parser.add_argument('--NUM_CLASS', type=int, default=2)
    parser.add_argument('--IMG_SIZE', type=int, default=256)
    parser.add_argument('--GPU_ID', type=int, default=0)
    parser.add_argument('--MODEL_PATH', type=str, required=True)
    
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device(f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = get_model(args)
    model = model.to(device)
    
    # 加载权重（修复版）
    if os.path.isfile(args.MODEL_PATH):
        print(f"=> 加载模型: {args.MODEL_PATH}")
        try:
            checkpoint = torch.load(args.MODEL_PATH, map_location=device, weights_only=True)
        except Exception:
            print(f"Note: 使用兼容模式加载模型")
            checkpoint = torch.load(args.MODEL_PATH, map_location=device, weights_only=False)
        
        # 关键修复：提取 state_dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                print("=> 检测到完整checkpoint，提取 state_dict")
                state_dict = checkpoint['state_dict']
                if 'miou' in checkpoint:
                    print(f"=> 训练时最佳mIoU: {checkpoint['miou']:.4f}")
                if 'epoch' in checkpoint:
                    print(f"=> 训练轮次: {checkpoint['epoch']}")
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 加载到模型
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"警告: {len(missing)} 个层使用随机初始化")
        if unexpected:
            print(f"警告: {len(unexpected)} 个权重被忽略")
        
        print("=> 模型加载成功")
    else:
        raise FileNotFoundError(f"未找到模型: {args.MODEL_PATH}")
    
    model.eval()

    # 获取输入图像
    image_paths = get_image_paths(args.INPUT)
    print(f"\n找到 {len(image_paths)} 张图像")
    if not image_paths:
        return

    # Warm-up：用第一张图预热，避免CUDA/CNN初始化时间污染首图性能
    print("正在进行 Warm-up（第一张图不计时）...")
    _ = process_single_image(image_paths[0], model, device, args,
                             output_dir=None, gt_dir=None, measure_time=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warm-up 完成\n")

    # 批量处理
    all_metrics = []
    for img_path in tqdm(image_paths, desc="处理进度"):
        metrics = process_single_image(img_path, model, device, args,
                                      args.output_dir, args.ground_truth_dir)
        if metrics:
            all_metrics.append(metrics)

    # 输出评估结果
    if args.ground_truth_dir and all_metrics:
        if args.output_dir:
            csv_path = os.path.join(args.output_dir, 'evaluation_metrics.csv')
            save_metrics_to_csv(all_metrics, csv_path)
        else:
            # 无output_dir时打印到终端
            print("\n" + "="*60)
            print("评估结果（无output_dir，仅终端输出）")
            print("="*60)
            for m in all_metrics:
                print(f"{m['image_name']}: P={m['precision']:.3f}, R={m['recall']:.3f}, "
                      f"F1={m['f1_score']:.3f}, mIoU={m['miou']:.3f}, "
                      f"Time={m['inference_time']:.4f}s, FPS={m['fps']:.2f}")

            avg_p = np.mean([m['precision'] for m in all_metrics])
            avg_r = np.mean([m['recall'] for m in all_metrics])
            avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
            avg_miou = np.mean([m['miou'] for m in all_metrics])
            avg_time = np.mean([m['inference_time'] for m in all_metrics])
            avg_fps = np.mean([m['fps'] for m in all_metrics])
            print("-"*60)
            print(f"AVERAGE:       P={avg_p:.3f}, R={avg_r:.3f}, F1={avg_f1:.3f}, mIoU={avg_miou:.3f}, "
                  f"Time={avg_time:.4f}s, FPS={avg_fps:.2f}")
            print("="*60)
    
    print(f"\n✓ 完成！共处理 {len(image_paths)} 张图像")
    if args.output_dir:
        print(f"✓ 结果保存至: {args.output_dir}")


if __name__ == '__main__':
    main()