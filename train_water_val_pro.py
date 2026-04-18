# encoding = utf-8

import os
import csv
import argparse
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data

from nets.unet import Unet
from nets.ENet import ENet
from nets.fcn import FCN16s, FCN8s, FCN32s
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

from torch.utils.data import DataLoader
from utils.dataset import Labeled_Model_Dataset
from utils.metrics import Evaluator
from utils.weight_init import weights_init
from utils.focal import FocalLoss
from tqdm import tqdm


def get_model_info(model):
    """获取模型静态信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


def compute_metrics_from_evaluator(evaluator, num_classes):
    """从 Evaluator 提取核心指标，增加前景类（最后一类）单独指标"""
    Acc = evaluator.OverAll_Accuracy()
    Kappa = evaluator.Kappa()
    mIoU, IoU = evaluator.mean_Intersection_over_Union()
    mPrecision, Precision = evaluator.Precision()
    mRecall, Recall = evaluator.Recall()
    mF1_score, F1_score = evaluator.F1_Score()
    
    # 提取前景类指标（假设最后一类为前景/目标类）
    # 对于二分类（0=背景，1=前景），取索引 1
    fg_index = -1  # 最后一类
    
    metrics = {
        'acc': Acc,
        'kappa': Kappa,
        'miou': mIoU,
        'precision': mPrecision,
        'recall': mRecall,
        'f1': mF1_score,
        # 前景类单独指标
        'fg_precision': Precision[fg_index] if len(Precision) > 1 else Precision[0],
        'fg_recall': Recall[fg_index] if len(Recall) > 1 else Recall[0],
        'fg_f1': F1_score[fg_index] if len(F1_score) > 1 else F1_score[0],
        'fg_miou': IoU[fg_index] if len(IoU) > 1 else IoU[0],
    }
    
    # 诊断：检测是否全背景预测（训练初期常见现象）
    if metrics['fg_recall'] == 0 and abs(metrics['recall'] - 0.5) < 0.01:
        print(f"\n[WARNING] 检测到模型预测全为背景（Foreground Recall=0）！")
        print("          这是训练初期的正常现象，建议：")
        print("          1. 增加训练 epoch（建议 20-50 轮）")
        print("          2. 或增大学习率（当前可能过小）")
        print("          3. 检查标签是否为 0/1 格式（而非 0/255）\n")
    
    return metrics


class MetricsLogger:
    """优化版指标记录器，支持实时 CSV 写入、前景类指标、自定义文件名"""
    
    def __init__(self, log_dir, log_name, model_info):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_info = model_info
        
        # 确定文件名
        if log_name:
            self.csv_path = self.log_dir / f'{log_name}.csv'
        else:
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')
            model_type = model_info.get('model_type', 'model')
            self.csv_path = self.log_dir / f'{model_type}_training_log_{time_str}.csv'
        
        # 包含前景类指标的表头
        self.header = [
            'epoch',
            'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_miou',
            'train_fg_precision', 'train_fg_recall', 'train_fg_f1', 'train_fg_miou',
            'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_miou',
            'val_fg_precision', 'val_fg_recall', 'val_fg_f1', 'val_fg_miou',
            'inference_time_ms', 'fps', 'learning_rate'
        ]
        
        # 立即创建文件并写入表头（防止训练中断后无文件）
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
        print(f"[INFO] 标准日志文件创建: {self.csv_path}")
        
    def log_epoch(self, epoch, train_loss, train_metrics, val_loss, val_metrics, 
                  inference_time_ms, fps, lr):
        """记录一轮数据并立即写入 CSV"""
        row = {
            'epoch': epoch,
            'train_loss': f"{train_loss:.6f}",
            'train_precision': f"{train_metrics.get('precision', 0):.6f}",
            'train_recall': f"{train_metrics.get('recall', 0):.6f}",
            'train_f1': f"{train_metrics.get('f1', 0):.6f}",
            'train_miou': f"{train_metrics.get('miou', 0):.6f}",
            'train_fg_precision': f"{train_metrics.get('fg_precision', 0):.6f}",
            'train_fg_recall': f"{train_metrics.get('fg_recall', 0):.6f}",
            'train_fg_f1': f"{train_metrics.get('fg_f1', 0):.6f}",
            'train_fg_miou': f"{train_metrics.get('fg_miou', 0):.6f}",
            'val_loss': f"{val_loss:.6f}",
            'val_precision': f"{val_metrics.get('precision', 0):.6f}",
            'val_recall': f"{val_metrics.get('recall', 0):.6f}",
            'val_f1': f"{val_metrics.get('f1', 0):.6f}",
            'val_miou': f"{val_metrics.get('miou', 0):.6f}",
            'val_fg_precision': f"{val_metrics.get('fg_precision', 0):.6f}",
            'val_fg_recall': f"{val_metrics.get('fg_recall', 0):.6f}",
            'val_fg_f1': f"{val_metrics.get('fg_f1', 0):.6f}",
            'val_fg_miou': f"{val_metrics.get('fg_miou', 0):.6f}",
            'inference_time_ms': f"{inference_time_ms:.4f}",
            'fps': f"{fps:.2f}",
            'learning_rate': f"{lr:.8f}",
        }
        
        # 立即追加写入（实时保存，防止崩溃丢失）
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writerow(row)
        
    def save_model_info(self):
        """保存模型信息到日志目录"""
        info_path = self.log_dir / f"{self.csv_path.stem}_model_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"Model Type: {self.model_info.get('model_type', 'Unknown')}\n")
            f.write(f"Backbone: {self.model_info.get('backbone', 'Unknown')}\n")
            f.write(f"Total Parameters: {self.model_info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {self.model_info['trainable_params']:,}\n")
            f.write(f"Model Size: {self.model_info['model_size_mb']:.2f} MB\n")
            f.write(f"Number of Classes: {self.model_info.get('num_classes', 'Unknown')}\n")
            f.write(f"Input Size: {self.model_info.get('img_size', 'Unknown')}\n")
            f.write(f"Loss Type: {self.model_info.get('loss_type', 'Unknown')}\n")
            f.write(f"Optimizer: {self.model_info.get('optimizer_type', 'Unknown')}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Water Segmentation Training (Optimized)")
    
    # 数据路径参数
    parser.add_argument('--TRAIN_IMAGE_DIR', type=str, required=True, help='训练集图像目录')
    parser.add_argument('--TRAIN_MASK_DIR', type=str, required=True, help='训练集标签目录')
    parser.add_argument('--VAL_IMAGE_DIR', type=str, required=True, help='验证集图像目录')
    parser.add_argument('--VAL_MASK_DIR', type=str, required=True, help='验证集标签目录')
    
    # 模型与训练参数
    parser.add_argument('--IMG_SIZE', type=int, default=256, help='输入图像尺寸')
    parser.add_argument('--CUDA', type=bool, default=True)
    parser.add_argument('--BANDS', type=int, default=3)
    parser.add_argument('--NUM_CLASS', type=int, default=2)
    parser.add_argument('--GPU_ID', type=int, default=0)
    
    # 优化器与学习率
    parser.add_argument('--INIT_LR', '--lr', '--learning-rate', dest='INIT_LR', type=float, default=1e-3,
                        help='初始学习率（支持 --INIT_LR, --lr 或 --learning-rate）')
    parser.add_argument('--LR_STEP', type=int, default=1)
    parser.add_argument('--STEP_RATIO', type=float, default=0.94)
    parser.add_argument('--MOMENTUM', type=float, default=0.9)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-4)
    parser.add_argument('--BATCH_SIZE', type=int, default=2)
    parser.add_argument('--START_EPOCH', type=int, default=1)
    parser.add_argument('--EPOCHS', type=int, default=1)
    
    # 模型配置
    parser.add_argument('--PRETRAIN_MODEL', type=str, default=None)
    parser.add_argument('--LOSS_TYPE', type=str, default='ce', choices=['ce', 'focal'])
    parser.add_argument('--OPTIMIZER_TYPE', type=str, default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--LR_SCHEDULER', type=str, default='poly', choices=['poly', 'step', 'cos', 'exp'])
    parser.add_argument('--MODEL_TYPE', type=str, default='upernet',
                        choices=['unet', 'deeplab', 'enet', 'pspnet', 'hrnet', 'segnet', 
                                 'refinenet', 'fcn', 'segformer', 'setr', 'upernet', 
                                 'ocrnet', 'mask2former', 'segnext'])
    parser.add_argument('--BACKBONE_TYPE', type=str, default=None)
    parser.add_argument('--ATTENTION_TYPE', type=str, default=None, choices=['senet', 'ecanet', 'cbam', 'vit', 'self_atten'])
    parser.add_argument('--INIT_TYPE', type=str, default='kaiming', choices=['kaiming', 'normal', 'xavier', 'orthogonal'])
    
    # 新增：输出路径与文件名控制
    parser.add_argument('--model-dir', type=str, default='checkpoints',
                        help='模型保存目录（默认: checkpoints）')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='训练日志保存目录（默认: logs）')
    parser.add_argument('--model-name', type=str, default=None,
                        help='模型保存文件名前缀，例如 upernet_exp1（默认使用 MODEL_TYPE）')
    parser.add_argument('--log-name', type=str, default=None,
                        help='日志文件名前缀，例如 upernet_exp1（默认自动生成）')
    parser.add_argument('--save-interval', type=int, default=0,
                        help='分步保存模型的 epoch 间隔，0 表示不保存中间模型（默认: 0）')
    
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, train_loader, val_loader):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = Evaluator(self.args.NUM_CLASS)
        self.best_miou = 0.0
        self.last_detailed_metrics = []

    def get_last_detailed_metrics(self):
        """获取上一轮验证的详细指标（用于兼容原有详细 CSV）"""
        return self.last_detailed_metrics

    def training(self, epoch):
        self.model.train()
        train_loss = 0.0
        train_loader = tqdm(self.train_loader)
        num_batch = len(self.train_loader)
        
        # 计算训练集指标（新增）
        train_evaluator = Evaluator(self.args.NUM_CLASS)

        for i, data in enumerate(train_loader):
            img, lbl = data

            if self.args.CUDA:
                img = img.cuda(self.args.GPU_ID, non_blocking=True).float()
                lbl = lbl.cuda(self.args.GPU_ID, non_blocking=True).long()

            output = self.model(img).float()
            loss = self.criterion(output, lbl)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_loader.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            
            # 收集训练预测
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            label = lbl.cpu().numpy()
            train_evaluator.add_batch(label, pred)

        print('Epoch: %d, numImages: %5d' % (epoch+1, num_batch * self.args.BATCH_SIZE))
        print('Train Loss: %.3f' % (train_loss / num_batch))
        
        # 计算训练指标（包含前景类）
        train_metrics = compute_metrics_from_evaluator(train_evaluator, self.args.NUM_CLASS)
        
        return train_loss / num_batch, train_metrics

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        val_loss = 0.0
        val_loader = tqdm(self.val_loader)
        num_batch = len(self.val_loader)
        
        # 测量推理时间
        inference_times = []

        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                image, label = sample
                if torch.cuda.is_available():
                    image = image.cuda(self.args.GPU_ID, non_blocking=True).float()
                    label = label.cuda(self.args.GPU_ID, non_blocking=True).long()
                
                # 测量推理时间
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                
                output = self.model(image).float()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append(time.time() - start)
                
                loss = self.criterion(output, label)
                val_loss = val_loss + loss.item()
                val_loader.set_description('Val loss: %.3f' % (val_loss / (i + 1)))
                
                pred = output.data.cpu().numpy()
                label = label.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                self.evaluator.add_batch(label, pred)

        # 计算指标
        Acc = self.evaluator.OverAll_Accuracy()
        Kappa = self.evaluator.Kappa()
        mIoU, IoU = self.evaluator.mean_Intersection_over_Union()
        mIoU0, mIoU1 = IoU
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        mPrecision, Precision = self.evaluator.Precision()
        Precision0, Precision1 = Precision
        mRecall, Recall = self.evaluator.Recall()
        Recall0, Recall1 = Recall
        mF1_score, F1_score = self.evaluator.F1_Score()
        F1_score0, F1_score1 = F1_score
        mF2_score, F2_score = self.evaluator.F2_Score()
        F2_score0, F2_score1 = F2_score

        # 保存详细指标用于 CSV 记录（兼容原有详细日志）
        self.last_detailed_metrics = [
            Acc, Kappa, mIoU, mIoU0, mIoU1, FWIoU,
            mPrecision, Precision0, Precision1,
            mRecall, Recall0, Recall1,
            mF1_score, F1_score0, F1_score1,
            mF2_score, F2_score0, F2_score1
        ]

        # 计算推理时间
        inference_time_ms = np.mean(inference_times) * 1000
        fps = self.args.BATCH_SIZE / np.mean(inference_times) if np.mean(inference_times) > 0 else 0

        # 提取核心指标（包含前景类）
        val_metrics = {
            'acc': Acc,
            'kappa': Kappa,
            'miou': mIoU,
            'precision': mPrecision,
            'recall': mRecall,
            'f1': mF1_score,
            'fg_precision': Precision1,  # 前景类（假设为类别1）
            'fg_recall': Recall1,
            'fg_f1': F1_score1,
            'fg_miou': mIoU1,
        }

        # 更新最佳 mIoU
        if mIoU > self.best_miou:
            self.best_miou = mIoU
            print(f"[INFO] New best mIoU: {mIoU:.4f}")

        print('Validation Result:')
        print('Epoch:%d, numImages: %5d' % (epoch+1, num_batch * self.args.BATCH_SIZE))
        print("Epoch:{}, Acc:{:.4f}, Kappa:{:.4f}, mIoU:{:.4f}, FWIoU: {:.4f},\n"
              "Precision: {:.4f}, Recall: {:.4f}, f1_score: {:.4f}, f2_score: {:.4f}."
              .format(epoch+1, Acc, Kappa, mIoU, FWIoU, mPrecision, mRecall, mF1_score, mF2_score))
        print("Foreground -> Precision: {:.4f}, Recall: {:.4f}, f1: {:.4f}, mIoU: {:.4f}".format(
            Precision1, Recall1, F1_score1, mIoU1))
        print('Val Loss: %.3f' % (val_loss / num_batch))
        print('Inference Time: %.4f ms, FPS: %.2f' % (inference_time_ms, fps))

        return val_loss/num_batch, val_metrics, inference_time_ms, fps


def main():
    args = parse_args()

    # 创建保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu')

    # 网络选择
    if args.MODEL_TYPE == 'unet':
        model = Unet(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=args.BACKBONE_TYPE, atten_type=args.ATTENTION_TYPE)
    elif args.MODEL_TYPE == 'deeplab':
        # 注意：原代码中 DeepLab 被注释掉了，如果需要使用需取消注释 nets.deeplabv3_plus 导入
        raise NotImplementedError('DeepLab model needs to be uncommented in imports')
    elif args.MODEL_TYPE == 'fcn':
        model = FCN16s(bands=args.BANDS, num_classes=args.NUM_CLASS)
    elif args.MODEL_TYPE == 'hrnet':
        model = hrnet(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=18, version='v2')
    elif args.MODEL_TYPE == 'pspnet':
        model = PSPNet(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=args.BACKBONE_TYPE, downsample_factor=8)
    elif args.MODEL_TYPE == 'refinenet':
        model = rf50(bands=args.BANDS, num_classes=args.NUM_CLASS)
    elif args.MODEL_TYPE == 'enet':
        model = ENet(bands=args.BANDS, num_classes=args.NUM_CLASS)
    elif args.MODEL_TYPE == 'segformer':
        model = Segformer(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone='b0')
    elif args.MODEL_TYPE == 'setr':
        model = SETR(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone='Base', img_size=args.IMG_SIZE)
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
        raise NotImplementedError(f'model type [{args.MODEL_TYPE}] is not implemented')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory Cleared: Allocated {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

    # 加载预训练模型
    if args.PRETRAIN_MODEL:
        if os.path.isfile(args.PRETRAIN_MODEL):
            print(f"=> Loading pretrained model from {args.PRETRAIN_MODEL}")
            checkpoint = torch.load(args.PRETRAIN_MODEL, map_location='cpu')
            pretrained_dict = checkpoint
            if 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
            model.load_state_dict(pretrained_dict, strict=False)
            print("=> Loaded pretrained weights (strict=False)")
    else:
        print("=> No pretrained model specified, using fresh initialization")
        weights_init(model, init_type=args.INIT_TYPE)

    if args.CUDA:
        torch.cuda.set_device(args.GPU_ID)
        model = model.cuda(args.GPU_ID)
        print(f"Information:\n"
              f"| model: {args.MODEL_TYPE}\n"
              f"| backbone: {args.BACKBONE_TYPE}\n"
              f"| optimizer: {args.OPTIMIZER_TYPE}\n"
              f"| batch size: {args.BATCH_SIZE}\n"
              f"| loss type: {args.LOSS_TYPE}\n"
              f"| init lr: {args.INIT_LR}\n"
              f"| lr scheduler: {args.LR_SCHEDULER}\n"
              f"| weight decay: {args.WEIGHT_DECAY}\n"
              f"| training epochs: {args.EPOCHS}\n"
              f"| init type: {args.INIT_TYPE}\n")
        print("Training on GPU: {}".format(args.GPU_ID))

    # 获取模型信息
    model_info = get_model_info(model)
    model_info.update({
        'model_type': args.MODEL_TYPE,
        'backbone': args.BACKBONE_TYPE,
        'num_classes': args.NUM_CLASS,
        'img_size': args.IMG_SIZE,
        'loss_type': args.LOSS_TYPE,
        'optimizer_type': args.OPTIMIZER_TYPE,
    })
    print(f"Model: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")

    # 类别权重
    if args.NUM_CLASS == 2:
        weight = np.array([1.0, 2.0], np.float32)
    else:
        weight = np.array([4.204673196, 48.29108289, 11.4838323], np.float32)
    weight = torch.from_numpy(weight.astype(np.float32)).cuda()

    # 损失函数
    if args.LOSS_TYPE == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-1, reduction='mean')
    elif args.LOSS_TYPE == 'focal':
        criterion = FocalLoss(alpha=weight, gamma=2.0, ignore_index=-1, reduction='mean')
    else:
        raise NotImplementedError(f'loss type [{args.LOSS_TYPE}] is not implemented')

    # 优化器
    if args.OPTIMIZER_TYPE == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.INIT_LR, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
    elif args.OPTIMIZER_TYPE == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.INIT_LR, weight_decay=args.WEIGHT_DECAY)
    else:
        raise NotImplementedError(f'optimizer type [{args.OPTIMIZER_TYPE}] is not implemented')

    # 学习率衰减
    if args.LR_SCHEDULER == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.LR_STEP, gamma=args.STEP_RATIO)
    elif args.LR_SCHEDULER == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.STEP_RATIO)
    elif args.LR_SCHEDULER == 'cos':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.EPOCHS)
    elif args.LR_SCHEDULER == 'poly':
        # PolynomialLR 在较新 PyTorch 版本中可用，如果不支持可使用 LambdaLR
        try:
            lr_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.EPOCHS, power=2)
        except AttributeError:
            # 降级方案
            lambda_lr = lambda epoch: (1 - epoch/args.EPOCHS)**2
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    else:
        raise NotImplementedError(f'lr scheduler type [{args.LR_SCHEDULER}] is not implemented')

    # 加载数据集
    print(f"Loading training data from: {args.TRAIN_IMAGE_DIR}")
    print(f"Loading validation data from: {args.VAL_IMAGE_DIR}")
    
    train_datasets = Labeled_Model_Dataset(args.TRAIN_IMAGE_DIR, args.TRAIN_MASK_DIR, bands=args.BANDS, img_size=args.IMG_SIZE)
    test_datasets = Labeled_Model_Dataset(args.VAL_IMAGE_DIR, args.VAL_MASK_DIR, bands=args.BANDS, img_size=args.IMG_SIZE)
    train_loader = DataLoader(train_datasets, shuffle=True, batch_size=args.BATCH_SIZE, num_workers=0, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_datasets, shuffle=False, batch_size=args.BATCH_SIZE, num_workers=0, pin_memory=True, drop_last=False)

    cudnn.benchmark = True

    # 创建指标记录器（优化版，实时写入）
    metrics_logger = MetricsLogger(args.log_dir, args.log_name, model_info)
    metrics_logger.save_model_info()

    # 保留原有的详细 CSV（用于详细分析）
    detailed_log_name = args.log_name if args.log_name else args.MODEL_TYPE
    detailed_log_path = os.path.join(args.log_dir, f'{detailed_log_name}_detailed.csv')
    with open(detailed_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'Acc', 'Kappa', 'mIoU', 
                        'mIoU0', 'mIoU1', 'FWIoU', 'Precision', 'Precision0', 'Precision1',
                        'Recall', 'Recall0', 'Recall1', 'F1_score', 'F1_score0', 'F1_score1',
                        'F2_score', 'F2_score0', 'F2_score1'])

    trainer = Trainer(args, model, criterion, optimizer, train_loader, test_loader)

    print('Starting Epoch:', trainer.args.START_EPOCH)
    print('Total Epoches:', trainer.args.EPOCHS)
    
    # 确定模型文件名前缀
    model_prefix = args.model_name if args.model_name else args.MODEL_TYPE

    for epoch in range(trainer.args.EPOCHS):
        print("Start training on GPU:{}...".format(args.GPU_ID))
        train_loss, train_metrics = trainer.training(epoch)
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        print("Current learning rate is:", current_lr)
        print("Training over.\n")

        print(f"Start validating on GPU:{args.GPU_ID}...")
        val_loss, val_metrics, inference_time_ms, fps = trainer.validation(epoch)
        print("Validating over.\n")

        # 实时记录标准格式 CSV（新增前景类指标）
        metrics_logger.log_epoch(
            epoch + 1, 
            train_loss, 
            train_metrics,
            val_loss, 
            val_metrics,
            inference_time_ms,
            fps,
            current_lr
        )

        # 记录到详细 CSV（追加）
        with open(detailed_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss] + trainer.get_last_detailed_metrics())

        # 保存最佳模型
        if val_metrics['miou'] > trainer.best_miou:
            trainer.best_miou = val_metrics['miou']
            best_path = os.path.join(args.model_dir, f'{model_prefix}_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'miou': trainer.best_miou,
                'optimizer': optimizer.state_dict(),
            }, best_path)
            print(f"[BEST] 模型已保存: {best_path}, mIoU: {trainer.best_miou:.4f}")

        # 分步保存中间模型（如果设置间隔 > 0）
        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            periodic_path = os.path.join(args.model_dir, f'{model_prefix}_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'miou': val_metrics['miou'],
                'optimizer': optimizer.state_dict(),
            }, periodic_path)
            print(f"[CHECKPOINT] 中间模型已保存: {periodic_path}")

    # 训练结束，保存最终模型（last）
    last_path = os.path.join(args.model_dir, f'{model_prefix}_last.pth')
    torch.save({
        'epoch': args.EPOCHS,
        'state_dict': model.state_dict(),
        'miou': val_metrics['miou'],
        'optimizer': optimizer.state_dict(),
    }, last_path)
    print(f"[LAST] 最终模型已保存: {last_path}")

    print(f"\n{'='*50}")
    print(f"训练完成!")
    print(f"最佳验证 mIoU: {trainer.best_miou:.4f}")
    print(f"最终验证 FG-Recall: {val_metrics['fg_recall']:.4f}")
    print(f"最佳模型: {args.model_dir}/{model_prefix}_best.pth")
    print(f"最终模型: {args.model_dir}/{model_prefix}_last.pth")
    print(f"标准日志: {metrics_logger.csv_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()