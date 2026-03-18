# encoding = utf-8

import os
import csv
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data

from nets.unet import Unet
# from nets.deeplabv3_plus import DeepLab
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


parser = argparse.ArgumentParser(description="Water Segmentation Training")
parser.add_argument('--TRAIN_IMAGE_DIR', type=str, required=True, help='训练集图像目录')
parser.add_argument('--TRAIN_MASK_DIR', type=str, required=True, help='训练集标签目录')
parser.add_argument('--VAL_IMAGE_DIR', type=str, required=True, help='验证集图像目录')
parser.add_argument('--VAL_MASK_DIR', type=str, required=True, help='验证集标签目录')
parser.add_argument('--IMG_SIZE', type=int, default=256, help='输入图像尺寸')
parser.add_argument('--CUDA', type=bool, default=True)
parser.add_argument('--BANDS', type=int, default=3)
parser.add_argument('--NUM_CLASS', type=int, default=2)
parser.add_argument('--GPU_ID', type=int, default=0)
parser.add_argument('--LR_STEP', type=int, default=1)
parser.add_argument('--STEP_RATIO', type=float, default=0.94)
parser.add_argument('--INIT_LR', type=float, default=1e-3)
parser.add_argument('--MOMENTUM', type=float, default=0.9)
parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-4)
parser.add_argument('--BATCH_SIZE', type=int, default=2)
parser.add_argument('--START_EPOCH', type=int, default=1)
parser.add_argument('--EPOCHS', type=int, default=1)
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
    """从 Evaluator 提取核心指标"""
    Acc = evaluator.OverAll_Accuracy()
    Kappa = evaluator.Kappa()
    mIoU, IoU = evaluator.mean_Intersection_over_Union()
    mPrecision, Precision = evaluator.Precision()
    mRecall, Recall = evaluator.Recall()
    mF1_score, F1_score = evaluator.F1_Score()
    
    return {
        'acc': Acc,
        'kappa': Kappa,
        'miou': mIoU,
        'precision': mPrecision,
        'recall': mRecall,
        'f1': mF1_score,
    }


class MetricsLogger:
    """简化版指标记录器，生成标准格式CSV"""
    
    def __init__(self, save_path, model_info):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_info = model_info
        
        self.header = [
            'epoch',
            'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_miou',
            'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_miou',
            'inference_time_ms', 'fps', 'learning_rate'
        ]
        self.rows = []
        
    def log_epoch(self, epoch, train_loss, train_metrics, val_loss, val_metrics, 
                  inference_time_ms, fps, lr):
        """记录一轮数据"""
        row = {
            'epoch': epoch,
            'train_loss': f"{train_loss:.6f}",
            'train_precision': f"{train_metrics.get('precision', 0):.6f}",
            'train_recall': f"{train_metrics.get('recall', 0):.6f}",
            'train_f1': f"{train_metrics.get('f1', 0):.6f}",
            'train_miou': f"{train_metrics.get('miou', 0):.6f}",
            'val_loss': f"{val_loss:.6f}",
            'val_precision': f"{val_metrics.get('precision', 0):.6f}",
            'val_recall': f"{val_metrics.get('recall', 0):.6f}",
            'val_f1': f"{val_metrics.get('f1', 0):.6f}",
            'val_miou': f"{val_metrics.get('miou', 0):.6f}",
            'inference_time_ms': f"{inference_time_ms:.4f}",
            'fps': f"{fps:.2f}",
            'learning_rate': f"{lr:.8f}",
        }
        self.rows.append(row)
        
    def save(self):
        """保存CSV"""
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"Standard training log saved to {self.save_path}")
        
    def save_model_info(self):
        """保存模型信息"""
        info_path = self.save_path.parent / f"{self.save_path.stem}_model_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Model Type: {self.model_info.get('model_type', 'Unknown')}\n")
            f.write(f"Backbone: {self.model_info.get('backbone', 'Unknown')}\n")
            f.write(f"Total Parameters: {self.model_info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {self.model_info['trainable_params']:,}\n")
            f.write(f"Model Size: {self.model_info['model_size_mb']:.2f} MB\n")
            f.write(f"Number of Classes: {self.model_info.get('num_classes', 'Unknown')}\n")
            f.write(f"Input Size: {self.model_info.get('img_size', 'Unknown')}\n")


def main():
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu')

    # 网络选择
    if args.MODEL_TYPE == 'unet':
        model = Unet(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=args.BACKBONE_TYPE, atten_type=args.ATTENTION_TYPE)
    elif args.MODEL_TYPE == 'deeplab':
        model = DeepLab(bands=args.BANDS, num_classes=args.NUM_CLASS, backbone=args.BACKBONE_TYPE, atten_type=args.ATTENTION_TYPE)
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
        raise NotImplementedError(f'model type [{args.MODEL_TYPE}] is not implemented')

    os.makedirs('pth_files', exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory Cleared: Allocated {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

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
        print(f"=> No pretrained model found at '{args.PRETRAIN_MODEL}'")
        weights_init(model, init_type=args.INIT_TYPE)

    if args.CUDA:
        torch.cuda.set_device(args.GPU_ID)
        model = model.cuda(args.GPU_ID)
        print(f"Information:\n"
              f"|model:{args.MODEL_TYPE}\n"
              f"|backbone:{args.BACKBONE_TYPE}\n"
              f"|optimizer:{args.OPTIMIZER_TYPE}\n"
              f"|batchsize:{args.BATCH_SIZE}\n"
              f"|loss type:{args.LOSS_TYPE}\n"
              f"|init lr:{args.INIT_LR}\n"
              f"|lr scheduler:{args.LR_SCHEDULER}\n"
              f"|weight decay:{args.WEIGHT_DECAY}\n"
              f"|training epochs:{args.EPOCHS}\n"
              f"|init type:{args.INIT_TYPE}.\n")
        print("Training on GPU: {}".format(args.GPU_ID))

    # 获取模型信息
    model_info = get_model_info(model)
    model_info.update({
        'model_type': args.MODEL_TYPE,
        'backbone': args.BACKBONE_TYPE,
        'num_classes': args.NUM_CLASS,
        'img_size': args.IMG_SIZE,
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
        lr_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.EPOCHS, power=2)
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

    # 创建指标记录器（新增标准格式CSV）
    standard_log_path = f'{args.MODEL_TYPE}_training_log_standard.csv'
    metrics_logger = MetricsLogger(standard_log_path, model_info)
    metrics_logger.save_model_info()

    # 保留原有的详细CSV
    with open(f'{args.MODEL_TYPE}_training_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'Acc', 'Kappa', 'mIoU', 
                        'mIoU0', 'mIoU1', 'FWIoU', 'Precision', 'Precision0', 'Precision1',
                        'Recall', 'Recall0', 'Recall1', 'F1_score', 'F1_score0', 'F1_score1',
                        'F2_score', 'F2_score0', 'F2_score1'])

    trainer = Trainer(args, model, criterion, optimizer, train_loader, test_loader)

    print('Starting Epoch:', trainer.args.START_EPOCH)
    print('Total Epoches:', trainer.args.EPOCHS)

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

        # 保存模型
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), 
                      'pth_files/%s-epoch%d-loss%.3f-val_loss%.3f.pth' % 
                      (args.MODEL_TYPE, (epoch+1), train_loss, val_loss))

        # 记录到原有详细CSV
        with open(f'{args.MODEL_TYPE}_training_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            # 从 trainer 获取详细指标
            writer.writerow([epoch+1, train_loss, val_loss] + trainer.get_last_detailed_metrics())

        # 记录到新增标准CSV
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

    # 保存标准CSV
    metrics_logger.save()
    print(f"Best Val mIoU: {trainer.best_miou:.4f}")


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
        """获取上一轮验证的详细指标"""
        return self.last_detailed_metrics

    def training(self, epoch):
        self.model.train()
        train_loss = 0.0
        train_loader = tqdm(self.train_loader)
        num_batch = len(self.train_loader)
        
        # 新增：计算训练集指标
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
        
        # 计算训练指标
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

        # 保存详细指标用于CSV记录
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

        # 提取核心指标
        val_metrics = {
            'acc': Acc,
            'kappa': Kappa,
            'miou': mIoU,
            'precision': mPrecision,
            'recall': mRecall,
            'f1': mF1_score,
        }

        # 更新最佳mIoU
        if mIoU > self.best_miou:
            self.best_miou = mIoU
            print(f"New best model! mIoU: {mIoU:.4f}")

        print('Validation Result:')
        print('Epoch:%d, numImages: %5d' % (epoch+1, num_batch * self.args.BATCH_SIZE))
        print("Epoch:{}, Acc:{:.4f}, Kappa:{:.4f}, mIoU:{:.4f}, FWIoU: {:.4f},\n"
              "Precision: {:.4f}, Recall: {:.4f}, f1_score: {:.4f}, f2_score: {:.4f}."
              .format(epoch+1, Acc, Kappa, mIoU, FWIoU, mPrecision, mRecall, mF1_score, mF2_score))
        print('Val Loss: %.3f' % (val_loss / num_batch))
        print('Inference Time: %.4f ms, FPS: %.2f' % (inference_time_ms, fps))

        return val_loss/num_batch, val_metrics, inference_time_ms, fps


if __name__ == '__main__':
    main()