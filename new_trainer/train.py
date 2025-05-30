#!/usr/bin/env python3
"""
Multi-GPU PyTorch Training Script for Chess Piece Recognition
Converted from TensorFlow, optimized for NVIDIA A40 GPUs on RunPod

Usage:
# Single GPU
python train_pytorch.py --data_dir ./data2/train --output_model chess_model.pth

# Multi-GPU (automatically detects available GPUs)  
python train_pytorch.py --data_dir ./data2/train --output_model chess_model.pth --multi_gpu

# Continue training
python train_pytorch.py --data_dir ./data2/train --output_model chess_model_v2.pth --resume_from chess_model.pth

# Match your original TensorFlow settings
python train_pytorch.py --data_dir ./data2/train --input_size 224 --batch_size 128 --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import timm

from PIL import Image
import os
import json
import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from datetime import datetime
import random
from collections import defaultdict

# A40-specific optimizations
def setup_a40_optimizations():
    """Configure PyTorch for optimal A40 performance"""
    # Enable Tensor Core optimizations for Ampere architecture
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Optimize memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'
    
    print("✅ A40 optimizations enabled")

class ChessPieceDataset(Dataset):
    """Optimized Dataset for chess piece images with heavy augmentation"""
    
    def __init__(self, data_dir, transform=None, is_training=True, class_mapping=None):
        self.data_dir = data_dir
        self.transform = transform
        self.is_training = is_training
        
        # Get all class directories
        self.class_dirs = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))]
        self.class_dirs.sort()
        
        # Create class mapping
        if class_mapping is None:
            self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_dirs)}
        else:
            self.class_to_idx = class_mapping
            
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # Load all image paths and labels
        self.samples = []
        self.class_counts = defaultdict(int)
        
        print(f"Loading dataset from {data_dir}...")
        for class_name in self.class_dirs:
            if class_name not in self.class_to_idx:
                continue
                
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob(os.path.join(class_dir, ext)))
            
            if not image_paths:
                print(f"⚠️  No images found in {class_dir}")
                continue
                
            for img_path in image_paths:
                self.samples.append((img_path, class_idx))
                self.class_counts[class_name] += 1
        
        print(f"Dataset loaded: {len(self.samples)} images across {self.num_classes} classes")
        for class_name, count in self.class_counts.items():
            print(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (244, 244), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(input_size=224, is_training=True):
    """Get transforms optimized for chess piece recognition"""
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            # Random erasing to simulate occlusions
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

class ChessPieceModel(nn.Module):
    """Chess piece recognition model using EfficientNet-B3"""
    
    def __init__(self, num_classes=14, pretrained=True):
        super(ChessPieceModel, self).__init__()
        
        # Use EfficientNet-B3 as backbone (better than MobileNetV2)
        if pretrained:
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b3(weights=weights)
        else:
            self.backbone = efficientnet_b3(weights=None)
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.classifier[1].in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)

def create_data_loaders(data_dir, batch_size=256, input_size=224, num_workers=8, 
                       validation_split=0.2, distributed=False):
    """Create optimized data loaders for A40 training"""
    
    # Create dataset
    full_transform = get_transforms(input_size, is_training=True)
    val_transform = get_transforms(input_size, is_training=False)
    
    full_dataset = ChessPieceDataset(data_dir, transform=full_transform)
    
    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update val dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else 2,
        drop_last=True  # Important for distributed training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    return train_loader, val_loader, full_dataset.class_to_idx

def setup_distributed(rank, world_size, port=12355):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, loss, 
                   filepath, class_mapping, is_distributed=False):
    """Save training checkpoint"""
    
    # Get the actual model (unwrap DDP if needed)
    model_state = model.module.state_dict() if is_distributed else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_accuracy': best_acc,
        'loss': loss,
        'class_mapping': class_mapping,
        'model_config': {
            'num_classes': len(class_mapping),
            'architecture': 'efficientnet_b3'
        }
    }
    
    torch.save(checkpoint, filepath)
    
    # Also save metadata
    metadata = {
        'class_mapping': class_mapping,
        'num_classes': len(class_mapping),
        'last_trained': datetime.now().isoformat(),
        'architecture': 'efficientnet_b3',
        'best_accuracy': best_acc
    }
    
    with open(filepath + '.metadata', 'w') as f:
        json.dump(metadata, f, indent=2)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return (checkpoint['epoch'], checkpoint.get('best_accuracy', 0.0), 
            checkpoint.get('class_mapping', {}))

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
               scaler=None, distributed=False, rank=0):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Progress bar only on main process
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Mixed precision forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += images.size(0)
        
        # Update progress bar on main process
        if rank == 0 and batch_idx % 10 == 0:
            current_acc = running_corrects.double() / total_samples
            pbar.set_postfix({
                'Loss': f'{running_loss/total_samples:.4f}',
                'Acc': f'{current_acc:.4f}'
            })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, val_loader, criterion, device, rank=0):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(val_loader, desc="Validation")
        else:
            pbar = val_loader
            
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def main_worker(rank, world_size, args):
    """Main training worker for distributed training"""
    
    # Setup distributed training
    is_distributed = world_size > 1
    if is_distributed:
        setup_distributed(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Setup A40 optimizations
    setup_a40_optimizations()
    
    # Initialize wandb only on main process
    if rank == 0 and args.use_wandb:
        wandb.init(
            project="chess-piece-recognition",
            name=f"efficientnet_b3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # Create data loaders
    train_loader, val_loader, class_mapping = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
        distributed=is_distributed
    )
    
    # Print class mapping on main process
    if rank == 0:
        print(f"\nClass mapping ({len(class_mapping)} classes):")
        for class_name, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
            print(f"  {idx}: {class_name}")
    
    # Create model
    model = ChessPieceModel(num_classes=len(class_mapping), pretrained=True)
    model = model.to(device)
    
    # Wrap model for distributed training
    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume_from and os.path.exists(args.resume_from):
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume_from}")
        start_epoch, best_acc, loaded_class_mapping = load_checkpoint(
            args.resume_from, model, optimizer, scheduler
        )
        
        # Verify class mapping consistency
        if loaded_class_mapping != class_mapping:
            if rank == 0:
                print("⚠️  Warning: Class mapping has changed!")
                print("Current classes:", sorted(class_mapping.items()))
                print("Loaded classes:", sorted(loaded_class_mapping.items()))
    
    # Training loop with optional two-phase training (matching TF script)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    if args.two_phase_training and args.resume_from is None:
        # Phase 1: Frozen backbone training (matching TF script)
        if rank == 0:
            print("🔒 Phase 1: Training with frozen backbone...")
            
        # Ensure backbone is frozen
        if hasattr(model, 'module'):
            backbone = model.module.backbone
        else:
            backbone = model.backbone
            
        # Freeze backbone except classifier
        for name, param in backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        
        # Phase 1 training
        phase1_epochs = min(args.unfreeze_epochs, args.epochs)
        for epoch in range(phase1_epochs):
            if is_distributed:
                train_loader.sampler.set_epoch(epoch)
            
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch + 1,
                scaler, is_distributed, rank
            )
            
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, rank)
            scheduler.step()
            
            if rank == 0:
                print(f"Phase 1 - Epoch {epoch + 1}/{phase1_epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                
                if args.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'phase': 1,
                        'train_loss': train_loss,
                        'train_accuracy': train_acc,
                        'val_loss': val_loss,
                        'val_accuracy': val_acc,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(
                        model, optimizer, scheduler, epoch + 1, best_acc, val_loss,
                        args.output_model, class_mapping, is_distributed
                    )
                    print(f"✅ New best model saved! Accuracy: {best_acc:.4f}")
        
        # Phase 2: Unfreeze backbone for fine-tuning
        if phase1_epochs < args.epochs:
            if rank == 0:
                print("🔓 Phase 2: Fine-tuning with unfrozen backbone...")
            
            # Unfreeze backbone layers
            for param in backbone.parameters():
                param.requires_grad = True
            
            # Lower learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate * 0.1
            
            # Continue training
            for epoch in range(phase1_epochs, args.epochs):
                if is_distributed:
                    train_loader.sampler.set_epoch(epoch)
                
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device, epoch + 1,
                    scaler, is_distributed, rank
                )
                
                val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, rank)
                scheduler.step()
                
                if rank == 0:
                    print(f"Phase 2 - Epoch {epoch + 1}/{args.epochs}")
                    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    
                    train_losses.append(train_loss)
                    train_accuracies.append(train_acc)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)
                    
                    if args.use_wandb:
                        wandb.log({
                            'epoch': epoch + 1,
                            'phase': 2,
                            'train_loss': train_loss,
                            'train_accuracy': train_acc,
                            'val_loss': val_loss,
                            'val_accuracy': val_acc,
                            'learning_rate': optimizer.param_groups[0]['lr']
                        })
                    
                    if val_acc > best_acc:
                        best_acc = val_acc
                        save_checkpoint(
                            model, optimizer, scheduler, epoch + 1, best_acc, val_loss,
                            args.output_model, class_mapping, is_distributed
                        )
                        print(f"✅ New best model saved! Accuracy: {best_acc:.4f}")
    
    else:
        # Single-phase training (when resuming or two-phase disabled)
        for epoch in range(start_epoch, args.epochs):
            # Set epoch for distributed sampler
            if is_distributed:
                train_loader.sampler.set_epoch(epoch)
            
            # Training phase
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch + 1,
                scaler, is_distributed, rank
            )
            
            # Validation phase
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, rank)
            
            # Step scheduler
            scheduler.step()
            
            # Logging and checkpointing (main process only)
            if rank == 0:
                print(f"\nEpoch {epoch + 1}/{args.epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
                
                # Save metrics
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                
                # Wandb logging
                if args.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'train_accuracy': train_acc,
                        'val_loss': val_loss,
                        'val_accuracy': val_acc,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(
                        model, optimizer, scheduler, epoch + 1, best_acc, val_loss,
                        args.output_model, class_mapping, is_distributed
                    )
                    print(f"✅ New best model saved! Accuracy: {best_acc:.4f}")
                
                # Save periodic checkpoint
                if (epoch + 1) % 10 == 0:
                    checkpoint_path = args.output_model.replace('.pth', f'_epoch_{epoch + 1}.pth')
                    save_checkpoint(
                        model, optimizer, scheduler, epoch + 1, best_acc, val_loss,
                        checkpoint_path, class_mapping, is_distributed
                    )
    
    # Save final results
    if rank == 0:
        print(f"\nTraining completed! Best validation accuracy: {best_acc:.4f}")
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_accuracies, label='Train')
        plt.plot(val_accuracies, label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if args.use_wandb:
            wandb.log({"training_history": wandb.Image("training_history.png")})
            wandb.finish()
    
    # Cleanup distributed training
    if is_distributed:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='Train chess piece recognition model with PyTorch')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--output_model', type=str, default='chess_model.pth',
                       help='Output model path')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint')
    
    # Model parameters
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size per GPU (will be scaled for multi-GPU, default matches A40 capacity)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (matching original TF script)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate (matching original TF script)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers per GPU')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training (default: enabled for A40)')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Use multi-GPU training if available')
    
    # Advanced training options (matching TensorFlow script features)
    parser.add_argument('--two_phase_training', action='store_true', default=True,
                       help='Use two-phase training (frozen then unfrozen backbone)')
    parser.add_argument('--unfreeze_epochs', type=int, default=20,
                       help='Epochs of frozen training before unfreezing')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    # Detect available GPUs
    world_size = torch.cuda.device_count()
    print(f"🎮 Detected {world_size} GPU(s)")
    
    for i in range(world_size):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")
    
    # Adjust batch size for multiple GPUs
    if args.multi_gpu and world_size > 1:
        print(f"🚀 Using multi-GPU training with {world_size} GPUs")
        # Scale batch size for multiple GPUs
        effective_batch_size = args.batch_size * world_size
        print(f"📊 Effective batch size: {effective_batch_size}")
        
        # Launch distributed training
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("🎯 Using single GPU training")
        main_worker(0, 1, args)

if __name__ == '__main__':
    main()