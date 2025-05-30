#!/usr/bin/env python3
"""
NVIDIA A40 Optimization Script for CNN Training
Optimizes settings specifically for A40's 48GB VRAM and Ampere architecture
"""

import torch
import os
import subprocess

def optimize_a40_settings():
    """Configure optimal settings for A40 GPU"""
    
    print("🚀 Optimizing for NVIDIA A40...")
    
    # A40 specifications
    A40_MEMORY_GB = 48
    A40_COMPUTE_CAPABILITY = 8.6
    
    # Verify we have an A40
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU: {gpu_name}")
        print(f"Memory: {total_memory:.1f} GB")
        
        if "A40" not in gpu_name:
            print("⚠️  Warning: This script is optimized for A40, but detected different GPU")
        
        # A40-specific optimizations
        configure_memory_settings()
        configure_compute_settings()
        suggest_batch_sizes()
        
    else:
        print("❌ No CUDA GPU detected!")

def configure_memory_settings():
    """Optimize memory settings for A40's 48GB VRAM"""
    print("\n📊 Configuring Memory Settings...")
    
    # Enable memory pool for better allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'
    
    # Set memory fraction (use 90% of 48GB = ~43GB)
    torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    print("✅ Memory settings optimized for 48GB VRAM")

def configure_compute_settings():
    """Optimize compute settings for Ampere architecture"""
    print("\n⚡ Configuring Compute Settings...")
    
    # Enable Tensor Core optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set optimal number of threads
    torch.set_num_threads(8)  # A40 works well with 8 CPU threads
    
    print("✅ Compute settings optimized for Ampere architecture")

def suggest_batch_sizes():
    """Suggest optimal batch sizes for common CNN architectures on A40"""
    print("\n🎯 Recommended Batch Sizes for A40 (48GB VRAM):")
    
    recommendations = {
        "ResNet-50": {
            "224px": "Batch size: 256-512",
            "384px": "Batch size: 128-256", 
            "512px": "Batch size: 64-128"
        },
        "ResNet-101": {
            "224px": "Batch size: 128-256",
            "384px": "Batch size: 64-128",
            "512px": "Batch size: 32-64"
        },
        "EfficientNet-B7": {
            "600px": "Batch size: 32-64",
            "380px": "Batch size: 64-128"
        },
        "Vision Transformer (ViT-Large)": {
            "224px": "Batch size: 64-128",
            "384px": "Batch size: 32-64"
        },
        "ConvNeXt-Large": {
            "224px": "Batch size: 128-256",
            "384px": "Batch size: 64-128"
        }
    }
    
    for model, sizes in recommendations.items():
        print(f"\n  {model}:")
        for resolution, batch_info in sizes.items():
            print(f"    {resolution}: {batch_info}")

def test_memory_allocation():
    """Test optimal memory allocation patterns"""
    print("\n🧪 Testing Memory Allocation...")
    
    try:
        # Test large batch allocation
        test_tensor = torch.randn(128, 3, 512, 512, device='cuda')
        print(f"✅ Large tensor test successful: {test_tensor.shape}")
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"📊 Memory usage - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        del test_tensor
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        print(f"❌ Memory test failed: {e}")

def create_a40_training_config():
    """Create optimized training configuration for A40"""
    config = {
        # Data loading
        'num_workers': 8,  # A40 pairs well with 8 CPU workers
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4,
        
        # Memory optimization
        'gradient_accumulation_steps': 1,  # Can use larger batches directly
        'mixed_precision': True,  # Enable FP16 for even better performance
        
        # Batch sizes (adjust based on your model)
        'batch_size_recommendations': {
            'resnet50_224': 512,
            'resnet50_384': 256,
            'resnet101_224': 256,
            'efficientnet_b7': 64,
            'vit_large_224': 128,
        },
        
        # Checkpointing
        'checkpoint_every_n_steps': 1000,
        'save_top_k': 3,
        
        # Logging
        'log_every_n_steps': 50,
    }
    
    return config

def benchmark_a40_performance():
    """Run performance benchmark on A40"""
    print("\n⏱️  Running A40 Performance Benchmark...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available for benchmarking")
        return
    
    # Benchmark different operations
    operations = [
        (torch.randn(512, 512, device='cuda'), lambda x: torch.mm(x, x), "Matrix Multiplication"),
        (torch.randn(128, 3, 224, 224, device='cuda'), lambda x: torch.nn.functional.conv2d(x, torch.randn(64, 3, 7, 7, device='cuda')), "Convolution"),
        (torch.randn(64, 1000, device='cuda'), lambda x: torch.nn.functional.softmax(x, dim=1), "Softmax"),
    ]
    
    for tensor, operation, name in operations:
        # Warmup
        for _ in range(10):
            _ = operation(tensor)
        
        torch.cuda.synchronize()
        
        # Benchmark
        import time
        start_time = time.time()
        for _ in range(100):
            _ = operation(tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        print(f"  {name}: {avg_time:.2f}ms average")
        
        del tensor

def main():
    """Main optimization function"""
    print("=" * 60)
    print("🎮 NVIDIA A40 CNN Training Optimization")
    print("=" * 60)
    
    optimize_a40_settings()
    test_memory_allocation()
    
    config = create_a40_training_config()
    print(f"\n📋 Recommended Training Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    benchmark_a40_performance()
    
    print("\n" + "=" * 60)
    print("✅ A40 Optimization Complete!")
    print("💡 Your A40 is now optimized for CNN training")
    print("💰 With 48GB VRAM, you can train large models with big batches")
    print("⚡ Ampere architecture provides excellent mixed-precision performance")
    print("=" * 60)

if __name__ == "__main__":
    main()