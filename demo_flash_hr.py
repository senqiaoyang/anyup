#!/usr/bin/env python3
"""
Flash Cross Attention 高分辨率图像效果测试

比较标准 AnyUp 和 Flash 版本在高分辨率图像上的:
1. 输出质量（PCA 可视化）
2. 运行速度
3. 显存占用

Usage:
    python demo_flash_hr.py
    python demo_flash_hr.py --resolution 2016  # 测试更高分辨率
    python demo_flash_hr.py --image path/to/your/image.jpg
"""

import io
import requests
import torch
import time
import gc
import argparse
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

from anyup.model import AnyUp


def load_image(image_path: str = None, target_height: int = None, target_width: int = None):
    """加载并预处理图像"""
    if image_path:
        img = Image.open(image_path).convert("RGB")
    else:
        print("Downloading sample image...")
        url = "https://cdn.pixabay.com/photo/2017/09/09/16/38/vegetables-2732589_1280.jpg"
        img = Image.open(io.BytesIO(requests.get(url, timeout=10).content)).convert("RGB")
    
    # 如果没有指定尺寸，使用接近原图尺寸并对齐到14的倍数
    if target_height is None or target_width is None:
        orig_w, orig_h = img.size
        target_height = (orig_h // 14) * 14
        target_width = (orig_w // 14) * 14
    
    return img, target_height, target_width


def extract_dinov2_features(img, target_height, target_width, device):
    """提取 DINOv2 特征"""
    print("Loading DINOv2 model...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True, token=False)
    model = Dinov2Model.from_pretrained("facebook/dinov2-base", token=False).to(device).eval()
    
    inputs = processor(
        images=img,
        do_resize=True,
        size={"height": target_height, "width": target_width},
        do_center_crop=False,
        return_tensors="pt",
    )
    hr_image = inputs["pixel_values"].to(device)
    
    print(f"Input image size: {target_width}x{target_height}")
    
    print("Extracting DINOv2 features...")
    with torch.no_grad():
        out = model(pixel_values=hr_image)
        tokens = out.last_hidden_state[:, 1:, :]  # drop [CLS]
    
    B, N, C = tokens.shape
    h = target_height // 14
    w = target_width // 14
    assert h * w == N, f"Token grid mismatch (N={N}, h={h}, w={w})"
    lr_features = tokens.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()
    print(f"LR features shape: {lr_features.shape}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return hr_image, lr_features


def create_anyup_models(device, flash_window_size=5):
    """创建标准版和 Flash 版 AnyUp 模型"""
    # 加载预训练权重
    print("Loading pretrained AnyUp weights...")
    pretrained = torch.hub.load("wimmerth/anyup", "anyup", verbose=False)
    state_dict = pretrained.state_dict()
    del pretrained
    gc.collect()
    torch.cuda.empty_cache()
    
    # 标准版本
    print("Creating standard AnyUp model...")
    model_standard = AnyUp(
        input_dim=3,
        qk_dim=128,
        kernel_size=1,
        kernel_size_lfu=5,
        window_ratio=0.1,
        num_heads=4,
        init_gaussian_derivatives=False,
        use_natten=False,
        use_flash=False,
    ).to(device).eval()
    model_standard.load_state_dict(state_dict, strict=True)
    
    # Flash 版本
    print(f"Creating Flash AnyUp model (window_size={flash_window_size})...")
    model_flash = AnyUp(
        input_dim=3,
        qk_dim=128,
        kernel_size=1,
        kernel_size_lfu=5,
        window_ratio=0.1,
        num_heads=4,
        init_gaussian_derivatives=False,
        use_natten=False,
        use_flash=True,
        flash_window_size=flash_window_size,
    ).to(device).eval()
    # Flash 版本加载相同的权重（应该兼容）
    model_flash.load_state_dict(state_dict, strict=False)
    
    return model_standard, model_flash


def benchmark_upsampling(model, hr_image, lr_features, name, q_chunk_size=None, warmup=2, repeats=5):
    """测试上采样性能"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            out = model(hr_image, lr_features, q_chunk_size=q_chunk_size)
    torch.cuda.synchronize()
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            out = model(hr_image, lr_features, q_chunk_size=q_chunk_size)
    torch.cuda.synchronize()
    
    elapsed = (time.perf_counter() - start) / repeats * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"  {name}: {elapsed:.1f} ms, {peak_mem:.2f} GB peak memory")
    
    return out, elapsed, peak_mem


def compute_pca_visualization(lr_features, hr_features_standard, hr_features_flash, target_height, target_width):
    """计算 Joint PCA 可视化"""
    print("Computing PCA visualization...")
    
    _, C, h, w = lr_features.shape
    
    with torch.no_grad():
        lr_flat = lr_features[0].permute(1, 2, 0).reshape(-1, C)
        hr_std_flat = hr_features_standard[0].permute(1, 2, 0).reshape(-1, C)
        hr_flash_flat = hr_features_flash[0].permute(1, 2, 0).reshape(-1, C)
        
        # Joint PCA on all features
        all_feats = torch.cat([lr_flat, hr_std_flat, hr_flash_flat], dim=0)
        mean = all_feats.mean(dim=0, keepdim=True)
        X = all_feats - mean
        
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        pcs = Vh[:3].T
        proj_all = X @ pcs
        
        # Split back
        n_lr = h * w
        n_hr = target_height * target_width
        proj_lr = proj_all[:n_lr].reshape(h, w, 3)
        proj_hr_std = proj_all[n_lr:n_lr + n_hr].reshape(target_height, target_width, 3)
        proj_hr_flash = proj_all[n_lr + n_hr:].reshape(target_height, target_width, 3)
        
        # Normalize jointly
        cmin = proj_all.min(dim=0).values
        cmax = proj_all.max(dim=0).values
        crng = (cmax - cmin).clamp(min=1e-6)
        
        lr_rgb = ((proj_lr - cmin) / crng).cpu().numpy()
        hr_std_rgb = ((proj_hr_std - cmin) / crng).cpu().numpy()
        hr_flash_rgb = ((proj_hr_flash - cmin) / crng).cpu().numpy()
    
    return lr_rgb, hr_std_rgb, hr_flash_rgb


def compute_difference_map(hr_std, hr_flash):
    """计算两个输出之间的差异"""
    with torch.no_grad():
        diff = (hr_std - hr_flash).abs()
        diff_per_pixel = diff.mean(dim=1)  # average over channels
        diff_mean = diff_per_pixel.mean().item()
        diff_max = diff_per_pixel.max().item()
        
        # Normalize for visualization
        diff_norm = diff_per_pixel[0].cpu().numpy()
        diff_norm = (diff_norm - diff_norm.min()) / (diff_norm.max() - diff_norm.min() + 1e-8)
    
    return diff_norm, diff_mean, diff_max


def save_results(img, target_width, target_height, lr_rgb, hr_std_rgb, hr_flash_rgb, diff_map, prefix="flash_hr"):
    """保存结果图像"""
    # 原图
    img_resized = img.resize((target_width, target_height))
    img_resized.save(f"{prefix}_1_original.png")
    print(f"Saved {prefix}_1_original.png ({target_width}x{target_height})")
    
    # LR 特征（放大显示）
    h, w = lr_rgb.shape[:2]
    lr_rgb_uint8 = (lr_rgb * 255).clip(0, 255).astype(np.uint8)
    lr_img = Image.fromarray(lr_rgb_uint8)
    lr_img_resized = lr_img.resize((target_width, target_height), Image.NEAREST)
    lr_img_resized.save(f"{prefix}_2_lr_features.png")
    print(f"Saved {prefix}_2_lr_features.png (resized from {w}x{h})")
    
    # 标准版 HR 特征
    hr_std_uint8 = (hr_std_rgb * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(hr_std_uint8).save(f"{prefix}_3_hr_standard.png")
    print(f"Saved {prefix}_3_hr_standard.png ({target_width}x{target_height})")
    
    # Flash 版 HR 特征
    hr_flash_uint8 = (hr_flash_rgb * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(hr_flash_uint8).save(f"{prefix}_4_hr_flash.png")
    print(f"Saved {prefix}_4_hr_flash.png ({target_width}x{target_height})")
    
    # 差异图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(diff_map, cmap='hot')
    plt.colorbar(label='Normalized Difference')
    plt.title('Standard vs Flash Difference Map')
    plt.axis('off')
    plt.savefig(f"{prefix}_5_difference.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {prefix}_5_difference.png")


def main():
    parser = argparse.ArgumentParser(description="Flash Cross Attention HR Image Test")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--resolution", type=int, default=None, 
                        help="Target resolution (square). If not set, uses original image size.")
    parser.add_argument("--flash-window-size", type=int, default=5, help="Flash attention window size")
    parser.add_argument("--q-chunk-size", type=int, default=256, help="Query chunk size for standard version")
    parser.add_argument("--output-prefix", type=str, default="flash_hr", help="Output filename prefix")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run benchmark, skip visualization")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 加载图像 - 默认使用 1274x854 (原图尺寸对齐到14的倍数)
    if args.resolution:
        target_height = target_width = (args.resolution // 14) * 14
        img, _, _ = load_image(args.image, target_height, target_width)
    else:
        target_height = 854  # 14 * 61
        target_width = 1274  # 14 * 91
        img, _, _ = load_image(args.image, target_height, target_width)
    
    print(f"Target resolution: {target_width}x{target_height} ({target_width * target_height / 1e6:.2f}M pixels)")
    
    # 提取 DINOv2 特征
    hr_image, lr_features = extract_dinov2_features(img, target_height, target_width, device)
    
    # 创建模型
    model_standard, model_flash = create_anyup_models(device, args.flash_window_size)
    
    print()
    print("=" * 60)
    print("Benchmarking...")
    print("=" * 60)
    
    # 测试标准版本
    try:
        hr_standard, std_time, std_mem = benchmark_upsampling(
            model_standard, hr_image, lr_features, 
            "Standard", q_chunk_size=args.q_chunk_size
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  Standard: OOM")
            hr_standard = None
            std_time = std_mem = float('inf')
            gc.collect()
            torch.cuda.empty_cache()
        else:
            raise
    
    # 测试 Flash 版本
    hr_flash, flash_time, flash_mem = benchmark_upsampling(
        model_flash, hr_image, lr_features, "Flash"
    )
    
    # 打印性能对比
    print()
    print("=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"Resolution: {target_width}x{target_height} ({target_width * target_height / 1e6:.2f}M pixels)")
    print(f"LR features: {lr_features.shape[2]}x{lr_features.shape[3]}")
    print()
    
    if hr_standard is not None:
        print(f"Standard AnyUp:")
        print(f"  - Time: {std_time:.1f} ms")
        print(f"  - Peak Memory: {std_mem:.2f} GB")
        print()
        print(f"Flash AnyUp:")
        print(f"  - Time: {flash_time:.1f} ms")
        print(f"  - Peak Memory: {flash_mem:.2f} GB")
        print()
        print(f"Speedup: {std_time / flash_time:.2f}x")
        print(f"Memory Savings: {(1 - flash_mem / std_mem) * 100:.1f}%")
    else:
        print("Standard version OOM, Flash version succeeded!")
        print(f"Flash AnyUp:")
        print(f"  - Time: {flash_time:.1f} ms")
        print(f"  - Peak Memory: {flash_mem:.2f} GB")
    
    if args.benchmark_only:
        print("\nBenchmark complete!")
        return
    
    # 可视化对比
    if hr_standard is not None:
        print()
        print("=" * 60)
        print("Generating visualizations...")
        print("=" * 60)
        
        # PCA 可视化
        lr_rgb, hr_std_rgb, hr_flash_rgb = compute_pca_visualization(
            lr_features, hr_standard, hr_flash, target_height, target_width
        )
        
        # 差异分析
        diff_map, diff_mean, diff_max = compute_difference_map(hr_standard, hr_flash)
        print(f"Output difference (Standard vs Flash):")
        print(f"  - Mean absolute difference: {diff_mean:.6f}")
        print(f"  - Max absolute difference: {diff_max:.6f}")
        
        # 保存结果
        save_results(img, target_width, target_height, lr_rgb, hr_std_rgb, hr_flash_rgb, diff_map, args.output_prefix)
    else:
        print("\nCannot generate comparison visualization (standard version OOM)")
        print("Generating Flash-only visualization...")
        
        # Flash-only PCA
        _, C, h, w = lr_features.shape
        with torch.no_grad():
            lr_flat = lr_features[0].permute(1, 2, 0).reshape(-1, C)
            hr_flash_flat = hr_flash[0].permute(1, 2, 0).reshape(-1, C)
            all_feats = torch.cat([lr_flat, hr_flash_flat], dim=0)
            mean = all_feats.mean(dim=0, keepdim=True)
            X = all_feats - mean
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            pcs = Vh[:3].T
            proj_all = X @ pcs
            
            n_lr = h * w
            proj_lr = proj_all[:n_lr].reshape(h, w, 3)
            proj_hr = proj_all[n_lr:].reshape(target_height, target_width, 3)
            
            cmin = proj_all.min(dim=0).values
            cmax = proj_all.max(dim=0).values
            crng = (cmax - cmin).clamp(min=1e-6)
            
            lr_rgb = ((proj_lr - cmin) / crng).cpu().numpy()
            hr_rgb = ((proj_hr - cmin) / crng).cpu().numpy()
        
        # 保存
        img_resized = img.resize((target_width, target_height))
        img_resized.save(f"{args.output_prefix}_1_original.png")
        
        lr_rgb_uint8 = (lr_rgb * 255).clip(0, 255).astype(np.uint8)
        lr_img = Image.fromarray(lr_rgb_uint8)
        lr_img.resize((target_width, target_height), Image.NEAREST).save(f"{args.output_prefix}_2_lr_features.png")
        
        hr_rgb_uint8 = (hr_rgb * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(hr_rgb_uint8).save(f"{args.output_prefix}_3_hr_flash.png")
        
        print(f"Saved {args.output_prefix}_*.png files")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
