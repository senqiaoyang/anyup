#!/usr/bin/env python3
"""
Flash Cross Attention Benchmark Script

对比原始 CrossAttentionBlock 和 FlashCrossAttentionBlock 的性能和显存使用。
测试推理模式和训练模式。

Usage:
    python benchmark.py
"""

import torch
import time
import gc
import argparse


def benchmark_inference(flash_block, orig_block, resolutions, patch_size, qk_dim, v_dim, device, dtype):
    """推理模式 benchmark (torch.no_grad)"""
    print("=" * 100)
    print("推理模式 (torch.no_grad)")
    print("=" * 100)
    print()
    print("| 分辨率 | 像素数 | 原始耗时 | Flash 耗时 | 加速比 | 原始显存 | Flash 显存 | 显存节省 |")
    print("|--------|--------|----------|------------|--------|----------|------------|----------|")

    for H, W in resolutions:
        h, w = H // patch_size, W // patch_size
        q = torch.randn(1, qk_dim, H, W, device=device, dtype=dtype)
        k = torch.randn(1, qk_dim, h, w, device=device, dtype=dtype)
        v = torch.randn(1, v_dim, h, w, device=device, dtype=dtype)

        # Flash timing
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = flash_block(q, k, v)  # warmup
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                out = flash_block(q, k, v)
        torch.cuda.synchronize()
        flash_time = (time.perf_counter() - start) / 10 * 1000
        flash_mem = torch.cuda.max_memory_allocated() / 1024**3
        del out

        # Original timing
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            with torch.no_grad():
                _ = orig_block(q, k, v, q_chunk_size=256)  # warmup
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(10):
                    out = orig_block(q, k, v, q_chunk_size=256)
            torch.cuda.synchronize()
            orig_time = (time.perf_counter() - start) / 10 * 1000
            orig_mem = torch.cuda.max_memory_allocated() / 1024**3
            del out

            speedup = orig_time / flash_time
            mem_save = (1 - flash_mem / orig_mem) * 100
            print(
                f"| {H}×{W} | {H*W/1e6:.2f}M | {orig_time:.1f} ms | {flash_time:.1f} ms | "
                f"**{speedup:.1f}×** | {orig_mem:.2f} GB | {flash_mem:.2f} GB | {mem_save:.0f}% |"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(
                    f"| {H}×{W} | {H*W/1e6:.2f}M | OOM | {flash_time:.1f} ms | "
                    f"- | OOM | {flash_mem:.2f} GB | - |"
                )
            else:
                raise

        del q, k, v
        gc.collect()
        torch.cuda.empty_cache()

    print()


def benchmark_training(flash_block, orig_block, resolutions, patch_size, qk_dim, v_dim, device, dtype):
    """训练模式 benchmark (forward + backward)"""
    print("=" * 100)
    print("训练模式 (forward + backward)")
    print("=" * 100)
    print()
    print("| 分辨率 | 像素数 | 原始耗时 | Flash 耗时 | 加速比 | 原始显存 | Flash 显存 | 显存节省 |")
    print("|--------|--------|----------|------------|--------|----------|------------|----------|")

    for H, W in resolutions:
        h, w = H // patch_size, W // patch_size

        # Flash - training mode
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        q = torch.randn(1, qk_dim, H, W, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(1, qk_dim, h, w, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(1, v_dim, h, w, device=device, dtype=dtype, requires_grad=True)

        # warmup
        out = flash_block(q, k, v)
        out.sum().backward()
        torch.cuda.synchronize()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        q = torch.randn(1, qk_dim, H, W, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(1, qk_dim, h, w, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(1, v_dim, h, w, device=device, dtype=dtype, requires_grad=True)

        start = time.perf_counter()
        for _ in range(5):
            out = flash_block(q, k, v)
            out.sum().backward()
            q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()
        flash_time = (time.perf_counter() - start) / 5 * 1000
        flash_mem = torch.cuda.max_memory_allocated() / 1024**3
        del out, q, k, v

        # Original - training mode
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            q = torch.randn(1, qk_dim, H, W, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(1, qk_dim, h, w, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(1, v_dim, h, w, device=device, dtype=dtype, requires_grad=True)

            # warmup
            out = orig_block(q, k, v, q_chunk_size=256)
            out.sum().backward()
            torch.cuda.synchronize()

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            q = torch.randn(1, qk_dim, H, W, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(1, qk_dim, h, w, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(1, v_dim, h, w, device=device, dtype=dtype, requires_grad=True)

            start = time.perf_counter()
            for _ in range(5):
                out = orig_block(q, k, v, q_chunk_size=256)
                out.sum().backward()
                q.grad = k.grad = v.grad = None
            torch.cuda.synchronize()
            orig_time = (time.perf_counter() - start) / 5 * 1000
            orig_mem = torch.cuda.max_memory_allocated() / 1024**3
            del out, q, k, v

            speedup = orig_time / flash_time
            mem_save = (1 - flash_mem / orig_mem) * 100
            print(
                f"| {H}×{W} | {H*W/1e6:.2f}M | {orig_time:.1f} ms | {flash_time:.1f} ms | "
                f"**{speedup:.1f}×** | {orig_mem:.2f} GB | {flash_mem:.2f} GB | {mem_save:.0f}% |"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(
                    f"| {H}×{W} | {H*W/1e6:.2f}M | OOM | {flash_time:.1f} ms | "
                    f"- | OOM | {flash_mem:.2f} GB | - |"
                )
            else:
                raise

        gc.collect()
        torch.cuda.empty_cache()

    print()


def main():
    parser = argparse.ArgumentParser(description="Flash Cross Attention Benchmark")
    parser.add_argument("--qk-dim", type=int, default=128, help="Q/K dimension")
    parser.add_argument("--v-dim", type=int, default=768, help="V dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--window-size", type=int, default=5, help="Flash attention window size")
    parser.add_argument("--patch-size", type=int, default=14, help="ViT patch size")
    parser.add_argument(
        "--resolutions",
        type=str,
        default="504,1008,1512,2016",
        help="Comma-separated list of resolutions to test (square images)",
    )
    parser.add_argument("--inference-only", action="store_true", help="Only run inference benchmark")
    parser.add_argument("--training-only", action="store_true", help="Only run training benchmark")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    resolutions = [(int(r), int(r)) for r in args.resolutions.split(",")]

    print()
    print("=" * 100)
    print("Flash Cross Attention Benchmark")
    print("=" * 100)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"dtype: {dtype}")
    print(f"qk_dim: {args.qk_dim}, v_dim: {args.v_dim}, num_heads: {args.num_heads}")
    print(f"window_size: {args.window_size}, patch_size: {args.patch_size}")
    print(f"Resolutions: {resolutions}")
    print()

    # Import and create models
    from anyup.layers.attention.flash_cross_attention import FlashCrossAttentionBlock
    from anyup.layers.attention.chunked_attention import CrossAttentionBlock

    flash_block = FlashCrossAttentionBlock(
        qk_dim=args.qk_dim,
        num_heads=args.num_heads,
        window_size=args.window_size,
    ).to(device).to(dtype)

    orig_block = CrossAttentionBlock(
        qk_dim=args.qk_dim,
        num_heads=args.num_heads,
        window_ratio=0.1,
    ).to(device).to(dtype)

    # Run benchmarks
    if not args.training_only:
        benchmark_inference(
            flash_block, orig_block, resolutions, args.patch_size, args.qk_dim, args.v_dim, device, dtype
        )

    if not args.inference_only:
        # Use smaller resolutions for training to avoid OOM
        train_resolutions = [(r, r) for r, _ in resolutions if r <= 1512]
        benchmark_training(
            flash_block, orig_block, train_resolutions, args.patch_size, args.qk_dim, args.v_dim, device, dtype
        )

    print("Benchmark complete!")


if __name__ == "__main__":
    main()
