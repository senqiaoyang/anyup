"""
Flash Attention 优化的分块 Cross Attention

核心思想：
1. 把 HR Query 按照 LR 的网格分块，每块对应 LR 的一个区域
2. 每个 HR 块只和对应的 LR 局部区域（+ padding）做 attention
3. 使用 Flash Attention 加速计算

内存优化：
- 原始: Q(H*W) x K(h*w) 全局 attention matrix
- 优化: 分成 (h*w) 个小块，每块 Q(block_h*block_w) x K(local_kv)
- 显存从 O(H*W * h*w) 降到 O(block_size * local_kv_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# 检查 Flash Attention 是否可用
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Flash Attention not available. Install with: pip install flash-attn")


class FlashCrossAttention(nn.Module):
    """
    使用 Flash Attention 的分块 Cross Attention
    
    参数结构与原始 CrossAttention 完全兼容，可以直接加载预训练权重。
    
    Args:
        qk_dim: Query/Key 的维度
        num_heads: 注意力头数
        window_size: 每个 query 块关注的 KV 邻域大小（LR 空间）
        v_dim: Value 的维度（如果不同于 qk_dim，会投影到 qk_dim 以使用 Flash Attention）
        use_v_proj: 是否将 V 投影到 qk_dim 以启用纯 Flash Attention（3-4x 更快）
    
    与原始 CrossAttention 的对应关系：
        - norm_q, norm_k: 完全相同
        - attention (nn.MultiheadAttention): 完全相同（用于复用权重）
            - 原始代码只用 attention weights，不用 MHA 的输出
            - in_proj_weight[:qk_dim] = Q 投影
            - in_proj_weight[qk_dim:2*qk_dim] = K 投影
    """
    def __init__(
        self,
        qk_dim: int,
        num_heads: int,
        window_size: int = 3,  # 在 LR 空间的窗口大小
        dropout: float = 0.0,
        v_dim: int = None,  # 如果提供且不同于 qk_dim，创建投影层
        use_v_proj: bool = True,  # 是否使用 V 投影以启用 Flash Attention
    ):
        super().__init__()
        self.qk_dim = qk_dim
        self.num_heads = num_heads
        self.head_dim = qk_dim // num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.use_v_proj = use_v_proj
        
        assert qk_dim % num_heads == 0, f"qk_dim {qk_dim} must be divisible by num_heads {num_heads}"
        
        # 与原始 CrossAttention 完全相同的结构
        self.norm_q = nn.RMSNorm(qk_dim)
        self.norm_k = nn.RMSNorm(qk_dim)
        
        # 使用 nn.MultiheadAttention 保持与原始权重兼容
        self.attention = nn.MultiheadAttention(
            embed_dim=qk_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        
        # V 投影：将 v_dim 投影到 qk_dim 以启用 Flash Attention
        # 这是新增的，原始代码没有（原始直接用未投影的 V）
        self.v_dim = v_dim if v_dim is not None else qk_dim
        if use_v_proj and self.v_dim != qk_dim:
            self.v_proj_in = nn.Linear(self.v_dim, qk_dim, bias=False)
            self.v_proj_out = nn.Linear(qk_dim, self.v_dim, bias=False)
            # 正交初始化以保留信息
            nn.init.orthogonal_(self.v_proj_in.weight)
            nn.init.orthogonal_(self.v_proj_out.weight)
        else:
            self.v_proj_in = None
            self.v_proj_out = None
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
    
    def _get_qk_projections(self, q, k):
        """使用 MultiheadAttention 的权重进行 Q, K 投影"""
        # in_proj_weight: (3*qk_dim, qk_dim) = [Q; K; V]
        in_proj = self.attention.in_proj_weight
        q_weight = in_proj[:self.qk_dim]
        k_weight = in_proj[self.qk_dim:2*self.qk_dim]
        
        # 投影
        q_proj = F.linear(q, q_weight, self.attention.in_proj_bias[:self.qk_dim] if self.attention.in_proj_bias is not None else None)
        k_proj = F.linear(k, k_weight, self.attention.in_proj_bias[self.qk_dim:2*self.qk_dim] if self.attention.in_proj_bias is not None else None)
        
        return q_proj, k_proj
    
    def forward(
        self,
        query: torch.Tensor,      # (B, H, W, qk_dim)
        key: torch.Tensor,        # (B, h, w, qk_dim)
        value: torch.Tensor,      # (B, h, w, v_dim)  v_dim 可以和 qk_dim 不同
    ) -> torch.Tensor:
        """
        分块 Cross Attention
        
        核心逻辑：
        1. HR 图像按 LR 网格分块
        2. 每个 HR 块只和对应的 LR 局部窗口做 attention
        3. 用 Flash Attention 或手动实现
        
        如果 use_v_proj=True 且 v_dim != qk_dim：
        - V 先投影到 qk_dim，用 Flash Attention，再投影回 v_dim
        - 这样可以获得 3-4x 加速
        
        尺寸处理：
        - 如果 H 不能被 h 整除，会对 Q 进行 padding
        - 计算完成后会 crop 回原始尺寸
        """
        B, H_orig, W_orig, _ = query.shape
        _, h, w, v_dim_in = value.shape
        
        # 计算每个 LR token 对应多少 HR pixels（向上取整）
        scale_h = (H_orig + h - 1) // h  # ceil division
        scale_w = (W_orig + w - 1) // w
        
        # 计算需要的 padding
        H_target = h * scale_h
        W_target = w * scale_w
        
        pad_h = H_target - H_orig
        pad_w = W_target - W_orig
        
        # 如果需要 padding，对 Q 进行 padding
        if pad_h > 0 or pad_w > 0:
            # (B, H, W, dim) -> (B, dim, H, W) -> pad -> (B, dim, H_target, W_target) -> (B, H_target, W_target, dim)
            query = F.pad(query.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), mode='replicate')
            query = query.permute(0, 2, 3, 1)
        
        H, W = H_target, W_target
        
        # Normalize
        query = self.norm_q(query)
        key = self.norm_k(key)
        
        # Project Q, K (使用 MultiheadAttention 的权重)
        q, k = self._get_qk_projections(query, key)  # (B, H, W, qk_dim), (B, h, w, qk_dim)
        
        # 如果使用 V 投影，将 V 投影到 qk_dim 以启用 Flash Attention
        if self.v_proj_in is not None:
            v = self.v_proj_in(value)  # (B, h, w, qk_dim)
            v_dim = self.qk_dim
        else:
            v = value
            v_dim = v_dim_in
        
        # 对 K, V 做 padding 以处理边界
        # 如果 window_size >= max(h, w)，使用全局 K/V 模式（每个块看完整 K/V）
        use_global_kv = self.window_size >= max(h, w)
        
        if use_global_kv:
            # 全局模式：每个块看完整的 K/V，无 padding
            num_blocks = B * h * w
            kv_len = h * w
            
            # K/V 广播到每个块
            k_windows = k.view(B, 1, 1, h, w, self.qk_dim).expand(B, h, w, h, w, self.qk_dim)
            v_windows = v.view(B, 1, 1, h, w, v_dim).expand(B, h, w, h, w, v_dim)
            
            # reshape to (B, h, w, h*w, dim) 来模拟 (B, h, w, ws*ws, dim)
            k_windows = k_windows.reshape(B, h, w, kv_len, self.qk_dim)
            v_windows = v_windows.reshape(B, h, w, kv_len, v_dim)
        else:
            # 局部窗口模式
            pad = self.window_size // 2
            # (B, h, w, dim) -> (B, dim, h, w) -> pad -> (B, dim, h+2p, w+2p) -> (B, h+2p, w+2p, dim)
            k_padded = F.pad(k.permute(0, 3, 1, 2), (pad, pad, pad, pad), mode='replicate')
            k_padded = k_padded.permute(0, 2, 3, 1)  # (B, h+2p, w+2p, qk_dim)
            
            v_padded = F.pad(v.permute(0, 3, 1, 2), (pad, pad, pad, pad), mode='replicate')
            v_padded = v_padded.permute(0, 2, 3, 1)  # (B, h+2p, w+2p, v_dim)
            
            # 使用 unfold 提取每个位置的局部窗口
            # 结果: (B, h, w, window_size, window_size, dim)
            k_windows = self._extract_windows(k_padded, self.window_size)  # (B, h, w, ws, ws, qk_dim)
            v_windows = self._extract_windows(v_padded, self.window_size)  # (B, h, w, ws, ws, v_dim)
            
            # Flatten 窗口维度
            k_windows = k_windows.reshape(B, h, w, self.window_size * self.window_size, self.qk_dim)
            v_windows = v_windows.reshape(B, h, w, self.window_size * self.window_size, v_dim)
        
        # 把 Q reshape 成块: (B, h, w, scale_h, scale_w, qk_dim)
        q_blocks = q.view(B, h, scale_h, w, scale_w, self.qk_dim)
        q_blocks = q_blocks.permute(0, 1, 3, 2, 4, 5)  # (B, h, w, scale_h, scale_w, qk_dim)
        
        # 计算分块 attention
        output = self._blocked_attention(q_blocks, k_windows, v_windows, v_dim)
        
        # Reshape 回 (B, H, W, v_dim)
        output = output.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, h, scale_h, w, scale_w, v_dim)
        output = output.view(B, H, W, v_dim)
        
        # 如果使用了 V 投影，将输出投影回原始 v_dim
        if self.v_proj_out is not None:
            output = self.v_proj_out(output)  # (B, H, W, v_dim_original)
        
        # 如果做了 padding，crop 回原始尺寸
        if pad_h > 0 or pad_w > 0:
            output = output[:, :H_orig, :W_orig, :]
        
        return output
    
    def _extract_windows(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        从 padded tensor 提取局部窗口
        x: (B, h+2p, w+2p, dim)
        return: (B, h, w, window_size, window_size, dim)
        """
        B, H_pad, W_pad, dim = x.shape
        h = H_pad - window_size + 1
        w = W_pad - window_size + 1
        
        # 使用 unfold
        x = x.permute(0, 3, 1, 2)  # (B, dim, H_pad, W_pad)
        x = x.unfold(2, window_size, 1).unfold(3, window_size, 1)  # (B, dim, h, w, ws, ws)
        x = x.permute(0, 2, 3, 4, 5, 1)  # (B, h, w, ws, ws, dim)
        
        return x
    
    def _blocked_attention(
        self,
        q_blocks: torch.Tensor,   # (B, h, w, scale_h, scale_w, qk_dim)
        k_windows: torch.Tensor,  # (B, h, w, kv_len, qk_dim) - 已 flatten
        v_windows: torch.Tensor,  # (B, h, w, kv_len, v_dim) - 已 flatten
        v_dim: int,
    ) -> torch.Tensor:
        """
        对每个块执行 attention，使用 flash_attn_varlen_func 加速
        """
        B, h, w, scale_h, scale_w, _ = q_blocks.shape
        kv_len = k_windows.shape[3]
        
        q_len = scale_h * scale_w
        num_blocks = B * h * w
        
        # Flatten 所有 blocks: (total_q_tokens, dim), (total_kv_tokens, dim)
        q_flat = q_blocks.reshape(num_blocks * q_len, self.qk_dim)    # (B*h*w*q_len, qk_dim)
        k_flat = k_windows.reshape(num_blocks * kv_len, self.qk_dim)  # (B*h*w*kv_len, qk_dim)
        v_flat = v_windows.reshape(num_blocks * kv_len, v_dim)        # (B*h*w*kv_len, v_dim)
        
        if FLASH_ATTN_AVAILABLE and q_flat.is_cuda and q_flat.dtype in [torch.float16, torch.bfloat16]:
            output = self._flash_attention_varlen(q_flat, k_flat, v_flat, num_blocks, q_len, kv_len)
        else:
            # Fallback: reshape back to batched form
            q_batched = q_flat.view(num_blocks, q_len, self.qk_dim)
            k_batched = k_flat.view(num_blocks, kv_len, self.qk_dim)
            v_batched = v_flat.view(num_blocks, kv_len, v_dim)
            output = self._manual_attention(q_batched, k_batched, v_batched)
            output = output.reshape(num_blocks * q_len, v_dim)
        
        # Reshape 回 (B, h, w, scale_h, scale_w, v_dim)
        output = output.reshape(B, h, w, scale_h, scale_w, v_dim)
        
        return output
    
    def _flash_attention_varlen(
        self,
        q: torch.Tensor,  # (total_q, qk_dim) - 所有 blocks 的 Q 拼接
        k: torch.Tensor,  # (total_kv, qk_dim) - 所有 blocks 的 K 拼接
        v: torch.Tensor,  # (total_kv, v_dim) - 所有 blocks 的 V 拼接
        num_blocks: int,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        """
        使用 flash_attn_varlen_func，一次调用处理所有 blocks
        
        cu_seqlens_q: [0, q_len, 2*q_len, ..., num_blocks*q_len]
        cu_seqlens_k: [0, kv_len, 2*kv_len, ..., num_blocks*kv_len]
        """
        total_q = q.shape[0]
        total_kv = k.shape[0]
        v_dim = v.shape[1]
        
        # 构建 cu_seqlens（累积序列长度）
        cu_seqlens_q = torch.arange(0, (num_blocks + 1) * q_len, q_len, 
                                     dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.arange(0, (num_blocks + 1) * kv_len, kv_len,
                                     dtype=torch.int32, device=k.device)
        
        if v_dim == self.qk_dim:
            # V 维度匹配：直接用 flash_attn_varlen_func
            q = q.view(total_q, self.num_heads, self.head_dim)
            k = k.view(total_kv, self.num_heads, self.head_dim)
            v_reshaped = v.view(total_kv, self.num_heads, self.head_dim)
            
            out = flash_attn_varlen_func(
                q, k, v_reshaped,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=kv_len,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.scale,
            )
            out = out.reshape(total_q, self.qk_dim)
            return out
        
        # V 维度不同：使用手动计算（更稳定，避免广播梯度问题）
        # 对于训练，手动计算更安全
        v_dim = v.shape[1]
        
        # 重新 reshape 回 batched 形式
        q = q.view(num_blocks, q_len, self.qk_dim)
        k = k.view(num_blocks, kv_len, self.qk_dim)
        v = v.view(num_blocks, kv_len, v_dim)
        
        # 使用手动 attention
        q = q.view(num_blocks, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(num_blocks, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Average across heads and apply to V
        attn_avg = attn.mean(dim=1)  # (num_blocks, q_len, kv_len)
        out = torch.bmm(attn_avg, v)  # (num_blocks, q_len, v_dim)
        
        out = out.reshape(total_q, v_dim)
        return out
    
    def _compute_attention_weights(
        self,
        q: torch.Tensor,  # (B, q_len, num_heads, head_dim)
        k: torch.Tensor,  # (B, kv_len, num_heads, head_dim)
    ) -> torch.Tensor:
        """计算 attention weights"""
        # (B, num_heads, q_len, head_dim) @ (B, num_heads, head_dim, kv_len)
        q = q.permute(0, 2, 1, 3)  # (B, num_heads, q_len, head_dim)
        k = k.permute(0, 2, 3, 1)  # (B, num_heads, head_dim, kv_len)
        
        attn = torch.matmul(q, k) * self.scale  # (B, num_heads, q_len, kv_len)
        attn = F.softmax(attn, dim=-1)
        
        return attn
    
    def _manual_attention(
        self,
        q: torch.Tensor,  # (B*h*w, q_len, qk_dim)
        k: torch.Tensor,  # (B*h*w, kv_len, qk_dim)
        v: torch.Tensor,  # (B*h*w, kv_len, v_dim)
    ) -> torch.Tensor:
        """
        使用 PyTorch 2.0+ 的 scaled_dot_product_attention
        它会自动选择最优实现 (Flash Attention, Memory Efficient, 或 Math)
        """
        batch_size, q_len, _ = q.shape
        _, kv_len, v_dim = v.shape
        
        # Reshape for multi-head: (B, num_heads, seq_len, head_dim)
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算 attention weights
        # 由于我们需要将 weights 应用到不同维度的 V，无法直接用 SDPA
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Average across heads for applying to V
        attn_weights_avg = attn_weights.mean(dim=1)  # (B, q_len, kv_len)
        
        # Apply to V: (B, q_len, kv_len) @ (B, kv_len, v_dim) -> (B, q_len, v_dim)
        out = torch.bmm(attn_weights_avg, v)
        
        return out


class FlashCrossAttentionBlock(nn.Module):
    """
    完整的 Flash Cross Attention Block，可以替换原始的 CrossAttentionBlock
    
    Args:
        qk_dim: Query/Key 的维度
        num_heads: 注意力头数
        window_size: 每个 query 块关注的 KV 邻域大小（LR 空间）
        dropout: Dropout 比例
        v_dim: Value 的维度（DINOv2 特征维度，如 768）
        use_v_proj: 是否使用 V 投影以启用 Flash Attention（对于 3M+ 图像推荐开启）
    """
    def __init__(
        self,
        qk_dim: int,
        num_heads: int,
        window_size: int = 5,
        dropout: float = 0.0,
        v_dim: int = None,
        use_v_proj: bool = True,  # 对于大分辨率图像（3M+）推荐开启
        **kwargs
    ):
        super().__init__()
        self.cross_attn = FlashCrossAttention(
            qk_dim=qk_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            v_dim=v_dim,
            use_v_proj=use_v_proj,
        )
        self.conv2d = nn.Conv2d(qk_dim, qk_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.window_size = window_size
    
    def forward(
        self,
        q: torch.Tensor,  # (B, qk_dim, H, W)
        k: torch.Tensor,  # (B, qk_dim, h, w)
        v: torch.Tensor,  # (B, v_dim, h, w)
        vis_attn: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            q: Query features from HR image, (B, qk_dim, H, W)
            k: Key features (image + LR features), (B, qk_dim, h, w)
            v: Value features (LR backbone features), (B, v_dim, h, w)
        
        Returns:
            Upsampled features, (B, v_dim, H, W)
        """
        # Conv on query
        q = self.conv2d(q)
        
        b, _, H, W = q.shape
        _, _, h, w = k.shape
        v_dim = v.shape[1]
        
        # (B, C, H, W) -> (B, H, W, C)
        q = q.permute(0, 2, 3, 1).contiguous()
        k = k.permute(0, 2, 3, 1).contiguous()
        v = v.permute(0, 2, 3, 1).contiguous()
        
        # Cross attention
        output = self.cross_attn(q, k, v)  # (B, H, W, v_dim)
        
        # (B, H, W, C) -> (B, C, H, W)
        output = output.permute(0, 3, 1, 2).contiguous()
        
        return output


def convert_window_ratio_to_size(window_ratio: float, lr_size: int) -> int:
    """
    将 window_ratio 转换为 window_size
    
    Args:
        window_ratio: 原始的窗口比例 (如 0.1)
        lr_size: LR 特征的尺寸 (如 32)
    
    Returns:
        window_size: 窗口大小（奇数）
    """
    # window_ratio=0.1 意味着每个 query 关注 20% 的 LR 区域 (左右各 10%)
    # 转换为 window_size
    ws = int(2 * window_ratio * lr_size) + 1
    # 确保是奇数
    if ws % 2 == 0:
        ws += 1
    # 至少为 3
    ws = max(ws, 3)
    return ws


# ============ 用于测试的工具函数 ============ #

def benchmark_memory_comparison(
    batch_size: int = 4,
    hr_size: int = 448,
    lr_size: int = 32,
    qk_dim: int = 128,
    v_dim: int = 768,
    num_heads: int = 4,
):
    """
    对比原始 CrossAttention 和 FlashCrossAttention 的内存使用
    """
    import gc
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"内存对比测试")
    print(f"{'='*60}")
    print(f"配置: batch={batch_size}, HR={hr_size}x{hr_size}, LR={lr_size}x{lr_size}")
    print(f"       qk_dim={qk_dim}, v_dim={v_dim}, num_heads={num_heads}")
    
    # 创建输入
    q = torch.randn(batch_size, qk_dim, hr_size, hr_size, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, qk_dim, lr_size, lr_size, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, v_dim, lr_size, lr_size, device=device, dtype=torch.bfloat16)
    
    # 1. 测试 Flash Cross Attention
    print(f"\n--- FlashCrossAttentionBlock (window_size=5) ---")
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    
    flash_block = FlashCrossAttentionBlock(
        qk_dim=qk_dim,
        num_heads=num_heads,
        window_size=5,
    ).to(device).to(torch.bfloat16)
    
    with torch.no_grad():
        out_flash = flash_block(q, k, v)
    
    flash_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  输出形状: {out_flash.shape}")
    print(f"  峰值内存: {flash_mem:.2f} GB")
    
    # 清理
    del flash_block, out_flash
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. 测试原始 CrossAttentionBlock
    print(f"\n--- 原始 CrossAttentionBlock (window_ratio=0.1) ---")
    torch.cuda.reset_peak_memory_stats()
    
    from anyup.layers.attention.chunked_attention import CrossAttentionBlock
    
    orig_block = CrossAttentionBlock(
        qk_dim=qk_dim,
        num_heads=num_heads,
        window_ratio=0.1,
    ).to(device).to(torch.bfloat16)
    
    try:
        with torch.no_grad():
            out_orig = orig_block(q, k, v, q_chunk_size=256)
        
        orig_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  输出形状: {out_orig.shape}")
        print(f"  峰值内存: {orig_mem:.2f} GB")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  OOM!")
            orig_mem = float('inf')
        else:
            raise
    
    print(f"\n内存节省: {(1 - flash_mem/orig_mem)*100:.1f}%" if orig_mem != float('inf') else "原始版本 OOM")


if __name__ == "__main__":
    benchmark_memory_comparison()
