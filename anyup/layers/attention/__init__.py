try:
    from .natten_attention import NATTENCrossAttentionBlock
except ImportError:
    NATTENCrossAttentionBlock = None

try:
    from .flash_cross_attention import FlashCrossAttentionBlock, convert_window_ratio_to_size
    FLASH_CROSS_ATTN_AVAILABLE = True
except ImportError:
    FlashCrossAttentionBlock = None
    FLASH_CROSS_ATTN_AVAILABLE = False

from .chunked_attention import CrossAttentionBlock
from typing import Optional
from torch import nn
import warnings


def setup_cross_attention_block(use_natten: bool,
                                qk_dim: int,
                                num_heads: int,
                                window_ratio: float = 0.1,
                                q_chunk_size: Optional[int] = None,
                                use_flash: bool = False,
                                flash_window_size: Optional[int] = None,
                                v_dim: int = 768,  # DINOv2 feature dimension
                                **kwargs) -> nn.Module:
    # Flash Attention 优先级最高
    if use_flash:
        if not FLASH_CROSS_ATTN_AVAILABLE:
            warnings.warn(
                "FlashCrossAttentionBlock is not available. "
                "Falling back to standard CrossAttentionBlock."
            )
        else:
            # 如果没有指定 window_size，根据 window_ratio 估算（假设 LR 大小为 32）
            ws = flash_window_size if flash_window_size else convert_window_ratio_to_size(window_ratio, 32)
            print(f"Using Flash Cross-Attention Block with window_size={ws}")
            return FlashCrossAttentionBlock(
                qk_dim=qk_dim,
                num_heads=num_heads,
                window_size=ws,
                v_dim=v_dim,
                use_v_proj=False,  # 为了与原始完全兼容，默认关闭 v_proj
                **kwargs
            )
    
    if use_natten:
        if NATTENCrossAttentionBlock is None:
            warnings.warn(
                "NATTENCrossAttentionBlock is not available."
                "Please ensure that the natten module is installed correctly."
                "Falling back to standard CrossAttentionBlock."
            )
            return CrossAttentionBlock(
                qk_dim=qk_dim,
                num_heads=num_heads,
                window_ratio=window_ratio,
                q_chunk_size=q_chunk_size,
                **kwargs
            )
        print("Using the optimized NATTEN Cross-Attention Block. Does not match the standard cross-attention exactly.")
        return NATTENCrossAttentionBlock(
            qk_dim=qk_dim,
            num_heads=num_heads,
            window_ratio=window_ratio,
            q_chunk_size=q_chunk_size,
            **kwargs
        )
    else:
        return CrossAttentionBlock(
            qk_dim=qk_dim,
            num_heads=num_heads,
            window_ratio=window_ratio,
            q_chunk_size=q_chunk_size,
            **kwargs
        )
