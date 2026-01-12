# Flash Cross Attention for AnyUp

基于 `flash_attn_varlen_func` 的高效分块 Cross Attention 实现，专为 ViT 特征上采样优化。

## 1. 我们的改动

### 核心改动

我们为 AnyUp 添加了 **Flash Attention 加速的分块 Cross Attention**，主要改动如下：

| 文件 | 改动 |
|------|------|
| `anyup/layers/attention/flash_cross_attention.py` | **新增** - Flash Cross Attention 核心实现 |
| `anyup/layers/attention/__init__.py` | 添加 `use_flash` 参数支持 |
| `anyup/model.py` | 添加 `use_flash` 和 `flash_window_size` 参数 |

### 技术方案

**原始方案**：全局 Attention + Mask
```
Q (H×W) × K (h×w) = Attention Matrix (H×W, h×w)
问题：3M 像素图像需要 ~37GB 显存
```

**Flash 方案**：分块 Attention + flash_attn_varlen_func
```
HR Query 按 LR 网格分块，每块对应一个 LR patch 的感受野
每个 Q 块只关注对应位置周围的 K/V 局部窗口
所有块合并成一次 flash_attn_varlen_func 调用
```

```
                            输入图像 (854×1274)
                        ┌─────────────────────────┐
                        │                         │
                        │      原始高分辨率图像     │
                        │                         │
                        └───────────┬─────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ↓                                               ↓
      Image Encoder                                    DINOv2 ViT
      (conv layers)                                   (frozen backbone)
            ↓                                               ↓
      enc_img (854×1274)                              LR Features (61×91)
      128 channels                                    768 channels
            │                                               │
    ┌───────┴───────┐                           ┌───────────┴───────────┐
    ↓               ↓                           ↓                       ↓
Query Encoder   Key Encoder                Key Features Encoder     Value (V)
    ↓           (pool到61×91)                   ↓                  = LR Features
    ↓               ↓                           ↓                       ↓
    Q           k_img (61×91)               k_feat (61×91)         (61×91, 768ch)
(854×1274)      128 ch                       128 ch
 128 ch             │                           │
                    └─────────┬─────────────────┘
                              ↓
                        Concat + Aggregation
                              ↓
                           K (61×91)
                           128 channels


    HR Query 按 LR 网格分块              LR Key/Value 网格
    (每块对应 14×14 HR pixels)           (每个格子 = 1 LR token)
    
    ┌─────┬─────┬─────┬─────┐           ┌───┬───┬───┬───┐
    │ Q₀₀ │ Q₀₁ │ Q₀₂ │ ... │           │K₀₀│K₀₁│K₀₂│...│
    │14×14│14×14│14×14│     │           ├───┼───┼───┼───┤
    ├─────┼─────┼─────┼─────┤           │K₁₀│K₁₁│K₁₂│...│
    │ Q₁₀ │ Q₁₁ │ Q₁₂ │ ... │           ├───┼───┼───┼───┤
    │14×14│14×14│14×14│     │           │K₂₀│K₂₁│K₂₂│...│
    ├─────┼─────┼─────┼─────┤           ├───┼───┼───┼───┤
    │ ... │ ... │ ... │ ... │           │...│...│...│...│
    └─────┴─────┴─────┴─────┘           └───┴───┴───┴───┘
      共 61×91 = 5,551 个块               共 61×91 = 5,551 个 tokens
      每块 196 个 Q tokens                每个 = 1 个 K/V token


    注意力计算示例：Q₁₁ 块 → 以 K₁₁ 为中心的 5×5 窗口
    
                                        ┌───────┬───────┬───────┬───────┬───────┐
    ┌─────┐                             │ K₋₁,₋₁│ K₋₁,₀ │ K₋₁,₁ │ K₋₁,₂ │ K₋₁,₃ │ ← 边界 padding
    │ Q₁₁ │  ───── attention ─────▶     ├───────┼───────┼───────┼───────┼───────┤
    │14×14│  (196 × 25 矩阵)            │ K₀,₋₁ │ K₀,₀  │ K₀,₁  │ K₀,₂  │ K₀,₃  │
    │=196 │                             ├───────┼───────┼───────┼───────┼───────┤
    │tokens│                            │ K₁,₋₁ │ K₁,₀  │  ███  │ K₁,₂  │ K₁,₃  │ ← K₁₁ 是中心
    └─────┘                             ├───────┼───────┼───────┼───────┼───────┤
                                        │ K₂,₋₁ │ K₂,₀  │ K₂,₁  │ K₂,₂  │ K₂,₃  │
                                        ├───────┼───────┼───────┼───────┼───────┤
                                        │ K₃,₋₁ │ K₃,₀  │ K₃,₁  │ K₃,₂  │ K₃,₃  │
                                        └───────┴───────┴───────┴───────┴───────┘
                                                  5×5 = 25 个 K/V tokens
    
    输出：196 个 HR tokens，每个融合了以 K₁₁ 为中心的 25 个 LR tokens 信息
```



## 2. 改动效果

### 性能对比

**测试环境**: NVIDIA H100 NVL (93.1 GB), PyTorch 2.9.0, bfloat16, batch_size=1

#### 推理模式 (torch.no_grad)

| 分辨率 | 像素数 | 原始耗时 | Flash 耗时 | 加速比 | 原始显存 | Flash 显存 | 显存节省 |
|--------|--------|----------|------------|--------|----------|------------|----------|
| 504×504 | 0.25M | 309 ms | 4.4 ms | **70×** | 1.61 GB | 1.43 GB | 12% |
| 1008×1008 | 1.0M | 1,422 ms | 31 ms | **45×** | 10.0 GB | 5.6 GB | 44% |
| 1512×1512 | 2.3M | 4,000 ms | 74 ms | **54×** | 36.3 GB | 12.6 GB | **65%** |
| 2016×2016 | 4.1M | OOM | 130 ms | - | OOM | 22.3 GB | - |

#### 训练模式 (forward + backward)

| 分辨率 | 像素数 | 原始耗时 | Flash 耗时 | 加速比 | 原始显存 | Flash 显存 | 显存节省 |
|--------|--------|----------|------------|--------|----------|------------|----------|
| 504×504 | 0.25M | 874 ms | 8.9 ms | **98×** | 5.2 GB | 1.7 GB | **67%** |
| 1008×1008 | 1.0M | 4,125 ms | 50 ms | **83×** | 67.1 GB | 6.7 GB | **90%** |
| 1512×1512 | 2.3M | OOM | 113 ms | - | OOM | 14.9 GB | - |


---

## 3. 如何快速应用



### 3.1 嵌入到自己的架构

只需复制一个文件：

```bash
cp anyup/layers/attention/flash_cross_attention.py your_project/
```

```python
import torch
import torch.nn as nn
from flash_cross_attention import FlashCrossAttentionBlock

class YourSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 1. Backbone (冻结)
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        # 2. Query/Key 编码器
        self.query_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
        )
        self.key_encoder = nn.Conv2d(768, 128, 1)
        
        # 3. Flash Cross Attention 上采样器
        self.upsampler = FlashCrossAttentionBlock(
            qk_dim=128,
            num_heads=4,
            window_size=7,      # 局部窗口
            v_dim=768,          # DINOv2 特征维度
            use_v_proj=False,   # False=与原始权重兼容, True=更快但需重新训练
        )
        
        # 4. 任务头
        self.head = nn.Conv2d(768, num_classes, 1)
    
    def forward(self, image):
        B, _, H, W = image.shape
        
        # 提取 LR 特征
        with torch.no_grad():
            tokens = self.backbone.forward_features(image)['x_norm_patchtokens']
            h, w = H // 14, W // 14
            lr_features = tokens.permute(0, 2, 1).view(B, 768, h, w)
        
        # 编码 Q, K
        q = self.query_encoder(image)      # (B, 128, H, W)
        k = self.key_encoder(lr_features)  # (B, 128, h, w)
        v = lr_features                    # (B, 768, h, w)
        
        # 上采样
        hr_features = self.upsampler(q, k, v)  # (B, 768, H, W)
        
        # 预测
        return self.head(hr_features)
```

### 3.2 训练配置

```python
# 训练时使用
model = YourSegmentationModel(num_classes=150).cuda()
model.train()

# 冻结 backbone
for p in model.backbone.parameters():
    p.requires_grad = False

# 优化器只更新 upsampler 和 head
optimizer = torch.optim.AdamW([
    {'params': model.query_encoder.parameters()},
    {'params': model.key_encoder.parameters()},
    {'params': model.upsampler.parameters()},
    {'params': model.head.parameters()},
], lr=1e-4)

# 训练循环
for images, labels in dataloader:
    images = images.cuda()
    labels = labels.cuda()
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 3.3 测试/推理

```python
model.eval()
with torch.no_grad():
    # 支持任意分辨率
    outputs = model(test_image)  # 自动处理非 14 倍数的尺寸
```

---

## 4. 参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `qk_dim` | 128 | Q/K 维度，需能被 `num_heads` 整除 |
| `num_heads` | 4 | 注意力头数 |
| `window_size` | 5-9 | LR 空间窗口大小，5 = 每块关注 25 个 K/V tokens |
| `v_dim` | 768 | Value 维度 (DINOv2-B=768, DINOv2-L=1024) |
| `use_v_proj` | False | True=更快(需重新训练), False=兼容原始权重 |

### window_size 选择

```
window_size=5:  每块关注 5×5=25 个 K/V tokens (最快，局部性最强)
window_size=7:  每块关注 7×7=49 个 K/V tokens (推荐平衡)
window_size=9:  每块关注 9×9=81 个 K/V tokens (更多上下文)
window_size≥max(h,w): 全局 attention (等价于原始)
```

---

## 5. 安装依赖

```bash
# Flash Attention (必需)
pip install flash-attn --no-build-isolation

# 或者使用已有的 anyup 环境
pip install -e .
```


```

---

## License

MIT License
