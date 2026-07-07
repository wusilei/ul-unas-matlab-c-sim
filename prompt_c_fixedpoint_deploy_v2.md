# C语言纯定点推理工程部署Prompt（v2 完整版）

## 一、工程目标

基于 MATLAB 工程路径：
```
D:\haidesi\haidesi\ul-unas-x2000-deploy\UL-UNAS_SE_FPversion_v3\UL-UNAS_SE_FPversion_v2
```

从零新建一套 **独立全新 C 纯定点推理工程**，输出存放路径：
```
D:\haidesi\haidesi\ul-unas-x2000-deploy\UL-UNAS_SE_FPversion_v3\UL-UNAS_SE_FPversion_v2\x2000_deploy_v2
```

**核心目标**：C 语言全程使用纯定点整数运算（int16_t / int32_t / int64_t 累加缓冲），**不使用 float/double 做卷积/BN/GRU 门控等主干计算的中间操作**，适配君正 X2000（MIPS32R2，无 FPU）硬件。数学运算、量化截断、移位舍入逻辑 1:1 对标 MATLAB 源码，严格对齐 MATLAB 的计算时序与截断规则。

**现阶段目标**：先完成 1:1 逻辑复刻，保证运算公式、量化规则、截断时机与 MATLAB 完全对齐。暂不处理定点累积 SNR 误差精细调优。

---

## 二、网络模型完整结构

### 2.1 推理管线总览 (Main_infer.m)

```
噪声音频 (16kHz, mono)
  ↓
STFT: 512点FFT, 256跳步, Hann窗 → [T帧, 257频点] 复数谱
  ↓
┌─ 逐帧循环 (t = 1:T) ─────────────────────────────────────┐
│                                                            │
│ ① Fix_point(real/imag, 's32f20')                           │
│    ↓                                                       │
│ ② log_gen(real, imag) → log10(√(real²+imag²)) → s32f20    │
│    输出: [1, 257]                                          │
│    ↓                                                       │
│ ③ BM_module(x, erbfc_weight) — ERB频带合并                 │
│    低频 0-2000Hz 直通 (bin 1-65)                           │
│    高频 2031.25-8000Hz 用 u16f15 权重合并 (bin 66-257→129)  │
│    输出: [1, 129]                                          │
│    ↓                                                       │
│ ④ Encoder_module (5层)                                     │
│    Layer 0 XConv:  [1,129] → [12,65]   (TConv + cTFA)     │
│    Layer 1 XMB0:   [12,65] → [24,33]   (PConv×2 + GConv + cTFA) │
│    Layer 2 XDWS0:  [24,33] → [24,33]   (PConv + GConv + cTFA)   │
│    Layer 3 XMB1:   [24,33] → [32,33]   (PConv×2 + nonGConv + cTFA) │
│    Layer 4 XDWS1:  [32,33] → [16,33]   (PConv + nonGConv + cTFA)  │
│    ↓                                                       │
│ ⑤ GDPRNN_module × 2 (分组双路径RNN, gdprnn_idx=0,1)       │
│    Intra-RNN: 分2组(各8ch) → BiGRU(nHidden=4) → FC → LN → 残差 │
│    Inter-RNN: 分2组(各8ch) → GRU(nHidden=8, 有状态缓存) → FC → LN → 残差 │
│    输入/输出: [16, 33] (转置为 [33, 16] 进入RNN)           │
│    ↓                                                       │
│ ⑥ Decoder_module (5层 + 跳跃连接)                          │
│    Layer 0 De_XDWS0: [16,33]+skip_e4 → [32,33]             │
│    Layer 1 De_XMB0:  [32,33]+skip_e3 → [24,33]             │
│    Layer 2 De_XDWS1: [24,33]+skip_e2 → [24,33]             │
│    Layer 3 De_XMB1:  [24,33]+skip_e1 → [12,65]             │
│    Layer 4 De_XConv: [12,65]+skip_e0 → [2,129]             │
│    ↓                                                       │
│ ⑦ sigmoid → u16f15                                        │
│ ⑧ BS_module — ERB频带分裂 (BM的逆过程) → [1, 257]          │
│ ⑨ MASK_module — 掩膜作用到原始复数谱 → [2, 257]            │
│                                                            │
└────────────────────────────────────────────────────────────┘
  ↓
ISTFT → 增强音频 (16kHz)
```

### 2.2 Encoder 5层详细规格

| 层 | 模块 | 子块 | 算子序列 | 时域缓存 | 维度变化 |
|---|------|------|---------|---------|---------|
| 0 | XConv | TConv | Conv2D(1→12,k=3×3,s=[1,2])→BN→AffinePReLU | 2帧(2×129) | 1×129→12×65 |
| | | cTFA_ta | Square+AvgPool(freq)→GRU(24)→FC→Sigmoid | 1帧GRU状态(1×24) | →u16f15[1,12] |
| | | cTFA_fa | Square+AvgPool(time)→pad→BiGRU(4,grouped)→FC→Sigmoid | 无 | →u16f15[1,65] |
| 1 | XMB0 | PConv0 | PConv(Cin=6→12,分组×2)→BN→AffinePReLU | 无 | 12×65→24×65 |
| | | Shuffle | interleave(1:2:end/2:2:end) | — | →24×65 |
| | | TConv | GConv(24→24,k=2×3,s=[1,2])→BN→AffinePReLU | 1帧(24×65) | →24×33 |
| | | PConv1 | PConv(12→12,分组×2)→BN→AffinePReLU | 无 | →24×33 |
| | | cTFA_ta/cTFA_fa | 同上结构, hidden dim不同 | 1帧(1×48) | →u16f15 weights |
| | | Shuffle | interleave(1:2:end/2:2:end) | — | →24×33 |
| 2 | XDWS0 | PConv | PConv(12→12,分组×2)→BN→AffinePReLU | 无 | 24×33→24×33 |
| | | Shuffle | interleave | — | →24×33 |
| | | TConv | GConv(24→24,k=2×3,s=[1,1])→BN→AffinePReLU | 1帧(24×33) | →24×33 |
| | | cTFA_ta/cTFA_fa | 同上 | 1帧(1×48) | →u16f15 weights |
| 3 | XMB1 | PConv0 | PConv(12→16,分组×2)→BN→AffinePReLU | 无 | 24×33→32×33 |
| | | Shuffle | interleave | — | →32×33 |
| | | nonTConv | nonGConv(32→32,k=1×5,s=[1,1])→BN→AffinePReLU | 无 | →32×33 |
| | | PConv1 | PConv(16→16,分组×2)→BN→AffinePReLU | 无 | →32×33 |
| | | cTFA_ta/cTFA_fa | 同上 | 1帧(1×64) | →u16f15 weights |
| | | Shuffle | interleave | — | →32×33 |
| 4 | XDWS1 | PConv | PConv(8→8,分组×2)→BN→AffinePReLU | 无 | 32×33→16×33 |
| | | Shuffle | interleave | — | →16×33 |
| | | nonTConv | nonGConv(16→16,k=1×5,s=[1,1])→BN→AffinePReLU | 无 | →16×33 |
| | | cTFA_ta/cTFA_fa | 同上 | 1帧(1×32) | →u16f15 weights |

### 2.3 Decoder 5层详细规格 (Encoder 的镜像 + skip connection)

| 层 | 模块 | 子块 | 算子序列 | 时域缓存 | 维度变化 |
|---|------|------|---------|---------|---------|
| 0 | De_XDWS0 | PConv+Shuffle+nonTConv+cTFA | 无T缓存 | 16×33→32×33 |
| 1 | De_XMB0 | PConv0+Shuffle+nonTConv+PConv1+cTFA | 无T缓存 | 32×33→24×33 |
| 2 | De_XDWS1 | PConv+Shuffle+TConv+cTFA | 24×33缓存 | 24×33→24×33 |
| 3 | De_XMB1 | PConv0+Shuffle+TConv+PConv1+cTFA | 12×33缓存 | 24×33→12×65 |
| 4 | De_XConv | TConv+cTFA | 2×129缓存 | 12×65→2×129 |

Decoder 关键特征：
- 每层先做 skip connection：`x_con = x + x_enc_skip`
- TConv 使用 `gtconv2d_func` / `non_gtconv2d_func`（转置卷积：zero-insertion + rot90 kernel）
- cTFA 结构与 Encoder 对称
- 每层结尾 Shuffle 维度各不同

### 2.4 GDPRNN 详细结构

```
输入: [16, 33] → 转置 → [33, 16]

Intra-RNN (gdprnn_idx=0/1):
  - 分组: x0=x[:,0:8], x1=x[:,8:16]
  - 每组: BiGRU(nHidden=4, Qr1=-13, Qr2=-8) → [33, 8]
  - Concat → [33, 16]
  - FC: round(x*fc_weight*2^-9) + fc_bias → [33, 16]
  - LN: 动态mean/var → normalize → scale(Qr=-14 或 -13)
  - 残差: y = x + x_ln → [33, 16]

Inter-RNN (gdprnn_idx=0/1):
  - 分组: x0=x[:,0:8], x1=x[:,8:16]
  - 每组: GRU(nHidden=8, Qr1=-13, Qr2=-8, 有状态缓存) → [33, 8]
  - Concat → [33, 16]
  - FC → LN → 残差
  - 返回: y=[33,16], h_cache=[33,16]

输出: [33, 16] → 转置 → [16, 33]
```

---

## 三、Q 格式完整定义（严格对标 Fix_point.m）

### 3.1 激活/信号的 Q 格式

| 宏定义名 | MATLAB 标识 | C 类型 | 小数位 | 缩放因子 | 使用场景 |
|----------|------------|--------|--------|---------|---------|
| `Q_ACT` | `s32f20` | `int32_t` | 20 | ×1048576 | 主干激活信号（卷积输入/输出、BN中间、残差） |
| `Q_LOG` | `u32f20` | `uint32_t` | 20 | ×1048576 | log_gen 输出, cTFA 聚合值 |
| `Q_GRU_H` | `s16f15` | `int16_t` | 15 | ×32768 | GRU/BiGRU 隐藏状态、tanh 输出 |
| `Q_SIG` | `u16f15` | `uint16_t` | 15 | ×32768 | sigmoid 输出、cTFA 注意力掩膜 |
| `Q_LN_VAR` | `u16f11` | `uint16_t` | 11 | ×2048 | LN 的 1/sqrt(var) |

### 3.2 权重的 Q 格式（从预量化 .mat 文件直接读取，禁止二次量化）

| 权重类别 | MATLAB 中的隐含格式 | C 类型 | Q 格式 | 说明 |
|---------|-------------------|--------|--------|------|
| Conv 权重 (Encoder Conv0, DeConv0) | `s16f14` | `int16_t` | Q14 | 大核 conv2d 权重 |
| PConv 权重 | `s16f13` | `int16_t` | Q13 | 1×1 逐点卷积权重 |
| GConv / nonGConv 权重 | `s16f13` | `int16_t` | Q13 | 分组/深度卷积权重 |
| TConv / nonGTConv 权重 | `s16f13` | `int16_t` | Q13 | 转置卷积权重 |
| GRU ih/hh 权重 | `s16f12` | `int16_t` | Q12 | GRU 输入/隐藏权重 |
| GRU ih/hh bias | `s16f10` | `int16_t` | Q10 | GRU bias |
| FC 权重 | `s16f13` | `int16_t` | Q13 | 全连接层权重 |
| FC bias | `s32f20` | `int32_t` | Q20 | FC bias (匹配激活格式) |
| BN weight | `s16f14` / `u16f14` | `int16_t` / `uint16_t` | Q14 | BN 缩放因子 |
| BN bias | `s32f20` | `int32_t` | Q20 | BN 偏移 |
| BN running_mean | `s32f20` | `int32_t` | Q20 | BN 均值 |
| BN running_var | `u16f11`-`u16f14` | `uint16_t` | Q11-Q14 | BN 逆标准差（不同 block Q 不同） |
| LN weight | `s16f12` | `int16_t` | Q12 | LN 缩放因子 |
| LN bias | `s16f12` | `int16_t` | Q12 | LN 偏移 |
| AffinePReLU weight | `s16f14` | `int16_t` | Q14 | PReLU 仿射权重 |
| AffinePReLU bias | `s32f20` | `int32_t` | Q20 | PReLU 仿射偏移 |
| AffinePReLU slope | `s16f13` | `int16_t` | Q13 | PReLU 负半轴斜率 |
| TA FC 权重 | `s16f13` | `int16_t` | Q13 | cTFA 时间注意力 FC |
| TA FC bias | `s32f20` | `int32_t` | Q20 | cTFA 时间注意力 FC bias |
| FA FC 权重 | `s16f13` | `int16_t` | Q13 | cTFA 频率注意力 FC |
| FA FC bias | `s32f20` | `int32_t` | Q20 | cTFA 频率注意力 FC bias |
| DPRNN FC 权重 | `s16f13` | `int16_t` | Q13 | DPRNN FC |
| DPRNN FC bias | `s32f20` | `int32_t` | Q20 | DPRNN FC bias |
| erbfc/ierbfc weight | `u16f15` | `uint16_t` | Q15 | ERB 合并/分裂矩阵 |

### 3.3 每层 Qr 移位参数配置

MATLAB 每个算子的最后一个参数 `Qr` 决定了 `round(x*w*2^Qr)` 中的右移量（负值表示右移）。以下是各算子的 Qr 配置规律：

| 算子类型 | Qr 典型值 | 含义（MATLAB 表达式） |
|---------|----------|---------------------|
| conv2d_func | -14 | `round(x_kernel .* kernel_chan * 2^(-14))` |
| pconv2d_func | -13 或 -14 | `round(x_chan * kernel_chan * 2^(Qr))` |
| gconv2d_func | -13 或 -14 | 同上 |
| non_gconv2d_func | -13 | 同上 |
| tconv2d_func | -13 | 同上 |
| non_gtconv2d_func | -13 | 同上 |
| bn_func Qr1 | -11 或 -14 | `round((x-mean) * var * 2^(Qr1))` |
| bn_func Qr2 | -14 | `round(x_norm * weight * 2^(Qr2))` |
| ln_func Qr | -13 或 -14 | `round(x_norm * weight * 2^(Qr))` |
| affineprelu Qr1 | -13 | `round(x_neg * slope * 2^(Qr1))` |
| affineprelu Qr2 | -13 | `round(x * weight * 2^(Qr2))` |
| GRU Qr1 (ih) | -13 | `round(x_t * ih_weight * 2^(-13))` |
| GRU Qr2 (hh) | -8 | `round(h_cache * hh_weight * 2^(-8))` |
| GRU Q_sig_tanh | -15 | `round(z_t .* h_cache * 2^(-15))` |
| cTFA attention | -15 | `round(tconv .* ta' * 2^(-15))` |
| BM/BS | -15 | `round(x * weight * 2^(-15))` |
| MASK | -15 | `round(x_real .* x_mask * 2^(-15))` |
| DPRNN FC | -9 | `round(x_gru * fc_weight * 2^(-9))` |
| cTFA FC | -8 或 -9 | `round(x_gru * fc_weight * 2^(-8))` |

**关键要求**：`extract_weights.m` 导出权重时，必须同时导出一个 `qr_config.h` 头文件，将上述每个算子的 {Qr1, Qr2} 定义为编译期宏常量，供 C 代码直接使用。

---

## 四、张量存储排布

### 4.1 通用规则

- **2D 张量** `[C, W]`（channels × width），C 语言中行主序存储：`data[c * W + w]`
- **3D 张量** `[C, H, W]`（如 tconv2d_func 的输入/权重），展开为 `data[c * H * W + h * W + w]`
- **权重张量** `[Cout, Cin, Kh, Kw]`，展开为 `weight[co * Cin * Kh * Kw + ci * Kh * Kw + kh * Kw + kw]`

### 4.2 缓存张量

| 缓存名称 | MATLAB 维度 | C 类型 | 大小 |
|---------|------------|--------|------|
| conv_cache_e0 | [2, 129] | int32_t | 258 |
| conv_cache_e1 | [24, 65] | int32_t | 1560 |
| conv_cache_e2 | [24, 33] | int32_t | 792 |
| conv_cache_d0 | [24, 33] | int32_t | 792 |
| conv_cache_d1 | [12, 33] | int32_t | 396 |
| conv_cache_d2 | [12, 2, 65] | int32_t | 1560 |
| tfa_cache_e0 | [1, 24] | int16_t | 24 |
| tfa_cache_e1 | [1, 48] | int16_t | 48 |
| tfa_cache_e2 | [1, 48] | int16_t | 48 |
| tfa_cache_e3 | [1, 64] | int16_t | 64 |
| tfa_cache_e4 | [1, 32] | int16_t | 32 |
| tfa_cache_d0-d4 | 对称 | int16_t | 同上 |
| inter_cache_0/1 | [33, 16] | int16_t | 528 each |

### 4.3 Shuffle 操作规范

MATLAB 中的 interleave/deinterleave 操作：
```matlab
% Shuffle: interleave
y_s(1:2:end, :) = y_pconv(1:N/2, :);   % 前半放奇数行
y_s(2:2:end, :) = y_pconv(N/2+1:end, :); % 后半放偶数行
```

C 语言实现规范：
- 使用 `memcpy` + 循环步长实现
- 提供通用 `shuffle_interleave(int32_t *dst, int32_t *src, int C, int W)` 和 `shuffle_deinterleave` 函数
- 确保不同 C/W 维度下均正确

---

## 五、算子定点实现规范（逐行对标 MATLAB）

### 5.1 卷积类算子

#### conv2d_func — 标准 2D 卷积
```c
// MATLAB: temp = round(x_kernel .* kernel_chan * 2^(Qr));
//         conv_result(h_id, w_id) = sum(temp, 'all');
// C 实现:
//   int64_t acc = 0;
//   for each kernel element:
//       acc += (int32_t)x_pixel * (int32_t)w_pixel;
//   conv_result = (int32_t)( (acc + round_const) >> (-Qr) );
//   // round_const = 1 << (-Qr - 1)
```
关键规则：
- 累加使用 **int64_t** 防止溢出
- 完成全部乘累加后再做 **一次性 round + 右移截断**
- **不采用逐元素即时截断**，匹配 MATLAB 的 `sum(round(...))` → MATLAB 是先 round 后 sum，所以实际上是逐元素 round 再 sum。仔细看 MATLAB: `temp = round(x_kernel.*kernel_chan*2^(Qr)); conv_result = sum(temp,'all');` — 是先 round 再 sum。C 实现应该先对每个乘法结果做 round+shift，再累加。

**修正理解**：重新审视 MATLAB 代码：
```matlab
temp = round( x_kernel.*kernel_chan*2^(Qr) );
conv_result(h_id,w_id) = sum(temp,'all');
```
MATLAB 是逐元素 round 后再累加。C 端应匹配：
```c
for each pixel:
    int64_t prod = (int64_t)x * w;
    int32_t rounded = (int32_t)((prod + round_const) >> (-Qr));  // 或 prod >> (-Qr) 配合舍入
    acc += rounded;
result = (int32_t)acc + bias;
```

#### pconv2d_func — 逐点卷积 (1×1)
```matlab
% MATLAB: conv_result = round(x_chan * kernel_chan * 2^(Qr));
```
C 实现：输入 [Cin, W]，权重 [Cout, Cin]，对每个输出通道做内积累加。

#### gconv2d_func — 分组时域卷积（有缓存）
- 每个输出通道独立处理
- 输入 = [cache[nOut]; x[nOut]]（沿时间维度拼接）
- 左右各 pad 2 列零 → 滑动 kernel → round + sum
- 更新 cache = x（当前帧作为下一帧的 cache）

#### non_gconv2d_func — 分组非时域卷积（无缓存）
- 与 gconv2d_func 类似，但无时间 cache 拼接
- kernel 沿频率维度滑动（1×5 核）

#### tconv2d_func — 转置卷积
- Zero-insertion: `x_insert(1:stride:end, 1:stride:end) = x_chan`
- kernel 做 rot90 旋转
- 标准 conv 滑动 → round + sum

#### non_gtconv2d_func — 分组非时域转置卷积
- Zero-insertion + kernel rot90（注意：MATLAB 用的是 rot90(kernel, 90)，即 90° 而非 180°）
- 其余同 non_gconv2d_func

### 5.2 归一化算子

#### bn_func — Batch Normalization
```matlab
% MATLAB:
%   x_norm = round((x - running_mean) .* running_var * 2^(Qr1));
%   y = round(x_norm .* weight * 2^(Qr2)) + bias;
```
C 实现：
```c
// x, running_mean, bias: int32_t (Q20)
// running_var: uint16_t (Q11-Q14, varies per block)
// weight: int16_t/uint16_t (Q14)
for (i = 0; i < N; i++) {
    int64_t diff = (int64_t)x[i] - running_mean[i];
    int64_t norm = diff * (int64_t)running_var[i];
    int32_t x_norm = (int32_t)((norm + round1) >> (-Qr1));
    int64_t scaled = (int64_t)x_norm * weight[i];
    int32_t y = (int32_t)((scaled + round2) >> (-Qr2)) + bias[i];
}
```
**关键**：不同 block 的 BN 使用不同的 Qr1/Qr2，需要从 `qr_config.h` 读取。

#### ln_func — Layer Normalization
```matlab
% MATLAB:
%   x_dq = x * 2^(-20);              % 反量化到 float
%   running_mean = mean(x_dq, 'all');
%   running_var = 1 / sqrt(var(x_dq) + 1e-8);
%   running_mean = Fix_point(running_mean, 's32f20');
%   running_var = Fix_point(running_var, 'u16f11');
%   x_norm = round((x - running_mean) * running_var * 2^(-11));
%   y = round(x_norm .* weight * 2^(Qr)) + bias;
```
C 实现策略：
- 均值和 1/sqrt(var) 的计算**允许使用浮点中间运算**（LN 是运行时动态统计量，非批量处理），或使用定点牛顿迭代法
- 推荐方案：对 LN 输入做 int32→float 反量化、计算统计量、float→定点重量化，再执行定点归一化
- 备选方案：定点 mean 用 int64 累加再除法，1/sqrt(var) 用 256 点 Q11 LUT + 牛顿迭代

### 5.3 激活函数算子

#### affineprelu_func — Affine PReLU
```matlab
% x_copy = x;
% index = x < 0;
% x(index) = round(x(index) .* slope(row) * 2^(Qr1));  % 负半轴PReLU
% y = round(x_copy .* weight * 2^(Qr2)) + bias + x;     % affine + 残差
```
注意：PReLU slope 的索引按 row（channel），不是逐元素。

#### sigmoid — Q15 定点 LUT
MATLAB 中的模式：
```matlab
r_t_dq = r_t * 2^(-20);      % 反量化
r_t_dq = sigmoid_func(r_t_dq); % float sigmoid
r_t = Fix_point(r_t_dq, 'u16f15'); % 重新量化
```
C 实现方案：
- 预生成 s32f20→u16f15 的 sigmoid LUT（建议 1024 点，在 MATLAB 端生成并导出为 C 头文件）
- 使用对称查表 + 线性插值
- 输入 int32_t (Q20)，输出 uint16_t (Q15)
- 同理 tanh 输出 int16_t (Q15)

**LUT 生成方法**（MATLAB 端）：
```matlab
% 在 export_all_layers.m 中生成
function gen_sigmoid_lut(N)
    x = linspace(-8, 8, N);  % sigmoid 有效范围
    y = 1 ./ (1 + exp(-x));
    y_q = round(y * 32768);  % u16f15
    % 导出为 C 数组
end
```

#### log_gen — 对数幅度压缩
```matlab
% mag = sqrt(x_real_dq.^2 + x_imag_dq.^2);  % float
% clamped = max(mag, 1e-12);
% y = log10(clamped);
% y = Fix_point(y, 's32f20');
```
C 实现方案：
- 定点计算 mag² = real² + imag² (int64 防止溢出)
- sqrt: 使用整数 sqrt 算法（如二分查找或 Newton 迭代），输入 Q40，输出 Q20
- log10: 使用 512 点 LUT + 线性插值
- 或者：使用快速反平方根近似 + 查表组合

### 5.4 RNN 算子

#### GRU_module — 单向 GRU
```matlab
% 重置门: r_t = round(x_t*ih_r_w*2^(Qr1)) + round(h* hh_r_w*2^(Qr2)) + ih_r_b + hh_r_b
%   → sigmoid → u16f15
% 更新门: z_t = round(x_t*ih_z_w*2^(Qr1)) + round(h* hh_z_w*2^(Qr2)) + ih_z_b + hh_z_b
%   → sigmoid → u16f15
% 候选态: h_t = round(h*hh_n_w*2^(Qr2)) + hh_n_b
%         n_t = round(x_t*ih_n_w*2^(Qr1)) + round(r_t.*h_t*2^(-15)) + ih_n_b
%   → tanh → s16f15
% 隐态更新: h = round((32768-z_t).*n_t*2^(-15)) + round(z_t.*h*2^(-15))
```
C 实现要点：
- x_t 输入为 int32_t (Q20)，h_cache 为 int16_t (Q15)
- ih 路径：x_t * w (Q20 × Q12) → 累加 → round >> 13
- hh 路径：h_cache * w (Q15 × Q12) → 累加 → round >> 8
- 门控计算后通过 sigmoid LUT 得到 uint16_t (Q15)
- 候选态通过 tanh LUT 得到 int16_t (Q15)
- 隐态更新中的 32768 是 Q15 的 1.0 值，`32768 - z_t` 在 uint16_t 下等价于 `1 - z_t`

#### BiGRU_module — 双向 GRU
- 正向 GRU: 逐时间步循环，h_cache 前向传递
- 反向 GRU: x_re = x(end:-1:1,:)，逐时间步循环
- 最终 concat: `y = cat(2, y1, y2(end:-1:1,:))`（注意反向输出需要再翻转回来）
- 无全局状态缓存（每帧独立计算）

#### Intra_RNN_module — GDPRNN 帧内 RNN
- 分组: x0 = x[:, 0:8], x1 = x[:, 8:16]
- 每组: BiGRU(nHidden=4)
- Concat → FC(Qr=-9) → LN(Qr=-14) → 残差

#### Inter_RNN_module — GDPRNN 帧间 RNN
- 分组: x0 = x[:, 0:8], x1 = x[:, 8:16]
- 每组: GRU(nHidden=8, 有状态缓存)
- Concat → FC(Qr=-9) → LN(Qr=-13) → 残差

### 5.5 ERB 与 MASK 算子

#### BM_module — 频带合并
```matlab
% y(1:65) = x(1:65);                  % 低频直通
% y(66:129) = round(x(66:257) * weight * 2^(-15));  % 高频合并
```
weight 维度: [257-65, 129-65] = [192, 64]，uint16_t (Q15)

#### BS_module — 频带分裂
```matlab
% y(1:65) = x(1:65);                  % 低频直通
% y(66:257) = round(x(66:129) * weight * 2^(-15));  % 高频分裂
```
weight 维度: [129-65, 257-65] = [64, 192]，uint16_t (Q15)

#### MASK_module — 掩膜作用
```matlab
% y_real = round(x_real .* x_mask * 2^(-15));
% y_imag = round(x_imag .* x_mask * 2^(-15));
% y = cat(1, y_real, y_imag) * 2^(-20);  % 输出反量化
```
C 实现：输入 x_mask uint16_t (Q15)，x_real/x_imag int32_t (Q20)

### 5.6 cTFA 时频注意力（所有 Encoder/Decoder 层通用）

每层 cTFA 包含 TA（时间注意力）和 FA（频率注意力）两个分支：

**TA 分支** (cTFA_ta_module):
```matlab
% x_dq = x * 2^(-20);
% x_squared = x_dq.^2;           % float 平方
% x_agg = mean(x_squared, 2);    % 沿频率维平均
% x_t = Fix_point(x_agg', 'u32f20');
% [x_gru, h_cache] = GRU_module(x_t, nHidden, h_cache, ...);
% x_fc = round(x_gru * ta_fc_weight * 2^(-8)) + ta_fc_bias;
% → sigmoid → u16f15
```
C 实现：
- 平方+均值聚合：**允许使用浮点中间运算**（此步骤在 MATLAB 中也反量化到 float），或使用 int64 累加 + 整数除法
- TA GRU(nHidden=24/48/64/32)：输入 uint32_t (Q20)，有状态缓存 h_cache int16_t (Q15)
- TA FC → sigmoid LUT

**FA 分支** (cTFA_fa_module):
```matlab
% x_dq = x * 2^(-20);
% x_squared = x_dq.^2;           % float 平方
% x_agg = mean(x_squared, 1);    % 沿时间维平均
% x_agg = Fix_point(x_agg, 'u32f20');
% pad_len = 3; x_pad = [x_agg zeros(1, pad_len)];
% x_t = reshape(x_pad, [4, 17])';
% → BiGRU(nHidden=4, 正向+反向)
% → FC(Qr=-9) → reshape → 去pad → sigmoid → u16f15
```
C 实现：
- 重塑维度: 将 [1, W+pad] 重塑为 [group=4, seg=17]
- FA BiGRU(nHidden=4)：沿 17 个时间步循环，正向+反向
- FA FC → sigmoid LUT

**cTFA 最终融合**：
```matlab
y_t = round(tconv .* ta' * 2^(-15));   % TA 注意力
y = round(y_t .* fa * 2^(-15));        % FA 注意力
```

---

## 六、非线性函数定点 LUT 策略

### 6.1 sigmoid LUT
- 输入范围: int32_t Q20，等效 float [-8, 8]
- 输出: uint16_t Q15 [0, 32768]
- LUT 大小: 1024 点（索引用输入 >> 11 取高 10bit）
- 线性插值: 用低 11bit 做分数插值
- 边界饱和: 输入 > 8 → 输出 32768; 输入 < -8 → 输出 0

### 6.2 tanh LUT
- 输入范围: int32_t Q20，等效 float [-4, 4]
- 输出: int16_t Q15 [-32768, 32767]
- LUT 大小: 1024 点
- 线性插值 + 边界饱和

### 6.3 log10 LUT
- 输入: uint32_t Q20（clamped magnitude, ≥ 1e-12×2^20 ≈ 1.05×10^-6 × 2^20 ≈ 1）
- 使用 512 点 LUT + 线性插值
- 或者使用快速 log2 近似 + 换底公式

### 6.4 定点 sqrt
- 对 mag²（Q40 格式的 uint64_t）做整数 sqrt
- 使用二分查找或 Newton 迭代
- 结果: uint32_t Q20

### 6.5 LUT 生成与导出
在 `export_all_layers.m` 中自动生成所有 LUT，导出为 `ulunas_lut.h`：
```c
// ulunas_lut.h
#define SIGMOID_LUT_SIZE 1024
#define TANH_LUT_SIZE 1024
#define LOG10_LUT_SIZE 512

extern const uint16_t sigmoid_lut_q15[SIGMOID_LUT_SIZE];
extern const int16_t tanh_lut_q15[TANH_LUT_SIZE];
extern const int32_t log10_lut_q20[LOG10_LUT_SIZE];

// 查表函数声明
uint16_t sigmoid_q20_to_q15(int32_t x_q20);
int16_t tanh_q20_to_q15(int32_t x_q20);
int32_t log10_q20(int32_t x_q20);
uint32_t sqrt_q40_to_q20(uint64_t x_q40);
```

---

## 七、权重导出脚本规范 (extract_weights.m)

### 7.1 核心原则
1. **预量化权重** (`para_in_mat_FP/` 目录下 409 个 .mat 文件) 已经是在 MATLAB 端完成量化的定点值，**C 端直接使用，禁止二次量化**
2. 自动区分各权重的数据类型：
   - `uint16_t`: ERB 权重 (erbfc/ierbfc)
   - `int16_t`: 卷积/GRU/FC/BN/PReLU 权重
   - `int32_t`: BN/FC/Affine bias, BN running_mean
   - `uint16_t`: BN running_var, LN running_var
3. 完整导出 BN 全套参数（weight, bias, running_mean, running_var），无遗漏
4. 检查数值饱和情况（int16 范围 [-32768, 32767]，uint16 范围 [0, 65535]）

### 7.2 extract_weights.m 功能清单
```
① 遍历 para_in_mat_FP/ 下所有 .mat 文件
② 按文件名中的模块前缀分类（encoder_en_convs_X, decoder_de_convs_X, dpgrnn_X, erb_*）
③ 识别每个参数的 MATLAB 类型（squeeze 后的维度判断是 weight/bias/mean/var/slope）
④ 写入对应的 C 数组文件（ulunas_matlab_weights.h + ulunas_matlab_weights.c）
⑤ 同时写入 const 声明、维度信息、Q 格式注释
⑥ 生成 qr_config.h（每层 Qr 配置宏）
⑦ 生成 layer_dims.h（每层 Cin/Cout/kernel/stride/W 维度宏）
```

### 7.3 权重 C 文件输出格式
```c
// ulunas_matlab_weights.h
typedef struct {
    const int16_t *data;
    int Cout;
    int Cin;
    int Kh;
    int Kw;
} conv_weight_t;

extern const conv_weight_t encoder_tconv_0_weight;
extern const int32_t encoder_tconv_0_bias[12];
// ... 所有层的声明
```

### 7.4 qr_config.h 输出格式
```c
// qr_config.h — 每层算子的 Qr 移位配置（从 MATLAB 源码提取）

// Encoder Layer 0 (XConv)
#define E0_TCONV_CONV_QR          (-14)
#define E0_TCONV_BN_QR1           (-14)
#define E0_TCONV_BN_QR2           (-14)
#define E0_TCONV_AFFINE_QR1       (-13)
#define E0_TCONV_AFFINE_QR2       (-13)
#define E0_CTFA_TA_GRU_QR1        (-13)
#define E0_CTFA_TA_GRU_QR2        (-8)
#define E0_CTFA_TA_FC_QR          (-8)
#define E0_CTFA_FA_GRU_QR1        (-13)
#define E0_CTFA_FA_GRU_QR2        (-8)
#define E0_CTFA_FA_FC_QR          (-9)
// ... 所有层的 Qr 配置
```

---

## 八、工程文件清单

### 8.1 C 源文件

| 文件名 | 内容 | 预估行数 |
|--------|------|---------|
| `ulunas_fp.h` | Q 格式宏、维度常量、算子声明、模型状态结构体、LUT 声明 | ~350 |
| `ulunas_fp.c` | 全套定点算子底层实现（conv/tconv/gconv/pconv/BN/LN/GRU/BiGRU/cTFA/log_gen/BM/BS/MASK/sigmoid_lut/tanh_lut） | ~2500 |
| `ulunas_modules.c` | 封装 Encoder Layer 0-4、Decoder Layer 0-4、GDPRNN、Intra/Inter RNN 独立子模块 | ~1500 |
| `ulunas_infer.c` | 单帧顶层完整推理管线（ulunas_infer_frame 函数） | ~400 |
| `ulunas_lut.h` | 非线性函数 LUT 声明 + 查表函数 | ~50 |
| `ulunas_lut.c` | 非线性函数 LUT 数据表（sigmoid/tanh/log10/sqrt） | ~200 |
| `ulunas_matlab_weights.h` | 所有权重/偏置的外部声明 + 结构体定义 | ~500 |
| `ulunas_matlab_weights.c` | 权重数据占位（或直接嵌入导出的 C 数组） | ~2000+ |
| `qr_config.h` | 每层 Qr 配置宏（由 extract_weights.m 生成） | ~150 |
| `layer_dims.h` | 每层维度/卷积参数宏（由 extract_weights.m 生成） | ~200 |
| `test_matlab_golden.c` | 逐层 golden 二进制比对 + SNR 计算测试程序 | ~600 |
| `Makefile` | PC gcc + 君正 X2000 mips交叉编译 | ~80 |

### 8.2 MATLAB 脚本

| 文件名 | 内容 |
|--------|------|
| `extract_weights.m` | 权重导出主脚本 |
| `export_all_layers.m` | 逐层 golden 真值导出 + LUT 生成 |
| `gen_lut_tables.m` | sigmoid/tanh/log10 LUT 预生成 |

### 8.3 复用模块

从旧版 `UL-UNAS_SE_FPversion_v2\c_version\x2000_deploy_v1\` 目录复用：
- `fft_q15.h` — Q15 定点 FFT（需要适配接口：输入从 Q15 桥接到 Q20）
- `noise_reduction_q15.c` — ISTFT/OLA 相关函数（需评估是否适用于当前版本）

**注意**：复用前需审查接口 Q 格式是否兼容。MATLAB STFT 输入/输出使用 s32f20，旧版 FFT 使用 Q15，需要插入格式转换层。

---

## 九、模型状态结构体

```c
// ulunas_fp.h
typedef struct {
    // Temporal convolution caches (Encoder)
    int32_t conv_cache_e0[2 * 129];
    int32_t conv_cache_e1[24 * 65];
    int32_t conv_cache_e2[24 * 33];

    // Temporal convolution caches (Decoder)
    int32_t conv_cache_d0[24 * 33];
    int32_t conv_cache_d1[12 * 33];
    int32_t conv_cache_d2[12 * 2 * 65];

    // cTFA TA GRU hidden state caches (Encoder)
    int16_t tfa_cache_e0[24];
    int16_t tfa_cache_e1[48];
    int16_t tfa_cache_e2[48];
    int16_t tfa_cache_e3[64];
    int16_t tfa_cache_e4[32];

    // cTFA TA GRU hidden state caches (Decoder)
    int16_t tfa_cache_d0[64];
    int16_t tfa_cache_d1[48];
    int16_t tfa_cache_d2[48];
    int16_t tfa_cache_d3[24];
    int16_t tfa_cache_d4[2];

    // GDPRNN Inter-RNN hidden state caches
    int16_t inter_cache_0[33 * 16];
    int16_t inter_cache_1[33 * 16];

} ulunas_state_t;

// 初始化函数
void ulunas_state_init(ulunas_state_t *state);
```

---

## 十、测试与验证策略

### 10.1 test_matlab_golden.c 功能
1. 读取 MATLAB 导出的 golden 二进制文件（逐层中间输出）
2. 运行 C 推理，逐层比对输出
3. 计算每层：MAX absolute error, MEAN absolute error, SNR (dB)
4. 输出通过/失败状态

### 10.2 MATLAB golden 导出 (export_all_layers.m)
- 在 Main_infer.m 的每个子模块后插入 save 语句
- 导出格式：`int32_t` 或 `int16_t` 二进制原始数据（与 C 类型一致）
- 文件命名：`golden_e0_tconv.bin`, `golden_e0_ctfa_ta.bin`, ...

### 10.3 误差判定标准
- 初期目标：逐层定点运算与 MATLAB 输出 **完全一致**（bit-exact）
- 对于 sigmoid/tanh LUT 查表，允许 ±1 LSB 误差
- 对于 sqrt 迭代，允许 ±2 LSB 误差

---

## 十一、实施顺序建议

| 阶段 | 任务 | 验证方式 |
|------|------|---------|
| Phase 1 | 编写 extract_weights.m + export_all_layers.m，验证权重导出正确性 | MATLAB 端检查 |
| Phase 2 | 实现 ulunas_fp.c 中的基础算子（conv/pconv/bn/affineprelu/sigmoid_lut） | 单算子 golden 比对 |
| Phase 3 | 实现 RNN 算子（GRU/BiGRU）+ LN | 单算子 golden 比对 |
| Phase 4 | 实现 cTFA + BM/BS/MASK/log_gen + sqrt | 单模块 golden 比对 |
| Phase 5 | 组装 Encoder 5层 + Decoder 5层 (ulunas_modules.c) | 逐层 golden 比对 |
| Phase 6 | 组装 GDPRNN + 完整管线 (ulunas_infer.c) | 端到端 golden 比对 |
| Phase 7 | 集成 FFT/ISTFT，完成音频输入→输出 | 整轨音频 SNR 评估 |
| Phase 8 | 君正 X2000 交叉编译 + 性能优化 | 实时率测试 |

---

## 十二、项目约束

1. **完全新建独立工程**，不复用、不修改历史存在截断/量化/类型 bug 的旧 C 代码（旧版 `UL-UNAS_SE_FPversion_v2\c_version\` 仅参考，不直接复制）
2. **现阶段仅保证运算逻辑、量化规则、截断时机与 MATLAB 完全对齐**，暂不处理定点累积 SNR 误差精细调优
3. **所有权重数据直接从 para_in_mat_FP/ 读取**，不使用 Python 或其他中间工具导出
4. **遵守 MATLAB 源码中的 Shuffle 语义**：每层的 interleave/deinterleave 分组方式各不相同，需对号入座
5. **转置卷积的 kernel 旋转方向**：tconv 用 rot90(kernel, 2) = 180°，non_gtconv 用 rot90(kernel, 90) = 90°，注意区分
6. **LUT 统一在 MATLAB 端生成**，不依赖运行时初始化
