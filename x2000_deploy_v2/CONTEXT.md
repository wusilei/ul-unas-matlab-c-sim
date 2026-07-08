# 新对话启动上下文 — UL-UNAS MATLAB→C 定点转换工程

## 工程路径
```
/media/sf_haidesi/haidesi/ul-unas-x2000-deploy/UL-UNAS_SE_FPversion_v3/UL-UNAS_SE_FPversion_v2/x2000_deploy_v2/
```

## 启动新对话的提示词
把下面这段发给 Claude：
"""
继续 UL-UNAS MATLAB→C 定点转换工程。
先读 x2000_deploy_v2/CONTEXT.md 了解项目状态，然后继续工作。
"""

## 编译
```
cd x2000_deploy_v2/
make            # PC (gcc)
make TARGET=x2000  # X2000 MIPS32R2
```

## Git
```
https://github.com/wusilei/ul-unas-matlab-c-sim
```

## 当前状态 (2026-07-08 最终)

### 编译: ✅ PC + X2000 均零错误

### 已修复 Bug 清单 (16个)

| # | 文件 | Bug | 修复 |
|---|------|-----|------|
| 1 | bn_fp | running_mean 按 N=C×W 索引 | `c = i / W` 逐信道 |
| 2 | ctfa_fa_fp | x_re[68] 栈硬编码太小 | malloc(nseg*ngrp) |
| 3 | gru_step_fp | r_t_q15[32] 硬编码 | 改为[64] |
| 4 | ctfa_ta_fp | 错误第一版实现 | 删除重写 |
| 5 | gtconv2d_fp | W公式算错 | 修正公式 |
| 6 | ulunas_modules.c | D4 TA GRU input_dim=1但传C=2 | 增加input_dim参数 |
| 7 | bn_fp | (int16_t)running_var 破坏>32767 | (uint32_t)running_var |
| 8 | ctfa_fa_fp | FA reshape grp*nseg+seg 映射反向 | grp+ngrp*seg |
| 9 | gru_step_fp | r_t/z_t int16_t 但sigmoid返uint16_t | 改为uint16_t |
| 10 | gconv/tconv/gtconv/non_gconv/non_gtconv | 权重行优先→列优先 | 修正索引公式 |
| 11 | pconv2d_fp | grouped conv权重stride=Cout→Co*nGroups | 增加wstride参数 |
| 12-13 | pconv_g2_aff/bn | 组1 offset Co→Co*Ch回退 | layout [Co,nGroups,Ch] 非 [Co,Ch,nGroups] |
| 14 | pconv2d_fp | wstride参数 | 添加wstride=Co*2 |
| 15 | E1 pconv_g2_bn | BN Qr1=-11→-14 (MATLAB说-14) | E1 XMB0 PConv_block_1 修正 |
| 16 | bn_fp | BN weight声明为int16_t但有值>32767 | weight改为const uint16_t* |

注: Bug 16 — D1 pconv BN weight[19]=33101, E1 pconv BN weight[13]=47668; int16_t读取→符号扩展→符号翻转

### 当前 Golden SNR 状态

| Test | SNR | 说明 |
|------|-----|------|
| BM | 146.83 dB | ✓ |
| E0 XConv (独立) | 70.98 dB | △ GRU LUT 精度瓶颈 |
| Full Encoder E4 | 1.76 dB | △ 5层级联累积 |
| DPRNN idx=0 | 1.81 dB | △ 依赖Encoder输出 |
| DPRNN idx=1 | 5.28 dB | △ |
| Decoder (真实输入) | -4.66 dB | ✗ 误差传播 |
| Decoder (全golden输入) | 14.91 dB | △ 纯Decoder问题 |

### 剩余瓶颈: GRU/BiGRU 精度 (Bug 17)

**E0 诊断**: 用 golden TA+FA masks = **999 dB** (ctfa_apply完美)。用 C 计算的 TA+FA = **70.98 dB** → 差在 GRU 隐态。

**根因定位**: tanh LUT 范围不足
- sigmoid LUT: [-8, 8], 1024 pts, step≈0.0156 → 全覆盖
- **tanh LUT: [-4, 4], 1024 pts, step≈0.0078 → 漏了 [-8,-4) 和 (4,8]**
- tanh(-4) = -32747 (Q15), tanh(-5.787) = -32767 — 但 LUT 把 <-4 全饱和到 -32768
- GRU 中 n_t 可达 -5.787 (Q20), tanh 输出差 2 LSB → 隐态更新累积

**修复**: 在 MATLAB 端运行 `gen_lut_tables.m`，将 tanh 范围扩展为 [-8, 8]（与 sigmoid 一致）。

**长期方向**: Q15 隐态精度理论上限约 84 dB。要达到 130+ dB，需要将 GRU 隐态从 Q15→Q20 (int16→int32)，同步升级 sigmoid/tanh LUT 到 Q20。

### 诊断工具

| 工具 | 用途 |
|------|------|
| `diag_e0.m` + `diag_e0_ctfa.m` | E0 逐子操作导出到 diag_e0/ |
| `diag_decoder.m` | D0-D4 逐子操作导出到 diag_decoder/ |
| `test_matlab_golden.c` | 逐层 binary golden 比对 (make test) |
| `test_decoder_diag.c` | 逐子操作 text golden 比对 (make diag) |

### 下一步

1. **MATLAB**: 运行 gen_lut_tables.m 重新生成 tanh LUT (范围[-8,8])
2. **C**: 重新编译，跑 `make test` 和 `make diag`，确认 GRU 精度改善
3. 如果 E0 > 80 dB: DPRNN/Decoder 可能自然改善 (依赖GRU的层很多)
4. 长远: 评估 Q15→Q20 GRU 隐态升级的工程投入

---

## 核心文件速查

| 文件 | 说明 |
|------|------|
| ulunas_fp.h | Q格式、状态结构体、算子声明 |
| ulunas_fp.c | **全部定点算子** |
| ulunas_modules.c | 10层 Encoder/Decoder + GDPRNN |
| ulunas_infer.c | 单帧推理管线 |
| ulunas_stft.c | Q15 FFT/STFT/ISTFT |
| ulunas_matlab_weights.h/c | 409个权重张量(1.4MB, MATLAB生成) |
| ulunas_lut.h/c | sigmoid/tanh/log10/sqrt LUT |
| layer_dims.h | 维度宏 |
| qr_config.h | Qr移位配置宏 |
| test_matlab_golden.c | Golden比对测试 |
| diag_e0/ | MATLAB导出的E0中间值(文本) |
| golden/ | MATLAB导出的逐层golden(二进制) |
| CONTEXT.md | 本文件 |

## 模型架构

```
STFT(512pt,256hop) → log_gen → BM(257→129) → Encoder×5 → GDPRNN×2 → Decoder×5(+skip) → sigmoid → BS(129→257) → MASK → ISTFT
```

| 层 | 输入 | 输出 | 关键算子 |
|---|------|------|---------|
| E0 XConv | [1,129] | [12,65] | Conv2D→BN→AffinePReLU+cTFA |
| E1 XMB0 | [12,65] | [24,33] | PConv×2→Shuffle→GConv(stride=2)→PConv×2+cTFA→Shuffle |
| E2 XDWS0 | [24,33] | [24,33] | PConv→Shuffle→GConv(stride=1)+cTFA |
| E3 XMB1 | [24,33] | [32,33] | PConv×2→Shuffle→nonGConv→PConv×2+cTFA→Shuffle |
| E4 XDWS1 | [32,33] | [16,33] | PConv→Shuffle→nonGConv+cTFA |
| D0-D4 | — | — | Decoder 5层(Encoder镜像+skip connection) |
| D4 De_XConv | [12,65]+skip | **[1,129]** | ← 单通道! |

## Q格式

| 格式 | C类型 | 小数位 | 使用场景 |
|------|-------|--------|---------|
| s32f20 | int32_t | 20 | 主干激活信号 |
| u32f20 | uint32_t | 20 | log_gen, cTFA聚合 |
| s16f15 | int16_t | 15 | GRU隐态, tanh输出 |
| u16f15 | uint16_t | 15 | sigmoid输出, 注意力掩膜 |
| u16f11 | uint16_t | 11 | LN running_var |

## 关键技术要点 (容易踩坑)

1. **所有权重数组是 MATLAB 列优先存储** — C代码必须用列优先索引
2. **weight 4D [Cout,Cin,Kh,Kw]**: `widx = co + Cout*ci + Cout*Cin*kh + Cout*Cin*Kh*kw`
3. **tconv weight [Cin,Cout,Kh,Kw]**: `widx = ci + Cin*co + Cin*Cout*kh + Cin*Cout*Kh*kw`
4. **PConv grouped [Co,nGroups,Ch]**: wstride = Co*nGroups (不是Co!)
5. **GRU ih_weight [input_dim, 3*nHidden]**: `IH_R_W(i,j) = weight[i + j*input_dim]`
6. **BN running_mean/var 逐信道**: `c = i / W`
7. **sigmoid返回 uint16_t** (0-32768)，GRU中需用uint16_t，不能用int16_t
8. **running_var** 有值 >32767，bn_fp中必须用uint32_t
9. **FA reshape**: MATLAB `reshape(x,[ngrp,nseg])'` = `A(grp,seg)` → `B(seg,grp)`
10. **D4 输出 [1,129]** 单通道，不是 [2,129]
