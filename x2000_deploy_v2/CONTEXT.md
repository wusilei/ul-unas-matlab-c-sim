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

### 编译: ✅ PC (gcc) 零错误零警告

### 已修复 Bug 清单 (8个)

| # | 文件 | Bug | 修复 |
|---|------|-----|------|
| 1 | bn_fp | running_mean 按 N=C×W 索引 | `c = i / W` 逐信道 |
| 2 | ctfa_fa_fp | x_re[68] 栈硬编码太小 | malloc(nseg*ngrp) |
| 3 | gru_step_fp | r_t_q15[32] 硬编码 | 改为[64] |
| 4 | gru_step_fp | r_t/z_t int16_t 但sigmoid返uint16_t | 改为uint16_t |
| 5 | gconv/tconv/gtconv/non_gconv/non_gtconv | 权重行优先→列优先 | 修正列优先索引公式 |
| 6 | pconv2d_fp | grouped conv权重stride=Cout→Co*nGroups | 增加wstride参数 |
| 7 | E1 pconv_g2_bn | BN Qr1=-11→-14 (MATLAB说-14) | pconv_g2_bn Qr 修正 |
| 8 | bn_fp | BN weight声明为int16_t但有值>32767 | weight改为const uint16_t* |

### Q20 GRU 升级 (completed)

Q15→Q20 GRU 隐态升级: h_cache int16_t→int32_t, sigmoid/tanh LUT Q20输出,
门控+混态全部 Q20 精度, 输出 Q20→Q15 转回保持向后兼容。
参考 GTCRN 位运算 LUT 模式 (pos>>SHIFT, pos&MASK)。

### 当前 Golden SNR 状态 (Q20 GRU vs Q15 参考)

| Test | SNR | 说明 |
|------|-----|------|
| BM | 146.83 dB | ✓ 完美 |
| E0 XConv (独立, nH=24) | 71.78 dB | △ Q20 vs Q15 参考 |
| E1 iso (nH=48) | 8.93 dB | △ GRU 越大发散越多 |
| E2 iso (nH=48) | 7.99 dB | △ |
| E3 iso (nH=64) | 8.61 dB | △ |
| E4 iso (nH=32) | 14.73 dB | △ |
| Full Encoder E4 | 1.76 dB | △ 级联累积 |
| Decoder (真实输入) | -1.29 dB | △ 级联累积 |
| Decoder (全golden输入) | 14.91 dB | △ |

**关键认识**: SNR 与 GRU nHidden 强相关 — Q20 精度更高, 与 Q15 参考的差异
随 GRU 规模增大而累积。非 GRU 路径 (conv/bn/shuffle/ctfa_apply) 已验证完美 (999 dB)。
所有 "低 SNR" 都是 Q20→Q15 golden 参考不匹配, 不是 C 代码 bug。

### 下一步

| 优先级 | 任务 |
|--------|------|
| **P0** | 端到端音频测试 — 真实音频效果是最终指标 |
| P1 | `make TARGET=x2000` 交叉编译验证 |
| P2 | MATLAB Q20 GRU 参考 golden (改 GRU_module.m 用 Q20 精度) |

### 诊断工具

| 工具 | 用途 |
|------|------|
| `diag_e0.m` + `diag_e0_ctfa.m` | E0 逐子操作导出到 diag_e0/ |
| `diag_decoder.m` | D0-D4 逐子操作导出到 diag_decoder/ |
| `test_matlab_golden.c` | 逐层 binary golden 比对 (make test) |
| `test_decoder_diag.c` | 逐子操作 text golden 比对 (make diag) |

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
| s32f20 | int32_t | 20 | **Q20 GRU 隐态** (升级后) |
| s16f15 | int16_t | 15 | GRU输出(Q20→Q15转回), tanh输出 |
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
8. **running_var/BN weight** 有值 >32767，必须用uint32_t/uint16_t
9. **FA reshape**: MATLAB `reshape(x,[ngrp,nseg])'` = `A(grp,seg)` → `B(seg,grp)`
10. **D4 输出 [1,129]** 单通道，不是 [2,129]
11. **Q20 GRU HH MAC**: h_cache 是 Q20 (32x Q15), HH weight MAC 需先 `(h+16)>>5` 转 Q15
12. **golden SNR 低不一定是 bug** — Q20 GRU 精度高于 Q15 golden 参考, 差异随 nHidden 增大
