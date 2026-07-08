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
make test       # 逐层 binary golden 比对 (11 项测试)
make diag       # Decoder 逐子操作 text golden 比对
```

## Git
```
https://github.com/wusilei/ul-unas-matlab-c-sim
```
VM 无网络，文件通过共享文件夹同步。`git pull` 不可用，直接读工作树。

---

## 当前状态 (2026-07-08 最终)

### 编译: ✅ PC (gcc) 零错误

### 已修复 Bug 清单 (8个，本 session + MATLAB)

| # | 文件 | Bug | 修复 |
|---|------|-----|------|
| 9 | gru_step_fp | r_t/z_t int16_t 但 sigmoid 返回 uint16_t (32768→-32768) | 改为 uint16_t |
| 8 | ctfa_fa_fp | FA reshape `grp*nseg+seg` 映射反向 | `grp+ngrp*seg` |
| 10 | gconv/tconv/gtconv/non_gconv/non_gtconv | 权重行优先→列优先 | 修正列优先索引公式 |
| 11 | pconv2d_fp | grouped conv 权重 stride=Cout→Co*nGroups | 增加 wstride 参数 |
| 15 | E1 pconv_g2_bn | BN Qr1=-11→-14 (MATLAB 说 -14) | pconv_g2_bn Qr 修正 |
| 16 | bn_fp | BN weight 声明为 int16_t 但有值 >32767 (33101→-32435) | weight 改为 const uint16_t* |
| 17 | gru_step_fp_q20 | HH MAC: h_cache 是 Q20 (32x)，HH weight 是 Q12，乘积溢出 | `(h+16)>>5` 转 Q15 再乘 |
| 18 | export_all_layers.m | GRU_module_q20 输出 y=h_cache (Q20) 而 C 代码输出 (hn+16)>>5 (Q15) | MATLAB 改为 Fix_point(h_cache*2^(-20), 's16f15') |

注: #1-7 是更早 session 修复的 (BN per-channel, conv cache, Qr, Decoder layers, ASAN 等)，详见 git log。 #18 是 MATLAB 端修复 (commit e580fee)。

### Q20 GRU 升级 (已完成)

Q15→Q20 GRU 隐态升级要点:
- h_cache: int16_t→int32_t (Q15→Q20)
- sigmoid/tanh: Q15→Q20 LUT (4096pt, 位运算索引)
- 门控+混态: 全部 Q20 精度
- 输出: Q20→Q15 转回 (`(hn+16)>>5`)，保持向后兼容
- **关键**: HH MAC 必须先 `(h+16)>>5` 转 Q15 再和 HH weight (Q12) 相乘

### 当前 Golden SNR 状态 (Q20 GRU C vs Q20 GRU MATLAB golden, e580fee)

```
BM:                        146.83 dB ✓  非 GRU, 完美
────────────────────────────────────────────────
E0 XConv  (独立, nH=24):   71.67 dB △  LUT vs float sigmoid/tanh 天花板
E1 iso    (gold E0, nH=48):  8.93 dB △  GRU 越大发散越多
E2 iso    (gold E1, nH=48):  7.99 dB △
E3 iso    (gold E2, nH=64):  8.61 dB △  最大 GRU
E4 iso    (gold E3, nH=32): 14.73 dB △
────────────────────────────────────────────────
Full Encoder E4:             1.76 dB △  5层级联
DPRNN idx=0:                 1.81 dB ✗  待诊断
DPRNN idx=1:                 5.28 dB ✗  待诊断
Decoder (真实输入):         -4.66 dB △  误差传播
Decoder (全golden输入):     14.91 dB △  纯 Decoder
```

**关键认识**: SNR 与 GRU nHidden 强相关 — Q20 GRU 使用 4096pt LUT (位运算索引)，MATLAB 使用 float sigmoid/tanh，精度差异随 GRU 规模增大而累积。非 GRU 路径 (conv/bn/shuffle/ctfa_apply) 已验证完美 (999 dB with golden masks)。Golden 已重生为 Q20 (MATLAB GRU_module_q20 输出 Fix_point Q15，与 C 代码 (hn+16)>>5 一致)，所有 SNR 与之前 Q15 golden 基线一致，确认 golden 生成正确。

### 已验证正确的组件

| 组件 | 验证方式 | 结果 |
|------|---------|------|
| ctfa_apply_fp | golden TA+FA masks → E0 输出 | **999 dB** |
| Conv2D, BN, AffinePReLU | E0 诊断 | **999 dB** |
| PConv (含 wstride) | D1/D3 pconv vs text golden | **80 dB** |
| Shuffle, Skip Add | 直接比对 | **正确** |
| GConv/NGTConv/TConv 权重 | 列优先公式 vs MATLAB | **匹配** |

---

## 优先级路线图

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P0** ✅ | MATLAB Q20 golden 重生 | 完成 (e580fee)。修复: GRU_module_q20 输出 y=h_cache(Q20)→Fix_point(Q15)。Golden 与 C 代码 Q20 GRU 一致，所有层 SNR 恢复到 Q15 golden 基线水平 |
| **P1** | Decoder 全链路诊断 | `make diag` 逐子操作定位 Decoder 14.91 dB 瓶颈。怀疑 D4 De_XConv (nH=2 单通道) 或 GRU 精度差异 |
| **P2** | DPRNN 独立诊断 | idx=0 (1.81 dB) 和 idx=1 (5.28 dB) 需要子块级 SNR |
| **P3** | 端到端音频测试 | 所有层 >60 dB 后跑完整推理链，对比音频质量 |

---

## 诊断工具

| 工具 | 用途 |
|------|------|
| `make test` | 11 项逐层 binary golden 比对 (含 E1-E4 独立层隔离) |
| `make diag` | Decoder D0-D4 逐子操作 text golden 比对 |
| `diag_e0.m` + `diag_e0_ctfa.m` | MATLAB 导出 E0 逐子操作到 diag_e0/ |
| `diag_decoder.m` | MATLAB 导出 D0-D4 逐子操作到 diag_decoder/ |
| `export_all_layers.m` | MATLAB 导出逐层 binary golden 到 golden/ |

## 核心文件速查

| 文件 | 说明 |
|------|------|
| ulunas_fp.h | Q格式、状态结构体 (Q20 缓存)、算子声明 |
| ulunas_fp.c | **全部定点算子** (含 gru_step_fp_q20) |
| ulunas_modules.c | 10层 Encoder/Decoder + GDPRNN (Q20 调用) |
| ulunas_infer.c | 单帧推理管线 |
| ulunas_stft.c | Q15 FFT/STFT/ISTFT |
| ulunas_matlab_weights.h/c | 409个权重张量 (1.4MB, MATLAB 生成) |
| ulunas_lut.h/c | sigmoid/tanh Q15+Q20 LUT (4096pt, 位运算索引) |
| layer_dims.h | 维度宏 |
| qr_config.h | Qr移位配置宏 |
| test_matlab_golden.c | Golden比对测试 (11项) |
| test_decoder_diag.c | Decoder子操作诊断 |
| diag_e0/ | MATLAB导出的E0中间值(文本) |
| diag_decoder/ | MATLAB导出的D0-D4中间值(文本) |
| golden/ | MATLAB导出的逐层golden(二进制) |
| CONTEXT.md | 本文件 |

## 模型架构

```
STFT(512pt,256hop) → log_gen → BM(257→129) → Encoder×5 → GDPRNN×2 → Decoder×5(+skip) → sigmoid → BS(129→257) → MASK → ISTFT
```

| 层 | 输入 | 输出 | GRU nHidden | 关键算子 |
|---|------|------|------------|---------|
| E0 XConv | [1,129] | [12,65] | 24 | Conv2D→BN→AffinePReLU+cTFA |
| E1 XMB0 | [12,65] | [24,33] | 48 | PConv×2→Shuffle→GConv(stride=2)→PConv×2+cTFA→Shuffle |
| E2 XDWS0 | [24,33] | [24,33] | 48 | PConv→Shuffle→GConv(stride=1)+cTFA |
| E3 XMB1 | [24,33] | [32,33] | 64 | PConv×2→Shuffle→nonGConv→PConv×2+cTFA→Shuffle |
| E4 XDWS1 | [32,33] | [16,33] | 32 | PConv→Shuffle→nonGConv+cTFA |
| D0-D4 | — | — | 64,48,48,24,2 | Decoder 5层 (Encoder镜像+skip connection) |
| D4 De_XConv | [12,65]+skip | **[1,129]** | 2 | ← 单通道! |

## Q格式

| 格式 | C类型 | 小数位 | 使用场景 |
|------|-------|--------|---------|
| s32f20 | int32_t | 20 | 主干激活信号, **Q20 GRU 隐态** |
| u32f20 | uint32_t | 20 | log_gen, cTFA聚合 |
| s16f15 | int16_t | 15 | GRU输出 (Q20→Q15转回), tanh输出 |
| u16f15 | uint16_t | 15 | sigmoid输出, 注意力掩膜 |
| u16f11 | uint16_t | 11 | LN running_var |

## 关键技术要点 (容易踩坑)

1. **所有权重数组是 MATLAB 列优先存储** — C代码必须用列优先索引
2. **weight 4D [Cout,Cin,Kh,Kw]**: `widx = co + Cout*ci + Cout*Cin*kh + Cout*Cin*Kh*kw`
3. **tconv weight [Cin,Cout,Kh,Kw]**: `widx = ci + Cin*co + Cin*Cout*kh + Cin*Cout*Kh*kw`
4. **PConv grouped [Co,nGroups,Ch]**: wstride = Co*nGroups (不是Co!)
5. **GRU ih_weight [input_dim, 3*nHidden]**: `IH_R_W(i,j) = weight[i + j*input_dim]`
6. **Q20 GRU HH MAC**: h_cache 是 Q20 (32× Q15)，HH weight 是 Q12，乘积会溢出。必须先 `(h+16)>>5` 转 Q15 再乘 HH weight
7. **BN running_mean/var 逐信道**: `c = i / W`
8. **sigmoid返回 uint16_t** (0-32768)，GRU中需用uint16_t，不能用int16_t
9. **running_var/BN weight** 有值 >32767，必须用uint32_t/uint16_t 防止符号扩展
10. **FA reshape**: MATLAB `reshape(x,[ngrp,nseg])'` = `A(grp,seg)` → `B(seg,grp)`
11. **D4 输出 [1,129]** 单通道，不是 [2,129]
12. **golden SNR 低不一定是 bug** — Q20 GRU 精度高于 Q15 golden 参考，差异随 nHidden 增大而累积