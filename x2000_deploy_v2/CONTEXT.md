# 新对话启动上下文 — UL-UNAS MATLAB→C 定点转换工程

## 工程路径
```
D:\haidesi\haidesi\ul-unas-x2000-deploy\UL-UNAS_SE_FPversion_v3\UL-UNAS_SE_FPversion_v2\x2000_deploy_v2\
```

## 启动新对话
```
继续 UL-UNAS MATLAB→C 定点转换工程。
先读 x2000_deploy_v2/CONTEXT.md 了解项目状态，然后继续工作。
```

## 编译
```bash
cd x2000_deploy_v2/
make                # PC (gcc)
make TARGET=x2000   # X2000 MIPS32R2 soft-float
```

## Git
```
https://github.com/wusilei/ul-unas-matlab-c-sim
```

---

## 当前状态 (2026-07-08 最终)

### 编译: ✅ PC + X2000 均零错误

### 已修复 Bug 清单 (10个)

| # | 文件 | Bug | 修复 |
|---|------|-----|------|
| 1 | ulunas_fp.c:bn_fp | running_mean 按 N=C×W 索引，数组只有C个元素 | `c = i / W` 逐信道 |
| 2 | ulunas_fp.c:ctfa_fa_fp | x_re[68] 栈硬编码太小 (W=129需132) | malloc(nseg*ngrp) |
| 3 | ulunas_fp.c:gru_step_fp | r_t_q15[32] 硬编码 (nHidden最大64) | 改为[64] |
| 4 | — | ctfa_ta_fp 错误第一版实现 | 删除重写 |
| 5 | ulunas_fp.c:gtconv2d_fp | W公式算错129而非33 | 修正公式 |
| 6 | ulunas_modules.c | D4 TA GRU input_dim=1但传C=2 | 增加input_dim参数 |
| 7 | ulunas_fp.c:bn_fp | `(int16_t)running_var[c]` 破坏>32767的值 | `(uint32_t)running_var[c]` |
| 8 | ulunas_fp.c:ctfa_fa_fp | FA reshape `grp*nseg+seg` 映射反向 | `grp+ngrp*seg` |
| 9 | ulunas_fp.c:gru_step_fp | `r_t_q15/z_t_q15` 是 `int16_t` 但sigmoid返回uint16_t, 32768→-32768 | 改为 `uint16_t` |
| 10 | ulunas_fp.c (5处) | gconv/tconv/gtconv/non_gconv/non_gtconv 权重用 `&weight[nOut*Kh*Kw]` 非列优先 | 改为列优先索引公式 |

### 待测试验证
- Bug 8-10 是本轮最后修复，需要在VM上编译+跑golden比对
- E0 cTFA 诊断数据在 `diag_e0/` 目录，修复后可逐步骤比对
- E1-E4 + Decoder 需要修复 conv 列优先后才能通过

---

## 核心文件速查

| 文件 | 行数 | 说明 |
|------|------|------|
| `ulunas_fp.h` | ~330 | Q格式定义、状态结构体、算子声明 |
| `ulunas_fp.c` | ~1300 | **全部定点算子** (conv/pconv/gconv/tconv/gtconv/BN/LN/AffinePReLU/GRU/BiGRU/FC/cTFA/sigmoid/tanh/log10/shuffle) |
| `ulunas_modules.c` | ~700 | 10层 Encoder/Decoder + GDPRNN 子模块 |
| `ulunas_infer.c` | ~200 | 单帧顶层推理管线 |
| `ulunas_stft.c` | — | Q15 FFT/STFT/ISTFT |
| `ulunas_matlab_weights.h` | ~1200 | 409个权重张量外部声明 (MATLAB extract_weights.m 生成) |
| `ulunas_matlab_weights.c` | ~11000 | 权重数据 (1.4MB) |
| `ulunas_lut.h/.c` | — | sigmoid/tanh/log10/sqrt LUT (gen_lut_tables.m 生成) |
| `layer_dims.h` | ~150 | 所有层维度/卷积核/步长宏 |
| `qr_config.h` | ~200 | 所有层 Qr 移位配置宏 |
| `test_matlab_golden.c` | — | Golden二进制比对+SNR测试 |
| `Makefile` | — | PC gcc + X2000 mips交叉编译 |
| `CONTEXT.md` | — | 本文件 |

### MATLAB 脚本 (x2000_deploy_v2/)
| 文件 | 说明 |
|------|------|
| `gen_lut_tables.m` | 生成 ulunas_lut.h/.c |
| `extract_weights.m` | 导出409个权重→C数组 + 生成 qr_config.h/layer_dims.h |
| `export_all_layers.m` | 导出前10帧逐层golden二进制到 golden/ |
| `diag_e0.m` | 导出E0 conv2d/BN/AffinePReLU中间值到 diag_e0/ |
| `diag_e0_ctfa.m` | 导出E0 cTFA TA/FA/Fusion中间值到 diag_e0/ |

---

## 模型架构速查

### 推理管线
```
STFT(512pt,256hop) → log_gen → BM(257→129) → Encoder×5 → GDPRNN×2 → Decoder×5(+skip) → sigmoid → BS(129→257) → MASK → ISTFT
```

### Encoder 5层
| 层 | 输入 | 输出 | 缓存 | 关键算子 |
|---|------|------|------|---------|
| E0 XConv | [1,129] | [12,65] | 2帧 | Conv2D→BN→AffinePReLU + cTFA |
| E1 XMB0 | [12,65] | [24,33] | 1帧 | PConv×2→Shuffle→GConv→PConv×2 + cTFA |
| E2 XDWS0 | [24,33] | [24,33] | 1帧 | PConv→Shuffle→GConv + cTFA |
| E3 XMB1 | [24,33] | [32,33] | 无 | PConv×2→Shuffle→nonGConv→PConv×2 + cTFA |
| E4 XDWS1 | [32,33] | [16,33] | 无 | PConv→Shuffle→nonGConv + cTFA |

### Decoder 5层 (Encoder镜像+skip connection)
| 层 | 输入+skip | 输出 |
|---|-----------|------|
| D0 De_XDWS0 | [16,33]+skip_e4 | [32,33] |
| D1 De_XMB0 | [32,33]+skip_e3 | [24,33] |
| D2 De_XDWS1 | [24,33]+skip_e2 | [24,33] |
| D3 De_XMB1 | [24,33]+skip_e1 | [12,65] |
| D4 De_XConv | [12,65]+skip_e0 | **[1,129]** ← 单通道! |

### GDPRNN
```
[16,33] → transpose → Intra-RNN (分组BiGRU×2→FC→LN→残差)
                    → Inter-RNN (分组GRU×2→FC→LN→残差) → transpose → [16,33]
```

---

## Q格式完整定义

| 格式 | C类型 | 小数位 | 缩放 | 使用场景 |
|------|-------|--------|------|---------|
| s32f20 | int32_t | 20 | ×1048576 | 主干激活信号 |
| u32f20 | uint32_t | 20 | ×1048576 | log_gen, cTFA聚合 |
| s16f15 | int16_t | 15 | ×32768 | GRU隐态, tanh输出 |
| u16f15 | uint16_t | 15 | ×32768 | sigmoid输出, 注意力掩膜, BM/BS |
| u16f11 | uint16_t | 11 | ×2048 | LN running_var |

### 关键 Qr 移位值
| 算子 | Qr值 | 使用处 |
|------|------|--------|
| conv2d | -14 | E0 TConv |
| pconv/gconv | -13/-14 | 各层PConv/TConv |
| BN Qr1/Qr2 | -11/-14 (E0:-14/-14, D4:-11/-11) | 所有BN |
| AffinePReLU Qr1/Qr2 | -13/-13 | 所有AffinePReLU |
| GRU ih (Qr1) | -13 | 所有GRU |
| GRU hh (Qr2) | -8 | 所有GRU |
| GRU gate mixing | -15 | 隐态更新 |
| cTFA attn | -15 | TA/FA融合 |
| DPRNN FC | -9 | Intra/Inter FC |
| DPRNN Intra LN | -14 | Intra LN |
| DPRNN Inter LN | -13 | Inter LN |
| BM/BS | -15 | ERB合并/分裂 |
| MASK | -15 | 掩膜作用 |

---

## 关键技术要点 (容易踩坑)

1. **所有权重数组 = MATLAB列优先存储** → C代码必须用列优先索引公式
2. **weight 4D [Cout,Cin,Kh,Kw]** 列优先: `widx = co + Cout*ci + Cout*Cin*kh + Cout*Cin*Kh*kw`
3. **weight 3D [Cout,1,Kh,Kw]** 列优先: `widx = co + Cout*kh + Cout*Kh*kw`
4. **tconv weight [Cin,Cout,Kh,Kw]**: `widx = ci + Cin*co + Cin*Cout*kh + Cin*Cout*Kh*kw` (注意Cin/Cout互换!)
5. **GRU ih_weight [input_dim, 3*nHidden]**: `IH_R_W(i,j) = weight[i + j*input_dim]`
6. **BN running_mean/var 逐信道**: `c = i / W` (每个信道广播到所有W个元素)
7. **D4 De_XConv 输出 [1,129]** 单通道，不是 [2,129]
8. **sigmoid返回 uint16_t** (u16f15: 0-32768)，GRU中需用uint16_t接收，不能用int16_t
9. **running_var** 有值 >32767，bn_fp中必须用uint32_t转换，不能用int16_t
10. **FA reshape**: MATLAB `reshape(x,[ngrp,nseg])'` = 列优先 `A(grp,seg)` 转置为 `B(seg,grp)`
11. **NGrp和Pad**: 所有层 ngrp=4, pad=3 → nseg=(W+3)/4

---

## 下一步
1. VM上 `make clean && make` 编译修复后的代码
2. 运行 golden 比对验证 E0 cTFA + E1-E4 + Decoder
3. 如仍有 SNR 问题，用 diag_e0/ 诊断数据逐步骤比对定位
