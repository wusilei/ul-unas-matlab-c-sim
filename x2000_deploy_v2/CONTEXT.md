# 新对话启动上下文 — UL-UNAS MATLAB→C 定点转换工程

## 工程路径
```
/media/sf_haidesi/haidesi/ul-unas-x2000-deploy/UL-UNAS_SE_FPversion_v3/UL-UNAS_SE_FPversion_v2/x2000_deploy_v2/
```

## 启动新对话的提示词
把下面这段发给 Claude：

"""
继续 UL-UNAS MATLAB→C 定点转换工程。
先读 CONTEXT.md、progress.md、ulunas_fp.h 了解项目状态，然后继续工作。
工程路径：x2000_deploy_v2/
"""

## 编译
```
cd x2000_deploy_v2/
make            # PC (gcc)
make TARGET=x2000  # X2000 MIPS32R2
```

## 当前状态（2026-07-08）
- 编译: PC + X2000 均零错误
- Git: https://github.com/wusilei/ul-unas-matlab-c-sim
- 已修复 9 个 bug
- **GRU/BiGRU 是最后一个待修复组件**

## E0 逐组件验证结果 (diag_e0/ MATLAB golden)
已验证 bit-exact (999 dB): CONV2D, BN, AffinePReLU, TA Energy, TA FC,
  FA Energy, FA Reshape, FA FC, ctfa_apply
待修复: TA GRU (11 dB), FA BiGRU (4 dB) — 隐态饱和在 int16 ±32768

## 核心文件
| 文件 | 说明 |
|------|------|
| ulunas_fp.h | Q格式、函数声明、状态结构体 |
| ulunas_fp.c | 全部定点算子实现 |
| ulunas_modules.c | 10层Encoder/Decoder + DPRNN |
| ulunas_infer.c | 单帧推理管线 |
| ulunas_stft.c | Q15 FFT/STFT/ISTFT |
| test_matlab_golden.c | Golden比对测试 |
| diag_e0/ | MATLAB导出的E0中间值(文本) |
| golden/ | MATLAB导出的逐层golden(二进制) |
| ulunas_matlab_weights.h/c | 409个权重张量(1.4MB, MATLAB生成) |
| layer_dims.h | 维度宏 |
| qr_config.h | Qr移位配置宏 |
| CONTEXT.md | 本文件 |
| progress.md | 进度记录 |

## 关键技术要点
1. **所有权重数组是 MATLAB 列优先存储** — C代码必须用列优先索引
2. **Q格式**: 激活 s32f20, GRU隐态 s16f15, sigmoid u16f15, LN方差 u16f11
3. **bn_fp**: running_mean/var 逐信道索引 (c = i / W)
4. **affineprelu_fp**: weight/bias 是 [C,W] 列优先, slope 是 [C]
5. **D4 输出**: [1,129] 单通道，不是 [2,129]
6. **GRU weight/macro**: IH_R_W(i,j)=weight[i+j*input_dim] 正确(列优先)
7. **FC weight**: weight[i+o*in_dim] 正确(列优先)

## 待修复的列优先问题（影响 E1-E4/Decoder）
- gconv2d_fp: 用 &weight[nOut*Kh*Kw] 但应列优先访问
- tconv2d_fp: 同样问题
- gtconv2d_fp: 同样
- non_gconv2d_fp / non_gtconv2d_fp: 同样

## GRU 调试线索
- TA GRU: energy聚合正确→GRU→FC(999dB on golden GRU)
- GRU权重精确匹配 MATLAB (ih_r[0,0], hh_r[0,0] 验证)
- 但 C GRU 输出多元素饱和在 ±32768
- diag_e0/ 有 TA/FA 各步骤 golden 值可逐步骤比对
- 怀疑: bias Q格式 或 sigmoid/tanh LUT 精度 或 hidden state mixing 公式
