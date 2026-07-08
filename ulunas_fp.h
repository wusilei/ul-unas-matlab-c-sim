/* ulunas_fp.h — UL-UNAS Fixed-Point Inference Engine
 * Pure integer arithmetic for X2000 (MIPS32R2, no FPU).
 * All Q-format, dimension, and operator definitions.
 */

#ifndef ULUNAS_FP_H
#define ULUNAS_FP_H

#include <stdint.h>
#include <string.h>
#include "qr_config.h"
#include "layer_dims.h"
#include "ulunas_lut.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================== */
/* 1. Q-FORMAT MACROS                                                        */
/* ======================================================================== */

/* Activation / Signal Q formats */
#define Q_ACT           20    /* s32f20: main activations (conv i/o, BN, residual) */
#define Q_LOG           20    /* u32f20: log_gen output, cTFA aggregation */
#define Q_GRU_H         15    /* s16f15: GRU/BiGRU hidden state, tanh output */
#define Q_SIG           15    /* u16f15: sigmoid output, cTFA attention mask */
#define Q_LN_VAR        11    /* u16f11: LN 1/sqrt(var) */

/* Scale factors */
#define SCALE_Q20       1048576    /* 2^20 */
#define SCALE_Q15       32768      /* 2^15 */
#define SCALE_Q14       16384      /* 2^14 */
#define SCALE_Q13       8192       /* 2^13 */
#define SCALE_Q12       4096       /* 2^12 */
#define SCALE_Q11       2048       /* 2^11 */
#define SCALE_Q10       1024       /* 2^10 */

/* Rounding constant */
#define ROUND_HALF(shift)  (1 << ((-(shift)) - 1))

/* Quantize / Dequantize helpers */
#define Q20_TO_FLOAT(x)   ((float)(x) / (float)SCALE_Q20)
#define Q15_TO_FLOAT(x)   ((float)(x) / (float)SCALE_Q15)
#define FLOAT_TO_Q20(x)   ((int32_t)roundf((x) * (float)SCALE_Q20))
#define FLOAT_TO_Q15(x)   ((int16_t)roundf((x) * (float)SCALE_Q15))

/* Saturate helpers */
static inline int32_t sat_s32(int64_t x) {
    if (x >  2147483647LL) return  2147483647;
    if (x < -2147483648LL) return -2147483648;
    return (int32_t)x;
}
static inline int16_t sat_s16(int32_t x) {
    if (x >  32767) return  32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}
static inline uint16_t sat_u16(int32_t x) {
    if (x > 65535) return 65535;
    if (x < 0)     return 0;
    return (uint16_t)x;
}

/* ======================================================================== */
/* 2. MODEL STATE STRUCTURE                                                  */
/* ======================================================================== */

typedef struct {
    /* Temporal convolution caches (Encoder) */
    int32_t conv_cache_e0[CACHE_E0_SIZE];   /* [2, 129] */
    int32_t conv_cache_e1[CACHE_E1_SIZE];   /* [24, 65] */
    int32_t conv_cache_e2[CACHE_E2_SIZE];   /* [24, 33] */

    /* Temporal convolution caches (Decoder) */
    int32_t conv_cache_d0[CACHE_D0_SIZE];   /* [24, 33] */
    int32_t conv_cache_d1[CACHE_D1_SIZE];   /* [12, 33] */
    int32_t conv_cache_d2[CACHE_D2_SIZE];   /* [12, 2, 65] */

    /* cTFA TA GRU hidden state caches (Encoder) — int16_t Q15 */
    int16_t tfa_cache_e0[E0_TA_GRU_HID];    /* 24 */
    int16_t tfa_cache_e1[E1_CTFA_TA_HID];   /* 48 */
    int16_t tfa_cache_e2[E2_CTFA_TA_HID];   /* 48 */
    int16_t tfa_cache_e3[E3_CTFA_TA_HID];   /* 64 */
    int16_t tfa_cache_e4[E4_CTFA_TA_HID];   /* 32 */

    /* cTFA TA GRU hidden state caches (Decoder) */
    int16_t tfa_cache_d0[D0_CTFA_TA_HID];   /* 64 */
    int16_t tfa_cache_d1[D1_CTFA_TA_HID];   /* 48 */
    int16_t tfa_cache_d2[D2_CTFA_TA_HID];   /* 48 */
    int16_t tfa_cache_d3[D3_CTFA_TA_HID];   /* 24 */
    int16_t tfa_cache_d4[D4_CTFA_TA_HID];   /* 2 */

    /* GDPRNN Inter-RNN hidden state caches */
    int16_t inter_cache_0[CACHE_INTER_SIZE]; /* [33, 16] */
    int16_t inter_cache_1[CACHE_INTER_SIZE]; /* [33, 16] */

} ulunas_state_t;

/* Initialize state to zero */
static inline void ulunas_state_init(ulunas_state_t *s) {
    memset(s, 0, sizeof(ulunas_state_t));
}

/* ======================================================================== */
/* 3. OPERATOR DECLARATIONS                                                  */
/* ======================================================================== */

/* ── Convolution operators ─────────────────────────────────────────────── */

/* Standard 2D convolution (stride, padding) */
void conv2d_fp(
    const int32_t *x, int Cin, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y);

/* Point-wise convolution (1×1)
 * wstride = stride between input channels in the weight array.
 *           For grouped conv with nGroups, wstride = Cout * nGroups. */
void pconv2d_fp(
    const int32_t *x, int Cin, int Cout, int Wout,
    const int16_t *weight, const int32_t *bias, int Qr, int wstride,
    int32_t *y);

/* Grouped temporal conv (with cache, 2-row input concat) */
void gconv2d_fp(
    const int32_t *x, const int32_t *conv_cache,
    int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y, int32_t *cache_out);

/* Grouped non-temporal conv (no cache, 1×K kernel) */
void non_gconv2d_fp(
    const int32_t *x, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y);

/* Transposed 2D convolution */
void tconv2d_fp(
    const int32_t *x, int Cin, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y);

/* Grouped transposed conv (with cache) */
void gtconv2d_fp(
    const int32_t *x, const int32_t *conv_cache,
    int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y, int32_t *cache_out);

/* Grouped non-temporal transposed conv (no cache) */
void non_gtconv2d_fp(
    const int32_t *x, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y);

/* ── Normalization ─────────────────────────────────────────────────────── */

/* Batch Normalization */
void bn_fp(
    const int32_t *x, int C, int W,
    const uint16_t *weight, const int32_t *bias,
    const int32_t *running_mean, const uint16_t *running_var,
    int Qr1, int Qr2,
    int32_t *y);

/* Layer Normalization (uses float for mean/var stats internally) */
void ln_fp(
    const int32_t *x, int N,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y);

/* ── Activation ────────────────────────────────────────────────────────── */

/* AffinePReLU: PReLU + affine + residual */
void affineprelu_fp(
    const int32_t *x, int C, int W,
    const int16_t *weight, const int32_t *bias,
    const int16_t *slope, int Qr1, int Qr2,
    int32_t *y);

/* ── RNN ───────────────────────────────────────────────────────────────── */

/* Single-direction GRU (one timestep) */
void gru_step_fp(
    const int32_t *x_t, int input_dim,
    int16_t *h_cache, int nHidden,
    const int16_t *ih_weight, const int32_t *ih_bias,
    const int16_t *hh_weight, const int32_t *hh_bias,
    int Qr1, int Qr2,
    int16_t *y_out);

/* Multi-timestep unidirectional GRU */
void gru_sequence_fp(
    const int32_t *x, int T, int input_dim,
    int16_t *h_cache, int nHidden,
    const int16_t *ih_weight, const int32_t *ih_bias,
    const int16_t *hh_weight, const int32_t *hh_bias,
    int Qr1, int Qr2,
    int16_t *y_out);

/* BiGRU over T timesteps */
void bigru_sequence_fp(
    const int32_t *x, int T, int input_dim,
    int nHidden,
    const int16_t *ih_weight, const int32_t *ih_bias,
    const int16_t *hh_weight, const int32_t *hh_bias,
    const int16_t *re_ih_weight, const int32_t *re_ih_bias,
    const int16_t *re_hh_weight, const int32_t *re_hh_bias,
    int Qr1, int Qr2,
    int16_t *y_out);

/* Full Connection (matrix multiply + bias) */
void fc_fp(
    const int32_t *x, int rows, int in_dim, int out_dim,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y);

/* fc_fp with int16 input (for RNN post-processing) */
void fc_fp_s16(
    const int16_t *x, int rows, int in_dim, int out_dim,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y);

/* ── ERB and MASK ──────────────────────────────────────────────────────── */

/* ERB Band Merge */
void bm_fp(
    const int32_t *x,
    const uint16_t *erbfc_weight,
    int32_t *y);

/* ERB Band Split */
void bs_fp(
    const uint16_t *x_mask,
    const uint16_t *ierbfc_weight,
    int32_t *y);

/* Mask application to complex spectrum */
void mask_fp(
    const int32_t *x_mask,
    const int32_t *spec_real, const int32_t *spec_imag,
    int N,
    int32_t *y_real, int32_t *y_imag);

/* ── cTFA (Time-Frequency Attention) ───────────────────────────────────── */

/* TA (Time Attention) branch */
void ctfa_ta_fp(
    const int32_t *x, int C, int W,
    int16_t *ta_h_cache, int nHidden, int input_dim,
    const int16_t *ih_weight, const int32_t *ih_bias,
    const int16_t *hh_weight, const int32_t *hh_bias,
    const int16_t *fc_weight, const int32_t *fc_bias,
    uint16_t *ta_out);

/* FA (Frequency Attention) branch */
void ctfa_fa_fp(
    const int32_t *x, int C, int W, int pad_len,
    int nHidden,
    const int16_t *ih_weight, const int32_t *ih_bias,
    const int16_t *hh_weight, const int32_t *hh_bias,
    const int16_t *re_ih_weight, const int32_t *re_ih_bias,
    const int16_t *re_hh_weight, const int32_t *re_hh_bias,
    const int16_t *fc_weight, const int32_t *fc_bias,
    uint16_t *fa_out);

/* cTFA fusion: apply TA and FA attention masks */
void ctfa_apply_fp(
    const int32_t *x, int C, int W,
    const uint16_t *ta, const uint16_t *fa,
    int32_t *y);

/* ── Non-linear functions ──────────────────────────────────────────────── */

/* sigmoid: int32_t Q20 → uint16_t Q15 (uses LUT + linear interp) */
uint16_t sigmoid_q20_to_q15(int32_t x_q20);

/* tanh: int32_t Q20 → int16_t Q15 (uses LUT + linear interp) */
int16_t tanh_q20_to_q15(int32_t x_q20);

/* log10 magnitude: real/imag (s32f20) → log magnitude (s32f20) */
int32_t log_gen_fp(int32_t real_q20, int32_t imag_q20);

/* ── Shuffle operations ────────────────────────────────────────────────── */

/* Interleave: first half → odd rows, second half → even rows */
void shuffle_interleave(const int32_t *src, int C, int W, int32_t *dst);

/* Deinterleave: odd rows → first half, even rows → second half */
void shuffle_deinterleave(const int32_t *src, int C, int W, int32_t *dst);

/* ── STFT / ISTFT ─────────────────────────────────────────────────────── */

void stft_frame_q15(const int16_t *audio, int win_len, const int16_t *window, int n_fft,
                    int32_t *real_out, int32_t *imag_out);
int  stft_process(const float *audio, int n_samples, int n_fft, int win_len, int hop,
                  const int16_t *window, int32_t *real_spec, int32_t *imag_spec);
void istft_frame_q15(const int32_t *real_in, const int32_t *imag_in,
                     const int16_t *window, int n_fft, int win_len, int16_t *audio_out);
int  istft_process(const int32_t *real_spec, const int32_t *imag_spec,
                   int n_frames, int n_fft, int win_len, int hop,
                   const int16_t *window, float *audio_out, int max_out);
void gen_hann_window_q15(int win_len, int16_t *window);
void spec_q15_to_q20(const int16_t *real_q15, const int16_t *imag_q15, int N,
                     int32_t *real_q20, int32_t *imag_q20);

/* ======================================================================== */
/* 4. HIGH-LEVEL MODULE MACROS (inline helpers)                              */
/* ======================================================================== */

/* Element-wise multiply with Qr shift */
#define ELTWISE_MUL_Q(x, w, qr)  ((int32_t)(((int64_t)(x) * (int64_t)(w) + ROUND_HALF(qr)) >> (-(qr))))

/* Element-wise multiply with uint16 weight */
#define ELTWISE_MUL_UQ(x, w, qr) ((int32_t)(((int64_t)(x) * (int64_t)(uint32_t)(w) + ROUND_HALF(qr)) >> (-(qr))))

#ifdef __cplusplus
}
#endif

#endif /* ULUNAS_FP_H */
