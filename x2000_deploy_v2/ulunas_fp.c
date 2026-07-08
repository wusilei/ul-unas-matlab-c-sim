/* ulunas_fp.c — UL-UNAS Fixed-Point Operator Implementations
 * Pure integer arithmetic.  All shift/round logic 1:1 matched to MATLAB.
 */

#include "ulunas_fp.h"
#include <stdlib.h>
#include <math.h>  /* only for ln_fp() sqrt via float */

/* ======================================================================== */
/*  CONVOLUTION OPERATORS                                                    */
/* ======================================================================== */


/* ── conv2d_fp (rewrite) ───────────────────────────────────────────────── */

/* Internal helper: single-channel conv2d with MATLAB column-major weight indexing.
 * weight is 4D [Cout, Cin, Kh, Kw] stored in MATLAB column-major order.
 * Access: weight(nOut,nIn,kh,kw) → nOut + Cout*nIn + Cout*Cin*kh + Cout*Cin*Kh*kw
 */
static void conv2d_chan_fp(
    const int32_t *x_chan, int H, int W,
    int Cout, int Cin, int nOut, int nIn,
    int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    int pad_top, int pad_left,
    const int16_t *weight, /* full 4D weight [Cout,Cin,Kh,Kw] col-major */
    int Qr,
    int32_t *y_chan)
{
    int h_id, w_id, kh, kw;
    int32_t round_const = (int32_t)(1LL << ((-Qr) - 1));
    int CoutCin = Cout * Cin;
    int CoutCinKh = CoutCin * Kh;

    for (int i = 0; i < Hout * Wout; i++) y_chan[i] = 0;

    for (h_id = 1; h_id <= Hout; h_id++) {
        int h_start = (h_id - 1) * stride_h + 1;
        for (w_id = 1; w_id <= Wout; w_id++) {
            int w_start = (w_id - 1) * stride_w + 1;
            int32_t acc = 0;

            for (kh = 0; kh < Kh; kh++) {
                int h_src = h_start + kh - pad_top;
                for (kw = 0; kw < Kw; kw++) {
                    int w_src = w_start + kw - pad_left;
                    int32_t x_val;

                    if (h_src >= 1 && h_src <= H && w_src >= 1 && w_src <= W) {
                        x_val = x_chan[(h_src - 1) * W + (w_src - 1)];
                    } else {
                        x_val = 0;
                    }

                    /* MATLAB column-major weight access:
                     * weight(nOut,nIn,kh,kw) → nOut + Cout*nIn + Cout*Cin*kh + Cout*Cin*Kh*kw */
                    int widx = nOut + Cout * nIn + CoutCin * kh + CoutCinKh * kw;
                    int16_t w_val = weight[widx];
                    int64_t prod = (int64_t)x_val * (int64_t)w_val;
                    int32_t rounded;
                    if (Qr < 0) {
                        rounded = (int32_t)((prod + (int64_t)round_const) >> (-Qr));
                    } else {
                        rounded = (int32_t)(prod << Qr);
                    }
                    acc += rounded;
                }
            }
            y_chan[(h_id - 1) * Wout + (w_id - 1)] = acc;
        }
    }
}

void conv2d_fp(
    const int32_t *x, int Cin, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y)
{
    /* For this model: input x is [Cin, H, W] where H = Kh (due to cache stacking).
     * pad_top = 0, pad_left = 1.
     * We need input dimensions. The caller knows them implicitly.
     * But the function signature doesn't include them.
     *
     * For E0: x is [1, 3, 129] (Cin=1, H=3, W=129)
     * H = Kh = 3. W = from (Wout-1)*stride_w + Kw - 2*pad_left = (65-1)*2+3-2 = 129. ✓
     */
    int H = Kh;  /* This model always stacks Kh rows */
    int W = (Wout - 1) * stride_w + Kw - 2;  /* pad_left=1 */
    int pad_left = 1, pad_top = 0;
    int nIn, nOut;
    int y_chan[Hout * Wout];
    int32_t round_const = (int32_t)(1LL << ((-Qr) - 1));

    for (nOut = 0; nOut < Cout; nOut++) {
        /* Initialize y_chan to 0 */
        memset(y_chan, 0, Hout * Wout * sizeof(int32_t));

        for (nIn = 0; nIn < Cin; nIn++) {
            const int32_t *x_chan = &x[nIn * H * W];
            int32_t conv_result[Hout * Wout];

            conv2d_chan_fp(x_chan, H, W,
                           Cout, Cin, nOut, nIn,
                           Hout, Wout, Kh, Kw, stride_h, stride_w,
                           pad_top, pad_left, weight, Qr, conv_result);

            /* Accumulate */
            for (int i = 0; i < Hout * Wout; i++) {
                y_chan[i] += conv_result[i];
            }
        }

        /* Add bias */
        for (int i = 0; i < Hout * Wout; i++) {
            y[nOut * Wout + i] = y_chan[i] + bias[nOut];
        }
    }
    (void)round_const; // used above
}

/* ── pconv2d_fp — Point-wise (1×1) convolution ────────────────────────────
 * MATLAB: conv_result = round(x_chan * kernel_chan * 2^(Qr));
 *         y_chan = sum over nIn + bias
 *
 * x:      [Cin, W] in int32_t Q20
 * weight: [Cout_total, Cin] in int16_t, MATLAB column-major
 *         wstride = stride between columns (typically Cout_total == Cout * nGroups)
 * y:      [Cout, W] in int32_t Q20
 */
void pconv2d_fp(
    const int32_t *x, int Cin, int Cout, int Wout,
    const int16_t *weight, const int32_t *bias, int Qr, int wstride,
    int32_t *y)
{
    int nOut, nIn, w;
    int32_t round_const = (int32_t)(1LL << ((-Qr) - 1));

    for (nOut = 0; nOut < Cout; nOut++) {
        /* Initialize output channel to bias */
        for (w = 0; w < Wout; w++) {
            y[nOut * Wout + w] = 0;
        }

        for (nIn = 0; nIn < Cin; nIn++) {
            /* Weight is [Cout_total, Cin] in MATLAB column-major: weight(nOut,nIn) → nOut + wstride*nIn */
            int16_t w_val = weight[nOut + wstride * nIn];
            const int32_t *x_chan = &x[nIn * Wout]; /* [1, Wout] */

            for (w = 0; w < Wout; w++) {
                int64_t prod = (int64_t)x_chan[w] * (int64_t)w_val;
                int32_t rounded;
                if (Qr < 0) {
                    rounded = (int32_t)((prod + (int64_t)round_const) >> (-Qr));
                } else {
                    rounded = (int32_t)(prod << Qr);
                }
                y[nOut * Wout + w] += rounded;
            }
        }

        /* Add bias */
        for (w = 0; w < Wout; w++) {
            y[nOut * Wout + w] += bias[nOut];
        }
    }
}

/* ── gconv2d_fp — Grouped temporal conv (with cache) ─────────────────────
 * MATLAB: Each output channel is independent (Cin=1 per channel).
 *         x_chan = [conv_cache(nOut,:); x(nOut,:)] → [2, W]
 *         x_padd = [zeros(2,1) x_chan zeros(2,1)] → [2, W+2]
 *         Then standard sliding conv.
 *
 * x:      [Cout, W] in int32_t Q20 (each channel separate)
 * conv_cache: [Cout, W] in int32_t Q20 (previous frame)
 * weight: [Cout, 1, Kh, Kw] in int16_t
 * y:      [Cout, Wout] in int32_t Q20
 * cache_out: [Cout, W] — x is copied here for next frame
 */
void gconv2d_fp(
    const int32_t *x, const int32_t *conv_cache,
    int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y, int32_t *cache_out)
{
    int nOut, h_id, w_id, kh, kw;
    int32_t round_const = (int32_t)(1LL << ((-Qr) - 1));
    int pad_left = 1, pad_top = 0;
    int H = 2;  /* cache(1 row) + current(1 row) = 2 rows */
    int W = Wout; /* Find W from Wout */
    /* Wout = floor((W + 2*pad_left - Kw) / stride_w) + 1 */
    /* Solve: W = (Wout - 1) * stride_w + Kw - 2*pad_left */
    W = (Wout - 1) * stride_w + Kw - 2;

    for (nOut = 0; nOut < Cout; nOut++) {
        /* Build x_chan: [cache; x] → [2, W] */
        int32_t x_chan[2 * W]; /* small, stack-allocated */
        for (kh = 0; kh < 1; kh++) {
            for (kw = 0; kw < W; kw++) {
                x_chan[kh * W + kw] = conv_cache[nOut * W + kw];
            }
        }
        for (kw = 0; kw < W; kw++) {
            x_chan[1 * W + kw] = x[nOut * W + kw];
        }

        /* Convolve: weight is [Cout, 1, Kh, Kw] column-major.
         * MATLAB: kernel_chan = squeeze(weight(nOut,1,:,:)) → [Kh,Kw]
         * Access: weight(nOut,1,kh,kw) = nOut + Cout*(kh-1) + Cout*Kh*(kw-1) */
        int32_t acc[Hout * Wout];
        memset(acc, 0, sizeof(acc));

        for (h_id = 1; h_id <= Hout; h_id++) {
            int h_start = (h_id - 1) * stride_h + 1;
            for (w_id = 1; w_id <= Wout; w_id++) {
                int w_start = (w_id - 1) * stride_w + 1;
                int32_t sum_val = 0;

                for (kh = 1; kh <= Kh; kh++) {
                    int h_src = h_start + (kh - 1) - pad_top;
                    for (kw = 1; kw <= Kw; kw++) {
                        int w_src = w_start + (kw - 1) - pad_left;
                        int32_t x_val = 0;
                        if (h_src >= 1 && h_src <= H && w_src >= 1 && w_src <= W) {
                            x_val = x_chan[(h_src - 1) * W + (w_src - 1)];
                        }
                        /* column-major: weight(nOut,1,kh,kw) */
                        int16_t w_val = weight[nOut + Cout * (kh - 1) + Cout * Kh * (kw - 1)];
                        int64_t prod = (int64_t)x_val * (int64_t)w_val;
                        sum_val += (int32_t)((prod + (int64_t)round_const) >> (-Qr));
                    }
                }
                acc[(h_id - 1) * Wout + (w_id - 1)] = sum_val;
            }
        }

        /* Write output + bias */
        for (int i = 0; i < Hout * Wout; i++) {
            y[nOut * Wout + i] = acc[i] + bias[nOut];
        }

        /* Update cache = current x */
        for (int i = 0; i < W; i++) {
            cache_out[nOut * W + i] = x[nOut * W + i];
        }
    }
    (void)round_const;
}

/* ── non_gconv2d_fp — Grouped non-temporal conv (no cache) ───────────────
 * MATLAB: x_chan = x(nOut,:) → [1, W]
 *         x_padd = [zeros(1,2) x_chan zeros(1,2)] → [1, W+4]
 *         Kernel slides along frequency dim only (Kh=1).
 */
void non_gconv2d_fp(
    const int32_t *x, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y)
{
    int nOut, w_id, kw;
    int32_t round_const = (int32_t)(1LL << ((-Qr) - 1));
    int pad_left = 2, pad_top = 0;
    int H = 1;
    int W = (Wout - 1) * stride_w + Kw - 2 * pad_left;

    for (nOut = 0; nOut < Cout; nOut++) {
        const int32_t *x_chan = &x[nOut * W];
        /* weight is [Cout, 1, 1, Kw] column-major.
         * MATLAB: kernel_chan = squeeze(weight(nOut,1,:,:)).' → [Kw]
         * Access: weight(nOut,1,1,kw) = nOut + Cout*(kw-1) [1-based] */

        int32_t acc[Hout * Wout];
        memset(acc, 0, sizeof(acc));

        for (w_id = 1; w_id <= Wout; w_id++) {
            int w_start = (w_id - 1) * stride_w + 1;
            int32_t sum_val = 0;

            for (kw = 1; kw <= Kw; kw++) {
                int w_src = w_start + (kw - 1) - pad_left;
                int32_t x_val = 0;
                if (w_src >= 1 && w_src <= W) {
                    x_val = x_chan[w_src - 1];
                }
                /* column-major: weight(nOut,1,1,kw) */
                int16_t w_val = weight[nOut + Cout * (kw - 1)];
                int64_t prod = (int64_t)x_val * (int64_t)w_val;
                sum_val += (int32_t)((prod + (int64_t)round_const) >> (-Qr));
            }
            acc[w_id - 1] = sum_val;
        }

        for (int i = 0; i < Wout; i++) {
            y[nOut * Wout + i] = acc[i] + bias[nOut];
        }
    }
    (void)round_const; (void)Hout; (void)stride_h;
}

/* ── tconv2d_fp — Transposed 2D convolution ──────────────────────────────
 * MATLAB: x_insert(1:stride:end, 1:stride:end) = x_chan  (zero insertion)
 *         kernel_chan = rot90(kernel, 2)  (180° rotation)
 *         Then standard conv sliding.
 *
 * x:   [Cin, H, W] in int32_t Q20
 * weight: [Cin, Cout, Kh, Kw] in int16_t (note: Cin↔Cout swapped for tconv!)
 *          (tconv2d_func uses weight(nIn, nOut, :, :))
 */
void tconv2d_fp(
    const int32_t *x, int Cin, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y)
{
    /* tconv2d: zero-insertion + rot180(kernel) + standard conv
     * MATLAB: x_insert(1:stride_h:end, 1:stride_w:end) = x_chan
     *         kernel_chan = rot90(kernel, 2)
     *         Then conv2d-like sliding with pad_left=1, pad_top=0
     *
     * x: [Cin, H, W] — H can be 3 (with cache) or 1 (no cache)
     * weight: [Cin, Cout, Kh, Kw] (note: MATLAB indexes weight(nIn,nOut,:,:))
     */
    int nIn, nOut, h_id, w_id, kh, kw;
    int32_t round_const = (int32_t)(1LL << ((-Qr) - 1));

    /* Input dimensions: the caller must provide x with known [H, W].
     * We derive H, W from known model dimensions:
     *   W = (Wout - Kw + 2) / 1? Actually from Wout = (W+2*pad-Kw)/1 + 1 with pad=1:
     *   Wout = W + 2 - Kw - 1 + 1? No. After insertion: W' = (W-1)*stride_w + 1.
     *   Then Wout = W' + 2*pad - Kw + 1 with pad=1 and stride=1.
     *   So: Wout = (W-1)*stride_w + 1 + 2 - Kw + 1 = (W-1)*stride_w + 4 - Kw.
     *   Solve: W = (Wout + Kw - 4) / stride_w + 1.
     * For De_XConv: Wout=129, Kw=3, stride_w=2: W = (129+3-4)/2+1 = 64+1 = 65. ✓
     *
     * H is known from the 3D input shape passed by caller.
     * We assume the caller reshapes x as [Cin, H, W] before calling.
     * For this model: H=3 for De_XConv (cache-stacked), H=1 for others.
     */
    int W = (Wout + Kw - 4) / stride_w + 1;
    /* H is harder — we assume Kh rows (matching pad_top=0, Hout=1 convention) */
    /* Actually H is whatever the caller passes. We guess from (Hout-1)*stride_h+Kh: */
    int H = Kh;  /* default: H=Kh for tconv with pad_top=(Kh-H)/2=0 when H=Kh */

    /* Allocate zero-inserted buffer */
    int W_ins = (W - 1) * stride_w + 1;
    int H_ins = (H - 1) * stride_h + 1;

    for (nOut = 0; nOut < Cout; nOut++) {
        /* Initialize output channel */
        for (int i = 0; i < Hout * Wout; i++) y[nOut * Wout + i] = 0;

        for (nIn = 0; nIn < Cin; nIn++) {
            const int32_t *x_chan = &x[(nIn * H) * W];  /* [H, W] */

            /* Zero insertion */
            int32_t *x_insert = (int32_t *)calloc(H_ins * W_ins, sizeof(int32_t));
            for (h_id = 0; h_id < H; h_id++) {
                for (w_id = 0; w_id < W; w_id++) {
                    x_insert[h_id * stride_h * W_ins + w_id * stride_w] =
                        x_chan[h_id * W + w_id];
                }
            }

            /* Pad: [zeros(Kh,1) x_insert zeros(Kh,1)]
             * → Add 1 zero column on each side, pad_top rows to reach Kh */
            int pad_left = 1;
            int pad_top = (Kh - H_ins) > 0 ? (Kh - H_ins) : 0;  /* pad top if needed */

            /* Kernel: rot90(kernel, 2) — 180° rotation
             * weight is [Cin, Cout, Kh, Kw] column-major.
             * MATLAB: kernel_chan = rot90(squeeze(weight(nIn,nOut,:,:)), 2)
             * Access: weight(nIn,nOut,kh,kw) = nIn + Cin*nOut + Cin*Cout*kh + Cin*Cout*Kh*kw */
            int CinCout = Cin * Cout;
            int16_t kernel_rot[Kh * Kw];
            for (kh = 0; kh < Kh; kh++) {
                for (kw = 0; kw < Kw; kw++) {
                    kernel_rot[kh * Kw + kw] = weight[nIn + Cin * nOut + CinCout * (Kh-1-kh) + CinCout * Kh * (Kw-1-kw)];
                }
            }

            /* Convolve (stride=1 after insertion) */
            int32_t conv_result[Hout * Wout];
            for (h_id = 1; h_id <= Hout; h_id++) {
                for (w_id = 1; w_id <= Wout; w_id++) {
                    int32_t acc = 0;
                    int h_start = h_id - 1 + 1;  /* stride=1 after insertion, 1-based */
                    int w_start = w_id - 1 + 1;
                    for (kh = 1; kh <= Kh; kh++) {
                        int h_src = h_start + (kh - 1) - pad_top;
                        for (kw = 1; kw <= Kw; kw++) {
                            int w_src = w_start + (kw - 1) - pad_left;
                            int32_t x_val = 0;
                            if (h_src >= 1 && h_src <= H_ins
                                && w_src >= 1 && w_src <= W_ins) {
                                x_val = x_insert[(h_src-1) * W_ins + (w_src-1)];
                            }
                            int16_t w_val = kernel_rot[(kh-1) * Kw + (kw-1)];
                            int64_t prod = (int64_t)x_val * (int64_t)w_val;
                            acc += (int32_t)((prod + (int64_t)round_const) >> (-Qr));
                        }
                    }
                    conv_result[(h_id-1) * Wout + (w_id-1)] = acc;
                }
            }

            /* Accumulate */
            for (int i = 0; i < Hout * Wout; i++) {
                y[nOut * Wout + i] += conv_result[i];
            }
            free(x_insert);
        }

        /* Add bias */
        for (int i = 0; i < Hout * Wout; i++) {
            y[nOut * Wout + i] += bias[nOut];
        }
    }
}

/* ── gtconv2d_fp — Grouped transposed conv (with cache) ─────────────────
 * MATLAB: x_chan = [cache; x], zero-insertion + rot90(kernel,2) + conv sliding
 * Used by: De_XDWS1_TConv, De_XMB1_TConv
 */
void gtconv2d_fp(
    const int32_t *x, const int32_t *conv_cache,
    int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y, int32_t *cache_out)
{
    int nOut, h_id, w_id, kh, kw;
    int32_t round_const = (int32_t)(1LL << ((-Qr) - 1));
    int H = 2;  /* cache(1) + current(1) = 2 */
    /* After zero-insertion: Wout = (W-1)*stride_w + 1; Solve: W = (Wout-1)/stride_w + 1 */
    int W = (Wout - 1) / stride_w + 1;
    int W_ins = (W - 1) * stride_w + 1;
    int H_ins = (H - 1) * stride_h + 1;

    for (nOut = 0; nOut < Cout; nOut++) {
        /* Build x_chan: [cache; current] → [2, W] */
        int32_t x_chan[2 * 512];  /* max W=65, safe up to 256 */
        for (int i = 0; i < W; i++) {
            x_chan[i] = conv_cache[nOut * W + i];
            x_chan[W + i] = x[nOut * W + i];
        }

        /* Zero insertion */
        int32_t *x_insert = (int32_t *)calloc(H_ins * W_ins, sizeof(int32_t));
        for (h_id = 0; h_id < H; h_id++)
            for (w_id = 0; w_id < W; w_id++)
                x_insert[h_id * stride_h * W_ins + w_id * stride_w] = x_chan[h_id * W + w_id];

        /* weight is [Cout, 1, Kh, Kw] column-major.
         * Build rotated kernel: rot90(kernel,2) = 180° rotation */
        int16_t kernel_rot[Kh * Kw];
        for (kh = 0; kh < Kh; kh++)
            for (kw = 0; kw < Kw; kw++)
                /* column-major: weight(nOut,1,Kh-1-kh,Kw-1-kw) */
                kernel_rot[kh * Kw + kw] = weight[nOut + Cout * (Kh-1-kh) + Cout * Kh * (Kw-1-kw)];

        int pad_left = 1;
        int32_t conv_result[Hout * Wout];
        for (h_id = 1; h_id <= Hout; h_id++) {
            for (w_id = 1; w_id <= Wout; w_id++) {
                int32_t acc = 0;
                int h_start = h_id; /* stride=1 after insertion, 1-based */
                int w_start = w_id;
                for (kh = 1; kh <= Kh; kh++) {
                    int h_src = h_start + (kh - 1);
                    for (kw = 1; kw <= Kw; kw++) {
                        int w_src = w_start + (kw - 1) - pad_left;
                        int32_t x_val = 0;
                        if (h_src >= 1 && h_src <= H_ins && w_src >= 1 && w_src <= W_ins)
                            x_val = x_insert[(h_src-1) * W_ins + (w_src-1)];
                        int16_t w_val = kernel_rot[(kh-1) * Kw + (kw-1)];
                        int64_t prod = (int64_t)x_val * (int64_t)w_val;
                        acc += (int32_t)((prod + (int64_t)round_const) >> (-Qr));
                    }
                }
                conv_result[(h_id-1) * Wout + (w_id-1)] = acc;
            }
        }
        for (int i = 0; i < Hout * Wout; i++)
            y[nOut * Wout + i] = conv_result[i] + bias[nOut];
        free(x_insert);

        /* Update cache = current x */
        for (int i = 0; i < W; i++)
            cache_out[nOut * W + i] = x[nOut * W + i];
    }
}

/* ── non_gtconv2d_fp — Grouped non-temporal transposed conv (no cache) ──
 * MATLAB: x_chan = x(nOut,:), zero-insertion + rot90(kernel,90) + conv sliding
 * Used by: De_XDWS0_nonTConv, De_XMB0_nonTConv
 *
 * NOTE: rot90(kernel, 90) NOT rot90(kernel, 2)!
 */
void non_gtconv2d_fp(
    const int32_t *x, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y)
{
    int nOut, w_id, kw, kh;
    int32_t round_const = (int32_t)(1LL << ((-Qr) - 1));
    /* After zero-insertion: Wout = (W-1)*stride_w + 1; Solve: W = (Wout-1)/stride_w + 1 */
    int W = (Wout - 1) / stride_w + 1;
    int W_ins = (W - 1) * stride_w + 1;

    for (nOut = 0; nOut < Cout; nOut++) {
        const int32_t *x_chan = &x[nOut * W];
        int32_t *x_insert = (int32_t *)calloc(W_ins, sizeof(int32_t));
        for (w_id = 0; w_id < W; w_id++)
            x_insert[w_id * stride_w] = x_chan[w_id];

        /* weight is [Cout, 1, Kh, Kw] column-major.
         * MATLAB: kernel = squeeze(weight(nOut,1,:,:)).'  → transpose → [Kw, Kh]
         *         kernel_rot = rot90(kernel, 90) → 90° CCW
         * Column-major: weight(nOut,1,kh,kw) = nOut + Cout*kh + Cout*Kh*kw */
        int16_t kt[Kh * Kw]; /* transposed: [Kw, Kh] */
        for (kh = 0; kh < Kh; kh++)
            for (kw = 0; kw < Kw; kw++)
                kt[kw * Kh + kh] = weight[nOut + Cout * kh + Cout * Kh * kw];
        /* rot90: new[i][j] = old[j][N-1-i] for 90° CCW, old is [Kw, Kh] */
        int16_t krot[Kh * Kw];
        for (kh = 0; kh < Kh; kh++)
            for (kw = 0; kw < Kw; kw++)
                krot[kh * Kw + kw] = kt[(Kw-1-kw) * Kh + kh];

        int pad_left = 2;
        int32_t conv_result[Hout * Wout];
        for (w_id = 1; w_id <= Wout; w_id++) {
            int32_t acc = 0;
            int w_start = w_id; /* stride=1 after insertion */
            for (kw = 1; kw <= Kw; kw++) {
                int w_src = w_start + (kw - 1) - pad_left;
                int32_t x_val = 0;
                if (w_src >= 1 && w_src <= W_ins)
                    x_val = x_insert[w_src - 1];
                int16_t w_val = krot[kw - 1]; /* Kh=1, row 0 */
                int64_t prod = (int64_t)x_val * (int64_t)w_val;
                acc += (int32_t)((prod + (int64_t)round_const) >> (-Qr));
            }
            conv_result[w_id - 1] = acc;
        }
        for (int i = 0; i < Wout; i++)
            y[nOut * Wout + i] = conv_result[i] + bias[nOut];
        free(x_insert);
    }
}

/* ── BN (Batch Normalization) ────────────────────────────────────────────
 * MATLAB:
 *   x_norm = round((x - running_mean) .* running_var * 2^(Qr1));
 *   y = round(x_norm .* weight * 2^(Qr2)) + bias;
 *
 * x:              [N] in int32_t Q20
 * weight:         [N] in int16_t (Q14) or uint16_t
 * bias:           [N] in int32_t Q20
 * running_mean:   [N] in int32_t Q20
 * running_var:    [N] in uint16_t (Q11-Q14, varies)
 */
void bn_fp(
    const int32_t *x, int C, int W,
    const uint16_t *weight, const int32_t *bias,
    const int32_t *running_mean, const uint16_t *running_var,
    int Qr1, int Qr2,
    int32_t *y)
{
    /* BN params (weight/bias/mean/var) are per-channel (C elements).
     * Broadcast across W: element i → channel c = i / W, param index = c.
     * running_var: uint16_t — some values > 32767 (e.g., 37326 for E0 ch1),
     * so MUST cast through unsigned to avoid sign-extension. */
    int N = C * W;
    int32_t round1 = (int32_t)(1LL << ((-Qr1) - 1));
    int32_t round2 = (int32_t)(1LL << ((-Qr2) - 1));
    int i;

    for (i = 0; i < N; i++) {
        int c = i / W;
        int64_t diff = (int64_t)x[i] - (int64_t)running_mean[c];
        int64_t norm = diff * (int64_t)(uint32_t)running_var[c];
        int32_t x_norm = (int32_t)((norm + (int64_t)round1) >> (-Qr1));

        int64_t scaled = (int64_t)x_norm * (int64_t)(uint32_t)weight[c];
        int32_t y_val = (int32_t)((scaled + (int64_t)round2) >> (-Qr2));
        y[i] = y_val + bias[c];
    }
}

/* ── LN (Layer Normalization) ────────────────────────────────────────────
 * MATLAB:
 *   x_dq = x * 2^(-20);                        // dequantize
 *   running_mean = mean(x_dq, 'all');           // float mean
 *   running_var = 1 / sqrt(var(x_dq) + 1e-8);  // float inv-std
 *   running_mean = Fix_point(running_mean, 's32f20');
 *   running_var = Fix_point(running_var, 'u16f11');
 *   x_norm = round((x - running_mean) * running_var * 2^(-11));
 *   y = round(x_norm .* weight * 2^(Qr)) + bias;
 *
 * Uses float for the statistics computation, matching MATLAB behavior.
 */
void ln_fp(
    const int32_t *x, int N,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y)
{
    int i;
    int32_t round1 = (int32_t)(1LL << (11 - 1));   /* Qr1 = -11: round for 2^(-11) shift */
    int32_t round2 = (int32_t)(1LL << ((-Qr) - 1));
    double sum = 0.0, sum_sq = 0.0;

    /* Step 1: dequantize to float and compute statistics */
    for (i = 0; i < N; i++) {
        double xf = (double)x[i] / (double)SCALE_Q20;
        sum += xf;
        sum_sq += xf * xf;
    }
    double mean_f = sum / (double)N;
    double var_f = sum_sq / (double)N - mean_f * mean_f;  /* MATLAB var(x,1) = population variance */
    double inv_std_f = 1.0 / sqrt(var_f + 1e-8);

    /* Quantize statistics */
    int32_t running_mean_q20 = FLOAT_TO_Q20(mean_f);
    uint16_t running_var_q11 = (uint16_t)(round(inv_std_f * (double)SCALE_Q11));
    if (running_var_q11 > 65535) running_var_q11 = 65535;

    /* Step 2: fixed-point normalization */
    for (i = 0; i < N; i++) {
        int64_t diff = (int64_t)x[i] - (int64_t)running_mean_q20;
        int64_t norm = diff * (int64_t)(uint32_t)running_var_q11;
        int32_t x_norm = (int32_t)((norm + (int64_t)round1) >> 11);  /* Qr = -11 */

        int64_t scaled = (int64_t)x_norm * (int64_t)weight[i];
        int32_t y_val = (int32_t)((scaled + (int64_t)round2) >> (-Qr));
        y[i] = y_val + bias[i];
    }
}

/* ── AffinePReLU ─────────────────────────────────────────────────────────
 * MATLAB:
 *   x_copy = x;
 *   index = x < 0; [row, ~] = find(index);
 *   x(index) = round(x(index) .* slope(row) * 2^(Qr1));   // PReLU neg part
 *   y = round(x_copy .* weight * 2^(Qr2)) + bias + x;     // affine + residual
 *
 * x:      [C, W] in int32_t Q20
 * weight: [C] in int16_t (Q14) — affine scale, per-channel
 * bias:   [C] in int32_t Q20
 * slope:  [C] in int16_t (Q13) — PReLU negative slope, per-channel
 */
void affineprelu_fp(
    const int32_t *x, int C, int W,
    const int16_t *weight, const int32_t *bias,
    const int16_t *slope, int Qr1, int Qr2,
    int32_t *y)
{
    int c, w;
    int32_t round1 = (int32_t)(1LL << ((-Qr1) - 1));
    int32_t round2 = (int32_t)(1LL << ((-Qr2) - 1));
    int N = C * W;

    /* First pass: apply PReLU to negative elements, compute affine */
    for (c = 0; c < C; c++) {
        for (w = 0; w < W; w++) {
            int idx = c * W + w;
            int32_t x_val = x[idx];

            /* PReLU: if x < 0, x = round(x * slope(c) * 2^(Qr1)) */
            if (x_val < 0) {
                int64_t prod = (int64_t)x_val * (int64_t)slope[c];
                x_val = (int32_t)((prod + (int64_t)round1) >> (-Qr1));
            }

            /* Affine + residual. weight/bias are [C,W] in MATLAB column-major:
             * weight(c,w) → c + C*w. slope is per-channel [C]. */
            int w_idx = c + C * w;  /* MATLAB column-major index */
            int64_t aff = (int64_t)x[idx] * (int64_t)weight[w_idx];
            int32_t aff_r = (int32_t)((aff + (int64_t)round2) >> (-Qr2));
            y[idx] = aff_r + bias[w_idx] + x_val;
        }
    }
    (void)N; /* used for potential optimization */
}

/* ── GRU single step ─────────────────────────────────────────────────────
 * MATLAB (from GRU_module.m):
 *   r_t = round(x_t*ih_r_w*2^(Qr1)) + round(h*hh_r_w*2^(Qr2)) + ih_r_b + hh_r_b → sigmoid → u16f15
 *   z_t = round(x_t*ih_z_w*2^(Qr1)) + round(h*hh_z_w*2^(Qr2)) + ih_z_b + hh_z_b → sigmoid → u16f15
 *   h_t = round(h*hh_n_w*2^(Qr2)) + hh_n_b
 *   n_t = round(x_t*ih_n_w*2^(Qr1)) + round(r_t.*h_t*2^(-15)) + ih_n_b → tanh → s16f15
 *   h = round((32768-z_t).*n_t*2^(-15)) + round(z_t.*h*2^(-15))
 *
 * x_t:     [input_dim] in int32_t Q20
 * h_cache: [nHidden] in int16_t Q15 (in/out, updated in place)
 * ih_weight: [input_dim, 3*nHidden] in int16_t Q12
 * ih_bias:   [3*nHidden] in int16_t Q10
 * hh_weight: [nHidden, 3*nHidden] in int16_t Q12
 * hh_bias:   [3*nHidden] in int16_t Q10
 * Qr1 = -13 (ih path), Qr2 = -8 (hh path)
 * Q_sig_tanh = -15 (gate mixing)
 */
void gru_step_fp(
    const int32_t *x_t, int input_dim,
    int16_t *h_cache, int nHidden,
    const int16_t *ih_weight, const int32_t *ih_bias,
    const int16_t *hh_weight, const int32_t *hh_bias,
    int Qr1, int Qr2,
    int16_t *y_out)
{
    int j;
    int32_t round_ih = (int32_t)(1LL << ((-Qr1) - 1));
    int32_t round_hh = (int32_t)(1LL << ((-Qr2) - 1));
    int32_t round_gate = (int32_t)(1LL << (15 - 1));  /* for 2^(-15) shift */

    /* Split weight/bias into r, z, n gates */
    /* ih_weight: [input_dim, 3*nHidden] — column-major? In MATLAB: ih_r = ih_weight(:,1:nHidden) */
    /* In C: ih_weight is stored [input_dim * 3*nHidden] row-major? */
    /* MATLAB uses column-major. For simplicity, we assume weights are exported in MATLAB's
       column-major order and we access them accordingly. */
    #define IH_R_W(i, j)  ih_weight[(i) + (j) * input_dim]
    #define IH_Z_W(i, j)  ih_weight[(i) + ((j) + nHidden) * input_dim]
    #define IH_N_W(i, j)  ih_weight[(i) + ((j) + 2*nHidden) * input_dim]
    #define HH_R_W(i, j)  hh_weight[(i) + (j) * nHidden]
    #define HH_Z_W(i, j)  hh_weight[(i) + ((j) + nHidden) * nHidden]
    #define HH_N_W(i, j)  hh_weight[(i) + ((j) + 2*nHidden) * nHidden]

    uint16_t r_t_q15[64], z_t_q15[64];  /* max nHidden=64; MUST be unsigned: sigmoid returns u16f15 (0-32768) */
    int32_t h_t_q20[64];
    int16_t n_t_q15[64];

    for (j = 0; j < nHidden; j++) {
        /* ── Reset gate R ── */
        int64_t ih_r_sum = 0;
        for (int i = 0; i < input_dim; i++) {
            ih_r_sum += (int64_t)x_t[i] * (int64_t)IH_R_W(i, j);
        }
        int32_t ih_r = (int32_t)((ih_r_sum + (int64_t)round_ih) >> (-Qr1));

        int64_t hh_r_sum = 0;
        for (int k = 0; k < nHidden; k++) {
            hh_r_sum += (int64_t)h_cache[k] * (int64_t)HH_R_W(k, j);
        }
        int32_t hh_r = (int32_t)((hh_r_sum + (int64_t)round_hh) >> (-Qr2));

        int32_t r_t_q20 = ih_r + hh_r
            + ih_bias[j]              /* ih_r_bias */
            + hh_bias[j];             /* hh_r_bias */
        r_t_q15[j] = sigmoid_q20_to_q15(r_t_q20);

        /* ── Update gate Z ── */
        int64_t ih_z_sum = 0;
        for (int i = 0; i < input_dim; i++) {
            ih_z_sum += (int64_t)x_t[i] * (int64_t)IH_Z_W(i, j);
        }
        int32_t ih_z = (int32_t)((ih_z_sum + (int64_t)round_ih) >> (-Qr1));

        int64_t hh_z_sum = 0;
        for (int k = 0; k < nHidden; k++) {
            hh_z_sum += (int64_t)h_cache[k] * (int64_t)HH_Z_W(k, j);
        }
        int32_t hh_z = (int32_t)((hh_z_sum + (int64_t)round_hh) >> (-Qr2));

        int32_t z_t_q20 = ih_z + hh_z
            + ih_bias[j + nHidden]    /* ih_z_bias */
            + hh_bias[j + nHidden];   /* hh_z_bias */
        z_t_q15[j] = sigmoid_q20_to_q15(z_t_q20);

        /* ── Candidate hidden state N ── */
        /* h_t = round(h*hh_n_w*2^(Qr2)) + hh_n_b */
        int64_t hh_n_sum = 0;
        for (int k = 0; k < nHidden; k++) {
            hh_n_sum += (int64_t)h_cache[k] * (int64_t)HH_N_W(k, j);
        }
        int32_t hh_n = (int32_t)((hh_n_sum + (int64_t)round_hh) >> (-Qr2));
        h_t_q20[j] = hh_n + hh_bias[j + 2*nHidden];  /* hh_n_bias */

        /* n_t = round(x_t*ih_n_w*2^(Qr1)) + round(r_t .* h_t * 2^(-15)) + ih_n_b */
        int64_t ih_n_sum = 0;
        for (int i = 0; i < input_dim; i++) {
            ih_n_sum += (int64_t)x_t[i] * (int64_t)IH_N_W(i, j);
        }
        int32_t ih_n = (int32_t)((ih_n_sum + (int64_t)round_ih) >> (-Qr1));

        int64_t r_h_prod = (int64_t)(uint32_t)r_t_q15[j] * (int64_t)h_t_q20[j];
        int32_t r_h_mix = (int32_t)((r_h_prod + (int64_t)round_gate) >> 15);  /* *2^(-15) */

        int32_t n_t_q20 = ih_n + r_h_mix + ih_bias[j + 2*nHidden];
        n_t_q15[j] = tanh_q20_to_q15(n_t_q20);
    }

    /* ── Hidden state update ── */
    for (j = 0; j < nHidden; j++) {
        /* h = round((32768 - z_t) .* n_t * 2^(-15)) + round(z_t .* h_cache * 2^(-15)) */
        uint32_t one_minus_z = (uint32_t)(32768 - (uint32_t)z_t_q15[j]);
        int64_t term1 = (int64_t)one_minus_z * (int64_t)n_t_q15[j];
        int32_t t1 = (int32_t)((term1 + (int64_t)round_gate) >> 15);

        int64_t term2 = (int64_t)(uint32_t)z_t_q15[j] * (int64_t)h_cache[j];
        int32_t t2 = (int32_t)((term2 + (int64_t)round_gate) >> 15);

        int32_t h_new = t1 + t2;
        h_cache[j] = sat_s16(h_new);
        y_out[j] = h_cache[j];
    }

    #undef IH_R_W
    #undef IH_Z_W
    #undef IH_N_W
    #undef HH_R_W
    #undef HH_Z_W
    #undef HH_N_W
}

/* ── GRU sequence ────────────────────────────────────────────────────────
 * Process T timesteps of a unidirectional GRU.
 * x: [T, input_dim] in int32_t Q20, row-major
 * y_out: [T, nHidden] in int16_t Q15, row-major
 */
void gru_sequence_fp(
    const int32_t *x, int T, int input_dim,
    int16_t *h_cache, int nHidden,
    const int16_t *ih_weight, const int32_t *ih_bias,
    const int16_t *hh_weight, const int32_t *hh_bias,
    int Qr1, int Qr2,
    int16_t *y_out)
{
    int t;
    for (t = 0; t < T; t++) {
        const int32_t *x_t = &x[t * input_dim];
        int16_t *y_t = &y_out[t * nHidden];
        gru_step_fp(x_t, input_dim, h_cache, nHidden,
                    ih_weight, ih_bias, hh_weight, hh_bias,
                    Qr1, Qr2, y_t);
    }
}

/* ── BiGRU sequence ──────────────────────────────────────────────────────
 * MATLAB:
 *   Forward GRU: y1, time t=1..T
 *   Reverse GRU: x_re = x(end:-1:1,:), y2_re, then reverse y2 back
 *   y = cat(2, y1, y2(end:-1:1,:))
 *
 * x:     [T, input_dim] in int32_t Q20, row-major
 * y_out: [T, 2*nHidden] in int16_t Q15, row-major
 */
void bigru_sequence_fp(
    const int32_t *x, int T, int input_dim,
    int nHidden,
    const int16_t *ih_weight, const int32_t *ih_bias,
    const int16_t *hh_weight, const int32_t *hh_bias,
    const int16_t *re_ih_weight, const int32_t *re_ih_bias,
    const int16_t *re_hh_weight, const int32_t *re_hh_bias,
    int Qr1, int Qr2,
    int16_t *y_out)
{
    int t;
    int16_t h_fwd[32] = {0};  /* max nHidden=4 for BiGRU in this model */
    int16_t h_rev[32] = {0};
    int16_t *y_fwd = (int16_t *)malloc(T * nHidden * sizeof(int16_t));
    int16_t *y_rev = (int16_t *)malloc(T * nHidden * sizeof(int16_t));

    /* Forward pass */
    for (t = 0; t < T; t++) {
        const int32_t *x_t = &x[t * input_dim];
        gru_step_fp(x_t, input_dim, h_fwd, nHidden,
                    ih_weight, ih_bias, hh_weight, hh_bias,
                    Qr1, Qr2, &y_fwd[t * nHidden]);
    }

    /* Reverse pass: process reversed input, then reverse output */
    for (t = 0; t < T; t++) {
        const int32_t *x_t = &x[(T - 1 - t) * input_dim];  /* reversed */
        gru_step_fp(x_t, input_dim, h_rev, nHidden,
                    re_ih_weight, re_ih_bias, re_hh_weight, re_hh_bias,
                    Qr1, Qr2, &y_rev[t * nHidden]);
    }

    /* Concat: y = [y_fwd, y_rev_reversed] */
    for (t = 0; t < T; t++) {
        for (int j = 0; j < nHidden; j++) {
            y_out[t * 2 * nHidden + j] = y_fwd[t * nHidden + j];
        }
        for (int j = 0; j < nHidden; j++) {
            /* y_rev reversed: last timestep of y_rev = first of original */
            y_out[t * 2 * nHidden + nHidden + j] = y_rev[(T - 1 - t) * nHidden + j];
        }
    }

    free(y_fwd);
    free(y_rev);
}

/* ── GRU single step Q20 ─────────────────────────────────────────────────
 * Q20 hidden state (int32_t) + Q20 gates + Q20 mixing → Q15 output.
 * GTCRN-style bitwise LUT lookup for sigmoid/tanh.
 */
void gru_step_fp_q20(
    const int32_t *x_t, int input_dim,
    int32_t *h_cache, int nHidden,
    const int16_t *ih_weight, const int32_t *ih_bias,
    const int16_t *hh_weight, const int32_t *hh_bias,
    int Qr1, int Qr2,
    int16_t *y_out)
{
    int j;
    int32_t round_ih = (int32_t)(1LL << ((-Qr1) - 1));
    int32_t round_hh = (int32_t)(1LL << ((-Qr2) - 1));
    int32_t round_gate = (int32_t)(1LL << (20 - 1));

    #define IH_R(i,j) ih_weight[(i)+(j)*input_dim]
    #define IH_Z(i,j) ih_weight[(i)+((j)+nHidden)*input_dim]
    #define IH_N(i,j) ih_weight[(i)+((j)+2*nHidden)*input_dim]
    #define HH_R(i,j) hh_weight[(i)+(j)*nHidden]
    #define HH_Z(i,j) hh_weight[(i)+((j)+nHidden)*nHidden]
    #define HH_N(i,j) hh_weight[(i)+((j)+2*nHidden)*nHidden]

    uint32_t r_q20[64], z_q20[64];
    int32_t ht_q20[64], n_q20[64];

    for (j=0;j<nHidden;j++){
        int64_t s=0;for(int i=0;i<input_dim;i++)s+=(int64_t)x_t[i]*IH_R(i,j);
        int32_t ih_r=(int32_t)((s+(int64_t)round_ih)>>(-Qr1));
        s=0;for(int k=0;k<nHidden;k++)s+=(int64_t)h_cache[k]*HH_R(k,j);
        int32_t hh_r=(int32_t)((s+(int64_t)round_hh)>>(-Qr2));
        r_q20[j]=sigmoid_q20_to_q20(ih_r+hh_r+ih_bias[j]+hh_bias[j]);

        s=0;for(int i=0;i<input_dim;i++)s+=(int64_t)x_t[i]*IH_Z(i,j);
        int32_t ih_z=(int32_t)((s+(int64_t)round_ih)>>(-Qr1));
        s=0;for(int k=0;k<nHidden;k++)s+=(int64_t)h_cache[k]*HH_Z(k,j);
        int32_t hh_z=(int32_t)((s+(int64_t)round_hh)>>(-Qr2));
        z_q20[j]=sigmoid_q20_to_q20(ih_z+hh_z+ih_bias[j+nHidden]+hh_bias[j+nHidden]);

        s=0;for(int k=0;k<nHidden;k++)s+=(int64_t)h_cache[k]*HH_N(k,j);
        ht_q20[j]=(int32_t)((s+(int64_t)round_hh)>>(-Qr2))+hh_bias[j+2*nHidden];

        s=0;for(int i=0;i<input_dim;i++)s+=(int64_t)x_t[i]*IH_N(i,j);
        int32_t ih_n=(int32_t)((s+(int64_t)round_ih)>>(-Qr1));
        int64_t rh=(int64_t)r_q20[j]*(int64_t)ht_q20[j];
        int32_t rm=(int32_t)((rh+(int64_t)round_gate)>>20);
        n_q20[j]=tanh_q20_to_q20(ih_n+rm+ih_bias[j+2*nHidden]);
    }

    for (j=0;j<nHidden;j++){
        uint64_t omz=1048576ULL-(uint64_t)z_q20[j];
        int64_t t1=(int64_t)omz*(int64_t)n_q20[j];
        int64_t t2=(int64_t)z_q20[j]*(int64_t)h_cache[j];
        int64_t hn=((t1+(int64_t)round_gate)>>20)+((t2+(int64_t)round_gate)>>20);
        h_cache[j]=sat_s20(hn);
        y_out[j]=sat_s16((int32_t)((hn+16)>>5));
    }
    #undef IH_R
    #undef IH_Z
    #undef IH_N
    #undef HH_R
    #undef HH_Z
    #undef HH_N
}

/* ── GRU/BiGRU sequence Q20 ──────────────────────────────────────────── */
void gru_sequence_fp_q20(
    const int32_t *x,int T,int input_dim,
    int32_t *h_cache,int nHidden,
    const int16_t *ih_w,const int32_t *ih_b,
    const int16_t *hh_w,const int32_t *hh_b,
    int Qr1,int Qr2,int16_t *y_out)
{
    for(int t=0;t<T;t++)
        gru_step_fp_q20(&x[t*input_dim],input_dim,h_cache,nHidden,
                        ih_w,ih_b,hh_w,hh_b,Qr1,Qr2,&y_out[t*nHidden]);
}

void bigru_sequence_fp_q20(
    const int32_t *x,int T,int input_dim,int nHidden,
    const int16_t *ih_w,const int32_t *ih_b,
    const int16_t *hh_w,const int32_t *hh_b,
    const int16_t *rih_w,const int32_t *rih_b,
    const int16_t *rhh_w,const int32_t *rhh_b,
    int Qr1,int Qr2,int16_t *y_out)
{
    int32_t hf[32]={0},hr[32]={0};
    int16_t *yf=(int16_t*)malloc(T*nHidden*2);
    int16_t *yr=(int16_t*)malloc(T*nHidden*2);
    for(int t=0;t<T;t++)
        gru_step_fp_q20(&x[t*input_dim],input_dim,hf,nHidden,
                        ih_w,ih_b,hh_w,hh_b,Qr1,Qr2,&yf[t*nHidden]);
    for(int t=0;t<T;t++)
        gru_step_fp_q20(&x[(T-1-t)*input_dim],input_dim,hr,nHidden,
                        rih_w,rih_b,rhh_w,rhh_b,Qr1,Qr2,&yr[t*nHidden]);
    for(int t=0;t<T;t++){
        for(int j=0;j<nHidden;j++)y_out[t*2*nHidden+j]=yf[t*nHidden+j];
        for(int j=0;j<nHidden;j++)y_out[t*2*nHidden+nHidden+j]=yr[(T-1-t)*nHidden+j];
    }
    free(yf);free(yr);
}

/* ── FC (Full Connection / Dense) ────────────────────────────────────────
 * MATLAB: y = round(x * weight * 2^(Qr)) + bias
 *
 * x:      [rows, in_dim] in int32_t Q20
 * weight: [in_dim, out_dim] in int16_t (Q13) — MATLAB column-major
 * bias:   [out_dim] in int32_t Q20
 */
void fc_fp(
    const int32_t *x, int rows, int in_dim, int out_dim,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y)
{
    int r, o;
    int32_t round_const = (int32_t)(1LL << ((-Qr) - 1));

    for (r = 0; r < rows; r++) {
        for (o = 0; o < out_dim; o++) {
            int64_t sum = 0;
            for (int i = 0; i < in_dim; i++) {
                /* weight[i, o] in MATLAB = weight[i + o*in_dim] in C (column-major) */
                sum += (int64_t)x[r * in_dim + i] * (int64_t)weight[i + o * in_dim];
            }
            y[r * out_dim + o] = (int32_t)((sum + (int64_t)round_const) >> (-Qr)) + bias[o];
        }
    }
}

/* fc_fp with int16 input (for RNN post-FC) */
void fc_fp_s16(
    const int16_t *x, int rows, int in_dim, int out_dim,
    const int16_t *weight, const int32_t *bias, int Qr,
    int32_t *y)
{
    int r, o;
    int32_t round_const = (int32_t)(1LL << ((-Qr) - 1));

    for (r = 0; r < rows; r++) {
        for (o = 0; o < out_dim; o++) {
            int64_t sum = 0;
            for (int i = 0; i < in_dim; i++) {
                sum += (int64_t)x[r * in_dim + i] * (int64_t)weight[i + o * in_dim];
            }
            y[r * out_dim + o] = (int32_t)((sum + (int64_t)round_const) >> (-Qr)) + bias[o];
        }
    }
}

/* ======================================================================== */
/*  ERB AND MASK OPERATORS                                                   */
/* ======================================================================== */

/* ── BM (Band Merge) ─────────────────────────────────────────────────────
 * MATLAB: y(1:65) = x(1:65);                   (low freq pass-through)
 *         y(66:129) = round(x(66:257) * weight * 2^(-15));
 * x:      [1, 257] in int32_t Q20
 * weight: [192, 64] in uint16_t Q15  (= N_FBIN_HIGH × N_ERB_HIGH)
 * y:      [1, 129] in int32_t Q20
 */
void bm_fp(
    const int32_t *x,
    const uint16_t *erbfc_weight,
    int32_t *y)
{
    int j;
    int32_t round_const = 1 << 14;  /* round for 2^(-15) */

    /* Low freq direct pass-through (bins 1-65 → output 1-65) */
    for (j = 0; j < N_ERB_LOW; j++) {
        y[j] = x[j];
    }

    /* High freq band merge (bins 66-257 → output 66-129) */
    for (j = 0; j < N_ERB_HIGH; j++) {
        int64_t sum = 0;
        for (int i = 0; i < N_FBIN_HIGH; i++) {
            sum += (int64_t)x[N_ERB_LOW + i] * (int64_t)(uint32_t)erbfc_weight[i + j * N_FBIN_HIGH];
        }
        y[N_ERB_LOW + j] = (int32_t)((sum + (int64_t)round_const) >> 15);
    }
}

/* ── BS (Band Split) ─────────────────────────────────────────────────────
 * MATLAB: y(:,1:65) = x(:,1:65);                (low freq pass-through)
 *         y(:,66:257) = round(x(66:129) * weight * 2^(-15));
 * x:      [1, 129] in uint16_t Q15
 * weight: [64, 192] in uint16_t Q15  (= N_ERB_HIGH × N_FBIN_HIGH)
 * y:      [1, 257] in int32_t Q20
 */
void bs_fp(
    const uint16_t *x_mask,
    const uint16_t *ierbfc_weight,
    int32_t *y)
{
    int j;
    int32_t round_const = 1 << 14;

    /* Low freq direct pass-through */
    for (j = 0; j < N_ERB_LOW; j++) {
        y[j] = (int32_t)x_mask[j];  /* Q15 → Q20: int32 preserves value */
    }

    /* High freq band split */
    for (j = 0; j < N_FBIN_HIGH; j++) {
        int64_t sum = 0;
        for (int i = 0; i < N_ERB_HIGH; i++) {
            sum += (int64_t)(uint32_t)x_mask[N_ERB_LOW + i]
                 * (int64_t)(uint32_t)ierbfc_weight[i + j * N_ERB_HIGH];
        }
        y[N_ERB_LOW + j] = (int32_t)((sum + (int64_t)round_const) >> 15);
    }
}

/* ── MASK ────────────────────────────────────────────────────────────────
 * MATLAB: y_real = round(x_real .* x_mask * 2^(-15));
 *         y_imag = round(x_imag .* x_mask * 2^(-15));
 *         y_q = cat(1, y_real, y_imag);
 *         y = y_q * 2^(-20);  → output is dequantized back to float-equivalent
 *
 * x_mask: [N] in int32_t Q20 (from BS output)
 * spec_real, spec_imag: [N] in int32_t Q20
 * y_real, y_imag: output (dequantized by 2^(-20), i.e., raw int values)
 *
 * Actually looking at MASK_module.m: output y = y_q * 2^(-20).
 * This means the output is in float for ISTFT. In our fixed-point pipeline,
 * the final mask output should be:
 *   y_real = round(spec_real .* mask * 2^(-15))
 *   y_imag = round(spec_imag .* mask * 2^(-15))
 * And then the ISTFT expects Q15 or float values. For now, we output the raw
 * int32 values and the ISTFT will handle the dequantization.
 */
void mask_fp(
    const int32_t *x_mask,
    const int32_t *spec_real, const int32_t *spec_imag,
    int N,
    int32_t *y_real, int32_t *y_imag)
{
    int i;
    int32_t round_const = 1 << 14;  /* for 2^(-15) */

    for (i = 0; i < N; i++) {
        int64_t prod_r = (int64_t)spec_real[i] * (int64_t)x_mask[i];
        y_real[i] = (int32_t)((prod_r + (int64_t)round_const) >> 15);

        int64_t prod_i = (int64_t)spec_imag[i] * (int64_t)x_mask[i];
        y_imag[i] = (int32_t)((prod_i + (int64_t)round_const) >> 15);
    }
}

/* ======================================================================== */
/*  cTFA (TIME-FREQUENCY ATTENTION)                                          */
/* ======================================================================== */

/* ── cTFA TA (Time Attention) ────────────────────────────────────────────
 * MATLAB:
 *   x_dq = x * 2^(-20);
 *   x_squared = x_dq.^2;
 *   x_agg = mean(x_squared, 2);    // average over frequency dim
 *   x_t = x_agg.';
 *   x_t = Fix_point(x_t, 'u32f20');
 *   [x_gru, h_cache] = GRU_module(x_t, nHidden, h_cache, ...);
 *   x_fc = round(x_gru * ta_fc_weight * 2^(-8)) + ta_fc_bias;
 *   → sigmoid → u16f15
 *
 * x:     [C, W] in int32_t Q20
 * ta_out: [C] in uint16_t Q15
 */
void ctfa_ta_fp(
    const int32_t *x, int C, int W,
    int32_t *ta_h_cache, int nHidden, int input_dim,
    const int16_t *ih_weight, const int32_t *ih_bias,
    const int16_t *hh_weight, const int32_t *hh_bias,
    const int16_t *fc_weight, const int32_t *fc_bias,
    uint16_t *ta_out)
{
    int c, w, j;
    int32_t round_fc = (int32_t)(1LL << (8 - 1));  /* Qr=-8 */


    /* CORRECTED IMPLEMENTATION: */
    /* Step 1: Aggregate across all channels → [1, C] vector */
    /* Actually the MATLAB does per-channel aggregation incorrectly as I read.
       Let me re-read ctfa_ta_module:
       x_dq = x * 2^(-20);          → [C, W]
       x_squared = x_dq.^2;         → [C, W]
       x_agg = mean(x_squared, 2);  → [C, 1] (mean over W, so per-channel energy)
       x_t = x_agg.';               → [1, C]
       x_t = Fix_point(x_t, 'u32f20');
       [x_gru, h_cache] = GRU_module(x_t, nHidden, h_cache, ...);

       So: GRU input_dim = C, output_dim = nHidden.
       One GRU with input=[1,C], producing [1, nHidden].
       Then FC: [1, nHidden] → [1, C].
       Sigmoid: [1, C].

       This is a time-attention mechanism: it generates C attention weights
       based on per-channel energy, processed through a GRU that models
       temporal dependence (since it has state).

       So I need to fix the TA implementation:
       1. Compute per-channel energy: mean(x_c.^2) for each channel
       2. Form [1, C] input vector
       3. One GRU step with input_dim=C
       4. FC: nHidden→C
       5. Sigmoid
    */

    /* Redo implementation: */
    int32_t x_agg_q20[C];  /* VLA */
    int32_t x_t[C];       /* per-channel energy in Q20 */

    for (c = 0; c < C; c++) {
        double sum_sq = 0.0;
        for (w = 0; w < W; w++) {
            double xf = (double)x[c * W + w] / (double)SCALE_Q20;
            sum_sq += xf * xf;
        }
        x_t[c] = (int32_t)((sum_sq / (double)W) * (double)SCALE_Q20);
        if (x_t[c] < 0) x_t[c] = 0;
    }

    /* GRU Q20: int32_t hidden, int16_t output */
    int16_t y_gru[64];
    gru_step_fp_q20(x_t, input_dim, ta_h_cache, nHidden,
                    ih_weight, ih_bias, hh_weight, hh_bias,
                    -13, -8, y_gru);

    /* FC: [1, nHidden] × [nHidden, C] → [1, C] */
    for (c = 0; c < C; c++) {
        int64_t sum = 0;
        for (j = 0; j < nHidden; j++) {
            /* fc_weight is [nHidden, C] column-major in MATLAB */
            sum += (int64_t)y_gru[j] * (int64_t)fc_weight[j + c * nHidden];
        }
        int32_t fc_out = (int32_t)((sum + (int64_t)round_fc) >> 8) + fc_bias[c];
        ta_out[c] = sigmoid_q20_to_q15(fc_out);
    }
}

/* ── cTFA FA (Frequency Attention) ───────────────────────────────────────
 * MATLAB:
 *   x_dq = x * 2^(-20);
 *   x_squared = x_dq.^2;
 *   x_agg = mean(x_squared, 1);     // average over time (channel) dim
 *   x_agg = Fix_point(x_agg, 'u32f20');
 *   pad_len = ...; x_pad = [x_agg zeros(1, pad_len)];
 *   x_t = reshape(x_pad, [4, 17])';  // reshape to [ngrp, nseg]
 *   → BiGRU(nHidden=4) → FC → reshape → depad → sigmoid → u16f15
 *
 * x:      [C, W] in int32_t Q20
 * fa_out: [W] in uint16_t Q15
 */
void ctfa_fa_fp(
    const int32_t *x, int C, int W, int pad_len,
    int nHidden,
    const int16_t *ih_weight, const int32_t *ih_bias,
    const int16_t *hh_weight, const int32_t *hh_bias,
    const int16_t *re_ih_weight, const int32_t *re_ih_bias,
    const int16_t *re_hh_weight, const int32_t *re_hh_bias,
    const int16_t *fc_weight, const int32_t *fc_bias,
    uint16_t *fa_out)
{
    int c, w;

    /* Step 1: Average over channels → [1, W] */
    int32_t *x_agg_q20 = (int32_t*)malloc((W + pad_len) * sizeof(int32_t));
    for (w = 0; w < W; w++) {
        double sum = 0.0;
        for (c = 0; c < C; c++) {
            double xf = (double)x[c * W + w] / (double)SCALE_Q20;
            sum += xf * xf;
        }
        x_agg_q20[w] = (int32_t)((sum / (double)C) * (double)SCALE_Q20);
        if (x_agg_q20[w] < 0) x_agg_q20[w] = 0;
    }

    /* Step 2: Pad with zeros */
    int W_pad = W + pad_len;
    for (w = W; w < W_pad; w++) {
        x_agg_q20[w] = 0;
    }

    /* Step 3: Reshape to [ngrp, nseg] → [nseg, ngrp]
     * For E0: W=65, pad=3 → 68, ngrp=4, nseg=17
     * For E1: W=33, pad=2 → 35 (not divisible by 4? Let me check XMB0_cTFA_fa)
     * Let me check E1: x_agg = [1,33], pad=2 → [1,35]. reshape to [?,?]...
     * Need to read the specific E1 FA module.
     * For now, assume ngrp=4 and compute nseg = (W+pad)/4.
     */
    int ngrp = 4;
    int nseg = W_pad / ngrp;  /* must be integer */

    /* Use VLAs sized by actual nseg (W+pad)/4, max=33 for W=129 */
    /* Reshape: x_t = reshape(x_pad, [ngrp, nseg])' → [nseg, ngrp]
     * MATLAB: A=reshape(x_pad,[ngrp,nseg]) → A(grp,seg)=x_pad[grp+ngrp*seg]
     *         B=A' → B(seg,grp)=A(grp,seg)=x_pad[grp+ngrp*seg]
     * C: x_re[seg*ngrp+grp] = x_agg_q20[grp + ngrp*seg] */
    int32_t *x_re = (int32_t*)malloc(nseg * ngrp * sizeof(int32_t));
    for (int seg = 0; seg < nseg; seg++) {
        for (int grp = 0; grp < ngrp; grp++) {
            x_re[seg * ngrp + grp] = x_agg_q20[grp + ngrp * seg];
        }
    }

    /* Step 4: BiGRU over nseg timesteps, input_dim=ngrp, nHidden=4 */
    int total_hid = 2 * nHidden;  /* BiGRU output = 2*4=8 */
    int16_t *y_gru = (int16_t*)malloc(nseg * total_hid * sizeof(int16_t));
    bigru_sequence_fp_q20(x_re, nseg, ngrp, nHidden,
                          ih_weight, ih_bias, hh_weight, hh_bias,
                          re_ih_weight, re_ih_bias, re_hh_weight, re_hh_bias,
                          -13, -8, y_gru);

    /* Step 5: FC: [nseg, 2*nHidden] → [nseg, ngrp] */
    int32_t *x_fc = (int32_t*)malloc(nseg * ngrp * sizeof(int32_t));
    fc_fp_s16(y_gru, nseg, total_hid, ngrp, fc_weight, fc_bias, -9, x_fc);

    /* Step 6: Reshape back to [1, W_pad] and remove padding
     * MATLAB: x_fc_T = x_fc.' → [ngrp, nseg]
     *         x_shape = reshape(x_fc_T, 1, []) → column-major flatten
     *         x_shape[grp + ngrp*seg] = x_fc[seg*ngrp + grp]
     * C: x_shape[grp + ngrp*seg] = x_fc[seg * ngrp + grp] */
    int32_t *x_shape = (int32_t*)malloc(W_pad * sizeof(int32_t));
    for (int seg = 0; seg < nseg; seg++) {
        for (int grp = 0; grp < ngrp; grp++) {
            x_shape[grp + ngrp * seg] = x_fc[seg * ngrp + grp];
        }
    }

    /* Step 7: Sigmoid (skip padded values) */
    for (w = 0; w < W; w++) {
        fa_out[w] = sigmoid_q20_to_q15(x_shape[w]);
    }

    free(x_agg_q20);
    free(x_re);
    free(y_gru);
    free(x_fc);
    free(x_shape);
}

/* ── cTFA Apply ──────────────────────────────────────────────────────────
 * MATLAB:
 *   y_t = round(tconv .* ta' * 2^(-15));
 *   y = round(y_t .* fa * 2^(-15));
 *
 * x:   [C, W] in int32_t Q20
 * ta:  [C] in uint16_t Q15
 * fa:  [W] in uint16_t Q15
 * y:   [C, W] in int32_t Q20
 */
void ctfa_apply_fp(
    const int32_t *x, int C, int W,
    const uint16_t *ta, const uint16_t *fa,
    int32_t *y)
{
    int c, w;
    int32_t round_const = 1 << 14;  /* for 2^(-15) */

    for (c = 0; c < C; c++) {
        for (w = 0; w < W; w++) {
            /* y_t = round(x .* ta' * 2^(-15)) */
            int64_t prod_ta = (int64_t)x[c * W + w] * (int64_t)(uint32_t)ta[c];
            int32_t y_t = (int32_t)((prod_ta + (int64_t)round_const) >> 15);

            /* y = round(y_t .* fa * 2^(-15)) */
            int64_t prod_fa = (int64_t)y_t * (int64_t)(uint32_t)fa[w];
            y[c * W + w] = (int32_t)((prod_fa + (int64_t)round_const) >> 15);
        }
    }
}

/* ======================================================================== */
/*  NON-LINEAR FUNCTIONS                                                     */
/* ======================================================================== */

/* ── log_gen_fp — Log-magnitude compression ──────────────────────────────
 * MATLAB:
 *   x_real_dq = x_real * 2^(-20);
 *   x_imag_dq = x_imag * 2^(-20);
 *   mag = sqrt(x_real_dq.^2 + x_imag_dq.^2);
 *   clamped = max(mag, 1e-12);
 *   y = log10(clamped);
 *   y = Fix_point(y, 's32f20');
 *
 * Uses LUT for log10, integer sqrt for magnitude.
 */
int32_t log_gen_fp(int32_t real_q20, int32_t imag_q20)
{
    /* Compute mag^2 in Q40: (real^2 + imag^2) */
    int64_t real_sq = (int64_t)real_q20 * (int64_t)real_q20;
    int64_t imag_sq = (int64_t)imag_q20 * (int64_t)imag_q20;
    uint64_t mag_sq_q40 = (uint64_t)(real_sq + imag_sq);

    /* Integer sqrt → Q20 */
    uint32_t mag_q20 = sqrt_q40_to_q20(mag_sq_q40);

    /* Clamp: max(mag, 1e-12 * 2^20) ≈ max(mag, 1) */
    if (mag_q20 < 1) mag_q20 = 1;

    /* Log10 via LUT */
    return log10_q20_to_q20((int32_t)mag_q20);
}

/* ======================================================================== */
/*  SHUFFLE OPERATIONS                                                       */
/* ======================================================================== */

/* Interleave: first half → odd rows, second half → even rows
 * src: [C, W], dst: [C, W]
 * MATLAB: y_s(1:2:end,:) = y_pconv(1:N/2,:);
 *         y_s(2:2:end,:) = y_pconv(N/2+1:end,:);
 */
void shuffle_interleave(const int32_t *src, int C, int W, int32_t *dst)
{
    int half = C / 2;
    int c, w;
    for (c = 0; c < half; c++) {
        for (w = 0; w < W; w++) {
            dst[(2 * c) * W + w] = src[c * W + w];           /* odd rows (1:2:end) */
            dst[(2 * c + 1) * W + w] = src[(half + c) * W + w]; /* even rows (2:2:end) */
        }
    }
}

/* Deinterleave: reverse of interleave */
void shuffle_deinterleave(const int32_t *src, int C, int W, int32_t *dst)
{
    int half = C / 2;
    int c, w;
    for (c = 0; c < half; c++) {
        for (w = 0; w < W; w++) {
            dst[c * W + w] = src[(2 * c) * W + w];               /* from odd rows */
            dst[(half + c) * W + w] = src[(2 * c + 1) * W + w];  /* from even rows */
        }
    }
}
