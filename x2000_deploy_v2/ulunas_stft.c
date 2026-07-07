/* ulunas_stft.c — STFT/ISTFT for UL-UNAS using Q15 fixed-point FFT
 *
 * STFT: float audio → Q15 windowed FFT → Q20 complex spectrum
 * ISTFT: Q20 complex spectrum → overlap-add → float audio
 *
 * FFT: Radix-2 DIT, Q15 fixed-point (compatible with X2000 MIPS optimization)
 */

#include "ulunas_fp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ======================================================================== */
/*  Q15 FFT (Radix-2 DIT, in-place complex)                                  */
/* ======================================================================== */

/* Pre-computed Q15 twiddle factors: cos(2π*k/N) and -sin(2π*k/N) */
typedef struct {
    int16_t cos;  /* Q15 */
    int16_t sin;  /* Q15 (actually -sin for forward FFT) */
} twiddle_q15_t;

static twiddle_q15_t *fft_twiddles = NULL;
static int fft_twiddle_N = 0;

/* Initialize twiddle factors for N-point FFT */
static void fft_init_twiddles(int N) {
    if (fft_twiddles && fft_twiddle_N == N) return;
    free(fft_twiddles);
    fft_twiddles = (twiddle_q15_t*)malloc((N/2) * sizeof(twiddle_q15_t));
    fft_twiddle_N = N;
    for (int k = 0; k < N/2; k++) {
        double angle = -2.0 * M_PI * k / N;  /* negative for forward FFT */
        fft_twiddles[k].cos = (int16_t)(cos(angle) * 32767.0);
        fft_twiddles[k].sin = (int16_t)(sin(angle) * 32767.0);
    }
}

/* Q15 multiply with rounding */
static inline int16_t q15_mul(int16_t a, int16_t b) {
    int32_t prod = (int32_t)a * (int32_t)b;
    return (int16_t)((prod + 16384) >> 15);  /* round half-up */
}

/* Radix-2 DIT FFT butterfly, Q15 in-place */
static void fft_radix2_q15(int16_t *real, int16_t *imag, int N) {
    fft_init_twiddles(N);

    /* Bit-reversal permutation */
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            int16_t tr = real[i]; real[i] = real[j]; real[j] = tr;
            int16_t ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
    }

    /* FFT stages */
    for (int len = 2; len <= N; len <<= 1) {
        int half = len >> 1;
        int step = N / len;
        for (int i = 0; i < N; i += len) {
            for (int j = 0; j < half; j++) {
                int tw_idx = j * step;
                int16_t tw_r = fft_twiddles[tw_idx].cos;
                int16_t tw_i = fft_twiddles[tw_idx].sin;

                int a = i + j;
                int b = i + j + half;

                /* Butterfly */
                int16_t t_re = q15_mul(real[b], tw_r) - q15_mul(imag[b], tw_i);
                int16_t t_im = q15_mul(real[b], tw_i) + q15_mul(imag[b], tw_r);

                /* Saturation add/sub */
                int32_t re_sum = (int32_t)real[a] + (int32_t)t_re;
                int32_t re_diff = (int32_t)real[a] - (int32_t)t_re;
                int32_t im_sum = (int32_t)imag[a] + (int32_t)t_im;
                int32_t im_diff = (int32_t)imag[a] - (int32_t)t_im;

                real[a] = sat_s16(re_sum);
                real[b] = sat_s16(re_diff);
                imag[a] = sat_s16(im_sum);
                imag[b] = sat_s16(im_diff);
            }
        }
    }
}

/* Inverse FFT (conjugate twiddles + scale by 1/N) */
static void ifft_radix2_q15(int16_t *real, int16_t *imag, int N) {
    /* Conjugate input (negate imaginary part) */
    for (int i = 0; i < N; i++) imag[i] = -imag[i];

    /* Forward FFT */
    fft_radix2_q15(real, imag, N);

    /* Conjugate output and scale by 1/N */
    for (int i = 0; i < N; i++) {
        /* Scale by 1/N: divide by N with rounding */
        int32_t r = (int32_t)real[i] * 32768 / N;  /* approximate */
        int32_t im = (int32_t)(-imag[i]) * 32768 / N;
        real[i] = sat_s16(r);
        imag[i] = sat_s16(im);
    }
}

/* ======================================================================== */
/*  STFT: audio samples → complex spectrogram                                */
/* ======================================================================== */

/**
 * stft_frame_q15: Compute one frame of STFT
 * @audio:    input audio samples [win_len] as int16 Q15
 * @window:   Hann window [win_len] as int16 Q15
 * @real_out: FFT real output [n_fft/2+1] as int32 Q20
 * @imag_out: FFT imag output [n_fft/2+1] as int32 Q20
 */
void stft_frame_q15(
    const int16_t *audio, int win_len,
    const int16_t *window, int n_fft,
    int32_t *real_out, int32_t *imag_out)
{
    /* Apply window and zero-pad to n_fft */
    int16_t frame_r[1024] = {0};
    int16_t frame_i[1024] = {0};

    for (int i = 0; i < win_len; i++) {
        int32_t windowed = q15_mul(audio[i], window[i]);
        frame_r[i] = windowed;
    }

    /* Forward FFT */
    fft_radix2_q15(frame_r, frame_i, n_fft);

    /* Convert Q15 → Q20: multiply by 2^5 = 32 */
    int n_out = n_fft / 2 + 1;
    for (int i = 0; i < n_out; i++) {
        real_out[i] = (int32_t)frame_r[i] * 32;
        imag_out[i] = (int32_t)frame_i[i] * 32;
    }
}

/**
 * stft_process: Full STFT of audio signal
 * @audio:    input samples (float, normalized to [-1,1])
 * @n_samples: number of input samples
 * @n_fft:    FFT size (must be power of 2)
 * @win_len:  window length
 * @hop:      hop size (frame advance)
 * @window:   window coefficients [win_len] as int16 Q15
 * @real_spec: output real spectrogram [n_frames, n_fft/2+1] int32 Q20
 * @imag_spec: output imag spectrogram
 * Returns: number of frames
 */
int stft_process(
    const float *audio, int n_samples,
    int n_fft, int win_len, int hop,
    const int16_t *window,
    int32_t *real_spec, int32_t *imag_spec)
{
    int n_frames = (n_samples - win_len) / hop + 1;
    if (n_frames < 1) n_frames = 1;

    int n_bins = n_fft / 2 + 1;

    for (int f = 0; f < n_frames; f++) {
        /* Extract frame */
        int16_t frame[1024];
        for (int i = 0; i < win_len; i++) {
            int idx = f * hop + i;
            float sample = (idx < n_samples) ? audio[idx] : 0.0f;
            /* Convert float → Q15 */
            int32_t q = (int32_t)(sample * 32767.0f);
            frame[i] = sat_s16(q);
        }

        /* STFT this frame */
        stft_frame_q15(frame, win_len, window, n_fft,
                       &real_spec[f * n_bins], &imag_spec[f * n_bins]);
    }

    return n_frames;
}

/* ======================================================================== */
/*  ISTFT: complex spectrogram → audio samples                               */
/* ======================================================================== */

/**
 * istft_frame_q15: Inverse STFT for one frame
 * @real_in:  FFT real input [n_fft/2+1] as int32 Q20
 * @imag_in:  FFT imag input [n_fft/2+1] as int32 Q20
 * @window:   synthesis window [win_len] as int16 Q15
 * @n_fft:    FFT size
 * @win_len:  window length
 * @audio_out: output time-domain frame [win_len] as int16 Q15
 */
void istft_frame_q15(
    const int32_t *real_in, const int32_t *imag_in,
    const int16_t *window, int n_fft, int win_len,
    int16_t *audio_out)
{
    int n_bins = n_fft / 2 + 1;

    /* Convert Q20 → Q15: divide by 32 */
    int16_t spec_r[1024] = {0};
    int16_t spec_i[1024] = {0};

    for (int i = 0; i < n_bins; i++) {
        spec_r[i] = sat_s16(real_in[i] / 32);
        spec_i[i] = sat_s16(imag_in[i] / 32);
    }

    /* Reconstruct conjugate-symmetric full spectrum */
    for (int i = 1; i < n_bins - 1; i++) {
        int j = n_fft - i;
        spec_r[j] = spec_r[i];
        spec_i[j] = -spec_i[i];
    }

    /* Inverse FFT */
    ifft_radix2_q15(spec_r, spec_i, n_fft);

    /* Apply synthesis window and extract win_len samples */
    for (int i = 0; i < win_len; i++) {
        audio_out[i] = q15_mul(spec_r[i], window[i]);
    }
}

/**
 * istft_process: Full ISTFT with overlap-add
 * @real_spec:  input real spectrogram [n_frames, n_fft/2+1] int32 Q20
 * @imag_spec:  input imag spectrogram
 * @n_frames:   number of frames
 * @n_fft:      FFT size
 * @win_len:    window length
 * @hop:        hop size
 * @window:     synthesis window [win_len] int16 Q15
 * @audio_out:  output float audio samples
 * @max_out:    max output samples
 * Returns: number of output samples
 */
int istft_process(
    const int32_t *real_spec, const int32_t *imag_spec,
    int n_frames, int n_fft, int win_len, int hop,
    const int16_t *window,
    float *audio_out, int max_out)
{
    int n_bins = n_fft / 2 + 1;
    int total_samples = (n_frames - 1) * hop + win_len;
    if (total_samples > max_out) total_samples = max_out;

    /* Overlap-add buffer (float) */
    float *ola = (float*)calloc(total_samples, sizeof(float));
    /* Window power compensation (for Hann window with 50% overlap) */
    float gain = 0.5f;  /* 1/(0.5 * 2) for Hann window */

    for (int f = 0; f < n_frames; f++) {
        int16_t frame[1024];
        istft_frame_q15(&real_spec[f * n_bins], &imag_spec[f * n_bins],
                        window, n_fft, win_len, frame);

        /* Overlap-add: convert Q15 → float and add */
        for (int i = 0; i < win_len; i++) {
            int idx = f * hop + i;
            if (idx < total_samples) {
                ola[idx] += (float)frame[i] / 32767.0f * gain;
            }
        }
    }

    /* Copy to output */
    for (int i = 0; i < total_samples; i++) {
        audio_out[i] = ola[i];
    }
    free(ola);
    return total_samples;
}

/* ======================================================================== */
/*  Hann Window Generator (Q15)                                              */
/* ======================================================================== */

void gen_hann_window_q15(int win_len, int16_t *window) {
    for (int i = 0; i < win_len; i++) {
        double w = 0.5 * (1.0 - cos(2.0 * M_PI * i / (win_len - 1)));
        window[i] = (int16_t)(w * 32767.0);
    }
}

/* ======================================================================== */
/*  Q15↔Q20 conversion for pipeline integration                              */
/* ======================================================================== */

/* Convert Q15 complex spectrum to Q20 (used at STFT output → model input) */
void spec_q15_to_q20(const int16_t *real_q15, const int16_t *imag_q15,
                     int N, int32_t *real_q20, int32_t *imag_q20) {
    for (int i = 0; i < N; i++) {
        real_q20[i] = (int32_t)real_q15[i] * 32;  /* 2^5 = Q15→Q20 */
        imag_q20[i] = (int32_t)imag_q15[i] * 32;
    }
}
