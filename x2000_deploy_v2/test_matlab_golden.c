/* test_matlab_golden.c — Layer-by-layer golden comparison for UL-UNAS C inference
 * Reads MATLAB-exported golden/*.bin files, runs C inference layer by layer,
 * reports SNR and max absolute error for each.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

/* ===== Golden file I/O ===== */

static int32_t* load_i32(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "MISSING: %s\n", path); return NULL; }
    int32_t *d = (int32_t*)malloc(n * sizeof(int32_t));
    fread(d, sizeof(int32_t), n, f); fclose(f); return d;
}
static int16_t* load_i16(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "MISSING: %s\n", path); return NULL; }
    int16_t *d = (int16_t*)malloc(n * sizeof(int16_t));
    fread(d, sizeof(int16_t), n, f); fclose(f); return d;
}
static uint16_t* load_u16(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "MISSING: %s\n", path); return NULL; }
    uint16_t *d = (uint16_t*)malloc(n * sizeof(uint16_t));
    fread(d, sizeof(uint16_t), n, f); fclose(f); return d;
}

/* ===== Metrics ===== */

typedef struct { double snr; int max_err; double mean_err; int match, total; } met_t;

static met_t cmp_i32(const int32_t *c, const int32_t *g, int N) {
    met_t m = {0}; m.total = N;
    double ss=0, sn=0, sa=0;
    for(int i=0;i<N;i++){
        double s=g[i], n=c[i]-g[i];
        ss+=s*s; sn+=n*n; sa+=fabs(n);
        int ae=abs(c[i]-g[i]);
        if(ae>m.max_err)m.max_err=ae;
        if(c[i]==g[i])m.match++;
    }
    m.snr = sn>0 ? 10*log10(ss/sn) : (sn==0?999:-999);
    m.mean_err = sa/N;
    return m;
}

static void report(const char *label, met_t m) {
    printf("  %-32s SNR=%8.2f dB  MAX=%4d  AVG=%.2f  ok=%d/%d",
           label, m.snr, m.max_err, m.mean_err, m.match, m.total);
    if(m.snr>=130 && m.max_err<=1) printf("  ✓");
    else if(m.snr>=60) printf("  △");
    else printf("  ✗");
    printf("\n");
}

/* ===== Forward declarations ===== */
extern void encoder_xconv_module(const int32_t *x, int32_t *cc, int16_t *tc, int32_t *y);
extern void encoder_module(const int32_t*,int32_t*,int16_t*,int32_t*,int16_t*,int32_t*,int16_t*,int16_t*,int16_t*,int32_t*,int32_t*,int32_t*,int32_t*,int32_t*);
extern void gdprnn_module(const int32_t*,int16_t*,int,int32_t*);
extern void decoder_module(const int32_t*,const int32_t*,const int32_t*,const int32_t*,const int32_t*,const int32_t*,int16_t*,int16_t*,int32_t*,int16_t*,int32_t*,int16_t*,int32_t*,int16_t*,int32_t*);

/* ERB weights */
extern const uint16_t erb_erb_fc_weight[];
extern const uint16_t erb_ierb_fc_weight[];

/* ===== Main ===== */

int main(int argc, char **argv) {
    const char *gd = "golden";
    if(argc>1) gd=argv[1];

    printf("=== UL-UNAS Layer-by-Layer Golden Test ===\nGolden: %s\n\n", gd);

    char p[512];
    #define LDPATH(fmt) (snprintf(p,sizeof(p),"%s/" fmt,gd),p)

    /* Load frame001 golden inputs */
    int32_t *g_log = load_i32(LDPATH("frame001_log_gen.bin"), 257);
    int32_t *g_bm  = load_i32(LDPATH("frame001_bm.bin"), 129);
    int32_t *g_e0  = load_i32(LDPATH("frame001_e0.bin"), 12*65);
    int32_t *g_e1  = load_i32(LDPATH("frame001_e1.bin"), 24*33);
    int32_t *g_e2  = load_i32(LDPATH("frame001_e2.bin"), 24*33);
    int32_t *g_e3  = load_i32(LDPATH("frame001_e3.bin"), 32*33);
    int32_t *g_e4  = load_i32(LDPATH("frame001_e4.bin"), 16*33);
    int32_t *g_rn1 = load_i32(LDPATH("frame001_rnn1.bin"), 16*33);
    int32_t *g_rn2 = load_i32(LDPATH("frame001_rnn2.bin"), 16*33);
    int32_t *g_dec = load_i32(LDPATH("frame001_dec.bin"), 2*129);
    uint16_t *g_sig = load_u16(LDPATH("frame001_sig.bin"), 2*129);
    int32_t *g_bs  = load_i32(LDPATH("frame001_bs.bin"), 2*257);
    int32_t *g_mr  = load_i32(LDPATH("frame001_mask_r.bin"), 257);
    int32_t *g_mi  = load_i32(LDPATH("frame001_mask_i.bin"), 257);

    /* Load state golden (first frame only) */
    int32_t *gs_ce0 = load_i32(LDPATH("state_conv_cache_e0.bin"), 2*129);
    int32_t *gs_ce1 = load_i32(LDPATH("state_conv_cache_e1.bin"), 24*65);
    int32_t *gs_ce2 = load_i32(LDPATH("state_conv_cache_e2.bin"), 24*33);
    int16_t *gs_te0 = load_i16(LDPATH("state_tfa_cache_e0.bin"), 24);
    int16_t *gs_te1 = load_i16(LDPATH("state_tfa_cache_e1.bin"), 48);
    int16_t *gs_te2 = load_i16(LDPATH("state_tfa_cache_e2.bin"), 48);
    int16_t *gs_te3 = load_i16(LDPATH("state_tfa_cache_e3.bin"), 64);
    int16_t *gs_te4 = load_i16(LDPATH("state_tfa_cache_e4.bin"), 32);
    int16_t *gs_ic0 = load_i16(LDPATH("state_inter_cache_0.bin"), 33*16);
    int16_t *gs_ic1 = load_i16(LDPATH("state_inter_cache_1.bin"), 33*16);

    ulunas_state_t st; ulunas_state_init(&st);
    int passed=0, total=0;

    /* ── Test 1: log_gen → BM ── */
    if(g_log && g_bm) {
        int32_t c[129]; bm_fp(g_log, erb_erb_fc_weight, c);
        met_t m = cmp_i32(c, g_bm, 129);
        report("1. BM (log_gen→BM)", m); total++;
        if(m.max_err<=1 && m.snr>130) passed++;
    }

    /* ── Test 2: Encoder E0 XConv ── */
    if(g_bm && g_e0) {
        int32_t c[12*65];
        encoder_xconv_module(g_bm, st.conv_cache_e0, st.tfa_cache_e0, c);
        met_t m = cmp_i32(c, g_e0, 12*65);
        report("2. Encoder E0 XConv", m); total++;
        if(m.max_err<=1 && m.snr>130) passed++;
    }

    /* ── Test 3: Full Encoder (E0→E4) ── */
    if(g_bm && g_e4) {
        /* Reset state for fair comparison */
        ulunas_state_init(&st);
        int32_t ye0[12*65], ye1[24*33], ye2[24*33], ye3[32*33], ye4[16*33];
        encoder_module(g_bm,
            st.conv_cache_e0, st.tfa_cache_e0,
            st.conv_cache_e1, st.tfa_cache_e1,
            st.conv_cache_e2, st.tfa_cache_e2,
            st.tfa_cache_e3, st.tfa_cache_e4,
            ye0, ye1, ye2, ye3, ye4);
        met_t m = cmp_i32(ye4, g_e4, 16*33);
        report("3. Encoder full (E0→E4)", m); total++;
        if(m.max_err<=1 && m.snr>130) passed++;
    }

    /* ── Test 4: GDPRNN idx=0 ── */
    if(g_e4 && g_rn1) {
        ulunas_state_init(&st);
        int32_t ye0[12*65],ye1[24*33],ye2[24*33],ye3[32*33],ye4[16*33];
        encoder_module(g_bm,
            st.conv_cache_e0,st.tfa_cache_e0,st.conv_cache_e1,st.tfa_cache_e1,
            st.conv_cache_e2,st.tfa_cache_e2,st.tfa_cache_e3,st.tfa_cache_e4,
            ye0,ye1,ye2,ye3,ye4);
        int32_t c[16*33];
        gdprnn_module(ye4, st.inter_cache_0, 0, c);
        met_t m = cmp_i32(c, g_rn1, 16*33);
        report("4. GDPRNN idx=0", m); total++;
        if(m.max_err<=1 && m.snr>130) passed++;
    }

    /* ── Test 5: GDPRNN idx=1 ── */
    if(g_rn1 && g_rn2) {
        int32_t c[16*33];
        gdprnn_module(g_rn1, st.inter_cache_1, 1, c);
        met_t m = cmp_i32(c, g_rn2, 16*33);
        report("5. GDPRNN idx=1", m); total++;
        if(m.max_err<=1 && m.snr>130) passed++;
    }

    /* ── Test 6: Decoder (full) ── */
    if(g_rn2 && g_dec) {
        ulunas_state_init(&st);
        int32_t ye0[12*65],ye1[24*33],ye2[24*33],ye3[32*33],ye4[16*33];
        encoder_module(g_bm,
            st.conv_cache_e0,st.tfa_cache_e0,st.conv_cache_e1,st.tfa_cache_e1,
            st.conv_cache_e2,st.tfa_cache_e2,st.tfa_cache_e3,st.tfa_cache_e4,
            ye0,ye1,ye2,ye3,ye4);
        int32_t yr1[16*33], yr2[16*33];
        gdprnn_module(ye4, st.inter_cache_0, 0, yr1);
        gdprnn_module(yr1, st.inter_cache_1, 1, yr2);
        int32_t c[2*129];
        decoder_module(yr2, ye4,ye3,ye2,ye1,ye0,
            st.tfa_cache_d0,st.tfa_cache_d1,
            st.conv_cache_d0,st.tfa_cache_d2,
            st.conv_cache_d1,st.tfa_cache_d3,
            st.conv_cache_d2,st.tfa_cache_d4, c);
        met_t m = cmp_i32(c, g_dec, 2*129);
        report("6. Decoder full (D0→D4)", m); total++;
        if(m.max_err<=1 && m.snr>130) passed++;
    }

    /* ── Test 7: Sigmoid ── */
    if(g_dec && g_sig) {
        uint16_t c[2*129];
        for(int i=0;i<2*129;i++) c[i]=sigmoid_q20_to_q15(g_dec[i]);
        met_t m = {0}; m.total=2*129;
        double ss=0,sn=0,sa=0;
        for(int i=0;i<2*129;i++){
            double s=g_sig[i], n=(double)c[i]-s;
            ss+=s*s; sn+=n*n; sa+=fabs(n);
            int ae=abs((int)c[i]-(int)g_sig[i]);
            if(ae>m.max_err)m.max_err=ae;
            if(c[i]==g_sig[i])m.match++;
        }
        m.snr = sn>0 ? 10*log10(ss/sn) : (sn==0?999:-999);
        m.mean_err=sa/(2*129);
        report("7. Sigmoid LUT", m); total++;
        if(m.max_err<=2 && m.snr>100) passed++;
    }

    /* ── Test 8: End-to-end single frame ── */
    if(g_mr && g_mi) {
        /* Run full pipeline: need spec_real/imag input */
        /* We'll use golden log_gen as input and compare final mask */
        /* For E2E test, load spec from log_gen golden */
        /* The spec_real/imag should come from STFT golden */
        /* For now: load frame001 initial state and run pipeline */

        /* Use spec_real/imag from the golden (approximated from log_gen) */
        /* This is a rough test — the full E2E requires STFT golden data */
        int32_t spec_r[257], spec_i[257];
        for(int i=0;i<257;i++){ spec_r[i]=g_log[i]; spec_i[i]=0; }

        /* Run full pipeline */
        ulunas_state_init(&st);
        int32_t out_r[257], out_i[257];
        ulunas_infer_frame(spec_r, spec_i, &st, out_r, out_i);

        met_t mr = cmp_i32(out_r, g_mr, 257);
        met_t mi = cmp_i32(out_i, g_mi, 257);
        report("8a. MASK real", mr); total++;
        report("8b. MASK imag", mi); total++;
        if(mr.max_err<=1 && mr.snr>130) passed++;
        if(mi.max_err<=1 && mi.snr>130) passed++;
    }

    printf("\n=== %d/%d tests passed ===\n", passed, total);

    /* Cleanup */
    free(g_log);free(g_bm);free(g_e0);free(g_e1);free(g_e2);free(g_e3);free(g_e4);
    free(g_rn1);free(g_rn2);free(g_dec);free(g_sig);free(g_bs);free(g_mr);free(g_mi);
    free(gs_ce0);free(gs_ce1);free(gs_ce2);
    free(gs_te0);free(gs_te1);free(gs_te2);free(gs_te3);free(gs_te4);
    free(gs_ic0);free(gs_ic1);

    return (passed==total) ? 0 : 1;
}
