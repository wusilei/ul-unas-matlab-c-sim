/* test_matlab_golden.c — Layer-by-layer golden comparison
 * Handles MATLAB column-major → C row-major transpose for 2D arrays.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

/* Load binary int32, with optional column-major→row-major transpose.
 * rows=1 → no transpose (1D array), rows>1 → transpose [rows,cols] → [rows,cols] row-major */
static int32_t* load_i32_t(const char *path, int rows, int cols, int *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "MISS: %s\n", path); return NULL; }
    int n = rows * cols; *out_n = n;
    int32_t *d = malloc(n * 4);
    if (rows == 1) {
        fread(d, 4, n, f);
    } else {
        /* Read MATLAB column-major, store C row-major */
        int32_t *tmp = malloc(n * 4);
        fread(tmp, 4, n, f);
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < rows; i++)
                d[i * cols + j] = tmp[i + j * rows];
        free(tmp);
    }
    fclose(f); return d;
}
/* Load uint16 similarly */
static uint16_t* load_u16_t(const char *path, int rows, int cols, int *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "MISS: %s\n", path); return NULL; }
    int n = rows * cols; *out_n = n;
    uint16_t *d = malloc(n * 2);
    if (rows == 1) {
        fread(d, 2, n, f);
    } else {
        uint16_t *tmp = malloc(n * 2);
        fread(tmp, 2, n, f);
        for (int j = 0; j < cols; j++)
            for (int i = 0; i < rows; i++)
                d[i * cols + j] = tmp[i + j * rows];
        free(tmp);
    }
    fclose(f); return d;
}

typedef struct { double snr; int max_err; double avg_err; int match, total; } met_t;
static met_t cmp(const int32_t *c, const int32_t *g, int N) {
    met_t m = {0}; m.total = N; double ss=0,sn=0;
    for (int i=0;i<N;i++) {
        double s=g[i], n=c[i]-g[i]; ss+=s*s; sn+=n*n;
        int ae=abs(c[i]-g[i]); if(ae>m.max_err)m.max_err=ae;
        m.avg_err+=ae; if(c[i]==g[i])m.match++;
    }
    m.snr = sn>0 ? 10*log10(ss/sn) : (sn==0?999:-999);
    m.avg_err /= N; return m;
}
static void rep(const char *l, met_t m) {
    printf("  %-32s SNR=%8.2f dB MAX=%6d AVG=%.2f ok=%d/%d",
           l,m.snr,m.max_err,m.avg_err,m.match,m.total);
    if(m.snr>=130 && m.max_err<=1) printf(" ✓\n");
    else if(m.snr>=60) printf(" △\n");
    else printf(" ✗\n");
}

extern void encoder_xconv_module(const int32_t*,int32_t*,int16_t*,int32_t*);
extern void encoder_module(const int32_t*,int32_t*,int16_t*,int32_t*,int16_t*,int32_t*,int16_t*,int16_t*,int16_t*,int32_t*,int32_t*,int32_t*,int32_t*,int32_t*);
extern void gdprnn_module(const int32_t*,int16_t*,int,int32_t*);
extern void decoder_module(const int32_t*,const int32_t*,const int32_t*,const int32_t*,const int32_t*,const int32_t*,int16_t*,int16_t*,int32_t*,int16_t*,int32_t*,int16_t*,int32_t*,int16_t*,int32_t*);
extern const uint16_t erb_erb_fc_weight[], erb_ierb_fc_weight[];

int main(int argc, char **argv) {
    const char *gd = argc>1 ? argv[1] : "golden";
    printf("=== UL-UNAS Golden Test ===\nGolden: %s\n\n", gd);
    char p[512];
    #define P(fmt) (snprintf(p,sizeof(p),"%s/" fmt,gd),p)

    int n; met_t m; int passed=0,total=0;
    int np=2; /* frames for encoder+ later tests: need frame001 */

    /* ── Test 1: BM ── */
    {
        int32_t *gl = load_i32_t(P("frame001_log_gen.bin"),1,257,&n);
        int32_t *gb = load_i32_t(P("frame001_bm.bin"),1,129,&n);
        if(gl&&gb){ int32_t c[129]; bm_fp(gl,erb_erb_fc_weight,c); rep("1. BM",cmp(c,gb,129)); total++; if(cmp(c,gb,129).snr>130)passed++; }
        free(gl);free(gb);
    }

    /* ── Test 2: Encoder E0 ── */
    {
        int32_t *gb = load_i32_t(P("frame001_bm.bin"),1,129,&n);
        int32_t *ge0= load_i32_t(P("frame001_e0.bin"),12,65,&n);
        if(gb&&ge0){
            ulunas_state_t st; ulunas_state_init(&st);
            int32_t c[780];
            encoder_xconv_module(gb,st.conv_cache_e0,st.tfa_cache_e0,c);
            rep("2. Encoder E0 XConv",cmp(c,ge0,780)); total++; if(cmp(c,ge0,780).snr>130)passed++;
        }
        free(gb);free(ge0);
    }

    /* ── Test 3: Full Encoder (E0→E4) ── */
    {
        int32_t *gb = load_i32_t(P("frame001_bm.bin"),1,129,&n);
        int32_t *ge4= load_i32_t(P("frame001_e4.bin"),16,33,&n);
        if(gb&&ge4){
            ulunas_state_t st; ulunas_state_init(&st);
            int32_t ye0[780],ye1[792],ye2[792],ye3[1056],ye4[528];
            encoder_module(gb,st.conv_cache_e0,st.tfa_cache_e0,st.conv_cache_e1,st.tfa_cache_e1,
                st.conv_cache_e2,st.tfa_cache_e2,st.tfa_cache_e3,st.tfa_cache_e4,
                ye0,ye1,ye2,ye3,ye4);
            rep("3. Encoder full (E0→E4)",cmp(ye4,ge4,528)); total++; if(cmp(ye4,ge4,528).snr>130)passed++;
        }
        free(gb);free(ge4);
    }

    /* ── Test 4: GDPRNN idx=0 ── */
    {
        int32_t *gb = load_i32_t(P("frame001_bm.bin"),1,129,&n);
        int32_t *gr1= load_i32_t(P("frame001_rnn1.bin"),16,33,&n);
        if(gb&&gr1){
            ulunas_state_t st; ulunas_state_init(&st);
            int32_t ye0[780],ye1[792],ye2[792],ye3[1056],ye4[528];
            encoder_module(gb,st.conv_cache_e0,st.tfa_cache_e0,st.conv_cache_e1,st.tfa_cache_e1,
                st.conv_cache_e2,st.tfa_cache_e2,st.tfa_cache_e3,st.tfa_cache_e4,ye0,ye1,ye2,ye3,ye4);
            int32_t c[528]; gdprnn_module(ye4,st.inter_cache_0,0,c);
            rep("4. GDPRNN idx=0",cmp(c,gr1,528)); total++; if(cmp(c,gr1,528).snr>130)passed++;
        }
        free(gb);free(gr1);
    }

    /* ── Test 5: GDPRNN idx=1 ── */
    {
        int32_t *gr1= load_i32_t(P("frame001_rnn1.bin"),16,33,&n);
        int32_t *gr2= load_i32_t(P("frame001_rnn2.bin"),16,33,&n);
        if(gr1&&gr2){
            ulunas_state_t st; ulunas_state_init(&st);
            int32_t c[528]; gdprnn_module(gr1,st.inter_cache_1,1,c);
            rep("5. GDPRNN idx=1",cmp(c,gr2,528)); total++; if(cmp(c,gr2,528).snr>130)passed++;
        }
        free(gr1);free(gr2);
    }

    /* ── Test 6: Decoder ── */
    {
        int32_t *gb  = load_i32_t(P("frame001_bm.bin"),1,129,&n);
        int32_t *gr2 = load_i32_t(P("frame001_rnn2.bin"),16,33,&n);
        int32_t *gdec= load_i32_t(P("frame001_dec.bin"),1,129,&n);
        if(gb&&gr2&&gdec){
            ulunas_state_t st; ulunas_state_init(&st);
            int32_t ye0[780],ye1[792],ye2[792],ye3[1056],ye4[528];
            encoder_module(gb,st.conv_cache_e0,st.tfa_cache_e0,st.conv_cache_e1,st.tfa_cache_e1,
                st.conv_cache_e2,st.tfa_cache_e2,st.tfa_cache_e3,st.tfa_cache_e4,ye0,ye1,ye2,ye3,ye4);
            int32_t yr1[528],yr2[528];
            gdprnn_module(ye4,st.inter_cache_0,0,yr1);
            gdprnn_module(yr1,st.inter_cache_1,1,yr2);
            int32_t c[129];
            decoder_module(yr2,ye4,ye3,ye2,ye1,ye0,st.tfa_cache_d0,st.tfa_cache_d1,
                st.conv_cache_d0,st.tfa_cache_d2,st.conv_cache_d1,st.tfa_cache_d3,
                st.conv_cache_d2,st.tfa_cache_d4,c);
            rep("6. Decoder (D0→D4)",cmp(c,gdec,129)); total++; if(cmp(c,gdec,129).snr>130)passed++;
        }
        free(gb);free(gr2);free(gdec);
    }

    printf("\n=== %d/%d tests passed ===\n", passed, total);
    return (passed==total)?0:1;
}
