/* ulunas_modules.c — Complete Encoder/Decoder/GDPRNN with all weight references
 * Auto-wired from MATLAB source analysis. Each function matches MATLAB 1:1.
 */
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

/* ===== GENERIC HELPERS ===== */

/* PConv grouped×2 → BN → AffinePReLU */
void pconv_g2_aff(
    const int32_t *x, int Ch, int Co, int W,
    const int16_t *pw, const int32_t *pb,
    const int16_t *bw, const int32_t *bb, const int32_t *bm, const uint16_t *bv,
    const int16_t *aw, const int32_t *ab, const int16_t *as,
    int cqr, int b1, int b2, int a1, int a2, int32_t *y)
{
    int Co2 = Co*2, N = Co2*W;
    int32_t y0[Co*W], y1[Co*W];
    pconv2d_fp(x, Ch, Co, W, pw, pb, cqr, Co*2, y0);
    pconv2d_fp(&x[Ch*W], Ch, Co, W, &pw[Co], &pb[Co], cqr, Co*2, y1);
    int32_t yc[Co2*W];
    for(int i=0;i<Co*W;i++){yc[i]=y0[i];yc[Co*W+i]=y1[i];}
    int32_t yb[Co2*W];
    bn_fp(yc,Co2,W,bw,bb,bm,bv,b1,b2,yb);
    affineprelu_fp(yb,Co2,W,aw,ab,as,a1,a2,y);
}

/* PConv grouped×2 → BN only (no AffinePReLU) */
void pconv_g2_bn(
    const int32_t *x, int Ch, int Co, int W,
    const int16_t *pw, const int32_t *pb,
    const int16_t *bw, const int32_t *bb, const int32_t *bm, const uint16_t *bv,
    int cqr, int b1, int b2, int32_t *y)
{
    int Co2=Co*2, N=Co2*W;
    int32_t y0[Co*W], y1[Co*W];
    pconv2d_fp(x,Ch,Co,W,pw,pb,cqr,Co*2,y0);
    pconv2d_fp(&x[Ch*W],Ch,Co,W,&pw[Co],&pb[Co],cqr,Co*2,y1);
    int32_t yc[Co2*W];
    for(int i=0;i<Co*W;i++){yc[i]=y0[i];yc[Co*W+i]=y1[i];}
    bn_fp(yc,Co2,W,bw,bb,bm,bv,b1,b2,y);
}

/* GConv (cached) → BN → AffinePReLU */
void gconv_aff(
    const int32_t *x, const int32_t *cc, int C, int Wo,
    int Kh, int Kw, int sh, int sw,
    const int16_t *cw, const int32_t *cb,
    const int16_t *bw, const int32_t *bb, const int32_t *bm, const uint16_t *bv,
    const int16_t *aw, const int32_t *ab, const int16_t *as,
    int cqr, int b1, int b2, int a1, int a2, int32_t *y, int32_t *cn)
{
    int N=C*Wo; int32_t yc[C*Wo];
    gconv2d_fp(x,cc,C,1,Wo,Kh,Kw,sh,sw,cw,cb,cqr,yc,cn);
    int32_t yb[C*Wo];
    bn_fp(yc,C,Wo,bw,bb,bm,bv,b1,b2,yb);
    affineprelu_fp(yb,C,Wo,aw,ab,as,a1,a2,y);
}

/* nonGConv → BN → AffinePReLU */
void ngconv_aff(
    const int32_t *x, int C, int Wo,
    int Kh, int Kw, int sh, int sw,
    const int16_t *cw, const int32_t *cb,
    const int16_t *bw, const int32_t *bb, const int32_t *bm, const uint16_t *bv,
    const int16_t *aw, const int32_t *ab, const int16_t *as,
    int cqr, int b1, int b2, int a1, int a2, int32_t *y)
{
    int N=C*Wo; int32_t yc[C*Wo];
    non_gconv2d_fp(x,C,1,Wo,Kh,Kw,sh,sw,cw,cb,cqr,yc);
    int32_t yb[C*Wo];
    bn_fp(yc,C,Wo,bw,bb,bm,bv,b1,b2,yb);
    affineprelu_fp(yb,C,Wo,aw,ab,as,a1,a2,y);
}

/* GTConv (cached) → BN → AffinePReLU */
void gtconv_aff(
    const int32_t *x, const int32_t *cc, int C, int Wo,
    int Kh, int Kw, int sh, int sw,
    const int16_t *cw, const int32_t *cb,
    const int16_t *bw, const int32_t *bb, const int32_t *bm, const uint16_t *bv,
    const int16_t *aw, const int32_t *ab, const int16_t *as,
    int cqr, int b1, int b2, int a1, int a2, int32_t *y, int32_t *cn)
{
    int N=C*Wo; int32_t yc[C*Wo];
    gtconv2d_fp(x,cc,C,1,Wo,Kh,Kw,sh,sw,cw,cb,cqr,yc,cn);
    int32_t yb[C*Wo];
    bn_fp(yc,C,Wo,bw,bb,bm,bv,b1,b2,yb);
    affineprelu_fp(yb,C,Wo,aw,ab,as,a1,a2,y);
}

/* nonGTConv → BN → AffinePReLU */
void ngtconv_aff(
    const int32_t *x, int C, int Wo,
    int Kh, int Kw, int sh, int sw,
    const int16_t *cw, const int32_t *cb,
    const int16_t *bw, const int32_t *bb, const int32_t *bm, const uint16_t *bv,
    const int16_t *aw, const int32_t *ab, const int16_t *as,
    int cqr, int b1, int b2, int a1, int a2, int32_t *y)
{
    int N=C*Wo; int32_t yc[C*Wo];
    non_gtconv2d_fp(x,C,1,Wo,Kh,Kw,sh,sw,cw,cb,cqr,yc);
    int32_t yb[C*Wo];
    bn_fp(yc,C,Wo,bw,bb,bm,bv,b1,b2,yb);
    affineprelu_fp(yb,C,Wo,aw,ab,as,a1,a2,y);
}

/* cTFA fusion: TA + FA + apply */
void ctfa(
    const int32_t *x, int C, int W, int pad,
    int16_t *tc, int tn, int fn, int tidim,
    const int16_t *tiw, const int32_t *tib, const int16_t *thw, const int32_t *thb,
    const int16_t *tfw, const int32_t *tfb,
    const int16_t *fiw, const int32_t *fib, const int16_t *fhw, const int32_t *fhb,
    const int16_t *riw, const int32_t *rib, const int16_t *rhw, const int32_t *rhb,
    const int16_t *ffw, const int32_t *ffb,
    int32_t *y)
{
    uint16_t ta[64], fa[256]; /* max C=32, W=129 */
    ctfa_ta_fp(x,C,W,tc,tn,tidim,tiw,tib,thw,thb,tfw,tfb,ta);
    ctfa_fa_fp(x,C,W,pad,fn,fiw,fib,fhw,fhb,riw,rib,rhw,rhb,ffw,ffb,fa);
    ctfa_apply_fp(x,C,W,ta,fa,y);
}

/* Shuffle interleave + return */
void shuf(const int32_t *x, int C, int W, int32_t *y) {
    shuffle_interleave(x,C,W,y);
}

/* Skip connection add */
void skip_add(const int32_t *a, const int32_t *b, int N, int32_t *y) {
    for(int i=0;i<N;i++) y[i]=a[i]+b[i];
}

/* ======================================================================== */
/*  ENCODER LAYER 0: XConv  (already complete in encoder_xconv_module)       */
/* ======================================================================== */

void encoder_xconv_module(
    const int32_t *x, int32_t *conv_cache, int16_t *tfa_cache, int32_t *y);

/* ======================================================================== */
/*  ENCODER LAYER 2: XDWS0  [24,33]→[24,33]                                 */
/* ======================================================================== */
static void e2_xdws0(const int32_t *x, int32_t *cc, int16_t *tc, int32_t *y) {
    int32_t yp[24*33];
    pconv_g2_aff(x,12,12,33,
        encoder_en_convs_2_pconv_0_weight,encoder_en_convs_2_pconv_0_bias,
        encoder_en_convs_2_pconv_1_weight,encoder_en_convs_2_pconv_1_bias,
        encoder_en_convs_2_pconv_1_running_mean,encoder_en_convs_2_pconv_1_running_var,
        encoder_en_convs_2_pconv_2_affine_weight,encoder_en_convs_2_pconv_2_affine_bias,
        encoder_en_convs_2_pconv_2_slope_weight,
        -14,-11,-14,-13,-13,yp);
    int32_t ys[24*33]; shuf(yp,24,33,ys);
    int32_t yt[24*33];
    gconv_aff(ys,cc,24,33,2,3,1,1,
        encoder_en_convs_2_dconv_1_weight,encoder_en_convs_2_dconv_1_bias,
        encoder_en_convs_2_dconv_2_weight,encoder_en_convs_2_dconv_2_bias,
        encoder_en_convs_2_dconv_2_running_mean,encoder_en_convs_2_dconv_2_running_var,
        encoder_en_convs_2_dconv_3_affine_weight,encoder_en_convs_2_dconv_3_affine_bias,
        encoder_en_convs_2_dconv_3_slope_weight,
        -13,-14,-14,-13,-13,yt,cc);
    ctfa(yt,24,33,3,tc,48,4,24,
        encoder_en_convs_2_dconv_4_ta_gru_weight_ih_l0,encoder_en_convs_2_dconv_4_ta_gru_bias_ih_l0,
        encoder_en_convs_2_dconv_4_ta_gru_weight_hh_l0,encoder_en_convs_2_dconv_4_ta_gru_bias_hh_l0,
        encoder_en_convs_2_dconv_4_ta_fc_weight,encoder_en_convs_2_dconv_4_ta_fc_bias,
        encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0,encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0,
        encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0,encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0,
        encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse,encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse,
        encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse,encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
        encoder_en_convs_2_dconv_4_fa_fc_weight,encoder_en_convs_2_dconv_4_fa_fc_bias,y);
}

/* ======================================================================== */
/*  ENCODER LAYER 3: XMB1  [24,33]→[32,33]                                  */
/* ======================================================================== */
static void e3_xmb1(const int32_t *x, int16_t *tc, int32_t *y) {
    int32_t yp0[32*33];
    pconv_g2_aff(x,12,16,33,
        encoder_en_convs_3_pconv1_0_weight,encoder_en_convs_3_pconv1_0_bias,
        encoder_en_convs_3_pconv1_1_weight,encoder_en_convs_3_pconv1_1_bias,
        encoder_en_convs_3_pconv1_1_running_mean,encoder_en_convs_3_pconv1_1_running_var,
        encoder_en_convs_3_pconv1_2_affine_weight,encoder_en_convs_3_pconv1_2_affine_bias,
        encoder_en_convs_3_pconv1_2_slope_weight,
        -13,-11,-14,-13,-13,yp0);
    int32_t ys[32*33]; shuf(yp0,32,33,ys);
    int32_t ynt[32*33];
    ngconv_aff(ys,32,33,1,5,1,1,
        encoder_en_convs_3_dconv_1_weight,encoder_en_convs_3_dconv_1_bias,
        encoder_en_convs_3_dconv_2_weight,encoder_en_convs_3_dconv_2_bias,
        encoder_en_convs_3_dconv_2_running_mean,encoder_en_convs_3_dconv_2_running_var,
        encoder_en_convs_3_dconv_3_affine_weight,encoder_en_convs_3_dconv_3_affine_bias,
        encoder_en_convs_3_dconv_3_slope_weight,
        -13,-14,-14,-13,-13,ynt);
    int32_t yp1[32*33];
    pconv_g2_bn(ynt,16,16,33,
        encoder_en_convs_3_pconv2_0_weight,encoder_en_convs_3_pconv2_0_bias,
        encoder_en_convs_3_pconv2_1_weight,encoder_en_convs_3_pconv2_1_bias,
        encoder_en_convs_3_pconv2_1_running_mean,encoder_en_convs_3_pconv2_1_running_var,
        -14,-14,-14,yp1);
    int32_t yc[32*33];
    ctfa(yp1,32,33,3,tc,64,4,32,
        encoder_en_convs_3_pconv2_2_ta_gru_weight_ih_l0,encoder_en_convs_3_pconv2_2_ta_gru_bias_ih_l0,
        encoder_en_convs_3_pconv2_2_ta_gru_weight_hh_l0,encoder_en_convs_3_pconv2_2_ta_gru_bias_hh_l0,
        encoder_en_convs_3_pconv2_2_ta_fc_weight,encoder_en_convs_3_pconv2_2_ta_fc_bias,
        encoder_en_convs_3_pconv2_2_fa_gru_weight_ih_l0,encoder_en_convs_3_pconv2_2_fa_gru_bias_ih_l0,
        encoder_en_convs_3_pconv2_2_fa_gru_weight_hh_l0,encoder_en_convs_3_pconv2_2_fa_gru_bias_hh_l0,
        encoder_en_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse,encoder_en_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse,
        encoder_en_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse,encoder_en_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
        encoder_en_convs_3_pconv2_2_fa_fc_weight,encoder_en_convs_3_pconv2_2_fa_fc_bias,yc);
    shuf(yc,32,33,y);
}

/* ======================================================================== */
/*  ENCODER LAYER 4: XDWS1  [32,33]→[16,33]                                 */
/* ======================================================================== */
static void e4_xdws1(const int32_t *x, int16_t *tc, int32_t *y) {
    int32_t yp[16*33];
    pconv_g2_aff(x,16,8,33,
        encoder_en_convs_4_pconv_0_weight,encoder_en_convs_4_pconv_0_bias,
        encoder_en_convs_4_pconv_1_weight,encoder_en_convs_4_pconv_1_bias,
        encoder_en_convs_4_pconv_1_running_mean,encoder_en_convs_4_pconv_1_running_var,
        encoder_en_convs_4_pconv_2_affine_weight,encoder_en_convs_4_pconv_2_affine_bias,
        encoder_en_convs_4_pconv_2_slope_weight,
        -14,-11,-14,-13,-13,yp);
    int32_t ys[16*33]; shuf(yp,16,33,ys);
    int32_t ynt[16*33];
    ngconv_aff(ys,16,33,1,5,1,1,
        encoder_en_convs_4_dconv_1_weight,encoder_en_convs_4_dconv_1_bias,
        encoder_en_convs_4_dconv_2_weight,encoder_en_convs_4_dconv_2_bias,
        encoder_en_convs_4_dconv_2_running_mean,encoder_en_convs_4_dconv_2_running_var,
        encoder_en_convs_4_dconv_3_affine_weight,encoder_en_convs_4_dconv_3_affine_bias,
        encoder_en_convs_4_dconv_3_slope_weight,
        -14,-14,-14,-13,-13,ynt);
    ctfa(ynt,16,33,3,tc,32,4,16,
        encoder_en_convs_4_dconv_4_ta_gru_weight_ih_l0,encoder_en_convs_4_dconv_4_ta_gru_bias_ih_l0,
        encoder_en_convs_4_dconv_4_ta_gru_weight_hh_l0,encoder_en_convs_4_dconv_4_ta_gru_bias_hh_l0,
        encoder_en_convs_4_dconv_4_ta_fc_weight,encoder_en_convs_4_dconv_4_ta_fc_bias,
        encoder_en_convs_4_dconv_4_fa_gru_weight_ih_l0,encoder_en_convs_4_dconv_4_fa_gru_bias_ih_l0,
        encoder_en_convs_4_dconv_4_fa_gru_weight_hh_l0,encoder_en_convs_4_dconv_4_fa_gru_bias_hh_l0,
        encoder_en_convs_4_dconv_4_fa_gru_weight_ih_l0_reverse,encoder_en_convs_4_dconv_4_fa_gru_bias_ih_l0_reverse,
        encoder_en_convs_4_dconv_4_fa_gru_weight_hh_l0_reverse,encoder_en_convs_4_dconv_4_fa_gru_bias_hh_l0_reverse,
        encoder_en_convs_4_dconv_4_fa_fc_weight,encoder_en_convs_4_dconv_4_fa_fc_bias,y);
}

/* ======================================================================== */
/*  ENCODER LAYER 1: XMB0  [12,65]→[24,33]                                  */
/* ======================================================================== */
static void e1_xmb0(const int32_t *x, int32_t *cc, int16_t *tc, int32_t *y) {
    int32_t yp0[24*65];
    pconv_g2_aff(x,6,12,65,
        encoder_en_convs_1_pconv1_0_weight,encoder_en_convs_1_pconv1_0_bias,
        encoder_en_convs_1_pconv1_1_weight,encoder_en_convs_1_pconv1_1_bias,
        encoder_en_convs_1_pconv1_1_running_mean,encoder_en_convs_1_pconv1_1_running_var,
        encoder_en_convs_1_pconv1_2_affine_weight,encoder_en_convs_1_pconv1_2_affine_bias,
        encoder_en_convs_1_pconv1_2_slope_weight,
        -14,-11,-14,-13,-13,yp0);
    int32_t ys0[24*65]; shuf(yp0,24,65,ys0);
    int32_t yt[24*33];
    gconv_aff(ys0,cc,24,33,2,3,1,2,
        encoder_en_convs_1_dconv_1_weight,encoder_en_convs_1_dconv_1_bias,
        encoder_en_convs_1_dconv_2_weight,encoder_en_convs_1_dconv_2_bias,
        encoder_en_convs_1_dconv_2_running_mean,encoder_en_convs_1_dconv_2_running_var,
        encoder_en_convs_1_dconv_3_affine_weight,encoder_en_convs_1_dconv_3_affine_bias,
        encoder_en_convs_1_dconv_3_slope_weight,
        -14,-11,-14,-13,-13,yt,cc);
    int32_t yp1[24*33];
    pconv_g2_bn(yt,12,12,33,
        encoder_en_convs_1_pconv2_0_weight,encoder_en_convs_1_pconv2_0_bias,
        encoder_en_convs_1_pconv2_1_weight,encoder_en_convs_1_pconv2_1_bias,
        encoder_en_convs_1_pconv2_1_running_mean,encoder_en_convs_1_pconv2_1_running_var,
        -14,-14,-14,yp1);
    int32_t yc[24*33];
    ctfa(yp1,24,33,3,tc,48,4,24,
        encoder_en_convs_1_pconv2_2_ta_gru_weight_ih_l0,encoder_en_convs_1_pconv2_2_ta_gru_bias_ih_l0,
        encoder_en_convs_1_pconv2_2_ta_gru_weight_hh_l0,encoder_en_convs_1_pconv2_2_ta_gru_bias_hh_l0,
        encoder_en_convs_1_pconv2_2_ta_fc_weight,encoder_en_convs_1_pconv2_2_ta_fc_bias,
        encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0,encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0,
        encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0,encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0,
        encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse,encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse,
        encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse,encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
        encoder_en_convs_1_pconv2_2_fa_fc_weight,encoder_en_convs_1_pconv2_2_fa_fc_bias,yc);
    shuf(yc,24,33,y);
}

/* ======================================================================== */
/*  DECODER LAYER 0: De_XDWS0  [16,33]+skip_e4→[32,33]                      */
/* ======================================================================== */
void d0_xdws0(const int32_t *x, const int32_t *sk, int16_t *tc, int32_t *y) {
    int32_t xs[16*33]; skip_add(x,sk,16*33,xs);
    int32_t yp[32*33];
    pconv_g2_aff(xs,8,16,33,
        decoder_de_convs_0_pconv_0_weight,decoder_de_convs_0_pconv_0_bias,
        decoder_de_convs_0_pconv_1_weight,decoder_de_convs_0_pconv_1_bias,
        decoder_de_convs_0_pconv_1_running_mean,decoder_de_convs_0_pconv_1_running_var,
        decoder_de_convs_0_pconv_2_affine_weight,decoder_de_convs_0_pconv_2_affine_bias,
        decoder_de_convs_0_pconv_2_slope_weight,
        -14,-14,-14,-13,-13,yp);
    int32_t ys[32*33]; shuf(yp,32,33,ys);
    int32_t ynt[32*33];
    ngtconv_aff(ys,32,33,1,5,1,1,
        decoder_de_convs_0_dconv_1_weight,decoder_de_convs_0_dconv_1_bias,
        decoder_de_convs_0_dconv_2_weight,decoder_de_convs_0_dconv_2_bias,
        decoder_de_convs_0_dconv_2_running_mean,decoder_de_convs_0_dconv_2_running_var,
        decoder_de_convs_0_dconv_3_affine_weight,decoder_de_convs_0_dconv_3_affine_bias,
        decoder_de_convs_0_dconv_3_slope_weight,
        -14,-14,-14,-13,-13,ynt);
    ctfa(ynt,32,33,3,tc,64,4,32,
        decoder_de_convs_0_dconv_4_ta_gru_weight_ih_l0,decoder_de_convs_0_dconv_4_ta_gru_bias_ih_l0,
        decoder_de_convs_0_dconv_4_ta_gru_weight_hh_l0,decoder_de_convs_0_dconv_4_ta_gru_bias_hh_l0,
        decoder_de_convs_0_dconv_4_ta_fc_weight,decoder_de_convs_0_dconv_4_ta_fc_bias,
        decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0,decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0,
        decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0,decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0,
        decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0_reverse,decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0_reverse,
        decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0_reverse,decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0_reverse,
        decoder_de_convs_0_dconv_4_fa_fc_weight,decoder_de_convs_0_dconv_4_fa_fc_bias,y);
}

/* ======================================================================== */
/*  DECODER LAYER 1: De_XMB0  [32,33]+skip_e3→[24,33]                       */
/* ======================================================================== */
void d1_xmb0(const int32_t *x, const int32_t *sk, int16_t *tc, int32_t *y) {
    int32_t xs[32*33]; skip_add(x,sk,32*33,xs);
    int32_t yp0[24*33];
    pconv_g2_aff(xs,16,12,33,
        decoder_de_convs_1_pconv1_0_weight,decoder_de_convs_1_pconv1_0_bias,
        decoder_de_convs_1_pconv1_1_weight,decoder_de_convs_1_pconv1_1_bias,
        decoder_de_convs_1_pconv1_1_running_mean,decoder_de_convs_1_pconv1_1_running_var,
        decoder_de_convs_1_pconv1_2_affine_weight,decoder_de_convs_1_pconv1_2_affine_bias,
        decoder_de_convs_1_pconv1_2_slope_weight,
        -13,-11,-14,-13,-13,yp0);
    int32_t ys0[24*33]; shuf(yp0,24,33,ys0);
    int32_t ynt[24*33];
    ngtconv_aff(ys0,24,33,1,5,1,1,
        decoder_de_convs_1_dconv_1_weight,decoder_de_convs_1_dconv_1_bias,
        decoder_de_convs_1_dconv_2_weight,decoder_de_convs_1_dconv_2_bias,
        decoder_de_convs_1_dconv_2_running_mean,decoder_de_convs_1_dconv_2_running_var,
        decoder_de_convs_1_dconv_3_affine_weight,decoder_de_convs_1_dconv_3_affine_bias,
        decoder_de_convs_1_dconv_3_slope_weight,
        -14,-11,-14,-13,-13,ynt);
    int32_t yp1[24*33];
    pconv_g2_bn(ynt,12,12,33,
        decoder_de_convs_1_pconv2_0_weight,decoder_de_convs_1_pconv2_0_bias,
        decoder_de_convs_1_pconv2_1_weight,decoder_de_convs_1_pconv2_1_bias,
        decoder_de_convs_1_pconv2_1_running_mean,decoder_de_convs_1_pconv2_1_running_var,
        -14,-11,-11,yp1);
    int32_t yc[24*33];
    ctfa(yp1,24,33,3,tc,48,4,24,
        decoder_de_convs_1_pconv2_2_ta_gru_weight_ih_l0,decoder_de_convs_1_pconv2_2_ta_gru_bias_ih_l0,
        decoder_de_convs_1_pconv2_2_ta_gru_weight_hh_l0,decoder_de_convs_1_pconv2_2_ta_gru_bias_hh_l0,
        decoder_de_convs_1_pconv2_2_ta_fc_weight,decoder_de_convs_1_pconv2_2_ta_fc_bias,
        decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0,decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0,
        decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0,decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0,
        decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse,decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse,
        decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse,decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
        decoder_de_convs_1_pconv2_2_fa_fc_weight,decoder_de_convs_1_pconv2_2_fa_fc_bias,yc);
    shuf(yc,24,33,y);
}

/* ======================================================================== */
/*  DECODER LAYER 2: De_XDWS1  [24,33]+skip_e2→[24,33]                      */
/* ======================================================================== */
void d2_xdws1(const int32_t *x, const int32_t *sk, int32_t *cc, int16_t *tc, int32_t *y) {
    int32_t xs[24*33]; skip_add(x,sk,24*33,xs);
    int32_t yp[24*33];
    pconv_g2_aff(xs,12,12,33,
        decoder_de_convs_2_pconv_0_weight,decoder_de_convs_2_pconv_0_bias,
        decoder_de_convs_2_pconv_1_weight,decoder_de_convs_2_pconv_1_bias,
        decoder_de_convs_2_pconv_1_running_mean,decoder_de_convs_2_pconv_1_running_var,
        decoder_de_convs_2_pconv_2_affine_weight,decoder_de_convs_2_pconv_2_affine_bias,
        decoder_de_convs_2_pconv_2_slope_weight,
        -14,-11,-14,-13,-13,yp);
    int32_t ys[24*33]; shuf(yp,24,33,ys);
    int32_t yt[24*33];
    gtconv_aff(ys,cc,24,33,2,3,1,1,
        decoder_de_convs_2_dconv_1_weight,decoder_de_convs_2_dconv_1_bias,
        decoder_de_convs_2_dconv_2_weight,decoder_de_convs_2_dconv_2_bias,
        decoder_de_convs_2_dconv_2_running_mean,decoder_de_convs_2_dconv_2_running_var,
        decoder_de_convs_2_dconv_3_affine_weight,decoder_de_convs_2_dconv_3_affine_bias,
        decoder_de_convs_2_dconv_3_slope_weight,
        -13,-11,-14,-13,-13,yt,cc);
    ctfa(yt,24,33,3,tc,48,4,24,
        decoder_de_convs_2_dconv_4_ta_gru_weight_ih_l0,decoder_de_convs_2_dconv_4_ta_gru_bias_ih_l0,
        decoder_de_convs_2_dconv_4_ta_gru_weight_hh_l0,decoder_de_convs_2_dconv_4_ta_gru_bias_hh_l0,
        decoder_de_convs_2_dconv_4_ta_fc_weight,decoder_de_convs_2_dconv_4_ta_fc_bias,
        decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0,decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0,
        decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0,decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0,
        decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse,decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse,
        decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse,decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
        decoder_de_convs_2_dconv_4_fa_fc_weight,decoder_de_convs_2_dconv_4_fa_fc_bias,y);
}

/* ======================================================================== */
/*  DECODER LAYER 3: De_XMB1  [24,33]+skip_e1→[12,65]                       */
/* ======================================================================== */
void d3_xmb1(const int32_t *x, const int32_t *sk, int32_t *cc, int16_t *tc, int32_t *y) {
    int32_t xs[24*33]; skip_add(x,sk,24*33,xs);
    int32_t yp0[12*33];
    pconv_g2_aff(xs,12,6,33,
        decoder_de_convs_3_pconv1_0_weight,decoder_de_convs_3_pconv1_0_bias,
        decoder_de_convs_3_pconv1_1_weight,decoder_de_convs_3_pconv1_1_bias,
        decoder_de_convs_3_pconv1_1_running_mean,decoder_de_convs_3_pconv1_1_running_var,
        decoder_de_convs_3_pconv1_2_affine_weight,decoder_de_convs_3_pconv1_2_affine_bias,
        decoder_de_convs_3_pconv1_2_slope_weight,
        -14,-11,-14,-13,-13,yp0);
    int32_t ys0[12*33]; shuf(yp0,12,33,ys0);
    int32_t yt[12*65];
    gtconv_aff(ys0,cc,12,65,2,3,1,2,
        decoder_de_convs_3_dconv_1_weight,decoder_de_convs_3_dconv_1_bias,
        decoder_de_convs_3_dconv_2_weight,decoder_de_convs_3_dconv_2_bias,
        decoder_de_convs_3_dconv_2_running_mean,decoder_de_convs_3_dconv_2_running_var,
        decoder_de_convs_3_dconv_3_affine_weight,decoder_de_convs_3_dconv_3_affine_bias,
        decoder_de_convs_3_dconv_3_slope_weight,
        -14,-11,-11,-13,-13,yt,cc);
    int32_t yp1[12*65];
    pconv_g2_bn(yt,6,6,65,
        decoder_de_convs_3_pconv2_0_weight,decoder_de_convs_3_pconv2_0_bias,
        decoder_de_convs_3_pconv2_1_weight,decoder_de_convs_3_pconv2_1_bias,
        decoder_de_convs_3_pconv2_1_running_mean,decoder_de_convs_3_pconv2_1_running_var,
        -14,-11,-11,yp1);
    int32_t yc[12*65];
    ctfa(yp1,12,65,3,tc,24,4,12,
        decoder_de_convs_3_pconv2_2_ta_gru_weight_ih_l0,decoder_de_convs_3_pconv2_2_ta_gru_bias_ih_l0,
        decoder_de_convs_3_pconv2_2_ta_gru_weight_hh_l0,decoder_de_convs_3_pconv2_2_ta_gru_bias_hh_l0,
        decoder_de_convs_3_pconv2_2_ta_fc_weight,decoder_de_convs_3_pconv2_2_ta_fc_bias,
        decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0,decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0,
        decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0,decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0,
        decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse,decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse,
        decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse,decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
        decoder_de_convs_3_pconv2_2_fa_fc_weight,decoder_de_convs_3_pconv2_2_fa_fc_bias,yc);
    shuf(yc,12,65,y);
}

/* ======================================================================== */
/*  DECODER LAYER 4: De_XConv  [12,65]+skip_e0→[1,129]                      */
/*  Verified: bias[1] weight[12×1×3×3=108] → Cout=1, cTFA TA input_dim=1    */
/* ======================================================================== */
void d4_xconv(const int32_t *x, const int32_t *sk, int32_t *cc, int16_t *tc, int32_t *y) {
    int32_t xs[12*65]; skip_add(x,sk,12*65,xs);
    /* Build 3D cache: x_cache = cat(2, cc[12,2,65], reshape(xs,[12,1,65])) */
    int32_t xc[12*3*65]; /* [12,3,65] */
    for(int c=0;c<12;c++){
        for(int f=0;f<2;f++) for(int w=0;w<65;w++) xc[(c*3+f)*65+w]=cc[(c*2+f)*65+w];
        for(int w=0;w<65;w++) xc[(c*3+2)*65+w]=xs[c*65+w];
    }
    /* TConv: tconv2d [12,3,65] → [1,129]  (Cin=12, Cout=1) */
    int32_t yt[1*129];
    tconv2d_fp(xc,12,1,1,129,3,3,1,2,
        decoder_de_convs_4_ops_1_weight,decoder_de_convs_4_ops_1_bias,-14,yt);
    /* Update cache: xc(:,2:3,:) → cc[12,2,65] */
    for(int c=0;c<12;c++){
        for(int f=0;f<2;f++) for(int w=0;w<65;w++) cc[(c*2+f)*65+w]=xc[(c*3+(f+1))*65+w];
    }
    /* BN: [1,129], Qr1=-11, Qr2=-11 (unique to D4) */
    int32_t yb[1*129];
    bn_fp(yt,1,129,
        decoder_de_convs_4_ops_2_weight,decoder_de_convs_4_ops_2_bias,
        decoder_de_convs_4_ops_2_running_mean,decoder_de_convs_4_ops_2_running_var,
        -11,-11,yb);
    /* cTFA: C=1, W=129, TA nHidden=2 input_dim=1, FA nHidden=4 */
    ctfa(yb,1,129,3,tc,2,4,1,
        decoder_de_convs_4_ops_4_ta_gru_weight_ih_l0,decoder_de_convs_4_ops_4_ta_gru_bias_ih_l0,
        decoder_de_convs_4_ops_4_ta_gru_weight_hh_l0,decoder_de_convs_4_ops_4_ta_gru_bias_hh_l0,
        decoder_de_convs_4_ops_4_ta_fc_weight,decoder_de_convs_4_ops_4_ta_fc_bias,
        decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0,decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0,
        decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0,decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0,
        decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0_reverse,decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0_reverse,
        decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0_reverse,decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0_reverse,
        decoder_de_convs_4_ops_4_fa_fc_weight,decoder_de_convs_4_ops_4_fa_fc_bias,y);
}

/* ======================================================================== */
/*  TOP-LEVEL ENCODER                                                        */
/* ======================================================================== */

void encoder_module(
    const int32_t *x,
    int32_t *conv_cache_e0, int16_t *tfa_cache_e0,
    int32_t *conv_cache_e1, int16_t *tfa_cache_e1,
    int32_t *conv_cache_e2, int16_t *tfa_cache_e2,
    int16_t *tfa_cache_e3,
    int16_t *tfa_cache_e4,
    int32_t *y_e0, int32_t *y_e1, int32_t *y_e2,
    int32_t *y_e3, int32_t *y_e4)
{
    encoder_xconv_module(x, conv_cache_e0, tfa_cache_e0, y_e0);
    e1_xmb0(y_e0, conv_cache_e1, tfa_cache_e1, y_e1);
    e2_xdws0(y_e1, conv_cache_e2, tfa_cache_e2, y_e2);
    e3_xmb1(y_e2, tfa_cache_e3, y_e3);
    e4_xdws1(y_e3, tfa_cache_e4, y_e4);
}

/* ======================================================================== */
/*  TOP-LEVEL DECODER                                                        */
/* ======================================================================== */

void decoder_module(
    const int32_t *x,
    const int32_t *skip_e4, const int32_t *skip_e3,
    const int32_t *skip_e2, const int32_t *skip_e1, const int32_t *skip_e0,
    int16_t *tfa_cache_d0, int16_t *tfa_cache_d1,
    int32_t *conv_cache_d0, int16_t *tfa_cache_d2,
    int32_t *conv_cache_d1, int16_t *tfa_cache_d3,
    int32_t *conv_cache_d2, int16_t *tfa_cache_d4,
    int32_t *y)
{
    int32_t y_d0[32*33], y_d1[24*33], y_d2[24*33], y_d3[12*65];
    d0_xdws0(x, skip_e4, tfa_cache_d0, y_d0);
    d1_xmb0(y_d0, skip_e3, tfa_cache_d1, y_d1);
    d2_xdws1(y_d1, skip_e2, conv_cache_d0, tfa_cache_d2, y_d2);
    d3_xmb1(y_d2, skip_e1, conv_cache_d1, tfa_cache_d3, y_d3);
    d4_xconv(y_d3, skip_e0, conv_cache_d2, tfa_cache_d4, y);
}

/* ======================================================================== */
/*  GDPRNN MODULE (unchanged from previous version)                           */
/* ======================================================================== */

void gdprnn_module(
    const int32_t *x, int16_t *inter_cache, int gdprnn_idx, int32_t *y)
{
    int32_t x_t[33*16];
    for(int t=0;t<33;t++) for(int c=0;c<16;c++) x_t[t*16+c]=x[c*33+t];

    /* Weight pointer tables */
    const int16_t *ir1_iw,*ir2_iw,*ir1_hw,*ir2_hw,*ir1_riw,*ir2_riw,*ir1_rhw,*ir2_rhw;
    const int32_t *ir1_ib,*ir2_ib,*ir1_hb,*ir2_hb,*ir1_rib,*ir2_rib,*ir1_rhb,*ir2_rhb;
    const int16_t *ifc_w,*iln_w; const int32_t *ifc_b,*iln_b;
    const int16_t *er1_iw,*er2_iw,*er1_hw,*er2_hw;
    const int32_t *er1_ib,*er2_ib,*er1_hb,*er2_hb;
    const int16_t *efc_w,*eln_w; const int32_t *efc_b,*eln_b;

    if(gdprnn_idx==0){
        ir1_iw=dpgrnn_0_intra_rnn_rnn1_weight_ih_l0; ir1_ib=dpgrnn_0_intra_rnn_rnn1_bias_ih_l0;
        ir1_hw=dpgrnn_0_intra_rnn_rnn1_weight_hh_l0; ir1_hb=dpgrnn_0_intra_rnn_rnn1_bias_hh_l0;
        ir1_riw=dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse; ir1_rib=dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse;
        ir1_rhw=dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse; ir1_rhb=dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse;
        ir2_iw=dpgrnn_0_intra_rnn_rnn2_weight_ih_l0; ir2_ib=dpgrnn_0_intra_rnn_rnn2_bias_ih_l0;
        ir2_hw=dpgrnn_0_intra_rnn_rnn2_weight_hh_l0; ir2_hb=dpgrnn_0_intra_rnn_rnn2_bias_hh_l0;
        ir2_riw=dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse; ir2_rib=dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse;
        ir2_rhw=dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse; ir2_rhb=dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse;
        ifc_w=dpgrnn_0_intra_fc_weight; ifc_b=dpgrnn_0_intra_fc_bias;
        iln_w=dpgrnn_0_intra_ln_weight; iln_b=dpgrnn_0_intra_ln_bias;
        er1_iw=dpgrnn_0_inter_rnn_rnn1_weight_ih_l0; er1_ib=dpgrnn_0_inter_rnn_rnn1_bias_ih_l0;
        er1_hw=dpgrnn_0_inter_rnn_rnn1_weight_hh_l0; er1_hb=dpgrnn_0_inter_rnn_rnn1_bias_hh_l0;
        er2_iw=dpgrnn_0_inter_rnn_rnn2_weight_ih_l0; er2_ib=dpgrnn_0_inter_rnn_rnn2_bias_ih_l0;
        er2_hw=dpgrnn_0_inter_rnn_rnn2_weight_hh_l0; er2_hb=dpgrnn_0_inter_rnn_rnn2_bias_hh_l0;
        efc_w=dpgrnn_0_inter_fc_weight; efc_b=dpgrnn_0_inter_fc_bias;
        eln_w=dpgrnn_0_inter_ln_weight; eln_b=dpgrnn_0_inter_ln_bias;
    }else{
        ir1_iw=dpgrnn_1_intra_rnn_rnn1_weight_ih_l0; ir1_ib=dpgrnn_1_intra_rnn_rnn1_bias_ih_l0;
        ir1_hw=dpgrnn_1_intra_rnn_rnn1_weight_hh_l0; ir1_hb=dpgrnn_1_intra_rnn_rnn1_bias_hh_l0;
        ir1_riw=dpgrnn_1_intra_rnn_rnn1_weight_ih_l0_reverse; ir1_rib=dpgrnn_1_intra_rnn_rnn1_bias_ih_l0_reverse;
        ir1_rhw=dpgrnn_1_intra_rnn_rnn1_weight_hh_l0_reverse; ir1_rhb=dpgrnn_1_intra_rnn_rnn1_bias_hh_l0_reverse;
        ir2_iw=dpgrnn_1_intra_rnn_rnn2_weight_ih_l0; ir2_ib=dpgrnn_1_intra_rnn_rnn2_bias_ih_l0;
        ir2_hw=dpgrnn_1_intra_rnn_rnn2_weight_hh_l0; ir2_hb=dpgrnn_1_intra_rnn_rnn2_bias_hh_l0;
        ir2_riw=dpgrnn_1_intra_rnn_rnn2_weight_ih_l0_reverse; ir2_rib=dpgrnn_1_intra_rnn_rnn2_bias_ih_l0_reverse;
        ir2_rhw=dpgrnn_1_intra_rnn_rnn2_weight_hh_l0_reverse; ir2_rhb=dpgrnn_1_intra_rnn_rnn2_bias_hh_l0_reverse;
        ifc_w=dpgrnn_1_intra_fc_weight; ifc_b=dpgrnn_1_intra_fc_bias;
        iln_w=dpgrnn_1_intra_ln_weight; iln_b=dpgrnn_1_intra_ln_bias;
        er1_iw=dpgrnn_1_inter_rnn_rnn1_weight_ih_l0; er1_ib=dpgrnn_1_inter_rnn_rnn1_bias_ih_l0;
        er1_hw=dpgrnn_1_inter_rnn_rnn1_weight_hh_l0; er1_hb=dpgrnn_1_inter_rnn_rnn1_bias_hh_l0;
        er2_iw=dpgrnn_1_inter_rnn_rnn2_weight_ih_l0; er2_ib=dpgrnn_1_inter_rnn_rnn2_bias_ih_l0;
        er2_hw=dpgrnn_1_inter_rnn_rnn2_weight_hh_l0; er2_hb=dpgrnn_1_inter_rnn_rnn2_bias_hh_l0;
        efc_w=dpgrnn_1_inter_fc_weight; efc_b=dpgrnn_1_inter_fc_bias;
        eln_w=dpgrnn_1_inter_ln_weight; eln_b=dpgrnn_1_inter_ln_bias;
    }

    /* Intra-RNN */
    int32_t x0[33*8],x1[33*8];
    for(int t=0;t<33;t++){for(int c=0;c<8;c++){x0[t*8+c]=x_t[t*16+c];x1[t*8+c]=x_t[t*16+8+c];}}
    int16_t g0[33*8],g1[33*8];
    bigru_sequence_fp(x0,33,8,4,ir1_iw,ir1_ib,ir1_hw,ir1_hb,ir1_riw,ir1_rib,ir1_rhw,ir1_rhb,-13,-8,g0);
    bigru_sequence_fp(x1,33,8,4,ir2_iw,ir2_ib,ir2_hw,ir2_hb,ir2_riw,ir2_rib,ir2_rhw,ir2_rhb,-13,-8,g1);
    int16_t xg[33*16];
    for(int t=0;t<33;t++){for(int c=0;c<8;c++){xg[t*16+c]=g0[t*8+c];xg[t*16+8+c]=g1[t*8+c];}}
    int32_t xf[33*16]; fc_fp_s16(xg,33,16,16,ifc_w,ifc_b,-9,xf);
    int32_t xl[33*16]; for(int t=0;t<33;t++)ln_fp(&xf[t*16],16,iln_w,iln_b,-14,&xl[t*16]);
    int32_t yi[33*16]; for(int i=0;i<33*16;i++)yi[i]=x_t[i]+xl[i];

    /* Inter-RNN */
    int32_t y0[33*8],y1[33*8];
    for(int t=0;t<33;t++){for(int c=0;c<8;c++){y0[t*8+c]=yi[t*16+c];y1[t*8+c]=yi[t*16+8+c];}}
    int16_t h0[33*8],h1[33*8];
    gru_sequence_fp(y0,33,8,&inter_cache[0],8,er1_iw,er1_ib,er1_hw,er1_hb,-13,-8,h0);
    gru_sequence_fp(y1,33,8,&inter_cache[8],8,er2_iw,er2_ib,er2_hw,er2_hb,-13,-8,h1);
    int16_t xig[33*16];
    for(int t=0;t<33;t++){for(int c=0;c<8;c++){xig[t*16+c]=h0[t*8+c];xig[t*16+8+c]=h1[t*8+c];}}
    int32_t xif[33*16]; fc_fp_s16(xig,33,16,16,efc_w,efc_b,-9,xif);
    int32_t xil[33*16]; for(int t=0;t<33;t++)ln_fp(&xif[t*16],16,eln_w,eln_b,-13,&xil[t*16]);
    int32_t yo[33*16]; for(int i=0;i<33*16;i++)yo[i]=yi[i]+xil[i];
    for(int c=0;c<16;c++) for(int t=0;t<33;t++) y[c*33+t]=yo[t*16+c];
}

/* Encoder Layer 0 — moved to top of file for dependencies */
void encoder_xconv_module(
    const int32_t *x, int32_t *conv_cache, int16_t *tfa_cache, int32_t *y)
{
    int32_t xc[3*129];
    for(int i=0;i<E0_CACHE_FRAMES*E0_IN_W;i++)xc[i]=conv_cache[i];
    for(int i=0;i<E0_IN_W;i++)xc[E0_CACHE_FRAMES*E0_IN_W+i]=x[i];
    int32_t yt[E0_OUT_C*E0_OUT_W];
    conv2d_fp(xc,E0_IN_C,E0_OUT_C,1,E0_OUT_W,E0_KH,E0_KW,E0_STRIDE_H,E0_STRIDE_W,
        encoder_en_convs_0_ops_1_weight,encoder_en_convs_0_ops_1_bias,E0_TCONV_CONV_QR,yt);
    for(int i=0;i<E0_CACHE_FRAMES*E0_IN_W;i++)conv_cache[i]=xc[E0_IN_W+i];
    int32_t yb[E0_OUT_C*E0_OUT_W];
    bn_fp(yt,E0_OUT_C,E0_OUT_W,
        encoder_en_convs_0_ops_2_weight,encoder_en_convs_0_ops_2_bias,
        encoder_en_convs_0_ops_2_running_mean,encoder_en_convs_0_ops_2_running_var,
        E0_TCONV_BN_QR1,E0_TCONV_BN_QR2,yb);
    int32_t ya[E0_OUT_C*E0_OUT_W];
    affineprelu_fp(yb,E0_OUT_C,E0_OUT_W,
        encoder_en_convs_0_ops_3_affine_weight,encoder_en_convs_0_ops_3_affine_bias,
        encoder_en_convs_0_ops_3_slope_weight,
        E0_TCONV_AFFINE_QR1,E0_TCONV_AFFINE_QR2,ya);
    uint16_t ta[12],fa[65];
    ctfa_ta_fp(ya,E0_OUT_C,E0_OUT_W,tfa_cache,E0_TA_GRU_HID,E0_OUT_C,
        encoder_en_convs_0_ops_4_ta_gru_weight_ih_l0,encoder_en_convs_0_ops_4_ta_gru_bias_ih_l0,
        encoder_en_convs_0_ops_4_ta_gru_weight_hh_l0,encoder_en_convs_0_ops_4_ta_gru_bias_hh_l0,
        encoder_en_convs_0_ops_4_ta_fc_weight,encoder_en_convs_0_ops_4_ta_fc_bias,ta);
    ctfa_fa_fp(ya,E0_OUT_C,E0_OUT_W,E0_FA_PADLEN,E0_FA_GRU_HID,
        encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0,encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0,
        encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0,encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0,
        encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0_reverse,encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0_reverse,
        encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0_reverse,encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0_reverse,
        encoder_en_convs_0_ops_4_fa_fc_weight,encoder_en_convs_0_ops_4_fa_fc_bias,fa);
    ctfa_apply_fp(ya,E0_OUT_C,E0_OUT_W,ta,fa,y);
}

