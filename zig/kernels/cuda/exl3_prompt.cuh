// Common EXL3 prompt helpers, after @BobClawblaw (#976) and MiaAI-Lab (Apache-2.0, patches 0004 and 0009).
#pragma once
#include "common/cp_async.cuh"
#include "common/qmm_frag.cuh"
namespace tf_exl3_prompt {
using namespace tf_exl3;
// One mcg tile of K2 half-bits a value (8: 4 bits, 6: 3 bits) from shared memory into this lane's B fragments.
template <int K2>
__device__ __forceinline__ void decode_tile_p(const uint32_t* tile, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t w[lane_words<K2>()];
    int word, offset;
    lane_start<K2>(lane, word, offset);
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = tile[(word + i) % tile_words<K2>()];
    decode_lane<K2, CB_MCG>(w, lane, b0, b1);
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
}
using tf_common::cp_commit;
using tf_common::cp_wait;
using qmm_frag::ldmatrix4;
}
namespace tf_exl3_prompt {

constexpr int TPM = 64, TNW = 8, TMT = 16 / TNW, TNB = TPM / 8, TT = TNW * 32, TPH = TPM / 2;
constexpr int TBK = 32, TKT = TBK / 16, TSTAGES = 3, TXS = TBK + 8;
template <int K2>
constexpr int twords() { return 16 * TKT * tile_words<K2>(); }   // a stage's weight words: 16 tiles by TKT
constexpr int GU_ES = 128 + 4, DN_ES = 256 + 4;                        // the epilogue sums' row strides (floats)
template <int XR, int K2>
constexpr int tstage_halves() { return XR * TXS + 2 * twords<K2>(); }
template <int K2>
constexpr int tsmem_bytes() {
    const int gu = TSTAGES * tstage_halves<2 * TPM, K2>() * 2;   // gate and up rows apart (no shared rotation)
    const int ep = 2 * TPH * GU_ES * 4, dn = TPH * DN_ES * 4;
    return gu > ep ? (gu > dn ? gu : dn) : (ep > dn ? ep : dn);
}

// Which (matrix, weight tile) warp-tile q (0..15) of column block cb is.
template <int MATS>
__device__ __forceinline__ void ttile(int q, int cb, int& mat, int& nt) {
    mat = MATS == 2 ? q >> 3 : 0;
    nt = MATS == 2 ? cb * 8 + (q & 7) : cb * 16 + q;
}

// The main loop (exl3p, after the Python recipe's K12 loop, patches/aevonix/0073): the same items, blocks, B fragments
// and fp32 chains as the loop it replaces (each output one chain of mma.m16n8k16 over the ascending k tiles from zero),
// fed differently:
//  - compiled for the item's NBA 8-member column blocks (1..8): one basic block a k tile, no predicate, no mma on a
//    column block without a member and no member-row load nobody reads;
//  - two decoded A fragment sets: the next k tile's trellis decode issues among the current k tile's mma, across the
//    stage barrier too (stage c + 1 is waited for at the top of stage c, so PS stages hold PS - 2 in flight);
//  - each thread's cp.async source and destination worked out once an item, not once a stage;
//  - the mcg mask in a register (mcg2), and 4-bit tiles decoded from the lane's word and its neighbour's in fewer steps
//    (decode_a8: decode_lane<8>'s states, so its values).
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
    uint32_t v;
    asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(a));
    return v;
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm2(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void cp16s(uint32_t dst, const void* src, int n) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
// decode2<CB_MCG>(s0, s1) with each (x & 0x8FFF8FFF) ^ 0x3B603B60 one LOP3 on a mask register the compiler keeps (for
// the C++ and/xor it re-creates an immediate before every LOP3); the same bits.
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    const uint32_t m = 0x8FFF8FFFu;
    uint32_t x0, x1;
    asm("lop3.b32 %0, %1, %2, 0x3B603B60, 0x6a;\n" : "=r"(x0) : "r"(s0 * 0xCBAC1FEDu), "r"(m));
    asm("lop3.b32 %0, %1, %2, 0x3B603B60, 0x6a;\n" : "=r"(x1) : "r"(s1 * 0xCBAC1FEDu), "r"(m));
    const uint32_t lo = __byte_perm(x0, x1, 0x5410), hi = __byte_perm(x0, x1, 0x7632);
    const half2 r = __hadd2(*reinterpret_cast<const half2*>(&lo), *reinterpret_cast<const half2*>(&hi));
    return *reinterpret_cast<const uint32_t*>(&r);
}
// A 4-bit tile's A fragment for this lane from its word w and the word of lane - 1, p: lane_states<8>'s eight states
// (s0 = p:w bits 43..28, s1 = 39..24, s2 = 35..20, s3 = w 31..16, s4 = 27..12, s5 = 23..8, s6 = 19..4, s7 = 15..0) in
// fewer shifts, in decode_tile_p's A order {b0[0], b1[0], b0[1], b1[1]}.
__device__ __forceinline__ void decode_a8(uint32_t w, uint32_t p, uint32_t (&a)[4]) {
    const uint32_t s = __funnelshift_r(w, p, 20), t = w >> 4;
    a[0] = mcg2(__byte_perm(s, 0u, 0x4421), (s >> 4) & 0xffffu);
    a[1] = mcg2(__byte_perm(t, 0u, 0x4421), __byte_perm(w, 0u, 0x4421));
    a[2] = mcg2(s & 0xffffu, w >> 16);
    a[3] = mcg2(__byte_perm(t, 0u, 0x4410), w & 0xffffu);
}
// Any width's tile from the lane's words: decode_lane<K2, CB_MCG>'s values, in A order.
template <int K2>
__device__ __forceinline__ void decode_ag(const uint32_t (&w)[lane_words<K2>()], int lane, uint32_t (&a)[4]) {
    uint32_t s[8];
    lane_states<K2>(w, lane, s);
    a[0] = mcg2(s[0], s[1]);
    a[1] = mcg2(s[4], s[5]);
    a[2] = mcg2(s[2], s[3]);
    a[3] = mcg2(s[6], s[7]);
}

// fwht128 with each cross-lane butterfly one FFMA, fmaf(s, v, o), s = +1 in the lower lane (o + v) and -1 in the upper
// (o - v): s v is exact, so the FFMA rounds o + v or o - v once, as the add does (fwht128 compiles to an FSEL of -v
// and an FADD; after the Python recipe's K12). The same butterflies in the same order, the same bits.
__device__ __forceinline__ void fwht128f(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const float sg = (lane & m) ? -1.f : 1.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = fmaf(sg, v[j], o);
        }
    }
}

// This thread's cp.async pieces of a stage, fixed for an item: member-row pieces (rows past the item's NBA blocks are
// not loaded, rows past its count are zeros) and trellis word pieces.
template <int MATS, int XR, int K2, int NBA>
struct TLoads {
    static constexpr int TW = tile_words<K2>(), PIECES = TW / 4;
    static constexpr int XPIECES = XR * (TBK / 8), XPT = (XPIECES + TT - 1) / TT;
    static constexpr int WPIECES = 16 * TKT * PIECES, WPT = (WPIECES + TT - 1) / TT;
    const half* xs[XPT];
    uint32_t xd[XPT];
    int xn[XPT];
    bool xon[XPT];
    const uint32_t* ws[WPT];
    uint32_t wd[WPT];
    bool won[WPT];
    size_t wstep;
    __device__ __forceinline__ void init(const half* X0, const half* X1, const uint32_t* T0, const uint32_t* T1, const int* rows_sh,
            int K, int NTILES, int cb, int xdiv, uint32_t sbase) {
#pragma unroll
        for (int j = 0; j < XPT; ++j) {
            const int idx = (int)threadIdx.x + j * TT;
            const int m = idx / (TPM * (TBK / 8)), rem = idx % (TPM * (TBK / 8)), i = rem / (TBK / 8), part = rem % (TBK / 8);
            const bool in = idx < XPIECES;
            const int r = in ? rows_sh[i] : -1;
            xs[j] = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : (r / xdiv)) * K + part * 8;
            xn[j] = r >= 0 ? 16 : 0;
            xon[j] = in && i < NBA * 8;
            xd[j] = sbase + (uint32_t)(((m * TPM + i) * TXS + part * 8) * 2);
        }
#pragma unroll
        for (int j = 0; j < WPT; ++j) {
            const int idx = (int)threadIdx.x + j * TT;
            won[j] = idx < WPIECES;
            const int q = won[j] ? idx / (TKT * PIECES) : 0, kk = (idx / PIECES) % TKT, part = idx % PIECES;
            int wm, nt;
            ttile<MATS>(q, cb, wm, nt);
            ws[j] = (wm ? T1 : T0) + ((size_t)kk * NTILES + nt) * TW + part * 4;
            wd[j] = sbase + (uint32_t)(XR * TXS * 2 + ((q * TKT + kk) * TW + part * 4) * 4);
        }
        wstep = (size_t)TKT * NTILES * TW;
    }
    __device__ __forceinline__ void issue(int c, uint32_t off) const {
#pragma unroll
        for (int j = 0; j < XPT; ++j)
            if (xon[j]) cp16s(xd[j] + off, xs[j] + (size_t)c * TBK, xn[j]);
#pragma unroll
        for (int j = 0; j < WPT; ++j)
            if (won[j]) cp16s(wd[j] + off, ws[j] + (size_t)c * wstep, 16);
    }
};

// Main loop: acc[i][nb] = warp tile i x members nb * 8 .. + 7 over K, nb < NBA; XR rows a stage (TPM if X0 == X1, else
// 2 TPM); PS stages (4 fit tsmem_bytes at XR = TPM, 3 at 2 TPM).
template <int MATS, int XR, int K2, int NBA, int PS>
__device__ __forceinline__ void tmain(const half* __restrict__ X0, const half* __restrict__ X1, const uint64_t* __restrict__ tp0,
        const uint64_t* __restrict__ tp1, const int* rows_sh, int e, int K, int NTILES, int cb, int xdiv, unsigned char* smem,
        float (&acc)[TMT][NBA][4]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    constexpr int SB = tstage_halves<XR, K2>() * 2, TB = TW * 4;   // a stage's bytes, a tile's bytes
    static_assert(PS >= 3 && PS * SB <= tsmem_bytes<K2>(), "a stage of lookahead in the kernel's shared buffer");
    static_assert(TKT % 2 == 0, "two A fragment sets: an even number of k tiles a stage");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int mat, nt_unused;
    ttile<MATS>(warp * TMT, cb, mat, nt_unused);
    const int chunks = K / TBK;
    const uint32_t* T0 = reinterpret_cast<const uint32_t*>(tp0[e]);
    const uint32_t* T1 = reinterpret_cast<const uint32_t*>(tp1[e]);
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int j = 0; j < NBA; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
    TLoads<MATS, XR, K2, NBA> ld;
    ld.init(X0, X1, T0, T1, rows_sh, K, NTILES, cb, xdiv, sbase);
#pragma unroll
    for (int s = 0; s < PS - 1; ++s) {
        if (s < chunks) ld.issue(s, (uint32_t)(s * SB));
        cp_commit();
    }
    // ldmatrix rows (members) and columns (k) of this lane; this warp's first tile; the lane's words in a tile
    const uint32_t xlane = (uint32_t)((((XR == TPM ? 0 : mat * TPM) + (lane & 7) + ((lane >> 4) & 1) * 8) * TXS + ((lane >> 3) & 1) * 8) * 2);
    const uint32_t wwarp = (uint32_t)(XR * TXS * 2 + warp * TMT * TKT * TB);
    uint32_t wo[LW];
    {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int i = 0; i < LW; ++i) wo[i] = (uint32_t)(((word + i) % TW) * 4);
    }
    auto decode = [&](uint32_t tile, uint32_t (&a)[4]) {
        uint32_t w[LW];
#pragma unroll
        for (int i = 0; i < LW; ++i) w[i] = lds32(tile + wo[i]);
        if constexpr (K2 == 8) decode_a8(w[1], w[0], a);   // lane_start<8>: the words of lane - 1 and lane
        else decode_ag<K2>(w, lane, a);
    };
    cp_wait<PS - 2>();
    __syncthreads();   // stage 0
    uint32_t A[2][TMT][4];
#pragma unroll
    for (int i = 0; i < TMT; ++i) decode(sbase + wwarp + (uint32_t)(i * TKT * TB), A[0][i]);
    uint32_t sc = sbase, sn = sbase + SB;   // this stage's and the next stage's buffers
    uint32_t fill = (uint32_t)((PS - 1) * SB);   // the buffer the next load goes to (an offset)
    for (int c = 0; c < chunks; ++c) {
        cp_wait<PS - 3>();   // stage c + 1 landed: its first k tile decodes in this stage
        __syncthreads();     // and every warp is past stage c - 1, whose buffer the load below takes
        if (c + PS - 1 < chunks) ld.issue(c + PS - 1, fill);
        cp_commit();
        fill = fill + SB == (uint32_t)(PS * SB) ? 0u : fill + SB;
#pragma unroll
        for (int kk = 0; kk < TKT; ++kk) {
            const int cur = kk & 1, nxt = cur ^ 1;
            const uint32_t src = kk + 1 < TKT ? sc + (uint32_t)((kk + 1) * TB) : sn;   // the next k tile's words
#pragma unroll
            for (int i = 0; i < TMT; ++i) decode(src + wwarp + (uint32_t)(i * TKT * TB), A[nxt][i]);
            const uint32_t xk = sc + xlane + (uint32_t)(kk * 32);
#pragma unroll
            for (int jp = 0; jp < (NBA + 1) / 2; ++jp) {
                const uint32_t xa = xk + (uint32_t)(jp * 16 * TXS * 2);
                if (2 * jp + 1 < NBA) {
                    uint32_t r[4];
                    ldsm4(r, xa);
                    const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
#pragma unroll
                    for (int i = 0; i < TMT; ++i) {
                        mma16816(acc[i][2 * jp], A[cur][i], bl);
                        mma16816(acc[i][2 * jp + 1], A[cur][i], bh);
                    }
                } else {
                    uint32_t r[2];
                    ldsm2(r, xa);
#pragma unroll
                    for (int i = 0; i < TMT; ++i) mma16816(acc[i][2 * jp], A[cur][i], r);
                }
            }
        }
        sc = sn;
        sn = sn + SB == sbase + (uint32_t)(PS * SB) ? sbase : sn + SB;
    }
    cp_wait<0>();
    __syncthreads();   // the stages become the epilogue's sums
}

// acc (members TPH * hf .. + TPH - 1 of the NBA blocks) -> E[member - TPH * hf][col] fp32, row stride ES; col0: first
// tile's column. Blocks past NBA hold no member and are not written (the epilogue reads members below the count).
template <int NBA>
__device__ __forceinline__ void tspill(float* E, int ES, int col0, const float (&acc)[TMT][NBA][4], int hf) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int q = 0; q < TPM / 16; ++q) {
            const int nb = hf * (TPM / 16) + q;
            if (nb < NBA) {
                const int tok = q * 8 + 2 * t, col = col0 + i * 16 + g;
                E[(size_t)tok * ES + col] = acc[i][nb][0];
                E[(size_t)(tok + 1) * ES + col] = acc[i][nb][1];
                E[(size_t)tok * ES + col + 8] = acc[i][nb][2];
                E[(size_t)(tok + 1) * ES + col + 8] = acc[i][nb][3];
            }
        }
}

// The item's 8-member column blocks: the main loop and epilogue compiled for each count (an item has 1 to TPM members).
#define TF_EXL3_PROMPT_NBA(cnt, CALL)            \
    switch (((cnt) + 7) >> 3) {                  \
        case 1: CALL(1); break;                  \
        case 2: CALL(2); break;                  \
        case 3: CALL(3); break;                  \
        case 4: CALL(4); break;                  \
        case 5: CALL(5); break;                  \
        case 6: CALL(6); break;                  \
        case 7: CALL(7); break;                  \
        default: CALL(8); break;                 \
    }

template <int XR, int K2, int NBA>
__device__ __forceinline__ void tgateup_n(const half* __restrict__ Xg, const half* __restrict__ Xu, const uint64_t* __restrict__ Tg,
        const uint64_t* __restrict__ Tu, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode, unsigned char* smem_raw, const int* rows_sh, int cb, int e,
        int cnt) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][NBA][4];
    tmain<2, XR, K2, NBA, XR == TPM ? 4 : 3>(Xg, Xu, Tg, Tu, rows_sh, e, K, N >> 4, cb, xdiv, smem_raw, acc);
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    ttile<2>(warp * TMT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();   // the first half's rows are read
        tspill<NBA>(Eg + (size_t)mat * TPH * GU_ES, GU_ES, col0, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
                uv[j] = Eg[(size_t)(TPH + i) * GU_ES + 4 * lane + j];
            }
            fwht128f(gv, lane);
            fwht128f(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]);
                float uu = uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]);
                if (act_mode == 0) { gg = bf16r(gg); uu = bf16r(uu); }
                if (limit > 0) { gg = upTo(gg, limit); uu = within(uu, limit); }
                float act = gg / (1.f + expf(-gg));
                if (act_mode == 0) act = bf16r(act);
                act *= uu;
                if (act_mode == 0) act = bf16r(act);
                v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
            }
            fwht128f(v, lane);
            // the lane's 4 halves in one 8-byte store (xd rows and n are 8-byte aligned: Kernels.run checks xd)
            const half2 o01 = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
            const half2 o23 = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
            uint2 packed;
            packed.x = *reinterpret_cast<const uint32_t*>(&o01);
            packed.y = *reinterpret_cast<const uint32_t*>(&o23);
            *reinterpret_cast<uint2*>(xd + (size_t)p * N + n) = packed;
        }
    }
}

template <int XR, int K2>
__device__ __forceinline__ void tgateup(const half* __restrict__ Xg, const half* __restrict__ Xu, const uint64_t* __restrict__ Tg,
        const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
        const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d, half* __restrict__ xd,
        int K, int N, int xdiv, float limit, int act_mode, unsigned char* smem_raw, int* rows_sh) {
    const int cb = blockIdx.x, item = blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
#define TF_EXL3_PROMPT_GU(NBA) \
    tgateup_n<XR, K2, NBA>(Xg, Xu, Tg, Tu, svh_g, svh_u, suh_d, xd, K, N, xdiv, limit, act_mode, smem_raw, rows_sh, cb, e, cnt)
    TF_EXL3_PROMPT_NBA(cnt, TF_EXL3_PROMPT_GU)
#undef TF_EXL3_PROMPT_GU
}

}  // namespace tf_exl3_prompt
