// The grouped EXL3 experts' projection (tf_exl3_grouped_*), shared with the contract's baseline and planted faults.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#include "exl3_decode.cuh"

namespace tf_exl3 {

// One projection's grouped arguments; the fields must stay in launch.zig add order.
struct GroupedArgs {
    const half* x0; // fp16 [P, K], the rotated gate (or down) inputs
    const half* x1; // fp16 [P, K], the rotated up inputs
    const long long* tp0; // int64 [E], one trellis pointer per expert
    const long long* tp1;
    const int* k2_0; // int32 [E], half-bits a value
    const int* k2_1;
    const int* uids; // int32 [nexp_max], distinct routed experts in id order
    const int* ucount; // int32 [1]
    const int* members; // int32 [nexp_max, maxm], row * 32 + slot, -1 after the last member
    float* z; // fp32 partials, [(mat * sk + split) * P + p] * N + col
    int k, n, p, sk, maxm, slots, nexp_max, mats, ns, standard, skip;
};

// How a warp reads its k tiles: main's loop, or a tile ahead (sm_120 up); SKIP and STALE are the contract's faults;
// SHFL every word once, a tile ahead (warp_tiles_shfl, sm_120 up); SHFL_SRC and SHFL_A its planted faults.
enum Load : int { PLAIN = 0, AHEAD = 1, SKIP = 2, STALE = 3, SHFL = 4, SHFL_SRC = 5, SHFL_A = 6, SHFL_PDL = 7 };

// Dependent launches (sm_90 up; a no-op in a kernel not launched as a dependent): wait for the grids before this one
// (their outputs visible after it), and let the next dependent launch begin.
__device__ __forceinline__ void pdl_wait() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// One warp's k tiles [kt0, kt0 + nkt) of an expert matrix into acc: per-lane loads and the shared decode (same bits).
template <int CB, int K2, int NT, int LOAD>
__device__ __forceinline__ void warp_tiles(const uint32_t* T, int ntiles, int kt0, int nkt, int nt0, const half* x0,
                                           const half* x1, bool ok0, bool ok1, int lane, float (&acc)[NT][2][4]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    const size_t kstride = (size_t)ntiles * TW;
    const uint32_t* tb = T + (size_t)kt0 * kstride + (size_t)nt0 * TW;
    uint32_t nx[NT][LW];  // the next tile's words, in flight while this one decodes
    if (LOAD != PLAIN && nkt > 0)
#pragma unroll
        for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(tb + j * TW, lane, nx[j]);
    for (int i = 0; i < nkt; ++i) {
        uint32_t w[NT][LW];
        if constexpr (LOAD == PLAIN) {
            const uint32_t* t0 = T + (size_t)(kt0 + i) * kstride + (size_t)nt0 * TW;
#pragma unroll
            for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(t0 + j * TW, lane, w[j]);
        } else {
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int l = 0; l < LW; ++l) w[j][l] = nx[j][l];
            const int next = LOAD == SKIP ? (i + 2) % nkt : i + 1;  // SKIP: tile i + 2, in the warp's range
            const int last = LOAD == STALE ? nkt - 1 : nkt;  // STALE: the last tile decodes the one before's words
            if (i + 1 < last)
#pragma unroll
                for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(tb + (size_t)next * kstride + j * TW, lane, nx[j]);
        }
        const int k = (kt0 + i) * 16;
        const uint32_t a[4] = {ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u,
                               ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u,
                               ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u,
                               ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u};
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            uint32_t b0[2], b1[2];
            decode_lane<K2, CB>(w[j], lane, b0, b1);
            mma16816(acc[j][0], a, b0);
            mma16816(acc[j][1], a, b1);
        }
    }
}

// lane_states<8>'s values (4-bit: windows at fixed nibble offsets): V = w0:w1 (w0 high), s[j] = (V >> (28 - 4 j)) & 0xffff.
__device__ __forceinline__ void lane_states8(uint32_t w0, uint32_t w1, uint32_t (&s)[8]) {
    s[0] = __funnelshift_r(w1, w0, 28) & 0xffffu;
    s[1] = __funnelshift_r(w1, w0, 24) & 0xffffu;
    s[2] = __funnelshift_r(w1, w0, 20) & 0xffffu;
    s[3] = w1 >> 16;
    s[4] = (w1 >> 12) & 0xffffu;
    s[5] = __byte_perm(w1, 0u, 0x4421);
    s[6] = (w1 >> 4) & 0xffffu;
    s[7] = w1 & 0xffffu;
}

// decode2<CB_MCG> with its AND mask in a register `m` (0x8FFF8FFF): the same operations on the same values.
__device__ __forceinline__ uint32_t decode2_mcg(uint32_t s0, uint32_t s1, uint32_t m) {
    uint32_t x0 = s0 * 0xCBAC1FEDu, x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & m) ^ 0x3B603B60u;
    x1 = (x1 & m) ^ 0x3B603B60u;
    const uint32_t lo = __byte_perm(x0, x1, 0x5410);
    const uint32_t hi = __byte_perm(x0, x1, 0x7632);
    const half2 r = __hadd2(*reinterpret_cast<const half2*>(&lo), *reinterpret_cast<const half2*>(&hi));
    return *reinterpret_cast<const uint32_t*>(&r);
}

// warp_tiles with every trellis word read once: lane l loads word l of each tile (one 128-byte warp load a 4-bit tile,
// where the lane-window loads read each word twice) and the next tile's words and A fragment a tile ahead; each lane
// takes its windows' words from their owners by shuffle. The same words, A values, decode and MMA chain as warp_tiles,
// so the same bits. Planted faults (contract): SHFL_SRC takes the last tile's second word from the wrong lane, SHFL_A
// gives the last tile the A fragment of the tile before.
// 4-bit mcg decodes with lane_states8 and decode2_mcg; `opq` (any value below 256, the call's matrix count) builds the
// mask so that ptxas keeps it in a register rather than rebuilding it for every LOP3 (26 MOVs a k step).
template <int CB, int K2, int NT, int LOAD>
__device__ __forceinline__ void warp_tiles_shfl(const uint32_t* T, int ntiles, int kt0, int nkt, int nt0, const half* x0,
                                                const half* x1, bool ok0, bool ok1, int lane, float (&acc)[NT][2][4],
                                                int opq) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    static_assert(TW <= 32, "a word a lane");
    const size_t kstride = (size_t)ntiles * TW;
    const uint32_t* tb = T + (size_t)kt0 * kstride + (size_t)nt0 * TW;
    int word, offset;
    lane_start<K2>(lane, word, offset);
    const bool own = TW == 32 || lane < TW;
    const uint32_t mask = 0x8FFF8FFFu | ((uint32_t)opq >> 8);  // exactly 0x8FFF8FFF (opq < 256), held in a register
    uint32_t nx[NT], na[4];  // the next tile's word `lane` of each tile and its A fragment, in flight while this decodes
    auto words = [&](int i) {  // the trellis: immutable, so it may stream in before a dependent launch's wait
        const uint32_t* t0 = tb + (size_t)i * kstride;
#pragma unroll
        for (int j = 0; j < NT; ++j) nx[j] = own ? ldg_stream_u32(t0 + j * TW + lane) : 0u;
    };
    auto rows = [&](int i) {  // the rows' inputs: the grid before this one's outputs
        const int k = (kt0 + i) * 16;
        na[0] = ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u;
        na[1] = ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u;
        na[2] = ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u;
        na[3] = ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u;
    };
    auto load = [&](int i) {
        words(i);
        rows(i);
    };
    if (nkt > 0) words(0);
    if constexpr (LOAD == SHFL_PDL) {  // a dependent launch: every input of this step and every write after the wait
        pdl_wait();
        pdl_trigger();
    }
    if (nkt > 0) rows(0);
    for (int i = 0; i < nkt; ++i) {
        uint32_t cur[NT], a[4];
#pragma unroll
        for (int j = 0; j < NT; ++j) cur[j] = nx[j];
#pragma unroll
        for (int c = 0; c < 4; ++c) a[c] = na[c];
        if (LOAD == SHFL_A && i == nkt - 1 && i > 0) {  // planted: the last tile keeps the tile before's A fragment
            const int k = (kt0 + i - 1) * 16;
            a[0] = ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u;
            a[2] = ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u;
        }
        if (i + 1 < nkt) load(i + 1);
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            uint32_t w[LW];
#pragma unroll
            for (int l = 0; l < LW; ++l) {
                const int src = (LOAD == SHFL_SRC && i == nkt - 1 && l == LW - 1 ? word + l + 1 : word + l) % TW;
                w[l] = __shfl_sync(0xffffffffu, cur[j], src);
            }
            uint32_t b0[2], b1[2];
            if constexpr (K2 == 8 && CB == CB_MCG) {
                uint32_t st[8];
                lane_states8(w[0], w[1], st);
                b0[0] = decode2_mcg(st[0], st[1], mask);
                b0[1] = decode2_mcg(st[2], st[3], mask);
                b1[0] = decode2_mcg(st[4], st[5], mask);
                b1[1] = decode2_mcg(st[6], st[7], mask);
            } else
                decode_lane<K2, CB>(w, lane, b0, b1);
            mma16816(acc[j][0], a, b0);
            mma16816(acc[j][1], a, b1);
        }
    }
}

// Program (expert u, n block, split, member tile): 16 members times W_q over the split's K range, summed in warp order.
template <int CB, int NT, int W, int LOAD>
__device__ __forceinline__ void grouped_body(const GroupedArgs a, int u, int nblock, int zz) {
    if (u >= a.ucount[0]) return;
    const int MT = (a.maxm + 15) / 16;
    const int mtile = zz % MT;
    const int split = (zz / MT) % a.sk;
    const int mat = zz / MT / a.sk;
    const half* X = mat ? a.x1 : a.x0;
    const int e = a.standard ? a.uids[3 * u] : a.uids[u];
    if (a.standard && e == a.skip) return;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? a.tp1[e] : a.tp0[e]);
    const int k2 = mat ? a.k2_1[e] : a.k2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int NTILES = a.ns > 0 ? a.ns : (a.n >> 4); // row stride of the trellis, in tiles

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int count = a.standard ? a.uids[3 * u + 2] : a.maxm;
        const int first = a.standard ? a.uids[3 * u + 1] : u * a.maxm;
        const int code = m < count ? a.members[first + m] : -1;
        rows_sh[threadIdx.x] = a.standard ? code : (code >= 0 ? (code >> 5) * a.slots + (code & 31) : -1);
    }
    __syncthreads();
    if (rows_sh[0] < 0) return; // members come first, so this tile is empty
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * a.k + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * a.k + 2 * t;
    const int per_warp = (a.k >> 4) / a.sk / W;
    const int kt0 = split * (per_warp * W) + warp * per_warp;
    const int nt0 = nblock * NT;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    __shared__ __align__(16) float red[W][16][NT * 16];
    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                   \
    case K2_:                                                                                                \
        warp_tiles<CB, K2_, NT, LOAD>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc);          \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }

#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        a.z[(((size_t)mat * a.sk + split) * a.p + r) * a.n + nt0 * 16 + col] = s;
    }
}

// grouped_body at one trellis width K2F (no width switch: fewer registers) and RR reduction rows (a member tile of at
// most RR rows: decode windows); every column keeps its k slices and warp-order sum, so the bits are grouped_body's.
// SHFL_PDL (a dependent launch): before its wait a block reads only the plan (uids, count, members: from the router two
// launches back, done before the grid before this one passed its own wait) and the immutable trellis pointers, widths
// and first tile; every exit waits first, so every CTA waits.
template <int CB, int NT, int W, int LOAD, int K2F, int RR>
__device__ __forceinline__ void grouped_body_v2(const GroupedArgs a, int u, int nblock, int zz) {
    constexpr bool PDLW = LOAD == SHFL_PDL;
    if (u >= a.ucount[0]) {
        if constexpr (PDLW) pdl_wait();
        return;
    }
    const int MT = (a.maxm + 15) / 16;
    const int mtile = zz % MT;
    const int split = (zz / MT) % a.sk;
    const int mat = zz / MT / a.sk;
    const half* X = mat ? a.x1 : a.x0;
    const int e = a.standard ? a.uids[3 * u] : a.uids[u];
    if (a.standard && e == a.skip) {
        if constexpr (PDLW) pdl_wait();
        return;
    }
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? a.tp1[e] : a.tp0[e]);
    const int k2 = mat ? a.k2_1[e] : a.k2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int NTILES = a.ns > 0 ? a.ns : (a.n >> 4); // row stride of the trellis, in tiles

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int count = a.standard ? a.uids[3 * u + 2] : a.maxm;
        const int first = a.standard ? a.uids[3 * u + 1] : u * a.maxm;
        const int code = m < count ? a.members[first + m] : -1;
        rows_sh[threadIdx.x] = a.standard ? code : (code >= 0 ? (code >> 5) * a.slots + (code & 31) : -1);
    }
    __syncthreads();
    if (rows_sh[0] < 0) { // members come first, so this tile is empty
        if constexpr (PDLW) pdl_wait();
        return;
    }
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * a.k + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * a.k + 2 * t;
    const int per_warp = (a.k >> 4) / a.sk / W;
    const int kt0 = split * (per_warp * W) + warp * per_warp;
    const int nt0 = nblock * NT;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    __shared__ __align__(16) float red[W][RR][NT * 16];
    if constexpr (K2F > 0) {
        if (k2 != K2F) __trap();  // the launcher picks it for a projection whose every expert is K2F wide
        if constexpr (LOAD >= SHFL)
            warp_tiles_shfl<CB, K2F, NT, LOAD>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc, a.mats);
        else
            warp_tiles<CB, K2F, NT, LOAD>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc);
    } else {
    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                   \
    case K2_:                                                                                                \
        warp_tiles<CB, K2_, NT, LOAD>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc);          \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }
    }

#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            if (g < RR) {
                red[warp][g][col] = acc[i][h][0];
                red[warp][g][col + 1] = acc[i][h][1];
            }
            if (g + 8 < RR) {
                red[warp][g + 8][col] = acc[i][h][2];
                red[warp][g + 8][col + 1] = acc[i][h][3];
            }
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < RR * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        a.z[(((size_t)mat * a.sk + split) * a.p + r) * a.n + nt0 * 16 + col] = s;
    }
}

}  // namespace tf_exl3
