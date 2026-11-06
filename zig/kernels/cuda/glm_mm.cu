// GLM dense linears, ours: one tensor-core kernel for any row count, 64-groups then K slices in order; row-stable bits.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "common/qmm_frag.cuh"

namespace {

using namespace qmm_frag;

// 16 codes from a packed chunk (ops.cu's layout), stored + 2^(BITS-1)
template <int BITS>
__device__ __forceinline__ void codes16(const uint32_t* __restrict__ p, int (&q)[16]) {
    if constexpr (BITS == 4) {
#pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = (int)((p[i / 8] >> (4 * (i % 8))) & 15u) - 8;
    } else if constexpr (BITS == 8) {
#pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = (int)((p[i / 4] >> (8 * (i % 4))) & 255u) - 128;
    } else {
#pragma unroll
        for (int i = 0; i < 15; ++i) q[i] = (int)((p[i / 5] >> (6 * (i % 5))) & 63u) - 32;
        q[15] = (int)((p[0] >> 30) | ((p[1] >> 30) << 2) | ((p[2] >> 30) << 4)) - 32;
    }
}

template <int BITS>
__device__ __forceinline__ void pack16(const int (&q)[16], uint32_t* __restrict__ out) {
    constexpr int half = 1 << (BITS - 1);
    uint32_t w[BITS / 2];
#pragma unroll
    for (int i = 0; i < BITS / 2; ++i) w[i] = 0u;
    if constexpr (BITS == 4) {
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i / 8] |= (uint32_t)(q[i] + half) << (4 * (i % 8));
    } else if constexpr (BITS == 8) {
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i / 4] |= (uint32_t)(q[i] + half) << (8 * (i % 4));
    } else {
#pragma unroll
        for (int i = 0; i < 15; ++i) w[i / 5] |= (uint32_t)(q[i] + half) << (6 * (i % 5));
        const uint32_t last = (uint32_t)(q[15] + half);
#pragma unroll
        for (int j = 0; j < 3; ++j) w[j] |= ((last >> (2 * j)) & 3u) << 30;
    }
#pragma unroll
    for (int i = 0; i < BITS / 2; ++i) out[i] = w[i];
}

// input of value v (0..15) of lane ``lane``'s chunk within its group of 64
__device__ __forceinline__ int frag_k(int lane, int v) {
    const int t = lane & 3, kt = v >> 2, j = v & 3;
    return kt * 16 + 2 * t + (j & 1) + (j >> 1) * 8;
}

// (t & 0x007F007F) | c and (t & 0x00800080) ^ c, one LOP3 each with c in a register (two immediates in one LOP3
// would cost a MOV a use)
__device__ __forceinline__ uint32_t low7_or(uint32_t t, uint32_t c) {
    uint32_t y;
    asm("lop3.b32 %0, %1, 0x007F007F, %2, 0xea;\n" : "=r"(y) : "r"(t), "r"(c));
    return y;
}
__device__ __forceinline__ uint32_t top_xor(uint32_t t, uint32_t c) {
    uint32_t y;
    asm("lop3.b32 %0, %1, 0x00800080, %2, 0x6a;\n" : "=r"(y) : "r"(t), "r"(c));
    return y;
}

// the B-fragment bf16 pairs of a chunk: pairs 2 kt and 2 kt + 1 are step kt's b0 and b1
template <int BITS>
__device__ __forceinline__ void bpairs(const uint32_t* __restrict__ p, uint32_t (&b)[8]) {
    if constexpr (BITS == 16) {
#pragma unroll
        for (int i = 0; i < 8; ++i) b[i] = p[i];
    } else {
        int q[16];
        codes16<BITS>(p, q);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const __nv_bfloat162 h = __floats2bfloat162_rn((float)q[2 * i], (float)q[2 * i + 1]);   // exact integers
            b[i] = *reinterpret_cast<const uint32_t*>(&h);
        }
    }
}

// bpairs<8>'s pairs with fewer instructions (the packed prompt kernels): a byte pair of codes u (stored q + 128) spread
// into the halves' low bytes; a = 128 + (u & 127) (0x4300 | the low 7 bits), c = 256, or 128 when u >= 128 (0x4380 with
// its lowest exponent bit cleared), and a - c = u - 128 exactly: both and the difference are integers bf16 holds, so
// the bf16 subtraction rounds nothing and gives bf16((float)(u - 128)), +0 at u = 128 as (float)0 does.
__device__ __forceinline__ void bpairs8_fast(const uint32_t* __restrict__ p, uint32_t (&b)[8]) {
    const uint32_t k43 = 0x43004300u, k438 = 0x43804380u;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t t = __byte_perm(p[i >> 1], 0u, (i & 1) ? 0x4342 : 0x4140);
        const uint32_t a = low7_or(t, k43), c = top_xor(t, k438);
        const __nv_bfloat162 d = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&a), *reinterpret_cast<const __nv_bfloat162*>(&c));
        b[i] = *reinterpret_cast<const uint32_t*>(&d);
    }
}

template <int BITS, int BM, int WM, int STAGES>
struct Tile {
    static constexpr int WN = 4, BN = 64, NT = 2, MT = BM / WM / 16, THREADS = WM * WN * 32;
    static constexpr int WPC = BITS / 2;
    static constexpr int ROW = 128, CHUNKS = 8;               // a row's 64 inputs: 128 bytes, 8 chunks of 16
    static constexpr int X = BM * ROW;
    static constexpr int W = 8 * 32 * WPC * 4;                // 8 column tiles' chunks
    static constexpr int S = 8 * 8 * 2;                       // their scales (fp16)
    static constexpr int STAGE = X + W + S;
    static constexpr int SMEM = STAGES * STAGE;
};

// ROWS_FIRST: row blocks fastest, so the blocks reading one weight tile run together and share it in L2 (prompts)
// P2 (the packed prompt kernels): each thread's cp.async sources, destinations and bounds worked out once a block (a
// stage adds its group's offset), and bpairs8_fast; the loads, MMAs and FMAs stay those of the other instances.
template <int BITS, int BM, int WM, int STAGES, bool ROWS_FIRST = false, bool P2 = false>
__device__ __forceinline__ void mm_kernel(const __nv_bfloat16* __restrict__ x, long long ldx,
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part,
        int f32, int M, int N, int K, int SK, int zslice = -1, int xtile = -1) {
    using T = Tile<BITS, BM, WM, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / T::WN, wn = warp % T::WN;
    const int KG = K >> 6, per = KG / SK, N8 = N >> 3;
    // a two-problem launch passes the block's column tile (xtile) and slice (zslice) of its problem
    const int n0 = (xtile >= 0 ? (unsigned)xtile : (ROWS_FIRST ? blockIdx.y : blockIdx.x)) * T::BN,
              m0 = (ROWS_FIRST ? blockIdx.x : blockIdx.y) * BM;
    const int slice = zslice >= 0 ? zslice : (int)blockIdx.z, g0 = slice * per;  // a pair launch's z is its problem
    const int t0 = n0 >> 3;                                   // the block's first column tile
    auto stage = [&](int st) { return buf + st * T::STAGE; };
    auto loadX = [&](int st, int g) {
        unsigned char* p = stage(st);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + (long long)row * ldx + g * 64 + ch * 8, m0 + r < M);
        }
    };
    auto loadW = [&](int st, int g) {
        unsigned char* pw = stage(st) + T::X;
        constexpr int PER_TILE = 32 * T::WPC * 4 / 16;        // 16-byte pieces of one column tile's chunks
        for (int c = tid; c < 8 * PER_TILE; c += T::THREADS) {
            const int tt = c / PER_TILE, off = c % PER_TILE;
            if (t0 + tt >= N8) continue;
            const long long src = ((long long)(t0 + tt) * KG + g) * 32 * T::WPC * 4 + off * 16;
            cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + src);
        }
        if constexpr (BITS != 16) {
            unsigned char* ps = pw + T::W;
            for (int tt = tid; tt < 8; tt += T::THREADS) {
                if (t0 + tt >= N8) continue;
                cp16(ps + tt * 16, s + ((long long)(t0 + tt) * KG + g) * 8);
            }
        }
    };
    // P2: the same pieces as loadX / loadW, sources at group g0, a stage adds (g - g0) groups
    constexpr int XPT = (BM * T::CHUNKS + T::THREADS - 1) / T::THREADS;
    constexpr int PER_TILE_ = 32 * T::WPC * 4 / 16, WPT = (8 * PER_TILE_ + T::THREADS - 1) / T::THREADS;
    const unsigned char* xsrc[P2 ? XPT : 1];
    uint32_t xdst[P2 ? XPT : 1];
    bool xin[P2 ? XPT : 1], xok[P2 ? XPT : 1];
    const unsigned char* wsrc[P2 ? WPT : 1];
    bool wok[P2 ? WPT : 1];
    const unsigned char* ssrc = nullptr;
    bool sok = false;
    if constexpr (P2) {
#pragma unroll
        for (int j = 0; j < XPT; ++j) {
            const int c = tid + j * T::THREADS, r = c / T::CHUNKS, ch = c % T::CHUNKS;
            xin[j] = c < BM * T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            xsrc[j] = reinterpret_cast<const unsigned char*>(x + (long long)row * ldx + (long long)g0 * 64 + ch * 8);
            xdst[j] = (uint32_t)(r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            xok[j] = m0 + r < M;
        }
#pragma unroll
        for (int j = 0; j < WPT; ++j) {
            const int c = tid + j * T::THREADS, tt = c / PER_TILE_, off = c % PER_TILE_;
            wok[j] = c < 8 * PER_TILE_ && t0 + tt < N8;
            wsrc[j] = reinterpret_cast<const unsigned char*>(w) + (wok[j] ? ((long long)(t0 + tt) * KG + g0) * 32 * T::WPC * 4 + off * 16 : 0);
        }
        if constexpr (BITS != 16) {
            sok = tid < 8 && t0 + tid < N8;
            ssrc = reinterpret_cast<const unsigned char*>(s + (sok ? ((long long)(t0 + tid) * KG + g0) * 8 : 0));
        }
    }
    auto loadPX = [&](int st, int g) {   // P2: loadX of group g into stage st
        unsigned char* p = stage(st);
        const long long dg = g - g0;
#pragma unroll
        for (int j = 0; j < XPT; ++j)
            if (xin[j]) cp16z(p + xdst[j], xsrc[j] + dg * 128, xok[j]);
    };
    auto loadPW = [&](int st, int g) {   // P2: loadW of group g into stage st
        unsigned char* pw = stage(st) + T::X;
        const long long dg = g - g0;
#pragma unroll
        for (int j = 0; j < WPT; ++j)
            if (wok[j]) cp16(pw + (tid + j * T::THREADS) * 16, wsrc[j] + dg * (32 * T::WPC * 4));
        if constexpr (BITS != 16)
            if (sok) cp16(pw + T::W + tid * 16, ssrc + dg * 16);
    };
    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    // dependent launch: first stages' weights stream in early; inputs and outputs only after griddepcontrol.wait
    grid_launch();
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st)
        if (st < per) {
            if constexpr (P2) loadPW(st, g0 + st);
            else loadW(st, g0 + st);
        }
    grid_wait();
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < per) {
            if constexpr (P2) loadPX(st, g0 + st);
            else loadX(st, g0 + st);
        }
        commit();
    }
    for (int it = 0; it < per; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) {
            if constexpr (P2) {
                loadPX(next % STAGES, g0 + next);
                loadPW(next % STAGES, g0 + next);
            } else {
                loadX(next % STAGES, g0 + next);
                loadW(next % STAGES, g0 + next);
            }
        }
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const __half* ps = reinterpret_cast<const __half*>(p + T::X + T::W);
        uint32_t b[T::NT][8];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            if constexpr (P2 && BITS == 8) bpairs8_fast(pw + ((wn * T::NT + j) * 32 + lane) * T::WPC, b[j]);
            else bpairs<BITS>(pw + ((wn * T::NT + j) * 32 + lane) * T::WPC, b[j]);
        }
        float d[T::MT][T::NT][4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    if (kt == 0) mma0(d[i][j], a[i], b[j][0], b[j][1]);
                    else mma(d[i][j], a[i], b[j][2 * kt], b[j][2 * kt + 1]);
                }
        }
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            float s0 = 1.0f, s1 = 1.0f;
            if constexpr (BITS != 16) {
                const __half2 h = *reinterpret_cast<const __half2*>(ps + (wn * T::NT + j) * 8 + (lane & 3) * 2);
                s0 = __low2float(h);
                s1 = __high2float(h);
            }
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = __fmaf_rn(d[i][j][e], (e & 1) ? s1 : s0, acc[i][j][e]);
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + (wn * T::NT + j) * 8 + (lane & 3) * 2;
            if (col >= N) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                if (SK > 1) {
                    float* dst = part + ((long long)slice * M + row) * N + col;
                    dst[0] = v0;
                    dst[1] = v1;
                } else if (f32) {
                    float* dst = reinterpret_cast<float*>(out) + (long long)row * N + col;
                    dst[0] = v0;
                    dst[1] = v1;
                } else {
                    *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + (long long)row * N + col) =
                        __floats2bfloat162_rn(v0, v1);
                }
            }
        }
}

}  // namespace

// The K slices' partials added in slice order, then rounded once. Dependent launch: it may start while the matmul
// that writes `part` runs, so every thread waits for that grid before it reads `part` or writes `out` (or exits).
extern "C" __global__ void glm_mm_reduce(const float* __restrict__ part, void* __restrict__ out, long long total, int SK, int f32) {
    qmm_frag::grid_launch();
    qmm_frag::grid_wait();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = part[i];
    for (int sl = 1; sl < SK; ++sl) acc = acc + part[sl * total + i];
    if (f32) reinterpret_cast<float*>(out)[i] = acc;
    else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
}

// one instance a (bits, row bucket): (16, 1 warp row), (32, 1), (64, 2), (128, 2); three stages
#define GLM_MM_ALL(B)                                                                                                  \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_16(const __nv_bfloat16* __restrict__ x, long long ldx, \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, \
        int f32, int M, int N, int K, int SK) { mm_kernel<B, 16, 1, 3>(x, ldx, w, s, out, part, f32, M, N, K, SK); }     \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_32(const __nv_bfloat16* __restrict__ x, long long ldx, \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, \
        int f32, int M, int N, int K, int SK) { mm_kernel<B, 32, 1, 3>(x, ldx, w, s, out, part, f32, M, N, K, SK); }     \
    extern "C" __global__ void __launch_bounds__(256) glm_mm##B##_64(const __nv_bfloat16* __restrict__ x, long long ldx, \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, \
        int f32, int M, int N, int K, int SK) { mm_kernel<B, 64, 2, 3>(x, ldx, w, s, out, part, f32, M, N, K, SK); }     \
    extern "C" __global__ void __launch_bounds__(256) glm_mm##B##_128(const __nv_bfloat16* __restrict__ x, long long ldx, \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, \
        int f32, int M, int N, int K, int SK) { mm_kernel<B, 128, 2, 3>(x, ldx, w, s, out, part, f32, M, N, K, SK); }
// prompts: 128-row blocks, row blocks fastest (one slice: a prompt's rows fill the GPU)
extern "C" __global__ void __launch_bounds__(256) glm_mm16_rows(const __nv_bfloat16* __restrict__ x, long long ldx,
    const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part,
    int f32, int M, int N, int K, int SK) { mm_kernel<16, 128, 2, 3, true>(x, ldx, w, s, out, part, f32, M, N, K, SK); }
GLM_MM_ALL(4)
GLM_MM_ALL(6)
GLM_MM_ALL(8)
GLM_MM_ALL(16)
// Packed prompts, two stages (41 KB: two blocks an SM instead of one). Each output's K groups run in the same order, so
// the sums are glm_mm{B}_128's bit for bit; a 768-row piece's 384 blocks fit one wave of 376 slots instead of 2.04.
#define GLM_MM_PROMPT2(B)                                                                                              \
    extern "C" __global__ void __launch_bounds__(256, 2) glm_mm##B##_128s2(const __nv_bfloat16* __restrict__ x,       \
        long long ldx, const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out,          \
        float* __restrict__ part, int f32, int M, int N, int K, int SK) { mm_kernel<B, 128, 2, 2, false, true>(x, ldx, w, s, out, part, f32, M, N, K, SK); }
GLM_MM_PROMPT2(4)
GLM_MM_PROMPT2(6)
GLM_MM_PROMPT2(8)

// Two problems of one shape and one K slice in one launch (z picks the problem): the same blocks and sums as two
// launches of glm_mm*_16 (GLM's KDA f_b and g_b projections of a decode window).
struct MmPair {
    const __nv_bfloat16* x[2];
    long long ldx;
    const uint32_t* w[2];
    const __half* s[2];
    void* out[2];
    int f32, M, N, K;
};
#define GLM_MM_PAIR(B)                                                                                                  \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_16x2(const MmPair p) {                              \
        const int z = blockIdx.z;                                                                                      \
        mm_kernel<B, 16, 1, 3>(p.x[z], p.ldx, p.w[z], p.s[z], p.out[z], nullptr, p.f32, p.M, p.N, p.K, 1, 0);           \
    }
GLM_MM_PAIR(4)
GLM_MM_PAIR(6)
GLM_MM_PAIR(8)
GLM_MM_PAIR(16)

// Two decode problems of one width in one launch, each with its own shape and its own K slices: x runs problem 0's
// column tiles times its slices, then problem 1's, and each block is the block a lone glm_mm{B}_{BM} launch runs for
// that (column tile, slice), so partials and outputs are two launches' bit for bit (GLM's DSA [q_a | kv_a] with the
// indexer's kw on the layer's rows, and q_b with the indexer's qb on the q rows).
struct MmTwo {
    const __nv_bfloat16* x[2];
    long long ldx[2];
    const uint32_t* w[2];
    const __half* s[2];
    void* out[2];
    float* part[2];
    int N[2], K[2], SK[2], cols[2];
    int blocks0, f32, M;
};
template <int B, int BM, int WM>
__device__ __forceinline__ void mm_two(const MmTwo& p) {
    const int b = (int)blockIdx.x, z = b >= p.blocks0 ? 1 : 0, l = b - (z ? p.blocks0 : 0), c = p.cols[z];
    mm_kernel<B, BM, WM, 3>(p.x[z], p.ldx[z], p.w[z], p.s[z], p.out[z], p.part[z], p.f32, p.M, p.N[z], p.K[z], p.SK[z],
                            l / c, l % c);
}
#define GLM_MM_TWO(B)                                                                                                   \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_16t(const MmTwo p) { mm_two<B, 16, 1>(p); }          \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_32t(const MmTwo p) { mm_two<B, 32, 1>(p); }          \
    extern "C" __global__ void __launch_bounds__(256) glm_mm##B##_64t(const MmTwo p) { mm_two<B, 64, 2>(p); }
GLM_MM_TWO(4)
GLM_MM_TWO(6)
GLM_MM_TWO(8)
GLM_MM_TWO(16)

// Decode windows (16 and 32 rows a block) with a deeper cp.async pipeline (TF_GLM_MM_STAGES=5, 6 or 8; 3 is the kernels
// above): a block keeps more weight bytes in flight on a GPU whose memory latency is long for its bandwidth (GB10).
// Each K group's loads, MMAs and FMAs are those of the three-stage kernel in the same order, so every output is the
// same bits; only how many groups ahead the stages are fetched changes. One instance a (bits, rows, stages).
template <int B, int BM, int WM, int S>
__device__ __forceinline__ void mm_two_s(const MmTwo& p) {
    const int b = (int)blockIdx.x, z = b >= p.blocks0 ? 1 : 0, l = b - (z ? p.blocks0 : 0), c = p.cols[z];
    mm_kernel<B, BM, WM, S>(p.x[z], p.ldx[z], p.w[z], p.s[z], p.out[z], p.part[z], p.f32, p.M, p.N[z], p.K[z], p.SK[z],
                            l / c, l % c);
}
#define GLM_MM_DECODE_STAGES(B, S)                                                                                      \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_16s##S(const __nv_bfloat16* __restrict__ x,         \
        long long ldx, const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out,          \
        float* __restrict__ part, int f32, int M, int N, int K, int SK) {                                              \
        mm_kernel<B, 16, 1, S>(x, ldx, w, s, out, part, f32, M, N, K, SK);                                             \
    }                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_32s##S(const __nv_bfloat16* __restrict__ x,         \
        long long ldx, const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out,          \
        float* __restrict__ part, int f32, int M, int N, int K, int SK) {                                              \
        mm_kernel<B, 32, 1, S>(x, ldx, w, s, out, part, f32, M, N, K, SK);                                             \
    }                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_16x2s##S(const MmPair p) {                          \
        const int z = blockIdx.z;                                                                                      \
        mm_kernel<B, 16, 1, S>(p.x[z], p.ldx, p.w[z], p.s[z], p.out[z], nullptr, p.f32, p.M, p.N, p.K, 1, 0);           \
    }                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_16ts##S(const MmTwo p) { mm_two_s<B, 16, 1, S>(p); } \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_32ts##S(const MmTwo p) { mm_two_s<B, 32, 1, S>(p); }
#define GLM_MM_DECODE_ALL(B) GLM_MM_DECODE_STAGES(B, 5) GLM_MM_DECODE_STAGES(B, 6) GLM_MM_DECODE_STAGES(B, 8)
GLM_MM_DECODE_ALL(4)
GLM_MM_DECODE_ALL(6)
GLM_MM_DECODE_ALL(8)
GLM_MM_DECODE_ALL(16)

// Both problems' K slices added in slice order, then rounded once: glm_mm_reduce's sums for each (x: problem 0's
// blocks, then problem 1's; a problem of one slice has none). Dependent launch (TF_GLM_PDL_CHAIN's reduce, as
// glm_mm_reduce): it may start while the two-problem matmul that writes `part` runs, so every thread waits for that
// grid before it reads `part` or writes `out` (or exits).
struct MmReduceTwo {
    const float* part[2];
    void* out[2];
    long long total[2];
    int SK[2];
    int blocks0, f32;
};
extern "C" __global__ void glm_mm_reduce_two(const MmReduceTwo r) {
    qmm_frag::grid_launch();
    qmm_frag::grid_wait();
    const int z = (int)blockIdx.x >= r.blocks0 ? 1 : 0;
    const long long i = (long long)((int)blockIdx.x - (z ? r.blocks0 : 0)) * blockDim.x + threadIdx.x;
    const long long total = r.total[z];
    if (i >= total) return;
    const float* part = r.part[z];
    const int SK = r.SK[z];
    float acc = part[i];
    for (int sl = 1; sl < SK; ++sl) acc = acc + part[sl * total + i];
    if (r.f32) reinterpret_cast<float*>(r.out[z])[i] = acc;
    else reinterpret_cast<__nv_bfloat16*>(r.out[z])[i] = __float2bfloat16_rn(acc);
}

// -- tiling at load: ops.cu's packed rows and fp16 scales [N][K / 64] -> the tiled chunks and scales ---------------
template <int BITS>
__device__ __forceinline__ void tile_packed(const uint32_t* __restrict__ src, const __half* __restrict__ ss,
                                            uint32_t* __restrict__ dst, __half* __restrict__ ds, int N, int K) {
    constexpr int WPC = BITS / 2;
    const int KG = K >> 6, N8 = N >> 3;
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // a destination chunk: (tile, group, lane)
    if (c >= (long long)N8 * KG * 32) return;
    const int lane = (int)(c & 31);
    const long long tg = c >> 5;
    const int g = (int)(tg % KG), tt = (int)(tg / KG);
    const int n = tt * 8 + (lane >> 2);
    const uint32_t* row = src + (long long)n * (K >> 4) * WPC;
    int q[16];
#pragma unroll
    for (int v = 0; v < 16; ++v) {
        const int k = g * 64 + frag_k(lane, v);
        int sq[16];
        codes16<BITS>(row + (long long)(k >> 4) * WPC, sq);
        q[v] = sq[k & 15];
    }
    pack16<BITS>(q, dst + c * WPC);
    if (lane < 8) ds[(tg * 8) + lane] = ss[(long long)(tt * 8 + lane) * KG + g];
}

extern "C" __global__ void glm_mm_tile4(const uint32_t* src, const __half* ss, uint32_t* dst, __half* ds, int N, int K) { tile_packed<4>(src, ss, dst, ds, N, K); }
extern "C" __global__ void glm_mm_tile6(const uint32_t* src, const __half* ss, uint32_t* dst, __half* ds, int N, int K) { tile_packed<6>(src, ss, dst, ds, N, K); }
extern "C" __global__ void glm_mm_tile8(const uint32_t* src, const __half* ss, uint32_t* dst, __half* ds, int N, int K) { tile_packed<8>(src, ss, dst, ds, N, K); }

// bf16 rows [N][K] -> the tiled chunks (16 bf16 a chunk, in pair order)
extern "C" __global__ void glm_mm_tile16(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst, int N, int K) {
    const int KG = K >> 6, N8 = N >> 3;
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (long long)N8 * KG * 32) return;
    const int lane = (int)(c & 31);
    const long long tg = c >> 5;
    const int g = (int)(tg % KG), tt = (int)(tg / KG);
    const int n = tt * 8 + (lane >> 2);
#pragma unroll
    for (int v = 0; v < 16; ++v) dst[c * 16 + v] = src[(long long)n * K + g * 64 + frag_k(lane, v)];
}

// Opt-in W8A8 prompt linears (TF_GLM_W8A8=1): see glm_mm_a8.cuh.
#include "glm_mm_a8.cuh"
