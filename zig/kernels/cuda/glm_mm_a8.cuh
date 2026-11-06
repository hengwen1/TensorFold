// Opt-in W8A8 for packed 8-bit prompt linears (TF_GLM_W8A8=1): activations rounded to int8 with one fp32 scale per row
// and 64-group, then int8 x int8 IMMA (m16n8k32, exact int32 sums per group) on glm_mm's tiled 8-bit codes as stored.
//
// The tiled codes give lane (t = lane & 3) of a column tile, in a group of 64, the inputs frag_k(t, v) for v = 0..15
// (bf16 m16n8k16 B-fragment order). A group's dot product does not depend on the order of its 64 terms, so the
// activations are written permuted instead: position k' = 32 s + 16 r + 4 t + b (IMMA k-step s, register r, byte b,
// the m16n8k32 B-fragment order) holds input frag_k(t, 8 s + 4 r + b). Then word (2 s + r) of a lane's tiled chunk is
// exactly its IMMA register r of k-step s, and the activations' rows load with ldmatrix as for bf16.
#pragma once

namespace glm_a8 {

using namespace qmm_frag;

// Input 8 c + i of a group (i in 0..7: thread c's chunk) -> its permuted position k'.
__device__ __forceinline__ int perm_of(int c, int i) {
  return 32 * (c >> 2) + 16 * ((c >> 1) & 1) + 4 * (i >> 1) + 2 * (c & 1) + (i & 1);
}

__device__ __forceinline__ void imma(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void imma0(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  const int z = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(z));
}
// a group's int32 sum (|i| < 2^21) to fp32, exactly (I2F measured ~3% faster here than the two-add trick)
__device__ __forceinline__ float i2f(int i) { return __int2float_rn(i); }

// 64-byte rows of 4 chunks: conflict-free ldmatrix over 8 consecutive rows
__device__ __forceinline__ int swz4(int r, int c) { return c ^ ((r >> 1) & 3); }

template <int BN, int STAGES, int GS = 64>
struct Tile8 {
  static constexpr int BM = 128, WM = 2, WN = 4, THREADS = 256, MT = BM / WM / 16, NT = BN / WN / 8, TILES = BN / 8;
  static constexpr int X = BM * 64, XS = BM * 4 * (64 / GS), W = TILES * 32 * 4 * 4, WS = TILES * 16;
  static constexpr int STAGE = X + XS + W + WS, SMEM = STAGES * STAGE;
};

// out [M][N] (fp32 with f32, else bf16) = sum over groups g of (q [M][64 g ..] . codes) * xs[g][m] * ws[n][g].
// xs is [K/GS][mp] (rows of a group contiguous, mp = M rounded up to 4, padded so 16-byte loads stay inside).
// GS 32: an activation scale per IMMA k-step (inputs 32 s .. 32 s + 31 of a 64-group, see perm_of), each step's int32
// sum scaled on its own; GS 64: one scale per 64-group, both steps summed in int32 first.
template <int BN, int STAGES, bool ROWS_FIRST, int GS = 64>
__device__ __forceinline__ void mm8a8(const int8_t* __restrict__ q, const float* __restrict__ xs, int mp,
                                      const uint32_t* __restrict__ w, const __half* __restrict__ s,
                                      void* __restrict__ out, int f32, int M, int N, int K) {
  using T = Tile8<BN, STAGES, GS>;
  constexpr int SPG = 64 / GS;  // activation scales a 64-group
  extern __shared__ __align__(128) unsigned char buf[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / T::WN, wn = warp % T::WN;
  const int KG = K >> 6, N8 = N >> 3;
  // ROWS_FIRST: the row blocks of one column tile run together and share its weight tile in L2 (each read once)
  const int n0 = (ROWS_FIRST ? blockIdx.y : blockIdx.x) * BN, m0 = (ROWS_FIRST ? blockIdx.x : blockIdx.y) * T::BM;
  const int t0 = n0 >> 3;
  auto stage = [&](int st) { return buf + st * T::STAGE; };
  auto loadX = [&](int st, int g) {
    unsigned char* p = stage(st);
#pragma unroll
    for (int c = tid; c < T::BM * 4; c += T::THREADS) {
      const int r = c >> 2, ch = c & 3;
      const int row = min(m0 + r, M - 1);
      cp16z(p + r * 64 + swz4(r, ch) * 16, q + (long long)row * K + g * 64 + ch * 16, m0 + r < M);
    }
    if (tid < T::BM / 4 * SPG) {
      const int h = tid / (T::BM / 4), c = tid % (T::BM / 4);
      cp16(p + T::X + h * T::BM * 4 + c * 16, xs + (long long)(g * SPG + h) * mp + m0 + c * 4);
    }
  };
  auto loadW = [&](int st, int g) {
    unsigned char* pw = stage(st) + T::X + T::XS;
    constexpr int PER_TILE = 32 * 4 * 4 / 16;
    for (int c = tid; c < T::TILES * PER_TILE; c += T::THREADS) {
      const int tt = c / PER_TILE, off = c % PER_TILE;
      if (t0 + tt >= N8) continue;
      cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + ((long long)(t0 + tt) * KG + g) * 512 + off * 16);
    }
    unsigned char* ps = pw + T::W;
    for (int tt = tid; tt < T::TILES; tt += T::THREADS) {
      if (t0 + tt >= N8) continue;
      cp16(ps + tt * 16, s + ((long long)(t0 + tt) * KG + g) * 8);
    }
  };
  float acc[T::MT][T::NT][4];
#pragma unroll
  for (int i = 0; i < T::MT; ++i)
#pragma unroll
    for (int j = 0; j < T::NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
  grid_launch();
#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st)
    if (st < KG) loadW(st, st);
  grid_wait();
#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) {
    if (st < KG) loadX(st, st);
    commit();
  }
  const int g4 = lane >> 2, t4 = lane & 3;
  for (int it = 0; it < KG; ++it) {
    wait<STAGES - 2>();
    __syncthreads();
    const int next = it + STAGES - 1;
    if (next < KG) {
      loadX(next % STAGES, next);
      loadW(next % STAGES, next);
    }
    commit();
    const unsigned char* p = stage(it % STAGES);
    const float* px = reinterpret_cast<const float*>(p + T::X);
    const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X + T::XS);
    const __half* ps = reinterpret_cast<const __half*>(p + T::X + T::XS + T::W);
    uint32_t b[T::NT][4];
#pragma unroll
    for (int j = 0; j < T::NT; ++j) {
      const uint4 u = *reinterpret_cast<const uint4*>(pw + ((wn * T::NT + j) * 32 + lane) * 4);
      b[j][0] = u.x ^ 0x80808080u, b[j][1] = u.y ^ 0x80808080u, b[j][2] = u.z ^ 0x80808080u, b[j][3] = u.w ^ 0x80808080u;
    }
    float sw[T::NT][2];
#pragma unroll
    for (int j = 0; j < T::NT; ++j) {
      const __half2 h = *reinterpret_cast<const __half2*>(ps + (wn * T::NT + j) * 8 + t4 * 2);
      sw[j][0] = __low2float(h), sw[j][1] = __high2float(h);
    }
    // a step's (or, GS 64, a group's) int32 sums into the fp32 accumulators with their scales
    auto scale = [&](const int (&d)[T::MT][T::NT][4], const float* pxs) {
      float sx[T::MT][2];
#pragma unroll
      for (int i = 0; i < T::MT; ++i) {
        const int r = wm * (T::BM / T::WM) + i * 16 + g4;
        sx[i][0] = pxs[r], sx[i][1] = pxs[r + 8];
      }
#pragma unroll
      for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            acc[i][j][e] = __fmaf_rn(i2f(d[i][j][e]), sx[i][e >> 1] * sw[j][e & 1], acc[i][j][e]);
    };
    int d[T::MT][T::NT][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      uint32_t a[T::MT][4];
#pragma unroll
      for (int i = 0; i < T::MT; ++i) {
        const int r = wm * (T::BM / T::WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        ldmatrix4(a[i], p + r * 64 + swz4(r, ks * 2 + (lane >> 4)) * 16);
      }
#pragma unroll
      for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
          if (ks == 0 || GS == 32) imma0(d[i][j], a[i], b[j][2 * ks], b[j][2 * ks + 1]);
          else imma(d[i][j], a[i], b[j][2], b[j][3]);
        }
      if constexpr (GS == 32) scale(d, px + ks * T::BM);
    }
    if constexpr (GS == 64) scale(d, px);
  }
  wait<0>();
#pragma unroll
  for (int i = 0; i < T::MT; ++i)
#pragma unroll
    for (int j = 0; j < T::NT; ++j) {
      const int col = n0 + (wn * T::NT + j) * 8 + t4 * 2;
      if (col >= N) continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = m0 + wm * (T::BM / T::WM) + i * 16 + g4 + h * 8;
        if (row >= M) continue;
        const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
        if (f32) {
          *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + (long long)row * N + col) = make_float2(v0, v1);
        } else {
          *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + (long long)row * N + col) =
              __floats2bfloat162_rn(v0, v1);
        }
      }
    }
}


}  // namespace glm_a8

// Activations for glm_mm8a8: x bf16 [M][K] (rows ldx apart) -> q int8 [M][K] (each 64-group permuted, see above) and
// xs fp32 [K/gs][mp]: per row and gs-group (gs 64 or 32), scale = max|x| / 127 and q = rint(x / scale) (0 for an
// all-zero group). 8 threads a 64-group (4 a 32-group), 32 64-groups a block of 256.
extern "C" __global__ void __launch_bounds__(256) glm_q8_rows(const __nv_bfloat16* __restrict__ x, long long ldx,
    int8_t* __restrict__ q, float* __restrict__ xs, int mp, int M, int K, int gs) {
  const int KG = K >> 6;
  const long long gid = (long long)blockIdx.x * 32 + (threadIdx.x >> 3);
  const int c = threadIdx.x & 7;
  const bool ok = gid < (long long)M * KG;
  const long long row = ok ? gid / KG : 0;
  const int g = ok ? (int)(gid % KG) : 0;
  float v[8];
  uint4 u = make_uint4(0, 0, 0, 0);
  if (ok) u = *reinterpret_cast<const uint4*>(x + row * ldx + g * 64 + c * 8);
  const __nv_bfloat16* hv = reinterpret_cast<const __nv_bfloat16*>(&u);
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = __bfloat162float(hv[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  const float a4 = __shfl_xor_sync(0xffffffffu, amax, 4);
  if (gs == 64) amax = fmaxf(amax, a4);
  if (!ok) return;
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  int8_t* dst = q + row * K + g * 64;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const int a = max(-127, min(127, __float2int_rn(v[i] * inv)));
    const int b = max(-127, min(127, __float2int_rn(v[i + 1] * inv)));
    *reinterpret_cast<uint16_t*>(dst + glm_a8::perm_of(c, i)) = (uint16_t)((a & 255) | ((b & 255) << 8));
  }
  if (gs == 64 && c == 0) xs[(long long)g * mp + row] = amax * (1.0f / 127.0f);
  if (gs == 32 && (c & 3) == 0) xs[(long long)(2 * g + (c >> 2)) * mp + row] = amax * (1.0f / 127.0f);
}

// The instances cuda_mm.zig launches (row blocks fastest): 64 columns a block for pieces (two blocks an SM), 128 for a
// whole pass. Measured at 2048 x 12576 x 4096: 0.67 ms against glm_mm8_128s2's 1.12 ms (column-first order, four
// stages, or the scale epilogue one group behind measured no faster).
#define GLM_MM8A8(name, BN, MINB, GS)                                                                                  \
  extern "C" __global__ void __launch_bounds__(256, MINB) name(const int8_t* __restrict__ q, const float* __restrict__ xs, \
      int mp, const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, int f32, int M,    \
      int N, int K) { glm_a8::mm8a8<BN, 3, true, GS>(q, xs, mp, w, s, out, f32, M, N, K); }
GLM_MM8A8(glm_mm8a8_64s3r, 64, 2, 64)
GLM_MM8A8(glm_mm8a8_128s3r, 128, 1, 64)
// TF_GLM_W8A8=32: an activation scale per 32 inputs (each IMMA k-step's own), the same codes
GLM_MM8A8(glm_mm8a8_64s3r_g32, 64, 2, 32)
GLM_MM8A8(glm_mm8a8_128s3r_g32, 128, 1, 32)
#undef GLM_MM8A8
