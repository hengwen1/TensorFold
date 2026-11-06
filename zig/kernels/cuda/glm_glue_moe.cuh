// GLM's MLP and MoE glue by hand: glue._swiglu, glue._topk and qmm._group_sums (the router's dot is glm_glue_router).
#pragma once

#include "exl3_group.cuh"

#include "glm_triton.cuh"

using namespace glm_triton;

namespace glm_moe {

// A thread's S consecutive values summed from the second, then the butterfly over the 64 / S lanes of their group.
template <int S>
__device__ __forceinline__ float group64(const float* q) {
  float g = q[0] + q[1];
  for (int j = 2; j < S; ++j) g = q[j] + g;
  for (int o = 32 / S; o > 0; o >>= 1) g = g + __shfl_xor_sync(0xffffffffu, g, o);
  return g;
}

__device__ __forceinline__ float swiglu1(float g, float u, float limit) {
  g = fminf(g, limit);
  u = fminf(fmaxf(u, 0.0f - limit), limit);
  const float s = rbf(div_full(g, exp_approx(0.0f - g) + 1.0f));
  return rbf(__fmul_rn(u, s));
}

}  // namespace glm_moe

// glue._swiglu with BLOCK 512 (128 threads, 4 values each, 64-group sums over 16 lanes), column block blockIdx.y.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_swiglu_512(const __nv_bfloat16* gu, __nv_bfloat16* out, float* xs, float limit, int64_t w) {
  const int64_t r = blockIdx.x, d0 = blockIdx.y * 512 + threadIdx.x * 4;
  const uint2 gv = *reinterpret_cast<const uint2*>(gu + r * 2 * w + d0);
  const uint2 uv = *reinterpret_cast<const uint2*>(gu + r * 2 * w + w + d0);
  const __nv_bfloat16* gh = reinterpret_cast<const __nv_bfloat16*>(&gv);
  const __nv_bfloat16* uh = reinterpret_cast<const __nv_bfloat16*>(&uv);
  float a[4];
  uint2 ov;
  __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ov);
  for (int j = 0; j < 4; ++j) {
    a[j] = glm_moe::swiglu1(bf(gh[j]), bf(uh[j]), limit);
    oh[j] = __float2bfloat16_rn(a[j]);
  }
  *reinterpret_cast<uint2*>(out + r * w + d0) = ov;
  const float g = glm_moe::group64<4>(a);
  if ((threadIdx.x & 15) == 0) xs[r * (w / 64) + d0 / 64] = g;
}

// glue._swiglu with BLOCK 64 (the TP=3 shared expert's 704): one value a thread, the warp's butterfly, then 2 warps.
extern "C" __global__ void __launch_bounds__(64)
    tf_glm_swiglu_64(const __nv_bfloat16* gu, __nv_bfloat16* out, float* xs, float limit, int64_t w) {
  __shared__ float smem[2];
  const int64_t r = blockIdx.x, d = blockIdx.y * 64 + threadIdx.x;
  const float a = glm_moe::swiglu1(bf(gu[r * 2 * w + d]), bf(gu[r * 2 * w + w + d]), limit);
  out[r * w + d] = __float2bfloat16_rn(a);
  const float g = block_sum<2>(a, smem);
  if (threadIdx.x == 0) xs[r * (w / 64) + blockIdx.y] = g;
}

// qmm._group_sums: row blockIdx.x's 64-input sums, 8 lanes a group of 8 consecutive inputs each, 16 groups a block.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_group_sums(const __nv_bfloat16* x, float* xs, int64_t x_stride, int64_t kg) {
  const int64_t m = blockIdx.x, g = blockIdx.y * 16 + threadIdx.x / 8;
  float q[8];
  for (int j = 0; j < 8; ++j) q[j] = g < kg ? bf(x[m * x_stride + g * 64 + (threadIdx.x & 7) * 8 + j]) : 0.0f;
  const float s = glm_moe::group64<8>(q);
  if ((threadIdx.x & 7) == 0 && g < kg) xs[m * kg + g] = s;
}

// glue._topk of row `r` by one warp: top_k by sigmoid + bias (ties to lower id), normed if `norm`, x scale; then the
// shared expert. `sh` (or null): the row's routed picks also at sh[r * top_k ..], compact, for the EXL3 grouping.
namespace glm_moe {
__device__ __forceinline__ void topk_row(int64_t r, int lane, const float* logits, const float* bias, int* pick, float* wts, float scale,
                                         int ne, int top_k, int norm, int* sh) {
  constexpr int MAXE = 16;  // 512 experts a row
  float score[MAXE], choice[MAXE];
  for (int i = 0; i < MAXE; ++i) {
    const int e = i * 32 + lane;
    score[i] = e < ne ? div_full(1.0f, exp_approx(0.0f - logits[r * ne + e]) + 1.0f) : 0.0f;
    choice[i] = e < ne ? score[i] + bias[e] : -INFINITY;
  }
  float total = 0.0f, mine = 0.0f;
  int mine_id = 0;
  for (int k = 0; k < top_k; ++k) {
    float mx = choice[0];
    for (int i = 1; i < MAXE; ++i) mx = fmaxf(mx, choice[i]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    int id = 1 << 30;
    for (int i = MAXE - 1; i >= 0; --i)
      if (choice[i] == mx) id = i * 32 + lane;
    for (int o = 16; o > 0; o >>= 1) id = min(id, __shfl_xor_sync(0xffffffffu, id, o));
    float sk = 0.0f;
    for (int i = 0; i < MAXE; ++i)
      if (i * 32 + lane == id) {
        sk = score[i];
        choice[i] = -INFINITY;
      }
    sk = __shfl_sync(0xffffffffu, sk, id & 31);
    total = total + sk;
    if (lane == k) {
      mine = sk;
      mine_id = id;
    }
  }
  if (lane < top_k) {
    const float v = norm ? div_full(mine, total + 1e-20f) : mine;
    pick[r * (top_k + 1) + lane] = mine_id;
    wts[r * (top_k + 1) + lane] = __fmul_rn(v, scale);
    if (sh) sh[r * top_k + lane] = mine_id;
  } else if (lane == top_k) {
    pick[r * (top_k + 1) + lane] = ne;
    wts[r * (top_k + 1) + lane] = 1.0f;
  }
}
}  // namespace glm_moe

// glue._topk, a warp a row.
extern "C" __global__ void __launch_bounds__(32)
    tf_glm_topk(const float* logits, const float* bias, int* pick, float* wts, float scale, int ne, int top_k,
                int norm) {
  glm_moe::topk_row(blockIdx.x, threadIdx.x, logits, bias, pick, wts, scale, ne, top_k, norm, nullptr);
}

namespace glm_moe {
// The router's K slices a fused select sums: glue.router_ks.
constexpr int ROUTER_KS = 8;

// An order-preserving key of a choice: a choice is never -0 (a sigmoid >= +0 plus a bias), and NaN takes the lowest
// key because tf_glm_topk's fmaxf passes over NaN: max and equality on keys are its fmaxf and == on choices.
__device__ __forceinline__ unsigned choice_key(float v) {
  const unsigned u = __float_as_uint(v);
  if (v != v) return 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// topk_row's selection of row `r` from its sigmoid scores `sc` [r][ne] and the bias `sb`, on choice keys with
// redux.sync (max key, then the lowest id holding it): the same picks in the same order, the same weights.
__device__ __forceinline__ void select_row_keys(int64_t r, int lane, const float* sc, const float* sb, int* pick,
                                                float* wts, float scale, int ne, int top_k, int norm, int* sh) {
  constexpr int MAXE = 16;                   // topk_row's 512 slots a row: e >= ne are -inf
  constexpr unsigned NEG_INF = 0x007fffffu;  // choice_key(-inf)
  float score[MAXE];
  unsigned key[MAXE];
#pragma unroll
  for (int i = 0; i < MAXE; ++i) {
    const int e = i * 32 + lane;
    score[i] = e < ne ? sc[r * ne + e] : 0.0f;
    key[i] = e < ne ? choice_key(score[i] + sb[e]) : NEG_INF;
  }
  float total = 0.0f, mine = 0.0f;
  int mine_id = 0;
  for (int k = 0; k < top_k; ++k) {
    unsigned mx = key[0];
#pragma unroll
    for (int i = 1; i < MAXE; ++i) mx = max(mx, key[i]);
    mx = __reduce_max_sync(0xffffffffu, mx);
    int id = 1 << 30;
    if (mx != 0u) {  // every slot NaN: topk_row's max is NaN and matches nothing
#pragma unroll
      for (int i = MAXE - 1; i >= 0; --i)
        if (key[i] == mx) id = i * 32 + lane;
    }
    id = static_cast<int>(__reduce_min_sync(0xffffffffu, static_cast<unsigned>(id)));
    float sk = 0.0f;
#pragma unroll
    for (int i = 0; i < MAXE; ++i)
      if (i * 32 + lane == id) {
        sk = score[i];
        key[i] = NEG_INF;
      }
    sk = __shfl_sync(0xffffffffu, sk, id & 31);
    total = total + sk;
    if (lane == k) {
      mine = sk;
      mine_id = id;
    }
  }
  if (lane < top_k) {
    const float v = norm ? div_full(mine, total + 1e-20f) : mine;
    pick[r * (top_k + 1) + lane] = mine_id;
    wts[r * (top_k + 1) + lane] = __fmul_rn(v, scale);
    if (sh) sh[r * top_k + lane] = mine_id;
  } else if (lane == top_k) {
    pick[r * (top_k + 1) + lane] = ne;
    wts[r * (top_k + 1) + lane] = 1.0f;
  }
}
// tf_exl3::group_rows' outputs from `n` picks (`slots` a row) in shared memory, for experts < 512: distinct experts
// (< `experts`) in id order in `uids`, their count in ucount[0], and for each its picks' row * 32 + slot in pick order,
// at most `maxm`, then -1 to maxm. Each pick counts itself into its expert (shared atomics: only counts) and finds
// its place among its expert's picks by scanning the picks before it; one thread an expert lays out uids and the -1
// tails; each pick writes its own member word. The same words as group_rows' expert-by-pick scans, in fewer steps.
// cnt and place: shared [512] each. Every thread of a GROUP_THREADS block calls it.
__device__ __forceinline__ void group_rows_fast(const int* sh_pick, int n, int slots, int experts, int maxm, int* uids,
                                                int* ucount, int* members, int* cnt, int* place) {
  __shared__ int warp_tot[tf_exl3::GROUP_THREADS / 32];
  for (int e = threadIdx.x; e < 512; e += blockDim.x) cnt[e] = 0;
  __syncthreads();
  int e = -1, rank = 0;
  if (threadIdx.x < n) {
    const int v = sh_pick[threadIdx.x];
    if (v >= 0 && v < experts) {
      e = v;
      atomicAdd(&cnt[e], 1);
      for (int i = 0; i < static_cast<int>(threadIdx.x); ++i) rank += sh_pick[i] == e;
    }
  }
  __syncthreads();
  // the distinct experts in id order: an exclusive scan of "used" over experts 4 a thread (group_rows' layout)
  constexpr int PER = tf_exl3::GROUP_PER_THREAD;
  int used = 0;
#pragma unroll
  for (int q = 0; q < PER; ++q) {
    const int x = threadIdx.x * PER + q;
    used += x < experts && x < 512 && cnt[x] > 0;
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int inc = used;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += v;
  }
  if (lane == 31) warp_tot[warp] = inc;
  __syncthreads();
  if (warp == 0) {
    const int v = warp_tot[lane];
    int sc = v;
    for (int o = 1; o < 32; o <<= 1) {
      const int x = __shfl_up_sync(0xffffffffu, sc, o);
      if (lane >= o) sc += x;
    }
    warp_tot[lane] = sc - v;  // exclusive per warp
    if (lane == 31) ucount[0] = sc;
  }
  __syncthreads();
  int at = warp_tot[warp] + inc - used;
#pragma unroll
  for (int q = 0; q < PER; ++q) {
    const int x = threadIdx.x * PER + q;
    if (x >= experts || x >= 512 || cnt[x] == 0) continue;
    place[x] = at;
    uids[at] = x;
    for (int j = min(cnt[x], maxm); j < maxm; ++j) members[at * maxm + j] = -1;
    ++at;
  }
  __syncthreads();
  if (e >= 0 && rank < maxm) members[place[e] * maxm + rank] = (threadIdx.x / slots) * 32 + (threadIdx.x % slots);
}
}  // namespace glm_moe

// A decode window's router logits, select and grouping in one launch: every thread sums the router's ROUTER_KS slices
// of rows x ne logits (tf_glm_slice_sum's adds, slice order) into `logits` and their sigmoids (topk_row's) into shared
// memory (rows x ne floats, dynamic), then a warp a row selects (select_row_keys) and the block groups (group_rows_fast, tf_exl3_group's
// words): tf_glm_slice_sum's then tf_glm_topk_group's outputs, bit for bit, without the slice-sum launch.
extern "C" __global__ void __launch_bounds__(tf_exl3::GROUP_THREADS)
    tf_glm_topk_group_sum(const float* part, float* logits, const float* bias, int* pick, float* wts, float scale, int ne,
                          int top_k, int norm, int rows, int* pick8, int* uids, int* ucount, int* members) {
  extern __shared__ float sc[];
  __shared__ float sb[512];
  __shared__ int sh_pick[tf_exl3::GROUP_THREADS];
  __shared__ int cnt[512], place[512];
  const int total = rows * ne;
  for (int i = threadIdx.x; i < ne; i += blockDim.x) sb[i] = bias[i];
  // Dependent launch (decode windows): the bias (immutable) is staged before the wait; the router's slices, an
  // earlier grid's, and every store come after it.
  pdl_wait();
  pdl_trigger();
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    float v[glm_moe::ROUTER_KS];
#pragma unroll
    for (int s = 0; s < glm_moe::ROUTER_KS; ++s) v[s] = part[static_cast<int64_t>(s) * total + i];
    float lg = v[0];
#pragma unroll
    for (int s = 1; s < glm_moe::ROUTER_KS; ++s) lg = lg + v[s];
    logits[i] = lg;
    sc[i] = div_full(1.0f, exp_approx(0.0f - lg) + 1.0f);
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp < rows) glm_moe::select_row_keys(warp, lane, sc, sb, pick, wts, scale, ne, top_k, norm, sh_pick);
  __syncthreads();
  const int n = rows * top_k;
  for (int i = threadIdx.x; i < n; i += tf_exl3::GROUP_THREADS) pick8[i] = sh_pick[i];
  glm_moe::group_rows_fast(sh_pick, n, top_k, ne, rows, uids, ucount, members, cnt, place);
}

// A decode window's glue._topk (rows <= 32, a warp a row, tf_glm_topk's own code) and then, for the EXL3 experts, the
// routed picks compact (top_k a row) and their grouping (tf_exl3_group's own code) from the same block: no copy, no
// separate group launch. 1024 threads (the grouping's block).
extern "C" __global__ void __launch_bounds__(tf_exl3::GROUP_THREADS)
    tf_glm_topk_group(const float* logits, const float* bias, int* pick, float* wts, float scale, int ne, int top_k, int norm,
                      int rows, int* pick8, int* uids, int* ucount, int* members) {
  __shared__ int sh_pick[tf_exl3::GROUP_THREADS];
  // Dependent launch (decode windows): the logits are an earlier grid's, so the wait comes first.
  pdl_wait();
  pdl_trigger();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp < rows) glm_moe::topk_row(warp, lane, logits, bias, pick, wts, scale, ne, top_k, norm, sh_pick);
  __syncthreads();
  const int n = rows * top_k;
  for (int i = threadIdx.x; i < n; i += tf_exl3::GROUP_THREADS) pick8[i] = sh_pick[i];
  tf_exl3::group_rows(sh_pick, n, top_k, ne, rows, uids, ucount, members);
}
