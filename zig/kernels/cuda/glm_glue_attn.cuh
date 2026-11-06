// latent.py's MLA attention and sparse.py's index scores by hand: the chunk kernels' online softmax, the chunk merge.
#pragma once

#include "glm_glue_router.cuh"
#include <cuda_fp8.h>

namespace glm_attn {

constexpr int LW = 512, KT = 32, CH = 512, KPAD = LW + 8;
// chunks_wide's block: its 8 warps' (key group, head block) split and the FP8 tiles' units a thread both assume it;
// cuda_latent.chunk_threads launches it, and a launch of any other width traps instead of leaving keys unloaded.
constexpr int CHUNK_THREADS = 256;
static_assert(KT * (LW / 8) % CHUNK_THREADS == 0, "an FP8 tile's 8-code units split evenly over the block");

// p fp32 summed as Triton's tile does: a thread's two columns, the quad (xor 2, 1), then warps (w0 + w2) + (w1 + w3).
__device__ __forceinline__ float quad_sum(float v) {
  v = v + __shfl_xor_sync(0xffffffffu, v, 2);
  return v + __shfl_xor_sync(0xffffffffu, v, 1);
}
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
}

__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
template <bool TRANS>
__device__ __forceinline__ void ldm4(uint32_t* r, const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  if (TRANS)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
  else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

// A 512-wide bf16 row's 16-byte unit u in shared memory, swizzled by the row so ldmatrix's eight rows hit eight banks.
__device__ __forceinline__ int wide(int row, int u) { return row * LW + ((u ^ (row & 7)) << 3); }
// p's [32][32] rows (64 bytes): unit u of row r.
__device__ __forceinline__ int narrow(int row, int u) { return row * KT + ((u ^ ((row >> 1) & 3)) << 3); }

// latent._dense_chunks/_sparse_chunks (sparse 0/1), block (row, HB heads, chunk), 8 warps: online softmax by 32 keys.
template <int HB>
constexpr int wide_smem() {
  return (HB * LW + 2 * KT * LW + HB * KT) * 2 + (HB * 4 + 2 * HB) * 4;
}

// PAGED8: an E4M3 cache on KV pages (pt, pstep bytes apart): a dense chunk is one page, a sparse key's logical token goes
// through the table to its page; codes and scales then load as on a window, so the arithmetic is the window's.
template <int HB, bool FP8 = false, bool PAGED8 = false>
__device__ void chunks_wide(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok,
                            const int* cnt, float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w,
                            float scale, int sparse, const int* pt, int64_t pstep, const float* scales = nullptr,
                            int64_t rr = -1, int64_t cc = -1) {
  constexpr int MT = HB / 16;
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* kvs = qs + HB * LW;
  __nv_bfloat16* ps = kvs + 2 * KT * LW;
  float* red = reinterpret_cast<float*>(ps + HB * KT);
  float* stat = red + HB * 4;
  if (blockDim.x != CHUNK_THREADS) __trap();  // the launch contract above, not a tuning knob
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, q = lane & 3;
  const int kg = warp & 3, hh = warp >> 2;
  // a segment-batched launch passes the block's row and chunk within its segment (rr, cc)
  const int64_t r = rr >= 0 ? rr : (int64_t)blockIdx.x, h0 = (int64_t)blockIdx.y * HB, c = cc >= 0 ? cc : (int64_t)blockIdx.z;
  const int64_t n = sparse ? cnt[r] : 0, limit = sparse ? 0 : (int64_t)pos[0] + r, start = c * CH;
  const bool live = sparse ? start < n : start <= limit;
  // a dense chunk's 512 keys are one page, its rows from that page's; a sparse key's row comes as the token list has it
  const __nv_bfloat16* keys_at = !FP8 && pt && !sparse && live ? lc + ((int64_t)pt[c] * (pstep / 2) - start * LW) : lc;
  // E4M3 tiles: each thread's 8-code units in one 8-byte load apiece, all issued before any is converted (fetch8),
  // then converted value for value as before (store8); the loop fetches tile + 2 before its math and stores it after.
  constexpr int UPT = KT * (LW / 8) / CHUNK_THREADS;  // units a thread moves a tile
  uint2 raw[FP8 ? UPT : 1];
  float rsc[FP8 ? UPT : 1];
  // on pages, a dense chunk is one page: its codes and scales are the window's at bases moved once (as keys_at above),
  // so its keys load with the window's arithmetic; a sparse key's logical token goes through the table to its page.
  const unsigned char* codes8 = reinterpret_cast<const unsigned char*>(lc);
  const float* scales8 = scales;
  if constexpr (PAGED8) {
    if (!sparse && live) {
      const int64_t page = (int64_t)pt[c] * pstep;
      codes8 += page - start * LW;
      scales8 = reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(scales) + page) - start;
    }
  }
  auto fetch8 = [&](int tile) {
    const int64_t first = start + tile * KT;
    if constexpr (PAGED8) {
      if (sparse) {
        const unsigned char* scale_bytes = reinterpret_cast<const unsigned char*>(scales);
#pragma unroll
        for (int i = 0; i < UPT; ++i) {
          const int idx = t + i * CHUNK_THREADS, key = idx / (LW / 8), u = idx % (LW / 8);
          const int64_t ki = first + key;
          const bool ok = ki < n;
          const int s = ok ? tok[r * w + ki] : 0;  // a visible position: 0 <= s < 2^31
          const int64_t page = ok ? (int64_t)pt[s >> 9] * pstep : 0;  // CH = 512 rows a page
          const int row = s & (CH - 1);
          raw[i] = ok ? __ldg(reinterpret_cast<const uint2*>(codes8 + page + (row * LW + u * 8))) : make_uint2(0u, 0u);
          rsc[i] = ok ? __ldg(reinterpret_cast<const float*>(scale_bytes + page + row * 4)) : 1.0f;
        }
        return;
      }
    }
#pragma unroll
    for (int i = 0; i < UPT; ++i) {
      const int idx = t + i * CHUNK_THREADS, key = idx / (LW / 8), u = idx % (LW / 8);
      const int64_t ki = first + key;
      const bool ok = sparse ? ki < n : ki <= limit;
      const int64_t src = ok ? (sparse ? (int64_t)tok[r * w + ki] : ki) : 0;
      raw[i] = ok ? __ldg(reinterpret_cast<const uint2*>(codes8 + src * LW + u * 8)) : make_uint2(0u, 0u);
      rsc[i] = ok ? __ldg(scales8 + src) : 1.0f;
    }
  };
  auto store8 = [&](int tile) {
    __nv_bfloat16* dst = kvs + (tile & 1) * KT * LW;
#pragma unroll
    for (int i = 0; i < UPT; ++i) {
      const int idx = t + i * CHUNK_THREADS, key = idx / (LW / 8), u = idx % (LW / 8);
      const uint32_t wv[2] = {raw[i].x, raw[i].y};
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        const uint16_t bits = (uint16_t)((wv[j / 4] >> (8 * (j % 4))) & 0xffffu);  // codes j, j + 1 (little-endian)
        const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3)));
        dst[wide(key, u) + j] = __float2bfloat16_rn(__fmul_rn(v.x, rsc[i]));
        dst[wide(key, u) + j + 1] = __float2bfloat16_rn(__fmul_rn(v.y, rsc[i]));
      }
    }
  };
  auto load_kv = [&](int tile) {
    if constexpr (FP8) {
      fetch8(tile);
      store8(tile);
      return;
    }
    __nv_bfloat16* dst = kvs + (tile & 1) * KT * LW;
    const int64_t first = start + tile * KT;
    for (int idx = t; idx < KT * (LW / 8); idx += blockDim.x) {
      const int key = idx / (LW / 8), u = idx % (LW / 8);
      const int64_t ki = first + key;
      const bool ok = sparse ? ki < n : ki <= limit;
      const int64_t src = ok ? (sparse ? (int64_t)tok[r * w + ki] : ki) : 0;
      cp16(dst + wide(key, u), keys_at + src * LW + u * 8, ok);
    }
  };
  float m[MT][2], l[MT][2], o[MT][8][4];
  for (int i = 0; i < MT; ++i)
    for (int e = 0; e < 2; ++e) m[i][e] = -INFINITY, l[i][e] = 0.0f;
  for (int i = 0; i < MT; ++i)
    for (int j = 0; j < 8; ++j)
      for (int e = 0; e < 4; ++e) o[i][j][e] = 0.0f;
  constexpr int tiles = CH / KT;
  if (live) {
    for (int idx = t; idx < HB * (LW / 8); idx += blockDim.x) {
      const int hd = idx / (LW / 8), u = idx % (LW / 8);
      const bool ok = h0 + hd < heads;
      cp16(qs + wide(hd, u), qa + (r * heads + (ok ? h0 + hd : 0)) * LW + u * 8, ok);
    }
    load_kv(0);
    commit();
    load_kv(1);
    commit();
  }
  for (int tile = 0; live && tile < tiles; ++tile) {
    if constexpr (FP8) {
      wait<0>();
    } else {
      if (tile + 1 < tiles)
        wait<1>();
      else
        wait<0>();
    }
    __syncthreads();
    if constexpr (FP8) {
      if (tile + 2 < tiles) fetch8(tile + 2);  // in flight during this tile's math
    }
    const __nv_bfloat16* kv = kvs + (tile & 1) * KT * LW;
    const int64_t first = start + tile * KT;
    float sc[4];
    if (hh < MT) {
      float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
      for (int kk = 0; kk < LW; kk += 32) {
        uint32_t a0[4], a1[4], b[4];
        ldm4<false>(a0, qs + wide(hh * 16 + (lane & 15), (kk >> 3) + (lane >> 4)));
        ldm4<false>(a1, qs + wide(hh * 16 + (lane & 15), ((kk + 16) >> 3) + (lane >> 4)));
        ldm4<false>(b, kv + wide(kg * 8 + (lane & 7), (kk >> 3) + (lane >> 3)));
        glm_router::mma16816(acc, a0, b);
        glm_router::mma16816(acc, a1, b + 2);
      }
      for (int e = 0; e < 2; ++e) {
        const int64_t ki = first + kg * 8 + q * 2 + e;
        const bool ok = sparse ? ki < n : ki <= limit;
        sc[e] = ok ? __fmul_rn(acc[e], scale) : -INFINITY;
        sc[e + 2] = ok ? __fmul_rn(acc[e + 2], scale) : -INFINITY;
      }
      for (int hf = 0; hf < 2; ++hf) {
        const float mx = quad_max(fmaxf(sc[hf * 2], sc[hf * 2 + 1]));
        if (q == 0) red[(hh * 16 + hf * 8 + g) * 4 + kg] = mx;
      }
    }
    __syncthreads();
    if (t < HB * 4) {
      float v = red[t];
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
      if ((t & 3) == 0) stat[t / 4] = v;
    }
    __syncthreads();
    float alpha[MT][2];
    for (int i = 0; i < MT; ++i)
      for (int hf = 0; hf < 2; ++hf) {
        const float tm = stat[i * 16 + hf * 8 + g];
        const bool active = tm != -INFINITY;
        const float nm = active ? fmaxf(m[i][hf], tm) : m[i][hf];
        alpha[i][hf] = active ? (m[i][hf] == -INFINITY ? 0.0f : glm_triton::exp_approx(m[i][hf] - nm)) : 1.0f;
        if (i == hh) {
          float p[2];
          for (int e = 0; e < 2; ++e) {
            const int64_t ki = first + kg * 8 + q * 2 + e;
            const bool ok = sparse ? ki < n : ki <= limit;
            p[e] = active && ok ? glm_triton::exp_approx(sc[hf * 2 + e] - nm) : 0.0f;
          }
          const int hr = i * 16 + hf * 8 + g, col = kg * 8 + q * 2;
          *reinterpret_cast<__nv_bfloat162*>(ps + narrow(hr, col >> 3) + (col & 7)) = __floats2bfloat162_rn(p[0], p[1]);
          const float s = quad_sum(p[0] + p[1]);
          if (q == 0) red[hr * 4 + kg] = s;
        }
        m[i][hf] = nm;
      }
    __syncthreads();
    if (t < HB * 4) {
      float v = red[t];
      v = v + __shfl_xor_sync(0xffffffffu, v, 2);
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      if ((t & 3) == 0) stat[HB + t / 4] = v;
    }
    for (int i = 0; i < MT; ++i)
      for (int j = 0; j < 8; ++j)
        for (int e = 0; e < 4; ++e) o[i][j][e] = __fmul_rn(o[i][j][e], alpha[i][e / 2]);
#pragma unroll
    for (int k16 = 0; k16 < KT; k16 += 16) {
      uint32_t a[MT][4];
      for (int i = 0; i < MT; ++i) ldm4<false>(a[i], ps + narrow(i * 16 + (lane & 15), (k16 >> 3) + (lane >> 4)));
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        uint32_t b[4];
        ldm4<true>(b, kv + wide(k16 + (lane & 15), ((warp * 64 + j * 8) >> 3) + (lane >> 4)));
        for (int i = 0; i < MT; ++i) {
          glm_router::mma16816(o[i][j], a[i], b);
          glm_router::mma16816(o[i][j + 1], a[i], b + 2);
        }
      }
    }
    __syncthreads();
    for (int i = 0; i < MT; ++i)
      for (int hf = 0; hf < 2; ++hf) l[i][hf] = __fmaf_rn(l[i][hf], alpha[i][hf], stat[HB + i * 16 + hf * 8 + g]);
    if constexpr (FP8) {
      if (tile + 2 < tiles) store8(tile + 2);
    } else if (tile + 2 < tiles) {
      load_kv(tile + 2);
    }
    commit();
  }
  for (int i = 0; i < MT; ++i)
    for (int e = 0; e < 4; ++e) {
      const int64_t hd = h0 + i * 16 + g + (e / 2) * 8;
      if (hd >= heads) continue;
      const int64_t base = (c * rows + r) * heads + hd;
      for (int j = 0; j < 8; ++j) po[base * LW + warp * 64 + j * 8 + q * 2 + e % 2] = o[i][j][e];
      if (warp == 0 && q == 0 && e % 2 == 0) {
        pm[base] = m[i][e / 2];
        pl[base] = l[i][e / 2];
      }
    }
}

// chunks_wide's sparse rows with tf_glm_merge in the block (block (row, HB heads), grid z 1): the row's nch chunks
// in turn, each computed as chunks_wide's own block computes it (fresh m, l, o; the same code, copied, so chunks_wide's
// kernels keep their bytes), then merged into the row's running result by tf_glm_merge's sequence, chunk by chunk;
// out bf16 [rows][heads][LW] as tf_glm_merge writes it, and count-0 rows left alone. No partials.
template <int HB, bool FP8 = false, bool PAGED8 = false>
__device__ void chunks_fused(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok,
                            const int* cnt, float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w,
                            float scale, int sparse, const int* pt, int64_t pstep, const float* scales = nullptr,
                            __nv_bfloat16* out = nullptr, int nch = 1) {
  constexpr int MT = HB / 16;
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* kvs = qs + HB * LW;
  __nv_bfloat16* ps = kvs + 2 * KT * LW;
  float* red = reinterpret_cast<float*>(ps + HB * KT);
  float* stat = red + HB * 4;
  if (blockDim.x != CHUNK_THREADS) __trap();  // the launch contract above, not a tuning knob
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, q = lane & 3;
  const int kg = warp & 3, hh = warp >> 2;
  const int64_t r = blockIdx.x, h0 = (int64_t)blockIdx.y * HB;
  const int64_t n = sparse ? cnt[r] : 0, limit = sparse ? 0 : (int64_t)pos[0] + r;
  if (n == 0) return;  // tf_glm_merge skips sparse count-0 rows
  float om[MT][8][4], mm[MT][2], lm[MT][2];  // the merge's running o, m, l (tf_glm_merge's registers)
  for (int64_t c = 0; c < nch; ++c) {
  const int64_t start = c * CH;
  const bool live = sparse ? start < n : start <= limit;
  // a dense chunk's 512 keys are one page, its rows from that page's; a sparse key's row comes as the token list has it
  const __nv_bfloat16* keys_at = !FP8 && pt && !sparse && live ? lc + ((int64_t)pt[c] * (pstep / 2) - start * LW) : lc;
  // E4M3 tiles: each thread's 8-code units in one 8-byte load apiece, all issued before any is converted (fetch8),
  // then converted value for value as before (store8); the loop fetches tile + 2 before its math and stores it after.
  constexpr int UPT = KT * (LW / 8) / CHUNK_THREADS;  // units a thread moves a tile
  uint2 raw[FP8 ? UPT : 1];
  float rsc[FP8 ? UPT : 1];
  // on pages, a dense chunk is one page: its codes and scales are the window's at bases moved once (as keys_at above),
  // so its keys load with the window's arithmetic; a sparse key's logical token goes through the table to its page.
  const unsigned char* codes8 = reinterpret_cast<const unsigned char*>(lc);
  const float* scales8 = scales;
  if constexpr (PAGED8) {
    if (!sparse && live) {
      const int64_t page = (int64_t)pt[c] * pstep;
      codes8 += page - start * LW;
      scales8 = reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(scales) + page) - start;
    }
  }
  auto fetch8 = [&](int tile) {
    const int64_t first = start + tile * KT;
    if constexpr (PAGED8) {
      if (sparse) {
        const unsigned char* scale_bytes = reinterpret_cast<const unsigned char*>(scales);
#pragma unroll
        for (int i = 0; i < UPT; ++i) {
          const int idx = t + i * CHUNK_THREADS, key = idx / (LW / 8), u = idx % (LW / 8);
          const int64_t ki = first + key;
          const bool ok = ki < n;
          const int s = ok ? tok[r * w + ki] : 0;  // a visible position: 0 <= s < 2^31
          const int64_t page = ok ? (int64_t)pt[s >> 9] * pstep : 0;  // CH = 512 rows a page
          const int row = s & (CH - 1);
          raw[i] = ok ? __ldg(reinterpret_cast<const uint2*>(codes8 + page + (row * LW + u * 8))) : make_uint2(0u, 0u);
          rsc[i] = ok ? __ldg(reinterpret_cast<const float*>(scale_bytes + page + row * 4)) : 1.0f;
        }
        return;
      }
    }
#pragma unroll
    for (int i = 0; i < UPT; ++i) {
      const int idx = t + i * CHUNK_THREADS, key = idx / (LW / 8), u = idx % (LW / 8);
      const int64_t ki = first + key;
      const bool ok = sparse ? ki < n : ki <= limit;
      const int64_t src = ok ? (sparse ? (int64_t)tok[r * w + ki] : ki) : 0;
      raw[i] = ok ? __ldg(reinterpret_cast<const uint2*>(codes8 + src * LW + u * 8)) : make_uint2(0u, 0u);
      rsc[i] = ok ? __ldg(scales8 + src) : 1.0f;
    }
  };
  auto store8 = [&](int tile) {
    __nv_bfloat16* dst = kvs + (tile & 1) * KT * LW;
#pragma unroll
    for (int i = 0; i < UPT; ++i) {
      const int idx = t + i * CHUNK_THREADS, key = idx / (LW / 8), u = idx % (LW / 8);
      const uint32_t wv[2] = {raw[i].x, raw[i].y};
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        const uint16_t bits = (uint16_t)((wv[j / 4] >> (8 * (j % 4))) & 0xffffu);  // codes j, j + 1 (little-endian)
        const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3)));
        dst[wide(key, u) + j] = __float2bfloat16_rn(__fmul_rn(v.x, rsc[i]));
        dst[wide(key, u) + j + 1] = __float2bfloat16_rn(__fmul_rn(v.y, rsc[i]));
      }
    }
  };
  auto load_kv = [&](int tile) {
    if constexpr (FP8) {
      fetch8(tile);
      store8(tile);
      return;
    }
    __nv_bfloat16* dst = kvs + (tile & 1) * KT * LW;
    const int64_t first = start + tile * KT;
    for (int idx = t; idx < KT * (LW / 8); idx += blockDim.x) {
      const int key = idx / (LW / 8), u = idx % (LW / 8);
      const int64_t ki = first + key;
      const bool ok = sparse ? ki < n : ki <= limit;
      const int64_t src = ok ? (sparse ? (int64_t)tok[r * w + ki] : ki) : 0;
      cp16(dst + wide(key, u), keys_at + src * LW + u * 8, ok);
    }
  };
  float m[MT][2], l[MT][2], o[MT][8][4];
  for (int i = 0; i < MT; ++i)
    for (int e = 0; e < 2; ++e) m[i][e] = -INFINITY, l[i][e] = 0.0f;
  for (int i = 0; i < MT; ++i)
    for (int j = 0; j < 8; ++j)
      for (int e = 0; e < 4; ++e) o[i][j][e] = 0.0f;
  constexpr int tiles = CH / KT;
  if (live) {
    if (c == 0)  // the row's heads once (the later chunks keep them)
      for (int idx = t; idx < HB * (LW / 8); idx += blockDim.x) {
        const int hd = idx / (LW / 8), u = idx % (LW / 8);
        const bool ok = h0 + hd < heads;
        cp16(qs + wide(hd, u), qa + (r * heads + (ok ? h0 + hd : 0)) * LW + u * 8, ok);
      }
    load_kv(0);
    commit();
    load_kv(1);
    commit();
  }
  for (int tile = 0; live && tile < tiles; ++tile) {
    if constexpr (FP8) {
      wait<0>();
    } else {
      if (tile + 1 < tiles)
        wait<1>();
      else
        wait<0>();
    }
    __syncthreads();
    if constexpr (FP8) {
      if (tile + 2 < tiles) fetch8(tile + 2);  // in flight during this tile's math
    }
    const __nv_bfloat16* kv = kvs + (tile & 1) * KT * LW;
    const int64_t first = start + tile * KT;
    float sc[4];
    if (hh < MT) {
      float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
      for (int kk = 0; kk < LW; kk += 32) {
        uint32_t a0[4], a1[4], b[4];
        ldm4<false>(a0, qs + wide(hh * 16 + (lane & 15), (kk >> 3) + (lane >> 4)));
        ldm4<false>(a1, qs + wide(hh * 16 + (lane & 15), ((kk + 16) >> 3) + (lane >> 4)));
        ldm4<false>(b, kv + wide(kg * 8 + (lane & 7), (kk >> 3) + (lane >> 3)));
        glm_router::mma16816(acc, a0, b);
        glm_router::mma16816(acc, a1, b + 2);
      }
      for (int e = 0; e < 2; ++e) {
        const int64_t ki = first + kg * 8 + q * 2 + e;
        const bool ok = sparse ? ki < n : ki <= limit;
        sc[e] = ok ? __fmul_rn(acc[e], scale) : -INFINITY;
        sc[e + 2] = ok ? __fmul_rn(acc[e + 2], scale) : -INFINITY;
      }
      for (int hf = 0; hf < 2; ++hf) {
        const float mx = quad_max(fmaxf(sc[hf * 2], sc[hf * 2 + 1]));
        if (q == 0) red[(hh * 16 + hf * 8 + g) * 4 + kg] = mx;
      }
    }
    __syncthreads();
    if (t < HB * 4) {
      float v = red[t];
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
      if ((t & 3) == 0) stat[t / 4] = v;
    }
    __syncthreads();
    float alpha[MT][2];
    for (int i = 0; i < MT; ++i)
      for (int hf = 0; hf < 2; ++hf) {
        const float tm = stat[i * 16 + hf * 8 + g];
        const bool active = tm != -INFINITY;
        const float nm = active ? fmaxf(m[i][hf], tm) : m[i][hf];
        alpha[i][hf] = active ? (m[i][hf] == -INFINITY ? 0.0f : glm_triton::exp_approx(m[i][hf] - nm)) : 1.0f;
        if (i == hh) {
          float p[2];
          for (int e = 0; e < 2; ++e) {
            const int64_t ki = first + kg * 8 + q * 2 + e;
            const bool ok = sparse ? ki < n : ki <= limit;
            p[e] = active && ok ? glm_triton::exp_approx(sc[hf * 2 + e] - nm) : 0.0f;
          }
          const int hr = i * 16 + hf * 8 + g, col = kg * 8 + q * 2;
          *reinterpret_cast<__nv_bfloat162*>(ps + narrow(hr, col >> 3) + (col & 7)) = __floats2bfloat162_rn(p[0], p[1]);
          const float s = quad_sum(p[0] + p[1]);
          if (q == 0) red[hr * 4 + kg] = s;
        }
        m[i][hf] = nm;
      }
    __syncthreads();
    if (t < HB * 4) {
      float v = red[t];
      v = v + __shfl_xor_sync(0xffffffffu, v, 2);
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      if ((t & 3) == 0) stat[HB + t / 4] = v;
    }
    for (int i = 0; i < MT; ++i)
      for (int j = 0; j < 8; ++j)
        for (int e = 0; e < 4; ++e) o[i][j][e] = __fmul_rn(o[i][j][e], alpha[i][e / 2]);
#pragma unroll
    for (int k16 = 0; k16 < KT; k16 += 16) {
      uint32_t a[MT][4];
      for (int i = 0; i < MT; ++i) ldm4<false>(a[i], ps + narrow(i * 16 + (lane & 15), (k16 >> 3) + (lane >> 4)));
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        uint32_t b[4];
        ldm4<true>(b, kv + wide(k16 + (lane & 15), ((warp * 64 + j * 8) >> 3) + (lane >> 4)));
        for (int i = 0; i < MT; ++i) {
          glm_router::mma16816(o[i][j], a[i], b);
          glm_router::mma16816(o[i][j + 1], a[i], b + 2);
        }
      }
    }
    __syncthreads();
    for (int i = 0; i < MT; ++i)
      for (int hf = 0; hf < 2; ++hf) l[i][hf] = __fmaf_rn(l[i][hf], alpha[i][hf], stat[HB + i * 16 + hf * 8 + g]);
    if constexpr (FP8) {
      if (tile + 2 < tiles) store8(tile + 2);
    } else if (tile + 2 < tiles) {
      load_kv(tile + 2);
    }
    commit();
  }
  // tf_glm_merge's step for chunk c on this thread's (head, column) elements: its m, l are the chunk's pm, pl
  for (int i = 0; i < MT; ++i)
    for (int hf = 0; hf < 2; ++hf) {
        const float cm = m[i][hf], cl = l[i][hf], was = c == 0 ? -INFINITY : mm[i][hf];
        const bool active = cl > 0.0f;
        const float nm = active ? fmaxf(was, cm) : was;
        const float b = active ? glm_triton::exp_approx(cm - nm) : 0.0f;
        if (c == 0) {
          for (int j = 0; j < 8; ++j)
            for (int e = 0; e < 2; ++e) om[i][j][hf * 2 + e] = __fmaf_rn(b, o[i][j][hf * 2 + e], 0.0f);
          lm[i][hf] = __fmaf_rn(b, cl, 0.0f);
        } else {
          const float a = active ? (was == -INFINITY ? 0.0f : glm_triton::exp_approx(was - nm)) : 1.0f;
          for (int j = 0; j < 8; ++j)
            for (int e = 0; e < 2; ++e)
              om[i][j][hf * 2 + e] = __fmaf_rn(om[i][j][hf * 2 + e], a, __fmul_rn(b, o[i][j][hf * 2 + e]));
          lm[i][hf] = __fmaf_rn(lm[i][hf], a, __fmul_rn(b, cl));
        }
        mm[i][hf] = nm;
      }
  }  // chunks
  for (int i = 0; i < MT; ++i)
    for (int e = 0; e < 4; ++e) {
      const int64_t hd = h0 + i * 16 + g + (e / 2) * 8;
      if (hd >= heads) continue;
      for (int j = 0; j < 8; ++j)
        out[(r * heads + hd) * LW + warp * 64 + j * 8 + q * 2 + e % 2] =
            __float2bfloat16_rn(glm_triton::div_full(om[i][j][e], lm[i][e / 2]));
    }
}

}  // namespace glm_attn

// The 16-head form (decode windows) and 32-head form (prompt chunks), each in glm_attn::wide_smem<HB>() bytes of smem.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok, const int* cnt,
                     float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w, float scale, int sparse,
                     const int* pt, int64_t pstep) {
  glm_attn::chunks_wide<16>(qa, lc, pos, tok, cnt, po, pm, pl, rows, heads, w, scale, sparse, pt, pstep);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok, const int* cnt,
                     float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w, float scale, int sparse,
                     const int* pt, int64_t pstep) {
  glm_attn::chunks_wide<32>(qa, lc, pos, tok, cnt, po, pm, pl, rows, heads, w, scale, sparse, pt, pstep);
}

// E4M3 cache variants reconstruct the existing BF16 shared tile before the unchanged attention math.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16_fp8(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                         const int* tok, const int* cnt, float* po, float* pm, float* pl,
                         int64_t rows, int64_t heads, int64_t w, float scale, int sparse) {
  glm_attn::chunks_wide<16, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), pos, tok, cnt,
                                 po, pm, pl, rows, heads, w, scale, sparse, nullptr, 0, scales);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_fp8(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                         const int* tok, const int* cnt, float* po, float* pm, float* pl,
                         int64_t rows, int64_t heads, int64_t w, float scale, int sparse) {
  glm_attn::chunks_wide<32, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), pos, tok, cnt,
                                 po, pm, pl, rows, heads, w, scale, sparse, nullptr, 0, scales);
}

// The E4M3 variants on KV pages: the stream's table `pt`, its pages `pstep` bytes apart (codes and scales in each).
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16_fp8_paged(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                               const int* tok, const int* cnt, float* po, float* pm, float* pl, int64_t rows,
                               int64_t heads, int64_t w, float scale, int sparse, const int* pt, int64_t pstep) {
  glm_attn::chunks_wide<16, true, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), pos, tok, cnt,
                                       po, pm, pl, rows, heads, w, scale, sparse, pt, pstep, scales);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_fp8_paged(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                               const int* tok, const int* cnt, float* po, float* pm, float* pl, int64_t rows,
                               int64_t heads, int64_t w, float scale, int sparse, const int* pt, int64_t pstep) {
  glm_attn::chunks_wide<32, true, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), pos, tok, cnt,
                                       po, pm, pl, rows, heads, w, scale, sparse, pt, pstep, scales);
}

// A prompt chunk's sparse rows with the merge in the kernel (latent.sparseAttentionFormat when the partials can't hold the
// call): block (row, HB heads), nch chunks a block, out bf16 as tf_glm_merge writes it; the same bytes.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16_fused(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* tok, const int* cnt,
                           __nv_bfloat16* out, int64_t rows, int64_t heads, int64_t w, float scale, int nch,
                           const int* pt, int64_t pstep) {
  glm_attn::chunks_fused<16, false, false>(qa, lc, nullptr, tok, cnt, nullptr, nullptr, nullptr, rows, heads, w,
                                                scale, 1, pt, pstep, nullptr, out, nch);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_fused(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* tok, const int* cnt,
                           __nv_bfloat16* out, int64_t rows, int64_t heads, int64_t w, float scale, int nch,
                           const int* pt, int64_t pstep) {
  glm_attn::chunks_fused<32, false, false>(qa, lc, nullptr, tok, cnt, nullptr, nullptr, nullptr, rows, heads, w,
                                                scale, 1, pt, pstep, nullptr, out, nch);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16_fp8_fused(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* tok,
                               const int* cnt, __nv_bfloat16* out, int64_t rows, int64_t heads, int64_t w, float scale,
                               int nch) {
  glm_attn::chunks_fused<16, true, false>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), nullptr, tok, cnt,
                                               nullptr, nullptr, nullptr, rows, heads, w, scale, 1, nullptr, 0, scales,
                                               out, nch);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_fp8_fused(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* tok,
                               const int* cnt, __nv_bfloat16* out, int64_t rows, int64_t heads, int64_t w, float scale,
                               int nch) {
  glm_attn::chunks_fused<32, true, false>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), nullptr, tok, cnt,
                                               nullptr, nullptr, nullptr, rows, heads, w, scale, 1, nullptr, 0, scales,
                                               out, nch);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16_fp8_paged_fused(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales,
                                     const int* tok, const int* cnt, __nv_bfloat16* out, int64_t rows, int64_t heads,
                                     int64_t w, float scale, int nch, const int* pt, int64_t pstep) {
  glm_attn::chunks_fused<16, true, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), nullptr, tok, cnt,
                                              nullptr, nullptr, nullptr, rows, heads, w, scale, 1, pt, pstep, scales,
                                              out, nch);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_fp8_paged_fused(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales,
                                     const int* tok, const int* cnt, __nv_bfloat16* out, int64_t rows, int64_t heads,
                                     int64_t w, float scale, int nch, const int* pt, int64_t pstep) {
  glm_attn::chunks_fused<32, true, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), nullptr, tok, cnt,
                                              nullptr, nullptr, nullptr, rows, heads, w, scale, 1, pt, pstep, scales,
                                              out, nch);
}

// latent._merge, block (row, head), 128 threads: chunks merged in order, o / l by div.full; skips sparse count-0 rows.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_merge(const float* po, const float* pm, const float* pl, __nv_bfloat16* out, const int* cnt, int64_t rows,
                 int64_t heads, int nch, int sparse) {
  constexpr int LW = glm_attn::LW;
  const int64_t r = blockIdx.x, h = blockIdx.y;
  if (sparse && cnt[r] == 0) return;
  const int k0 = threadIdx.x * 4;
  float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int c = 0; c < nch; ++c) {
    const int64_t base = (c * rows + r) * heads + h;
    const float cm = pm[base], cl = pl[base];
    const bool active = cl > 0.0f;
    const float nm = active ? fmaxf(m, cm) : m;
    const float b = active ? glm_triton::exp_approx(cm - nm) : 0.0f;
    if (c == 0) {
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(b, po[base * LW + k0 + j], 0.0f);
      l = __fmaf_rn(b, cl, 0.0f);
    } else {
      const float a = active ? (m == -INFINITY ? 0.0f : glm_triton::exp_approx(m - nm)) : 1.0f;
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(o[j], a, __fmul_rn(b, po[base * LW + k0 + j]));
      l = __fmaf_rn(l, a, __fmul_rn(b, cl));
    }
    m = nm;
  }
  for (int j = 0; j < 4; ++j) out[(r * heads + h) * LW + k0 + j] = __float2bfloat16_rn(glm_triton::div_full(o[j], l));
}

// tf_glm_merge's code for a segment-batched launch: row r of its segment (the same loop, word for word).
namespace glm_attn {
__device__ __forceinline__ void merge_body(const float* po, const float* pm, const float* pl, __nv_bfloat16* out,
                                           const int* cnt, int64_t rows, int64_t heads, int nch, int sparse, int64_t r) {
  const int64_t h = blockIdx.y;
  if (sparse && cnt[r] == 0) return;
  const int k0 = threadIdx.x * 4;
  float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int c = 0; c < nch; ++c) {
    const int64_t base = (c * rows + r) * heads + h;
    const float cm = pm[base], cl = pl[base];
    const bool active = cl > 0.0f;
    const float nm = active ? fmaxf(m, cm) : m;
    const float b = active ? glm_triton::exp_approx(cm - nm) : 0.0f;
    if (c == 0) {
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(b, po[base * LW + k0 + j], 0.0f);
      l = __fmaf_rn(b, cl, 0.0f);
    } else {
      const float a = active ? (m == -INFINITY ? 0.0f : glm_triton::exp_approx(m - nm)) : 1.0f;
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(o[j], a, __fmul_rn(b, po[base * LW + k0 + j]));
      l = __fmaf_rn(l, a, __fmul_rn(b, cl));
    }
    m = nm;
  }
  for (int j = 0; j < 4; ++j) out[(r * heads + h) * LW + k0 + j] = __float2bfloat16_rn(glm_triton::div_full(o[j], l));
}
}  // namespace glm_attn

// A shared window's latent attention with every segment (stream) in one launch: x runs over the window's rows and
// each block finds its segment; z runs to the most chunks a segment has, and blocks past their segment's chunks leave.
// Each block is the block its segment's own tf_glm_chunks_16_fp8_paged / tf_glm_merge launch runs (its row and chunk
// within the segment; the segment's queries, cache table, position, token lists and partials region), so the bits are
// the per-segment launches'.
namespace glm_attn {
struct AttnSeg {
  const __nv_bfloat16* qa;  // the segment's absorbed queries (its first row)
  const unsigned char* lc;  // the stream's latent cache (on pages: this layer's place in page 0)
  const float* scales;
  const int* pos;           // the stream's position on the device
  const int* tok;           // sparse: the segment's token lists and counts (its first row)
  const int* cnt;
  float* po;                // the segment's own partials region
  float* pm;
  float* pl;
  const int* pt;            // the stream's page table
  int64_t rows;
  int nch;                  // the segment's chunks
  int row0;                 // its first row in the window
};
constexpr int ATTN_SEGS = 16;  // Engine.max_shared
struct AttnSegs {
  AttnSeg s[ATTN_SEGS];
  int n;
};
__device__ __forceinline__ const AttnSeg& seg_of(const AttnSegs& t, int row) {
  int k = 0;
  while (k + 1 < t.n && row >= t.s[k + 1].row0) ++k;
  return t.s[k];
}
}  // namespace glm_attn

extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16_fp8_paged_segs(int64_t heads, int64_t w, float scale, int sparse, int64_t pstep,
                                    const glm_attn::AttnSegs t) {
  const glm_attn::AttnSeg& g = glm_attn::seg_of(t, static_cast<int>(blockIdx.x));
  const int64_t c = blockIdx.z;
  if (c >= g.nch) return;  // past the segment's own grid
  glm_attn::chunks_wide<16, true, true>(g.qa, reinterpret_cast<const __nv_bfloat16*>(g.lc), g.pos, g.tok, g.cnt, g.po,
                                       g.pm, g.pl, g.rows, heads, w, scale, sparse, g.pt, pstep, g.scales,
                                       static_cast<int64_t>(blockIdx.x) - g.row0, c);
}

extern "C" __global__ void __launch_bounds__(128)
    tf_glm_merge_segs(__nv_bfloat16* out, int64_t heads, int sparse, const glm_attn::AttnSegs t) {
  const glm_attn::AttnSeg& g = glm_attn::seg_of(t, static_cast<int>(blockIdx.x));
  glm_attn::merge_body(g.po, g.pm, g.pl, out + static_cast<int64_t>(g.row0) * heads * glm_attn::LW, g.cnt, g.rows,
                       heads, g.nch, sparse, static_cast<int64_t>(blockIdx.x) - g.row0);
}

// sparse._scores, block (row, L pool blocks) of 4 warps, 16 pools a warp: sum_h w relu(dots scale), -inf past visible.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_scores(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride, const __nv_bfloat16* pk,
                  float* out, const int* pos, int64_t np, int64_t nb, float scale, float wscale, int64_t heads,
                  int loop, const int* pt, int64_t ps) {
  constexpr int D = 128, HP = 32, BP = 64;
  __shared__ __align__(16) __nv_bfloat16 sm[4 * 2 * 16 * D];  // q's 32 rows, then each warp's two 16-pool buffers
  auto at = [](int row, int u) { return row * D + ((u ^ (row & 7)) << 3); };
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int64_t r = blockIdx.x, npool = ((int64_t)pos[0] + r + 1) / 4;
  for (int idx = threadIdx.x; idx < HP * (D / 8); idx += 128) {
    const int hh = idx / (D / 8), u = idx % (D / 8);
    const bool ok = hh < heads;
    glm_attn::cp16(sm + at(hh, u), qi + (r * heads + (ok ? hh : 0)) * D + u * 8, ok);
  }
  glm_attn::commit();
  float wv[4];
  for (int i = 0; i < 4; ++i) {
    const int64_t hh = g + 8 * i;
    wv[i] = hh < heads ? glm_triton::fmul(__bfloat162float(wts[r * w_stride + hh]), wscale) : 0.0f;
  }
  // A block's pools for this warp: rows p0 .. p0 + 15 of pk (16 | p0: one page), 16 chunks of 16 bytes, 8 a lane.
  auto stage = [&](int buf, int64_t p0) {
    const __nv_bfloat16* src = pk + glm_pages::at<128>(pt, ps / 2, p0 < npool ? p0 : 0, D);
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int idx = lane + t * 32, row = idx / (D / 8), u = idx % (D / 8);
      const int64_t p = p0 + row;
      const bool ok = p < npool;
      glm_attn::cp16(sm + (warp * 2 + buf) * 16 * D + at(row, u), src + (ok ? row : 0) * D + u * 8, ok);
    }
  };
  const int64_t first = (int64_t)blockIdx.y * loop;
  // Blocks that score: those under nb with visible pools (pb * BP < npool); the rest only write -inf.
  auto live = [&](int j) { const int64_t pb = first + j; return j < loop && pb < nb && pb * BP < npool; };
  glm_attn::wait<0>();
  __syncthreads();
  uint32_t aq[D / 16][2][4];
#pragma unroll
  for (int kq = 0; kq < D / 16; ++kq)
#pragma unroll
    for (int i = 0; i < 2; ++i) glm_attn::ldm4<false>(aq[kq][i], sm + at(i * 16 + (lane & 15), kq * 2 + (lane >> 4)));
  __syncthreads();  // q's fragments are in registers: the key buffers take its place
  if (live(0)) stage(0, first * BP + warp * 16);
  glm_attn::commit();
  for (int j = 0; j < loop; ++j) {
    const int64_t pb = first + j;
    if (pb >= nb) break;
    const int64_t p0 = pb * BP + warp * 16;
    if (!live(j)) {
      for (int e = lane; e < 16; e += 32)
        if (p0 + e < np) out[r * np + p0 + e] = -INFINITY;
      continue;
    }
    if (live(j + 1)) stage((j + 1) & 1, p0 + BP);
    glm_attn::commit();
    glm_attn::wait<1>();
    __syncwarp();
    float acc[2][2][4] = {};
    const __nv_bfloat16* keys = sm + (warp * 2 + (j & 1)) * 16 * D;
#pragma unroll
    for (int kq = 0; kq < D / 16; ++kq) {
      uint32_t b[4];
      glm_attn::ldm4<false>(b, keys + at((lane >> 4) * 8 + (lane & 7), kq * 2 + ((lane >> 3) & 1)));
      for (int nt = 0; nt < 2; ++nt)
        for (int i = 0; i < 2; ++i) glm_router::mma16816(acc[i][nt], aq[kq][i], b + 2 * nt);
    }
    __syncwarp();
    for (int nt = 0; nt < 2; ++nt)
      for (int e = 0; e < 2; ++e) {
        const float v0 = fmaxf(__fmul_rn(acc[0][nt][e], scale), 0.0f),
                    v1 = fmaxf(__fmul_rn(acc[0][nt][e + 2], scale), 0.0f);
        const float v2 = fmaxf(__fmul_rn(acc[1][nt][e], scale), 0.0f),
                    v3 = fmaxf(__fmul_rn(acc[1][nt][e + 2], scale), 0.0f);
        float s = glm_triton::fmul(wv[1], v1);
        s = __fmaf_rn(wv[0], v0, s);
        s = __fmaf_rn(wv[2], v2, s);
        s = __fmaf_rn(wv[3], v3, s);
        for (int o = 16; o >= 4; o >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, o);
        const int64_t p = p0 + nt * 8 + q * 2 + e;
        if (g == 0 && p < np) out[r * np + p] = p < npool ? s : -INFINITY;
      }
  }
}
