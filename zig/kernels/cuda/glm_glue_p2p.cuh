// Two ranks on one host: a small all-gather as one-shot stores into the peer's memory (cuda/p2p_gather.zig).
#pragma once

#include <stdint.h>

#include "glm_triton.cuh"

namespace glm_p2p {

constexpr int kMaxBlocks = 32;  // p2p_gather.zig max_blocks: a flag line per block and parity
constexpr int kFlagWords = 32;  // one 128-byte line a flag

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_volatile(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_volatile(uint32_t* p, uint32_t v) {
  asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint64_t now_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

}  // namespace glm_p2p

// all = [rank 0's packs][rank 1's]: block b stores its share of `mine` in the peer's inbox (parity epoch & 1),
// releases its flag there, acquires its own flag, then copies the peer's share from its inbox to all[peer].
// A rank reaches epoch e + 2 only after the peer's e + 1 flag, so a parity's slot is never written while read.
// A wait past timeout_ns or a raised abort word (status[0]) writes the epoch to status[1]; the host checks it.
// Dependent launch: the gather waits for the grid that wrote `mine` (and so for every earlier one) before it touches
// any memory, then lets the next grid launch, whose weights stream in during the exchange; that grid still waits for
// this one to complete before it reads `all`. Flags, parities and abort handling are unchanged.
extern "C" __global__ void __launch_bounds__(512) tf_glm_p2p_gather(const uint4* mine, uint4* all, int64_t n16,
                                                                     int rank, int copy_mine, uint4* peer_inbox,
                                                                     uint32_t* peer_flags, const uint4* my_inbox,
                                                                     const uint32_t* my_flags, int64_t slot16,
                                                                     uint32_t epoch, uint64_t timeout_ns,
                                                                     uint32_t* status) {
  using namespace glm_p2p;
  glm_triton::pdl_wait();
  glm_triton::pdl_trigger();
  const int b = blockIdx.x;
  const int64_t per = (n16 + gridDim.x - 1) / gridDim.x;
  const int64_t lo = min(n16, (int64_t)b * per), hi = min(n16, lo + per);
  const int par = epoch & 1;
  uint4* to = peer_inbox + par * slot16;
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) to[i] = mine[i];
  if (copy_mine)
    for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) all[rank * n16 + i] = mine[i];
  __shared__ int ok;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_release_sys(peer_flags + (par * kMaxBlocks + b) * kFlagWords, epoch);
    const uint32_t* f = my_flags + (par * kMaxBlocks + b) * kFlagWords;
    int good = 1;
    uint64_t t0 = 0;
    for (uint32_t spins = 1; (int32_t)(ld_acquire_sys(f) - epoch) < 0; ++spins) {
      if ((spins & 1023) != 0) continue;
      const uint64_t t = now_ns();
      if (t0 == 0) t0 = t;
      if (t - t0 > timeout_ns || ld_volatile(status) != 0) {
        good = 0;
        break;
      }
    }
    if (!good) st_volatile(status + 1, epoch);
    ok = good;
  }
  __syncthreads();
  if (!ok) return;
  const uint4* from = my_inbox + par * slot16;
  uint4* out = all + (1 - rank) * n16;
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) out[i] = __ldcg(from + i);
}
