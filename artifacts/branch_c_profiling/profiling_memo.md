# Profiling Memo: IsoQuant NPT=8 Fused Kernel Bottlenecks

**Date:** April 27, 2026
**Model:** Qwen3.6-35B-A3B-nvfp4 (D=256)
**Context Lengths:** 4K, 8K (Tiled Path)

## 1. Executive Summary

Profiling of the fused NPT=8 IsoQuant decode kernel reveals a **2.9x latency gap** vs. default KV at 8K context (36.8ms vs 12.5ms). The bottleneck is primarily in the **KV write path (compression and packing)** and the **merge/rotation overhead** on the read path, rather than the Metal kernel ALU performance itself.

## 2. Gap Attribution (T=8192)

| Component | Avg ms/step (All Layers) | % of Instrumented Total | Gap Attribution* |
|-----------|--------------------------|-------------------------|------------------|
| **Total (Unpatched)** | **36.76 ms** | - | **100%** |
| **Default Baseline** | **12.51 ms** | - | - |
| **Gap** | **24.25 ms** | - | - |
| `compression_and_packing` | 31.55 ms | 83.0% | ~125% |
| `metal_kernel` | 8.68 ms | 22.8% | ~35% |
| `inverse_rotation` | 8.12 ms | 21.3% | ~32% |
| `fa2_merge` | 5.40 ms | 14.2% | ~22% |
| `query_rotation` | 2.51 ms | 6.6% | ~10% |

*\*Attributions sum to >100% due to instrumentation overhead (sync fences across 64+ layers).*

## 3. Analysis

1.  **Write Path Dominance:** The `compression_and_packing` component accounts for the majority of the overhead. Every decode step requires compressing the new key/value and packing them into 3-bit storage. This O(L) overhead (where L is layers) is currently the "floor" of IsoQuant latency.
2.  **Kernel vs. Merge:** The tiled Metal kernel dispatch (`metal_kernel`) is only ~15% of the total step time. The combined overhead of the Python-side `fa2_merge` and the `inverse_rotation` (13.5ms total) is nearly double the time spent in the GPU kernel itself.
3.  **ALU vs. BW:** While the kernel is efficient, the high cost of inverse rotation suggests that the SO(4) math, even if fused, is adding significant latency when implemented as a separate step.

## 4. Recommendations

1.  **Optimize Packing:** The 3-bit packing is currently a bottleneck. Investigate moving packing into a fused kernel or using a more cache-friendly bit-packing representation.
2.  **Fuse Inverse Rotation:** The `inverse_rotation` is currently a separate call after the FA2 merge. Fusing the inverse SO(4) rotation directly into the T-tiled Metal kernel's final reduction would eliminate ~8ms of latency.
3.  **C++ Merge:** Moving the FA2 tile merge from Python/MLX-ops to a dedicated C++ or Metal reduction would eliminate ~5ms of overhead.

## 5. Next Steps

*   **Task 4 (Instruments):** Confirm if `metal_kernel` is memory-bound or ALU-bound (suspected memory-bound due to 3-bit unpacking).
*   **Decision:** Proceed with **Phase 6: Fused Representation Redesign**, focusing on fusing the inverse rotation and optimizing the packing path.
