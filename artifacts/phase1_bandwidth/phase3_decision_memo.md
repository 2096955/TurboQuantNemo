# Phase 1 -> Phase 3 Priority Decision

Tiled V-accum bandwidth at T={1024, 2048, 4096, 8192, 16384}: see `tiled_v_accum_bw.json`.

Peak BW assumption: 300 GB/s realistic for this M4 Max access pattern.

## Measured

| T | ms/call | best-case GB/s | best % peak | worst-case GB/s |
|---|---:|---:|---:|---:|
| 1024 | 0.526 | 1.04 | 0.3% | 26.44 |
| 2048 | 0.366 | 2.95 | 1.0% | 150.40 |
| 4096 | 0.358 | 6.00 | 2.0% | 612.28 |
| 8192 | 0.401 | 10.66 | 3.6% | 2179.72 |
| 16384 | 0.519 | 16.43 | 5.5% | 6725.57 |

The best-case model assumes full cache reuse across query heads sharing a KV head. Even under that optimistic model, the maximum achieved bandwidth is 16.43 GB/s, or 5.5% of peak. The worst-case model intentionally over-counts DRAM traffic and exceeds hardware peak at large T, which confirms that caching is already amortizing much of the apparent re-read traffic.

## Decision

Decision per Phase 1 rubric:

- >= 60% of peak: traffic-reduction path -> fusion gives upper-end gain. Proceed Phase 3 as planned.
- 20-60%: mixed dispatch + traffic. Proceed; expect mid-range gain.
- < 20%: instruction-level waste. Investigate scattered centroid-load patterns first; consider a software-pipelined Kernel C variant before fusing.

Conclusion: **<20% branch**. The tiled V-accum kernel is not memory-throughput-limited under the best-case cache-reuse model. Phase 3's traffic-reduction upside is likely smaller than the optimistic forecast; before committing to the full NPT=8 fused-kernel implementation, investigate centroid-load patterns, dispatch overhead, and software-pipelined Kernel C variants.

**Effect on the written plan’s fusion forecast:** Phase 1 **does not** support the plan’s original premise that V-accum is **bandwidth-limited** in a way that makes **traffic-reduction fusion** the primary lever. The plan’s “optimistic / conservative ms ranges” for Phase 3 should be treated as **unvalidated** until re-measured after Phase 2. A fused NPT=8 kernel may still reduce **dispatch and inter-kernel sync**; that is a **different** hypothesis from roofline / peak-GB/s traffic reduction.

## Recommendation

Proceed to Phase 2 immediately. The 8K attribution shows `pack_indices_3bit` at roughly 11.8 ms/step, making it the largest IsoQuant-attributable cost and a cleaner target than Phase 3 fusion. After Phase 2, rerun attribution; if pack cost drops as expected, use the new residual profile to decide whether Phase 3 still clears the value bar.
