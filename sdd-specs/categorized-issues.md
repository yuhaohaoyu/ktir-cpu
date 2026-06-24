# ktir-cpu Open Issues (without open PRs addressing them)

> Generated 2026-06-24 from [torch-spyre/ktir-cpu](https://github.com/torch-spyre/ktir-cpu/issues)

Issues already addressed by open PRs (excluded): #74 ← PR #80, #87 ← PR #137, #91 ← PR #132, #99 ← PR #111, #128 ← PR #117

---

## Tier 1 — Critical

High importance, well-scoped issues that block real workloads or produce incorrect results.

| &emsp;#&emsp; | Title | &emsp;&emsp;Cat&nbsp;1&emsp;&emsp; | &emsp;&emsp;Cat&nbsp;2&emsp;&emsp; | &emsp;Importance&emsp; | &emsp;Rigor&emsp; | Rationale |
|:---:|-------|:---:|:---:|:---:|:---:|-----------|
| [#101](https://github.com/torch-spyre/ktir-cpu/issues/101) | construct_memory_view: base_ptr treated as stick index instead of element index | 2) Distributed Memory | — | High | High | Any kernel computing its own pointer silently reads from the wrong HBM location. |
| [#25](https://github.com/torch-spyre/ktir-cpu/issues/25) | Fix cross-core communication: ktdp.reduce and ktdp.transfer correctness | 2) Distributed Memory | 5) Executor | High | High | BSP-replay execution produces incorrect results for every multi-core kernel using inter-core comm. |
| [#139](https://github.com/torch-spyre/ktir-cpu/issues/139) | mlir_frontend path missing outs_operands and use_counts *(use_counts partially fixed by PR #140; outs_operands still open)* | 4) Parsing | 5) Executor | High | High | MLIRFrontendParser users get incorrect LX accounting — accumulators double-charged, tiles never freed. |
| [#122](https://github.com/torch-spyre/ktir-cpu/issues/122) | roofline() reports the wrong dominant compute unit for mixed systolic/SIMD kernels | 3) Perf Modeling | — | High | High | Roofline picks wrong unit and wrong ceiling for any kernel mixing systolic and SIMD work. |
| [#119](https://github.com/torch-spyre/ktir-cpu/issues/119) | Support for `linalg.matmul_transpose_b` vs matmul with an indexing map | 4) Parsing | 5) Executor | High | Medium | Required for LLM forward passes — without it, transposed-B weights must be materialized at load or forward time. |
| [#2](https://github.com/torch-spyre/ktir-cpu/issues/2) | [RFC conformance] Add RFC-explicit non-ktdp ops | 6) Completeness | — | High | Medium | Blocks running any RFC example file end-to-end; the remaining ops are enumerated. |
| [#3](https://github.com/torch-spyre/ktir-cpu/issues/3) | Transformer decoder op coverage: gap analysis and MLIR examples | 6) Completeness | — | High | Medium | No workload-driven coverage of a full decoder sub-graph — individual op tests pass but integration gaps remain invisible. |

---

## Tier 2 — Important

Medium-to-high importance with good rigor: meaningful improvements to correctness, performance modeling, or infrastructure.

| &emsp;#&emsp; | Title | &emsp;&emsp;Cat&nbsp;1&emsp;&emsp; | &emsp;&emsp;Cat&nbsp;2&emsp;&emsp; | &emsp;Importance&emsp; | &emsp;Rigor&emsp; | Rationale |
|:---:|-------|:---:|:---:|:---:|:---:|-----------|
| [#76](https://github.com/torch-spyre/ktir-cpu/issues/76) | linalg.reduce latency: charges output element count instead of deferring to combiner region | 3) Perf Modeling | 5) Executor | Medium | High | Double-counts cycles for every reduce op — combiner region already fires, then output elements are charged on top. |
| [#56](https://github.com/torch-spyre/ktir-cpu/issues/56) | MemoryOps contract undefined for non-rectangular variables_space_set | 1) Indirect Access | — | Medium | High | ValueError crash when vss enumerates fewer points than shape.prod() — no graceful path. |
| [#67](https://github.com/torch-spyre/ktir-cpu/issues/67) | read_scattered cross-allocation safety: hard-guard the caller invariant | 1) Indirect Access | — | Medium | High | Silent memory corruption if scattered read spans two allocations; only a docstring guards today. |
| [#102](https://github.com/torch-spyre/ktir-cpu/issues/102) | HBM/LX memory bootstrap: explicit placement API and allocation-aware next_ptr | 2) Distributed Memory | — | Medium | High | No way to place tensors at specific addresses — tests are brittle and silent aliasing is possible. |
| [#108](https://github.com/torch-spyre/ktir-cpu/issues/108) | comm: add efficient ReduceBackend variants + attribute-based dispatch | 2) Distributed Memory | — | Medium | High | CW-only ring models worst case; real hardware is bidirectional, so latency estimates are 2x pessimistic. |
| [#31](https://github.com/torch-spyre/ktir-cpu/issues/31) | HBM memory model: cross-load stick coalescing not modeled | 2) Distributed Memory | 3) Perf Modeling | Medium | High | Adjacent loads sharing a stick are charged independently — latency model over-reports for dense access patterns. |
| [#127](https://github.com/torch-spyre/ktir-cpu/issues/127) | roofline(): add a time-based core-occupancy metric (true SM Active %) | 3) Perf Modeling | — | Medium | High | Binary grid_coverage masks barely-active cores; no way to distinguish full utilization from minimal dispatch. |
| [#126](https://github.com/torch-spyre/ktir-cpu/issues/126) | latency_demo notebook: surface cross-core reduce (communication) cost | 3) Perf Modeling | 2) Distributed Memory | Medium | High | Comm latency is computed but invisible in the analysis surface — users cannot see where comm dominates. |
| [#125](https://github.com/torch-spyre/ktir-cpu/issues/125) | chip-level HBM bandwidth throughput | 3) Perf Modeling | — | Medium | High | No bandwidth-axis dual of chip_throughput — memory-bound kernels lack a chip-level efficiency metric. |
| [#92](https://github.com/torch-spyre/ktir-cpu/issues/92) | test_ops.py: incomplete dtype coverage for arithmetic and math ops | 6) Completeness | — | Medium | High | `_tile` helper ignores dtype arg — all tests secretly run float16 regardless of declared dtype. |
| [#88](https://github.com/torch-spyre/ktir-cpu/issues/88) | ktdp.reduce / ktdp.coreid are non-spec ops in the regex executor | 5) Executor | 6) Completeness | Medium | High | Executor registers handlers for ops that don't exist in the authoritative dialect — spec drift. |
| [#98](https://github.com/torch-spyre/ktir-cpu/issues/98) | Vectorized indirect coord build, reusing one subscript dispatch | 1) Indirect Access | — | Medium | High | Reverted fast path had 2x duplicate logic; need to reintroduce without the second copy of subscript handling. |
| [#97](https://github.com/torch-spyre/ktir-cpu/issues/97) | MemoryOps.load/store accept ndarray coords to skip tolist()/tuple() conversion | 1) Indirect Access | — | Medium | High | Python tolist()/tuple() conversion is the dominant overhead once vectorized coord build is reinstated. |
| [#96](https://github.com/torch-spyre/ktir-cpu/issues/96) | Preserve BoxSet structure under permuted variables_space_order | 1) Indirect Access | — | Medium | High | Permuted box falls back to O(n) point enumeration even though a permuted box is still a box. |
| [#95](https://github.com/torch-spyre/ktir-cpu/issues/95) | Strided sub-rectangle fast path in MemoryOps.load/store | 1) Indirect Access | — | Medium | High | Sub-tile loads that are strided but rectangular miss the contiguous fast path and fall to element-wise. |
| [#40](https://github.com/torch-spyre/ktir-cpu/issues/40) | @register should accept an optional attributes schema | 4) Parsing | — | Medium | High | Attributes are dict[str, Any] with no per-op contract — consumers assuming int silently break on str entries. |
| [#7](https://github.com/torch-spyre/ktir-cpu/issues/7) | [CI] Install ktir-mlir-frontend Python wheels | 6) Completeness | — | Medium | High | Frontend adapter tests auto-skip in CI — regressions in the MLIR path are invisible until manual local runs. |
| [#94](https://github.com/torch-spyre/ktir-cpu/issues/94) | Add latency estimation demo notebook | 3) Perf Modeling | — | Medium | High | The primary analysis surface for cycle-approximate latency has no committed, tested notebook in the repo. |
| [#78](https://github.com/torch-spyre/ktir-cpu/issues/78) | Parallel GridExecutor to speed up multi-core simulation | 5) Executor | 2) Distributed Memory | Medium | Medium | Simulation wall-clock scales linearly with core count even for independent cores; 32-core runs are impractical. |
| [#77](https://github.com/torch-spyre/ktir-cpu/issues/77) | Kernel test plan: simple feedforward (FFN / SwiGLU) | 6) Completeness | — | Medium | Medium | No realistic sub-graph regression test — op-level tests pass but interaction bugs across a real workload are invisible. |
| [#48](https://github.com/torch-spyre/ktir-cpu/issues/48) | Potential issue with #digits being excluded by dialect-specific operand regexes | 4) Parsing | — | Medium | Medium | Dialect parsers using `%\w+` would silently truncate `%base#N` to `%base` — needs audit to confirm scope. |
| [#123](https://github.com/torch-spyre/ktir-cpu/issues/123) | [Tracking] latency_demo notebook — roofline improvements | 3) Perf Modeling | — | Medium | Medium | Umbrella tracking issue coordinating roofline correctness, metrics, and demo coverage work. |
| [#53](https://github.com/torch-spyre/ktir-cpu/issues/53) | Should construct_indirect_access_tile compose with DistributedMemRef? | 1) Indirect Access | 2) Distributed Memory | Medium | Low | Architectural decision blocking the indirect+distributed composition — no implementation path chosen yet. |

---

## Tier 3 — Backlog

Lower importance, currently harmless, open design questions, or incremental coverage improvements.

| &emsp;#&emsp; | Title | &emsp;&emsp;Cat&nbsp;1&emsp;&emsp; | &emsp;&emsp;Cat&nbsp;2&emsp;&emsp; | &emsp;Importance&emsp; | &emsp;Rigor&emsp; | Rationale |
|:---:|-------|:---:|:---:|:---:|:---:|-----------|
| [#85](https://github.com/torch-spyre/ktir-cpu/issues/85) | linalg.reduce: multi-axis reduce and outs-init not handled per MLIR semantics | 5) Executor | 6) Completeness | Low | High | Diverges from MLIR spec but currently harmless — frontend never emits multi-axis reduce or non-zero init. |
| [#75](https://github.com/torch-spyre/ktir-cpu/issues/75) | Fix misleading linalg.matmul comment in LatencyEstimator._estimate | 3) Perf Modeling | — | Low | High | Comment says "no HBM traffic" as if verified — real reason is systolic ops are modeled compute-only by design. |
| [#55](https://github.com/torch-spyre/ktir-cpu/issues/55) | Add distributed memory view examples and test_examples coverage | 2) Distributed Memory | 6) Completeness | Low | High | Existing distributed-view-copy.mlir is not wired into test_examples.py — no automated frontend coverage. |
| [#47](https://github.com/torch-spyre/ktir-cpu/issues/47) | Add more dynamic-shape MLIR examples (varying dtypes, 2-D, multi-core) | 6) Completeness | — | Low | High | Only one minimal single-core dynamic example exists — dtype/rank/multi-core variants untested. |
| [#68](https://github.com/torch-spyre/ktir-cpu/issues/68) | [Question] cso direct-path: pointwise or sort-then-gather | 1) Indirect Access | — | Low | Low | Open design question on whether direct-path cso should match indirect-path sort-then-gather semantics. |
| [#60](https://github.com/torch-spyre/ktir-cpu/issues/60) | [Question] Coordinate collision semantics for indirect_store / scatter | 1) Indirect Access | — | Low | Low | Last-writer-wins works in practice; question is whether to spec deterministic or undefined behavior. |
| [#28](https://github.com/torch-spyre/ktir-cpu/issues/28) | [Question] Out-of-Set Element Semantics for ktdp.load / ktdp.store | 1) Indirect Access | — | Low | Low | Out-of-set positions are zero-filled on load and ignored on store — works, but not formally spec'd. |

---

## Category Legend

| &emsp;#&emsp; | &emsp;&emsp;&emsp;&emsp;Category&emsp;&emsp;&emsp;&emsp; |
|:---:|:---:|
| 1 | Indirect Access |
| 2 | Distributed Memory |
| 3 | Performance Modeling and Projection |
| 4 | Parsing |
| 5 | Executor |
| 6 | Completeness and Others |

---

## Summary

| &emsp;&emsp;&emsp;Tier&emsp;&emsp;&emsp; | &emsp;Count&emsp; | Profile |
|:---:|:---:|-----------|
| Tier 1 — Critical | 7 | High importance; blocks workloads or produces wrong results |
| Tier 2 — Important | 23 | Meaningful correctness, perf, or infra improvements |
| Tier 3 — Backlog | 7 | Currently harmless, incremental, or open design questions |
| **Total** | **32** | **37 open issues, 5 already have PRs** |

---

Note: PR #138 (open) is related to #122 and #126 but tackles a narrower fix (excluding comm bytes from roofline denominator) rather than fully resolving either.
