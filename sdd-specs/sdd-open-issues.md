# SDD: ktir-cpu Open Issue Landscape

| Field | Value |
|-------|-------|
| **Document** | System Design Document — Issue Triage & Dependency Map |
| **Repository** | [torch-spyre/ktir-cpu](https://github.com/torch-spyre/ktir-cpu) |
| **Date** | 2026-06-24 |
| **Status** | Living document |
| **Scope** | All open issues without an open PR directly addressing them |

---

## 1. Purpose

This document categorizes and prioritizes the open issue backlog for `ktir-cpu`, identifies inter-issue dependencies, and groups related work into coherent execution clusters. It serves as the basis for sprint planning and contributor onboarding.

---

## 2. Methodology

1. Pulled all open issues via `gh issue list --state open`.
2. Pulled all open PRs and matched `Fixes #N` / `Closes #N` references.
3. Excluded issues already under active PR coverage.
4. Categorized each issue into one or two of six domain categories.
5. Scored **Importance** (does it block workloads or produce wrong results?) and **Rigor** (is the scope well-defined and actionable?).
6. Tiered by combined score and grouped correlated issues for adjacent execution.

---

## 3. Exclusions (Issues with Open PRs)

| Issue | PR | Status |
|:-----:|:--:|--------|
| #74 | PR #80 | e2e symbolic-BoxSet test |
| #87 | PR #137 | Frontend parser distributed memory views |
| #91 | PR #132 | Bundled multi-result LHS parsing |
| #99 | PR #111 | Dynamic dims in distributed view result type |
| #128 | PR #117 | Rust port |

---

## 4. Categories

| ID | Category | Description |
|:--:|----------|-------------|
| 1 | Indirect Access | Coordinate-set enumeration, indirect load/store, gather/scatter primitives |
| 2 | Distributed Memory | HBM/LX model, memory views, cross-core communication |
| 3 | Performance Modeling | Latency estimation, roofline analysis, throughput metrics |
| 4 | Parsing | Regex and MLIR frontend parsers, op registration |
| 5 | Executor | Interpreter dispatch, grid execution, op handler correctness |
| 6 | Completeness | RFC conformance, test coverage, CI, documentation |

---

## 5. Tiered Issue Map

### 5.1 Tier 1 — Critical

> High importance, well-scoped. Blocks real workloads or produces incorrect results.

#### Cluster A: Distributed Memory Correctness

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #101 | construct_memory_view: base_ptr treated as stick index instead of element index | 2 | — | High | High | Any kernel computing its own pointer silently reads from the wrong HBM location. Related to: #102 (placement API follows up on this fix). |
| #25 | Fix cross-core communication: ktdp.reduce and ktdp.transfer correctness | 2 | 5 | High | High | BSP-replay execution produces incorrect results for every multi-core kernel using inter-core comm. Related to: #108 (efficient reduce backends), #126 (comm cost visibility). |

#### Cluster B: Frontend / Parsing Blockers

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #139 | mlir_frontend path missing outs_operands and use_counts | 4 | 5 | High | High | MLIRFrontendParser users get incorrect LX accounting — accumulators double-charged, tiles never freed. *use_counts partially fixed by PR #140; outs_operands still open.* Related to: #119 (both block real LLM workloads on the frontend path). |
| #119 | Support for `linalg.matmul_transpose_b` vs matmul with an indexing map | 4 | 5 | High | Medium | Required for LLM forward passes — without it, transposed-B weights must be materialized at load or forward time. Related to: #139 (both block frontend LLM execution), #3 (decoder coverage). |

#### Cluster C: Roofline Correctness

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #122 | roofline() reports the wrong dominant compute unit for mixed systolic/SIMD kernels | 3 | — | High | High | Roofline picks wrong unit and wrong ceiling for any kernel mixing systolic and SIMD work. Related to: #123 (tracking), #127 (occupancy metric), #125 (bandwidth dual). |

#### Cluster D: Op / Workload Completeness

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #2 | [RFC conformance] Add RFC-explicit non-ktdp ops | 6 | — | High | Medium | Blocks running any RFC example file end-to-end; the remaining ops are enumerated. Related to: #3 (decoder workload needs these ops). |
| #3 | Transformer decoder op coverage: gap analysis and MLIR examples | 6 | — | High | Medium | No workload-driven coverage of a full decoder sub-graph — individual op tests pass but integration gaps remain invisible. Related to: #2 (RFC ops prerequisite), #77 (FFN test plan), #119 (transpose_b needed). |

---

### 5.2 Tier 2 — Important

> Medium-to-high importance with good rigor. Meaningful improvements to correctness, performance modeling, or infrastructure.

#### Cluster E: Indirect Access — Safety Hardening

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #56 | MemoryOps contract undefined for non-rectangular variables_space_set | 1 | — | Medium | High | ValueError crash when vss enumerates fewer points than shape.prod() — no graceful path. Related to: #67 (both are MemoryOps contract gaps). |
| #67 | read_scattered cross-allocation safety: hard-guard the caller invariant | 1 | — | Medium | High | Silent memory corruption if scattered read spans two allocations; only a docstring guards today. Related to: #56 (both are MemoryOps safety hardening). |

#### Cluster F: Indirect Access — Performance Chain (split from PR #52)

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #95 | Strided sub-rectangle fast path in MemoryOps.load/store | 1 | — | Medium | High | Sub-tile loads that are strided but rectangular miss the contiguous fast path and fall to element-wise. Related to: #96, #97, #98 (perf chain — each unblocks the next). |
| #96 | Preserve BoxSet structure under permuted variables_space_order | 1 | — | Medium | High | Permuted box falls back to O(n) point enumeration even though a permuted box is still a box. Related to: #95, #97, #98 (perf chain from PR #52). |
| #97 | MemoryOps.load/store accept ndarray coords to skip tolist()/tuple() conversion | 1 | — | Medium | High | Python tolist()/tuple() conversion is the dominant overhead once vectorized coord build is reinstated. Related to: #95, #96, #98 (perf chain from PR #52). |
| #98 | Vectorized indirect coord build, reusing one subscript dispatch | 1 | — | Medium | High | Reverted fast path had 2x duplicate logic; need to reintroduce without the second copy of subscript handling. Related to: #95, #96, #97 (perf chain capstone). |
| #53 | Should construct_indirect_access_tile compose with DistributedMemRef? | 1 | 2 | Medium | Low | Architectural decision blocking the indirect+distributed composition — no implementation path chosen yet. Related to: #95–#98 (indirect perf), #102 (distributed placement). |

#### Cluster G: Distributed Memory Model

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #102 | HBM/LX memory bootstrap: explicit placement API and allocation-aware next_ptr | 2 | — | Medium | High | No way to place tensors at specific addresses — tests are brittle and silent aliasing is possible. Related to: #101 (follows up on the base_ptr fix), #31 (memory model accuracy). |
| #31 | HBM memory model: cross-load stick coalescing not modeled | 2 | 3 | Medium | High | Adjacent loads sharing a stick are charged independently — latency model over-reports for dense access patterns. Related to: #102 (memory model), #76 (latency accuracy). |
| #108 | comm: add efficient ReduceBackend variants + attribute-based dispatch | 2 | — | Medium | High | CW-only ring models worst case; real hardware is bidirectional, so latency estimates are 2x pessimistic. Related to: #25 (comm correctness prerequisite), #126 (comm visibility). |

#### Cluster H: Performance Modeling & Roofline

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #123 | [Tracking] latency_demo notebook — roofline improvements | 3 | — | Medium | Medium | Umbrella tracking issue coordinating roofline correctness, metrics, and demo coverage work. Related to: #122, #127, #125, #126, #94 (all are children or siblings). |
| #127 | roofline(): add a time-based core-occupancy metric (true SM Active %) | 3 | — | Medium | High | Binary grid_coverage masks barely-active cores; no way to distinguish full utilization from minimal dispatch. Related to: #122 (roofline correctness), #123 (tracking). |
| #125 | chip-level HBM bandwidth throughput | 3 | — | Medium | High | No bandwidth-axis dual of chip_throughput — memory-bound kernels lack a chip-level efficiency metric. Related to: #122 (roofline gaps), #123 (tracking). |
| #126 | latency_demo notebook: surface cross-core reduce (communication) cost | 3 | 2 | Medium | High | Comm latency is computed but invisible in the analysis surface — users cannot see where comm dominates. Related to: #25 (comm correctness), #108 (reduce backends), #123 (tracking). |
| #94 | Add latency estimation demo notebook | 3 | — | Medium | High | The primary analysis surface for cycle-approximate latency has no committed, tested notebook in the repo. Related to: #123 (tracking), #126 (comm view), #76 (reduce latency). |
| #76 | linalg.reduce latency: charges output element count instead of deferring to combiner region | 3 | 5 | Medium | High | Double-counts cycles for every reduce op — combiner region already fires, then output elements are charged on top. Related to: #85 (linalg.reduce semantics), #94 (notebook would expose this). |

#### Cluster I: Parsing Robustness

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #40 | @register should accept an optional attributes schema | 4 | — | Medium | High | Attributes are dict[str, Any] with no per-op contract — consumers assuming int silently break on str entries. Related to: #48 (both are parser robustness gaps). |
| #48 | Potential issue with #digits being excluded by dialect-specific operand regexes | 4 | — | Medium | Medium | Dialect parsers using `%\w+` would silently truncate `%base#N` to `%base` — needs audit to confirm scope. Related to: #40 (parser robustness), #139 (frontend path issues). |

#### Cluster J: Executor

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #88 | ktdp.reduce / ktdp.coreid are non-spec ops in the regex executor | 5 | 6 | Medium | High | Executor registers handlers for ops that don't exist in the authoritative dialect — spec drift. Related to: #25 (ktdp.reduce correctness), #2 (RFC conformance). |
| #78 | Parallel GridExecutor to speed up multi-core simulation | 5 | 2 | Medium | Medium | Simulation wall-clock scales linearly with core count even for independent cores; 32-core runs are impractical. Related to: #25 (multi-core correctness should land first). |

#### Cluster K: Test Coverage & CI

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #92 | test_ops.py: incomplete dtype coverage for arithmetic and math ops | 6 | — | Medium | High | `_tile` helper ignores dtype arg — all tests secretly run float16 regardless of declared dtype. Related to: #77 (both are test coverage gaps). |
| #77 | Kernel test plan: simple feedforward (FFN / SwiGLU) | 6 | — | Medium | Medium | No realistic sub-graph regression test — op-level tests pass but interaction bugs across a real workload are invisible. Related to: #3 (decoder coverage), #92 (op test gaps). |
| #7 | [CI] Install ktir-mlir-frontend Python wheels | 6 | — | Medium | High | Frontend adapter tests auto-skip in CI — regressions in the MLIR path are invisible until manual local runs. Related to: #139 (frontend path bugs stay hidden without CI). |

---

### 5.3 Tier 3 — Backlog

> Lower importance, currently harmless, open design questions, or incremental coverage improvements.

#### Cluster L: Spec Divergence (harmless today)

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #85 | linalg.reduce: multi-axis reduce and outs-init not handled per MLIR semantics | 5 | 6 | Low | High | Diverges from MLIR spec but currently harmless — frontend never emits multi-axis reduce or non-zero init. Related to: #76 (linalg.reduce latency bug). |
| #75 | Fix misleading linalg.matmul comment in LatencyEstimator._estimate | 3 | — | Low | High | Comment says "no HBM traffic" as if verified — real reason is systolic ops are modeled compute-only by design. Related to: #122 (roofline correctness context). |

#### Cluster M: Example & Test Coverage

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #55 | Add distributed memory view examples and test_examples coverage | 2 | 6 | Low | High | Existing distributed-view-copy.mlir is not wired into test_examples.py — no automated frontend coverage. Related to: #47 (both are example/test coverage), #7 (CI for frontend). |
| #47 | Add more dynamic-shape MLIR examples (varying dtypes, 2-D, multi-core) | 6 | — | Low | High | Only one minimal single-core dynamic example exists — dtype/rank/multi-core variants untested. Related to: #55 (both are example coverage), #92 (dtype gap). |

#### Cluster N: Open Design Questions (indirect access semantics)

| # | Title | Cat 1 | Cat 2 | Imp. | Rigor | Rationale |
|:-:|-------|:-----:|:-----:|:----:|:-----:|-----------|
| #68 | [Question] cso direct-path: pointwise or sort-then-gather | 1 | — | Low | Low | Open design question on whether direct-path cso should match indirect-path sort-then-gather semantics. Related to: #60, #28 (all are indirect-access semantic questions). |
| #60 | [Question] Coordinate collision semantics for indirect_store / scatter | 1 | — | Low | Low | Last-writer-wins works in practice; question is whether to spec deterministic or undefined behavior. Related to: #68, #28 (all are indirect-access semantic questions). |
| #28 | [Question] Out-of-Set Element Semantics for ktdp.load / ktdp.store | 1 | — | Low | Low | Out-of-set positions are zero-filled on load and ignored on store — works, but not formally spec'd. Related to: #68, #60 (all are indirect-access semantic questions). |

---

## 6. Dependency Graph (critical paths)

```
Legend:  ──▶ "should land before"    ···▶ "informs design of"

Tier 1 foundations
──────────────────
#101 (base_ptr bug) ──▶ #102 (placement API) ──▶ #31 (coalescing model)
#25  (comm correctness) ──▶ #108 (efficient backends) ──▶ #126 (comm visibility)
                        ──▶ #78  (parallel grid executor)
#139 (frontend LX)  ···▶ #7  (CI catches regressions)
#119 (transpose_b)  ···▶ #3  (decoder coverage)
#2   (RFC ops)      ──▶ #3  (decoder workload)  ··▶ #77 (FFN test)

Indirect access perf chain
──────────────────────────
#95 (strided fast path) ──▶ #96 (BoxSet under permuted vso) ──▶ #97 (ndarray coords) ──▶ #98 (vectorized build)
                                                                                          ···▶ #53 (compose with distributed?)

Roofline cluster
────────────────
#122 (dominant unit) ──▶ #127 (occupancy) ┐
                     ──▶ #125 (bandwidth)  ├──▶ #94 (demo notebook) ──▶ #123 (tracking)
                     ──▶ #126 (comm cost)  ┘
#76  (reduce latency) ···▶ #85 (reduce semantics)
```

---

## 7. Summary

| Tier | Count | Profile |
|:----:|:-----:|---------|
| Tier 1 — Critical | 7 | Blocks workloads or produces wrong results |
| Tier 2 — Important | 23 | Correctness, perf, or infra improvements |
| Tier 3 — Backlog | 7 | Harmless, incremental, or open questions |
| **Total** | **32** | 37 open issues, 5 already have PRs |

| Category | Tier 1 | Tier 2 | Tier 3 | Total |
|----------|:------:|:------:|:------:|:-----:|
| 1) Indirect Access | 0 | 7 | 3 | 10 |
| 2) Distributed Memory | 2 | 3 | 1 | 6 |
| 3) Perf Modeling | 1 | 6 | 2 | 9 |
| 4) Parsing | 2 | 2 | 0 | 4 |
| 5) Executor | 0 | 2 | 1 | 3 |
| 6) Completeness | 2 | 3 | 1 | 6 |

> **Note:** Some issues appear in two categories (secondary noted in Cat 2 column). Totals above count primary category only. PR #138 (open) is related to #122 and #126 but tackles a narrower fix (excluding comm bytes from roofline denominator) rather than fully resolving either.

---

## 8. Recommended Execution Order

1. **Tier 1 Cluster A** (#101, #25) — unblock multi-core and pointer-arithmetic kernels.
2. **Tier 1 Cluster B** (#139, #119) — unblock LLM workloads on the MLIR frontend path.
3. **Tier 1 Cluster C** (#122) — fix roofline before building new metrics on top.
4. **Tier 2 Cluster G** (#102, #31, #108) — solidify the memory model while Cluster A context is fresh.
5. **Tier 2 Cluster F** (#95 → #96 → #97 → #98) — sequential perf chain; highest payoff at #98.
6. **Tier 2 Cluster H** (#127, #125, #126, #94, #76) — roofline improvements now that #122 is fixed.
7. **Tier 1 Cluster D + Tier 2 Cluster K** (#2, #3, #77, #92, #7) — coverage and CI; can be parallelized with above.
8. **Remaining Tier 2** (#56, #67, #40, #48, #88, #78, #53) — independent, pick by contributor interest.
9. **Tier 3** — backlog; address opportunistically or during onboarding.
