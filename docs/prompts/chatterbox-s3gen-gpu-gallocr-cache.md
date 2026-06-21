# Handover — Chatterbox S3Gen CFM: alternative single-all-GPU `ggml_gallocr` path with graph cache

**Goal.** Add an *alternative*, env-gated compute path for the Chatterbox S3Gen
CFM (the flow-matching UNet1D denoiser) that runs the whole UNet on **one GPU
backend** allocated with **raw `ggml_gallocr`** (no `ggml_backend_sched`), and
**caches the compute graph + allocation across the 10 Euler steps**. This is the
single biggest remaining chatterbox perf lever on M1/Metal. The current
(legacy) path rebuilds + re-allocates the graph every Euler step via
`ggml_backend_sched`, which is ~50–55 % of total synthesis and **cannot be
cached** (proven below). Keep the legacy path; gate the new one behind an env
var; flip the default only after parity + a real speedup are demonstrated.

This is a **runtime compute-graph task** — per `feedback_no_agents_for_runtime_graphs`,
do the graph/alloc math yourself against the diff harness + the parity protocol
below. Do **not** delegate the graph wiring to a sub-agent.

---

## Why this is worth doing (measured, M1 Metal, q8 models, 3.4 s clip)

Per-stage bench (`CHATTERBOX_BENCH=1`), HISTORY §207:

| stage | time | % |
|---|---|---|
| **S3Gen CFM Euler** | ~15–21 s | **~54 %** |
| T3 prefill + decode | ~14 s | ~37 % |
| HiFT vocoder | ~3.3 s | ~9 % |

The CFM's ~1.5–2.0 s/step is almost all **overhead** (graph rebuild + sched
alloc + split planning + Metal command-buffer dispatch), not matmul. The
reference impl **gianni-cor/chatterbox.cpp** hits RTF 0.16 on M3 Ultra in part
because it uses **raw `ggml_gallocr` on a single all-GPU backend and reuses the
graph across CFM steps** (their `chatterbox_tts.cpp` `time_mlp_cache` + gallocr
pattern; 8× `ggml_gallocr_alloc_graph`, zero `ggml_backend_sched`). A fresh
read-only clone is at `/Volumes/backups/code/gianni-chatterbox-cpp` — study
their CFM/UNet build + alloc as the existence proof, **do not copy their code**
(different tensor namespace `model/h0/attn/qkv/w`, their own license).

## Why the legacy path can't be cached (don't repeat this)

Two CrispASR sessions (the §205/§207 work and the §206-era graph-cache branch
`perf/chatterbox-graph-cache`) both tried to cache the legacy `ggml_backend_sched`
graph and **both failed identically**:

- **Reuse the built graph across `sched_reset`+`alloc_graph`** → deterministic
  **SIGSEGV at CFM step ~4**, even on a tiny clip (not OOM).
  `ggml_backend_sched_alloc_graph` *mutates the graph* (inserts split-copy
  nodes) and leaves stale `backend_buffer` pointers on intermediate tensors;
  re-allocating the same graph object corrupts it. This is the meaning of the
  existing comment at `cfm_euler_solve` ("gallocr state is not reusable").
- **Reuse the *allocation*** (alloc once, skip `sched_reset` across steps) →
  the `parallel=true` sched stops updating the velocity output (`x_rms` frozen)
  → wrong audio.

So the lever is **not** "cache the sched graph." It is "switch to a single
all-GPU **raw gallocr** path, where the graph is *not* mutated on alloc and can
be reused." That is what this task builds.

---

## Current code (read these first — `src/chatterbox_s3gen.cpp`, line numbers ~main @ §207)

- `chatterbox_s3gen_context` struct (~280–360): `backend`, `backend_cpu`,
  `sched`, `compute_meta`, `unet_on_gpu`, `force_unet_cpu`, `dequant_cfm_f16`,
  `ctx_f16`/`buf_f16` (the q8→F16 CFM weight copies — see §205).
- `ggml_backend_sched_new(..., use_parallel, false)` (~810) — the legacy sched.
  `use_parallel = (backend != backend_cpu)`.
- **`build_graph_unet1d_b2(c, T_mel)`** (~2140) — the batch=2 cond+uncond CFG
  graph (the default; `use_cfg_b2` at ~2542). Builds into `c->compute_meta`,
  `ggml_free(ctx0)` at end. Inputs: `unet_input_b2` (T_mel,320,2), `time_emb`
  (1024), `mask`; output: `denoiser_out_b2` (T,80,2).
- `build_graph_unet1d(c, T_mel)` (~2248) — the single-pass (batch=1) variant
  used when `CRISPASR_S3GEN_UNET_CFG_SINGLE=1`.
- **`cfm_euler_solve`** (~2443) — the 10-step loop. The b2 branch (~2569+):
  `build_graph_unet1d_b2` → `ggml_backend_sched_reset` →
  `s3gen_maybe_pin_graph_to_cpu` → `ggml_backend_sched_alloc_graph` → set 3
  inputs → `ggml_backend_sched_graph_compute` → read `denoiser_out_b2` → Euler
  update `x += dt*((1+cfg)*vc − cfg*vu)`.
- `s3gen_maybe_pin_graph_to_cpu` (~482) — no-op in the F16-dequant default
  (`force_unet_cpu=false`, `unet_pin_mm_cpu=false`).

### Constraints you must respect (these are why it's not trivial)

1. **`parallel=true` / Bug B (PLAN #83 r9 #5, comment ~797).** The legacy sched
   needs `parallel=true` because consecutive computes with CPU-`memcpy`'d inputs
   (the per-step `tensor_set`) plus `parallel=false`'s `waitUntilCompleted` do
   **not** invalidate Metal's view of a shared-storage buffer overwritten between
   command buffers → CFG uncond pass diverges. **A single-backend raw-gallocr
   path side-steps the sched entirely**, so re-verify whether Bug B even applies
   (it may not, since there's no GPU↔CPU copy boundary), but you must prove CFG
   cond/uncond stay correct (parity protocol below). This is the #1 risk.
2. **F16-dequant weights (§205).** On Metal the q8 `s3.fd.*` weights are
   dequantized to F16 at load (`dequant_cfm_f16`) so the CFM uses
   `mul_mm_f16_f32_hp`, not the activation-requantizing `mul_mv_q8_0_q8_0` (which
   NaNs/garbages). Your gallocr path **must keep using the F16 weights**
   (`c->tensors` already points at them) — never feed q8 weights to the GPU CFM.
3. **`c->sched` is shared.** The encoder (before the CFM) and the HiFT vocoder
   (after) both `sched_reset`+`alloc` on `c->sched`. A new gallocr path for the
   UNet must use its **own** `ggml_gallocr_t` (+ its own persistent graph
   context), independent of `c->sched`, so the encoder/vocoder are untouched.
4. **The b2 graph has 31 backend "splits"** under the sched even when nominally
   GPU-resident. On a single all-GPU backend with raw gallocr there is no split
   machinery, which is precisely where the per-step overhead goes — but confirm
   every op in `build_graph_unet1d_b2` is Metal-supported on a single backend
   (if any op currently silently falls to CPU via the sched, it will now run on
   GPU or fail — check with `ggml_backend_supports_op`).

---

## What to build

Add an alternative UNet1D execution path, selected by env, with three pieces:

1. **A dedicated single GPU backend handle + raw `ggml_gallocr`** for the CFM,
   created at init only when the new path is active and a GPU backend exists.
   Do **not** reuse `c->sched`. Model it on the project's own `ggml_gallocr`
   pattern (see `docs/` "ggml_gallocr (graph allocator) — CrispASR pattern" and
   e.g. `kyutai_stt.cpp` / `moonshine_streaming.cpp` §176s encoder caches), and
   on gianni's `time_mlp_cache`/gallocr usage as the existence proof.

2. **A cached b2 graph** built once per `T_mel` into a persistent ctx + meta
   buffer (like the §176s `cached_enc_*` pattern, and like the reverted attempt
   in HISTORY §207 — but now with `ggml_gallocr_alloc_graph` instead of the
   sched, which is the part that makes reuse legal). Per Euler step: set the 3
   inputs (`tensor_set`), `ggml_backend_graph_compute` (single backend), read
   `denoiser_out_b2`. Rebuild only when `T_mel` changes; free on T_mel change /
   in the destructor.

3. **Env gating** (follow `CLAUDE.md` "Env var gating" convention):
   - `CRISPASR_S3GEN_UNET_GALLOCR=1` → new single-all-GPU raw-gallocr cached
     path (this task). `=0`/unset → legacy `ggml_backend_sched` path (default
     until validated).
   - Keep all existing knobs working: `CRISPASR_S3GEN_UNET_CPU=1` (CPU route),
     `CRISPASR_S3GEN_UNET_CFG_SINGLE=1` (batch=1), `CRISPASR_CHATTERBOX_SEED`,
     `--tts-steps N`. The new path only replaces the **b2 GPU compute** inside
     `cfm_euler_solve`; if `UNET_CPU=1` or backend is CPU-only, fall back to
     legacy automatically.
   - Document every new env in this file's "Env vars" section and PLAN §176.

Wire it as: in `cfm_euler_solve`'s b2 branch, `if (c->unet_gallocr_active)` use
the cached gallocr path, `else` the legacy sched path (unchanged). Set
`unet_gallocr_active` at init from the env + GPU availability.

---

## Validation — MANDATORY, this path is correctness-critical

Use the §207 protocol (fixed seed → only the compute path varies):

1. **Build** the worktree (`CCACHE_DIR=$HOME/.ccache`, Ninja, Release).
2. **Parity run** vs the legacy path, identical inputs:
   `CRISPASR_CHATTERBOX_SEED=42 CRISPASR_S3GEN_UNET_GALLOCR=1 CHATTERBOX_BENCH=1 CRISPASR_S3GEN_DUMP=1`
   `crispasr --backend chatterbox -m <t3-q8> --codec-model <s3gen-q8> --seed 1234 --voice samples/jfk.wav --i-have-rights --no-spoken-disclaimer --tts "The quick brown fox jumps over the lazy dog." --tts-output gallocr.wav`
   and the same with `CRISPASR_S3GEN_UNET_GALLOCR=0` → `legacy.wav`.
   Models: `/Volumes/backups/ai/crispasr/chatterbox-{t3,s3gen}-q8_0.gguf`.
3. **Per-step parity:** `s3gen: CFM step N/10 x_rms=...` must **grow** smoothly
   (1.0 → ~4–11) and **match the legacy run step-for-step** within ~1e-3. A
   frozen `x_rms` = the velocity isn't updating (the alloc-reuse bug); a NaN =
   wrong weights/op on GPU.
4. **Output parity:** compare `gallocr.wav` vs `legacy.wav` with a
   **phase-invariant log-magnitude-spectrogram correlation** (the §207 metric);
   require **≥ 0.999** (this should be near-bit-exact — same graph, same
   weights, just a different allocator — unlike the §207 step-count change which
   was an approximation). Also ASR-roundtrip both (e.g.
   `--backend moonshine -m moonshine-base-q4_k.gguf`) → identical text.
5. **Speedup:** `CFM euler` ms/step from `CHATTERBOX_BENCH=1` must drop
   meaningfully vs legacy (target: eliminate most of the per-step rebuild+alloc
   overhead). Report before/after ms/step and total RTF. Run **sequentially**
   (no parallel runs — they contend on M1 and contaminate timing).
6. **q8 + F16 + Q4_K**, and **base + turbo (meanflow 2-step)** variants must all
   still produce intelligible audio (turbo uses `build_graph_unet1d` single-pass
   / meanflow — confirm the gate handles non-b2 paths or leaves them legacy).
7. **`crispasr-diff chatterbox`** stages (`cfm_step0_result`, mel) must still
   pass — the diff harness pins 10 steps; run it with the gallocr path on and
   confirm no regression. Reference dump: `cstr/chatterbox-GGUF/diff-harness-ref/`
   (and the local copies under `/Volumes/backups/ai/`).

Acceptance: parity ≥ 0.999 spectral + identical ASR + step-wise x_rms match +
measurable CFM speedup, with the legacy path bit-unchanged when the env is off.
Only then propose flipping the default (and even then keep the legacy path +
env opt-out forever — never delete the gate; it's the regression-bisection
mechanism).

## Failure modes to watch (seen this session)

- `ggml_metal_buffer_get_id: ... buffer is nil` → a tensor wasn't allocated by
  the gallocr before compute (mark inputs with `ggml_set_input`, allocate before
  `tensor_set`; see the gallocr doc pattern: `tensor_set`/`tensor_get` go via
  `ggml_backend_tensor_*`, not `memcpy`).
- Frozen `x_rms` → reused allocation but inputs/outputs not re-bound per step.
- NaN/garbage mel → q8 weights reached the GPU CFM (must be the F16-dequant
  copies), or an op silently needed the CPU split that the sched used to provide.
- CFG cond/uncond divergence (Bug B) on the b2 path → the single-backend path
  must still produce correct uncond; if it diverges, that's the Bug-B mechanism
  resurfacing — investigate the input-copy/cache-coherency, don't paper over it.

## References

- HISTORY **§205** (FFT crash + q8/Metal F16-dequant), **§207** (CFM 10→6 +
  graph-cache-is-non-viable-under-sched). LEARNINGS §205 (q8 mat-vec) + §207.
- `feedback_no_agents_for_runtime_graphs`, `feedback_diff_alignment`,
  `feedback_tts_validation` (ASR-roundtrip every TTS output).
- `docs/diff-harness-coverage.md` (chatterbox ref + the `diff-harness-ref/` HF
  archive), `tools/benchmark_chatterbox.py` (CrispASR-vs-Python bench).
- Reference impl (read-only): `/Volumes/backups/code/gianni-chatterbox-cpp`
  (`src/chatterbox_tts.cpp` — gallocr + `time_mlp_cache`; `--cfm-steps` knob).
- ggml gallocr usage pattern: `docs/` "ggml_gallocr (graph allocator) — CrispASR
  pattern"; §176s encoder caches in `kyutai_stt.cpp` / `moonshine_streaming.cpp`.

## Env vars (add to PLAN §176 + the backend's section)

- `CRISPASR_S3GEN_UNET_GALLOCR=1` — opt into the single-all-GPU raw-gallocr
  cached UNet path (this task). Default off (legacy `ggml_backend_sched`).
- (existing, keep working) `CRISPASR_S3GEN_UNET_CPU=1`,
  `CRISPASR_S3GEN_UNET_CFG_SINGLE=1`, `CRISPASR_CHATTERBOX_SEED`, `--tts-steps`.
