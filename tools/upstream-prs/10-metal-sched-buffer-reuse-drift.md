**Title:** `metal/ggml-alloc : long F32 GPU graphs accumulate drift sensitive to in-place buffer reuse pattern`

---

This is a bug report rather than a patch. It documents a precision
issue we hit running a 14-block diffusion UNet (the chatterbox-tts
S3Gen conditional flow-matching decoder) on Apple Silicon Metal, and
the bisect that points at `ggml-alloc`'s in-place buffer reuse rather
than any single op kernel. Filing because the root-cause investigation
needs ggml-internals knowledge beyond what an application can do.

## Repro shape

Pipeline: 1 down block + 12 mid blocks + 1 up block + final
projection. Each block: `causal_resnet_block` (CausalConv1d + LayerNorm
+ Mish + time-MLP add + CausalConv1d) + 4× BasicTransformerBlock
(LayerNorm + Q/K/V mul_mat + flash_attn_ext + output mul_mat + add +
LayerNorm + FFN up mul_mat + GELU + FFN down mul_mat + add). Weights
mostly Q8_0 with some F16. Activations all F32. Graph: ~396 mul_mats
per pass; run 10 times in a CFM Euler ODE solver.

Reference output dumped from a known-good CPU pass.

## Observed drift

CPU path (whole UNet computed on `ggml_backend_cpu`):
`cos_min = 1.000000`, max_abs vs ref = 0.

Pure GPU path (whole UNet on Metal):
`cos_min = 0.940`, max_abs vs ref = 14.4.

The GPU output is structurally similar (cos_mean = 0.976) but with
elements deviating by up to 14× the activation scale at the worst
positions. Downstream this produces unintelligible synthesised audio.

## Bisect findings — *not* a kernel-precision bug

We chased this through several rounds. Each is independent evidence
that no single op kernel is the source:

1. **Bit-match `mul_mat`.** We added a `GGML_PREC_F32` dispatch for
   Q8_0 × F32 that pre-quantises input to Q8_0 and runs an integer-dot
   kernel matching `ggml_vec_dot_q8_0_q8_0_generic` bit-for-bit (PR
   09). Confirmed bit-identical to CPU mul_mat output. With this
   kernel firing on all 350 prec-tagged UNet mul_mats, `cos_min` moves
   from 0.940 to **0.947** — essentially no change.

2. **Per-op pin bisect.** With `ggml_backend_sched_set_tensor_backend`
   used to pin a single op type to the CPU backend (and the rest of
   the graph on GPU), we measured `cos_min` per op type pinned:

   | Pin to CPU | cos_min | Frequency in UNet |
   | - | - | - |
   | `mul_mat` | 1.000 | high (~7/block) |
   | `norm`, `mul`, `add`, `flash_attn_ext`, `gelu` | 1.000 | high (≥1/block) |
   | `reshape`, `cont`, `concat`, `permute` | 1.000 | high (memory ops) |
   | `conv_1d`, `soft_max`, `mish`, `silu`, `scale` | 0.940 | low (≤1/block) |

   The clean correlation is **op frequency**, not op identity. Pinning
   any op that occurs frequently restores parity; pinning a sparse op
   doesn't. This points at sync-barrier density, not op correctness.

3. **`ggml_set_output` bisect.** `ggml_set_output` marks a tensor as a
   graph output, which makes `ggml_gallocr` skip the in-place buffer
   reuse path for it (see `ggml-alloc.c` around line 644). We tested:

   | What's marked output | cos_min |
   | - | - |
   | nothing extra (default) | 0.940 |
   | the first block's resnet output only | **0.879** (worse!) |
   | all 62 sub-block outputs | **1.000** |

   The single-checkpoint result is the smoking gun: a `set_output` on
   one specific tensor changes the allocator's reuse decisions
   downstream, and the new pattern produces *more* drift, not less.

## What this means

The chatterbox UNet output is correct with default GPU compute except
for a path-dependent compounding effect that is sensitive to which
buffers `ggml_gallocr` chooses to reuse in-place. There's no single
buggy op — there's a combination of "approximate enough" GPU op
outputs whose error trajectory through ~4000 GPU ops depends on the
allocator's reuse decisions in a non-monotonic way.

Our production workaround is to load the UNet's weight tensors on the
CPU backend (via `ggml_backend_sched`'s weight-residency routing) so
the entire UNet executes on CPU. That's a 0% perf win for us (the
encoder/vocoder around it stay on GPU). The whole-UNet `set_output`
approach restores precision at small T but produces NaN at production
T (see related issue on scheduler NaN at large T with mixed CPU+GPU
ops).

## Investigation pointers

The two ggml entry points we could trace it to:

- `ggml-alloc.c:622` `ggml_gallocr_allocate_node` — the in-place reuse
  decision is made per node based on `n_children == 1 && n_views ==
  0`. The reuse choice depends on traversal order, so adding an output
  marker can flip cascading reuse decisions downstream.

- `ggml-metal/ggml-metal-ops.cpp:159` `ggml_metal_op_concurrency_check`
  — the mem-range overlap check that adds Metal command-buffer
  barriers. Looks correct on inspection (SRC/DST overlap detection
  matches what you'd expect), but worth eyeballing for the case where
  an op writes to a buffer in-place that was also read by an earlier
  enqueued-but-not-yet-executed op.

A "disable inplace reuse for graphs marked as F32-precision" knob
in `ggml-alloc` would let applications opt into bit-equivalent GPU
output at a memory cost, and would expose this issue for further
study.

## How to reproduce

Standalone repro requires the chatterbox model files (50 MB) and our
diff-harness scaffolding; we can extract a minimal `test-backend-ops`
case if helpful — flag this issue and we'll prepare one.
