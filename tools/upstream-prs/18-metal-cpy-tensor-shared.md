**Title:** `metal : implement cpy_tensor for shared buffers (avoid host-staging copy)`

**Repo:** ggml-org/ggml (direct — `src/ggml-metal/**`, same as merged PR #04).

---

## Background

`ggml_backend_tensor_copy(src, dst)` copies between two tensors. When
neither buffer reports `is_host`, it delegates to
`dst_buf->iface.cpy_tensor(...)`; if that returns `false` it falls back
to a **malloc + `get_tensor` (device→host) + `set_tensor` (host→device)
+ free** staging round-trip (`ggml-backend.cpp`):

```c
} else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
    size_t nbytes = ggml_nbytes(src);
    void * data = malloc(nbytes);
    ggml_backend_tensor_get(src, data, 0, nbytes);
    ggml_backend_tensor_set(dst, data, 0, nbytes);
    free(data);
}
```

The Metal **shared** (unified-memory) buffer never implements the fast
path — `ggml_backend_metal_buffer_shared_cpy_tensor` returns `false`
unconditionally:

```c
static bool ggml_backend_metal_buffer_shared_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ...
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;   // <-- always
}
```

So every same-backend Metal→Metal `ggml_backend_tensor_copy` allocates a
full-tensor host buffer and bounces the data through it — on unified
memory, where the source and destination are *already the same kind of
host-addressable pointer*. The shared buffer's own `get_tensor` /
`set_tensor` are plain unsynchronized `memcpy`s against
`get_base()`-derived pointers, so a direct `memcpy` between two shared
tensors is both correct and strictly cheaper (no allocation, no double
copy).

## Where it bites

Discovered while porting CrispASR's branched beam-search KV-cache
snapshot to on-device copies (our issue #161). The branched-beam decoder
snapshots/restores the decoder KV cache once per beam per step via
`ggml_backend_tensor_copy`. We expected device-to-device blits; on Metal
the unconditional `false` silently routed every snapshot through the
malloc-bounce path, which (with a recycled snapshot pool) measured
*slower* than the original explicit `tensor_get`/`set` it replaced
(`UNACCOUNTED` 705 → 2168 ms on M1 before we special-cased Metal back to
pooled host buffers). The slowdown is pure allocator + redundant-copy
overhead introduced by the fallback, not the copy itself.

Any caller doing same-backend tensor copies on Metal (KV snapshots, beam
search, graph i/o staging) hits this.

## Fix

Implement the shared-buffer `cpy_tensor`: when the source is also
host-addressable (a Metal shared buffer of the same buffer type, or a
host buffer), copy in place with a single `memcpy`; otherwise return
`false` and let the generic path handle it. Sketch:

```c
static bool ggml_backend_metal_buffer_shared_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;
    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    // dst is in this shared (unified-memory) buffer. If src is also
    // host-addressable, the copy is a plain memcpy with no staging.
    if (ggml_backend_buffer_is_host(src->buffer) ||
        src->buffer->buft == dst->buffer->buft) {   // both Metal shared
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}
```

Reference patch: `18-metal-cpy-tensor-shared.patch` (1 file, ~10 LOC).

This matches the existing shared `get_tensor`/`set_tensor` contract:
they already `memcpy` against unified pointers without inserting a GPU
sync, so callers are already responsible for ordering w.r.t. in-flight
command buffers (e.g. `ggml_backend_sched_graph_compute` synchronizes
before any host-visible read). `cpy_tensor` inherits exactly that
contract — no new synchronization semantics.

The same pattern can be applied to the **private** buffer iface
(`ggml_backend_metal_buffer_private_cpy_tensor`, also `return false`)
when both src and dst are private Metal buffers of the same type, via a
blit encoder — left out of this PR to keep it to the unified-memory case
that the macOS/iOS/Asahi GPUs actually use.

## Provenance & verification

Ours — found via the CrispASR #161 KV-snapshot work; the fix is a couple
lines mirroring the existing `get_tensor` access pattern. Validate with
`test-backend-ops` on Metal (any op whose graph i/o triggers a
same-backend copy) and a direct two-shared-tensor `ggml_backend_tensor_copy`
round-trip checked bit-exact against the source.

> Filing note (per `README.md` AI policy): re-author this description in
> your own words before opening the PR and disclose mechanical AI
> assistance in the Requirements section — do not paste this draft.
