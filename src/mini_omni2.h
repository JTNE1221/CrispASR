// mini_omni2.h — C API for Mini-Omni2 (gpt-omni/mini-omni2).
//
// Architecture: Whisper-small encoder (80 mel, 12L, 768d)
//             + whisperMLP adapter (SwiGLU 768→4864→896)
//             + Qwen2-0.5B LLM (896d, 24L, GQA 14/2, SwiGLU, RMSNorm)
//
// ASR path only in Phase 1 — text-only greedy decode from audio input.
// Speech output (SNAC 24kHz) is a follow-on.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mini_omni2_context;

struct mini_omni2_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    float temperature; // 0 = greedy argmax, >0 = softmax sampling
};

struct mini_omni2_context_params mini_omni2_context_default_params(void);

struct mini_omni2_context* mini_omni2_init_from_file(const char* path_model, struct mini_omni2_context_params params);

void mini_omni2_free(struct mini_omni2_context* ctx);

// High-level: transcribe raw 16 kHz mono PCM audio.
// Returns malloc'd UTF-8 string, caller frees with free().
char* mini_omni2_transcribe(struct mini_omni2_context* ctx, const float* samples, int n_samples);

// Pipeline building blocks for differential testing:

// Compute mel spectrogram from raw PCM. Returns malloc'd (n_mels, T_mel) F32.
float* mini_omni2_compute_mel(struct mini_omni2_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                              int* out_T_mel);

// Run Whisper encoder on mel. Returns malloc'd (T_enc, 768) F32.
float* mini_omni2_run_encoder(struct mini_omni2_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                              int* out_dim);

// Run whisperMLP adapter on encoder output. Returns malloc'd (T_enc, 896) F32.
float* mini_omni2_run_adapter(struct mini_omni2_context* ctx, const float* enc, int T_enc, int enc_dim, int* out_T,
                              int* out_dim);

// KV cache management.
bool mini_omni2_kv_init(struct mini_omni2_context* ctx, int max_ctx);
void mini_omni2_kv_reset(struct mini_omni2_context* ctx);

// Run LLM forward with KV cache. Returns malloc'd logits (vocab_size,) F32.
float* mini_omni2_run_llm_kv(struct mini_omni2_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                             int* out_n_tokens, int* out_vocab_size);

// Token text lookup.
const char* mini_omni2_token_text(struct mini_omni2_context* ctx, int id);

#ifdef __cplusplus
}
#endif
