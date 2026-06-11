// mini_omni2.cpp — Mini-Omni2 runtime (gpt-omni/mini-omni2).
//
// Architecture: Whisper-small (80 mel, 12L, 768d, sinusoidal pos, LayerNorm+bias)
//             + whisperMLP adapter (SwiGLU 768→4864→896, no bias)
//             + Qwen2-0.5B LLM (896d, 24L, GQA 14/2, SwiGLU, RMSNorm)
//
// Phase 1: per-stage APIs for diff-harness validation.
// Phase 2: full ASR transcribe path.

#include "mini_omni2.h"

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "core/attention.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/mel.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ===========================================================================
// Model structures
// ===========================================================================

struct mo2_hparams {
    // Audio encoder (Whisper-small)
    int enc_hidden = 768;
    int enc_n_layers = 12;
    int enc_n_heads = 12;
    int enc_ff = 3072;
    int n_mels = 80;
    int enc_max_pos = 1500;
    // Adapter
    int adapter_dim = 768; // input dim (= enc_hidden)
    int adapter_ff = 4864; // intermediate (= llm_ff)
    int adapter_out = 896; // output dim (= llm_hidden)
    // LLM decoder (Qwen2-0.5B)
    int llm_hidden = 896;
    int llm_n_layers = 24;
    int llm_n_heads = 14;
    int llm_n_kv_heads = 2;
    int llm_ff = 4864;
    int llm_vocab = 181120;
    int llm_max_pos = 2048;
    float rms_eps = 1e-6f;
    int rope_base = 1000000;
    bool tie_word_embeddings = true;
    // Vocab split
    int text_vocab_size = 152000;
    int audio_vocab_size = 4160;
    // Derived
    int enc_head_dim = 64; // enc_hidden / enc_n_heads
    int llm_head_dim = 64; // llm_hidden / llm_n_heads
};

// Whisper encoder block tensors
struct mo2_enc_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* attn_out_b = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_norm_b = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
};

// Qwen2 LLM block tensors
struct mo2_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct mo2_model {
    mo2_hparams hp;

    // Audio encoder (Whisper-small)
    struct {
        ggml_tensor* conv1_w = nullptr;
        ggml_tensor* conv1_b = nullptr;
        ggml_tensor* conv2_w = nullptr;
        ggml_tensor* conv2_b = nullptr;
        ggml_tensor* pos_embd = nullptr; // sinusoidal (1500, 768)
        ggml_tensor* ln_post_w = nullptr;
        ggml_tensor* ln_post_b = nullptr;
        std::vector<mo2_enc_block> blocks;
    } audio;

    // whisperMLP adapter (SwiGLU)
    struct {
        ggml_tensor* fc_1_w = nullptr; // (768, 4864) gate
        ggml_tensor* fc_2_w = nullptr; // (768, 4864) up
        ggml_tensor* proj_w = nullptr; // (4864, 896) down
    } adapter;

    // LLM (Qwen2-0.5B)
    struct {
        ggml_tensor* token_embd_w = nullptr;
        ggml_tensor* output_norm_w = nullptr;
        ggml_tensor* lm_head_w = nullptr; // may be nullptr if tied
        std::vector<mo2_llm_block> blocks;
    } llm;

    // Tokenizer
    std::vector<std::string> vocab;

    // GGUF context
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_buffer_t buf_cpu = nullptr;
};

struct mini_omni2_context {
    mini_omni2_context_params params;
    mo2_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;

    int n_threads = 4;
};

// ===========================================================================
// Implementation
// ===========================================================================

extern "C" struct mini_omni2_context_params mini_omni2_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1, /*use_gpu=*/true, /*temperature=*/0.0f};
}

extern "C" struct mini_omni2_context* mini_omni2_init_from_file(const char* path_model,
                                                                struct mini_omni2_context_params params) {
    auto* ctx = new mini_omni2_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    auto& m = ctx->model;
    auto& hp = m.hp;

    // ---- pass 1: read hparams + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path_model);
        if (!gctx) {
            fprintf(stderr, "mini_omni2: failed to open '%s'\n", path_model);
            delete ctx;
            return nullptr;
        }
        hp.enc_hidden = core_gguf::kv_u32(gctx, "mini_omni2.audio.hidden_size", hp.enc_hidden);
        hp.enc_n_layers = core_gguf::kv_u32(gctx, "mini_omni2.audio.num_layers", hp.enc_n_layers);
        hp.enc_n_heads = core_gguf::kv_u32(gctx, "mini_omni2.audio.num_heads", hp.enc_n_heads);
        hp.enc_ff = core_gguf::kv_u32(gctx, "mini_omni2.audio.intermediate_size", hp.enc_ff);
        hp.n_mels = core_gguf::kv_u32(gctx, "mini_omni2.audio.num_mel_bins", hp.n_mels);
        hp.enc_max_pos = core_gguf::kv_u32(gctx, "mini_omni2.audio.max_position_embeddings", hp.enc_max_pos);

        hp.adapter_dim = core_gguf::kv_u32(gctx, "mini_omni2.adapter.input_dim", hp.adapter_dim);
        hp.adapter_ff = core_gguf::kv_u32(gctx, "mini_omni2.adapter.intermediate_size", hp.adapter_ff);
        hp.adapter_out = core_gguf::kv_u32(gctx, "mini_omni2.adapter.output_dim", hp.adapter_out);

        hp.llm_hidden = core_gguf::kv_u32(gctx, "mini_omni2.llm.hidden_size", hp.llm_hidden);
        hp.llm_n_layers = core_gguf::kv_u32(gctx, "mini_omni2.llm.num_layers", hp.llm_n_layers);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "mini_omni2.llm.num_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "mini_omni2.llm.num_kv_heads", hp.llm_n_kv_heads);
        hp.llm_ff = core_gguf::kv_u32(gctx, "mini_omni2.llm.intermediate_size", hp.llm_ff);
        hp.llm_vocab = core_gguf::kv_u32(gctx, "mini_omni2.llm.vocab_size", hp.llm_vocab);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "mini_omni2.llm.max_position_embeddings", hp.llm_max_pos);
        hp.rms_eps = core_gguf::kv_f32(gctx, "mini_omni2.llm.rms_norm_eps", hp.rms_eps);
        hp.rope_base = core_gguf::kv_u32(gctx, "mini_omni2.llm.rope_base", hp.rope_base);

        hp.tie_word_embeddings = core_gguf::kv_bool(gctx, "mini_omni2.tie_word_embeddings", hp.tie_word_embeddings);
        hp.text_vocab_size = core_gguf::kv_u32(gctx, "mini_omni2.text_vocab_size", hp.text_vocab_size);
        hp.audio_vocab_size = core_gguf::kv_u32(gctx, "mini_omni2.audio_vocab_size", hp.audio_vocab_size);

        hp.enc_head_dim = hp.enc_hidden / hp.enc_n_heads;
        hp.llm_head_dim = hp.llm_hidden / hp.llm_n_heads;

        // Tokenizer
        m.vocab.resize(hp.llm_vocab);
        const int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (tok_key >= 0) {
            const int n = gguf_get_arr_n(gctx, tok_key);
            for (int i = 0; i < n && i < hp.llm_vocab; i++) {
                const char* s = gguf_get_arr_str(gctx, tok_key, i);
                if (s)
                    m.vocab[i] = s;
            }
        }

        gguf_free(gctx);
    }

    // ---- pass 2: load tensor data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "mini_omni2", wl)) {
        fprintf(stderr, "mini_omni2: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    m.ctx = wl.ctx;
    m.buf = wl.buf;
    m.buf_cpu = wl.buf_cpu;

    // Map tensors
    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        if (it == wl.tensors.end()) {
            fprintf(stderr, "mini_omni2: tensor '%s' not found\n", name);
            return nullptr;
        }
        return it->second;
    };
    auto try_get = [&](const char* name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        return it != wl.tensors.end() ? it->second : nullptr;
    };

    // Audio encoder
    m.audio.conv1_w = get("audio.conv1.weight");
    m.audio.conv1_b = get("audio.conv1.bias");
    m.audio.conv2_w = get("audio.conv2.weight");
    m.audio.conv2_b = get("audio.conv2.bias");
    m.audio.pos_embd = get("audio.positional_embedding");
    m.audio.ln_post_w = get("audio.norm.weight");
    m.audio.ln_post_b = try_get("audio.norm.bias");
    m.audio.blocks.resize(hp.enc_n_layers);
    for (int i = 0; i < hp.enc_n_layers; i++) {
        char buf[128];
        auto& b = m.audio.blocks[i];
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_norm.weight", i);
        b.attn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_norm.bias", i);
        b.attn_norm_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_q.weight", i);
        b.attn_q_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_q.bias", i);
        b.attn_q_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_k.weight", i);
        b.attn_k_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_v.weight", i);
        b.attn_v_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_v.bias", i);
        b.attn_v_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_out.weight", i);
        b.attn_out_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.attn_out.bias", i);
        b.attn_out_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn_norm.weight", i);
        b.ffn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn_norm.bias", i);
        b.ffn_norm_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn.up.weight", i);
        b.ffn_up_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn.up.bias", i);
        b.ffn_up_b = try_get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn.down.weight", i);
        b.ffn_down_w = get(buf);
        snprintf(buf, sizeof(buf), "audio.blk.%d.ffn.down.bias", i);
        b.ffn_down_b = try_get(buf);
    }

    // Adapter
    m.adapter.fc_1_w = get("adapter.fc_1.weight");
    m.adapter.fc_2_w = get("adapter.fc_2.weight");
    m.adapter.proj_w = get("adapter.proj.weight");

    // LLM
    m.llm.token_embd_w = get("llm.token_embd.weight");
    m.llm.output_norm_w = get("llm.output_norm.weight");
    m.llm.lm_head_w = try_get("lm_head.weight");
    if (!m.llm.lm_head_w && hp.tie_word_embeddings)
        m.llm.lm_head_w = m.llm.token_embd_w;
    m.llm.blocks.resize(hp.llm_n_layers);
    for (int i = 0; i < hp.llm_n_layers; i++) {
        char buf[128];
        auto& b = m.llm.blocks[i];
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_norm.weight", i);
        b.attn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.ffn_norm.weight", i);
        b.ffn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_q.weight", i);
        b.attn_q_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_q.bias", i);
        b.attn_q_b = try_get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_k.weight", i);
        b.attn_k_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_k.bias", i);
        b.attn_k_b = try_get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_v.weight", i);
        b.attn_v_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_v.bias", i);
        b.attn_v_b = try_get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.attn_out.weight", i);
        b.attn_out_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.ffn.gate.weight", i);
        b.ffn_gate_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.ffn.up.weight", i);
        b.ffn_up_w = get(buf);
        snprintf(buf, sizeof(buf), "llm.blk.%d.ffn.down.weight", i);
        b.ffn_down_w = get(buf);
    }

    // Scheduler
    int n_be = 1;
    ggml_backend_t backends[2] = {ctx->backend, nullptr};
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend) {
        backends[n_be++] = ctx->backend_cpu;
    }
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        fprintf(stderr, "mini_omni2: loaded %d enc + %d llm layers, vocab %d\n", hp.enc_n_layers, hp.llm_n_layers,
                hp.llm_vocab);
    }

    return ctx;
}

extern "C" void mini_omni2_free(struct mini_omni2_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.buf_cpu)
        ggml_backend_buffer_free(ctx->model.buf_cpu);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" const char* mini_omni2_token_text(struct mini_omni2_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->model.vocab.size())
        return "";
    return ctx->model.vocab[id].c_str();
}

// ===========================================================================
// Mel spectrogram
// ===========================================================================

extern "C" float* mini_omni2_compute_mel(struct mini_omni2_context* ctx, const float* samples, int n_samples,
                                         int* out_n_mels, int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->model.hp;

    // Get mel_filters and mel_window from model tensors
    ggml_tensor* mel_fb_t = ggml_get_tensor(ctx->model.ctx, "audio.mel_filters");
    ggml_tensor* mel_win_t = ggml_get_tensor(ctx->model.ctx, "audio.mel_window");
    if (!mel_fb_t || !mel_win_t) {
        fprintf(stderr, "mini_omni2: mel_filters or mel_window not found\n");
        return nullptr;
    }

    int n_freqs = (int)mel_fb_t->ne[0];
    int n_mels_fb = (int)mel_fb_t->ne[1];
    if (n_mels_fb == 201) {
        std::swap(n_freqs, n_mels_fb);
    }
    std::vector<float> mel_fb((size_t)n_freqs * n_mels_fb);
    ggml_backend_tensor_get(mel_fb_t, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    int win_len = (int)mel_win_t->ne[0];
    std::vector<float> mel_win(win_len);
    ggml_backend_tensor_get(mel_win_t, mel_win.data(), 0, mel_win.size() * sizeof(float));

    // Whisper mel: n_fft=400, hop=160, 80 mels, log10, GlobalClipMax
    core_mel::Params p;
    p.n_mels = hp.n_mels;
    p.n_fft = 400;
    p.hop_length = 160;
    p.win_length = 400;
    p.center_pad = true;
    p.center_pad_reflect = true;
    p.log_base = core_mel::LogBase::Log10;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.log_eps = 1e-10f;
    p.norm = core_mel::Normalization::GlobalClipMax;
    p.layout = core_mel::Layout::MelsTime;
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.matmul = core_mel::MatmulPrecision::Double;

    // DFT for n_fft=400 (not a power of 2). Whisper's torch.stft uses
    // n_fft=400 directly, so the frequency bins are at k*sr/400 spacing.
    // Zero-padding to 512 would put bins at k*sr/512 — wrong spacing for
    // the mel filterbank. O(N*N_freqs) is acceptable for N=400.
    auto mo2_fft = [](const float* in, int N, float* out) {
        const int n_freqs = N / 2 + 1;
        for (int k = 0; k < n_freqs; k++) {
            double re = 0.0, im = 0.0;
            const double ang = -2.0 * M_PI * k / N;
            for (int n = 0; n < N; n++) {
                const double theta = ang * n;
                re += (double)in[n] * cos(theta);
                im += (double)in[n] * sin(theta);
            }
            out[2 * k] = (float)re;
            out[2 * k + 1] = (float)im;
        }
        // Fill remaining bins (core_mel expects 2*N output floats)
        for (int k = n_freqs; k < N; k++) {
            out[2 * k] = 0.0f;
            out[2 * k + 1] = 0.0f;
        }
    };

    int T_mel = 0;
    auto mel =
        core_mel::compute(samples, n_samples, mel_win.data(), win_len, mel_fb.data(), n_freqs, mo2_fft, p, T_mel);
    if (mel.empty())
        return nullptr;

    // Whisper's torch.stft drops the last frame (stft[..., :-1]). core_mel
    // produces one extra frame. Truncate to match Whisper's output.
    const int T_whisper = n_samples / p.hop_length;
    if (T_mel > T_whisper) {
        // MelsTime layout: data is [mel_0[0..T-1], mel_1[0..T-1], ...].
        // Truncating means removing trailing frames per mel band.
        std::vector<float> trimmed((size_t)hp.n_mels * T_whisper);
        for (int m = 0; m < hp.n_mels; m++) {
            memcpy(trimmed.data() + (size_t)m * T_whisper, mel.data() + (size_t)m * T_mel, T_whisper * sizeof(float));
        }
        mel = std::move(trimmed);
        T_mel = T_whisper;
    }

    if (out_n_mels)
        *out_n_mels = hp.n_mels;
    if (out_T_mel)
        *out_T_mel = T_mel;

    float* result = (float*)malloc(mel.size() * sizeof(float));
    if (!result)
        return nullptr;
    memcpy(result, mel.data(), mel.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Whisper encoder
// ===========================================================================

static ggml_cgraph* mo2_build_encoder(mini_omni2_context* ctx, int T_mel) {
    const auto& m = ctx->model;
    const auto& hp = m.hp;
    const int d = hp.enc_hidden;          // 768
    const int n_heads = hp.enc_n_heads;   // 12
    const int hd = hp.enc_head_dim;       // 64
    const int n_layers = hp.enc_n_layers; // 12
    const int n_mels = hp.n_mels;         // 80
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input mel (n_mels, T_mel) — Whisper convention
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // Conv stem: conv1(80→768, k=3, s=1, p=1) + GELU + conv2(768→768, k=3, s=2, p=1) + GELU
    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };

    ggml_tensor* cur = ggml_conv_1d(ctx0, m.audio.conv1_w, mel, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv1_b));
    cur = ggml_gelu_erf(ctx0, cur);

    cur = ggml_conv_1d(ctx0, m.audio.conv2_w, cur, 2, 1, 1);
    cur = ggml_add(ctx0, cur, bias_1d(m.audio.conv2_b));
    cur = ggml_gelu_erf(ctx0, cur);

    cur = ggml_cont(ctx0, cur);
    const int T_conv = (T_mel + 1) / 2; // stride-2 conv output (may be 1 extra)
    // ggml_conv_1d output is (d, T_conv, 1). Reshape to 2D first.
    cur = ggml_reshape_2d(ctx0, cur, T_conv, d);
    // Clamp to max positional embedding length (Whisper-small: 1500).
    // core_mel center-pad can produce one extra frame vs the Python STFT.
    const int T_enc = std::min(T_conv, hp.enc_max_pos);
    if (T_enc < T_conv) {
        cur = ggml_view_2d(ctx0, cur, T_enc, d, cur->nb[1], 0);
    }
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // (d, T_enc) → (T_enc, d)

    // Add sinusoidal positional embedding: pos_embd is (1500, 768), slice to T_enc.
    // pos_embd may be F16 from the GGUF — cast to F32 for the add.
    ggml_tensor* pos = ggml_view_2d(ctx0, m.audio.pos_embd, d, T_enc, m.audio.pos_embd->nb[1], 0);
    if (pos->type != GGML_TYPE_F32)
        pos = ggml_cast(ctx0, pos, GGML_TYPE_F32);
    cur = ggml_add(ctx0, cur, pos);

    // 12 × Whisper encoder blocks (pre-norm LayerNorm with bias, no RoPE — sinusoidal pos)
    for (int il = 0; il < n_layers; il++) {
        const auto& b = m.audio.blocks[il];
        ggml_tensor* residual = cur;

        // Pre-attention LayerNorm
        ggml_tensor* x = ggml_norm(ctx0, cur, 1e-5f);
        x = ggml_mul(ctx0, x, b.attn_norm_w);
        if (b.attn_norm_b)
            x = ggml_add(ctx0, x, b.attn_norm_b);

        // Self-attention (no RoPE — Whisper uses sinusoidal positional embedding)
        {
            ggml_tensor* Q = ggml_mul_mat(ctx0, b.attn_q_w, x);
            if (b.attn_q_b)
                Q = ggml_add(ctx0, Q, b.attn_q_b);
            ggml_tensor* K = ggml_mul_mat(ctx0, b.attn_k_w, x);
            ggml_tensor* V = ggml_mul_mat(ctx0, b.attn_v_w, x);
            if (b.attn_v_b)
                V = ggml_add(ctx0, V, b.attn_v_b);

            // Reshape to (hd, n_heads, T_enc) for flash_attn
            Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, T_enc);
            K = ggml_reshape_3d(ctx0, K, hd, n_heads, T_enc);
            V = ggml_reshape_3d(ctx0, V, hd, n_heads, T_enc);

            // Permute to (hd, T_enc, n_heads) for flash_attn
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

            // Bidirectional flash attention (no causal mask)
            ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, scale, 0.0f, 0.0f);
            attn = ggml_reshape_2d(ctx0, attn, n_heads * hd, T_enc);

            // Output projection
            attn = ggml_mul_mat(ctx0, b.attn_out_w, attn);
            if (b.attn_out_b)
                attn = ggml_add(ctx0, attn, b.attn_out_b);

            cur = ggml_add(ctx0, residual, attn);
        }

        // FFN: pre-norm LN → up(GELU) → down + residual
        residual = cur;
        ggml_tensor* xn = ggml_norm(ctx0, cur, 1e-5f);
        xn = ggml_mul(ctx0, xn, b.ffn_norm_w);
        if (b.ffn_norm_b)
            xn = ggml_add(ctx0, xn, b.ffn_norm_b);

        xn = ggml_mul_mat(ctx0, b.ffn_up_w, xn);
        if (b.ffn_up_b)
            xn = ggml_add(ctx0, xn, b.ffn_up_b);
        xn = ggml_gelu_erf(ctx0, xn);
        xn = ggml_mul_mat(ctx0, b.ffn_down_w, xn);
        if (b.ffn_down_b)
            xn = ggml_add(ctx0, xn, b.ffn_down_b);

        cur = ggml_add(ctx0, residual, xn);
    }

    // Post-norm LayerNorm
    cur = ggml_norm(ctx0, cur, 1e-5f);
    cur = ggml_mul(ctx0, cur, m.audio.ln_post_w);
    if (m.audio.ln_post_b)
        cur = ggml_add(ctx0, cur, m.audio.ln_post_b);

    ggml_set_name(cur, "encoder_output");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

extern "C" float* mini_omni2_run_encoder(struct mini_omni2_context* ctx, const float* mel_data, int n_mels, int T_mel,
                                         int* out_T_enc, int* out_dim) {
    if (!ctx || !mel_data || n_mels <= 0 || T_mel <= 0)
        return nullptr;

    ggml_cgraph* gf = mo2_build_encoder(ctx, T_mel);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    // Set mel input
    ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_t, mel_data, 0, (size_t)n_mels * T_mel * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* enc_out = ggml_graph_get_tensor(gf, "encoder_output");
    const int T_enc = (int)enc_out->ne[1];
    const int dim = (int)enc_out->ne[0];

    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_dim)
        *out_dim = dim;

    float* result = (float*)malloc((size_t)T_enc * dim * sizeof(float));
    if (!result)
        return nullptr;
    ggml_backend_tensor_get(enc_out, result, 0, (size_t)T_enc * dim * sizeof(float));
    return result;
}

// ===========================================================================
// whisperMLP adapter
// ===========================================================================

extern "C" float* mini_omni2_run_adapter(struct mini_omni2_context* ctx, const float* enc, int T_enc, int enc_dim,
                                         int* out_T, int* out_dim) {
    if (!ctx || !enc || T_enc <= 0 || enc_dim <= 0)
        return nullptr;
    const auto& m = ctx->model;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);

    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, enc_dim, T_enc);
    ggml_set_name(inp, "enc_input");
    ggml_set_input(inp);

    // whisperMLP: SwiGLU(fc_1, fc_2) → proj
    // out = proj(silu(fc_1(x)) * fc_2(x))
    ggml_tensor* gate = ggml_mul_mat(ctx0, m.adapter.fc_1_w, inp);
    ggml_tensor* up = ggml_mul_mat(ctx0, m.adapter.fc_2_w, inp);
    ggml_tensor* mlp = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
    ggml_tensor* out = ggml_mul_mat(ctx0, m.adapter.proj_w, mlp);

    ggml_set_name(out, "adapter_output");
    ggml_set_output(out);

    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    ggml_tensor* inp_t = ggml_graph_get_tensor(gf, "enc_input");
    ggml_backend_tensor_set(inp_t, enc, 0, (size_t)T_enc * enc_dim * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "adapter_output");
    const int out_T_val = (int)out_t->ne[1];
    const int out_dim_val = (int)out_t->ne[0];

    if (out_T)
        *out_T = out_T_val;
    if (out_dim)
        *out_dim = out_dim_val;

    float* result = (float*)malloc((size_t)out_T_val * out_dim_val * sizeof(float));
    if (!result)
        return nullptr;
    ggml_backend_tensor_get(out_t, result, 0, (size_t)out_T_val * out_dim_val * sizeof(float));
    return result;
}

// ===========================================================================
// KV cache
// ===========================================================================

extern "C" bool mini_omni2_kv_init(struct mini_omni2_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    if (ctx->kv_k)
        return true;
    const auto& hp = ctx->model.hp;
    const int hd = hp.llm_head_dim;
    const int n_kv = hp.llm_n_kv_heads;
    const int nl = hp.llm_n_layers;

    ggml_init_params ip = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    ctx->kv_ctx = ggml_init(ip);
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("mini_omni2");
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, nl);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");
    size_t kb = ggml_nbytes(ctx->kv_k), vb = ggml_nbytes(ctx->kv_v);
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "mini_omni2");
    ctx->kv_buf = ggml_backend_alloc_buffer(kv_backend, kb + vb);
    if (!ctx->kv_buf) {
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kb);
    mini_omni2_kv_reset(ctx);
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "mini_omni2: kv cache %zu MiB\n", (kb + vb) >> 20);
    }
    return true;
}

extern "C" void mini_omni2_kv_reset(struct mini_omni2_context* ctx) {
    if (!ctx || !ctx->kv_buf)
        return;
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
}

// ===========================================================================
// LLM forward (stub — will be implemented after encoder/adapter validation)
// ===========================================================================

extern "C" float* mini_omni2_run_llm_kv(struct mini_omni2_context* ctx, const float* inputs_embeds, int n_tokens,
                                        int n_past, int* out_n_tokens, int* out_vocab_size) {
    // TODO: implement after encoder + adapter pass diff validation
    (void)ctx;
    (void)inputs_embeds;
    (void)n_tokens;
    (void)n_past;
    if (out_n_tokens)
        *out_n_tokens = 0;
    if (out_vocab_size)
        *out_vocab_size = 0;
    fprintf(stderr, "mini_omni2: run_llm_kv not yet implemented\n");
    return nullptr;
}

// ===========================================================================
// Full transcribe (stub)
// ===========================================================================

extern "C" char* mini_omni2_transcribe(struct mini_omni2_context* ctx, const float* samples, int n_samples) {
    // TODO: implement after LLM path is validated
    (void)ctx;
    (void)samples;
    (void)n_samples;
    fprintf(stderr, "mini_omni2: transcribe not yet implemented\n");
    return nullptr;
}
