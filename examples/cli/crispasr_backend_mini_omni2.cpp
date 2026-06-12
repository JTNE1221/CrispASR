// crispasr_backend_mini_omni2.cpp — adapter for Mini-Omni2.
//
// Whisper-small encoder + whisperMLP adapter + Qwen2-0.5B LLM.
// Audio understanding (not pure ASR) — the model responds to speech.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "mini_omni2.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

class MiniOmni2Backend : public CrispasrBackend {
public:
    MiniOmni2Backend() = default;
    ~MiniOmni2Backend() override { MiniOmni2Backend::shutdown(); }

    const char* name() const override { return "mini-omni2"; }

    uint32_t capabilities() const override { return CAP_AUTO_DOWNLOAD; }

    bool init(const whisper_params& p) override {
        mini_omni2_context_params mp = mini_omni2_context_default_params();
        mp.n_threads = p.n_threads;
        mp.verbosity = p.no_prints ? 0 : 1;
        mp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = mini_omni2_init_from_file(p.model.c_str(), mp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[mini-omni2]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        char* text = mini_omni2_transcribe(ctx_, samples, n_samples);
        if (!text)
            return out;

        crispasr_segment seg;
        seg.text = text;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples * 100.0 / 16000.0);
        free(text);

        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            mini_omni2_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    mini_omni2_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_mini_omni2_backend() {
    return std::make_unique<MiniOmni2Backend>();
}
