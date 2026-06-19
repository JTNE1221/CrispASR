// test-wyoming-proto.cpp — unit tests for Wyoming wire format, PCM conversion,
// and linear-interpolation resampler.
//
// Tests the logic that lives in examples/cli/wyoming.cpp without needing a
// running server or any model. On POSIX, the wire-format tests use socketpair
// to drive a loopback read/write. On Windows, those cases are compiled out
// (socketpair not available) but the math tests still run.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Inline copies of wyoming.cpp pure-math helpers (anon-namespace originals
// cannot be called from outside the TU, so we replicate them here and test
// the same invariants).
// ---------------------------------------------------------------------------

static std::vector<float> resample_f32(const float* src, int n_src, int src_rate, int dst_rate) {
    if (src_rate == dst_rate)
        return std::vector<float>(src, src + n_src);
    const int n_dst = (int)((double)n_src * dst_rate / src_rate + 0.5);
    std::vector<float> out((size_t)n_dst);
    const double ratio = (double)(n_src - 1) / std::max(n_dst - 1, 1);
    for (int i = 0; i < n_dst; i++) {
        double pos = i * ratio;
        int i0 = (int)pos;
        int i1 = std::min(i0 + 1, n_src - 1);
        float alpha = (float)(pos - (double)i0);
        out[(size_t)i] = src[i0] * (1.0f - alpha) + src[i1] * alpha;
    }
    return out;
}

static std::vector<int16_t> f32_to_s16(const float* src, int n) {
    std::vector<int16_t> out((size_t)n);
    for (int i = 0; i < n; i++) {
        float v = src[i];
        if (v > 1.0f)
            v = 1.0f;
        if (v < -1.0f)
            v = -1.0f;
        out[(size_t)i] = (int16_t)(v * 32767.0f);
    }
    return out;
}

static std::vector<float> s16_to_f32(const int16_t* src, int n) {
    std::vector<float> out((size_t)n);
    for (int i = 0; i < n; i++)
        out[(size_t)i] = (float)src[i] / 32768.0f;
    return out;
}

// ---------------------------------------------------------------------------
// f32 ↔ s16 conversion
// ---------------------------------------------------------------------------

TEST_CASE("Wyoming f32→s16 clamping", "[unit][wyoming]") {
    float over[4] = {2.0f, 1.0f, -1.0f, -2.0f};
    auto s16 = f32_to_s16(over, 4);
    REQUIRE(s16[0] == 32767);
    REQUIRE(s16[1] == 32767);
    REQUIRE(s16[2] == -32767);
    REQUIRE(s16[3] == -32767);
}

TEST_CASE("Wyoming s16→f32 zero maps to 0.0", "[unit][wyoming]") {
    int16_t zeros[4] = {0, 0, 0, 0};
    auto f32 = s16_to_f32(zeros, 4);
    for (auto v : f32)
        REQUIRE(v == 0.0f);
}

TEST_CASE("Wyoming f32→s16→f32 roundtrip within quantisation error", "[unit][wyoming]") {
    float orig[3] = {0.5f, -0.5f, 0.0f};
    auto s16 = f32_to_s16(orig, 3);
    auto rt = s16_to_f32(s16.data(), 3);
    const float eps = 2.0f / 32768.0f;
    REQUIRE(std::abs(rt[0] - 0.5f) < eps);
    REQUIRE(std::abs(rt[1] + 0.5f) < eps);
    REQUIRE(rt[2] == 0.0f);
}

TEST_CASE("Wyoming s16→f32 sign preserved", "[unit][wyoming]") {
    int16_t samples[2] = {16384, -16384};
    auto f32 = s16_to_f32(samples, 2);
    REQUIRE(f32[0] > 0.0f);
    REQUIRE(f32[1] < 0.0f);
    // Magnitude should be ~0.5
    REQUIRE(f32[0] == Catch::Approx(0.5f).epsilon(0.01));
    REQUIRE(f32[1] == Catch::Approx(-0.5f).epsilon(0.01));
}

// ---------------------------------------------------------------------------
// resample_f32
// ---------------------------------------------------------------------------

TEST_CASE("Wyoming resample_f32 identity (same rate)", "[unit][wyoming]") {
    float src[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    auto out = resample_f32(src, 4, 16000, 16000);
    REQUIRE(out.size() == 4);
    for (int i = 0; i < 4; i++)
        REQUIRE(out[(size_t)i] == Catch::Approx(src[i]));
}

TEST_CASE("Wyoming resample_f32 upsample 2x — output length and endpoints", "[unit][wyoming]") {
    float src[3] = {0.0f, 1.0f, 0.0f};
    auto out = resample_f32(src, 3, 8000, 16000);
    // n_dst = round(3 * 16000/8000) = 6
    REQUIRE(out.size() == 6);
    REQUIRE(out.front() == Catch::Approx(src[0]));
    REQUIRE(out.back() == Catch::Approx(src[2]));
    // Linear interpolation: grid at pos = i * 2/5, so src[1]=1.0 is not hit
    // exactly (no integer i satisfies i*0.4 == 1.0). Peak is 0.8 (at i=2,3).
    float peak = *std::max_element(out.begin(), out.end());
    REQUIRE(peak >= 0.75f);
    REQUIRE(peak <= 1.0f);
    // All intermediate values must be non-negative (positive triangle pulse)
    for (auto v : out)
        REQUIRE(v >= 0.0f);
}

TEST_CASE("Wyoming resample_f32 downsample 2x — DC preserved", "[unit][wyoming]") {
    std::vector<float> src(100, 0.75f);
    auto out = resample_f32(src.data(), 100, 16000, 8000);
    // n_dst = round(100 * 8000/16000) = 50
    REQUIRE(out.size() == 50);
    for (auto v : out)
        REQUIRE(v == Catch::Approx(0.75f));
}

TEST_CASE("Wyoming resample_f32 4x upsample — endpoints preserved", "[unit][wyoming]") {
    float src[5] = {0.1f, 0.5f, 0.9f, 0.3f, 0.7f};
    auto out = resample_f32(src, 5, 24000, 96000);
    // n_dst = round(5 * 96000/24000) = 20
    REQUIRE(out.size() == 20);
    REQUIRE(out.front() == Catch::Approx(src[0]));
    REQUIRE(out.back() == Catch::Approx(src[4]));
}

TEST_CASE("Wyoming resample_f32 single sample — no crash", "[unit][wyoming]") {
    float src[1] = {0.5f};
    auto out = resample_f32(src, 1, 16000, 8000);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0] == Catch::Approx(0.5f));
}

// ---------------------------------------------------------------------------
// Wyoming wire-format (POSIX only — socketpair)
// ---------------------------------------------------------------------------
#ifndef _WIN32

#include <sys/socket.h>
#include <unistd.h>

// Minimal JSON-line writer: produces {"type":"T","data":{},"payload_length":N}\n
static void write_wyoming_header(int fd, const char* type, int payload_len) {
    std::string line =
        std::string("{\"type\":\"") + type + "\",\"data\":{},\"payload_length\":" + std::to_string(payload_len) + "}\n";
    send(fd, line.data(), line.size(), MSG_NOSIGNAL);
}

static std::string read_line(int fd) {
    std::string line;
    char c;
    while (true) {
        auto r = recv(fd, &c, 1, 0);
        if (r <= 0)
            return line;
        if (c == '\n')
            return line;
        line += c;
    }
}

TEST_CASE("Wyoming wire format — header round-trip via socketpair", "[unit][wyoming]") {
    int fds[2];
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);

    write_wyoming_header(fds[1], "describe", 0);
    ::shutdown(fds[1], SHUT_WR);

    std::string line = read_line(fds[0]);
    close(fds[0]);
    close(fds[1]);

    REQUIRE(!line.empty());
    // Must start with '{' and contain the type key
    REQUIRE(line[0] == '{');
    REQUIRE(line.find("\"type\"") != std::string::npos);
    REQUIRE(line.find("\"describe\"") != std::string::npos);
    REQUIRE(line.find("\"payload_length\"") != std::string::npos);
}

TEST_CASE("Wyoming wire format — binary payload framing", "[unit][wyoming]") {
    int fds[2];
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);

    std::vector<int16_t> samples = {100, 200, -300, 400};
    int payload_len = (int)(samples.size() * sizeof(int16_t));

    write_wyoming_header(fds[1], "audio-chunk", payload_len);
    send(fds[1], samples.data(), (size_t)payload_len, MSG_NOSIGNAL);
    ::shutdown(fds[1], SHUT_WR);

    // Read header line
    std::string line = read_line(fds[0]);
    REQUIRE(line.find("\"audio-chunk\"") != std::string::npos);
    REQUIRE(line.find(std::to_string(payload_len)) != std::string::npos);

    // Read payload
    std::vector<int16_t> recv_buf(samples.size());
    ssize_t got = recv(fds[0], recv_buf.data(), (size_t)payload_len, MSG_WAITALL);
    close(fds[0]);
    close(fds[1]);

    REQUIRE(got == payload_len);
    REQUIRE(recv_buf == samples);
}

TEST_CASE("Wyoming wire format — multiple sequential messages", "[unit][wyoming]") {
    int fds[2];
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);

    const char* types[] = {"transcribe", "audio-start", "audio-stop"};
    for (auto t : types)
        write_wyoming_header(fds[1], t, 0);
    ::shutdown(fds[1], SHUT_WR);

    for (auto t : types) {
        std::string line = read_line(fds[0]);
        REQUIRE(line.find(t) != std::string::npos);
    }
    close(fds[0]);
    close(fds[1]);
}

#endif // !_WIN32
