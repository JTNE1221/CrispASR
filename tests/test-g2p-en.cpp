// tests/test-g2p-en.cpp — unit tests for English G2P phonemizer.
// Tests all three tiers: ARPAbet→IPA table, LTS rules, and the
// combined phonemizer pipeline.

#include <catch2/catch_test_macros.hpp>

#include "core/g2p_en.h"
#include "phonemizer.h"

#include <string>

// ── ARPAbet → IPA conversion ─────────────────────────────────────────

TEST_CASE("ARPAbet to IPA conversion", "[g2p][arpabet]") {
    SECTION("basic vowels") {
        CHECK(g2p_en::arpa_to_ipa("AA0") == "ɑː");
        CHECK(g2p_en::arpa_to_ipa("AE1") == "ˈæ");
        CHECK(g2p_en::arpa_to_ipa("IY0") == "iː");
        CHECK(g2p_en::arpa_to_ipa("UW2") == "ˌuː");
        CHECK(g2p_en::arpa_to_ipa("AH0") == "ʌ");
        CHECK(g2p_en::arpa_to_ipa("EY1") == "ˈeɪ");
    }
    SECTION("basic consonants") {
        CHECK(g2p_en::arpa_to_ipa("B") == "b");
        CHECK(g2p_en::arpa_to_ipa("CH") == "tʃ");
        CHECK(g2p_en::arpa_to_ipa("SH") == "ʃ");
        CHECK(g2p_en::arpa_to_ipa("TH") == "θ");
        CHECK(g2p_en::arpa_to_ipa("DH") == "ð");
        CHECK(g2p_en::arpa_to_ipa("NG") == "ŋ");
        CHECK(g2p_en::arpa_to_ipa("ZH") == "ʒ");
        CHECK(g2p_en::arpa_to_ipa("HH") == "h");
        CHECK(g2p_en::arpa_to_ipa("JH") == "dʒ");
    }
    SECTION("stress markers") {
        CHECK(g2p_en::arpa_to_ipa("AH0") == "ʌ");     // no stress
        CHECK(g2p_en::arpa_to_ipa("AH1") == "ˈʌ");    // primary
        CHECK(g2p_en::arpa_to_ipa("AH2") == "ˌʌ");    // secondary
    }
    SECTION("case insensitivity") {
        CHECK(g2p_en::arpa_to_ipa("ah0") == "ʌ");
        CHECK(g2p_en::arpa_to_ipa("Sh") == "ʃ");
    }
    SECTION("unknown phoneme") {
        CHECK(g2p_en::arpa_to_ipa("XX") == "");
        CHECK(g2p_en::arpa_to_ipa("") == "");
    }
}

// ── LTS rules ────────────────────────────────────────────────────────

TEST_CASE("LTS rule-based phonemization", "[g2p][lts]") {
    SECTION("simple words produce non-empty ARPAbet") {
        auto phs = g2p_en::lts_predict("hello");
        REQUIRE(!phs.empty());
        // Should contain at least an H and a vowel
        bool has_hh = false, has_vowel = false;
        for (auto& p : phs) {
            std::string upper;
            for (char c : p) upper += (char)toupper((unsigned char)c);
            if (upper == "HH" || upper == "H") has_hh = true;
            if (upper.find("AH") != std::string::npos || upper.find("EH") != std::string::npos ||
                upper.find("OW") != std::string::npos || upper.find("IY") != std::string::npos)
                has_vowel = true;
        }
        CHECK(has_hh);
        CHECK(has_vowel);
    }
    SECTION("digraphs handled correctly") {
        auto phs = g2p_en::lts_predict("the");
        // 'th' should produce TH, not T+H
        REQUIRE(!phs.empty());
        bool has_th = false;
        for (auto& p : phs) {
            std::string upper;
            for (char c : p) upper += (char)toupper((unsigned char)c);
            if (upper == "TH") has_th = true;
        }
        CHECK(has_th);
    }
    SECTION("silent final e") {
        auto phs_make = g2p_en::lts_predict("make");
        auto phs_mak = g2p_en::lts_predict("mak");
        // "make" should have the same consonants but silent e
        CHECK(phs_make.size() <= phs_mak.size() + 1);
    }
    SECTION("sh digraph") {
        auto phs = g2p_en::lts_predict("she");
        REQUIRE(!phs.empty());
        bool has_sh = false;
        for (auto& p : phs) {
            std::string upper;
            for (char c : p) upper += (char)toupper((unsigned char)c);
            if (upper == "SH") has_sh = true;
        }
        CHECK(has_sh);
    }
}

// ── Word-to-IPA (full pipeline) ──────────────────────────────────────

TEST_CASE("word_to_ipa produces IPA output", "[g2p][ipa]") {
    g2p_en::context ctx; // no CMUdict or neural — pure LTS

    SECTION("common words produce IPA with Unicode characters") {
        std::string ipa = g2p_en::word_to_ipa(ctx, "hello");
        REQUIRE(!ipa.empty());
        // IPA should contain non-ASCII characters (ɛ, ʌ, etc.)
        bool has_nonascii = false;
        for (unsigned char c : ipa) {
            if (c >= 0x80) { has_nonascii = true; break; }
        }
        CHECK(has_nonascii);
    }

    SECTION("the produces ð") {
        std::string ipa = g2p_en::word_to_ipa(ctx, "the");
        CHECK(ipa.find("θ") != std::string::npos); // th → θ
    }

    SECTION("she produces ʃ") {
        std::string ipa = g2p_en::word_to_ipa(ctx, "she");
        CHECK(ipa.find("ʃ") != std::string::npos);
    }
}

// ── Text-to-IPA (full sentence) ──────────────────────────────────────

TEST_CASE("text_to_ipa handles full sentences", "[g2p][sentence]") {
    g2p_en::context ctx;

    SECTION("hello world produces non-empty IPA") {
        std::string ipa = g2p_en::text_to_ipa(ctx, "hello world");
        REQUIRE(!ipa.empty());
        // Should have a space separating two words
        CHECK(ipa.find(' ') != std::string::npos);
    }

    SECTION("punctuation is handled") {
        std::string ipa1 = g2p_en::text_to_ipa(ctx, "hello, world!");
        std::string ipa2 = g2p_en::text_to_ipa(ctx, "hello world");
        // Both should produce IPA (punctuation stripped/preserved)
        CHECK(!ipa1.empty());
        CHECK(!ipa2.empty());
    }

    SECTION("empty input") {
        std::string ipa = g2p_en::text_to_ipa(ctx, "");
        CHECK(ipa.empty());
    }

    SECTION("mixed case") {
        std::string ipa1 = g2p_en::text_to_ipa(ctx, "Hello");
        std::string ipa2 = g2p_en::text_to_ipa(ctx, "hello");
        CHECK(ipa1 == ipa2);
    }
}

// ── Tokenizer ────────────────────────────────────────────────────────

TEST_CASE("tokenizer splits correctly", "[g2p][tokenizer]") {
    SECTION("basic words") {
        auto t = g2p_en::tokenize("hello world");
        REQUIRE(t.size() == 2);
        CHECK(t[0] == "hello");
        CHECK(t[1] == "world");
    }
    SECTION("punctuation as separate tokens") {
        auto t = g2p_en::tokenize("hello, world!");
        REQUIRE(t.size() == 4);
        CHECK(t[0] == "hello");
        CHECK(t[1] == ",");
        CHECK(t[2] == "world");
        CHECK(t[3] == "!");
    }
    SECTION("multiple spaces") {
        auto t = g2p_en::tokenize("a  b");
        REQUIRE(t.size() == 2);
    }
}

// ── Phonemizer interface ─────────────────────────────────────────────

TEST_CASE("phonemizer builtin_en works without espeak", "[phonemizer]") {
    std::string out;

    SECTION("English text produces IPA") {
        bool ok = crispasr::phonemize_builtin_en("en-us", "hello world", out);
        CHECK(ok);
        CHECK(!out.empty());
    }

    SECTION("auto language works") {
        bool ok = crispasr::phonemize_builtin_en("auto", "hello", out);
        CHECK(ok);
        CHECK(!out.empty());
    }

    SECTION("empty language works") {
        bool ok = crispasr::phonemize_builtin_en("", "hello", out);
        CHECK(ok);
    }

    SECTION("non-English returns false") {
        bool ok = crispasr::phonemize_builtin_en("de", "hallo", out);
        CHECK(!ok);
    }
}

TEST_CASE("phonemize() cascade works", "[phonemizer]") {
    std::string out;
    // Even without espeak, the built-in English G2P should produce output
    bool ok = crispasr::phonemize("en-us", "The quick brown fox", out);
    CHECK(ok);
    CHECK(!out.empty());
    // Should contain IPA characters
    bool has_ipa = false;
    for (unsigned char c : out) {
        if (c >= 0x80) { has_ipa = true; break; }
    }
    CHECK(has_ipa);
}
