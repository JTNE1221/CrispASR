// tests/test-g2p-de.cpp — unit tests for German G2P.

#include <catch2/catch_test_macros.hpp>

#include "core/g2p_de.h"

#include <string>

// ── German LTS rules ─────────────────────────────────────────────────

TEST_CASE("German LTS: sch/ch/tsch digraphs", "[g2p_de][lts]") {
    SECTION("sch → ʃ") {
        std::string ipa = g2p_de::lts_word_to_ipa("schule");
        CHECK(ipa.find("\xCA\x83") != std::string::npos); // ʃ
    }
    SECTION("tsch → tʃ") {
        std::string ipa = g2p_de::lts_word_to_ipa("deutsch");
        CHECK(ipa.find("t\xCA\x83") != std::string::npos); // tʃ
    }
    SECTION("ich-Laut: ch after front vowel → ç") {
        std::string ipa = g2p_de::lts_word_to_ipa("ich");
        CHECK(ipa.find("\xC3\xA7") != std::string::npos); // ç
    }
    SECTION("ach-Laut: ch after back vowel → x") {
        std::string ipa = g2p_de::lts_word_to_ipa("dach");
        CHECK(ipa.find("x") != std::string::npos);
    }
}

TEST_CASE("German LTS: vowel digraphs", "[g2p_de][lts]") {
    SECTION("ei → aɪ̯") {
        std::string ipa = g2p_de::lts_word_to_ipa("mein");
        CHECK(ipa.find("a\xC9\xAA") != std::string::npos); // aɪ
    }
    SECTION("eu → ɔʏ̯") {
        std::string ipa = g2p_de::lts_word_to_ipa("heute");
        CHECK(ipa.find("\xC9\x94\xCA\x8F") != std::string::npos); // ɔʏ
    }
    SECTION("au → aʊ̯") {
        std::string ipa = g2p_de::lts_word_to_ipa("haus");
        CHECK(ipa.find("a\xCA\x8A") != std::string::npos); // aʊ
    }
    SECTION("ie → iː") {
        std::string ipa = g2p_de::lts_word_to_ipa("liebe");
        CHECK(ipa.find("i\xCB\x90") != std::string::npos); // iː
    }
}

TEST_CASE("German LTS: consonant specifics", "[g2p_de][lts]") {
    SECTION("z → t͡s") {
        std::string ipa = g2p_de::lts_word_to_ipa("zeit");
        CHECK(ipa.find("ts") != std::string::npos); // t͡s
    }
    SECTION("w → v") {
        std::string ipa = g2p_de::lts_word_to_ipa("welt");
        CHECK(ipa[0] == 'v');
    }
    SECTION("v → f") {
        std::string ipa = g2p_de::lts_word_to_ipa("vater");
        CHECK(ipa[0] == 'f');
    }
    SECTION("initial sp → ʃp") {
        std::string ipa = g2p_de::lts_word_to_ipa("sprechen");
        CHECK(ipa.find("\xCA\x83""p") != std::string::npos); // ʃp
    }
    SECTION("initial st → ʃt") {
        std::string ipa = g2p_de::lts_word_to_ipa("stunde");
        CHECK(ipa.find("\xCA\x83""t") != std::string::npos); // ʃt
    }
    SECTION("ng → ŋ") {
        std::string ipa = g2p_de::lts_word_to_ipa("lang");
        CHECK(ipa.find("\xC5\x8B") != std::string::npos); // ŋ
    }
    SECTION("r → ʁ") {
        std::string ipa = g2p_de::lts_word_to_ipa("rot");
        CHECK(ipa.find("\xCA\x81") != std::string::npos); // ʁ
    }
}

TEST_CASE("German LTS: final schwa and -er", "[g2p_de][lts]") {
    SECTION("final -e → ə") {
        std::string ipa = g2p_de::lts_word_to_ipa("habe");
        CHECK(ipa.find("\xC9\x99") != std::string::npos); // ə
    }
    SECTION("final -er → ɐ") {
        std::string ipa = g2p_de::lts_word_to_ipa("besser");
        CHECK(ipa.find("\xC9\x90") != std::string::npos); // ɐ
    }
}

TEST_CASE("German LTS: s voicing", "[g2p_de][lts]") {
    SECTION("s before vowel → z") {
        std::string ipa = g2p_de::lts_word_to_ipa("sonne");
        CHECK(ipa[0] == 'z');
    }
    SECTION("ss → s") {
        std::string ipa = g2p_de::lts_word_to_ipa("wasser");
        CHECK(ipa.find("s") != std::string::npos);
    }
}

// ── Full word IPA ────────────────────────────────────────────────────

TEST_CASE("German word_to_ipa produces IPA", "[g2p_de][word]") {
    g2p_de::context ctx;
    SECTION("common words produce non-empty IPA with Unicode") {
        std::string ipa = g2p_de::word_to_ipa(ctx, "hallo");
        REQUIRE(!ipa.empty());
    }
    SECTION("welt") {
        std::string ipa = g2p_de::word_to_ipa(ctx, "welt");
        CHECK(ipa[0] == 'v'); // w → v
        CHECK(!ipa.empty());
    }
}

// ── Full text ────────────────────────────────────────────────────────

TEST_CASE("German text_to_ipa handles sentences", "[g2p_de][sentence]") {
    g2p_de::context ctx;
    SECTION("hallo welt") {
        std::string ipa = g2p_de::text_to_ipa(ctx, "hallo welt");
        CHECK(!ipa.empty());
        CHECK(ipa.find(' ') != std::string::npos);
    }
    SECTION("longer sentence") {
        std::string ipa = g2p_de::text_to_ipa(ctx, "Ich bin ein Berliner");
        CHECK(!ipa.empty());
    }
}

// ── Dictionary loading ───────────────────────────────────────────────

TEST_CASE("German IPA dict loading", "[g2p_de][dict]") {
    g2p_de::context ctx;
    int n = g2p_de::load_ipa_dict_file(ctx.dict, "/tmp/ipa_dict_de.txt");
    if (n == 0) {
        SKIP("German IPA dict not available at /tmp/ipa_dict_de.txt");
    }
    INFO("Loaded " << n << " entries");
    CHECK(n > 100000);

    SECTION("hallo lookup") {
        std::string ipa = g2p_de::word_to_ipa(ctx, "hallo");
        CHECK(!ipa.empty());
        // Dict entry should contain IPA characters
        bool has_ipa = false;
        for (unsigned char c : ipa) { if (c >= 0x80) { has_ipa = true; break; } }
        CHECK(has_ipa);
    }
    SECTION("loanword: Restaurant") {
        std::string ipa = g2p_de::word_to_ipa(ctx, "Restaurant");
        CHECK(!ipa.empty());
    }
}
