// core/g2p_de.h — German grapheme-to-phoneme (text → IPA).
//
// Two-tier pipeline:
//   1. Dictionary lookup (IPA dict from open-dict-data, CC-BY-SA, auto-downloadable)
//   2. Rule-based LTS (German orthography rules, MIT, always available)
//
// German is much more regular than English — rules cover ~90-95% of
// native words correctly. Loanwords (Chance, Restaurant) need the dict.

#pragma once

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace g2p_de {

// ── Dictionary ──────────────────────────────────────────────────────

struct dictionary {
    // word (lowercase) → IPA string (e.g. "halo" → "halˈoː")
    std::map<std::string, std::string> entries;
    bool loaded = false;
};

// Load open-dict-data format: "word\t/IPA1/, /IPA2/\n"
// Takes first pronunciation only. Returns entry count.
inline int load_ipa_dict_file(dictionary& dict, const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return 0;
    char line[1024];
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        // Find tab separator
        char* tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        std::string word = line;
        // Lowercase the word for lookup
        for (auto& c : word) c = (char)tolower((unsigned char)c);
        // Skip if already present (keep first pronunciation)
        if (dict.entries.count(word)) continue;
        // Parse IPA: skip leading /, take up to next / or comma
        char* ipa_start = tab + 1;
        while (*ipa_start == '/' || *ipa_start == ' ') ipa_start++;
        std::string ipa;
        for (char* p = ipa_start; *p && *p != '/' && *p != ','; p++) {
            ipa += *p;
        }
        // Trim trailing whitespace
        while (!ipa.empty() && (ipa.back() == ' ' || ipa.back() == '/')) ipa.pop_back();
        if (!ipa.empty() && !word.empty()) {
            dict.entries[word] = ipa;
            count++;
        }
    }
    fclose(f);
    dict.loaded = count > 0;
    return count;
}

// ── Rule-based German G2P ───────────────────────────────────────────
// German orthography is largely regular. These rules handle the main
// patterns. Covers native German words well; loanwords need the dict.

inline std::string lts_word_to_ipa(const std::string& word) {
    std::string ipa;
    std::string w;
    for (char c : word) w += (char)tolower((unsigned char)c);
    int len = (int)w.size();

    for (int i = 0; i < len;) {
        // Helper: lookahead characters
        auto at = [&](int offset) -> char {
            int idx = i + offset;
            return (idx >= 0 && idx < len) ? w[idx] : 0;
        };
        char c = at(0), c1 = at(1), c2 = at(2), c3 = at(3);

        // --- 4-char sequences ---
        if (c == 't' && c1 == 's' && c2 == 'c' && c3 == 'h') { ipa += "t\xCA\x83"; i += 4; continue; } // tsch → tʃ

        // --- 3-char sequences ---
        if (c == 's' && c1 == 'c' && c2 == 'h') { ipa += "\xCA\x83"; i += 3; continue; }     // sch → ʃ
        if (c == 'c' && c1 == 'h' && c2 == 's') { ipa += "ks"; i += 3; continue; }            // chs → ks

        // --- 2-char sequences ---
        // ch: ç after front vowels (e,i,ä,ö,ü,ei,eu), x after back vowels (a,o,u,au)
        if (c == 'c' && c1 == 'h') {
            char prev = (i > 0) ? w[i-1] : 0;
            if (prev == 'a' || prev == 'o' || prev == 'u') {
                ipa += "x"; // ach-Laut
            } else {
                ipa += "\xC3\xA7"; // ç ich-Laut
            }
            i += 2; continue;
        }
        if (c == 'c' && c1 == 'k') { ipa += "k"; i += 2; continue; }
        if (c == 'p' && c1 == 'h') { ipa += "f"; i += 2; continue; }
        if (c == 'p' && c1 == 'f') { ipa += "pf"; i += 2; continue; }  // pf → p͡f
        if (c == 't' && c1 == 'h') { ipa += "t"; i += 2; continue; }              // th → t
        if (c == 't' && c1 == 'z') { ipa += "ts"; i += 2; continue; }   // tz → t͡s
        if (c == 'd' && c1 == 't') { ipa += "t"; i += 2; continue; }
        if (c == 'n' && c1 == 'g') { ipa += "\xC5\x8B"; i += 2; continue; }       // ng → ŋ
        if (c == 'n' && c1 == 'k') { ipa += "\xC5\x8B""k"; i += 2; continue; }    // nk → ŋk
        if (c == 'q' && c1 == 'u') { ipa += "kv"; i += 2; continue; }

        // Vowel digraphs
        if (c == 'e' && c1 == 'i') { ipa += "a\xC9\xAA\xCC\xAF"; i += 2; continue; }   // ei → aɪ̯
        if (c == 'e' && c1 == 'u') { ipa += "\xC9\x94\xCA\x8F\xCC\xAF"; i += 2; continue; } // eu → ɔʏ̯
        if (c == 'a' && c1 == 'u') { ipa += "a\xCA\x8A\xCC\xAF"; i += 2; continue; }   // au → aʊ̯
        if (c == 'i' && c1 == 'e') { ipa += "i\xCB\x90"; i += 2; continue; }             // ie → iː
        if (c == 'e' && c1 == 'e') { ipa += "e\xCB\x90"; i += 2; continue; }             // ee → eː
        if (c == 'o' && c1 == 'o') { ipa += "o\xCB\x90"; i += 2; continue; }             // oo → oː
        if (c == 'a' && c1 == 'a') { ipa += "a\xCB\x90"; i += 2; continue; }             // aa → aː
        if (c == 'e' && c1 == 'h') { ipa += "e\xCB\x90"; i += 2; continue; }             // eh → eː
        if (c == 'a' && c1 == 'h') { ipa += "a\xCB\x90"; i += 2; continue; }             // ah → aː
        if (c == 'o' && c1 == 'h') { ipa += "o\xCB\x90"; i += 2; continue; }             // oh → oː
        if (c == 'u' && c1 == 'h') { ipa += "u\xCB\x90"; i += 2; continue; }             // uh → uː
        if (c == 'i' && c1 == 'h') { ipa += "i\xCB\x90"; i += 2; continue; }             // ih → iː

        // ä-digraphs
        // äu → ɔʏ̯ (encoded as UTF-8 ä = C3 A4)
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xA4) {
            // ä followed by u?
            if (at(2) == 'u') { ipa += "\xC9\x94\xCA\x8F\xCC\xAF"; i += 3; continue; }   // äu → ɔʏ̯
            ipa += "\xC9\x9B"; i += 2; continue;  // ä → ɛ
        }
        // ö = C3 B6
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xB6) {
            if (at(2) == 'h') { ipa += "\xC3\xB8\xCB\x90"; i += 3; continue; } // öh → øː
            ipa += "\xC5\x93"; i += 2; continue; // ö → œ (short)
        }
        // ü = C3 BC
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0xBC) {
            if (at(2) == 'h') { ipa += "y\xCB\x90"; i += 3; continue; } // üh → yː
            ipa += "y"; i += 2; continue; // ü → y (short)
        }

        // Double consonants (gemination = short preceding vowel marker, single sound)
        if (c == c1 && c1 >= 'a' && c1 <= 'z' && c != 'e' && c != 'a' && c != 'o') {
            // Just emit single consonant, skip double
            // Fall through to single-char rules below
        }

        // st/sp at start → ʃt/ʃp
        if (i == 0 || (i > 0 && (w[i-1] == ' ' || w[i-1] == '-'))) {
            if (c == 's' && c1 == 't') { ipa += "\xCA\x83""t"; i += 2; continue; }
            if (c == 's' && c1 == 'p') { ipa += "\xCA\x83""p"; i += 2; continue; }
        }
        if (c == 's' && c1 == 's') { ipa += "s"; i += 2; continue; }
        // ß = C3 9F
        if ((unsigned char)c == 0xC3 && (unsigned char)c1 == 0x9F) { ipa += "s"; i += 2; continue; }

        // --- Single characters ---
        if (c == 'a') { ipa += "a"; i++; continue; }
        if (c == 'b') { ipa += "b"; i++; continue; }
        if (c == 'c') { ipa += "k"; i++; continue; }  // standalone c → k
        if (c == 'd') { ipa += "d"; i++; continue; }
        if (c == 'e') {
            // Final -e is schwa, -er is ɐ
            if (i == len - 1) { ipa += "\xC9\x99"; i++; continue; }  // schwa
            if (c1 == 'r' && (i + 2 == len || at(2) == ' ' || at(2) == '-')) {
                ipa += "\xC9\x90"; i += 2; continue; // -er → ɐ
            }
            ipa += "\xC9\x9B"; i++; continue; // short e → ɛ
        }
        if (c == 'f') { ipa += "f"; i++; continue; }
        if (c == 'g') { ipa += "\xC9\xA1"; i++; continue; }  // g → ɡ
        if (c == 'h') {
            // h after vowel is silent (lengthening marker), handled in digraphs
            ipa += "h"; i++; continue;
        }
        if (c == 'i') { ipa += "\xC9\xAA"; i++; continue; }  // short i → ɪ
        if (c == 'j') { ipa += "j"; i++; continue; }
        if (c == 'k') { ipa += "k"; i++; continue; }
        if (c == 'l') { ipa += "l"; i++; continue; }
        if (c == 'm') { ipa += "m"; i++; continue; }
        if (c == 'n') { ipa += "n"; i++; continue; }
        if (c == 'o') { ipa += "\xC9\x94"; i++; continue; }  // short o → ɔ
        if (c == 'p') { ipa += "p"; i++; continue; }
        if (c == 'r') { ipa += "\xCA\x81"; i++; continue; }  // r → ʁ
        if (c == 's') {
            // s before vowel → z
            if (c1 == 'a' || c1 == 'e' || c1 == 'i' || c1 == 'o' || c1 == 'u' ||
                (unsigned char)c1 == 0xC3) {
                ipa += "z"; i++; continue;
            }
            ipa += "s"; i++; continue;
        }
        if (c == 't') { ipa += "t"; i++; continue; }
        if (c == 'u') { ipa += "\xCA\x8A"; i++; continue; }  // short u → ʊ
        if (c == 'v') { ipa += "f"; i++; continue; }  // German v → f (mostly)
        if (c == 'w') { ipa += "v"; i++; continue; }  // German w → v
        if (c == 'x') { ipa += "ks"; i++; continue; }
        if (c == 'y') { ipa += "y"; i++; continue; }
        if (c == 'z') { ipa += "ts"; i++; continue; } // z → t͡s

        i++; // skip unknown
    }
    return ipa;
}

// ── Context ─────────────────────────────────────────────────────────

struct context {
    dictionary dict;
};

// ── Tokenizer ───────────────────────────────────────────────────────

inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        if (c == ' ' || c == ',' || c == '.' || c == '!' || c == '?' ||
            c == ';' || c == ':' || c == '-' || c == '\n') {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
            if (c != ' ') tokens.push_back(std::string(1, c));
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

// ── Main API ────────────────────────────────────────────────────────

inline std::string word_to_ipa(const context& ctx, const std::string& word) {
    // Lowercase for dict lookup
    std::string lower;
    for (size_t i = 0; i < word.size(); i++) {
        unsigned char c = (unsigned char)word[i];
        if (c >= 'A' && c <= 'Z') lower += (char)(c + 32);
        else lower += word[i];
    }
    // Tier 1: dictionary
    if (ctx.dict.loaded) {
        auto it = ctx.dict.entries.find(lower);
        if (it != ctx.dict.entries.end()) return it->second;
    }
    // Tier 2: rules
    return lts_word_to_ipa(lower);
}

inline std::string text_to_ipa(const context& ctx, const std::string& text) {
    auto words = tokenize(text);
    std::string ipa;
    for (const auto& w : words) {
        if (w.size() == 1 && (w[0] == ',' || w[0] == '.' || w[0] == '!' ||
            w[0] == '?' || w[0] == ';' || w[0] == ':' || w[0] == '-')) {
            continue;
        }
        if (!ipa.empty()) ipa += ' ';
        ipa += word_to_ipa(ctx, w);
    }
    return ipa;
}

} // namespace g2p_de
