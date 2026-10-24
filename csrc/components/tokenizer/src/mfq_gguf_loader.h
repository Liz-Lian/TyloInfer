#pragma once

// GGUF metadata adapter used by MFQ's integrated tokenizer.
// Portions derived from ggml-org/llama.cpp.
// SPDX-License-Identifier: MIT

#include "gguf.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

enum mfq_gguf_key {
    TYLOINFER_GGUF_KEY_GENERAL_ARCHITECTURE,
    TYLOINFER_GGUF_KEY_GENERAL_NAME,
    TYLOINFER_GGUF_KEY_VOCAB_SIZE,
    TYLOINFER_GGUF_KEY_TOKENIZER_MODEL,
    TYLOINFER_GGUF_KEY_TOKENIZER_PRE,
    TYLOINFER_GGUF_KEY_TOKENIZER_LIST,
    TYLOINFER_GGUF_KEY_TOKENIZER_TOKEN_TYPE,
    TYLOINFER_GGUF_KEY_TOKENIZER_TOKEN_TYPE_COUNT,
    TYLOINFER_GGUF_KEY_TOKENIZER_SCORES,
    TYLOINFER_GGUF_KEY_TOKENIZER_MERGES,
    TYLOINFER_GGUF_KEY_TOKENIZER_BOS_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_EOS_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_EOG_IDS,
    TYLOINFER_GGUF_KEY_TOKENIZER_EOT_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_EOM_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_UNK_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_SEP_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_PAD_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_MASK_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_ADD_BOS,
    TYLOINFER_GGUF_KEY_TOKENIZER_ADD_EOS,
    TYLOINFER_GGUF_KEY_TOKENIZER_ADD_SEP,
    TYLOINFER_GGUF_KEY_TOKENIZER_ADD_PREFIX,
    TYLOINFER_GGUF_KEY_TOKENIZER_REMOVE_EXTRA_WS,
    TYLOINFER_GGUF_KEY_TOKENIZER_PRECOMPILED_CHARSMAP,
    TYLOINFER_GGUF_KEY_TOKENIZER_NORMALIZER_LOWERCASE,
    TYLOINFER_GGUF_KEY_TOKENIZER_NORMALIZER_STRIP_ACCENTS,
    TYLOINFER_GGUF_KEY_TOKENIZER_FIM_PRE_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_FIM_SUF_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_FIM_MID_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_FIM_PAD_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_FIM_REP_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_FIM_SEP_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_SUPPRESS_TOKENS,
    TYLOINFER_GGUF_KEY_TOKENIZER_PREFIX_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_SUFFIX_ID,
    TYLOINFER_GGUF_KEY_TOKENIZER_MIDDLE_ID,
};

struct mfq_gguf_keys {
    explicit mfq_gguf_keys(std::string architecture = {})
        : architecture(std::move(architecture)) {}

    std::string operator()(mfq_gguf_key key) const {
        switch (key) {
            case TYLOINFER_GGUF_KEY_GENERAL_ARCHITECTURE: return "general.architecture";
            case TYLOINFER_GGUF_KEY_GENERAL_NAME: return "general.name";
            case TYLOINFER_GGUF_KEY_VOCAB_SIZE:
                return (architecture.empty() ? "unknown" : architecture) +
                    ".vocab_size";
            case TYLOINFER_GGUF_KEY_TOKENIZER_MODEL: return "tokenizer.ggml.model";
            case TYLOINFER_GGUF_KEY_TOKENIZER_PRE: return "tokenizer.ggml.pre";
            case TYLOINFER_GGUF_KEY_TOKENIZER_LIST: return "tokenizer.ggml.tokens";
            case TYLOINFER_GGUF_KEY_TOKENIZER_TOKEN_TYPE: return "tokenizer.ggml.token_type";
            case TYLOINFER_GGUF_KEY_TOKENIZER_TOKEN_TYPE_COUNT: return "tokenizer.ggml.token_type_count";
            case TYLOINFER_GGUF_KEY_TOKENIZER_SCORES: return "tokenizer.ggml.scores";
            case TYLOINFER_GGUF_KEY_TOKENIZER_MERGES: return "tokenizer.ggml.merges";
            case TYLOINFER_GGUF_KEY_TOKENIZER_BOS_ID: return "tokenizer.ggml.bos_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_EOS_ID: return "tokenizer.ggml.eos_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_EOG_IDS: return "tokenizer.ggml.eog_token_ids";
            case TYLOINFER_GGUF_KEY_TOKENIZER_EOT_ID: return "tokenizer.ggml.eot_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_EOM_ID: return "tokenizer.ggml.eom_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_UNK_ID: return "tokenizer.ggml.unknown_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_SEP_ID: return "tokenizer.ggml.seperator_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_PAD_ID: return "tokenizer.ggml.padding_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_MASK_ID: return "tokenizer.ggml.mask_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_ADD_BOS: return "tokenizer.ggml.add_bos_token";
            case TYLOINFER_GGUF_KEY_TOKENIZER_ADD_EOS: return "tokenizer.ggml.add_eos_token";
            case TYLOINFER_GGUF_KEY_TOKENIZER_ADD_SEP: return "tokenizer.ggml.add_sep_token";
            case TYLOINFER_GGUF_KEY_TOKENIZER_ADD_PREFIX: return "tokenizer.ggml.add_space_prefix";
            case TYLOINFER_GGUF_KEY_TOKENIZER_REMOVE_EXTRA_WS: return "tokenizer.ggml.remove_extra_whitespaces";
            case TYLOINFER_GGUF_KEY_TOKENIZER_PRECOMPILED_CHARSMAP: return "tokenizer.ggml.precompiled_charsmap";
            case TYLOINFER_GGUF_KEY_TOKENIZER_NORMALIZER_LOWERCASE: return "tokenizer.ggml.normalizer.lowercase";
            case TYLOINFER_GGUF_KEY_TOKENIZER_NORMALIZER_STRIP_ACCENTS: return "tokenizer.ggml.normalizer.strip_accents";
            case TYLOINFER_GGUF_KEY_TOKENIZER_FIM_PRE_ID: return "tokenizer.ggml.fim_pre_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_FIM_SUF_ID: return "tokenizer.ggml.fim_suf_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_FIM_MID_ID: return "tokenizer.ggml.fim_mid_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_FIM_PAD_ID: return "tokenizer.ggml.fim_pad_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_FIM_REP_ID: return "tokenizer.ggml.fim_rep_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_FIM_SEP_ID: return "tokenizer.ggml.fim_sep_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_SUPPRESS_TOKENS: return "tokenizer.ggml.suppress_tokens";
            case TYLOINFER_GGUF_KEY_TOKENIZER_PREFIX_ID: return "tokenizer.ggml.prefix_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_SUFFIX_ID: return "tokenizer.ggml.suffix_token_id";
            case TYLOINFER_GGUF_KEY_TOKENIZER_MIDDLE_ID: return "tokenizer.ggml.middle_token_id";
        }
        throw std::runtime_error("unknown tokenizer metadata key");
    }

    std::string architecture;
};

struct mfq_gguf_loader {
    mfq_gguf_loader(gguf_context * metadata, mfq_gguf_keys keys)
        : metadata(metadata), keys(std::move(keys)) {}

    template<typename T>
    bool get_key(const std::string & key, T & result, bool required = true) const {
        const int index = gguf_find_key(metadata, key.c_str());
        if (index < 0) {
            if (required) throw std::runtime_error("GGUF key not found: " + key);
            return false;
        }
        const gguf_type type = gguf_get_kv_type(metadata, index);
        if constexpr (std::is_same_v<T, std::string>) {
            if (type != GGUF_TYPE_STRING) throw std::runtime_error("GGUF key is not a string: " + key);
            result = gguf_get_val_str(metadata, index);
        } else if constexpr (std::is_same_v<T, bool>) {
            if (type != GGUF_TYPE_BOOL) throw std::runtime_error("GGUF key is not a bool: " + key);
            result = gguf_get_val_bool(metadata, index);
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            if (type == GGUF_TYPE_UINT32) result = gguf_get_val_u32(metadata, index);
            else if (type == GGUF_TYPE_INT32) result = static_cast<uint32_t>(gguf_get_val_i32(metadata, index));
            else throw std::runtime_error("GGUF key is not a 32-bit integer: " + key);
        } else {
            static_assert(!sizeof(T), "unsupported GGUF metadata type");
        }
        return true;
    }

    template<typename T>
    bool get_key(mfq_gguf_key key, T & result, bool required = true) const {
        return get_key(keys(key), result, required);
    }

    gguf_context * metadata;
    mfq_gguf_keys keys;
};
