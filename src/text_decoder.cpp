#include "text_decoder.h"
#include "timing.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <ggml-impl.h>
#include <ggml-cpu.h>

#define QWEN3_ASR_MAX_NODES 8192

namespace qwen3_asr {

TextDecoder::TextDecoder() = default;

TextDecoder::~TextDecoder() {
    free_kv_cache(state_.cache);
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend_gpu) {
        ggml_backend_free(state_.backend_gpu);
        state_.backend_gpu = nullptr;
    }
    if (state_.backend_cpu) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }
    free_decoder_model(model_);
}

void TextDecoder::set_n_threads(int n_threads) {
    n_threads_ = std::max(1, n_threads);
    if (state_.backend_cpu) {
        ggml_backend_cpu_set_n_threads(state_.backend_cpu, n_threads_);
    }
}

bool TextDecoder::load_model(const std::string & model_path) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    
    gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open GGUF file: " + model_path;
        return false;
    }
    
    if (!parse_config(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!create_tensors(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!load_tensor_data(model_path, ctx)) {
        free_decoder_model(model_);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!load_vocab(ctx)) {
        free_decoder_model(model_);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);
    
    state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!state_.backend_cpu) {
        error_msg_ = "Failed to initialize CPU backend";
        return false;
    }

    state_.backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);

    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> backend_bufts;

    if (state_.backend_gpu) {
        backends.push_back(state_.backend_gpu);
        backend_bufts.push_back(ggml_backend_get_default_buffer_type(state_.backend_gpu));
    }

    backends.push_back(state_.backend_cpu);
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_get_default_buffer_type(state_.backend_cpu);
    if (state_.backend_gpu) {
        ggml_backend_dev_t gpu_dev = ggml_backend_get_device(state_.backend_gpu);
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gpu_dev);
        if (host_buft) cpu_buft = host_buft;
    }
    backend_bufts.push_back(cpu_buft);

    state_.sched = ggml_backend_sched_new(backends.data(), backend_bufts.data(), backends.size(), QWEN3_ASR_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }

    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_ASR_MAX_NODES + ggml_graph_overhead());

    set_n_threads(n_threads_);
    
    return true;
}

bool TextDecoder::parse_config(gguf_context * ctx) {
    auto get_u32 = [&](const char * key, int32_t default_val) -> int32_t {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) {
            GGML_LOG_WARN("TextDecoder::parse_config(): Failed to find key '%s', using default value %d\n", key, default_val);
            return default_val;
        }
        return (int32_t)gguf_get_val_u32(ctx, idx);
    };
    
    auto get_f32 = [&](const char * key, float default_val) -> float {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) {
            GGML_LOG_WARN("TextDecoder::parse_config(): Failed to find key '%s', using default value %f\n", key, default_val);
            return default_val;
        }
        return gguf_get_val_f32(ctx, idx);
    };
    
    auto & cfg = model_.config;
    cfg.vocab_size = get_u32("qwen3-asr.vocab_size", 151936);
    cfg.hidden_size = get_u32("qwen3-asr.embedding_length", 1024);
    cfg.n_decoder_layers = get_u32("qwen3-asr.block_count", 28);
    cfg.n_attention_heads = get_u32("qwen3-asr.attention.head_count", 16);
    cfg.n_key_value_heads = get_u32("qwen3-asr.attention.head_count_kv", 8);
    cfg.intermediate_size = get_u32("qwen3-asr.feed_forward_length", 3072);
    cfg.head_dim = get_u32("qwen3-asr.attention.key_length", 128);
    cfg.rms_norm_eps = get_f32("qwen3-asr.attention.layer_norm_rms_epsilon", 1e-6f);
    cfg.rope_theta = get_f32("qwen3-asr.rope.freq_base", 1000000.0f);
    
    cfg.pad_token_id = 151643;
    cfg.eos_token_id = 151645;
    cfg.audio_start_token_id = get_u32("qwen3-asr.audio.start_token_id", 151669);
    cfg.audio_end_token_id = get_u32("qwen3-asr.audio.end_token_id", 151670);
    cfg.audio_pad_token_id = get_u32("qwen3-asr.audio.pad_token_id", 151676);
    
    return true;
}

bool TextDecoder::create_tensors(gguf_context * ctx) {
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const auto & cfg = model_.config;
    
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to create GGML context";
        return false;
    }
    
    model_.layers.resize(cfg.n_decoder_layers);
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        ggml_type type = gguf_get_tensor_type(ctx, i);
        
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        int n_dims = 0;
        
        if (strstr(name, "audio.encoder")) {
            continue;
        } else if (strstr(name, "token_embd.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.vocab_size;
            n_dims = 2;
        } else if (strstr(name, "output_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "attn_output.weight")) {
            ne[0] = cfg.n_attention_heads * cfg.head_dim;
            ne[1] = cfg.hidden_size;
            n_dims = 2;
        } else if (strstr(name, "output.weight")) {
            // Weight tying: output.weight aliases token_embd.weight
            continue;
        } else if (strstr(name, "attn_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "attn_q_norm.weight")) {
            ne[0] = cfg.head_dim;
            n_dims = 1;
        } else if (strstr(name, "attn_k_norm.weight")) {
            ne[0] = cfg.head_dim;
            n_dims = 1;
        } else if (strstr(name, "attn_q.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.n_attention_heads * cfg.head_dim;
            n_dims = 2;
        } else if (strstr(name, "attn_k.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.n_key_value_heads * cfg.head_dim;
            n_dims = 2;
        } else if (strstr(name, "attn_v.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.n_key_value_heads * cfg.head_dim;
            n_dims = 2;
        } else if (strstr(name, "ffn_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "ffn_gate.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.intermediate_size;
            n_dims = 2;
        } else if (strstr(name, "ffn_up.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.intermediate_size;
            n_dims = 2;
        } else if (strstr(name, "ffn_down.weight")) {
            ne[0] = cfg.intermediate_size;
            ne[1] = cfg.hidden_size;
            n_dims = 2;
        } else {
            continue;
        }
        
        struct ggml_tensor * tensor = ggml_new_tensor(model_.ctx, type, n_dims, ne);
        if (!tensor) {
            error_msg_ = "Failed to create tensor: " + std::string(name);
            return false;
        }
        ggml_set_name(tensor, name);
        model_.tensors[name] = tensor;
        
        if (strstr(name, "token_embd.weight")) {
            model_.token_embd = tensor;
        } else if (strstr(name, "output_norm.weight")) {
            model_.output_norm = tensor;
        } else if (strstr(name, "blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < cfg.n_decoder_layers) {
                auto & layer = model_.layers[layer_idx];
                
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        }
    }
    
    // Weight tying: LM head shares token embedding weights
    model_.output = model_.token_embd;
    
    return true;
}

bool TextDecoder::load_tensor_data(const std::string & path, gguf_context * ctx) {
    if (!map_file_readonly(path, model_.mmap, error_msg_)) {
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(ctx);
    if (data_offset > model_.mmap.size) {
        error_msg_ = "Invalid GGUF data offset in file: " + path;
        unmap_file(model_.mmap);
        return false;
    }

    const size_t total_size = model_.mmap.size - data_offset;
    uint8_t * data_base = static_cast<uint8_t *>(model_.mmap.addr) + data_offset;
    
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    size_t max_tensor_size = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        size_t sz = gguf_get_tensor_size(ctx, i);
        if (sz > max_tensor_size) max_tensor_size = sz;
    }

    // Try GPU device buffer (zero-copy on Apple Silicon unified memory)
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    bool use_vram = getenv("QWEN_USE_VRAM") != nullptr;
    if (!use_vram && gpu_dev) {
        const char * dev_name = ggml_backend_dev_name(gpu_dev);
        if (dev_name && (strstr(dev_name, "CUDA") != nullptr || strstr(dev_name, "cuda") != nullptr)) {
            use_vram = true;
        }
    }

    if (use_vram) {
        ggml_backend_t backend = ggml_backend_dev_init(gpu_dev, nullptr);
        model_.buffer = ggml_backend_alloc_ctx_tensors(model_.ctx, backend);
        if (model_.buffer) {
            ggml_backend_buffer_set_usage(model_.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (int64_t i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(ctx, i);
                size_t offset = gguf_get_tensor_offset(ctx, i);
                auto it = model_.tensors.find(name);
                if (it == model_.tensors.end()) continue;
                ggml_backend_tensor_set(it->second, data_base + offset, 0, ggml_nbytes(it->second));
            }
        }
        ggml_backend_free(backend);
    } else {
        if (gpu_dev) {
            model_.buffer = ggml_backend_dev_buffer_from_host_ptr(gpu_dev, data_base, total_size, max_tensor_size);
        }
        if (!model_.buffer) {
            model_.buffer = ggml_backend_cpu_buffer_from_ptr(data_base, total_size);
        }
        if (!model_.buffer) {
            error_msg_ = "Failed to create buffer from mmap";
            unmap_file(model_.mmap);
            return false;
        }

        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            size_t offset = gguf_get_tensor_offset(ctx, i);

            auto it = model_.tensors.find(name);
            if (it == model_.tensors.end()) continue;

            ggml_tensor * tensor = it->second;
            tensor->buffer = model_.buffer;
            tensor->data = data_base + offset;
        }
    }

    return true;
}

bool TextDecoder::init_kv_cache(int32_t n_ctx) {
    const auto & cfg = model_.config;
    
    free_kv_cache(state_.cache);
    
    state_.cache.n_ctx = n_ctx;
    state_.cache.n_used = 0;
    state_.cache.head_dim = cfg.head_dim;
    state_.cache.n_kv_heads = cfg.n_key_value_heads;
    state_.cache.n_layers = cfg.n_decoder_layers;
    
    const size_t n_tensors = cfg.n_decoder_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    state_.cache.ctx = ggml_init(params);
    if (!state_.cache.ctx) {
        error_msg_ = "Failed to create KV cache context";
        return false;
    }
    
    state_.cache.k_cache.resize(cfg.n_decoder_layers);
    state_.cache.v_cache.resize(cfg.n_decoder_layers);
    
    for (int il = 0; il < cfg.n_decoder_layers; ++il) {
        state_.cache.k_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.k_cache[il], "k_cache_%d", il);
        
        state_.cache.v_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.v_cache[il], "v_cache_%d", il);
    }
    
    ggml_backend_t kv_backend = state_.backend_gpu ? state_.backend_gpu : state_.backend_cpu;
    state_.cache.buffer = ggml_backend_alloc_ctx_tensors(state_.cache.ctx, kv_backend);
    if (!state_.cache.buffer) {
        error_msg_ = "Failed to allocate KV cache buffer";
        return false;
    }
    
    return true;
}

void TextDecoder::clear_kv_cache() {
    state_.cache.n_used = 0;
}

struct ggml_cgraph * TextDecoder::build_graph(
    const int32_t * tokens, int32_t n_tokens, int32_t n_past,
    const float * audio_embd, int32_t n_audio, int32_t audio_start_pos) {
    
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_decoder_layers;
    
    ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_ASR_MAX_NODES, false);
    
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    ggml_tensor * inp_audio = nullptr;
    if (audio_embd && n_audio > 0) {
        inp_audio = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_audio);
        ggml_set_name(inp_audio, "inp_audio");
        ggml_set_input(inp_audio);
    }
    
    ggml_tensor * cur = ggml_get_rows(ctx0, model_.token_embd, inp_tokens);
    
    if (inp_audio && n_audio > 0 && audio_start_pos >= 0 && audio_start_pos + n_audio <= n_tokens) {
        ggml_tensor * embd_before = nullptr;
        ggml_tensor * embd_after = nullptr;
        
        if (audio_start_pos > 0) {
            embd_before = ggml_view_2d(ctx0, cur, hidden_size, audio_start_pos,
                                       cur->nb[1], 0);
        }
        
        if (audio_start_pos + n_audio < n_tokens) {
            int after_start = audio_start_pos + n_audio;
            int after_len = n_tokens - after_start;
            embd_after = ggml_view_2d(ctx0, cur, hidden_size, after_len,
                                      cur->nb[1], after_start * cur->nb[1]);
        }
        
        if (embd_before && embd_after) {
            ggml_tensor * tmp = ggml_concat(ctx0, embd_before, inp_audio, 1);
            cur = ggml_concat(ctx0, tmp, embd_after, 1);
        } else if (embd_before) {
            cur = ggml_concat(ctx0, embd_before, inp_audio, 1);
        } else if (embd_after) {
            cur = ggml_concat(ctx0, inp_audio, embd_after, 1);
        } else {
            cur = inp_audio;
        }
        ggml_set_name(cur, "embd_with_audio");
        ggml_set_output(cur);
    }
    
    ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    // Flash attention causal mask: [n_kv, n_tokens], F16
    int n_kv = n_past + n_tokens;
    ggml_tensor * fa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv, n_tokens);
    ggml_set_name(fa_mask, "fa_mask");
    ggml_set_input(fa_mask);
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        if (!layer.attn_norm || !layer.attn_q || !layer.attn_k || !layer.attn_v || 
            !layer.attn_output || !layer.ffn_norm || !layer.ffn_gate || 
            !layer.ffn_up || !layer.ffn_down) {
            return nullptr;
        }
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        ggml_tensor * k_cache = state_.cache.k_cache[il];
        ggml_tensor * v_cache = state_.cache.v_cache[il];
        
        ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2]);
        
        ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2]);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2], 0);
        
        ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2], 0);
        
        // flash_attn_ext expects: Q[head_dim, n_tokens, n_head], K[head_dim, n_kv, n_kv_head], V[head_dim, n_kv, n_kv_head]
        ggml_tensor * Qfa = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        
        cur = ggml_flash_attn_ext(ctx0, Qfa, K, V, fa_mask, KQscale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx0, cur, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
        ggml_format_name(cur, "ffn_out_%d", il);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;

    if (n_tokens > 1) {
        cur = ggml_view_2d(ctx0, cur, hidden_size, 1, cur->nb[1], (n_tokens - 1) * cur->nb[1]);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    ggml_set_name(cur, "result_norm");
    
    cur = ggml_mul_mat(ctx0, model_.output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

bool TextDecoder::forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                          std::vector<float> & output) {
    return forward_with_audio(tokens, n_tokens, nullptr, 0, -1, n_past, output);
}

bool TextDecoder::forward_with_audio(
    const int32_t * tokens, int32_t n_tokens,
    const float * audio_embd, int32_t n_audio,
    int32_t audio_start_pos, int32_t n_past,
    std::vector<float> & output) {
    QWEN3_TIMER("decoder.forward");
    
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (state_.cache.n_ctx == 0) {
        if (!init_kv_cache(1024)) {
            return false;
        }
    }
    
    if (n_past + n_tokens > state_.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }
    
    ggml_cgraph * gf = build_graph(tokens, n_tokens, n_past,
                                          audio_embd, n_audio, audio_start_pos);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (!inp_tokens) {
        error_msg_ = "Failed to find inp_tokens tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    ggml_backend_tensor_set(inp_tokens, tokens, 0, n_tokens * sizeof(int32_t));
    
    ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            positions[i] = n_past + i;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_tokens * sizeof(int32_t));
    }
    
    ggml_tensor * fa_mask_t = ggml_graph_get_tensor(gf, "fa_mask");
    if (fa_mask_t) {
        int n_kv = n_past + n_tokens;
        std::vector<ggml_fp16_t> mask_data((size_t)n_kv * n_tokens);
        const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_f16 = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            for (int k = 0; k < n_kv; ++k) {
                mask_data[k + q * n_kv] = (k <= n_past + q) ? zero_f16 : neginf_f16;
            }
        }
        ggml_backend_tensor_set(fa_mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }
    
    if (audio_embd && n_audio > 0) {
        ggml_tensor * inp_audio = ggml_graph_get_tensor(gf, "inp_audio");
        if (inp_audio) {
            ggml_backend_tensor_set(inp_audio, audio_embd, 0, 
                                    n_audio * model_.config.hidden_size * sizeof(float));
        }
    }
    
    {
        QWEN3_TIMER("decoder.compute");
        if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
    }
    
    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t vocab_size = logits->ne[0];
    int64_t n_logit_rows = logits->ne[1];
    output.resize(n_logit_rows * vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    
    state_.cache.n_used = n_past + n_tokens;
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

bool TextDecoder::forward_debug(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                                std::vector<float> & output,
                                std::map<std::string, std::vector<float>> & debug_tensors) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    if (state_.cache.n_ctx == 0) {
        if (!init_kv_cache(1024)) {
            return false;
        }
    }
    
    ggml_cgraph * gf = build_graph(tokens, n_tokens, n_past, nullptr, 0, -1);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens) {
        ggml_backend_tensor_set(inp_tokens, tokens, 0, n_tokens * sizeof(int32_t));
    }
    
    ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            positions[i] = n_past + i;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_tokens * sizeof(int32_t));
    }
    
    ggml_tensor * fa_mask_t = ggml_graph_get_tensor(gf, "fa_mask");
    if (fa_mask_t) {
        int n_kv = n_past + n_tokens;
        std::vector<ggml_fp16_t> mask_data((size_t)n_kv * n_tokens);
        const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_f16 = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            for (int k = 0; k < n_kv; ++k) {
                mask_data[k + q * n_kv] = (k <= n_past + q) ? zero_f16 : neginf_f16;
            }
        }
        ggml_backend_tensor_set(fa_mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (logits) {
        int64_t vocab_size = logits->ne[0];
        output.resize(n_tokens * vocab_size);
        ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    }
    
    const char * debug_names[] = {"debug_norm0", "debug_q0_raw", "debug_q0_normed", "debug_q0_rope", 
                                   "debug_attn0_out", "debug_kq_scaled", "debug_kq_masked", "debug_kq_softmax"};
    for (const char * name : debug_names) {
        ggml_tensor * t = ggml_graph_get_tensor(gf, name);
        if (t) {
            size_t nbytes = ggml_nbytes(t);
            std::vector<float> data(nbytes / sizeof(float));
            ggml_backend_tensor_get(t, data.data(), 0, nbytes);
            debug_tensors[name] = std::move(data);
        }
    }
    
    state_.cache.n_used = n_past + n_tokens;
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

void free_decoder_model(text_decoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    if (model.mmap.addr) {
        unmap_file(model.mmap);
    }
    model.tensors.clear();
    model.layers.clear();
}

void free_kv_cache(kv_cache & cache) {
    if (cache.buffer) {
        ggml_backend_buffer_free(cache.buffer);
        cache.buffer = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.n_ctx = 0;
    cache.n_used = 0;
}

bool TextDecoder::load_vocab(gguf_context * ctx) {
    int64_t tokens_idx = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tokens_idx < 0) {
        error_msg_ = "Vocabulary not found in GGUF file";
        return false;
    }
    
    int64_t n_vocab = gguf_get_arr_n(ctx, tokens_idx);
    if (n_vocab <= 0) {
        error_msg_ = "Empty vocabulary in GGUF file";
        return false;
    }
    
    vocab_.resize(n_vocab);
    for (int64_t i = 0; i < n_vocab; ++i) {
        vocab_[i] = gguf_get_arr_str(ctx, tokens_idx, i);
    }
    
    return true;
}

// GPT-2 byte-level BPE: reverse mapping from Unicode codepoints back to raw bytes.
// See HuggingFace tokenizers bytes_to_unicode() — printable bytes map to themselves,
// non-printable bytes map to codepoints 256+n.
static std::vector<int> build_unicode_to_byte_table() {
    std::vector byte_to_cp(256, 0);
    std::vector assigned(256, false);

    auto mark = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) {
            byte_to_cp[b] = b;
            assigned[b] = true;
        }
    };
    mark(0x21, 0x7E);
    mark(0xA1, 0xAC);
    mark(0xAE, 0xFF);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!assigned[b]) {
            byte_to_cp[b] = 256 + n;
            ++n;
        }
    }

    std::vector cp_to_byte(512, -1);
    for (int b = 0; b < 256; ++b) {
        cp_to_byte[byte_to_cp[b]] = b;
    }
    return cp_to_byte;
}

std::string TextDecoder::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int32_t)vocab_.size()) {
        return "";
    }

    std::string token = vocab_[token_id];

    if (token == "<asr_text>") {
        return "";
    }

    if (token.size() >= 11 && token.substr(0, 9) == "<|bytes_" &&
        token.substr(token.size() - 2) == "|>") {
        try {
            const int byte_val = std::stoi(token.substr(9, token.size() - 11));
            if (byte_val >= 0 && byte_val <= 255) {
                return std::string(1, static_cast<char>(byte_val));
            }
        } catch (...) {
        }
    }

    // Skip special tokens like <|...|> and [PAD...]
    if (token.size() >= 3 && token[0] == '<' && token[1] == '|' &&
        token[token.size()-1] == '>' && token[token.size()-2] == '|') {
        return "";
    }
    if (token.size() >= 5 && token.substr(0, 4) == "[PAD") {
        return "";
    }

    // Byte-level BPE decode: each Unicode codepoint in the token string maps to a byte
    // via the GPT-2 bytes_to_unicode table.
    static const std::vector<int> cp_to_byte = build_unicode_to_byte_table();

    std::string bytes;
    bytes.reserve(token.size());

    size_t i = 0;
    while (i < token.size()) {
        // Decode one UTF-8 codepoint from the token string
        uint32_t cp = 0;
        auto c = static_cast<unsigned char>(token[i]);
        size_t len = 0;

        if (c < 0x80) {
            cp = c; len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F; len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F; len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07; len = 4;
        } else {
            // Invalid UTF-8 start byte, pass through
            bytes += static_cast<char>(c);
            ++i;
            continue;
        }

        if (i + len > token.size()) {
            // Truncated sequence, pass through remaining bytes
            while (i < token.size()) {
                bytes += token[i++];
            }
            break;
        }

        for (size_t j = 1; j < len; ++j) {
            cp = (cp << 6) | (static_cast<unsigned char>(token[i + j]) & 0x3F);
        }
        i += len;

        // Look up the byte value for this codepoint
        if (cp < cp_to_byte.size() && cp_to_byte[cp] >= 0) {
            bytes += static_cast<char>(cp_to_byte[cp]);
        } else {
            // Codepoint not in BPE table — encode back as UTF-8 (shouldn't happen for valid vocab)
            if (cp < 0x80) {
                bytes += static_cast<char>(cp);
            } else if (cp < 0x800) {
                bytes += static_cast<char>(0xC0 | (cp >> 6));
                bytes += static_cast<char>(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                bytes += static_cast<char>(0xE0 | (cp >> 12));
                bytes += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                bytes += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                bytes += static_cast<char>(0xF0 | (cp >> 18));
                bytes += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                bytes += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                bytes += static_cast<char>(0x80 | (cp & 0x3F));
            }
        }
    }

    return bytes;
}

std::string TextDecoder::decode_tokens(const std::vector<int32_t> & tokens) const {
    std::string result;
    for (int32_t token : tokens) {
        result += decode_token(token);
    }
    return result;
}

} // namespace qwen3_asr
