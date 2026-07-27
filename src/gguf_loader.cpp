#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <ggml-impl.h>

namespace qwen3_asr {

GGUFLoader::GGUFLoader() = default;

GGUFLoader::~GGUFLoader() = default;

bool GGUFLoader::load(const std::string & path, audio_encoder_model & model) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    
    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open GGUF file: " + path;
        return false;
    }
    if (!parse_hparams(ctx, model) || !create_tensors(ctx, model) || !load_tensor_data(path, ctx, model)) {
        free_model(model);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);
    return true;
}

bool GGUFLoader::parse_hparams(gguf_context * ctx, audio_encoder_model & model) {
    auto get_u32 = [&](const char * key, int32_t default_val) -> int32_t {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) {
            GGML_LOG_WARN("GGUFLoader::parse_hparams(): Failed to find key '%s', using default value %d\n", key, default_val);
            return default_val;
        }
        return (int32_t)gguf_get_val_u32(ctx, idx);
    };
    
    auto get_f32 = [&](const char * key, float default_val) -> float {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) {
            GGML_LOG_WARN("GGUFLoader::parse_hparams(): Failed to find key '%s', using default value %f\n", key, default_val);
            return default_val;
        }
        return gguf_get_val_f32(ctx, idx);
    };
    
    auto & hp = model.hparams;
    hp.n_encoder_layers     = get_u32("qwen3-asr.audio.encoder.layer_count", 18);
    hp.d_model              = get_u32("qwen3-asr.audio.encoder.embedding_length", 896);
    hp.n_attention_heads    = get_u32("qwen3-asr.audio.encoder.attention.head_count", 14);
    hp.ffn_dim              = get_u32("qwen3-asr.audio.encoder.feed_forward_length", 3584);
    hp.conv_channels        = get_u32("qwen3-asr.audio.conv_channels", 480);
    hp.conv_out_dim         = get_u32("qwen3-asr.audio.encoder.embedding_length", 896);
    hp.n_mel_bins           = get_u32("qwen3-asr.audio.num_mel_bins", 128);
    hp.n_window_infer       = 800;      // Default value from Qwen3-ASR, hardcoded directly
    hp.layer_norm_eps       = get_f32("qwen3-asr.attention.layer_norm_rms_epsilon", 1e-5f);
    
    auto & thp = model.text_hparams;
    thp.hidden_size         = get_u32("qwen3-asr.embedding_length", 1024);
    thp.n_decoder_layers    = get_u32("qwen3-asr.block_count", 28);
    thp.n_attention_heads   = get_u32("qwen3-asr.attention.head_count", 16);
    thp.n_key_value_heads   = get_u32("qwen3-asr.attention.head_count_kv", 8);
    thp.intermediate_size   = get_u32("qwen3-asr.feed_forward_length", 6144);
    thp.rms_norm_eps        = get_f32("qwen3-asr.attention.layer_norm_rms_epsilon", 1e-6f);
    
    return true;
}

bool GGUFLoader::create_tensors(gguf_context * ctx, audio_encoder_model & model) {
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    model.ctx = ggml_init(params);
    if (!model.ctx) {
        error_msg_ = "Failed to create GGML context";
        return false;
    }
    
    model.layers.resize(model.hparams.n_encoder_layers);
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        ggml_type type = gguf_get_tensor_type(ctx, i);
        
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        int n_dims = 0;
        
        size_t offset = gguf_get_tensor_offset(ctx, i);
        size_t size = gguf_get_tensor_size(ctx, i);
        
        (void)offset;
        
        size_t type_size = ggml_type_size(type);
        size_t blck_size = ggml_blck_size(type);
        
        int64_t total_elements = 1;
        
        if (type_size > 0 && blck_size > 0) {
            total_elements = (size * blck_size) / type_size;
        }
        
        if (strstr(name, "encoder.conv1.weight")) {
            ne[0] = 3; ne[1] = 3; ne[2] = 1; ne[3] = model.hparams.conv_channels;
            n_dims = 4;
        } else if (strstr(name, "encoder.conv2.weight") || strstr(name, "encoder.conv3.weight")) {
            ne[0] = 3; ne[1] = 3; ne[2] = model.hparams.conv_channels; ne[3] = model.hparams.conv_channels;
            n_dims = 4;
        } else if ((strstr(name, "encoder.conv1.bias") || strstr(name, "encoder.conv2.bias") || 
                    strstr(name, "encoder.conv3.bias"))) {
            ne[0] = model.hparams.conv_channels;
            n_dims = 1;
        } else if (strstr(name, "encoder.conv_out.weight")) {
            ne[0] = model.hparams.conv_channels * 16;
            ne[1] = model.hparams.d_model;
            n_dims = 2;
        } else if (strstr(name, "attn_q.weight") || strstr(name, "attn_k.weight") || 
                   strstr(name, "attn_v.weight") || strstr(name, "attn_out.weight")) {
            ne[0] = model.hparams.d_model;
            ne[1] = model.hparams.d_model;
            n_dims = 2;
        } else if (strstr(name, "attn_q.bias") || strstr(name, "attn_k.bias") || 
                   strstr(name, "attn_v.bias") || strstr(name, "attn_out.bias") ||
                   strstr(name, "attn_norm.weight") || strstr(name, "attn_norm.bias")) {
            ne[0] = model.hparams.d_model;
            n_dims = 1;
        } else if (strstr(name, "ffn_up.weight")) {
            ne[0] = model.hparams.d_model;
            ne[1] = model.hparams.ffn_dim;
            n_dims = 2;
        } else if (strstr(name, "ffn_down.weight")) {
            ne[0] = model.hparams.ffn_dim;
            ne[1] = model.hparams.d_model;
            n_dims = 2;
        } else if (strstr(name, "ffn_up.bias")) {
            ne[0] = model.hparams.ffn_dim;
            n_dims = 1;
        } else if (strstr(name, "ffn_down.bias") || strstr(name, "ffn_norm.weight") || 
                   strstr(name, "ffn_norm.bias")) {
            ne[0] = model.hparams.d_model;
            n_dims = 1;
        } else if (strstr(name, "ln_post.weight") || strstr(name, "ln_post.bias")) {
            ne[0] = model.hparams.d_model;
            n_dims = 1;
        } else if (strstr(name, "encoder.proj1.weight")) {
            ne[0] = model.hparams.d_model;
            ne[1] = model.hparams.d_model;
            n_dims = 2;
        } else if (strstr(name, "encoder.proj1.bias")) {
            ne[0] = model.hparams.d_model;
            n_dims = 1;
        } else if (strstr(name, "encoder.proj2.weight")) {
            ne[0] = model.hparams.d_model;
            ne[1] = model.text_hparams.hidden_size;
            n_dims = 2;
        } else if (strstr(name, "encoder.proj2.bias")) {
            ne[0] = model.text_hparams.hidden_size;
            n_dims = 1;
        } else {
            int64_t remaining = total_elements;
            ne[0] = remaining;
            n_dims = 1;
        }
        
        ggml_tensor * tensor = ggml_new_tensor(model.ctx, type, n_dims, ne);
        if (!tensor) {
            error_msg_ = "Failed to create tensor: " + std::string(name);
            return false;
        }
        ggml_set_name(tensor, name);
        model.tensors[name] = tensor;
        
        if (strstr(name, "encoder.conv1.weight")) {
            model.conv2d1_w = tensor;
        } else if (strstr(name, "encoder.conv1.bias")) {
            model.conv2d1_b = tensor;
        } else if (strstr(name, "encoder.conv2.weight")) {
            model.conv2d2_w = tensor;
        } else if (strstr(name, "encoder.conv2.bias")) {
            model.conv2d2_b = tensor;
        } else if (strstr(name, "encoder.conv3.weight")) {
            model.conv2d3_w = tensor;
        } else if (strstr(name, "encoder.conv3.bias")) {
            model.conv2d3_b = tensor;
        } else if (strstr(name, "encoder.conv_out.weight")) {
            model.conv_out_w = tensor;
        } else if (strstr(name, "encoder.ln_post.weight")) {
            model.ln_post_w = tensor;
        } else if (strstr(name, "encoder.ln_post.bias")) {
            model.ln_post_b = tensor;
        } else if (strstr(name, "encoder.proj1.weight")) {
            model.proj1_w = tensor;
        } else if (strstr(name, "encoder.proj1.bias")) {
            model.proj1_b = tensor;
        } else if (strstr(name, "encoder.proj2.weight")) {
            model.proj2_w = tensor;
        } else if (strstr(name, "encoder.proj2.bias")) {
            model.proj2_b = tensor;
        } else if (strstr(name, "audio.encoder.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "audio.encoder.blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < model.hparams.n_encoder_layers) {
                auto & layer = model.layers[layer_idx];
                
                if (strstr(name, "attn_q.weight")) layer.attn_q_w = tensor;
                else if (strstr(name, "attn_q.bias")) layer.attn_q_b = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k_w = tensor;
                else if (strstr(name, "attn_k.bias")) layer.attn_k_b = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v_w = tensor;
                else if (strstr(name, "attn_v.bias")) layer.attn_v_b = tensor;
                else if (strstr(name, "attn_out.weight")) layer.attn_out_w = tensor;
                else if (strstr(name, "attn_out.bias")) layer.attn_out_b = tensor;
                else if (strstr(name, "attn_norm.weight")) layer.attn_norm_w = tensor;
                else if (strstr(name, "attn_norm.bias")) layer.attn_norm_b = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up_w = tensor;
                else if (strstr(name, "ffn_up.bias")) layer.ffn_up_b = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down_w = tensor;
                else if (strstr(name, "ffn_down.bias")) layer.ffn_down_b = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm_w = tensor;
                else if (strstr(name, "ffn_norm.bias")) layer.ffn_norm_b = tensor;
            }
        }
    }
    
    return true;
}

bool GGUFLoader::load_tensor_data(const std::string & path, gguf_context * ctx,
                                   audio_encoder_model & model) {
    if (!map_file_readonly(path, model.mmap, error_msg_)) {
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(ctx);
    if (data_offset > model.mmap.size) {
        error_msg_ = "Invalid GGUF data offset in file: " + path;
        unmap_file(model.mmap);
        return false;
    }

    const size_t total_size = model.mmap.size - data_offset;
    uint8_t * data_base = static_cast<uint8_t *>(model.mmap.addr) + data_offset;
    
    // Find largest tensor for max_tensor_size hint
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    size_t max_tensor_size = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        size_t sz = gguf_get_tensor_size(ctx, i);
        if (sz > max_tensor_size) max_tensor_size = sz;
    }

    // Try GPU device buffer (zero-copy on Apple Silicon unified memory)
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    bool use_vram = gpu_dev && getenv("QWEN_USE_VRAM") != nullptr;
    if (!use_vram && gpu_dev) {
        const char * dev_name = ggml_backend_dev_name(gpu_dev);
        if (dev_name && (strstr(dev_name, "CUDA") != nullptr || strstr(dev_name, "cuda") != nullptr)) {
            use_vram = true;
        }
    }

    if (use_vram) {
        ggml_backend_t backend = ggml_backend_dev_init(gpu_dev, nullptr);
        model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, backend);
        if (model.buffer) {
            ggml_backend_buffer_set_usage(model.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (int64_t i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(ctx, i);
                size_t offset = gguf_get_tensor_offset(ctx, i);
                auto it = model.tensors.find(name);
                if (it == model.tensors.end()) continue;
                ggml_backend_tensor_set(it->second, data_base + offset, 0, ggml_nbytes(it->second));
            }
            ggml_backend_free(backend);
            return true;
        }
        ggml_backend_free(backend);
    }

    if (gpu_dev) {
        model.buffer = ggml_backend_dev_buffer_from_host_ptr(gpu_dev, data_base, total_size, max_tensor_size);
    }
    if (!model.buffer) {
        model.buffer = ggml_backend_cpu_buffer_from_ptr(data_base, total_size);
    }
    if (!model.buffer) {
        error_msg_ = "Failed to create buffer from mmap";
        unmap_file(model.mmap);
        return false;
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);

        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) continue;

        ggml_tensor * tensor = it->second;
        tensor->buffer = model.buffer;
        tensor->data = data_base + offset;
    }

    return true;
}

void free_model(audio_encoder_model & model) {
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

} // namespace qwen3_asr
