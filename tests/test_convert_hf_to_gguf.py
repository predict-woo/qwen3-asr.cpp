"""Offline unit tests for scripts/convert_hf_to_gguf.py.

Verifies conversion of both:
1. Native HuggingFace transformers>=5.21 Qwen3-ASR checkpoints (flat
   audio_config/text_config, model.audio_tower.*, model.multi_modal_projector.*,
   model.language_model.*, tied embeddings without lm_head.weight, tokenizer.json)
   matching the schema of oruk/oruk-fourier-2b and Qwen/Qwen3-ASR-1.7B-hf.
2. Legacy thinker_config / thinker.* checkpoints with vocab.json + merges.txt.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gguf  # noqa: E402
from convert_hf_to_gguf import Qwen3ASRConverter  # noqa: E402


def _make_native_hf_checkpoint(model_dir: Path) -> None:
    """Write a tiny synthetic checkpoint using oruk/oruk-fourier-2b's exact schema."""
    model_dir.mkdir(parents=True, exist_ok=True)
    d_model = 64
    ffn_dim = 128
    conv_ch = 16
    hidden = 64
    inter = 128
    head_dim = 32
    n_heads = 2
    n_kv = 1
    vocab_size = 64

    config = {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "audio_config": {
            "d_model": d_model,
            "downsample_hidden_size": conv_ch,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": ffn_dim,
            "encoder_layers": 1,
            "max_position_embeddings": 13,
            "n_window": 50,
            "n_window_infer": 800,
            "num_mel_bins": 128,
            "output_dim": hidden,
        },
        "audio_token_id": 12,
        "eos_token_id": [10, 11],
        "model_type": "qwen3_asr",
        "text_config": {
            "head_dim": head_dim,
            "hidden_size": hidden,
            "intermediate_size": inter,
            "num_attention_heads": n_heads,
            "num_hidden_layers": 1,
            "num_key_value_heads": n_kv,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
            "tie_word_embeddings": True,
            "vocab_size": vocab_size,
        },
        "tie_word_embeddings": True,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    tokenizer_json = {
        "added_tokens": [
            {"id": 10, "content": "<|endoftext|>", "special": True},
            {"id": 11, "content": "<|im_end|>", "special": True},
            {"id": 12, "content": "<|audio_pad|>", "special": True},
            {"id": 13, "content": "<asr_text>", "special": False},
        ],
        "model": {
            "vocab": {f"tok{i}": i for i in range(10)},
            "merges": [["t", "o"], "to k"],
        },
    }
    (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json), encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"}),
        encoding="utf-8",
    )

    tensors = {
        "model.audio_tower.conv2d1.weight": torch.zeros((conv_ch, 1, 3, 3), dtype=torch.bfloat16),
        "model.audio_tower.conv2d1.bias": torch.zeros((conv_ch,), dtype=torch.bfloat16),
        "model.audio_tower.conv2d2.weight": torch.zeros((conv_ch, conv_ch, 3, 3), dtype=torch.bfloat16),
        "model.audio_tower.conv2d2.bias": torch.zeros((conv_ch,), dtype=torch.bfloat16),
        "model.audio_tower.conv2d3.weight": torch.zeros((conv_ch, conv_ch, 3, 3), dtype=torch.bfloat16),
        "model.audio_tower.conv2d3.bias": torch.zeros((conv_ch,), dtype=torch.bfloat16),
        "model.audio_tower.conv_out.weight": torch.zeros((d_model, conv_ch * 16), dtype=torch.bfloat16),
        "model.audio_tower.ln_post.weight": torch.ones((d_model,), dtype=torch.bfloat16),
        "model.audio_tower.ln_post.bias": torch.zeros((d_model,), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.self_attn_layer_norm.weight": torch.ones((d_model,), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.self_attn_layer_norm.bias": torch.zeros((d_model,), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.self_attn.q_proj.weight": torch.zeros((d_model, d_model), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.self_attn.q_proj.bias": torch.zeros((d_model,), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.self_attn.k_proj.weight": torch.zeros((d_model, d_model), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.self_attn.k_proj.bias": torch.zeros((d_model,), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.self_attn.v_proj.weight": torch.zeros((d_model, d_model), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.self_attn.v_proj.bias": torch.zeros((d_model,), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.self_attn.out_proj.weight": torch.zeros((d_model, d_model), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.self_attn.out_proj.bias": torch.zeros((d_model,), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.final_layer_norm.weight": torch.ones((d_model,), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.final_layer_norm.bias": torch.zeros((d_model,), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.fc1.weight": torch.zeros((ffn_dim, d_model), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.fc1.bias": torch.zeros((ffn_dim,), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.fc2.weight": torch.zeros((d_model, ffn_dim), dtype=torch.bfloat16),
        "model.audio_tower.layers.0.fc2.bias": torch.zeros((d_model,), dtype=torch.bfloat16),
        "model.multi_modal_projector.linear_1.weight": torch.zeros((d_model, d_model), dtype=torch.bfloat16),
        "model.multi_modal_projector.linear_1.bias": torch.zeros((d_model,), dtype=torch.bfloat16),
        "model.multi_modal_projector.linear_2.weight": torch.zeros((hidden, d_model), dtype=torch.bfloat16),
        "model.multi_modal_projector.linear_2.bias": torch.zeros((hidden,), dtype=torch.bfloat16),
        # Tied embeddings: no lm_head.weight tensor
        "model.language_model.embed_tokens.weight": torch.zeros((vocab_size, hidden), dtype=torch.bfloat16),
        "model.language_model.norm.weight": torch.ones((hidden,), dtype=torch.bfloat16),
        "model.language_model.layers.0.input_layernorm.weight": torch.ones((hidden,), dtype=torch.bfloat16),
        "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones((hidden,), dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros((n_heads * head_dim, hidden), dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.zeros((n_kv * head_dim, hidden), dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.zeros((n_kv * head_dim, hidden), dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.zeros((hidden, n_heads * head_dim), dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones((head_dim,), dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones((head_dim,), dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.zeros((inter, hidden), dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.zeros((inter, hidden), dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.zeros((hidden, inter), dtype=torch.bfloat16),
    }
    save_file(tensors, str(model_dir / "model.safetensors"))


class ConvertHFToGGUFTests(unittest.TestCase):
    def test_converts_native_hf_tied_fourier_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "oruk-fourier-2b"
            out_gguf = Path(tmp) / "oruk-fourier-2b-q8_0.gguf"
            _make_native_hf_checkpoint(model_dir)

            converter = Qwen3ASRConverter(model_dir, out_gguf, output_type="q8_0")
            self.assertEqual(converter.model_name, "Oruk-Fourier-2B")
            self.assertEqual(converter.audio_d_model, 64)
            self.assertEqual(converter.text_rope_theta, 1000000)
            self.assertEqual(converter.audio_pad_token_id, 12)

            converter.convert()
            self.assertTrue(out_gguf.exists())

            reader = gguf.GGUFReader(str(out_gguf))
            tensor_names = {t.name for t in reader.tensors}
            self.assertEqual(len(tensor_names), 42)
            for required in (
                "audio.encoder.conv1.weight",
                "audio.encoder.conv_out.weight",
                "audio.encoder.proj1.weight",
                "audio.encoder.proj2.weight",
                "audio.encoder.blk.0.attn_q.weight",
                "token_embd.weight",
                "output_norm.weight",
                "blk.0.attn_q_norm.weight",
                "blk.0.ffn_down.weight",
            ):
                self.assertIn(required, tensor_names)
            self.assertNotIn("output.weight", tensor_names)

            fields = reader.fields
            self.assertEqual(fields["general.architecture"].contents(), "qwen3-asr")
            self.assertEqual(fields["general.name"].contents(), "Oruk-Fourier-2B")
            self.assertEqual(fields["qwen3-asr.audio.pad_token_id"].contents(), 12)
            self.assertEqual(fields["tokenizer.ggml.eos_token_id"].contents(), 11)
            self.assertEqual(fields["tokenizer.ggml.merges"].contents(), ["t o", "to k"])

    def test_legacy_thinker_mapping_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "Qwen3-ASR-1.7B"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "thinker_config": {
                            "audio_config": {"d_model": 1024, "encoder_layers": 24},
                            "text_config": {"hidden_size": 2048, "rope_theta": 1000000},
                            "audio_token_id": 151676,
                        }
                    }
                ),
                encoding="utf-8",
            )
            converter = Qwen3ASRConverter(model_dir, Path(tmp) / "out.gguf", output_type="f16")
            self.assertEqual(converter.model_name, "Qwen3-ASR-1.7B")
            self.assertEqual(
                converter._map_tensor_name("thinker.audio_tower.proj1.weight"),
                "audio.encoder.proj1.weight",
            )
            self.assertEqual(
                converter._map_tensor_name("model.multi_modal_projector.linear_2.bias"),
                "audio.encoder.proj2.bias",
            )
            self.assertEqual(
                converter._map_tensor_name("thinker.model.layers.3.self_attn.o_proj.weight"),
                "blk.3.attn_output.weight",
            )
            self.assertEqual(
                converter._map_tensor_name("model.language_model.layers.3.self_attn.o_proj.weight"),
                "blk.3.attn_output.weight",
            )


if __name__ == "__main__":
    unittest.main()
