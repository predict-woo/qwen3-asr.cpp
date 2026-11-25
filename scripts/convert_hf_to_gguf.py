#!/usr/bin/env python3
"""
Convert HuggingFace Qwen3-ASR and Qwen3-ForcedAligner models to GGUF format.
https://huggingface.co/Qwen/Qwen3-ASR-0.6B/
https://huggingface.co/Qwen/Qwen3-ASR-1.7B/
https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/
https://huggingface.co/oruk/oruk-fourier-2b/

Supports both legacy `thinker_config` (`thinker.*`) checkpoints and native
HuggingFace `transformers>=5.21` (`audio_config`/`text_config`,
`model.audio_tower.*`, `model.multi_modal_projector.*`,
`model.language_model.*`) checkpoints with tied embeddings and `tokenizer.json`.

Usage:
    python scripts/convert_hf_to_gguf.py \
        --input /path/to/Qwen3-ASR-0.6B \
        --output models/qwen3-asr-0.6b-f16.gguf \
        --type f16
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

# Add gguf-py to path
GGUF_PY_PATH = Path("/root/llama.cpp/gguf-py")
if GGUF_PY_PATH.exists():
    sys.path.insert(0, str(GGUF_PY_PATH))

import gguf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32


def get_folder_size(dir_path: Path) -> int:
    """Get the total size of a folder in bytes."""
    path = Path(dir_path)
    total_size = 0
    for file in path.rglob("*"):
        if file.is_file():
            try:
                total_size += file.stat().st_size
            except OSError:
                pass
    return total_size


class Qwen3ASRConverter:
    """Converter for Qwen3-ASR and Qwen3-ForcedAligner models to GGUF format."""

    # Tensor name mapping from HuggingFace to GGML conventions
    TENSOR_MAP = {
        # Audio Tower - Conv2D front-end (legacy thinker.* and native HF model.*)
        "thinker.audio_tower.conv2d1.weight": "audio.encoder.conv1.weight",
        "thinker.audio_tower.conv2d1.bias": "audio.encoder.conv1.bias",
        "thinker.audio_tower.conv2d2.weight": "audio.encoder.conv2.weight",
        "thinker.audio_tower.conv2d2.bias": "audio.encoder.conv2.bias",
        "thinker.audio_tower.conv2d3.weight": "audio.encoder.conv3.weight",
        "thinker.audio_tower.conv2d3.bias": "audio.encoder.conv3.bias",
        "thinker.audio_tower.conv_out.weight": "audio.encoder.conv_out.weight",
        "thinker.audio_tower.conv_out.bias": "audio.encoder.conv_out.bias",
        "model.audio_tower.conv2d1.weight": "audio.encoder.conv1.weight",
        "model.audio_tower.conv2d1.bias": "audio.encoder.conv1.bias",
        "model.audio_tower.conv2d2.weight": "audio.encoder.conv2.weight",
        "model.audio_tower.conv2d2.bias": "audio.encoder.conv2.bias",
        "model.audio_tower.conv2d3.weight": "audio.encoder.conv3.weight",
        "model.audio_tower.conv2d3.bias": "audio.encoder.conv3.bias",
        "model.audio_tower.conv_out.weight": "audio.encoder.conv_out.weight",
        "model.audio_tower.conv_out.bias": "audio.encoder.conv_out.bias",
        # Audio Tower - Layer norms
        "thinker.audio_tower.layer_norm.weight": "audio.encoder.ln.weight",
        "thinker.audio_tower.layer_norm.bias": "audio.encoder.ln.bias",
        "thinker.audio_tower.ln_post.weight": "audio.encoder.ln_post.weight",
        "thinker.audio_tower.ln_post.bias": "audio.encoder.ln_post.bias",
        "model.audio_tower.layer_norm.weight": "audio.encoder.ln.weight",
        "model.audio_tower.layer_norm.bias": "audio.encoder.ln.bias",
        "model.audio_tower.ln_post.weight": "audio.encoder.ln_post.weight",
        "model.audio_tower.ln_post.bias": "audio.encoder.ln_post.bias",
        # Audio Tower - Positional embedding
        "thinker.audio_tower.embed_positions.weight": "audio.encoder.pos_embd.weight",
        "model.audio_tower.embed_positions.weight": "audio.encoder.pos_embd.weight",
        # Audio Tower - Projection layers (audio to text embedding space)
        "thinker.audio_tower.proj1.weight": "audio.encoder.proj1.weight",
        "thinker.audio_tower.proj1.bias": "audio.encoder.proj1.bias",
        "thinker.audio_tower.proj2.weight": "audio.encoder.proj2.weight",
        "thinker.audio_tower.proj2.bias": "audio.encoder.proj2.bias",
        "model.multi_modal_projector.linear_1.weight": "audio.encoder.proj1.weight",
        "model.multi_modal_projector.linear_1.bias": "audio.encoder.proj1.bias",
        "model.multi_modal_projector.linear_2.weight": "audio.encoder.proj2.weight",
        "model.multi_modal_projector.linear_2.bias": "audio.encoder.proj2.bias",
        # Text Decoder
        "thinker.model.embed_tokens.weight": "token_embd.weight",
        "thinker.model.norm.weight": "output_norm.weight",
        "thinker.lm_head.weight": "output.weight",
        "model.language_model.embed_tokens.weight": "token_embd.weight",
        "model.language_model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
        # ForcedAligner specific
        "thinker.classify_head.weight": "classify_head.weight",
        "thinker.classify_head.bias": "classify_head.bias",
    }

    # Regex patterns for layer-specific tensors (legacy thinker.* and native HF model.*)
    AUDIO_LAYER_PATTERNS = [
        # Audio encoder transformer layers - attention weights
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.weight", "audio.encoder.blk.{}.attn_q.weight"),
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.weight", "audio.encoder.blk.{}.attn_k.weight"),
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.weight", "audio.encoder.blk.{}.attn_v.weight"),
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.weight", "audio.encoder.blk.{}.attn_out.weight"),
        # Audio encoder transformer layers - attention biases
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.bias", "audio.encoder.blk.{}.attn_q.bias"),
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.bias", "audio.encoder.blk.{}.attn_k.bias"),
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.bias", "audio.encoder.blk.{}.attn_v.bias"),
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.bias", "audio.encoder.blk.{}.attn_out.bias"),
        # Audio encoder transformer layers - layer norms
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.weight", "audio.encoder.blk.{}.attn_norm.weight"),
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.bias", "audio.encoder.blk.{}.attn_norm.bias"),
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.final_layer_norm\.weight", "audio.encoder.blk.{}.ffn_norm.weight"),
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.final_layer_norm\.bias", "audio.encoder.blk.{}.ffn_norm.bias"),
        # Audio encoder transformer layers - FFN
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.fc1\.weight", "audio.encoder.blk.{}.ffn_up.weight"),
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.fc1\.bias", "audio.encoder.blk.{}.ffn_up.bias"),
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.fc2\.weight", "audio.encoder.blk.{}.ffn_down.weight"),
        (r"(?:thinker|model)\.audio_tower\.layers\.(\d+)\.fc2\.bias", "audio.encoder.blk.{}.ffn_down.bias"),
    ]

    TEXT_LAYER_PATTERNS = [
        # Text decoder transformer layers - attention
        (r"(?:thinker\.model|model\.language_model)\.layers\.(\d+)\.input_layernorm\.weight", "blk.{}.attn_norm.weight"),
        (r"(?:thinker\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.q_proj\.weight", "blk.{}.attn_q.weight"),
        (r"(?:thinker\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.k_proj\.weight", "blk.{}.attn_k.weight"),
        (r"(?:thinker\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.v_proj\.weight", "blk.{}.attn_v.weight"),
        (r"(?:thinker\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.o_proj\.weight", "blk.{}.attn_output.weight"),
        # Text decoder transformer layers - QK norms (Qwen3 uses RMSNorm on Q and K)
        (r"(?:thinker\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.q_norm\.weight", "blk.{}.attn_q_norm.weight"),
        (r"(?:thinker\.model|model\.language_model)\.layers\.(\d+)\.self_attn\.k_norm\.weight", "blk.{}.attn_k_norm.weight"),
        # Text decoder transformer layers - FFN
        (r"(?:thinker\.model|model\.language_model)\.layers\.(\d+)\.post_attention_layernorm\.weight", "blk.{}.ffn_norm.weight"),
        (r"(?:thinker\.model|model\.language_model)\.layers\.(\d+)\.mlp\.gate_proj\.weight", "blk.{}.ffn_gate.weight"),
        (r"(?:thinker\.model|model\.language_model)\.layers\.(\d+)\.mlp\.up_proj\.weight", "blk.{}.ffn_up.weight"),
        (r"(?:thinker\.model|model\.language_model)\.layers\.(\d+)\.mlp\.down_proj\.weight", "blk.{}.ffn_down.weight"),
    ]

    def __init__(
            self,
            input_dir: Path,
            output_path: Path,
            output_type: str = "f16",
            model_name: str | None = None,
    ):
        self.input_dir = input_dir
        self.output_path = output_path
        self.output_type = output_type
        self._explicit_model_name = model_name

        # Load config
        self.config = self._load_config()
        self.is_forced_aligner = self._is_forced_aligner()

        # Determine model parameters
        self._extract_params()

    def _load_config(self) -> dict[str, Any]:
        """Load model configuration from config.json."""
        config_path = self.input_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _is_forced_aligner(self) -> bool:
        """Check if this is a ForcedAligner model."""
        thinker_config = self.config.get("thinker_config", {})
        return (
            thinker_config.get("model_type") == "qwen3_forced_aligner"
            or self.config.get("model_type") == "qwen3_forced_aligner"
        )

    def _extract_params(self) -> None:
        """Extract model parameters from legacy or native HF config."""
        # Legacy Qwen/Qwen3-ASR checkpoints nest audio_config/text_config under
        # thinker_config; native transformers>=5.21 checkpoints place them at top level.
        root_cfg = self.config.get("thinker_config") or self.config
        audio_config = root_cfg.get("audio_config", {})
        text_config = root_cfg.get("text_config", {})

        # Audio encoder parameters
        self.audio_encoder_layers = audio_config.get("encoder_layers", audio_config.get("num_hidden_layers", 18))
        self.audio_d_model = audio_config.get("d_model", 896)
        self.audio_attention_heads = audio_config.get("encoder_attention_heads", 14)
        self.audio_ffn_dim = audio_config.get("encoder_ffn_dim", 3584)
        self.audio_num_mel_bins = audio_config.get("num_mel_bins", 128)
        self.audio_downsample_hidden_size = audio_config.get("downsample_hidden_size", 480)

        # Text decoder parameters
        self.text_decoder_layers = text_config.get("num_hidden_layers", 28)
        self.text_hidden_size = text_config.get("hidden_size", 1024)
        self.text_attention_heads = text_config.get("num_attention_heads", 16)
        self.text_kv_heads = text_config.get("num_key_value_heads", 8)
        self.text_intermediate_size = text_config.get("intermediate_size", 3072)
        rope_parameters = text_config.get("rope_parameters") or {}
        self.text_rope_theta = text_config.get(
            "rope_theta",
            rope_parameters.get("rope_theta", 1000000),
        )
        self.text_rms_norm_eps = text_config.get("rms_norm_eps", 1e-6)
        self.text_head_dim = text_config.get("head_dim", 128)
        self.vocab_size = text_config.get("vocab_size", 151936)

        # Special tokens
        self.audio_start_token_id = root_cfg.get("audio_start_token_id", 151669)
        self.audio_end_token_id = root_cfg.get("audio_end_token_id", 151670)
        self.audio_pad_token_id = root_cfg.get("audio_token_id", 151676)

        # ForcedAligner specific
        if self.is_forced_aligner:
            self.classify_num = root_cfg.get("classify_num", 5000)
            self.timestamp_token_id = self.config.get("timestamp_token_id", 151705)
        else:
            self.classify_num = None
            self.timestamp_token_id = None

        # Model name
        if self._explicit_model_name:
            self.model_name = self._explicit_model_name
        elif self.is_forced_aligner:
            self.model_name = "Qwen3-ForcedAligner-0.6B"
        elif "fourier" in self.input_dir.name.lower():
            self.model_name = "Oruk-Fourier-2B"
        elif self.audio_d_model >= 1024 or self.text_hidden_size >= 2048 or get_folder_size(self.input_dir) > 1024 * 1024 * 1024 * 3:
            self.model_name = "Qwen3-ASR-1.7B"
        else:
            self.model_name = "Qwen3-ASR-0.6B"

    def _load_tokenizer_config(self) -> dict[str, Any]:
        """Load tokenizer_config.json if present."""
        tokenizer_config_path = self.input_dir / "tokenizer_config.json"
        if not tokenizer_config_path.exists():
            return {}

        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_tokenizer_json(self) -> dict[str, Any]:
        """Load tokenizer.json if present."""
        tokenizer_json_path = self.input_dir / "tokenizer.json"
        if not tokenizer_json_path.exists():
            return {}

        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_vocab_dict(self) -> dict[str, int]:
        """Load base tokenizer vocabulary from vocab.json or tokenizer.json."""
        vocab_path = self.input_dir / "vocab.json"
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                return json.load(f)

        tokenizer_json = self._load_tokenizer_json()
        model_vocab = (tokenizer_json.get("model") or {}).get("vocab")
        if isinstance(model_vocab, dict) and model_vocab:
            return dict(model_vocab)

        raise FileNotFoundError(
            f"Neither vocab.json nor tokenizer.json with model.vocab found in {self.input_dir}"
        )

    def _map_tensor_name(self, hf_name: str) -> str | None:
        """Map HuggingFace tensor name to GGML convention."""
        # Check direct mapping first
        if hf_name in self.TENSOR_MAP:
            return self.TENSOR_MAP[hf_name]

        # Check audio layer patterns
        for pattern, template in self.AUDIO_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                layer_idx = match.group(1)
                return template.format(layer_idx)

        # Check text layer patterns
        for pattern, template in self.TEXT_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                layer_idx = match.group(1)
                return template.format(layer_idx)

        return None

    def _get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate over all tensors from safetensors files."""
        safetensor_files = list(self.input_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {self.input_dir}")

        for sf_path in sorted(safetensor_files):
            logger.info(f"Loading tensors from {sf_path.name}")
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)

    def _should_quantize(self, tensor_name: str) -> bool:
        """Determine if a tensor should be quantized (Q8_0) or kept in F16.

        Tensors to keep in F16 for quality:
        - Embeddings (token_embd, output, pos_embd)
        - Layer norms (attn_norm, ffn_norm, output_norm, ln)
        - Biases
        """
        # Keep embeddings in F16
        if any(x in tensor_name for x in ["token_embd", "output.weight", "pos_embd"]):
            return False

        # Keep layer norms in F16
        if any(x in tensor_name for x in ["_norm", ".ln", "ln_post"]):
            return False

        # Keep biases in F16 (they're usually 1D anyway, handled separately)
        if ".bias" in tensor_name:
            return False

        # Quantize weight matrices
        return True

    def _convert_dtype(self, tensor: torch.Tensor, tensor_name: str = "") -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
        """Convert tensor to appropriate dtype for GGUF."""
        # Convert to numpy
        if tensor.dtype == torch.bfloat16:
            # Convert bfloat16 to float32 first, then to float16
            data = tensor.float().numpy()
        else:
            data = tensor.numpy()

        # Determine output type
        n_dims = len(data.shape)

        # Handle conv2d weight permutation
        # PyTorch conv2d weights: [out_ch, in_ch, kh, kw]
        # GGML ggml_conv_2d expects kernel shape [OC, IC, KH, KW] in GGML notation
        # which means ne[0]=KW, ne[1]=KH, ne[2]=IC, ne[3]=OC
        # The memory layout should have KW varying fastest, then KH, then IC, then OC
        # PyTorch's C-order layout already has KW varying fastest (last dim varies fastest)
        # So we just need to ensure the data is contiguous, no transpose needed
        if n_dims == 4 and "conv" in tensor_name and "weight" in tensor_name:
            data = np.ascontiguousarray(data)
            logger.debug(f"Conv2d weight {tensor_name}: shape {data.shape}")

        # NOTE: Do NOT transpose embedding tensors!
        # HuggingFace stores embeddings as [vocab_size, hidden_size] in row-major
        # gguf-py reverses the shape when writing, so GGUF stores [hidden_size, vocab_size]
        # GGML reads in column-major, which effectively transposes the data
        # Net effect: GGML sees [hidden_size, vocab_size] with each column being a token's embedding
        # This is exactly what ggml_get_rows expects (ne[0] = hidden_size)

        # 1D tensors (norms, biases) should be F32
        if n_dims <= 1:
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32

        # For 2D+ tensors, use the specified output type
        if self.output_type == "f32":
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32
        elif self.output_type == "f16":
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16
        elif self.output_type == "q8_0":
            # Q8_0 quantization: 8-bit with block size 32
            # Keep some tensors in F16 for quality (embeddings, norms)
            if not self._should_quantize(tensor_name):
                logger.debug(f"Keeping {tensor_name} in F16 (not quantizing)")
                return data.astype(np.float16), gguf.GGMLQuantizationType.F16

            # Quantize to Q8_0
            # Data must be float32 for quantization
            data = data.astype(np.float32)
            try:
                quantized = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
                return quantized, gguf.GGMLQuantizationType.Q8_0
            except Exception as e:
                logger.warning(f"Q8_0 quantization failed for {tensor_name}: {e}, falling back to F16")
                return data.astype(np.float16), gguf.GGMLQuantizationType.F16
        elif self.output_type == "q4_1":
            if not self._should_quantize(tensor_name):
                logger.debug(f"Keeping {tensor_name} in F16 (not quantizing)")
                return data.astype(np.float16), gguf.GGMLQuantizationType.F16
            # Quantize to Q4_1
            # Data must be float32 for quantization
            data = data.astype(np.float32)
            try:
                quantized = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q4_1)
                return quantized, gguf.GGMLQuantizationType.Q4_1
            except Exception as e:
                logger.warning(f"Q4_1 quantization failed for {tensor_name}: {e}, falling back to F16")
                return data.astype(np.float16), gguf.GGMLQuantizationType.F16
        else:
            # Default to F16
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16

    def _load_tokenizer(self) -> tuple[list[str], list[int], list[str], dict[str, int]]:
        """Load tokenizer vocabulary and merges from vocab.json/merges.txt or tokenizer.json."""
        vocab_dict = self._load_vocab_dict()
        tokenizer_config = self._load_tokenizer_config()
        tokenizer_json = self._load_tokenizer_json()
        merges_path = self.input_dir / "merges.txt"

        added_tokens: dict[int, dict[str, Any]] = {}
        for tok in tokenizer_json.get("added_tokens", []):
            tok_id = tok.get("id")
            if tok_id is not None:
                added_tokens[int(tok_id)] = tok
        for tok_id_str, tok_info in tokenizer_config.get("added_tokens_decoder", {}).items():
            added_tokens[int(tok_id_str)] = tok_info

        max_vocab_id = max(vocab_dict.values()) if vocab_dict else -1
        max_added_id = max(added_tokens.keys(), default=-1)
        token_count = max(self.vocab_size, max_vocab_id + 1, max_added_id + 1)

        tokens = [f"[PAD{i}]" for i in range(token_count)]
        toktypes = [gguf.TokenType.UNUSED] * token_count

        for token, token_id in vocab_dict.items():
            tokens[token_id] = token
            toktypes[token_id] = gguf.TokenType.NORMAL

        for token_id, token_info in added_tokens.items():
            token = token_info.get("content")
            if token is None:
                continue

            tokens[token_id] = token
            toktypes[token_id] = (
                gguf.TokenType.CONTROL
                if token_info.get("special", False)
                else gguf.TokenType.NORMAL
            )
            vocab_dict[token] = token_id

        # Load merges from merges.txt or tokenizer.json
        merges: list[str] = []
        if merges_path.exists():
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        merges.append(line)
        else:
            raw_merges = (tokenizer_json.get("model") or {}).get("merges") or []
            for entry in raw_merges:
                if isinstance(entry, str):
                    merges.append(entry)
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    merges.append(f"{entry[0]} {entry[1]}")

        source_desc = "vocab.json" if (self.input_dir / "vocab.json").exists() else "tokenizer.json"
        logger.info(
            f"Using tokenizer vocab built from {source_desc} + tokenizer_config.json "
            f"({len(vocab_dict)} entries)"
        )
        return tokens, toktypes, merges, vocab_dict

    def convert(self) -> None:
        """Convert the model to GGUF format."""
        logger.info(f"Converting {self.model_name} to GGUF format")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Output type: {self.output_type}")

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize GGUF writer
        arch = "qwen3-asr"
        writer = gguf.GGUFWriter(path=None, arch=arch)

        # Add metadata
        self._add_metadata(writer)

        # Add tokenizer
        self._add_tokenizer(writer)

        # Process tensors
        tensor_count = 0
        skipped_count = 0

        logger.info("Processing tensors...")
        for hf_name, tensor in tqdm(list(self._get_tensors()), desc="Converting"):
            ggml_name = self._map_tensor_name(hf_name)

            if ggml_name is None:
                logger.warning(f"Skipping unmapped tensor: {hf_name}")
                skipped_count += 1
                continue

            # Convert tensor
            data, dtype = self._convert_dtype(tensor, ggml_name)

            # Add tensor to writer
            writer.add_tensor(ggml_name, data, raw_dtype=dtype)
            tensor_count += 1

            logger.debug(f"  {hf_name} -> {ggml_name} [{dtype.name}] {data.shape}")

        logger.info(f"Converted {tensor_count} tensors, skipped {skipped_count}")

        # Write to file
        logger.info(f"Writing GGUF file to {self.output_path}")
        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

        logger.info("Conversion complete!")

    def _add_metadata(self, writer: gguf.GGUFWriter) -> None:
        """Add model metadata to GGUF writer."""
        # General metadata
        writer.add_name(self.model_name)
        writer.add_type(gguf.GGUFType.MODEL)

        # File type
        match self.output_type:
            case "f32":
                ftype = gguf.LlamaFileType.ALL_F32
            case "f16":
                ftype = gguf.LlamaFileType.MOSTLY_F16
            case "q8_0":
                ftype = gguf.LlamaFileType.MOSTLY_Q8_0
            case "q4_1":
                ftype = gguf.LlamaFileType.MOSTLY_Q4_1
            case _:
                ftype = gguf.LlamaFileType.MOSTLY_F16
        writer.add_file_type(ftype)

        # Quantization version
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

        # Text decoder parameters (main architecture)
        writer.add_block_count(self.text_decoder_layers)
        writer.add_embedding_length(self.text_hidden_size)
        writer.add_feed_forward_length(self.text_intermediate_size)
        writer.add_head_count(self.text_attention_heads)
        writer.add_head_count_kv(self.text_kv_heads)
        writer.add_key_length(self.text_head_dim)
        writer.add_value_length(self.text_head_dim)
        writer.add_rope_freq_base(self.text_rope_theta)
        writer.add_layer_norm_rms_eps(self.text_rms_norm_eps)
        writer.add_vocab_size(self.vocab_size)

        # Audio encoder parameters (custom keys)
        arch = "qwen3-asr"
        writer.add_uint32(f"{arch}.audio.encoder.layer_count", self.audio_encoder_layers)
        writer.add_uint32(f"{arch}.audio.encoder.embedding_length", self.audio_d_model)
        writer.add_uint32(f"{arch}.audio.encoder.attention.head_count", self.audio_attention_heads)
        writer.add_uint32(f"{arch}.audio.encoder.feed_forward_length", self.audio_ffn_dim)
        writer.add_uint32(f"{arch}.audio.num_mel_bins", self.audio_num_mel_bins)
        writer.add_uint32(f"{arch}.audio.conv_channels", self.audio_downsample_hidden_size)

        # Special token IDs
        writer.add_uint32(f"{arch}.audio.start_token_id", self.audio_start_token_id)
        writer.add_uint32(f"{arch}.audio.end_token_id", self.audio_end_token_id)
        writer.add_uint32(f"{arch}.audio.pad_token_id", self.audio_pad_token_id)

        # ForcedAligner specific
        if self.is_forced_aligner:
            writer.add_uint32(f"{arch}.classify_num", self.classify_num)
            writer.add_uint32(f"{arch}.timestamp_token_id", self.timestamp_token_id)
            writer.add_uint32(f"{arch}.timestamp_segment_time", 80)

        logger.info("Added model metadata")

    def _add_tokenizer(self, writer: gguf.GGUFWriter) -> None:
        """Add tokenizer to GGUF writer."""
        tokens, toktypes, merges, vocab = self._load_tokenizer()
        tokenizer_config = self._load_tokenizer_config()

        # Tokenizer model type
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("qwen2")

        # Token list
        writer.add_token_list(tokens)
        writer.add_token_types(toktypes)

        # Merges
        if merges:
            writer.add_token_merges(merges)

        # Special tokens from tokenizer_config.json
        eos_token = tokenizer_config.get("eos_token")
        if isinstance(eos_token, dict):
            eos_token = eos_token.get("content")
        if eos_token and eos_token in vocab:
            writer.add_eos_token_id(vocab[eos_token])

        pad_token = tokenizer_config.get("pad_token")
        if isinstance(pad_token, dict):
            pad_token = pad_token.get("content")
        if pad_token and pad_token in vocab:
            writer.add_pad_token_id(vocab[pad_token])

        chat_template = tokenizer_config.get("chat_template")
        if chat_template:
            writer.add_chat_template(chat_template)

        logger.info(
            "Added tokenizer with "
            f"{len(tokens)} tokens and {len(merges)} merges"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-ASR/ForcedAligner models to GGUF format"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to HuggingFace model directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["f16", "f32", "q8_0", "q4_1"],
        default="f16",
        help="Output data type (default: f16). q8_0 provides ~50%% size reduction with minimal quality loss."
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional GGUF general.name override (default: inferred from config/directory)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    converter = Qwen3ASRConverter(
        input_dir=args.input,
        output_path=args.output,
        output_type=args.type,
        model_name=args.name,
    )
    converter.convert()


if __name__ == "__main__":
    main()
