# AGENTS.md

## Project Overview

qwen3-asr.cpp is a C++ GGML implementation of Qwen3-ASR and Qwen3-ForcedAligner.
It performs speech recognition and forced alignment with word-level timestamps.

## Architecture

### Core Components

- `cli/main.cpp` — CLI entry point, mode dispatch (transcription, alignment, combined)
- `src/server.cpp` — OpenAI-compatible HTTP API server (`/v1/audio/transcriptions`)
- `src/qwen3_asr.cpp` `include/qwen3_asr.h` — High-level ASR orchestration (mel → encoder → decoder)
- `src/forced_aligner.cpp` `include/forced_aligner.h` — Forced aligner (separate encoder + decoder, chunked convolution, BPE tokenizer, Korean word splitting)
- `src/text_decoder.cpp` `src/text_decoder.h` — Qwen2-based text decoder with KV cache, flash attention, RoPE
- `src/audio_encoder.cpp` `src/audio_encoder.h` — Audio feature encoder with Metal GPU backend
- `src/mel_spectrogram.cpp` `src/mel_spectrogram.h` — Mel spectrogram computation (vDSP/Accelerate optimized)
- `src/audio_injection.cpp` `include/audio_injection.h` — Audio embedding injection into token sequence
- `src/gguf_loader.cpp` `src/gguf_loader.h` — GGUF model file loading with mmap
- `src/mman_multiplatform.h` — Cross-platform mmap wrapper
- `src/stat_multiplatform.h` — Cross-platform large file support wrapper
- `src/timing.h` — Simple timing macros

### Model Architecture

**ASR encoder**: Whisper-style audio encoder (conv frontend + transformer)

**ASR decoder**: Qwen2 (28 layers, GQA 16/4 heads, head_dim=64, hidden=1024, RoPE theta=1M)

**Forced aligner encoder**: Larger conv frontend (3x Conv2d stride=2 → 480ch → linear 1024) + 24-layer transformer with windowed attention

**Forced aligner decoder**: Same Qwen2 architecture but with classify_head (5000 classes × 80ms = 400s range) instead of lm_head, uses CAUSAL attention

### Key Design Decisions

- **GGML tensor library** (not PyTorch/ONNX) for minimal dependencies
- **Dual CPU+Metal GPU backends** with ggml_backend_sched for optimal placement
- **mmap weight loading** with zero-copy GPU transfer via `ggml_backend_dev_buffer_from_host_ptr`
- **F16 KV cache** to reduce memory bandwidth
- **Flash attention** (`ggml_flash_attn_ext`) for decode speedup
- **Weight tying** (token_embd = output weight) to save memory
- **Korean word splitting** ported from soynlp LTokenizer with bundled dictionary

### Build System

- CMake 3.14+, C++17
- GGML included as git submodule at `./ggml`
- `third_party/httplib.h` provides the single-header HTTP server dependency
- Accelerate framework linked on Apple for vDSP mel spectrogram
- Build: `cmake --build build -j$(sysctl -n hw.ncpu)`

### Testing

- **Test audio**: any 16kHz mono WAV
- **ASR**: `./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav`
- **Align**: `./build/qwen3-asr-cli -m models/qwen3-forced-aligner-0.6b-f16.gguf -f audio.wav --align --text "text" --lang korean`
- **Combined**: `./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf -f audio.wav --transcribe-align`
- **Server**: `./build/qwen3-asr-server -m models/qwen3-asr-0.6b-q8_0.gguf --host 127.0.0.1 --port 8080`

### Conventions

- **Namespace**: `qwen3_asr::`
- **Error handling**: bool return + error_msg_ member
- **Timing**: `QWEN3_TIMER_SCOPED("name")` macros from `src/timing.h`
- **Memory**: RAII with explicit cleanup in destructors; mmap cleanup via munmap
- **Tensor naming**: follows HuggingFace naming convention for weight mapping
- **Server concurrency**: serialize requests around a loaded `Qwen3ASR` instance because the decoder KV cache is mutable

### Important Caveats

- The forced aligner decoder MUST use causal attention (model was trained with `self_attn.is_causal: True`)
- The forced aligner encoder uses windowed attention (block-diagonal mask, window_aftercnn=104)
- Korean word splitting requires `assets/korean_dict_jieba.dict` — auto-discovered relative to model/executable
- The ASR output text starts with "language <Name>" prefix (e.g. "language Korean...") which must be stripped before alignment
- Special token IDs: audio_start=151669, audio_end=151670, audio_pad=151676, timestamp=151705

## Performance Notes

Benchmark on 92-second Korean audio, Apple M2 Pro:
- Mel spectrogram: 98 ms
- Audio encoding: 715 ms
- Text decoding: 4,194 ms (323 tokens)
- Total ASR: 5,007 ms
- Forced alignment: 12,998 ms (183 words)
- Combined: 18,005 ms

Memory: ~247 MB RSS, ~294 MB Metal

## Development Guidelines

### When Adding Features

1. **Maintain compatibility** with GGML submodule API
2. **Profile before optimizing** using `QWEN3_ASR_TIMING` build flag
3. **Follow existing patterns** for error handling and memory management
4. **Test on real audio** (not just synthetic data)
5. **Update benchmarks** if modifying hot paths

### Common Pitfalls

- Don't break mmap compatibility (weight tensors must use mmap buffer)
- Don't change tensor layout without updating GGUF conversion script
- Flash attention requires contiguous memory layout
- Korean dictionary must be UTF-8 encoded
- Audio must be 16kHz mono PCM (conversion required otherwise)
- Server `--convert` shells out to ffmpeg for uploaded non-WAV audio; without it, uploads must already be 16kHz mono PCM WAV

### File Organization

- Headers (`.h`) contain public API and documentation
- Implementation (`.cpp`) contains internal logic
- Test files (`test_*.cpp`) demonstrate usage and verify correctness
- Timing macros only active when `QWEN3_ASR_TIMING` defined

## Contributing

When making changes:
1. Run existing tests to verify no regression
2. Add new tests for new functionality
3. Update AGENTS.md if architecture changes
4. Document any new conventions or caveats
5. Benchmark performance impact on Apple Silicon

## Resources

- [GGML Documentation](https://github.com/ggerganov/ggml)
- [Qwen3-ASR HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ForcedAligner HuggingFace](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- Korean dictionary: 17,968 words from soynlp corpus
