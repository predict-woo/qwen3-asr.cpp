# Qwen3-ASR CLI Usage Guide

Complete documentation for the `qwen3-asr-cli` command-line interface and the `qwen3-asr-server` HTTP API server.

## Synopsis

```
qwen3-asr-cli [options]
```

## Options

### Required Options

| Option | Description |
|--------|-------------|
| `-f, --audio <path>` | Path to input audio file (WAV, 16kHz mono) |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model <path>` | `models/qwen3-asr-0.6b-f16.gguf` | Path to GGUF model file |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output <path>` | stdout | Output file path |

### Transcription Options

| Option | Default | Description |
|--------|---------|-------------|
| `-l, --language <code>` | auto-detect | Language code (e.g., `en`, `zh`, `ja`) |
| `-t, --threads <n>` | 4 | Number of CPU threads |
| `--max-tokens <n>` | 1024 | Maximum tokens to generate |
| `--progress` | off | Print progress during transcription |
| `--no-timing` | off | Suppress timing information |
| `--tokens` | off | Print token IDs |

### Forced Alignment Options

| Option | Description |
|--------|-------------|
| `--align` | Enable forced alignment mode |
| `--text <text>` | Reference transcript for alignment (required with `--align`) |

### Help

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |

## Transcription Mode

### Basic Usage

```bash
# Transcribe audio file
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav
```

### Output

Transcription outputs plain text to stdout (or file with `-o`):

```
Hello world, this is a test transcription from the audio file.
```

### Examples

```bash
# Use quantized model for faster inference
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-q8_0.gguf \
    -f audio.wav

# Specify language for better accuracy
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f chinese_audio.wav \
    -l zh

# Multi-threaded processing
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f audio.wav \
    -t 8

# Save to file with progress
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f audio.wav \
    -o transcript.txt \
    --progress

# Long audio with more tokens
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f long_audio.wav \
    --max-tokens 4096

# Debug mode with token IDs
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f audio.wav \
    --tokens
```

## Forced Alignment Mode

Forced alignment synchronizes a reference transcript with audio, producing word-level timestamps.

### Basic Usage

```bash
./build/qwen3-asr-cli \
    -m models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --align \
    --text "Hello world, this is a test."
```

### Output Format

Alignment outputs JSON with word-level timestamps:

```json
{
  "words": [
    {"word": "Hello", "start": 0.000, "end": 0.320},
    {"word": "world", "start": 0.340, "end": 0.640},
    {"word": ",", "start": 0.640, "end": 0.680},
    {"word": "this", "start": 0.720, "end": 0.880},
    {"word": "is", "start": 0.900, "end": 0.980},
    {"word": "a", "start": 1.000, "end": 1.040},
    {"word": "test", "start": 1.060, "end": 1.320},
    {"word": ".", "start": 1.320, "end": 1.360}
  ]
}
```

### Examples

```bash
# Basic alignment
./build/qwen3-asr-cli \
    -m models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --align \
    --text "The quick brown fox jumps over the lazy dog."

# Save alignment to file
./build/qwen3-asr-cli \
    -m models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --align \
    --text "Hello world" \
    -o alignment.json

# Alignment with timing info
./build/qwen3-asr-cli \
    -m models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --align \
    --text "Test sentence"
    # Timing info printed to stderr by default

# Suppress timing info
./build/qwen3-asr-cli \
    -m models/qwen3-forced-aligner-0.6b-f16.gguf \
    -f audio.wav \
    --align \
    --text "Test sentence" \
    --no-timing
```

## Audio Requirements

### Supported Format

- **Format**: WAV (PCM)
- **Sample Rate**: 16,000 Hz
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit signed integer

### Converting Audio

Use ffmpeg to convert audio to the required format:

```bash
# Convert MP3 to WAV
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav

# Convert M4A to WAV
ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le output.wav

# Convert stereo to mono
ffmpeg -i stereo.wav -ar 16000 -ac 1 -c:a pcm_s16le mono.wav

# Extract audio from video
ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 -c:a pcm_s16le audio.wav
```

## OpenAI-Compatible Server

`qwen3-asr-server` serves a minimal OpenAI-compatible transcription API at `/v1/audio/transcriptions`.

### Server Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model <path>` | `models/qwen3-asr-0.6b-q8_0.gguf` | Path to ASR GGUF model |
| `--host <host>` | `127.0.0.1` | Hostname or IP address to bind |
| `--port <port>` | `8080` | Port to bind |
| `-t, --threads <n>` | 4 | Number of inference threads |
| `--max-tokens <n>` | 1024 | Maximum tokens to generate |
| `--max-upload-mb <n>` | 512 | Maximum upload size in MiB |
| `--tmp-dir <path>` | `/tmp` | Directory for uploaded audio files |
| `--convert` | off | Convert uploaded audio to 16 kHz mono WAV with ffmpeg |

### Start Server

```bash
./build/qwen3-asr-server \
    -m models/qwen3-asr-0.6b-q8_0.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    -t 8
```

By default uploaded audio must already be 16 kHz mono PCM WAV. To accept formats such as MP3 or M4A, start the server with `--convert` and make sure `ffmpeg` is installed:

```bash
./build/qwen3-asr-server \
    -m models/qwen3-asr-0.6b-q8_0.gguf \
    --convert
```

### Transcription Request

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
    -F file=@audio.wav \
    -F model=qwen3-asr \
    -F response_format=json
```

Supported multipart fields:

| Field | Description |
|-------|-------------|
| `file` | Audio file upload (required) |
| `model` | Accepted for OpenAI client compatibility; the server uses the model loaded at startup |
| `language` | Optional language hint |
| `response_format` | `json`, `text`, or `verbose_json` |

`response_format=json` returns:

```json
{"text":"transcribed text"}
```

`response_format=verbose_json` returns a minimal verbose payload with `task`, `language`, `duration`, `text`, and an empty `segments` array.

### Health Check

```bash
curl http://127.0.0.1:8080/health
```

Expected response:

```json
{"status":"ok"}
```

## Performance Tips

### Thread Count

Optimal thread count depends on your CPU:

```bash
# Use all available cores
./build/qwen3-asr-cli -m model.gguf -f audio.wav -t $(nproc)

# For hyperthreaded CPUs, use physical core count
./build/qwen3-asr-cli -m model.gguf -f audio.wav -t $(nproc --all)
```

### Quantized Models

Q8_0 quantized models offer:
- ~40% smaller file size
- Faster inference on CPU
- Minimal quality loss

```bash
# Use quantized model
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-q8_0.gguf -f audio.wav
```

### Memory Usage

| Model | Memory (approx) |
|-------|-----------------|
| F16 | ~2.5 GB |
| Q8_0 | ~1.8 GB |

### Batch Processing

For multiple files, use a shell loop:

```bash
for f in *.wav; do
    ./build/qwen3-asr-cli \
        -m models/qwen3-asr-0.6b-q8_0.gguf \
        -f "$f" \
        -o "${f%.wav}.txt" \
        --no-timing
done
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Error (model loading, audio loading, transcription failure) |

## Timing Output

When timing is enabled (default), the following metrics are printed to stderr:

**Transcription mode:**
```
Timing:
  Mel spectrogram: 45 ms
  Audio encoding:  120 ms
  Text decoding:   850 ms
  Total:           1015 ms
```

**Alignment mode:**
```
Timing:
  Mel spectrogram: 45 ms
  Audio encoding:  120 ms
  Text decoding:   350 ms
  Total:           515 ms
  Words aligned:   8
```

## Troubleshooting

### Common Errors

**"Error: Audio file path is required"**
```bash
# Solution: Provide audio file with -f
./build/qwen3-asr-cli -m model.gguf -f audio.wav
```

**"Error: Reference text is required for alignment mode"**
```bash
# Solution: Provide text with --text when using --align
./build/qwen3-asr-cli -m model.gguf -f audio.wav --align --text "Your text"
```

**"Error: Failed to load model"**
```bash
# Check model path exists
ls -la models/qwen3-asr-0.6b-f16.gguf

# Ensure model is valid GGUF format
file models/qwen3-asr-0.6b-f16.gguf
```

**"Error: Could not load audio file"**
```bash
# Check audio format
ffprobe audio.wav

# Convert to correct format
ffmpeg -i audio.wav -ar 16000 -ac 1 -c:a pcm_s16le audio_fixed.wav
```

### Debug Mode

Enable token output for debugging:

```bash
./build/qwen3-asr-cli \
    -m models/qwen3-asr-0.6b-f16.gguf \
    -f audio.wav \
    --tokens
```

This prints token IDs which can help diagnose decoding issues.
