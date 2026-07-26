#include "mel_spectrogram.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// FFT Implementation (ported from whisper.cpp)
// ============================================================================

// Precomputed sin/cos table for FFT - must match FFT size exactly
constexpr int SIN_COS_N_COUNT = QWEN_N_FFT;

struct GlobalCache {
    float sin_vals[SIN_COS_N_COUNT];
    float cos_vals[SIN_COS_N_COUNT];
    double hann_window[QWEN_N_FFT];
    float hann_window_f[QWEN_N_FFT];

    GlobalCache() {
        fill_sin_cos_table();
        fill_hann_window(QWEN_N_FFT, true, hann_window);
        for (int i = 0; i < QWEN_N_FFT; i++) {
            hann_window_f[i] = static_cast<float>(hann_window[i]);
        }
    }

    void fill_sin_cos_table() {
        for (int i = 0; i < SIN_COS_N_COUNT; i++) {
            double theta = (2.0 * M_PI * i) / SIN_COS_N_COUNT;
            sin_vals[i] = sinf(static_cast<float>(theta));
            cos_vals[i] = cosf(static_cast<float>(theta));
        }
    }

    void fill_hann_window(int length, bool periodic, double* output) {
        int offset = periodic ? 0 : -1;
        for (int i = 0; i < length; i++) {
            output[i] = 0.5 * (1.0 - cos((2.0 * M_PI * i) / (length + offset)));
        }
    }
};

static GlobalCache global_cache;

// Naive DFT for non-power-of-2 sizes
static void dft(const float* in, int N, float* out) {
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % SIN_COS_N_COUNT;
            re += in[n] * global_cache.cos_vals[idx];
            im -= in[n] * global_cache.sin_vals[idx];
        }

        out[k * 2 + 0] = re;
        out[k * 2 + 1] = im;
    }
}

// Cooley-Tukey FFT (recursive, in-place)
static void fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    const int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        // N is odd, fall back to DFT
        dft(in, N, out);
        return;
    }

    // Split into even and odd
    float* even = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i] = in[2 * i];
    }
    float* even_fft = out + 2 * N;
    fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2 * i + 1];
    }
    float* odd_fft = even_fft + N;
    fft(odd, half_N, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < half_N; k++) {
        int idx = k * sin_cos_step;
        float re = global_cache.cos_vals[idx];
        float im = -global_cache.sin_vals[idx];

        float re_odd = odd_fft[2 * k + 0];
        float im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + half_N) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

// ============================================================================
// WAV Loading (using dr_wav-style parsing, minimal implementation)
// ============================================================================

bool load_wav(const std::string& path, std::vector<float>& samples, int& sample_rate) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open WAV file: %s\n", path.c_str());
        return false;
    }

    // Read RIFF header
    char riff[4];
    file.read(riff, 4);
    if (strncmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "Error: Not a valid WAV file (missing RIFF header)\n");
        return false;
    }

    uint32_t file_size;
    file.read(reinterpret_cast<char*>(&file_size), 4);

    char wave[4];
    file.read(wave, 4);
    if (strncmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "Error: Not a valid WAV file (missing WAVE header)\n");
        return false;
    }

    // Find fmt chunk
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sr = 0;
    uint16_t bits_per_sample = 0;

    while (file.good()) {
        char chunk_id[4];
        uint32_t chunk_size;
        file.read(chunk_id, 4);
        file.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&sr), 4);
            uint32_t byte_rate;
            file.read(reinterpret_cast<char*>(&byte_rate), 4);
            uint16_t block_align;
            file.read(reinterpret_cast<char*>(&block_align), 2);
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            
            // Skip any extra format bytes
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
        } else if (strncmp(chunk_id, "data", 4) == 0) {
            // Found data chunk
            if (audio_format != 1) {
                fprintf(stderr, "Error: Only PCM format supported (got format %d)\n", audio_format);
                return false;
            }
            if (bits_per_sample != 16) {
                fprintf(stderr, "Error: Only 16-bit samples supported (got %d bits)\n", bits_per_sample);
                return false;
            }

            sample_rate = static_cast<int>(sr);
            int num_samples = chunk_size / (bits_per_sample / 8) / num_channels;
            samples.resize(num_samples);

            std::vector<int16_t> raw_samples(num_samples * num_channels);
            file.read(reinterpret_cast<char*>(raw_samples.data()), chunk_size);

            // Convert to float and handle stereo->mono if needed
            for (int i = 0; i < num_samples; i++) {
                if (num_channels == 1) {
                    samples[i] = raw_samples[i] / 32768.0f;
                } else {
                    // Average channels for stereo
                    float sum = 0;
                    for (int c = 0; c < num_channels; c++) {
                        sum += raw_samples[i * num_channels + c];
                    }
                    samples[i] = (sum / num_channels) / 32768.0f;
                }
            }
            return true;
        } else {
            // Skip unknown chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    fprintf(stderr, "Error: No data chunk found in WAV file\n");
    return false;
}

// ============================================================================
// NPY File I/O
// ============================================================================

// Simple NPY header parser
static bool parse_npy_header(std::ifstream& file, std::vector<size_t>& shape, 
                             std::string& dtype, bool& fortran_order) {
    char magic[6];
    file.read(magic, 6);
    if (magic[0] != '\x93' || strncmp(magic + 1, "NUMPY", 5) != 0) {
        return false;
    }

    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len;
    if (major == 1) {
        uint16_t len16;
        file.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else {
        file.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    // Parse dtype
    size_t descr_pos = header.find("'descr':");
    if (descr_pos != std::string::npos) {
        size_t start = header.find("'", descr_pos + 8);
        size_t end = header.find("'", start + 1);
        dtype = header.substr(start + 1, end - start - 1);
    }

    // Parse fortran_order
    fortran_order = header.find("'fortran_order': True") != std::string::npos;

    // Parse shape
    size_t shape_pos = header.find("'shape':");
    if (shape_pos != std::string::npos) {
        size_t start = header.find("(", shape_pos);
        size_t end = header.find(")", start);
        std::string shape_str = header.substr(start + 1, end - start - 1);
        
        shape.clear();
        size_t pos = 0;
        while (pos < shape_str.size()) {
            size_t comma = shape_str.find(",", pos);
            if (comma == std::string::npos) comma = shape_str.size();
            std::string num_str = shape_str.substr(pos, comma - pos);
            // Trim whitespace
            size_t first = num_str.find_first_not_of(" \t");
            if (first != std::string::npos) {
                size_t last = num_str.find_last_not_of(" \t,");
                num_str = num_str.substr(first, last - first + 1);
                if (!num_str.empty()) {
                    shape.push_back(std::stoull(num_str));
                }
            }
            pos = comma + 1;
        }
    }

    return true;
}

bool load_mel_filters_npy(const std::string& path, MelFilters& filters) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open mel filters file: %s\n", path.c_str());
        return false;
    }

    std::vector<size_t> shape;
    std::string dtype;
    bool fortran_order;

    if (!parse_npy_header(file, shape, dtype, fortran_order)) {
        fprintf(stderr, "Error: Invalid NPY header in %s\n", path.c_str());
        return false;
    }

    if (shape.size() != 2) {
        fprintf(stderr, "Error: Expected 2D array for mel filters, got %zu dimensions\n", shape.size());
        return false;
    }

    // Expected shape: (201, 128) from HuggingFace
    // We need to transpose to (128, 201) for our use
    size_t n_fft = shape[0];
    size_t n_mel = shape[1];

    filters.n_mel = static_cast<int32_t>(n_mel);
    filters.n_fft = static_cast<int32_t>(n_fft);
    filters.data.resize(n_mel * n_fft);

    // Read data
    std::vector<double> raw_data(n_fft * n_mel);
    if (dtype == "<f8" || dtype == "float64") {
        file.read(reinterpret_cast<char*>(raw_data.data()), raw_data.size() * sizeof(double));
        // Transpose from (n_fft, n_mel) to (n_mel, n_fft)
        for (size_t i = 0; i < n_fft; i++) {
            for (size_t j = 0; j < n_mel; j++) {
                filters.data[j * n_fft + i] = static_cast<float>(raw_data[i * n_mel + j]);
            }
        }
    } else if (dtype == "<f4" || dtype == "float32") {
        std::vector<float> raw_float(n_fft * n_mel);
        file.read(reinterpret_cast<char*>(raw_float.data()), raw_float.size() * sizeof(float));
        // Transpose from (n_fft, n_mel) to (n_mel, n_fft)
        for (size_t i = 0; i < n_fft; i++) {
            for (size_t j = 0; j < n_mel; j++) {
                filters.data[j * n_fft + i] = raw_float[i * n_mel + j];
            }
        }
    } else {
        fprintf(stderr, "Error: Unsupported dtype: %s\n", dtype.c_str());
        return false;
    }

    return true;
}

// ============================================================================
// Mel Filterbank Generation
// ============================================================================

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void generate_mel_filters(MelFilters& filters, int n_mels, int n_fft, int sample_rate) {
    int n_fft_bins = 1 + n_fft / 2;  // 201
    float fmax = sample_rate / 2.0f;  // Nyquist frequency
    float fmin = 0.0f;

    filters.n_mel = n_mels;
    filters.n_fft = n_fft_bins;
    filters.data.resize(n_mels * n_fft_bins, 0.0f);

    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);

    // Create n_mels + 2 equally spaced points in mel scale
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }

    // Convert back to Hz
    std::vector<float> hz_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    // Convert to FFT bin indices
    std::vector<float> bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        bin_points[i] = (n_fft + 1) * hz_points[i] / sample_rate;
    }

    // Create triangular filters
    for (int m = 0; m < n_mels; m++) {
        float left = bin_points[m];
        float center = bin_points[m + 1];
        float right = bin_points[m + 2];

        for (int k = 0; k < n_fft_bins; k++) {
            float weight = 0.0f;
            if (k >= left && k <= center) {
                weight = (k - left) / (center - left);
            } else if (k >= center && k <= right) {
                weight = (right - k) / (right - center);
            }
            filters.data[m * n_fft_bins + k] = weight;
        }
    }

    // Normalize filters (slaney normalization)
    for (int m = 0; m < n_mels; m++) {
        float enorm = 2.0f / (hz_points[m + 2] - hz_points[m]);
        for (int k = 0; k < n_fft_bins; k++) {
            filters.data[m * n_fft_bins + k] *= enorm;
        }
    }
}

// ============================================================================
// Log Mel Spectrogram Computation
// ============================================================================

static void log_mel_spectrogram_worker(int ith, const float* hann, 
                                        const std::vector<float>& samples,
                                        int n_samples, int frame_size, int frame_step,
                                        int n_threads, const MelFilters& filters,
                                        MelSpectrogram& mel) {
    std::vector<float> fft_in(frame_size * 2, 0.0f);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fft = filters.n_fft;
    int i = ith;

    // Calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // Apply Hann window
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // Fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0f);
        }

        // FFT
        fft(fft_in.data(), frame_size, fft_out.data());

        // Calculate modulus^2 of complex numbers
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = fft_out[2 * j + 0] * fft_out[2 * j + 0] + 
                         fft_out[2 * j + 1] * fft_out[2 * j + 1];
        }

        // Mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            // Unrolled loop for performance
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum += fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                       fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                       fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                       fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }
            // Handle remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }
            sum = log10(std::max(sum, 1e-10));
            mel.data[j * mel.n_len + i] = static_cast<float>(sum);
        }
    }

    // Fill remaining frames with log10(1e-10)
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = static_cast<float>(sum);
        }
    }
}

bool log_mel_spectrogram(const float* samples, int n_samples,
                         const MelFilters& filters, MelSpectrogram& mel,
                         int n_threads) {
    const int frame_size = QWEN_N_FFT;
    const int frame_step = QWEN_HOP_LENGTH;

    // Center padding: n_fft//2 on each side (matches HuggingFace/librosa center=True)
    int pad_amount = frame_size / 2;

    // Create padded samples with reflective padding on both sides
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + 2 * pad_amount);

    std::copy(samples, samples + n_samples, samples_padded.begin() + pad_amount);

    for (int i = 0; i < pad_amount; i++) {
        int src_idx = pad_amount - i;
        if (src_idx < n_samples) {
            samples_padded[i] = samples[src_idx];
        } else {
            samples_padded[i] = 0.0f;
        }
    }

    for (int i = 0; i < pad_amount; i++) {
        int src_idx = n_samples - 2 - i;
        if (src_idx >= 0) {
            samples_padded[n_samples + pad_amount + i] = samples[src_idx];
        } else {
            samples_padded[n_samples + pad_amount + i] = 0.0f;
        }
    }

    int total_frames = (static_cast<int>(samples_padded.size()) - frame_size) / frame_step + 1;

    mel.n_mel = filters.n_mel;
    mel.n_len = total_frames - 1;
    mel.n_len_org = mel.n_len;
    mel.data.resize(mel.n_mel * mel.n_len);

    int compute_frames = total_frames;
    int n_fft = filters.n_fft;

#ifdef __APPLE__
    const float* hann_f = global_cache.hann_window_f;

    std::vector<float> W_cos(n_fft * frame_size);
    std::vector<float> W_sin(n_fft * frame_size);
    for (int k = 0; k < n_fft; k++) {
        for (int n = 0; n < frame_size; n++) {
            float angle = static_cast<float>(2.0 * M_PI * k * n / frame_size);
            W_cos[k * frame_size + n] = cosf(angle);
            W_sin[k * frame_size + n] = sinf(angle);
        }
    }

    std::vector<float> windowed(frame_size);
    std::vector<float> dft_re(n_fft);
    std::vector<float> dft_im(n_fft);
    std::vector<float> power(n_fft);

    std::vector<double> temp_data(mel.n_mel * compute_frames);

    for (int i = 0; i < compute_frames; i++) {
        const int offset = i * frame_step;

        vDSP_vmul(hann_f, 1, &samples_padded[offset], 1, windowed.data(), 1, frame_size);

        cblas_sgemv(CblasRowMajor, CblasNoTrans, n_fft, frame_size,
                    1.0f, W_cos.data(), frame_size, windowed.data(), 1,
                    0.0f, dft_re.data(), 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n_fft, frame_size,
                    -1.0f, W_sin.data(), frame_size, windowed.data(), 1,
                    0.0f, dft_im.data(), 1);

        DSPSplitComplex split = { dft_re.data(), dft_im.data() };
        vDSP_zvmags(&split, 1, power.data(), 1, n_fft);

        for (int j = 0; j < mel.n_mel; j++) {
            float dot;
            vDSP_dotpr(power.data(), 1, &filters.data[j * n_fft], 1, &dot, n_fft);
            temp_data[j * compute_frames + i] = log10(std::max(static_cast<double>(dot), 1e-10));
        }
    }

#else
    const double* hann = global_cache.hann_window;

    std::vector<double> temp_data(mel.n_mel * compute_frames);

    std::vector<float> fft_in(2 * frame_size, 0.0f);
    std::vector<float> fft_out(8 * frame_size);
    std::vector<double> power(n_fft);

    for (int i = 0; i < compute_frames; i++) {
        const int offset = i * frame_step;

        for (int j = 0; j < frame_size; j++) {
            fft_in[j] = static_cast<float>(hann[j] * static_cast<double>(samples_padded[offset + j]));
        }

        fft(fft_in.data(), frame_size, fft_out.data());

        for (int k = 0; k < n_fft; k++) {
            float re = fft_out[2 * k + 0];
            float im = fft_out[2 * k + 1];
            power[k] = (double)re * re + (double)im * im;
        }

        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            for (int k = 0; k < n_fft; k++) {
                sum += power[k] * static_cast<double>(filters.data[j * n_fft + k]);
            }
            temp_data[j * compute_frames + i] = log10(std::max(sum, 1e-10));
        }
    }
#endif

    // Clamping and normalization in double precision
    double mmax = -1e20;
    for (int j = 0; j < mel.n_mel; j++) {
        for (int i = 0; i < mel.n_len; i++) {
            double val = temp_data[j * compute_frames + i];
            if (val > mmax) {
                mmax = val;
            }
        }
    }

    mmax -= 8.0;

    for (int j = 0; j < mel.n_mel; j++) {
        for (int i = 0; i < mel.n_len; i++) {
            double val = temp_data[j * compute_frames + i];
            if (val < mmax) {
                val = mmax;
            }
            val = (val + 4.0) / 4.0;
            mel.data[j * mel.n_len + i] = static_cast<float>(val);
        }
    }

    return true;
}

// ============================================================================
// NPY Save/Load for Mel Spectrogram
// ============================================================================

bool save_mel_npy(const std::string& path, const MelSpectrogram& mel) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot create file: %s\n", path.c_str());
        return false;
    }

    // NPY header
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    header += std::to_string(mel.n_mel) + ", " + std::to_string(mel.n_len) + "), }";
    
    // Pad header to multiple of 64 bytes
    int header_len = header.size();
    int padding = 64 - ((10 + header_len) % 64);
    if (padding < 1) padding += 64;
    header.resize(header_len + padding - 1, ' ');
    header += '\n';

    // Write magic number and version
    file.write("\x93NUMPY", 6);
    uint8_t version[2] = {1, 0};
    file.write(reinterpret_cast<char*>(version), 2);
    
    // Write header length and header
    uint16_t hlen = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<char*>(&hlen), 2);
    file.write(header.c_str(), header.size());

    // Write data
    file.write(reinterpret_cast<const char*>(mel.data.data()), 
               mel.data.size() * sizeof(float));

    return true;
}

bool load_mel_npy(const std::string& path, MelSpectrogram& mel) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file: %s\n", path.c_str());
        return false;
    }

    std::vector<size_t> shape;
    std::string dtype;
    bool fortran_order;

    if (!parse_npy_header(file, shape, dtype, fortran_order)) {
        fprintf(stderr, "Error: Invalid NPY header in %s\n", path.c_str());
        return false;
    }

    if (shape.size() != 2) {
        fprintf(stderr, "Error: Expected 2D array, got %zu dimensions\n", shape.size());
        return false;
    }

    mel.n_mel = static_cast<int32_t>(shape[0]);
    mel.n_len = static_cast<int32_t>(shape[1]);
    mel.n_len_org = mel.n_len;
    mel.data.resize(mel.n_mel * mel.n_len);

    if (dtype == "<f4" || dtype == "float32") {
        file.read(reinterpret_cast<char*>(mel.data.data()), 
                  mel.data.size() * sizeof(float));
    } else if (dtype == "<f8" || dtype == "float64") {
        std::vector<double> raw_data(mel.n_mel * mel.n_len);
        file.read(reinterpret_cast<char*>(raw_data.data()), 
                  raw_data.size() * sizeof(double));
        for (size_t i = 0; i < raw_data.size(); i++) {
            mel.data[i] = static_cast<float>(raw_data[i]);
        }
    } else {
        fprintf(stderr, "Error: Unsupported dtype: %s\n", dtype.c_str());
        return false;
    }

    return true;
}

float compare_mel(const MelSpectrogram& a, const MelSpectrogram& b) {
    if (a.n_mel != b.n_mel || a.n_len != b.n_len) {
        fprintf(stderr, "Error: Mel spectrogram dimensions don't match: "
                "(%d, %d) vs (%d, %d)\n", a.n_mel, a.n_len, b.n_mel, b.n_len);
        return -1.0f;
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < a.data.size(); i++) {
        float diff = std::abs(a.data[i] - b.data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    return max_diff;
}
