#include "qwen3_asr.h"

#include "httplib.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace qwen3_asr;

struct server_params {
    std::string host = "127.0.0.1";
    std::string model_path = "models/qwen3-asr-0.6b-q8_0.gguf";
#if defined(_WIN32)
    std::string tmp_dir;
#else
    std::string tmp_dir = "/tmp";
#endif
    int32_t port = 8080;
    int32_t n_threads = std::min<int32_t>(4, std::max<int32_t>(1, std::thread::hardware_concurrency()));
    int32_t max_tokens = 1024;
    int32_t max_upload_mb = 512;
    int32_t read_timeout_sec = 600;
    int32_t write_timeout_sec = 600;
    bool convert_audio = false;
};

static void print_usage(const char * prog, const server_params & params) {
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <path>       Path to ASR GGUF model (default: %s)\n", params.model_path.c_str());
    fprintf(stderr, "  --host <host>            Host to bind (default: %s)\n", params.host.c_str());
    fprintf(stderr, "  --port <port>            Port to bind (default: %d)\n", params.port);
    fprintf(stderr, "  -t, --threads <n>        Number of inference threads (default: %d)\n", params.n_threads);
    fprintf(stderr, "  --max-tokens <n>         Maximum tokens to generate (default: %d)\n", params.max_tokens);
    fprintf(stderr, "  --max-upload-mb <n>      Maximum upload size in MiB (default: %d)\n", params.max_upload_mb);
    fprintf(stderr, "  --tmp-dir <path>         Directory for uploaded audio files (default: %s)\n", params.tmp_dir.c_str());
    fprintf(stderr, "  --convert                Convert uploaded audio to 16kHz mono WAV with ffmpeg\n");
    fprintf(stderr, "  -h, --help               Show this help message\n\n");
    fprintf(stderr, "OpenAI-compatible endpoint:\n");
    fprintf(stderr, "  POST /v1/audio/transcriptions\n\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  curl http://%s:%d/v1/audio/transcriptions \\\n", params.host.c_str(), params.port);
    fprintf(stderr, "    -F file=@audio.wav \\\n");
    fprintf(stderr, "    -F model=qwen3-asr \\\n");
    fprintf(stderr, "    -F response_format=json\n");
}

static bool parse_args(int argc, char ** argv, server_params & params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "-m" || arg == "--model") {
            const char * value = require_value(arg.c_str());
            if (!value) return false;
            params.model_path = value;
        } else if (arg == "--host") {
            const char * value = require_value(arg.c_str());
            if (!value) return false;
            params.host = value;
        } else if (arg == "--port") {
            const char * value = require_value(arg.c_str());
            if (!value) return false;
            params.port = std::atoi(value);
        } else if (arg == "-t" || arg == "--threads") {
            const char * value = require_value(arg.c_str());
            if (!value) return false;
            params.n_threads = std::atoi(value);
        } else if (arg == "--max-tokens") {
            const char * value = require_value(arg.c_str());
            if (!value) return false;
            params.max_tokens = std::atoi(value);
        } else if (arg == "--max-upload-mb") {
            const char * value = require_value(arg.c_str());
            if (!value) return false;
            params.max_upload_mb = std::atoi(value);
        } else if (arg == "--tmp-dir") {
            const char * value = require_value(arg.c_str());
            if (!value) return false;
            params.tmp_dir = value;
        } else if (arg == "--convert") {
            params.convert_audio = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0], params);
            std::exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    if (params.port <= 0 || params.port > 65535) {
        fprintf(stderr, "error: invalid port: %d\n", params.port);
        return false;
    }
    if (params.n_threads <= 0) {
        fprintf(stderr, "error: --threads must be positive\n");
        return false;
    }
    if (params.max_tokens <= 0) {
        fprintf(stderr, "error: --max-tokens must be positive\n");
        return false;
    }
    if (params.max_upload_mb <= 0) {
        fprintf(stderr, "error: --max-upload-mb must be positive\n");
        return false;
    }

    return true;
}

static std::string json_escape(const std::string & input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (unsigned char c : input) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

static std::string make_error_json(const std::string & message, const std::string & type = "invalid_request_error") {
    return std::string("{\"error\":{\"message\":\"") + json_escape(message) +
           "\",\"type\":\"" + json_escape(type) + "\"}}";
}

static std::string form_value(const httplib::Request & req, const char * name, const std::string & fallback = "") {
    if (!req.has_file(name)) {
        return fallback;
    }
    return req.get_file_value(name).content;
}

static std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static std::string normalize_language_token(std::string lang) {
    lang.erase(lang.begin(), std::find_if(lang.begin(), lang.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    lang.erase(std::find_if(lang.rbegin(), lang.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), lang.end());

    lang = lower_copy(lang);
    if (lang == "<|zh|>") return "chinese";
    if (lang == "<|en|>") return "english";
    if (lang == "<|ko|>") return "korean";
    if (lang == "<|ja|>") return "japanese";
    return lang;
}

static std::string make_temp_path(const std::string & dir, const std::string & suffix) {
#if defined(_WIN32)
    char temp_dir[MAX_PATH + 1] = {};
    if (!dir.empty()) {
        std::string dir_with_sep = dir;
        if (dir_with_sep.back() != '\\' && dir_with_sep.back() != '/') {
            dir_with_sep += '\\';
        }
        std::snprintf(temp_dir, sizeof(temp_dir), "%s", dir_with_sep.c_str());
    } else if (GetTempPathA(MAX_PATH, temp_dir) == 0) {
        return "";
    }

    char temp_file[MAX_PATH + 1] = {};
    if (GetTempFileNameA(temp_dir, "qas", 0, temp_file) == 0) {
        return "";
    }

    std::string path = temp_file;
    DeleteFileA(path.c_str());
    if (!suffix.empty()) {
        path += suffix;
    }
    return path;
#else
    std::string tmpl = dir;
    if (!tmpl.empty() && tmpl.back() != '/') {
        tmpl += '/';
    }
    tmpl += "qwen3-asr-server-XXXXXX";
    tmpl += suffix;

    std::vector<char> path(tmpl.begin(), tmpl.end());
    path.push_back('\0');

    int fd = mkstemps(path.data(), static_cast<int>(suffix.size()));
    if (fd < 0) {
        return "";
    }
    close(fd);
    return std::string(path.data());
#endif
}

static bool write_file(const std::string & path, const std::string & content, std::string & error) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        error = "failed to open temporary file for writing: " + path;
        return false;
    }
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!out) {
        error = "failed to write temporary file: " + path;
        return false;
    }
    return true;
}

static std::string shell_quote(const std::string & s) {
#if defined(_WIN32)
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        out += c;
    }
    out += "\"";
    return out;
#else
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
#endif
}

static bool convert_to_wav(const std::string & input_path, const std::string & output_path, std::string & error) {
    const std::string cmd = "ffmpeg -hide_banner -loglevel error -y -i " +
                            shell_quote(input_path) +
                            " -ar 16000 -ac 1 -c:a pcm_s16le " +
                            shell_quote(output_path);
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        error = "ffmpeg failed to convert uploaded audio to 16kHz mono WAV";
        return false;
    }
    return true;
}

static std::string extension_from_filename(const std::string & filename) {
    const size_t slash = filename.find_last_of("/\\");
    const size_t dot = filename.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return ".audio";
    }
    std::string ext = filename.substr(dot);
    if (ext.size() > 16) {
        return ".audio";
    }
    for (char c : ext) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '.') {
            return ".audio";
        }
    }
    return ext;
}

static std::string detect_language(const std::string & asr_text) {
    const std::string prefix = "language ";
    if (asr_text.size() < prefix.size() || asr_text.compare(0, prefix.size(), prefix) != 0) {
        return "";
    }

    size_t pos = prefix.size();
    if (pos >= asr_text.size() || !std::isupper(static_cast<unsigned char>(asr_text[pos]))) {
        return "";
    }

    ++pos;
    while (pos < asr_text.size() && std::islower(static_cast<unsigned char>(asr_text[pos]))) {
        ++pos;
    }

    std::string lang = asr_text.substr(prefix.size(), pos - prefix.size());
    return lower_copy(lang);
}

static std::string extract_transcript(const std::string & asr_text) {
    const std::string prefix = "language ";
    if (asr_text.size() < prefix.size() || asr_text.compare(0, prefix.size(), prefix) != 0) {
        return asr_text;
    }

    size_t pos = prefix.size();
    if (pos >= asr_text.size() || !std::isupper(static_cast<unsigned char>(asr_text[pos]))) {
        return asr_text;
    }

    ++pos;
    while (pos < asr_text.size() && std::islower(static_cast<unsigned char>(asr_text[pos]))) {
        ++pos;
    }
    while (pos < asr_text.size() && std::isspace(static_cast<unsigned char>(asr_text[pos]))) {
        ++pos;
    }

    return asr_text.substr(pos);
}

static std::string response_json(const std::string & raw_text) {
    const std::string text = extract_transcript(raw_text);
    return std::string("{\"text\":\"") + json_escape(text) + "\"}";
}

static std::string response_verbose_json(const transcribe_result & result, const std::string & raw_text, double duration_sec) {
    const std::string text = extract_transcript(raw_text);
    const std::string language = result.language.empty() ? detect_language(raw_text) : normalize_language_token(result.language);

    char duration_buf[64];
    std::snprintf(duration_buf, sizeof(duration_buf), "%.3f", duration_sec);

    return std::string("{\"task\":\"transcribe\",\"language\":\"") + json_escape(language) +
           "\",\"duration\":" + duration_buf +
           ",\"text\":\"" + json_escape(text) +
           "\",\"segments\":[]}";
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    server_params sparams;
    if (!parse_args(argc, argv, sparams)) {
        print_usage(argv[0], sparams);
        return 1;
    }

    Qwen3ASR asr;
    fprintf(stderr, "Loading model: %s\n", sparams.model_path.c_str());
    if (!asr.load_model(sparams.model_path)) {
        fprintf(stderr, "error: failed to load model: %s\n", asr.get_error().c_str());
        return 2;
    }

    std::mutex asr_mutex;
    std::atomic<bool> ready{true};

    httplib::Server server;
    server.set_read_timeout(sparams.read_timeout_sec, 0);
    server.set_write_timeout(sparams.write_timeout_sec, 0);
    server.set_payload_max_length(static_cast<size_t>(sparams.max_upload_mb) * 1024 * 1024);
    server.set_default_headers({
        {"Server", "qwen3-asr.cpp"},
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "content-type, authorization"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
    });

    server.Options("/v1/audio/transcriptions", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
    });

    server.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(ready.load() ? "{\"status\":\"ok\"}" : "{\"status\":\"loading\"}", "application/json");
    });

    server.Get("/", [&](const httplib::Request &, httplib::Response & res) {
        const std::string body =
            "qwen3-asr.cpp OpenAI-compatible server\n\n"
            "POST /v1/audio/transcriptions\n"
            "  multipart/form-data fields:\n"
            "    file=@audio.wav\n"
            "    model=qwen3-asr\n"
            "    language=<optional>\n"
            "    response_format=json|text|verbose_json\n";
        res.set_content(body, "text/plain");
    });

    server.Post("/v1/audio/transcriptions", [&](const httplib::Request & req, httplib::Response & res) {
        if (!req.is_multipart_form_data()) {
            res.status = 400;
            res.set_content(make_error_json("request must be multipart/form-data"), "application/json");
            return;
        }
        if (!req.has_file("file")) {
            res.status = 400;
            res.set_content(make_error_json("missing required multipart field: file"), "application/json");
            return;
        }

        const auto audio_file = req.get_file_value("file");
        const std::string response_format = lower_copy(form_value(req, "response_format", "json"));

        if (response_format != "json" && response_format != "text" && response_format != "verbose_json") {
            res.status = 400;
            res.set_content(make_error_json("unsupported response_format: " + response_format), "application/json");
            return;
        }

        const std::string input_suffix = extension_from_filename(audio_file.filename);
        std::string input_path = make_temp_path(sparams.tmp_dir, input_suffix);
        if (input_path.empty()) {
            res.status = 500;
            res.set_content(make_error_json("failed to create temporary upload file", "server_error"), "application/json");
            return;
        }

        std::string error;
        if (!write_file(input_path, audio_file.content, error)) {
            std::remove(input_path.c_str());
            res.status = 500;
            res.set_content(make_error_json(error, "server_error"), "application/json");
            return;
        }

        std::string wav_path = input_path;
        if (sparams.convert_audio) {
            wav_path = make_temp_path(sparams.tmp_dir, ".wav");
            if (wav_path.empty()) {
                std::remove(input_path.c_str());
                res.status = 500;
                res.set_content(make_error_json("failed to create temporary WAV file", "server_error"), "application/json");
                return;
            }
            if (!convert_to_wav(input_path, wav_path, error)) {
                std::remove(input_path.c_str());
                std::remove(wav_path.c_str());
                res.status = 400;
                res.set_content(make_error_json(error), "application/json");
                return;
            }
        }

        transcribe_params params;
        params.n_threads = sparams.n_threads;
        params.max_tokens = sparams.max_tokens;
        params.print_timing = false;
        params.print_progress = false;
        params.language = form_value(req, "language", "");

        double audio_duration_sec = 0.0;
        if (response_format == "verbose_json") {
            std::vector<float> samples;
            int sample_rate = 0;
            if (load_audio_file(wav_path, samples, sample_rate) && sample_rate > 0) {
                audio_duration_sec = static_cast<double>(samples.size()) / sample_rate;
            }
        }

        fprintf(stderr, "transcribing upload '%s' (%zu bytes)\n",
                audio_file.filename.c_str(), audio_file.content.size());

        transcribe_result result;
        {
            std::lock_guard<std::mutex> lock(asr_mutex);
            result = asr.transcribe(wav_path, params);
        }

        std::remove(input_path.c_str());
        if (wav_path != input_path) {
            std::remove(wav_path.c_str());
        }

        if (!result.success) {
            res.status = 400;
            res.set_content(make_error_json(result.error_msg), "application/json");
            return;
        }

        if (response_format == "text") {
            res.set_content(extract_transcript(result.text), "text/plain; charset=utf-8");
        } else if (response_format == "verbose_json") {
            res.set_content(response_verbose_json(result, result.text, audio_duration_sec), "application/json");
        } else {
            res.set_content(response_json(result.text), "application/json");
        }
    });

    fprintf(stderr, "qwen3-asr server listening at http://%s:%d\n", sparams.host.c_str(), sparams.port);
    fprintf(stderr, "OpenAI-compatible endpoint: http://%s:%d/v1/audio/transcriptions\n",
            sparams.host.c_str(), sparams.port);
    if (sparams.convert_audio) {
        fprintf(stderr, "ffmpeg conversion enabled for uploaded audio\n");
    } else {
        fprintf(stderr, "ffmpeg conversion disabled; uploaded audio must be 16kHz mono PCM WAV\n");
    }

    if (!server.listen(sparams.host.c_str(), sparams.port)) {
        fprintf(stderr, "error: failed to bind server to %s:%d\n", sparams.host.c_str(), sparams.port);
        return 3;
    }

    return 0;
}
