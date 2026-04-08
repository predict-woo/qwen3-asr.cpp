cmake_minimum_required(VERSION 3.14)
project(qwen3-asr-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

# Timing instrumentation option
option(QWEN3_ASR_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_ASR_TIMING)
    add_compile_definitions(QWEN3_ASR_TIMING)
endif()

# Find threads
find_package(Threads REQUIRED)

# OpenMP — required by ggml-cpu when libggml-cpu.a is built with OpenMP
# enabled (the default on Linux). Without this, every executable that
# transitively depends on ggml-cpu will fail to link with undefined
# references to GOMP_parallel / GOMP_barrier / omp_get_thread_num.
# On macOS the Apple Accelerate path is used instead, so OpenMP is optional.
if(NOT APPLE)
    find_package(OpenMP)
endif()

# GGML library paths
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
set(GGML_BUILD_DIR "${GGML_DIR}/build")

# If the bundled ggml submodule is present and the user hasn't already
# built it, build it as a subdirectory of this project so the user only
# needs one cmake/cmake --build cycle. The legacy
# ${GGML_BUILD_DIR}/src target_link_directories path below still works
# either way (CMake's static .a files end up in the same place).
if(EXISTS "${GGML_DIR}/CMakeLists.txt"
   AND NOT EXISTS "${GGML_BUILD_DIR}/src/libggml.a"
   AND NOT EXISTS "${GGML_BUILD_DIR}/src/libggml.so")
    message(STATUS "qwen3-asr: building bundled ggml submodule via add_subdirectory(${GGML_DIR})")
    set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(${GGML_DIR} ${CMAKE_CURRENT_BINARY_DIR}/_ggml EXCLUDE_FROM_ALL)
    # When ggml is built via add_subdirectory the targets are real CMake
    # imported targets named `ggml`/`ggml-base`/`ggml-cpu`, so the
    # downstream target_link_libraries(... ggml ...) lines work without
    # needing the legacy target_link_directories ${GGML_BUILD_DIR}/src
    # path. Set a flag so we can keep both styles working.
    set(QWEN3_ASR_GGML_AS_SUBDIRECTORY ON)
endif()

# Helper: link OpenMP if available, otherwise no-op (Apple). Used by every
# target that pulls in libggml-cpu.a.
function(qwen3_link_openmp target)
    if(NOT APPLE AND OpenMP_CXX_FOUND)
        target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
    endif()
endfunction()

# Mel spectrogram library
add_library(mel_spectrogram STATIC
    src/mel_spectrogram.cpp
)
target_include_directories(mel_spectrogram PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(APPLE)
    target_link_libraries(mel_spectrogram PRIVATE
        Threads::Threads
        "-framework Accelerate"
    )
else()
    target_link_libraries(mel_spectrogram PRIVATE
        Threads::Threads
    )
endif()

# Audio encoder library (GGML-based)
add_library(audio_encoder STATIC
    src/gguf_loader.cpp
    src/audio_encoder.cpp
)
target_include_directories(audio_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(audio_encoder PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(audio_encoder PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
qwen3_link_openmp(audio_encoder)

# Text decoder library (GGML-based)
add_library(text_decoder STATIC
    src/text_decoder.cpp
)
target_include_directories(text_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(text_decoder PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(text_decoder PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
qwen3_link_openmp(text_decoder)

# Audio injection library (pure C++, no GGML dependency)
add_library(audio_injection STATIC
    src/audio_injection.cpp
)
target_include_directories(audio_injection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Qwen3 ASR library (full pipeline)
add_library(qwen3_asr STATIC
    src/qwen3_asr.cpp
)
target_include_directories(qwen3_asr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(qwen3_asr PUBLIC
    mel_spectrogram
    audio_encoder
    text_decoder
    audio_injection
    Threads::Threads
)

# ForcedAligner library
add_library(forced_aligner STATIC
    src/forced_aligner.cpp
)
target_include_directories(forced_aligner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(forced_aligner PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(forced_aligner PUBLIC
    mel_spectrogram
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
qwen3_link_openmp(forced_aligner)

# CLI executable. OpenMP propagates transitively from audio_encoder /
# text_decoder / forced_aligner via the qwen3_link_openmp() helper above.
add_executable(qwen3-asr-cli
    src/main.cpp
)
target_link_libraries(qwen3-asr-cli PRIVATE
    qwen3_asr
    forced_aligner
)

# Test executable for mel spectrogram
add_executable(test_mel
    tests/test_mel.cpp
)
target_link_libraries(test_mel PRIVATE
    mel_spectrogram
    Threads::Threads
)

# Test executable for audio encoder
add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_encoder
    Threads::Threads
)

# Test executable for text decoder
add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    text_decoder
    Threads::Threads
)

# Test executable for audio injection
add_executable(test_injection
    tests/test_injection.cpp
)
target_link_libraries(test_injection PRIVATE
    audio_injection
)

# Simple decoder test
add_executable(test_decoder_simple
    tests/test_decoder_simple.cpp
)
target_link_libraries(test_decoder_simple PRIVATE
    text_decoder
    Threads::Threads
)

# Embedding test
add_executable(test_embedding
    tests/test_embedding.cpp
)
target_include_directories(test_embedding PRIVATE
    ${GGML_DIR}/include
)
target_link_directories(test_embedding PRIVATE
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(test_embedding PRIVATE
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
qwen3_link_openmp(test_embedding)

# Debug decoder test
add_executable(test_decoder_debug
    tests/test_decoder_debug.cpp
)
target_link_libraries(test_decoder_debug PRIVATE
    text_decoder
    Threads::Threads
)

# Test decoder with reference audio
add_executable(test_decoder_with_audio
    tests/test_decoder_with_audio.cpp
)
target_link_libraries(test_decoder_with_audio PRIVATE
    text_decoder
    Threads::Threads
)

# Test decoder without audio (long sequence)
add_executable(test_decoder_no_audio
    tests/test_decoder_no_audio.cpp
)
target_link_libraries(test_decoder_no_audio PRIVATE
    text_decoder
    Threads::Threads
)

# Test decoder last position
add_executable(test_decoder_last_pos
    tests/test_decoder_last_pos.cpp
)
target_link_libraries(test_decoder_last_pos PRIVATE
    text_decoder
    Threads::Threads
)

# Test decoder lengths
add_executable(test_decoder_lengths
    tests/test_decoder_lengths.cpp
)
target_link_libraries(test_decoder_lengths PRIVATE
    text_decoder
    Threads::Threads
)

# Test attention comparison
add_executable(test_attention_compare
    tests/test_attention_compare.cpp
)
target_link_libraries(test_attention_compare PRIVATE
    text_decoder
    Threads::Threads
)

# Test Q/K comparison
add_executable(test_qk_compare
    tests/test_qk_compare.cpp
)
target_link_libraries(test_qk_compare PRIVATE
    text_decoder
    Threads::Threads
)

# Test conv only (no chunking)
add_executable(test_conv_only
    tests/test_conv_only.cpp
)
target_link_libraries(test_conv_only PRIVATE
    audio_encoder
    Threads::Threads
)

# Test encoder without chunking
add_executable(test_encoder_no_chunk
    tests/test_encoder_no_chunk.cpp
)
target_link_libraries(test_encoder_no_chunk PRIVATE
    audio_encoder
    Threads::Threads
)

# Install targets
install(TARGETS mel_spectrogram audio_encoder text_decoder audio_injection qwen3_asr forced_aligner qwen3-asr-cli
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES src/mel_spectrogram.h src/audio_encoder.h src/gguf_loader.h src/text_decoder.h src/audio_injection.h src/qwen3_asr.h src/forced_aligner.h
    DESTINATION include
)

# CTest support
enable_testing()
add_test(NAME mel_spectrogram_test
    COMMAND test_mel
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME audio_encoder_test
    COMMAND test_encoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME text_decoder_test
    COMMAND test_decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME audio_injection_test
    COMMAND test_injection
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Test conv1 output
add_executable(test_conv1
    tests/test_conv1.cpp
)
target_include_directories(test_conv1 PRIVATE
    ${GGML_DIR}/include
)
target_link_libraries(test_conv1 PRIVATE
    audio_encoder
    Threads::Threads
)
