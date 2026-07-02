cmake_minimum_required(VERSION 3.14)
project(qwen3-asr.cpp VERSION 0.1.1 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

option(QWEN3_ASR_TEST "Enable test mode" ON)
option(GGML_AVX "ggml: enable AVX" ON)
option(GGML_AVX2 "ggml: enable AVX2" ON)
option(GGML_FMA "ggml: enable FMA" ON)
option(GGML_F16C "ggml: enable F16c" ON)
option(GGML_VULKAN "ggml: enable Vulkan backend" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
elseif(MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
    )
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# Timing instrumentation option
option(QWEN3_ASR_TIMING "Enable detailed timing instrumentation" ON)
if(QWEN3_ASR_TIMING)
    add_compile_definitions(QWEN3_ASR_TIMING)
endif()

add_subdirectory(ggml)
file(GLOB SOURCES "src/*.*")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/server.cpp")

find_package(Threads REQUIRED)

if (GGML_VULKAN)
    find_package(Vulkan REQUIRED)
endif ()

# ==================================== Library ====================================
add_library(qwen3-asr SHARED
        ${SOURCES}
)

if(WIN32)
    target_compile_definitions(ggml-base PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    target_compile_definitions(qwen3-asr PRIVATE QWEN3ASR_EXPORTS)
    set_target_properties(qwen3-asr PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            IMPORT_PREFIX ""
            IMPORT_SUFFIX ".lib"
    )

    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR}")

endif()

if (APPLE)
    target_link_libraries(qwen3-asr PRIVATE "-framework Accelerate")
endif ()

target_include_directories(qwen3-asr PUBLIC
        ggml/include
        include
        src
)
target_include_directories(qwen3-asr PRIVATE ggml/src)

target_link_libraries(qwen3-asr PUBLIC ggml PRIVATE Threads::Threads)

# ==================================== CLI ====================================
add_executable(qwen3-asr-cli
        cli/main.cpp
)

target_include_directories(qwen3-asr-cli PRIVATE
        ggml/include
        ggml/src
        include
        src
)

target_link_libraries(qwen3-asr-cli PRIVATE qwen3-asr ggml Threads::Threads)

# ==================================== Server ====================================
add_executable(qwen3-asr-server
        src/server.cpp
)

target_include_directories(qwen3-asr-server PRIVATE
        ggml/include
        ggml/src
        include
        src
        third_party
)

target_link_libraries(qwen3-asr-server PRIVATE qwen3-asr ggml Threads::Threads)

# ==================================== Test  ====================================
if (QWEN3_ASR_TEST)
    # general test
    add_executable(qwen3asr-test tests/qwen3asr_test.cpp)
    target_include_directories(qwen3asr-test PRIVATE ggml/include include src)
    target_link_libraries(qwen3asr-test PRIVATE qwen3-asr ggml Threads::Threads)

    # unit test
    enable_testing()
    add_executable(test_attention_compare tests/test_attention_compare.cpp)
    add_executable(test_conv1 tests/test_conv1.cpp)
    add_executable(test_conv2d tests/test_conv2d.cpp)
    add_executable(test_conv_no_chunk tests/test_conv_no_chunk.cpp)
    add_executable(test_conv_only tests/test_conv_only.cpp)
    add_executable(test_decoder tests/test_decoder.cpp)
    add_executable(test_decoder_50 tests/test_decoder_50.cpp)
    add_executable(test_decoder_debug tests/test_decoder_debug.cpp)
    add_executable(test_decoder_last_pos tests/test_decoder_last_pos.cpp)
    add_executable(test_decoder_lengths tests/test_decoder_lengths.cpp)
    add_executable(test_decoder_no_audio tests/test_decoder_no_audio.cpp)
    add_executable(test_decoder_simple tests/test_decoder_simple.cpp)
    add_executable(test_decoder_trace tests/test_decoder_trace.cpp)
    add_executable(test_decoder_with_audio tests/test_decoder_with_audio.cpp)
    add_executable(test_embedding tests/test_embedding.cpp)
    add_executable(test_encoder tests/test_encoder.cpp)
    add_executable(test_encoder_no_chunk tests/test_encoder_no_chunk.cpp)
    add_executable(test_injection tests/test_injection.cpp)
    add_executable(test_kernel_load tests/test_kernel_load.cpp)
    add_executable(test_mel tests/test_mel.cpp)
    add_executable(test_mel_load tests/test_mel_load.cpp)
    add_executable(test_qk_compare tests/test_qk_compare.cpp)
    add_executable(test_tokenization tests/test_tokenization.cpp)

    target_include_directories(test_attention_compare PRIVATE ggml/include include src)
    target_include_directories(test_conv1 PRIVATE ggml/include include src)
    target_include_directories(test_conv2d PRIVATE ggml/include include src)
    target_include_directories(test_conv_no_chunk PRIVATE ggml/include include src)
    target_include_directories(test_conv_only PRIVATE ggml/include include src)
    target_include_directories(test_decoder PRIVATE ggml/include include src)
    target_include_directories(test_decoder_50 PRIVATE ggml/include include src)
    target_include_directories(test_decoder_debug PRIVATE ggml/include include src)
    target_include_directories(test_decoder_last_pos PRIVATE ggml/include include src)
    target_include_directories(test_decoder_lengths PRIVATE ggml/include include src)
    target_include_directories(test_decoder_no_audio PRIVATE ggml/include include src)
    target_include_directories(test_decoder_simple PRIVATE ggml/include include src)
    target_include_directories(test_decoder_trace PRIVATE ggml/include include src)
    target_include_directories(test_decoder_with_audio PRIVATE ggml/include include src)
    target_include_directories(test_embedding PRIVATE ggml/include include src)
    target_include_directories(test_encoder PRIVATE ggml/include include src)
    target_include_directories(test_encoder_no_chunk PRIVATE ggml/include include src)
    target_include_directories(test_injection PRIVATE ggml/include include src)
    target_include_directories(test_kernel_load PRIVATE ggml/include include src)
    target_include_directories(test_mel PRIVATE ggml/include include src)
    target_include_directories(test_mel_load PRIVATE ggml/include include src)
    target_include_directories(test_qk_compare PRIVATE ggml/include include src)
    target_include_directories(test_tokenization PRIVATE ggml/include include src)

    target_link_libraries(test_attention_compare PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_conv1 PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_conv2d PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_conv_no_chunk PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_conv_only PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_decoder PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_decoder_50 PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_decoder_debug PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_decoder_last_pos PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_decoder_lengths PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_decoder_no_audio PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_decoder_simple PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_decoder_trace PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_decoder_with_audio PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_embedding PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_encoder PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_encoder_no_chunk PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_injection PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_kernel_load PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_mel PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_mel_load PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_qk_compare PRIVATE qwen3-asr ggml Threads::Threads)
    target_link_libraries(test_tokenization PRIVATE qwen3-asr ggml Threads::Threads)

    add_test(NAME test_tokenization COMMAND test_tokenization)
endif ()

install(TARGETS qwen3-asr qwen3-asr-cli qwen3-asr-server
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    include/audio_injection.h
    include/forced_aligner.h
    include/qwen3_asr.h
    src/audio_encoder.h
    src/gguf_loader.h
    src/mel_spectrogram.h
    src/platform_mmap.h
    src/qwen3asr_win_export.h
    src/text_decoder.h
    src/timing.h
    DESTINATION include
)
