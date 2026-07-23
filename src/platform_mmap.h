#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace qwen3_asr {

struct mapped_file_view {
    void * addr = nullptr;
    size_t size = 0;
    void * handle = nullptr;
};

#ifdef _WIN32

inline std::string win32_error_string(const char * prefix, DWORD error_code) {
    return std::string(prefix) + " (Win32 error " + std::to_string(error_code) + ")";
}

inline bool map_file_readonly(const std::string & path, mapped_file_view & view, std::string & error_msg) {
    if (view.addr != nullptr) {
        error_msg = "Mapped view already initialized";
        return false;
    }

    HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        error_msg = win32_error_string(("Failed to open file for mapping: " + path).c_str(), GetLastError());
        return false;
    }

    LARGE_INTEGER file_size = {};
    if (!GetFileSizeEx(file, &file_size)) {
        error_msg = win32_error_string(("Failed to get file size: " + path).c_str(), GetLastError());
        CloseHandle(file);
        return false;
    }

    if (file_size.QuadPart <= 0) {
        error_msg = "Cannot map empty file: " + path;
        CloseHandle(file);
        return false;
    }

    if (static_cast<unsigned long long>(file_size.QuadPart) > std::numeric_limits<size_t>::max()) {
        error_msg = "File too large to map in current process: " + path;
        CloseHandle(file);
        return false;
    }

    HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    DWORD mapping_error = GetLastError();
    CloseHandle(file);

    if (!mapping) {
        error_msg = win32_error_string(("Failed to create file mapping: " + path).c_str(), mapping_error);
        return false;
    }

    void * addr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        error_msg = win32_error_string(("Failed to map file view: " + path).c_str(), GetLastError());
        CloseHandle(mapping);
        return false;
    }

    view.addr = addr;
    view.size = static_cast<size_t>(file_size.QuadPart);
    view.handle = mapping;
    return true;
}

inline void unmap_file(mapped_file_view & view) {
    if (view.addr) {
        UnmapViewOfFile(view.addr);
        view.addr = nullptr;
    }
    if (view.handle) {
        CloseHandle(static_cast<HANDLE>(view.handle));
        view.handle = nullptr;
    }
    view.size = 0;
}

#else

inline bool map_file_readonly(const std::string & path, mapped_file_view & view, std::string & error_msg) {
    if (view.addr != nullptr) {
        error_msg = "Mapped view already initialized";
        return false;
    }

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        error_msg = "Failed to open file for mmap: " + path + " (" + std::strerror(errno) + ")";
        return false;
    }

    struct stat st = {};
    if (fstat(fd, &st) != 0) {
        error_msg = "Failed to stat file: " + path + " (" + std::strerror(errno) + ")";
        close(fd);
        return false;
    }

    void * addr = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (addr == MAP_FAILED) {
        error_msg = "Failed to mmap file: " + path + " (" + std::strerror(errno) + ")";
        return false;
    }

    view.addr = addr;
    view.size = static_cast<size_t>(st.st_size);
    return true;
}

inline void unmap_file(mapped_file_view & view) {
    if (view.addr) {
        munmap(view.addr, view.size);
        view.addr = nullptr;
    }
    view.size = 0;
    view.handle = nullptr;
}

#endif

} // namespace qwen3_asr
