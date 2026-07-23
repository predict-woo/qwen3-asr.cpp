#ifndef MMAN_MULTIPLATFORM_H
#define MMAN_MULTIPLATFORM_H

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>

namespace qwen3_asr {

#define PROT_READ  0x01
#define PROT_WRITE 0x02
#define MAP_PRIVATE 0x02
#define MAP_SHARED  0x01
#define MAP_FAILED ((void*)-1)


    static void* mmap(void* /*ptr*/, size_t size, int prot, int flags, int fd, size_t offset) {

        if (fd < 0 || !(prot & PROT_READ) || !(flags & MAP_PRIVATE)) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return MAP_FAILED;
        }

        if (size == 0) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return MAP_FAILED;
        }

        HANDLE win_fd = (HANDLE)_get_osfhandle(fd);
        if (win_fd == INVALID_HANDLE_VALUE) {
            SetLastError(ERROR_INVALID_HANDLE);
            return MAP_FAILED;
        }

        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(win_fd, &fileSize)) {
            DWORD error = GetLastError();
            SetLastError(error);
            return MAP_FAILED;
        }

        if ((offset + size) > (size_t)fileSize.QuadPart) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return MAP_FAILED;
        }

        DWORD sizeLow = (DWORD)(size & 0xFFFFFFFF);
        DWORD sizeHigh = (DWORD)((size >> 32) & 0xFFFFFFFF);


        HANDLE mapping = CreateFileMappingW(
            win_fd,
            nullptr,
            PAGE_READONLY,
            sizeHigh,
            sizeLow,
            nullptr
        );
        if (mapping == nullptr) {
            DWORD error = GetLastError();
            SetLastError(error);
            return MAP_FAILED;
        }


        DWORD offsetLow = (DWORD)(offset & 0xFFFFFFFF);
        DWORD offsetHigh = (DWORD)((offset >> 32) & 0xFFFFFFFF);


        void* mapped_ptr = MapViewOfFile(
            mapping,
            FILE_MAP_READ,
            offsetHigh,
            offsetLow,
            size
        );
        CloseHandle(mapping);

        if (mapped_ptr == nullptr) {
            DWORD error = GetLastError();
            SetLastError(error);
            return MAP_FAILED;
        }

        return mapped_ptr;
    }

    static int munmap(void* ptr, size_t size) {
        (void)size;

        if (ptr == MAP_FAILED || ptr == nullptr) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return -1;
        }

        if (!UnmapViewOfFile(ptr)) {
            SetLastError(GetLastError());
            return -1;
        }

        return 0;
    }

}  // namespace qwen3_asr
#else
    #include <sys/mman.h>
    #include <unistd.h>
    #define O_BINARY O_RDONLY
#endif  // _WIN32

#endif // MMAN_MULTIPLATFORM_H