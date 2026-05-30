// Nakshatra fabric shm ring — C++ side of the Phase A primitive.
//
// Mirrors `scripts/fabric/shm_ring.py` byte-for-byte: same 64-byte
// header at the same offsets, same framed-message layout (u32 length
// prefix + payload), same wrap-aware read/write. The Python side
// creates the file + writes the header; the C++ side attaches to it.
//
// Single producer / single consumer per ring. The daemon uses one
// ring for receiving (Python writes / daemon reads) and one for
// sending (daemon writes / Python reads). Monotonic u64 cursors —
// ~584 years of headroom at 1 GB/s before wrap.
//
// Memory ordering relies on x86 TSO + ARM acquire/release semantics
// on aligned u64 loads/stores; we use GCC __atomic_* builtins to make
// the intent explicit (acquire on cursor read, release on cursor
// write — so the consumer never observes a cursor advance before the
// payload write that caused it).
//
// C++17 single-header. No external dependencies beyond POSIX
// <sys/mman.h>, <fcntl.h>, <unistd.h>.
//
// Header-only so the existing CMakeLists.txt (which builds
// worker_daemon.cpp as a single executable target) needs no new
// translation units — just an #include in worker_daemon.cpp.

#pragma once

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace nakshatra { namespace fabric {

// ── Wire constants (MUST match scripts/fabric/shm_ring.py) ─────────

constexpr size_t SHM_HEADER_SIZE = 64;
constexpr uint32_t SHM_VERSION_MAJOR = 1;

// "RING" — first 4 header bytes. Python's MAGIC = b"RING".
constexpr uint8_t SHM_MAGIC[4] = { 'R', 'I', 'N', 'G' };

// Header field offsets — explicit so Python + C++ never drift.
constexpr size_t SHM_OFF_MAGIC = 0;          // 4 bytes
constexpr size_t SHM_OFF_VERSION = 4;        // 4 bytes (u32 LE)
constexpr size_t SHM_OFF_CAPACITY = 8;       // 8 bytes (u64 LE)
constexpr size_t SHM_OFF_WRITE_CURSOR = 16;  // 8 bytes (u64 LE)
constexpr size_t SHM_OFF_READ_CURSOR = 24;   // 8 bytes (u64 LE)

constexpr size_t SHM_LENGTH_PREFIX_SIZE = 4; // u32 LE

// Default poll interval — matches the Python side. 50µs is well
// below typical daemon decode latencies and avoids burning a CPU
// in a tight spin loop while waiting on the producer.
constexpr long SHM_POLL_INTERVAL_NS = 50 * 1000;  // 50 µs


class ShmRingError : public std::runtime_error {
public:
    explicit ShmRingError(const std::string& msg)
        : std::runtime_error(msg) {}
};


class ShmRing {
public:
    // Attach to an existing ring at `path`. Parent (Python) created
    // it; we just open + mmap + validate the header.
    //
    // Throws ShmRingError on bad magic, unsupported version, or
    // any I/O error.
    static ShmRing attach(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDWR);
        if (fd < 0) {
            throw ShmRingError(
                "ShmRing::attach: open(" + path + ") failed: "
                + std::strerror(errno));
        }
        struct stat st;
        if (::fstat(fd, &st) != 0) {
            int e = errno;
            ::close(fd);
            throw ShmRingError(
                "ShmRing::attach: fstat failed: " + std::string(std::strerror(e)));
        }
        size_t size = (size_t)st.st_size;
        if (size < SHM_HEADER_SIZE) {
            ::close(fd);
            throw ShmRingError(
                "ShmRing::attach: file too small for header: "
                + std::to_string(size));
        }
        void* map = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            int e = errno;
            ::close(fd);
            throw ShmRingError(
                "ShmRing::attach: mmap failed: " + std::string(std::strerror(e)));
        }
        uint8_t* base = (uint8_t*)map;
        // Magic check.
        if (std::memcmp(base + SHM_OFF_MAGIC, SHM_MAGIC, 4) != 0) {
            ::munmap(map, size);
            ::close(fd);
            throw ShmRingError(
                "ShmRing::attach: bad magic at " + path);
        }
        // Version check — v1 only; v2 is forbidden (mirrors Python).
        uint32_t version;
        std::memcpy(&version, base + SHM_OFF_VERSION, 4);
        if (version != SHM_VERSION_MAJOR) {
            ::munmap(map, size);
            ::close(fd);
            throw ShmRingError(
                "ShmRing::attach: unsupported version " + std::to_string(version));
        }
        uint64_t capacity;
        std::memcpy(&capacity, base + SHM_OFF_CAPACITY, 8);
        return ShmRing(fd, base, size, (size_t)capacity);
    }

    // Move-only.
    ShmRing(ShmRing&& other) noexcept
        : fd_(other.fd_), base_(other.base_), size_(other.size_),
          capacity_(other.capacity_) {
        other.fd_ = -1;
        other.base_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    ShmRing& operator=(ShmRing&& other) noexcept {
        if (this != &other) {
            release();
            fd_ = other.fd_;
            base_ = other.base_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.fd_ = -1;
            other.base_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    ShmRing(const ShmRing&) = delete;
    ShmRing& operator=(const ShmRing&) = delete;
    ~ShmRing() { release(); }

    size_t capacity() const noexcept { return capacity_; }

    // Non-blocking read. Returns true + fills `out` on success;
    // returns false when empty or producer hasn't committed yet.
    // Throws on a corrupt length header (catches the rare case where
    // a bug in the producer would otherwise read past the buffer).
    bool try_read_message(std::vector<uint8_t>& out) {
        uint64_t r = load_read_cursor();
        uint64_t w = load_write_cursor();
        uint64_t avail = w - r;
        if (avail < SHM_LENGTH_PREFIX_SIZE) return false;
        uint32_t length;
        buf_read(r, (uint8_t*)&length, SHM_LENGTH_PREFIX_SIZE);
        if ((uint64_t)length > (uint64_t)capacity_ - SHM_LENGTH_PREFIX_SIZE) {
            throw ShmRingError(
                "ShmRing::try_read_message: corrupt length "
                + std::to_string(length));
        }
        if (avail < SHM_LENGTH_PREFIX_SIZE + length) return false;
        out.resize(length);
        if (length > 0) {
            buf_read(r + SHM_LENGTH_PREFIX_SIZE, out.data(), length);
        }
        store_read_cursor(r + SHM_LENGTH_PREFIX_SIZE + length);
        return true;
    }

    // Blocking read — spin with short sleeps until a message arrives.
    // `should_exit`, if provided, is checked between polls so callers
    // can break out cleanly on signal / shutdown.
    void read_message_blocking(std::vector<uint8_t>& out,
                                const volatile bool* should_exit = nullptr) {
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = SHM_POLL_INTERVAL_NS;
        while (true) {
            if (try_read_message(out)) return;
            if (should_exit && *should_exit) {
                throw ShmRingError("read aborted by should_exit");
            }
            ::nanosleep(&ts, nullptr);
        }
    }

    // Non-blocking write. Returns true on success; false when ring
    // lacks space (caller retries). Throws on oversize message —
    // that's a misconfigured ring size, never recoverable in a loop.
    bool try_write_message(const uint8_t* data, size_t n) {
        size_t needed = SHM_LENGTH_PREFIX_SIZE + n;
        if (needed > (size_t)capacity_) {
            throw ShmRingError(
                "ShmRing::try_write_message: message size "
                + std::to_string(needed) + " exceeds capacity "
                + std::to_string(capacity_));
        }
        uint64_t r = load_read_cursor();
        uint64_t w = load_write_cursor();
        size_t free_bytes = capacity_ - (size_t)(w - r);
        if (needed > free_bytes) return false;
        uint32_t length = (uint32_t)n;
        buf_write(w, (const uint8_t*)&length, SHM_LENGTH_PREFIX_SIZE);
        if (n > 0) buf_write(w + SHM_LENGTH_PREFIX_SIZE, data, n);
        // Cursor advance IS the publication barrier — the consumer
        // either sees the old cursor (no message) or the new cursor
        // along with all payload bytes that preceded it.
        store_write_cursor(w + needed);
        return true;
    }

    // Blocking write — spin with short sleeps until space appears.
    void write_message_blocking(const uint8_t* data, size_t n) {
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = SHM_POLL_INTERVAL_NS;
        while (!try_write_message(data, n)) {
            ::nanosleep(&ts, nullptr);
        }
    }

private:
    ShmRing(int fd, uint8_t* base, size_t size, size_t capacity)
        : fd_(fd), base_(base), size_(size), capacity_(capacity) {}

    void release() noexcept {
        if (base_) { ::munmap(base_, size_); base_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }

    // Cursor access — acquire/release semantics so the payload bytes
    // ordered before the cursor store are visible after the cursor
    // load on the consumer side.
    uint64_t load_read_cursor() const {
        uint64_t* p = (uint64_t*)(base_ + SHM_OFF_READ_CURSOR);
        return __atomic_load_n(p, __ATOMIC_ACQUIRE);
    }
    uint64_t load_write_cursor() const {
        uint64_t* p = (uint64_t*)(base_ + SHM_OFF_WRITE_CURSOR);
        return __atomic_load_n(p, __ATOMIC_ACQUIRE);
    }
    void store_read_cursor(uint64_t v) {
        uint64_t* p = (uint64_t*)(base_ + SHM_OFF_READ_CURSOR);
        __atomic_store_n(p, v, __ATOMIC_RELEASE);
    }
    void store_write_cursor(uint64_t v) {
        uint64_t* p = (uint64_t*)(base_ + SHM_OFF_WRITE_CURSOR);
        __atomic_store_n(p, v, __ATOMIC_RELEASE);
    }

    // Wrap-aware payload area access. `cursor` is the monotonic u64;
    // we mod into capacity_ here.
    void buf_write(uint64_t cursor, const uint8_t* data, size_t n) {
        size_t offset = (size_t)(cursor % capacity_);
        size_t end = offset + n;
        if (end <= capacity_) {
            std::memcpy(base_ + SHM_HEADER_SIZE + offset, data, n);
            return;
        }
        size_t first = capacity_ - offset;
        std::memcpy(base_ + SHM_HEADER_SIZE + offset, data, first);
        std::memcpy(base_ + SHM_HEADER_SIZE, data + first, n - first);
    }
    void buf_read(uint64_t cursor, uint8_t* out, size_t n) {
        size_t offset = (size_t)(cursor % capacity_);
        size_t end = offset + n;
        if (end <= capacity_) {
            std::memcpy(out, base_ + SHM_HEADER_SIZE + offset, n);
            return;
        }
        size_t first = capacity_ - offset;
        std::memcpy(out, base_ + SHM_HEADER_SIZE + offset, first);
        std::memcpy(out + first, base_ + SHM_HEADER_SIZE, n - first);
    }

    int fd_;
    uint8_t* base_;
    size_t size_;
    size_t capacity_;
};

} } // namespace nakshatra::fabric
