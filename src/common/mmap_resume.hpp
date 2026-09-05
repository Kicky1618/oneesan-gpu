#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace oneesan::mmap_resume {

inline constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
inline constexpr uint64_t kFnvPrime = 1099511628211ULL;

inline uint64_t fnv1a_update(uint64_t h, const void* data, size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= kFnvPrime;
    }
    return h;
}

inline uint64_t fnv1a_bytes(const void* data, size_t bytes) {
    return fnv1a_update(kFnvOffset, data, bytes);
}

inline std::runtime_error system_error(const std::string& what, int err = errno) {
    return std::runtime_error(what + ": " + std::strerror(err));
}

inline void write_all(int fd, const void* data, size_t bytes) {
    const auto* p = static_cast<const std::byte*>(data);
    while (bytes != 0) {
        const ssize_t n = ::write(fd, p, bytes);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw system_error("write");
        }
        if (n == 0) throw std::runtime_error("write returned 0");
        p += n;
        bytes -= static_cast<size_t>(n);
    }
}

inline void pread_all(int fd, void* data, size_t bytes, off_t off) {
    auto* p = static_cast<std::byte*>(data);
    while (bytes != 0) {
        const ssize_t n = ::pread(fd, p, bytes, off);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw system_error("pread");
        }
        if (n == 0) throw std::runtime_error("unexpected EOF in undo journal");
        p += n;
        off += n;
        bytes -= static_cast<size_t>(n);
    }
}

inline uint64_t fnv1a_fd_range(int fd, off_t off, uint64_t bytes) {
    std::array<std::byte, 1 << 20> buf{};
    uint64_t h = kFnvOffset;
    while (bytes != 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes, buf.size()));
        pread_all(fd, buf.data(), chunk, off);
        h = fnv1a_update(h, buf.data(), chunk);
        off += static_cast<off_t>(chunk);
        bytes -= chunk;
    }
    return h;
}

inline void fsync_directory(const std::filesystem::path& dir) {
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) throw system_error("open directory for fsync");
    const int rc = ::fsync(fd);
    const int saved = errno;
    ::close(fd);
    if (rc != 0) throw system_error("fsync directory", saved);
}

class DirectoryLock {
    int fd_ = -1;
public:
    DirectoryLock() = default;
    explicit DirectoryLock(const std::filesystem::path& store) { acquire(store); }
    DirectoryLock(const DirectoryLock&) = delete;
    DirectoryLock& operator=(const DirectoryLock&) = delete;
    ~DirectoryLock() { if (fd_ >= 0) ::close(fd_); }

    void acquire(const std::filesystem::path& store) {
        if (fd_ >= 0) throw std::logic_error("directory lock already acquired");
        const auto path = store / ".mmap.lock";
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) throw system_error("open store lock");
        if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            const int saved = errno;
            ::close(fd_);
            fd_ = -1;
            if (saved == EWOULDBLOCK || saved == EAGAIN) {
                throw std::runtime_error("external-store directory is already in use: " + store.string());
            }
            throw system_error("flock store", saved);
        }
    }
};

inline uint64_t fnv1a_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot fingerprint executable: " + path.string());
    uint64_t h = kFnvOffset;
    std::array<char, 1 << 20> buf{};
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const auto n = in.gcount();
        h = fnv1a_update(h, buf.data(), static_cast<size_t>(n));
    }
    return h;
}

inline std::filesystem::path self_executable_path() {
#ifdef __linux__
    std::array<char, 4096> buf{};
    const ssize_t n = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
    if (n > 0) return std::filesystem::path(std::string(buf.data(), static_cast<size_t>(n)));
#endif
    throw std::runtime_error("cannot determine executable path for resume fingerprint");
}

inline uint64_t self_fingerprint() { return fnv1a_file(self_executable_path()); }

inline std::string hex_encode(const std::vector<uint8_t>& bytes) {
    if (bytes.empty()) return "-";
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.resize(bytes.size() * 2);
    for (size_t i = 0; i < bytes.size(); ++i) {
        out[2 * i] = kHex[bytes[i] >> 4];
        out[2 * i + 1] = kHex[bytes[i] & 15];
    }
    return out;
}

inline std::vector<uint8_t> hex_decode(const std::string& s) {
    if (s == "-") return {};
    if (s.size() % 2 != 0) throw std::runtime_error("invalid checkpoint bitmap hex length");
    auto nibble = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
        throw std::runtime_error("invalid checkpoint bitmap hex digit");
    };
    std::vector<uint8_t> out(s.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<uint8_t>((nibble(s[2 * i]) << 4) | nibble(s[2 * i + 1]));
    return out;
}

struct Checkpoint {
    static constexpr uint32_t kVersion = 2;
    int n = 0;
    uint64_t modulus = 0;
    int target_mib = 0;
    int max_window = 0;
    uint64_t executable_fingerprint = 0;
    uint64_t main_count = 0;
    uint64_t block_count = 0;
    int row = 0;
    int p_hi = 0;
    int p_lo = 0;
    uint32_t groups = 0;
    std::vector<uint8_t> done;
    bool complete = false;

    bool is_done(uint32_t g) const {
        if (g >= groups) throw std::out_of_range("group id outside checkpoint");
        return (done[g >> 3] >> (g & 7)) & 1u;
    }
    void mark_done(uint32_t g) {
        if (g >= groups) throw std::out_of_range("group id outside checkpoint");
        done[g >> 3] |= static_cast<uint8_t>(1u << (g & 7));
    }
    bool all_done() const {
        for (uint32_t g = 0; g < groups; ++g) if (!is_done(g)) return false;
        return true;
    }
};

inline std::string serialize_checkpoint_body(const Checkpoint& c) {
    std::ostringstream out;
    out << "ONEESAN_MMAP_CHECKPOINT_V" << Checkpoint::kVersion << '\n'
        << "n " << c.n << '\n'
        << "modulus " << c.modulus << '\n'
        << "target_mib " << c.target_mib << '\n'
        << "max_window " << c.max_window << '\n'
        << "executable_fingerprint " << c.executable_fingerprint << '\n'
        << "main_count " << c.main_count << '\n'
        << "block_count " << c.block_count << '\n'
        << "row " << c.row << '\n'
        << "p_hi " << c.p_hi << '\n'
        << "p_lo " << c.p_lo << '\n'
        << "groups " << c.groups << '\n'
        << "complete " << (c.complete ? 1 : 0) << '\n'
        << "done " << hex_encode(c.done) << '\n';
    return out.str();
}

inline std::string serialize(const Checkpoint& c) {
    const std::string body = serialize_checkpoint_body(c);
    std::ostringstream out;
    out << body << "checksum " << fnv1a_bytes(body.data(), body.size()) << '\n';
    return out.str();
}

inline void validate_checkpoint_for_save(const Checkpoint& c) {
    const size_t expected_bitmap = (static_cast<size_t>(c.groups) + 7) / 8;
    if (c.done.size() != expected_bitmap) throw std::runtime_error("checkpoint bitmap size mismatch before save");
    if (c.complete && c.groups != 0 && !c.all_done()) throw std::runtime_error("cannot save complete checkpoint with unfinished groups");
}

inline void save_checkpoint_atomic(const std::filesystem::path& path, const Checkpoint& c) {
    validate_checkpoint_for_save(c);
    const auto tmp = path.string() + ".tmp";
    const std::string body = serialize(c);
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) throw system_error("open checkpoint temp");
    try {
        write_all(fd, body.data(), body.size());
        if (::fdatasync(fd) != 0) throw system_error("fdatasync checkpoint");
    } catch (...) {
        ::close(fd);
        ::unlink(tmp.c_str());
        throw;
    }
    ::close(fd);
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        const int saved = errno;
        ::unlink(tmp.c_str());
        throw system_error("rename checkpoint", saved);
    }
    fsync_directory(path.parent_path());
}

inline Checkpoint load_checkpoint(const std::filesystem::path& path) {
    std::ifstream raw(path, std::ios::binary);
    if (!raw) throw std::runtime_error("cannot open checkpoint: " + path.string());
    const std::string file((std::istreambuf_iterator<char>(raw)), std::istreambuf_iterator<char>());
    if (file.empty() || file.back() != '\n') throw std::runtime_error("truncated mmap checkpoint");

    const size_t checksum_pos = file.rfind("checksum ");
    if (checksum_pos == std::string::npos || (checksum_pos != 0 && file[checksum_pos - 1] != '\n')) {
        throw std::runtime_error("missing mmap checkpoint checksum");
    }
    const std::string body = file.substr(0, checksum_pos);
    const std::string checksum_line = file.substr(checksum_pos);
    std::istringstream cs(checksum_line);
    std::string cs_key, cs_extra;
    uint64_t saved_checksum = 0;
    if (!(cs >> cs_key >> saved_checksum) || cs_key != "checksum" || (cs >> cs_extra)) {
        throw std::runtime_error("malformed mmap checkpoint checksum");
    }
    if (saved_checksum != fnv1a_bytes(body.data(), body.size())) {
        throw std::runtime_error("mmap checkpoint checksum mismatch");
    }

    std::istringstream in(body);
    std::string magic;
    std::getline(in, magic);
    if (magic != "ONEESAN_MMAP_CHECKPOINT_V2") throw std::runtime_error("unsupported mmap checkpoint format: " + magic);

    Checkpoint c;
    std::string key, done_hex;
    std::unordered_set<std::string> seen;
    const std::unordered_set<std::string> required = {
        "n", "modulus", "target_mib", "max_window", "executable_fingerprint",
        "main_count", "block_count", "row", "p_hi", "p_lo", "groups", "complete", "done"
    };
    while (in >> key) {
        if (!required.count(key)) throw std::runtime_error("unknown mmap checkpoint field: " + key);
        if (!seen.insert(key).second) throw std::runtime_error("duplicate mmap checkpoint field: " + key);
        if (key == "n") in >> c.n;
        else if (key == "modulus") in >> c.modulus;
        else if (key == "target_mib") in >> c.target_mib;
        else if (key == "max_window") in >> c.max_window;
        else if (key == "executable_fingerprint") in >> c.executable_fingerprint;
        else if (key == "main_count") in >> c.main_count;
        else if (key == "block_count") in >> c.block_count;
        else if (key == "row") in >> c.row;
        else if (key == "p_hi") in >> c.p_hi;
        else if (key == "p_lo") in >> c.p_lo;
        else if (key == "groups") in >> c.groups;
        else if (key == "complete") {
            int x = -1;
            in >> x;
            if (x != 0 && x != 1) throw std::runtime_error("invalid checkpoint complete flag");
            c.complete = x != 0;
        } else if (key == "done") in >> done_hex;
        if (!in) throw std::runtime_error("malformed mmap checkpoint field: " + key);
    }
    for (const auto& field : required) {
        if (!seen.count(field)) throw std::runtime_error("missing mmap checkpoint field: " + field);
    }
    c.done = hex_decode(done_hex);
    const size_t expected = (static_cast<size_t>(c.groups) + 7) / 8;
    if (c.done.size() != expected) throw std::runtime_error("checkpoint bitmap size mismatch");
    if (c.complete && c.groups != 0 && !c.all_done()) throw std::runtime_error("complete checkpoint has unfinished groups");
    return c;
}

inline void validate_checkpoint_identity(const Checkpoint& c, int n, uint64_t modulus,
                                         int target_mib, int max_window, uint64_t executable_fingerprint,
                                         uint64_t main_count, uint64_t block_count) {
    auto mismatch = [](const std::string& name, auto got, auto expected) {
        if (got != expected) {
            std::ostringstream s;
            s << "resume checkpoint " << name << " mismatch: " << got << " != " << expected;
            throw std::runtime_error(s.str());
        }
    };
    mismatch("n", c.n, n);
    mismatch("modulus", c.modulus, modulus);
    mismatch("target_mib", c.target_mib, target_mib);
    mismatch("max_window", c.max_window, max_window);
    mismatch("executable_fingerprint", c.executable_fingerprint, executable_fingerprint);
    mismatch("main_count", c.main_count, main_count);
    mismatch("block_count", c.block_count, block_count);
}

#pragma pack(push, 1)
struct JournalHeader {
    char magic[16];
    uint32_t version;
    int32_t n;
    int32_t row;
    int32_t p_hi;
    int32_t p_lo;
    uint32_t group;
    uint64_t main_count;
    uint64_t block_count;
    uint64_t payload_checksum;
    uint64_t header_checksum;
};
#pragma pack(pop)

inline uint64_t journal_header_checksum(JournalHeader h) {
    h.header_checksum = 0;
    return fnv1a_bytes(&h, sizeof(h));
}

inline JournalHeader make_journal_header(int n, int row, int p_hi, int p_lo, uint32_t group,
                                         uint64_t main_count, uint64_t block_count) {
    JournalHeader h{};
    std::memcpy(h.magic, "ONEESAN_UNDO_V2", 15);
    h.version = 2;
    h.n = n;
    h.row = row;
    h.p_hi = p_hi;
    h.p_lo = p_lo;
    h.group = group;
    h.main_count = main_count;
    h.block_count = block_count;
    return h;
}

inline bool valid_journal_header(const JournalHeader& h) {
    return std::memcmp(h.magic, "ONEESAN_UNDO_V2", 15) == 0 &&
           h.version == 2 && h.header_checksum == journal_header_checksum(h);
}

inline std::filesystem::path journal_name(const std::filesystem::path& dir,
                                          int row, int p_hi, int p_lo, uint32_t group) {
    std::ostringstream name;
    name << "r" << std::setw(2) << std::setfill('0') << row
         << "_p" << std::setw(2) << p_hi << '_' << std::setw(2) << p_lo
         << "_g" << std::setw(8) << group << ".undo";
    return dir / name.str();
}

inline JournalHeader read_journal_header(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw system_error("open undo journal");
    JournalHeader h{};
    try { pread_all(fd, &h, sizeof(h), 0); }
    catch (...) { ::close(fd); throw; }
    ::close(fd);
    if (!valid_journal_header(h)) throw std::runtime_error("invalid undo journal header: " + path.string());
    return h;
}

template <class Count>
inline void write_journal_atomic(const std::filesystem::path& final_path, JournalHeader h,
                                 const Count* main_data, size_t main_count,
                                 const Count* block_data, size_t block_count) {
    if (main_count != h.main_count || block_count != h.block_count) {
        throw std::runtime_error("undo journal payload count/header mismatch");
    }
    uint64_t payload_hash = kFnvOffset;
    if (main_count) payload_hash = fnv1a_update(payload_hash, main_data, main_count * sizeof(Count));
    if (block_count) payload_hash = fnv1a_update(payload_hash, block_data, block_count * sizeof(Count));
    h.payload_checksum = payload_hash;
    h.header_checksum = journal_header_checksum(h);

    std::filesystem::create_directories(final_path.parent_path());
    const auto tmp = final_path.string() + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) throw system_error("open undo journal temp");
    try {
        write_all(fd, &h, sizeof(h));
        if (main_count) write_all(fd, main_data, main_count * sizeof(Count));
        if (block_count) write_all(fd, block_data, block_count * sizeof(Count));
        if (::fdatasync(fd) != 0) throw system_error("fdatasync undo journal");
    } catch (...) {
        ::close(fd);
        ::unlink(tmp.c_str());
        throw;
    }
    ::close(fd);
    if (::rename(tmp.c_str(), final_path.c_str()) != 0) {
        const int saved = errno;
        ::unlink(tmp.c_str());
        throw system_error("rename undo journal", saved);
    }
    fsync_directory(final_path.parent_path());
}

template <class Count, class Interval>
inline void sync_intervals(Count* base, const std::vector<Interval>& intervals) {
    if (intervals.empty()) return;
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0) throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
    uintptr_t run_begin = 0, run_end = 0;
    auto flush = [&] {
        if (run_begin == run_end) return;
        if (::msync(reinterpret_cast<void*>(run_begin), run_end - run_begin, MS_SYNC) != 0) {
            throw system_error("msync external-store interval");
        }
    };
    for (const auto& x : intervals) {
        if (x.len == 0) continue;
        const uintptr_t raw_begin = reinterpret_cast<uintptr_t>(base + x.global);
        const uintptr_t begin = (raw_begin / static_cast<uintptr_t>(page_size)) * static_cast<uintptr_t>(page_size);
        const uintptr_t raw_end = reinterpret_cast<uintptr_t>(base + x.global + x.len);
        const uintptr_t end = ((raw_end + static_cast<uintptr_t>(page_size) - 1) / static_cast<uintptr_t>(page_size)) * static_cast<uintptr_t>(page_size);
        if (run_begin == run_end) { run_begin = begin; run_end = end; }
        else if (begin <= run_end) run_end = std::max(run_end, end);
        else { flush(); run_begin = begin; run_end = end; }
    }
    flush();
}

template <class Count, class Interval>
inline void restore_journal(const std::filesystem::path& path,
                            const JournalHeader& expected,
                            Count* main_base, const std::vector<Interval>& main_intervals,
                            Count* block_base, const std::vector<Interval>& block_intervals) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw system_error("open undo journal for restore");
    JournalHeader h{};
    try {
        pread_all(fd, &h, sizeof(h), 0);
        if (!valid_journal_header(h)) throw std::runtime_error("invalid undo journal header");
        if (h.n != expected.n || h.row != expected.row || h.p_hi != expected.p_hi ||
            h.p_lo != expected.p_lo || h.group != expected.group ||
            h.main_count != expected.main_count || h.block_count != expected.block_count) {
            throw std::runtime_error("undo journal metadata mismatch: " + path.string());
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0) throw system_error("fstat undo journal");
        const uint64_t payload_bytes = (h.main_count + h.block_count) * sizeof(Count);
        const uint64_t expected_bytes = sizeof(JournalHeader) + payload_bytes;
        if (st.st_size < 0 || static_cast<uint64_t>(st.st_size) != expected_bytes) {
            throw std::runtime_error("undo journal size mismatch: " + path.string());
        }
        const uint64_t payload_hash = fnv1a_fd_range(fd, static_cast<off_t>(sizeof(JournalHeader)), payload_bytes);
        if (payload_hash != h.payload_checksum) {
            throw std::runtime_error("undo journal payload checksum mismatch: " + path.string());
        }

        const off_t main_off = static_cast<off_t>(sizeof(JournalHeader));
        for (const auto& x : main_intervals) {
            pread_all(fd, main_base + x.global, static_cast<size_t>(x.len) * sizeof(Count),
                      main_off + static_cast<off_t>(x.local * sizeof(Count)));
        }
        const off_t block_off = main_off + static_cast<off_t>(h.main_count * sizeof(Count));
        for (const auto& x : block_intervals) {
            pread_all(fd, block_base + x.global, static_cast<size_t>(x.len) * sizeof(Count),
                      block_off + static_cast<off_t>(x.local * sizeof(Count)));
        }
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
    sync_intervals(main_base, main_intervals);
    sync_intervals(block_base, block_intervals);
}

inline void durable_unlink(const std::filesystem::path& path) {
    if (::unlink(path.c_str()) != 0 && errno != ENOENT) throw system_error("unlink");
    fsync_directory(path.parent_path());
}

}  // namespace oneesan::mmap_resume
