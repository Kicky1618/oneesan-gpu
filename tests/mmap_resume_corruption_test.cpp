#include "../src/common/mmap_resume.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using Count = std::uint64_t;
struct Interval { std::uint64_t global, local, len; };
namespace r = oneesan::mmap_resume;

static std::filesystem::path temp_dir() {
    std::filesystem::create_directories("build");
    std::string pattern = "build/oneesan-mmap-corrupt-XXXXXX";
    std::vector<char> buf(pattern.begin(), pattern.end());
    buf.push_back('\0');
    char* p = ::mkdtemp(buf.data());
    if (!p) throw std::runtime_error("mkdtemp failed");
    return p;
}

template <class F>
static void must_throw(F&& f, const char* contains) {
    try {
        f();
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(contains) == std::string::npos) {
            throw std::runtime_error(std::string("wrong exception: ") + e.what());
        }
        return;
    }
    throw std::runtime_error(std::string("expected exception containing: ") + contains);
}

static r::Checkpoint checkpoint_fixture() {
    r::Checkpoint cp;
    cp.n = 9;
    cp.modulus = 4294967291ULL;
    cp.target_mib = 256;
    cp.max_window = 7;
    cp.executable_fingerprint = 123456789;
    cp.main_count = 16;
    cp.block_count = 8;
    cp.row = 2;
    cp.p_hi = 8;
    cp.p_lo = 5;
    cp.groups = 3;
    cp.done.assign(1, 0);
    cp.mark_done(0);
    return cp;
}

int main() {
    const auto dir = temp_dir();
    try {
        const auto checkpoint = dir / "checkpoint.state";
        auto cp = checkpoint_fixture();
        r::save_checkpoint_atomic(checkpoint, cp);
        (void)r::load_checkpoint(checkpoint);

        // A single-byte metadata corruption must be detected by the file checksum.
        {
            std::fstream io(checkpoint, std::ios::in | std::ios::out | std::ios::binary);
            std::string file((std::istreambuf_iterator<char>(io)), std::istreambuf_iterator<char>());
            const auto pos = file.find("n 9\n");
            assert(pos != std::string::npos);
            file[pos + 2] = '8';
            io.clear();
            io.seekp(0);
            io.write(file.data(), static_cast<std::streamsize>(file.size()));
            io.close();
        }
        must_throw([&] { (void)r::load_checkpoint(checkpoint); }, "checksum mismatch");

        // Even with a recomputed checksum, duplicate semantic fields are rejected.
        {
            std::string body = r::serialize_checkpoint_body(cp);
            const auto insert_at = body.find("modulus ");
            assert(insert_at != std::string::npos);
            body.insert(insert_at, "n 9\n");
            std::ofstream out(checkpoint, std::ios::binary | std::ios::trunc);
            out << body << "checksum " << r::fnv1a_bytes(body.data(), body.size()) << '\n';
        }
        must_throw([&] { (void)r::load_checkpoint(checkpoint); }, "duplicate mmap checkpoint field: n");

        const auto main_path = dir / "main.bin";
        const auto block_path = dir / "block.bin";
        const int mfd = ::open(main_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        const int bfd = ::open(block_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        assert(mfd >= 0 && bfd >= 0);
        assert(::ftruncate(mfd, 8 * sizeof(Count)) == 0);
        assert(::ftruncate(bfd, 4 * sizeof(Count)) == 0);
        auto* main_map = static_cast<Count*>(::mmap(nullptr, 8 * sizeof(Count), PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0));
        auto* block_map = static_cast<Count*>(::mmap(nullptr, 4 * sizeof(Count), PROT_READ | PROT_WRITE, MAP_SHARED, bfd, 0));
        assert(main_map != MAP_FAILED && block_map != MAP_FAILED);
        for (Count i = 0; i < 8; ++i) main_map[i] = 100 + i;
        for (Count i = 0; i < 4; ++i) block_map[i] = 200 + i;
        std::vector<Interval> mi{{1, 0, 3}};
        std::vector<Interval> bi{{1, 0, 2}};
        std::vector<Count> mo{101, 102, 103}, bo{201, 202};
        const auto undo = r::journal_name(dir / "undo", 2, 8, 5, 1);
        const auto expected = r::make_journal_header(9, 2, 8, 5, 1, mo.size(), bo.size());

        // Payload bit flip is rejected before any restore occurs.
        r::write_journal_atomic(undo, expected, mo.data(), mo.size(), bo.data(), bo.size());
        {
            const int fd = ::open(undo.c_str(), O_RDWR);
            assert(fd >= 0);
            unsigned char x = 0;
            assert(::pread(fd, &x, 1, sizeof(r::JournalHeader) + 1) == 1);
            x ^= 0x40;
            assert(::pwrite(fd, &x, 1, sizeof(r::JournalHeader) + 1) == 1);
            assert(::fdatasync(fd) == 0);
            ::close(fd);
        }
        const auto main_before = main_map[1];
        must_throw([&] { r::restore_journal(undo, expected, main_map, mi, block_map, bi); }, "payload checksum mismatch");
        assert(main_map[1] == main_before);

        // A truncated journal is rejected by exact file-size validation.
        r::write_journal_atomic(undo, expected, mo.data(), mo.size(), bo.data(), bo.size());
        {
            const auto sz = std::filesystem::file_size(undo);
            assert(sz > sizeof(r::JournalHeader));
            assert(::truncate(undo.c_str(), static_cast<off_t>(sz - 1)) == 0);
        }
        must_throw([&] { r::restore_journal(undo, expected, main_map, mi, block_map, bi); }, "size mismatch");

        ::munmap(main_map, 8 * sizeof(Count));
        ::munmap(block_map, 4 * sizeof(Count));
        ::close(mfd);
        ::close(bfd);
        std::filesystem::remove_all(dir);
        std::cout << "mmap corruption detection: PASS\n";
    } catch (...) {
        std::filesystem::remove_all(dir);
        throw;
    }
}
