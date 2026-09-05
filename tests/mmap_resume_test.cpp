#include "../src/common/mmap_resume.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using Count = std::uint64_t;
struct Interval { std::uint64_t global, local, len; };

static std::filesystem::path make_temp_dir() {
    std::filesystem::create_directories("build");
    std::string pattern = "build/oneesan-mmap-resume-XXXXXX";
    std::vector<char> buf(pattern.begin(), pattern.end());
    buf.push_back('\0');
    char* p = ::mkdtemp(buf.data());
    if (!p) throw std::runtime_error("mkdtemp failed");
    return p;
}

int main() {
    namespace r = oneesan::mmap_resume;
    const auto dir = make_temp_dir();
    try {
        r::DirectoryLock lock(dir);
        bool second_lock_rejected = false;
        try { r::DirectoryLock second(dir); }
        catch (const std::exception&) { second_lock_rejected = true; }
        assert(second_lock_rejected);

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
        cp.groups = 10;
        cp.done.assign((cp.groups + 7) / 8, 0);
        cp.mark_done(1);
        cp.mark_done(9);
        const auto checkpoint = dir / "checkpoint.state";
        r::save_checkpoint_atomic(checkpoint, cp);
        auto loaded = r::load_checkpoint(checkpoint);
        assert(loaded.is_done(1));
        assert(loaded.is_done(9));
        assert(!loaded.is_done(8));
        r::validate_checkpoint_identity(loaded, 9, 4294967291ULL, 256, 7, 123456789, 16, 8);

        const auto main_path = dir / "main.bin";
        const auto block_path = dir / "block.bin";
        const int mfd = ::open(main_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        const int bfd = ::open(block_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        assert(mfd >= 0 && bfd >= 0);
        assert(::ftruncate(mfd, 16 * sizeof(Count)) == 0);
        assert(::ftruncate(bfd, 8 * sizeof(Count)) == 0);
        auto* main_map = static_cast<Count*>(::mmap(nullptr, 16 * sizeof(Count), PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0));
        auto* block_map = static_cast<Count*>(::mmap(nullptr, 8 * sizeof(Count), PROT_READ | PROT_WRITE, MAP_SHARED, bfd, 0));
        assert(main_map != MAP_FAILED && block_map != MAP_FAILED);
        for (Count i = 0; i < 16; ++i) main_map[i] = 1000 + i;
        for (Count i = 0; i < 8; ++i) block_map[i] = 2000 + i;
        assert(::msync(main_map, 16 * sizeof(Count), MS_SYNC) == 0);
        assert(::msync(block_map, 8 * sizeof(Count), MS_SYNC) == 0);

        std::vector<Interval> mi{{2, 0, 3}, {10, 3, 2}};
        std::vector<Interval> bi{{1, 0, 2}, {6, 2, 2}};
        std::vector<Count> main_original(5), block_original(4);
        for (const auto& x : mi) std::copy_n(main_map + x.global, x.len, main_original.data() + x.local);
        for (const auto& x : bi) std::copy_n(block_map + x.global, x.len, block_original.data() + x.local);

        const auto undo_dir = dir / "undo";
        const auto undo = r::journal_name(undo_dir, 2, 8, 5, 3);
        const auto header = r::make_journal_header(9, 2, 8, 5, 3, main_original.size(), block_original.size());
        r::write_journal_atomic(undo, header,
                                main_original.data(), main_original.size(),
                                block_original.data(), block_original.size());
        assert(std::filesystem::exists(undo));
        assert(!std::filesystem::exists(undo.string() + ".tmp"));

        // Simulate a crash after an in-place update has reached the mmap file but
        // before the group completion bit was committed.
        for (const auto& x : mi) std::fill_n(main_map + x.global, x.len, 9000);
        for (const auto& x : bi) std::fill_n(block_map + x.global, x.len, 8000);
        r::sync_intervals(main_map, mi);
        r::sync_intervals(block_map, bi);

        r::restore_journal(undo, header, main_map, mi, block_map, bi);
        for (const auto& x : mi) {
            for (std::uint64_t i = 0; i < x.len; ++i) assert(main_map[x.global + i] == main_original[x.local + i]);
        }
        for (const auto& x : bi) {
            for (std::uint64_t i = 0; i < x.len; ++i) assert(block_map[x.global + i] == block_original[x.local + i]);
        }
        r::durable_unlink(undo);
        assert(!std::filesystem::exists(undo));

        ::munmap(main_map, 16 * sizeof(Count));
        ::munmap(block_map, 8 * sizeof(Count));
        ::close(mfd);
        ::close(bfd);
        std::filesystem::remove_all(dir);
        std::cout << "mmap resume primitives: PASS\n";
        return 0;
    } catch (...) {
        std::filesystem::remove_all(dir);
        throw;
    }
}
