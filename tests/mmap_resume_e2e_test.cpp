#include "../src/common/mmap_resume.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using Count = std::uint64_t;
struct Interval { std::uint64_t global, local, len; };
namespace r = oneesan::mmap_resume;

struct Group {
    std::uint32_t id;
    std::vector<Interval> main_iv;
    std::vector<Interval> block_iv;
};

struct Mapping {
    int fd = -1;
    Count* p = nullptr;
    size_t count = 0;
    void open(const std::filesystem::path& path, size_t n, bool fresh) {
        count = n;
        fd = ::open(path.c_str(), O_RDWR | O_CREAT | (fresh ? O_TRUNC : 0), 0644);
        if (fd < 0) throw std::runtime_error("open failed");
        if (fresh && ::ftruncate(fd, static_cast<off_t>(n * sizeof(Count))) != 0) throw std::runtime_error("ftruncate failed");
        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size != static_cast<off_t>(n * sizeof(Count))) throw std::runtime_error("mapped file size mismatch");
        p = static_cast<Count*>(::mmap(nullptr, n * sizeof(Count), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
    }
    void close() {
        if (p && p != MAP_FAILED) { ::munmap(p, count * sizeof(Count)); p = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }
    ~Mapping() { close(); }
};

static std::filesystem::path temp_dir() {
    std::filesystem::create_directories("build");
    std::string pattern = "build/oneesan-mmap-e2e-XXXXXX";
    std::vector<char> buf(pattern.begin(), pattern.end());
    buf.push_back('\0');
    char* p = ::mkdtemp(buf.data());
    if (!p) throw std::runtime_error("mkdtemp failed");
    return p;
}

static size_t interval_count(const std::vector<Interval>& iv) {
    size_t n = 0;
    for (const auto& x : iv) n = std::max(n, static_cast<size_t>(x.local + x.len));
    return n;
}

static std::vector<Count> gather(const Count* base, const std::vector<Interval>& iv) {
    std::vector<Count> out(interval_count(iv));
    for (const auto& x : iv) std::copy_n(base + x.global, x.len, out.data() + x.local);
    return out;
}

static void apply_group(Count* main, Count* block, const Group& g) {
    for (const auto& x : g.main_iv) {
        for (std::uint64_t i = 0; i < x.len; ++i) main[x.global + i] += Count(1000 * (g.id + 1) + x.local + i);
    }
    for (const auto& x : g.block_iv) {
        for (std::uint64_t i = 0; i < x.len; ++i) block[x.global + i] += Count(7000 * (g.id + 1) + x.local + i);
    }
}

static void apply_group(std::vector<Count>& main, std::vector<Count>& block, const Group& g) {
    apply_group(main.data(), block.data(), g);
}

static r::JournalHeader header_for(const r::Checkpoint& cp, const Group& g) {
    return r::make_journal_header(cp.n, cp.row, cp.p_hi, cp.p_lo, g.id,
                                  interval_count(g.main_iv), interval_count(g.block_iv));
}

static std::filesystem::path undo_for(const std::filesystem::path& dir, const r::Checkpoint& cp, const Group& g) {
    return r::journal_name(dir / "undo", cp.row, cp.p_hi, cp.p_lo, g.id);
}

static void journal_before_update(const std::filesystem::path& dir, const r::Checkpoint& cp,
                                  Count* main, Count* block, const Group& g) {
    const auto main_old = gather(main, g.main_iv);
    const auto block_old = gather(block, g.block_iv);
    r::write_journal_atomic(undo_for(dir, cp, g), header_for(cp, g),
                            main_old.data(), main_old.size(), block_old.data(), block_old.size());
}

static void commit_group(const std::filesystem::path& dir, const std::filesystem::path& checkpoint,
                         r::Checkpoint& cp, Count* main, Count* block, const Group& g) {
    journal_before_update(dir, cp, main, block, g);
    apply_group(main, block, g);
    r::sync_intervals(main, g.main_iv);
    r::sync_intervals(block, g.block_iv);
    cp.mark_done(g.id);
    r::save_checkpoint_atomic(checkpoint, cp);
    r::durable_unlink(undo_for(dir, cp, g));
}

int main() {
    const auto dir = temp_dir();
    try {
        r::DirectoryLock process_lock(dir);
        const auto main_path = dir / "main.bin";
        const auto block_path = dir / "blocked.bin";
        const auto checkpoint_path = dir / "checkpoint.state";
        Mapping main_map, block_map;
        main_map.open(main_path, 24, true);
        block_map.open(block_path, 12, true);

        std::vector<Count> initial_main(24), initial_block(12);
        for (size_t i = 0; i < initial_main.size(); ++i) initial_main[i] = 100 + i;
        for (size_t i = 0; i < initial_block.size(); ++i) initial_block[i] = 500 + i;
        std::copy(initial_main.begin(), initial_main.end(), main_map.p);
        std::copy(initial_block.begin(), initial_block.end(), block_map.p);
        assert(::msync(main_map.p, initial_main.size() * sizeof(Count), MS_SYNC) == 0);
        assert(::msync(block_map.p, initial_block.size() * sizeof(Count), MS_SYNC) == 0);

        const std::vector<Group> groups{
            {0, {{0, 0, 3}, {12, 3, 2}}, {{0, 0, 2}}},
            {1, {{3, 0, 4}, {14, 4, 2}}, {{2, 0, 3}}},
            {2, {{7, 0, 5}, {16, 5, 3}}, {{5, 0, 4}}},
        };

        // The groups are intentionally disjoint, matching the production scheduler invariant.
        std::vector<int> main_owner(24, -1), block_owner(12, -1);
        for (const auto& g : groups) {
            for (const auto& x : g.main_iv) for (std::uint64_t i = 0; i < x.len; ++i) assert(std::exchange(main_owner[x.global + i], g.id) == -1);
            for (const auto& x : g.block_iv) for (std::uint64_t i = 0; i < x.len; ++i) assert(std::exchange(block_owner[x.global + i], g.id) == -1);
        }

        auto expected_main = initial_main;
        auto expected_block = initial_block;
        for (const auto& g : groups) apply_group(expected_main, expected_block, g);

        r::Checkpoint cp;
        cp.n = 9;
        cp.modulus = 4294967291ULL;
        cp.target_mib = 256;
        cp.max_window = 7;
        cp.executable_fingerprint = 0x12345678ULL;
        cp.main_count = initial_main.size();
        cp.block_count = initial_block.size();
        cp.row = 3;
        cp.p_hi = 8;
        cp.p_lo = 5;
        cp.groups = groups.size();
        cp.done.assign((cp.groups + 7) / 8, 0);
        r::save_checkpoint_atomic(checkpoint_path, cp);

        // Parallel completion can be out of group-id order: g0 and g2 commit first.
        commit_group(dir, checkpoint_path, cp, main_map.p, block_map.p, groups[0]);
        commit_group(dir, checkpoint_path, cp, main_map.p, block_map.p, groups[2]);
        assert(cp.is_done(0) && !cp.is_done(1) && cp.is_done(2));

        // g1 reaches durable mmap state, then the process dies before mark_done().
        journal_before_update(dir, cp, main_map.p, block_map.p, groups[1]);
        apply_group(main_map.p, block_map.p, groups[1]);
        r::sync_intervals(main_map.p, groups[1].main_iv);
        r::sync_intervals(block_map.p, groups[1].block_iv);
        assert(std::filesystem::exists(undo_for(dir, cp, groups[1])));

        // Simulate process death/restart by closing and reopening all persistent state.
        main_map.close();
        block_map.close();
        cp = {};
        main_map.open(main_path, 24, false);
        block_map.open(block_path, 12, false);
        cp = r::load_checkpoint(checkpoint_path);
        assert(cp.is_done(0) && !cp.is_done(1) && cp.is_done(2));
        r::validate_checkpoint_identity(cp, 9, 4294967291ULL, 256, 7, 0x12345678ULL, 24, 12);

        // Only the incomplete group has an undo journal and is rolled back.
        const auto g1_header = header_for(cp, groups[1]);
        r::restore_journal(undo_for(dir, cp, groups[1]), g1_header,
                           main_map.p, groups[1].main_iv, block_map.p, groups[1].block_iv);
        r::durable_unlink(undo_for(dir, cp, groups[1]));
        assert(!std::filesystem::exists(undo_for(dir, cp, groups[1])));

        // Already committed groups must survive recovery unchanged.
        auto after_recovery_main = initial_main;
        auto after_recovery_block = initial_block;
        apply_group(after_recovery_main, after_recovery_block, groups[0]);
        apply_group(after_recovery_main, after_recovery_block, groups[2]);
        assert(std::equal(after_recovery_main.begin(), after_recovery_main.end(), main_map.p));
        assert(std::equal(after_recovery_block.begin(), after_recovery_block.end(), block_map.p));

        // Resume executes only g1, then finalizes the checkpoint.
        commit_group(dir, checkpoint_path, cp, main_map.p, block_map.p, groups[1]);
        assert(cp.all_done());
        cp.complete = true;
        r::save_checkpoint_atomic(checkpoint_path, cp);
        const auto final_cp = r::load_checkpoint(checkpoint_path);
        assert(final_cp.complete && final_cp.all_done());

        assert(std::equal(expected_main.begin(), expected_main.end(), main_map.p));
        assert(std::equal(expected_block.begin(), expected_block.end(), block_map.p));
        for (const auto& g : groups) assert(!std::filesystem::exists(undo_for(dir, cp, g)));

        main_map.close();
        block_map.close();
        std::filesystem::remove_all(dir);
        std::cout << "mmap crash/restart e2e: PASS\n";
    } catch (...) {
        std::filesystem::remove_all(dir);
        throw;
    }
}
