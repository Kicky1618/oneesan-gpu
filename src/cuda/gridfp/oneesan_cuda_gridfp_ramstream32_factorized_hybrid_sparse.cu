#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN

#include "ramstream32_cpu_low_sparse_persistent.hpp"
#include "ramstream32_cpu_high.hpp"
#include "ramstream32_cpu_high_direct_persistent.hpp"

static void hybrid_sparse_release_dense_host(StorageFactorHost& storage) {
    storage.low_packed_rank.clear(); storage.low_packed_rank.shrink_to_fit();
    storage.high_packed_rank.clear(); storage.high_packed_rank.shrink_to_fit();
    storage.low_all_codes.clear(); storage.low_all_codes.shrink_to_fit();
    storage.high_all_codes.clear(); storage.high_all_codes.shrink_to_fit();
    G_FACTOR.low_packed_rank.clear(); G_FACTOR.low_packed_rank.shrink_to_fit();
    G_FACTOR.high_packed_rank.clear(); G_FACTOR.high_packed_rank.shrink_to_fit();
    G_FACTOR.low_all_codes.clear(); G_FACTOR.low_all_codes.shrink_to_fit();
    G_FACTOR.high_all_codes.clear(); G_FACTOR.high_all_codes.shrink_to_fit();
    G_FACTOR.high_main_base.clear(); G_FACTOR.high_main_base.shrink_to_fit();
    G_FACTOR.high_block_base.clear(); G_FACTOR.high_block_base.shrink_to_fit();
}

static double env_nonnegative_double(const char* name, double fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (!end || *end || v < 0.0) {
        std::cerr << name << " must be a non-negative number\n";
        std::exit(2);
    }
    return v;
}

static int env_positive_int(const char* name, int fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end || v <= 0 || v > 1'000'000) {
        std::cerr << name << " must be a positive integer\n";
        std::exit(2);
    }
    return int(v);
}

static bool env_bool(const char* name, bool fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    if (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0
        || std::strcmp(s, "yes") == 0 || std::strcmp(s, "on") == 0) return true;
    if (std::strcmp(s, "0") == 0 || std::strcmp(s, "false") == 0
        || std::strcmp(s, "no") == 0 || std::strcmp(s, "off") == 0) return false;
    std::cerr << name << " must be 0/1, false/true, no/yes, or off/on\n";
    std::exit(2);
}

static bool env_nonempty(const char* name) {
    const char* s = std::getenv(name);
    return s && *s;
}

static std::vector<uint8_t> load_cpu_high_group_file(
    const char* path, size_t group_count, size_t& requested
) {
    requested = 0;
    std::vector<uint8_t> selected(group_count, 0);
    std::ifstream in(path);
    if (!in) {
        std::cerr << "cannot open CPU_HIGH_GROUPS_FILE: " << path << '\n';
        std::exit(2);
    }

    std::string line;
    size_t lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        const char* s = line.c_str();
        while (*s && std::isspace(static_cast<unsigned char>(*s))) ++s;
        if (!*s) continue;

        char* end = nullptr;
        unsigned long g = std::strtoul(s, &end, 10);
        if (end == s) {
            std::cerr << "invalid CPU HIGH group at " << path << ':' << lineno << '\n';
            std::exit(2);
        }
        while (*end && std::isspace(static_cast<unsigned char>(*end))) ++end;
        if (*end || g >= group_count) {
            std::cerr << "invalid CPU HIGH group at " << path << ':' << lineno
                      << " value=" << g << " group_count=" << group_count << '\n';
            std::exit(2);
        }
        if (!selected[size_t(g)]) {
            selected[size_t(g)] = 1;
            ++requested;
        }
    }

    std::cerr << "cpu_high_group_file path=" << path
              << " requested_groups=" << requested << '\n';
    return selected;
}

static uint64_t cpu_high_selection_hash(const std::vector<uint8_t>& selected) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t x : selected) {
        h ^= uint64_t(x);
        h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    int gpu_target_mib = argc > 3 ? std::atoi(argv[3]) : 12288;
    int cpu_workers = argc > 4 ? std::max(1, std::atoi(argv[4])) : 4;
    bool plan_only = argc > 5 && std::strcmp(argv[5], "--plan-only") == 0;
    double cpu_high_max_mib = env_nonnegative_double("CPU_HIGH_MAX_MIB", 0.0);
    int cpu_high_workers = env_positive_int("CPU_HIGH_WORKERS", cpu_workers);
    bool cpu_high_overlap = env_bool("CPU_HIGH_OVERLAP", false);
    const char* cpu_high_groups_file = std::getenv("CPU_HIGH_GROUPS_FILE");
    bool cpu_high_file_policy = cpu_high_groups_file && *cpu_high_groups_file;
    bool cpu_high_explicit_affinity = env_nonempty("CPU_HIGH_CPU_LIST");
    bool cpu_low_explicit_affinity = env_nonempty("CPU_LOW_CPU_LIST");
    const char* cpu_high_mode_env = std::getenv("CPU_HIGH_MODE");
    bool cpu_high_direct_mode = cpu_high_mode_env
        && std::strcmp(cpu_high_mode_env, "direct") == 0;
    if (cpu_high_mode_env && *cpu_high_mode_env
        && std::strcmp(cpu_high_mode_env, "scratch") != 0
        && std::strcmp(cpu_high_mode_env, "direct") != 0) {
        std::cerr << "CPU_HIGH_MODE must be scratch or direct\n";
        return 2;
    }
    const char* cpu_high_mode = cpu_high_direct_mode ? "direct" : "scratch";
    const char* cpu_high_policy = cpu_high_file_policy ? "file" : "threshold";

    int W = n + 1;
    if (W != TARGET_W || n < 2 || W > MAXW) return 1;
    if (gpu_target_mib <= 0 || cpu_workers <= 0) return 1;
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);

    auto meta0 = std::chrono::steady_clock::now();
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuLowSparseHost sparse = build_cpu_low_sparse(storage, layout, lowdesc, orbit);
    double meta_build_s = ram_seconds_since(meta0);

    WindowPlan high_wp = make_direct2d_window(true);
    WindowPlan low_wp = make_direct2d_window(false);
    auto high_jobs = make_direct2d_jobs(high_wp);
    auto all_cpu_high_jobs = make_cpu_high_jobs(W, high_wp);
    auto cpu_low_jobs = make_cpu_low_jobs(W, low_wp);

    size_t high_group_count = size_t(1) << high_wp.fixed_pos.size();
    std::vector<uint8_t> file_selected;
    size_t cpu_high_policy_requested = 0;
    if (cpu_high_file_policy)
        file_selected = load_cpu_high_group_file(
            cpu_high_groups_file, high_group_count, cpu_high_policy_requested);

    std::vector<uint8_t> cpu_high_selected(high_group_count, 0);
    std::vector<const CpuHighJob*> selected_cpu_high_jobs;
    selected_cpu_high_jobs.reserve(all_cpu_high_jobs.size());
    long double cpu_high_removed_bytes_per_row = 0.0L;
    size_t cpu_high_limit_bytes = cpu_high_max_mib > 0.0
        ? size_t(cpu_high_max_mib * double(1ULL << 20)) : 0;
    size_t gpu_high_max_bytes = 0;
    size_t gpu_high_nonempty_groups = 0;

    for (const auto& job : all_cpu_high_jobs) {
        if (!job.main_size && !job.block_size) continue;
        bool use_cpu = cpu_high_file_policy
            ? bool(file_selected[size_t(job.g)])
            : (cpu_high_limit_bytes && job.scratch_bytes <= cpu_high_limit_bytes);
        if (use_cpu) {
            cpu_high_selected[size_t(job.g)] = 1;
            selected_cpu_high_jobs.push_back(&job);
            cpu_high_removed_bytes_per_row += job.scratch_bytes;
        } else {
            gpu_high_max_bytes = std::max(gpu_high_max_bytes, job.scratch_bytes);
            ++gpu_high_nonempty_groups;
        }
    }

    uint64_t selection_hash = cpu_high_selection_hash(cpu_high_selected);
    if (cpu_high_file_policy && selected_cpu_high_jobs.size() != cpu_high_policy_requested) {
        std::cerr << "cpu_high_group_file includes "
                  << (cpu_high_policy_requested - selected_cpu_high_jobs.size())
                  << " empty HIGH groups\n";
    }

    size_t gpu_target = size_t(gpu_target_mib) << 20;
    if (gpu_high_max_bytes > gpu_target) {
        std::cerr << "hybrid-sparse remaining GPU HIGH group does not fit: high_gib="
                  << double(gpu_high_max_bytes) / double(1ULL << 30)
                  << " target_gib=" << double(gpu_target) / double(1ULL << 30)
                  << " cpu_high_policy=" << cpu_high_policy
                  << " cpu_high_max_mib=" << cpu_high_max_mib << '\n';
        return 4;
    }

    CpuHighDirectHost cpu_high_direct_meta;
    double cpu_high_direct_meta_build_s = 0.0;
    if (!selected_cpu_high_jobs.empty() && cpu_high_direct_mode) {
        auto t = std::chrono::steady_clock::now();
        cpu_high_direct_meta = build_cpu_high_direct(storage, layout, highdesc);
        cpu_high_direct_meta_build_s = ram_seconds_since(t);
    }

    MateID init = MateID(R) << (2 * (W - 1));
    Code init_rank = storage_rank_main_host(init, storage, layout);
    Code answer_rank = storage_rank_main_host(MateID(R), storage, layout);

    double highdesc_mib = double((highdesc.main_desc.size()+highdesc.block_desc.size())*sizeof(uint32_t))/(1<<20);
    double sparse_nn_orbit_mib = double(sparse.nn_orbit_ops.size()*sizeof(CpuLowSparseOrbitOp))/(1<<20);
    double sparse_nr_orbit_mib = double(sparse.nr_orbit_ops.size()*sizeof(CpuLowSparseOrbitOp))/(1<<20);
    double sparse_nl_orbit_mib = double(sparse.nl_orbit_ops.size()*sizeof(CpuLowSparseOrbitOp))/(1<<20);
    double sparse_orbit_mib = sparse_nn_orbit_mib + sparse_nr_orbit_mib + sparse_nl_orbit_mib;
    double sparse_local_closure_mib = double(sparse.local_closure_ops.size()*sizeof(CpuLowSparseClosureOp))/(1<<20);
    double sparse_cross_closure_mib = double(sparse.cross_closure_ops.size()*sizeof(CpuLowSparseClosureOp))/(1<<20);
    double sparse_closure_mib = sparse_local_closure_mib + sparse_cross_closure_mib;
    double sparse_offsets_mib = double((sparse.nn_orbit_off.size()+sparse.nr_orbit_off.size()
        +sparse.nl_orbit_off.size()+sparse.local_closure_off.size()
        +sparse.cross_closure_off.size())*sizeof(uint32_t))/(1<<20);
    double sparse_cross_mib = double(sparse.high_cross_rank.size()*sizeof(uint16_t))/(1<<20);
    double cpu_high_cross_mib_est = selected_cpu_high_jobs.empty() ? 0.0
        : double(size_t(LOW_LUT_K) * G_FACTOR.low_mask_codes.size() * sizeof(uint16_t))/(1<<20);
    double cpu_high_direct_meta_mib = double(
        cpu_high_direct_meta.orbit_ops.size()*sizeof(CpuHighOrbitOp)
        + cpu_high_direct_meta.closure_ops.size()*sizeof(CpuHighClosureOp)
        + (cpu_high_direct_meta.orbit_off.size()+cpu_high_direct_meta.closure_off.size())*sizeof(uint32_t))/(1<<20);
    double mask_mib = double((G_FACTOR.low_mask_codes.size()+G_FACTOR.low_mask_off.size()
        +G_FACTOR.high_mask_codes.size()+G_FACTOR.high_mask_off.size())*sizeof(uint32_t))/(1<<20);
    double dense_host_release_mib = double((storage.low_packed_rank.size()+storage.high_packed_rank.size()
        +G_FACTOR.low_packed_rank.size()+G_FACTOR.high_packed_rank.size())*sizeof(uint32_t))/(1<<20);
    double dense_cpu_meta_mib = double((lowdesc.main_desc.size()+lowdesc.block_desc.size())*sizeof(uint32_t)
        + orbit.rec.size()*sizeof(uint64_t))/(1<<20);

    long double auth_bytes = static_cast<long double>(layout.main_size + layout.block_size) * sizeof(Count);
    long double pcie_baseline_bytes = 2.0L * W * auth_bytes;
    long double pcie_removed_bytes = W * cpu_high_removed_bytes_per_row;
    long double pcie_remaining_bytes = pcie_baseline_bytes - pcie_removed_bytes;
    double pcie_baseline_tib = double(pcie_baseline_bytes / double(1ULL << 40));
    double pcie_removed_tib = double(pcie_removed_bytes / double(1ULL << 40));
    double pcie_remaining_tib = double(pcie_remaining_bytes / double(1ULL << 40));
    double pcie_fraction_removed = pcie_baseline_bytes
        ? double(pcie_removed_bytes / pcie_baseline_bytes) : 0.0;
    double pcie_remaining_50gib_s = double(pcie_remaining_bytes / (50.0L * double(1ULL << 30)));

    if (plan_only) {
        std::cout
            << "backend=gridfp-ramstream32-factorized-hybrid-sparse-v5.15-plan"
            << " n=" << n
            << " gpu_high_desc_mib=" << highdesc_mib
            << " gpu_mask_mib=" << mask_mib
            << " cpu_sparse_nn_orbit_mib=" << sparse_nn_orbit_mib
            << " cpu_sparse_nr_orbit_mib=" << sparse_nr_orbit_mib
            << " cpu_sparse_nl_orbit_mib=" << sparse_nl_orbit_mib
            << " cpu_sparse_orbit_mib=" << sparse_orbit_mib
            << " cpu_sparse_local_closure_mib=" << sparse_local_closure_mib
            << " cpu_sparse_cross_closure_mib=" << sparse_cross_closure_mib
            << " cpu_sparse_closure_mib=" << sparse_closure_mib
            << " cpu_sparse_offsets_mib=" << sparse_offsets_mib
            << " cpu_sparse_cross_rank_mib=" << sparse_cross_mib
            << " cpu_high_cross_rank_mib=" << cpu_high_cross_mib_est
            << " cpu_high_direct_meta_mib=" << cpu_high_direct_meta_mib
            << " cpu_high_direct_meta_build_s=" << cpu_high_direct_meta_build_s
            << " cpu_dense_meta_replaced_mib=" << dense_cpu_meta_mib
            << " dense_host_release_mib=" << dense_host_release_mib
            << " meta_build_s=" << meta_build_s
            << " gpu_high_window_max_gib=" << double(gpu_high_max_bytes)/double(1ULL<<30)
            << " cpu_workers=" << cpu_workers
            << " cpu_high_workers=" << cpu_high_workers
            << " cpu_high_mode=" << cpu_high_mode
            << " cpu_high_overlap=" << int(cpu_high_overlap)
            << " cpu_high_policy=" << cpu_high_policy
            << " cpu_high_persistent_workers=1"
            << " cpu_high_async_overlap=1"
            << " cpu_low_persistent_workers=1"
            << " cpu_low_pointer_workspace=stack"
            << " cpu_high_affinity=" << (cpu_high_explicit_affinity ? "explicit" : "default")
            << " cpu_low_affinity=" << (cpu_low_explicit_affinity ? "explicit" : "default")
            << " cpu_high_selection_hash=" << selection_hash
            << " cpu_high_max_mib=" << cpu_high_max_mib
            << " cpu_high_groups=" << selected_cpu_high_jobs.size()
            << " gpu_high_groups=" << gpu_high_nonempty_groups
            << " pcie_baseline_tib_per_residue=" << pcie_baseline_tib
            << " pcie_removed_tib_per_residue=" << pcie_removed_tib
            << " pcie_remaining_tib_per_residue=" << pcie_remaining_tib
            << " pcie_fraction_removed=" << pcie_fraction_removed
            << " pcie_remaining_50gib_s=" << pcie_remaining_50gib_s
            << '\n';
        return 0;
    }

    CpuHighCrossHost cpu_high_cross;
    if (!selected_cpu_high_jobs.empty())
        cpu_high_cross = build_cpu_high_cross(storage);

    int visible=0; ck(cudaGetDeviceCount(&visible),"cudaGetDeviceCount");
    if(visible<1)return 2;
    ck(cudaSetDevice(0),"cudaSetDevice");

    BidescMaskDeviceTables mask_tables; mask_tables.install(G_FACTOR);
    HighDescDeviceTables highdesc_tables; highdesc_tables.install(highdesc);

    hybrid_sparse_release_dense_host(storage);
    if (selected_cpu_high_jobs.empty() || cpu_high_direct_mode) {
        highdesc.main_desc.clear(); highdesc.main_desc.shrink_to_fit();
        highdesc.block_desc.clear(); highdesc.block_desc.shrink_to_fit();
    }
    lowdesc.main_desc.clear(); lowdesc.main_desc.shrink_to_fit();
    lowdesc.block_desc.clear(); lowdesc.block_desc.shrink_to_fit();
    orbit.rec.clear(); orbit.rec.shrink_to_fit();

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size,"mmap hybrid-sparse main");
    block_auth.alloc(layout.block_size,"mmap hybrid-sparse block");
    main_auth.ptr[init_rank]=1;

    Direct2DCtx gpu; gpu.init(mod);
    CpuLowSparsePersistentPool cpu_low(cpu_workers);
    CpuHighPool cpu_high_scratch(cpu_high_workers);
    CpuHighDirectPersistentPool cpu_high_direct(cpu_high_workers);
    int gpu_threads=256;

    auto run_gpu_high = [&] {
        for(const auto& job:high_jobs) {
            if (!job.work || cpu_high_selected[size_t(job.g)]) continue;
            process_group_bidesc_compact(
                gpu,main_auth,block_auth,storage,layout,W,high_wp,job.g,gpu_threads);
        }
    };
    auto run_cpu_high = [&] {
        if (selected_cpu_high_jobs.empty()) return;
        if (cpu_high_direct_mode) {
            cpu_high_direct.run(selected_cpu_high_jobs, main_auth, block_auth,
                                storage, layout, cpu_high_direct_meta,
                                cpu_high_cross, mod);
        } else {
            cpu_high_scratch.run(selected_cpu_high_jobs, main_auth, block_auth,
                                 storage, layout, highdesc, cpu_high_cross, mod);
        }
    };

    auto wall0=std::chrono::steady_clock::now();
    for(int row=0;row<W;++row){
        if (cpu_high_overlap && !selected_cpu_high_jobs.empty()
            && gpu_high_nonempty_groups) {
            if (cpu_high_direct_mode) {
                bool started = cpu_high_direct.start_run(
                    selected_cpu_high_jobs, main_auth, block_auth,
                    storage, layout, cpu_high_direct_meta, cpu_high_cross, mod);
                run_gpu_high();
                if (started) cpu_high_direct.wait_run();
            } else {
                std::thread cpu_thread(run_cpu_high);
                run_gpu_high();
                cpu_thread.join();
            }
        } else {
            run_gpu_high();
            run_cpu_high();
        }
        cpu_low.run(cpu_low_jobs,main_auth,block_auth,storage,layout,sparse,mod);
        uint64_t cpu_high_groups = cpu_high_direct_mode
            ? cpu_high_direct.groups() : cpu_high_scratch.groups();
        std::cerr<<"row "<<row+1<<'/'<<W
                 <<" gpu_groups="<<gpu.groups
                 <<" cpu_high_groups="<<cpu_high_groups
                 <<" cpu_low_groups="<<cpu_low.groups()<<'\n';
    }

    double wall_s=ram_seconds_since(wall0);
    Count answer=main_auth.ptr[answer_rank];
    uint64_t cpu_high_groups = cpu_high_direct_mode
        ? cpu_high_direct.groups() : cpu_high_scratch.groups();
    double cpu_high_wall_s = cpu_high_direct_mode
        ? cpu_high_direct.wall_s : cpu_high_scratch.wall_s;
    double cpu_high_kernel_sum_s = cpu_high_direct_mode
        ? cpu_high_direct.kernel_s() : cpu_high_scratch.kernel_s();
    double cpu_high_pack_sum_s = cpu_high_direct_mode ? 0.0 : cpu_high_scratch.pack_s();
    double cpu_high_unpack_sum_s = cpu_high_direct_mode ? 0.0 : cpu_high_scratch.unpack_s();
    double cpu_high_peak_scratch_mib = cpu_high_direct_mode ? 0.0
        : double(cpu_high_scratch.peak_scratch_bytes())/double(1<<20);

    std::cout
        << "backend=gridfp-ramstream32-factorized-hybrid-sparse-v5.15"
        << " n="<<n<<" residue="<<answer<<" modulus="<<mod
        << " gpu_high_desc_mib="<<highdesc_mib<<" gpu_mask_mib="<<mask_mib
        << " cpu_sparse_nn_orbit_mib="<<sparse_nn_orbit_mib
        << " cpu_sparse_nr_orbit_mib="<<sparse_nr_orbit_mib
        << " cpu_sparse_nl_orbit_mib="<<sparse_nl_orbit_mib
        << " cpu_sparse_orbit_mib="<<sparse_orbit_mib
        << " cpu_sparse_local_closure_mib="<<sparse_local_closure_mib
        << " cpu_sparse_cross_closure_mib="<<sparse_cross_closure_mib
        << " cpu_sparse_closure_mib="<<sparse_closure_mib
        << " cpu_sparse_cross_rank_mib="<<sparse_cross_mib
        << " cpu_high_cross_rank_mib="<<cpu_high_cross_mib_est
        << " cpu_high_direct_meta_mib="<<cpu_high_direct_meta_mib
        << " gpu_groups="<<gpu.groups
        << " cpu_high_groups="<<cpu_high_groups
        << " cpu_low_groups="<<cpu_low.groups()
        << " cpu_workers="<<cpu_workers
        << " cpu_high_workers="<<cpu_high_workers
        << " cpu_high_mode="<<cpu_high_mode
        << " cpu_high_overlap="<<int(cpu_high_overlap)
        << " cpu_high_policy="<<cpu_high_policy
        << " cpu_high_persistent_workers=1"
        << " cpu_high_async_overlap=1"
        << " cpu_low_persistent_workers=1"
        << " cpu_low_pointer_workspace=stack"
        << " cpu_high_affinity=" << (cpu_high_explicit_affinity ? "explicit" : "default")
        << " cpu_low_affinity=" << (cpu_low_explicit_affinity ? "explicit" : "default")
        << " cpu_high_worker_start_s="<<cpu_high_direct.worker_start_s
        << " cpu_low_worker_start_s="<<cpu_low.worker_start_s
        << " cpu_high_selection_hash="<<selection_hash
        << " cpu_high_max_mib="<<cpu_high_max_mib
        << " cpu_high_peak_worker_scratch_mib="<<cpu_high_peak_scratch_mib
        << " h2d_s="<<gpu.h2d_s<<" gpu_kernel_s="<<gpu.kernel_s<<" d2h_s="<<gpu.d2h_s
        << " cpu_high_pack_sum_s="<<cpu_high_pack_sum_s
        << " cpu_high_kernel_sum_s="<<cpu_high_kernel_sum_s
        << " cpu_high_unpack_sum_s="<<cpu_high_unpack_sum_s
        << " cpu_high_wall_s="<<cpu_high_wall_s
        << " cpu_low_kernel_sum_s="<<cpu_low.kernel_s()<<" cpu_low_wall_s="<<cpu_low.wall_s
        << " pcie_baseline_tib_per_residue="<<pcie_baseline_tib
        << " pcie_removed_tib_per_residue="<<pcie_removed_tib
        << " pcie_remaining_tib_per_residue="<<pcie_remaining_tib
        << " pcie_fraction_removed="<<pcie_fraction_removed
        << " wall_s="<<wall_s<<'\n';

    cpu_high_scratch.release();
    gpu.destroy(); highdesc_tables.release(); mask_tables.release();
    main_auth.release(); block_auth.release();
    return 0;
}