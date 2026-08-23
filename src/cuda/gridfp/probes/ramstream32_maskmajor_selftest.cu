#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_maskmajor.hpp"

static void mm_enum_rec(int pos, int h, MateID m, std::vector<MateID>& out) {
    if (pos < 0) { if (h == 0) out.push_back(m); return; }
    mm_enum_rec(pos - 1, h, mset(m, pos, N), out);
    if (h > 0) mm_enum_rec(pos - 1, h - 1, mset(m, pos, R), out);
    mm_enum_rec(pos - 1, h + 1, mset(m, pos, ::L), out);
}
static std::vector<MateID> mm_enum_states(int width) {
    std::vector<MateID> out; mm_enum_rec(width - 1, 1, 0, out); return out;
}
static inline Count mm_ref_add(Count a, Count b, Count mod) {
    return (a >= mod - b) ? a - (mod - b) : a + b;
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W == LOW_LUT_K + HIGH_LUT_K + 1);
    static_assert(W <= 12, "mask-major selftest intentionally uses a small width");

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout logical = build_storage_layout(storage);
    LowMaskMajorLayout mm = build_lowmask_major_layout(storage, logical);
    LowDescHost lowdesc = build_low_descriptors(storage, logical);
    LowOrbitHost orbit = build_cpu_low_orbit(storage, logical, lowdesc);
    CpuLowMaskSparseHost sparse = build_cpu_low_maskmajor_sparse(storage, logical, lowdesc, orbit);

    auto main_states = mm_enum_states(W);
    auto block_states = mm_enum_states(W - 1);
    if (main_states.size() != mm.main_size || block_states.size() != mm.block_size) return 2;

    std::unordered_map<MateID,size_t> mi, di;
    mi.reserve(main_states.size()*2); di.reserve(block_states.size()*2);
    for (size_t i=0;i<main_states.size();++i) mi.emplace(main_states[i],i);
    for (size_t i=0;i<block_states.size();++i) di.emplace(block_states[i],i);

    // Bijection check for the physical ranker.
    std::vector<uint8_t> seen_m(size_t(mm.main_size)), seen_d(size_t(mm.block_size));
    for (MateID m: main_states) {
        Code r = lowmask_major_rank_main_host(m, storage, logical, mm);
        if (r >= mm.main_size || seen_m[size_t(r)]++) { std::cerr << "main rank collision\n"; return 3; }
    }
    for (MateID m: block_states) {
        Code r = lowmask_major_rank_block_host(m, storage, logical, mm);
        if (r >= mm.block_size || seen_d[size_t(r)]++) { std::cerr << "block rank collision\n"; return 4; }
    }

    std::vector<Count> init_m(main_states.size()), init_d(block_states.size());
    std::mt19937_64 rng(1618);
    for (auto& v:init_m) v=Count(rng()%mod);
    for (auto& v:init_d) v=Count(rng()%mod);

    RamCounts main_auth, block_auth;
    main_auth.alloc(mm.main_size,"maskmajor selftest main");
    block_auth.alloc(mm.block_size,"maskmajor selftest block");
    for(size_t i=0;i<main_states.size();++i)
        main_auth.ptr[lowmask_major_rank_main_host(main_states[i],storage,logical,mm)] = init_m[i];
    for(size_t i=0;i<block_states.size();++i)
        block_auth.ptr[lowmask_major_rank_block_host(block_states[i],storage,logical,mm)] = init_d[i];

    std::vector<Count> rm=init_m, rd=init_d;
    for(int p=LOW_LUT_K;p>=1;--p){
        std::vector<Count> nm=rm, nd(rd.size(),0);
        for(size_t i=0;i<main_states.size();++i){
            Count c=rm[i]; auto z=oneesan::gridfp::include_horizontal(main_states[i],W,p);
            if(!z.valid) continue;
            if(z.blocked){auto it=di.find(z.mate); if(it==di.end()) return 5; nd[it->second]=mm_ref_add(nd[it->second],c,mod);}
            else{auto it=mi.find(z.mate); if(it==mi.end()) return 6; nm[it->second]=mm_ref_add(nm[it->second],c,mod);}
        }
        for(size_t i=0;i<block_states.size();++i){
            Count c=rd[i]; MateID z=oneesan::gridfp::blocked_exclude(block_states[i],p);
            auto it=mi.find(z); if(it==mi.end()) return 7;
            nm[it->second]=mm_ref_add(nm[it->second],c,mod);
        }
        rm.swap(nm); rd.swap(nd);
    }

    WindowPlan low_wp = make_direct2d_window(false);
    auto jobs = make_cpu_low_jobs(W, low_wp);
    CpuLowMaskMajorPool pool(2);
    pool.run(jobs,main_auth,block_auth,storage,logical,mm,sparse,mod);

    for(size_t i=0;i<main_states.size();++i){
        Count got=main_auth.ptr[lowmask_major_rank_main_host(main_states[i],storage,logical,mm)];
        if(got!=rm[i]){std::cerr<<"FAIL maskmajor main i="<<i<<" got="<<got<<" want="<<rm[i]<<'\n'; return 10;}
    }
    for(size_t i=0;i<block_states.size();++i){
        Count got=block_auth.ptr[lowmask_major_rank_block_host(block_states[i],storage,logical,mm)];
        if(got!=rd[i]){std::cerr<<"FAIL maskmajor block i="<<i<<" got="<<got<<" want="<<rd[i]<<'\n'; return 11;}
    }

    std::cout << "maskmajor-selftest OK W="<<W
              << " main="<<main_states.size()<<" block="<<block_states.size()
              << " groups="<<pool.groups()
              << " cpu_scratch_mib=0"
              << " sparse_meta_mib="
              << double(sparse.orbit_ops.size()*sizeof(CpuLowMaskOrbitOp)
                        +sparse.closure_ops.size()*sizeof(CpuLowMaskClosureOp))/(1<<20)
              << '\n';
    main_auth.release(); block_auth.release();
    return 0;
}
