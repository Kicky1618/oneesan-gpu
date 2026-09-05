#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_segment_major_builder_microprobe_main_unused
#include "gridfp_reduced_production_p2p_segment_major_builder_microprobe.cu"
#pragma pop_macro("main")

#include <set>

namespace {

static constexpr unsigned long long MAJOR_PAIR_SEG_INC = 1ULL << 32;
static constexpr unsigned long long MAJOR_PAIR_RUN_MASK = 0xffffffffULL;

struct MajorBuildLayout {
    std::vector<Rank64> net_header_base, net_source_base, net_run_base;
    std::vector<Rank64> net_group_seg_begin, net_group_run_begin;
    std::vector<Rank64> local_header_base, local_run_base;
    std::vector<Rank64> local_group_cycle_begin, local_group_run_begin;
    Rank64 total_net_headers = 0, total_net_segments = 0, total_net_runs = 0;
    Rank64 total_local_headers = 0, total_local_runs = 0;
    HostMajorPlan skeleton;
};

MajorBuildLayout make_major_build_layout(
    const MajorCountHost& count, int ngpu, int batches
) {
    MajorBuildLayout x;
    const int ngroups = ngpu * batches * MAJOR_PC_CLASSES;
    const int nlocal = ngpu * MAJOR_PC_CLASSES;
    x.net_header_base.resize(static_cast<std::size_t>(ngpu * batches));
    x.net_source_base.resize(static_cast<std::size_t>(ngpu * batches));
    x.net_run_base.resize(static_cast<std::size_t>(ngpu * batches));
    x.net_group_seg_begin.resize(static_cast<std::size_t>(ngroups));
    x.net_group_run_begin.resize(static_cast<std::size_t>(ngroups));
    x.local_header_base.resize(static_cast<std::size_t>(ngpu));
    x.local_run_base.resize(static_cast<std::size_t>(ngpu));
    x.local_group_cycle_begin.resize(static_cast<std::size_t>(nlocal));
    x.local_group_run_begin.resize(static_cast<std::size_t>(nlocal));
    x.skeleton.batch.resize(static_cast<std::size_t>(ngpu));
    for (auto& g : x.skeleton.batch) g.resize(static_cast<std::size_t>(batches));
    x.skeleton.local.resize(static_cast<std::size_t>(ngpu));
    x.skeleton.scratch_states.assign(static_cast<std::size_t>(ngpu), 0);

    for (int g = 0; g < ngpu; ++g) {
        x.local_header_base[static_cast<std::size_t>(g)] = x.total_local_headers;
        x.local_run_base[static_cast<std::size_t>(g)] = x.total_local_runs;
        Rank64 cp = 0, rp = 0;
        auto& hlo = x.skeleton.local[static_cast<std::size_t>(g)];
        for (int cls = 0; cls < MAJOR_PC_CLASSES; ++cls) {
            const int idx = g * MAJOR_PC_CLASSES + cls;
            const Rank64 nc = count.local_cycles[static_cast<std::size_t>(idx)];
            const Rank64 nr = count.local_runs[static_cast<std::size_t>(idx)];
            if (nc >= (Rank64(1) << 32) || nr >= (Rank64(1) << 32))
                fail("major fill local paired cursor range");
            x.local_group_cycle_begin[static_cast<std::size_t>(idx)] = cp;
            x.local_group_run_begin[static_cast<std::size_t>(idx)] = rp;
            auto& gm = hlo.group[static_cast<std::size_t>(cls)];
            gm.begin = static_cast<std::uint32_t>(cp);
            gm.end = static_cast<std::uint32_t>(cp + nc);
            gm.pc = catalan(cls + 1);
            cp += nc; rp += nr;
        }
        x.total_local_headers += cp + 1;
        x.total_local_runs += rp;
        hlo.run_begin.resize(static_cast<std::size_t>(cp + 1));
        hlo.local_low.resize(static_cast<std::size_t>(rp));
        hlo.local_high.resize(static_cast<std::size_t>(rp));
        x.skeleton.local_cycles += cp;
        x.skeleton.local_run_records += rp;

        for (int b = 0; b < batches; ++b) {
            const int gb = g * batches + b;
            x.net_header_base[static_cast<std::size_t>(gb)] = x.total_net_headers;
            x.net_source_base[static_cast<std::size_t>(gb)] = x.total_net_segments;
            x.net_run_base[static_cast<std::size_t>(gb)] = x.total_net_runs;
            Rank64 sp = 0, brp = 0, scratch = 0;
            auto& hb = x.skeleton.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            for (int cls = 0; cls < MAJOR_PC_CLASSES; ++cls) {
                const int idx = major_group_index(g, b, cls, batches);
                const Rank64 ns = count.network_segments[static_cast<std::size_t>(idx)];
                const Rank64 nr = count.network_runs[static_cast<std::size_t>(idx)];
                if (ns >= (Rank64(1) << 32) || nr >= (Rank64(1) << 32))
                    fail("major fill network paired cursor range");
                x.net_group_seg_begin[static_cast<std::size_t>(idx)] = sp;
                x.net_group_run_begin[static_cast<std::size_t>(idx)] = brp;
                auto& gm = hb.group[static_cast<std::size_t>(cls)];
                gm.begin = static_cast<std::uint32_t>(sp);
                gm.end = static_cast<std::uint32_t>(sp + ns);
                gm.scratch_base = scratch;
                gm.pc = catalan(cls + 1);
                sp += ns; brp += nr; scratch += ns * gm.pc;
            }
            x.total_net_headers += sp + 1;
            x.total_net_segments += sp;
            x.total_net_runs += brp;
            hb.run_begin.resize(static_cast<std::size_t>(sp + 1));
            hb.source_low.resize(static_cast<std::size_t>(sp));
            hb.source_high.resize(static_cast<std::size_t>(sp));
            hb.local_low.resize(static_cast<std::size_t>(brp));
            hb.local_high.resize(static_cast<std::size_t>(brp));
            hb.scratch_states = scratch;
            x.skeleton.scratch_states[static_cast<std::size_t>(g)] = std::max(
                x.skeleton.scratch_states[static_cast<std::size_t>(g)], scratch);
            x.skeleton.network_segments += sp;
            x.skeleton.local_run_records += brp;
        }
    }
    return x;
}

__device__ __forceinline__ void major_fill_rank_cycle(
    std::uint32_t root_support,
    bool blocked,
    int len,
    int W,
    int q,
    int K,
    bool reverse,
    int old_start,
    int ngpu,
    const Rank64* owner_begin,
    int (&owner)[RP_MAX_W],
    Rank64 (&local)[RP_MAX_W],
    int* error
) {
    std::uint32_t cur = root_support;
    for (int h = 0; h < len; ++h) {
        const DeviceKey key = equal_run_key0_device(cur, blocked, W, q, reverse);
        const GroupedDeviceRank gr = grouped_rank_device(
            key, W, q, reverse, old_start, K, ngpu, owner_begin);
        owner[h] = gr.owner;
        local[h] = gr.local;
        const int cheap = p2p_support_owner_device(cur, W, old_start, K, reverse, ngpu);
        if (gr.owner != cheap || gr.owner < 0 || gr.owner >= ngpu ||
            gr.local >= (Rank64(1) << 36)) {
            atomicCAS(error, 0, 461);
            return;
        }
        cur = shift_next_support_device(cur, blocked, W, q, K, K, reverse);
    }
}

__global__ void segment_major_fill_kernel(
    Rank64 base_supports,
    int W,
    int K,
    bool reverse,
    int ngpu,
    int batches,
    const Rank64* __restrict__ owner_begin,
    const Rank64* __restrict__ net_header_base,
    const Rank64* __restrict__ net_source_base,
    const Rank64* __restrict__ net_run_base,
    const Rank64* __restrict__ net_group_seg_begin,
    const Rank64* __restrict__ net_group_run_begin,
    const Rank64* __restrict__ local_header_base,
    const Rank64* __restrict__ local_run_base,
    const Rank64* __restrict__ local_group_cycle_begin,
    const Rank64* __restrict__ local_group_run_begin,
    unsigned long long* __restrict__ net_cursor,
    unsigned long long* __restrict__ local_cursor,
    std::uint32_t* __restrict__ net_run_begin,
    std::uint32_t* __restrict__ source_low,
    std::uint8_t* __restrict__ source_high,
    std::uint32_t* __restrict__ net_local_low,
    std::uint8_t* __restrict__ net_local_high,
    std::uint32_t* __restrict__ local_run_begin,
    std::uint32_t* __restrict__ local_low,
    std::uint8_t* __restrict__ local_high,
    int* error
) {
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);
    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed root = seeds[ri];
            const bool blocked = root.blocked != 0;
            const int len = shift_cycle_leader_length_device(
                root.support, blocked, W, q, K, K, reverse);
            if (len < 0 || len > RP_MAX_W) { atomicCAS(error, 0, 462); continue; }
            if (len <= 1) continue;
            const int occupied = __popc(root.support);
            if (!(occupied & 1)) { atomicCAS(error, 0, 463); continue; }
            const int cls = (occupied + 1) / 2 - 1;
            if (cls < 0 || cls >= MAJOR_PC_CLASSES) { atomicCAS(error, 0, 464); continue; }

            int owner[RP_MAX_W]{};
            Rank64 local_rank[RP_MAX_W]{};
            major_fill_rank_cycle(
                root.support, blocked, len, W, q, K, reverse, old_start, ngpu,
                owner_begin, owner, local_rank, error);
            if (*error) continue;

            int boundaries = 0;
            for (int h = 0; h < len; ++h)
                boundaries += owner[h] != owner[(h + len - 1) % len];
            if (!boundaries) {
                const int g = owner[0];
                const int idx = g * MAJOR_PC_CLASSES + cls;
                const unsigned long long add = MAJOR_PAIR_SEG_INC |
                    static_cast<unsigned long long>(len);
                const unsigned long long old = atomicAdd(local_cursor + idx, add);
                const Rank64 ci = old >> 32;
                const Rank64 rr = old & MAJOR_PAIR_RUN_MASK;
                if (rr + Rank64(len) > (Rank64(1) << 32)) {
                    atomicCAS(error, 0, 465); continue;
                }
                const Rank64 header_slot = local_header_base[g] +
                    local_group_cycle_begin[idx] + ci;
                local_run_begin[header_slot] = static_cast<std::uint32_t>(
                    local_group_run_begin[idx] + rr);
                const Rank64 run_slot = local_run_base[g] + local_group_run_begin[idx] + rr;
                for (int h = 0; h < len; ++h)
                    p2p_pack_run39_device(owner[h], local_rank[h],
                        local_low[run_slot + h], local_high[run_slot + h]);
                continue;
            }

            const int batch = int(major_batch_hash_device(root.support, blocked, reverse) %
                                  std::uint32_t(batches));
            for (int start = 0; start < len; ++start) {
                const int pred = (start + len - 1) % len;
                if (owner[start] == owner[pred]) continue;
                const int g = owner[start];
                int seg_len = 1;
                while (seg_len < len && owner[(start + seg_len) % len] == g) ++seg_len;
                if (seg_len >= len) { atomicCAS(error, 0, 466); break; }
                const int idx = major_group_index(g, batch, cls, batches);
                const unsigned long long add = MAJOR_PAIR_SEG_INC |
                    static_cast<unsigned long long>(seg_len);
                const unsigned long long old = atomicAdd(net_cursor + idx, add);
                const Rank64 si = old >> 32;
                const Rank64 rr = old & MAJOR_PAIR_RUN_MASK;
                if (rr + Rank64(seg_len) > (Rank64(1) << 32)) {
                    atomicCAS(error, 0, 467); break;
                }
                const int gb = g * batches + batch;
                const Rank64 seg_local = net_group_seg_begin[idx] + si;
                const Rank64 run_local_pos = net_group_run_begin[idx] + rr;
                const Rank64 header_slot = net_header_base[gb] + seg_local;
                const Rank64 source_slot = net_source_base[gb] + seg_local;
                const Rank64 run_slot = net_run_base[gb] + run_local_pos;
                net_run_begin[header_slot] = static_cast<std::uint32_t>(run_local_pos);
                p2p_pack_run39_device(owner[pred], local_rank[pred],
                    source_low[source_slot], source_high[source_slot]);
                for (int j = 0; j < seg_len; ++j) {
                    const int h = (start + j) % len;
                    p2p_pack_run39_device(owner[h], local_rank[h],
                        net_local_low[run_slot + j], net_local_high[run_slot + j]);
                }
            }
        }
    }
}

template<class T>
void fill_copy_vector(std::vector<T>& dst, const T* src, Rank64 n, const char* what) {
    dst.resize(static_cast<std::size_t>(n));
    if (n) ck(cudaMemcpy(dst.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost), what);
}

void verify_major_plan_multiset(const HostMajorPlan& want, const HostMajorPlan& got, int ngpu, int batches) {
    if (want.network_segments != got.network_segments || want.local_cycles != got.local_cycles ||
        want.local_run_records != got.local_run_records || want.scratch_states != got.scratch_states)
        fail("major fill plan aggregate mismatch");
    using Seq = std::vector<std::uint64_t>;
    for (int g = 0; g < ngpu; ++g) {
        for (int cls = 0; cls < MAJOR_PC_CLASSES; ++cls) {
            std::multiset<Seq> a, b;
            for (const auto* p : {&want.local[static_cast<std::size_t>(g)],
                                  &got.local[static_cast<std::size_t>(g)]}) {
                std::multiset<Seq>& out = (p == &want.local[static_cast<std::size_t>(g)]) ? a : b;
                const auto gm = p->group[static_cast<std::size_t>(cls)];
                for (std::uint32_t i = gm.begin; i < gm.end; ++i) {
                    Seq s;
                    for (std::uint32_t r = p->run_begin[i]; r < p->run_begin[i + 1]; ++r)
                        s.push_back(unpack_run_39(p->local_low[r], p->local_high[r]));
                    out.insert(std::move(s));
                }
            }
            if (a != b) fail("major fill local multiset mismatch");
        }
        for (int batch = 0; batch < batches; ++batch) {
            const auto& wa = want.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            const auto& ga = got.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            for (int cls = 0; cls < MAJOR_PC_CLASSES; ++cls) {
                std::multiset<Seq> a, b;
                for (int which = 0; which < 2; ++which) {
                    const auto& p = which ? ga : wa;
                    std::multiset<Seq>& out = which ? b : a;
                    const auto gm = p.group[static_cast<std::size_t>(cls)];
                    for (std::uint32_t i = gm.begin; i < gm.end; ++i) {
                        Seq s;
                        s.push_back(unpack_run_39(p.source_low[i], p.source_high[i]));
                        for (std::uint32_t r = p.run_begin[i]; r < p.run_begin[i + 1]; ++r)
                            s.push_back(unpack_run_39(p.local_low[r], p.local_high[r]));
                        out.insert(std::move(s));
                    }
                }
                if (a != b) fail("major fill network multiset mismatch");
            }
        }
    }
}

void run_segment_major_fill_probe(
    int W, int K, bool reverse, int ngpu, int batches, unsigned blocks
) {
    ProductionFactorTables tables(W);
    const HostTilePlan tile = make_host_tile_plan(tables, K, ngpu);
    const HostSegmentPlan hp = make_segment_plan(W, K, reverse, ngpu, batches, tables, tile);
    const HostMajorPlan want = make_major_plan(hp, ngpu, batches);
    const MajorCountHost count = expected_major_counts(want, ngpu, batches);
    const MajorBuildLayout layout = make_major_build_layout(count, ngpu, batches);

    ck(cudaSetDevice(0), "major fill set device");
    install_tables(tables);
    Rank64* d_owner_begin = nullptr;
    ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)), "major fill alloc owner begin");
    ck(cudaMemcpy(d_owner_begin, tile.owner_begin.data(), ngpu * sizeof(Rank64),
                  cudaMemcpyHostToDevice), "major fill copy owner begin");

    auto copy_u64 = [](Rank64*& dst, const std::vector<Rank64>& src, const char* what) {
        ck(cudaMalloc(&dst, std::max<std::size_t>(1, src.size()) * sizeof(Rank64)), what);
        if (!src.empty()) ck(cudaMemcpy(dst, src.data(), src.size() * sizeof(Rank64), cudaMemcpyHostToDevice), what);
    };
    Rank64 *d_nhb=nullptr,*d_nsb=nullptr,*d_nrb=nullptr,*d_ngs=nullptr,*d_ngr=nullptr;
    Rank64 *d_lhb=nullptr,*d_lrb=nullptr,*d_lgc=nullptr,*d_lgr=nullptr;
    copy_u64(d_nhb, layout.net_header_base, "major fill nhb");
    copy_u64(d_nsb, layout.net_source_base, "major fill nsb");
    copy_u64(d_nrb, layout.net_run_base, "major fill nrb");
    copy_u64(d_ngs, layout.net_group_seg_begin, "major fill ngs");
    copy_u64(d_ngr, layout.net_group_run_begin, "major fill ngr");
    copy_u64(d_lhb, layout.local_header_base, "major fill lhb");
    copy_u64(d_lrb, layout.local_run_base, "major fill lrb");
    copy_u64(d_lgc, layout.local_group_cycle_begin, "major fill lgc");
    copy_u64(d_lgr, layout.local_group_run_begin, "major fill lgr");

    const int ngroups = ngpu * batches * MAJOR_PC_CLASSES;
    const int nlocal = ngpu * MAJOR_PC_CLASSES;
    unsigned long long *d_nc=nullptr,*d_lc=nullptr;
    ck(cudaMalloc(&d_nc, ngroups * sizeof(unsigned long long)), "major fill net cursor");
    ck(cudaMalloc(&d_lc, nlocal * sizeof(unsigned long long)), "major fill local cursor");
    ck(cudaMemset(d_nc, 0, ngroups * sizeof(unsigned long long)), "major fill zero net cursor");
    ck(cudaMemset(d_lc, 0, nlocal * sizeof(unsigned long long)), "major fill zero local cursor");

    std::uint32_t *d_nh=nullptr,*d_sl=nullptr,*d_nll=nullptr,*d_lh=nullptr,*d_ll=nullptr;
    std::uint8_t *d_sh=nullptr,*d_nlh=nullptr,*d_lhigh=nullptr;
    ck(cudaMalloc(&d_nh, std::max<Rank64>(1,layout.total_net_headers)*4), "major fill net headers");
    ck(cudaMalloc(&d_sl, std::max<Rank64>(1,layout.total_net_segments)*4), "major fill source low");
    ck(cudaMalloc(&d_sh, std::max<Rank64>(1,layout.total_net_segments)), "major fill source high");
    ck(cudaMalloc(&d_nll, std::max<Rank64>(1,layout.total_net_runs)*4), "major fill net local low");
    ck(cudaMalloc(&d_nlh, std::max<Rank64>(1,layout.total_net_runs)), "major fill net local high");
    ck(cudaMalloc(&d_lh, std::max<Rank64>(1,layout.total_local_headers)*4), "major fill local headers");
    ck(cudaMalloc(&d_ll, std::max<Rank64>(1,layout.total_local_runs)*4), "major fill local low");
    ck(cudaMalloc(&d_lhigh, std::max<Rank64>(1,layout.total_local_runs)), "major fill local high");
    int* d_error=nullptr; ck(cudaMalloc(&d_error,sizeof(int)),"major fill error");
    ck(cudaMemset(d_error,0,sizeof(int)),"major fill zero error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const unsigned threads=256;
    const unsigned launch_blocks=static_cast<unsigned>(std::max<Rank64>(1,std::min<Rank64>(blocks,(base_supports+threads-1)/threads)));
    cudaEvent_t a{},b{}; ck(cudaEventCreate(&a),"major fill event a"); ck(cudaEventCreate(&b),"major fill event b");
    ck(cudaEventRecord(a),"major fill record a");
    segment_major_fill_kernel<<<launch_blocks,threads>>>(
        base_supports,W,K,reverse,ngpu,batches,d_owner_begin,
        d_nhb,d_nsb,d_nrb,d_ngs,d_ngr,d_lhb,d_lrb,d_lgc,d_lgr,d_nc,d_lc,
        d_nh,d_sl,d_sh,d_nll,d_nlh,d_lh,d_ll,d_lhigh,d_error);
    ck(cudaGetLastError(),"major fill launch"); ck(cudaEventRecord(b),"major fill record b"); ck(cudaEventSynchronize(b),"major fill sync");
    float ms=0; ck(cudaEventElapsedTime(&ms,a,b),"major fill elapsed");
    int error=0; ck(cudaMemcpy(&error,d_error,sizeof(error),cudaMemcpyDeviceToHost),"major fill copy error");
    if(error) fail("segment-major fill device error="+std::to_string(error));

    // Final sentinels are the only headers not written by paired cursors.
    for(int g=0;g<ngpu;++g){
        const auto& lo=layout.skeleton.local[static_cast<std::size_t>(g)];
        const std::uint32_t end=static_cast<std::uint32_t>(lo.local_low.size());
        const Rank64 slot=layout.local_header_base[static_cast<std::size_t>(g)]+lo.run_begin.size()-1;
        ck(cudaMemcpy(d_lh+slot,&end,sizeof(end),cudaMemcpyHostToDevice),"major fill local sentinel");
        for(int batch=0;batch<batches;++batch){
            const int gb=g*batches+batch;
            const auto& hb=layout.skeleton.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            const std::uint32_t bend=static_cast<std::uint32_t>(hb.local_low.size());
            const Rank64 hslot=layout.net_header_base[static_cast<std::size_t>(gb)]+hb.run_begin.size()-1;
            ck(cudaMemcpy(d_nh+hslot,&bend,sizeof(bend),cudaMemcpyHostToDevice),"major fill net sentinel");
        }
    }

    HostMajorPlan got=layout.skeleton;
    for(int g=0;g<ngpu;++g){
        auto& lo=got.local[static_cast<std::size_t>(g)];
        const Rank64 hb=layout.local_header_base[static_cast<std::size_t>(g)], rb=layout.local_run_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(lo.run_begin.data(),d_lh+hb,lo.run_begin.size()*4,cudaMemcpyDeviceToHost),"major fill read local headers");
        if(!lo.local_low.empty()){
            ck(cudaMemcpy(lo.local_low.data(),d_ll+rb,lo.local_low.size()*4,cudaMemcpyDeviceToHost),"major fill read local low");
            ck(cudaMemcpy(lo.local_high.data(),d_lhigh+rb,lo.local_high.size(),cudaMemcpyDeviceToHost),"major fill read local high");
        }
        for(int batch=0;batch<batches;++batch){
            const int gb=g*batches+batch;
            auto& hbv=got.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            const Rank64 hh=layout.net_header_base[static_cast<std::size_t>(gb)], sb=layout.net_source_base[static_cast<std::size_t>(gb)], nr=layout.net_run_base[static_cast<std::size_t>(gb)];
            ck(cudaMemcpy(hbv.run_begin.data(),d_nh+hh,hbv.run_begin.size()*4,cudaMemcpyDeviceToHost),"major fill read net headers");
            if(!hbv.source_low.empty()){
                ck(cudaMemcpy(hbv.source_low.data(),d_sl+sb,hbv.source_low.size()*4,cudaMemcpyDeviceToHost),"major fill read source low");
                ck(cudaMemcpy(hbv.source_high.data(),d_sh+sb,hbv.source_high.size(),cudaMemcpyDeviceToHost),"major fill read source high");
            }
            if(!hbv.local_low.empty()){
                ck(cudaMemcpy(hbv.local_low.data(),d_nll+nr,hbv.local_low.size()*4,cudaMemcpyDeviceToHost),"major fill read net low");
                ck(cudaMemcpy(hbv.local_high.data(),d_nlh+nr,hbv.local_high.size(),cudaMemcpyDeviceToHost),"major fill read net high");
            }
        }
    }
    verify_major_plan_multiset(want,got,ngpu,batches);
    std::cout<<"gridfp-reduced-production-p2p-segment-major-fill"
             <<" W="<<W<<" K="<<K<<" direction="<<(reverse?"reverse":"forward")
             <<" ngpu="<<ngpu<<" batches="<<batches
             <<" network_segments="<<got.network_segments
             <<" local_cycles="<<got.local_cycles
             <<" fill_ms="<<ms
             <<" paired_u64_cursor=1 independent_segment_run_atomics=0 semantic_multiset_exact=1\n";

    cudaFree(d_error);cudaFree(d_lhigh);cudaFree(d_ll);cudaFree(d_lh);cudaFree(d_nlh);cudaFree(d_nll);cudaFree(d_sh);cudaFree(d_sl);cudaFree(d_nh);
    cudaFree(d_lc);cudaFree(d_nc);cudaFree(d_lgr);cudaFree(d_lgc);cudaFree(d_lrb);cudaFree(d_lhb);cudaFree(d_ngr);cudaFree(d_ngs);cudaFree(d_nrb);cudaFree(d_nsb);cudaFree(d_nhb);cudaFree(d_owner_begin);
    cudaEventDestroy(a);cudaEventDestroy(b);
}

} // namespace

int main(int argc,char** argv){
    const int W=argc>1?std::atoi(argv[1]):10;
    const int K=argc>2?std::atoi(argv[2]):(W-2)/2;
    const int batches=argc>3?std::atoi(argv[3]):4;
    const unsigned blocks=argc>4?static_cast<unsigned>(std::strtoul(argv[4],nullptr,10)):256u;
    const int ngpu=argc>5?std::atoi(argv[5]):2;
    if(W<8||W>12||(W&1)||K!=(W-2)/2||batches<1||batches>64||!blocks||ngpu<2||ngpu>P2P_MAX_GPU)return 2;
    int visible=0;ck(cudaGetDeviceCount(&visible),"segment-major fill device count");if(visible<1)return 3;
    run_segment_major_fill_probe(W,K,false,ngpu,batches,blocks);
    run_segment_major_fill_probe(W,K,true,ngpu,batches,blocks);
    std::cout<<"ALL_OK production_p2p_segment_major_fill_builder=1\n";
    return 0;
}
