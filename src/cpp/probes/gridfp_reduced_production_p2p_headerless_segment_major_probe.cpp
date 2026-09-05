#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_segment_major_probe_main_unused2
#include "gridfp_reduced_production_p2p_segment_major_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <map>

namespace {

static constexpr int HL_CLASSES = 14;
static constexpr int HL_MAX_LEN = 28;

struct HeaderlessClassMeta {
    Rank begin = 0;
    Rank end = 0;
    Rank scratch_base = 0;
    Rank pc = 0;
    std::array<Rank,HL_MAX_LEN+2> length_begin{}; // [1..29], 29=end
    std::array<Rank,HL_MAX_LEN+1> run_base{};     // [1..28]
};

struct HeaderlessGroup {
    PackedRunSoA source; // empty for local-only schedules
    PackedRunSoA local;
    std::array<HeaderlessClassMeta,HL_CLASSES> cls{};
    Rank items = 0;
};

struct HeaderlessGpu {
    HeaderlessGroup local;
    std::vector<HeaderlessGroup> network;
    Rank scratch_states = 0;
};

struct HeaderlessSchedule {
    std::vector<HeaderlessGpu> gpu;
    Rank network_segments = 0;
    Rank local_cycles = 0;
    Rank local_run_records = 0;
};

struct HlSeq {
    std::uint64_t source = 0;
    std::vector<std::uint64_t> local;
};

void hl_push(PackedRunSoA& dst, std::uint64_t z) {
    std::uint32_t low = 0;
    std::uint8_t high = 0;
    pack_run_39(z, low, high);
    dst.low.push_back(low);
    dst.high.push_back(high);
}

std::uint64_t hl_get(const PackedRunSoA& x, Rank i) {
    return unpack_run_39(
        x.low.at(static_cast<std::size_t>(i)),
        x.high.at(static_cast<std::size_t>(i)));
}

HeaderlessGroup make_headerless_group(
    const SegmentMajorNetworkGroup& src,
    int cls_only,
    bool network
) {
    HeaderlessGroup out;
    Rank item_cursor = 0, run_cursor = 0;
    for (int cls = 0; cls < HL_CLASSES; ++cls) {
        HeaderlessClassMeta& cm = out.cls[static_cast<std::size_t>(cls)];
        cm.begin = item_cursor;
        cm.scratch_base = 0;
        cm.pc = catalan(cls + 1);
        for (int len = 1; len <= HL_MAX_LEN; ++len) {
            cm.length_begin[static_cast<std::size_t>(len)] = item_cursor;
            cm.run_base[static_cast<std::size_t>(len)] = run_cursor;
            if (cls == cls_only) {
                for (std::size_t i = 0; i + 1 < src.run_begin.size(); ++i) {
                    const std::uint32_t a = src.run_begin[i], b = src.run_begin[i + 1];
                    if (int(b - a) != len) continue;
                    if (network) hl_push(out.source, unpack_run_39(src.source.low[i], src.source.high[i]));
                    for (std::uint32_t r = a; r < b; ++r)
                        hl_push(out.local, unpack_run_39(src.local.low[r], src.local.high[r]));
                    ++item_cursor;
                    run_cursor += len;
                }
            }
        }
        cm.length_begin[HL_MAX_LEN + 1] = item_cursor;
        cm.end = item_cursor;
    }
    out.items = item_cursor;
    return out;
}

HeaderlessGroup make_headerless_local(
    const SegmentMajorLocalGroup& src,
    int cls_only
) {
    SegmentMajorNetworkGroup shim;
    shim.run_begin = src.run_begin;
    shim.local = src.local;
    return make_headerless_group(shim, cls_only, false);
}

HeaderlessSchedule compile_headerless(
    const SegmentMajorSchedule& major,
    int ngpu,
    int batches
) {
    HeaderlessSchedule out;
    out.gpu.resize(static_cast<std::size_t>(ngpu));
    for (int g = 0; g < ngpu; ++g) {
        auto& hg = out.gpu[static_cast<std::size_t>(g)];
        hg.network.resize(static_cast<std::size_t>(batches));

        // Merge the 14 already-classed local groups into one headerless stream.
        Rank item_cursor = 0, run_cursor = 0;
        for (int cls = 0; cls < HL_CLASSES; ++cls) {
            const auto& src = major.gpu[static_cast<std::size_t>(g)].local[static_cast<std::size_t>(cls)];
            auto& cm = hg.local.cls[static_cast<std::size_t>(cls)];
            cm.begin = item_cursor;
            cm.pc = catalan(cls + 1);
            for (int len = 1; len <= HL_MAX_LEN; ++len) {
                cm.length_begin[static_cast<std::size_t>(len)] = item_cursor;
                cm.run_base[static_cast<std::size_t>(len)] = run_cursor;
                for (std::size_t i = 0; i + 1 < src.run_begin.size(); ++i) {
                    const auto a = src.run_begin[i], b = src.run_begin[i + 1];
                    if (int(b - a) != len) continue;
                    for (std::uint32_t r = a; r < b; ++r)
                        hl_push(hg.local.local, unpack_run_39(src.local.low[r], src.local.high[r]));
                    ++item_cursor; run_cursor += len;
                    ++out.local_cycles;
                    out.local_run_records += len;
                }
            }
            cm.length_begin[HL_MAX_LEN + 1] = item_cursor;
            cm.end = item_cursor;
        }
        hg.local.items = item_cursor;

        for (int b = 0; b < batches; ++b) {
            HeaderlessGroup& dst = hg.network[static_cast<std::size_t>(b)];
            item_cursor = 0; run_cursor = 0; Rank scratch = 0;
            for (int cls = 0; cls < HL_CLASSES; ++cls) {
                const auto& src = major.gpu[static_cast<std::size_t>(g)]
                                      .network[static_cast<std::size_t>(b)]
                                              [static_cast<std::size_t>(cls)];
                auto& cm = dst.cls[static_cast<std::size_t>(cls)];
                cm.begin = item_cursor;
                cm.scratch_base = scratch;
                cm.pc = catalan(cls + 1);
                for (int len = 1; len <= HL_MAX_LEN; ++len) {
                    cm.length_begin[static_cast<std::size_t>(len)] = item_cursor;
                    cm.run_base[static_cast<std::size_t>(len)] = run_cursor;
                    for (std::size_t i = 0; i + 1 < src.run_begin.size(); ++i) {
                        const auto a = src.run_begin[i], e = src.run_begin[i + 1];
                        if (int(e - a) != len) continue;
                        hl_push(dst.source, unpack_run_39(src.source.low[i], src.source.high[i]));
                        for (std::uint32_t r = a; r < e; ++r)
                            hl_push(dst.local, unpack_run_39(src.local.low[r], src.local.high[r]));
                        ++item_cursor; run_cursor += len;
                        ++out.network_segments;
                        out.local_run_records += len;
                    }
                }
                cm.length_begin[HL_MAX_LEN + 1] = item_cursor;
                cm.end = item_cursor;
                scratch += (cm.end - cm.begin) * cm.pc;
            }
            dst.items = item_cursor;
            if (dst.source.low.size() != item_cursor)
                fail("headerless source/item shape");
            hg.scratch_states = std::max(hg.scratch_states, scratch);
        }
    }
    if (out.network_segments != major.network_segments ||
        out.local_cycles != major.local_cycles ||
        out.local_run_records != major.local_run_records)
        fail("headerless aggregate coverage");
    return out;
}

std::pair<int,int> hl_class_len(const HeaderlessGroup& g, Rank item) {
    for (int cls = 0; cls < HL_CLASSES; ++cls) {
        const auto& cm = g.cls[static_cast<std::size_t>(cls)];
        if (item >= cm.end) continue;
        for (int len = 1; len <= HL_MAX_LEN; ++len)
            if (item < cm.length_begin[static_cast<std::size_t>(len + 1)])
                return {cls,len};
        fail("headerless length lookup");
    }
    fail("headerless class lookup");
}

Rank hl_run_offset(const HeaderlessGroup& g, Rank item, int cls, int len) {
    const auto& cm = g.cls[static_cast<std::size_t>(cls)];
    return cm.run_base[static_cast<std::size_t>(len)] +
           (item - cm.length_begin[static_cast<std::size_t>(len)]) * Rank(len);
}

void verify_headerless_execution(
    const CompactSegmentSchedule& compact,
    const HeaderlessSchedule& hl,
    int ngpu,
    int batches
) {
    std::map<std::uint64_t,Rank> initial, got, want;
    Rank serial = 1;
    for (Rank r = 0; r < compact.run.low.size(); ++r) {
        const auto k = packed_key(compact.run.low[r], compact.run.high[r]);
        if (!initial.emplace(k,serial++).second) fail("headerless duplicate run key");
    }
    got = initial;
    for (const auto& h : compact.header) {
        const int len = compiled_header_len(h);
        for (int i = 0; i < len; ++i) {
            const Rank d = Rank(h.run_begin) + i;
            const Rank s = Rank(h.run_begin) + (i + len - 1) % len;
            want[packed_key(compact.run.low[d],compact.run.high[d])] =
                initial.at(packed_key(compact.run.low[s],compact.run.high[s]));
        }
    }

    for (int gpu = 0; gpu < ngpu; ++gpu) {
        const auto& g = hl.gpu[static_cast<std::size_t>(gpu)].local;
        for (Rank item = 0; item < g.items; ++item) {
            const auto [cls,len] = hl_class_len(g,item);
            const Rank ro = hl_run_offset(g,item,cls,len);
            std::vector<std::uint64_t> key;
            for (int j=0;j<len;++j) key.push_back(hl_get(g.local,ro+j));
            Rank temp=got.at(key.back());
            for(int j=len-1;j>0;--j) got[key[j]]=got.at(key[j-1]);
            got[key[0]]=temp;
        }
    }

    for (int b=0;b<batches;++b) {
        std::vector<std::vector<Rank>> scratch(static_cast<std::size_t>(ngpu));
        for(int gpu=0;gpu<ngpu;++gpu)
            scratch[gpu].resize(static_cast<std::size_t>(hl.gpu[gpu].scratch_states));
        for(int gpu=0;gpu<ngpu;++gpu){
            const auto& g=hl.gpu[gpu].network[b];
            for(Rank item=0;item<g.items;++item){
                const auto [cls,len]=hl_class_len(g,item);(void)len;
                const auto& cm=g.cls[cls];
                const Rank off=cm.scratch_base+(item-cm.begin)*cm.pc;
                scratch[gpu][off]=got.at(hl_get(g.source,item));
            }
        }
        for(int gpu=0;gpu<ngpu;++gpu){
            const auto& g=hl.gpu[gpu].network[b];
            for(Rank item=0;item<g.items;++item){
                const auto [cls,len]=hl_class_len(g,item);
                const auto& cm=g.cls[cls];
                const Rank ro=hl_run_offset(g,item,cls,len);
                std::vector<std::uint64_t> key;
                for(int j=0;j<len;++j)key.push_back(hl_get(g.local,ro+j));
                for(int j=len-1;j>0;--j)got[key[j]]=got.at(key[j-1]);
                const Rank off=cm.scratch_base+(item-cm.begin)*cm.pc;
                got[key[0]]=scratch[gpu][off];
            }
        }
    }
    if(got!=want)fail("headerless segment-major execution mismatch");
}

void verify_headerless(int W,bool reverse,int ngpu,int batches){
    const int K=(W-2)/2;
    const auto compact=compile_compact_segments(W,K,reverse,ngpu,batches);
    const auto major=compile_segment_major(compact,ngpu,batches);
    const auto hl=compile_headerless(major,ngpu,batches);
    verify_headerless_execution(compact,hl,ngpu,batches);
    Rank run_bytes=0,table_bytes=0,max_scratch=0;
    for(const auto& gpu:hl.gpu){
        max_scratch=std::max(max_scratch,gpu.scratch_states);
        run_bytes+=gpu.local.local.low.size()*5ULL;
        table_bytes+=sizeof(HeaderlessClassMeta)*HL_CLASSES;
        for(const auto& g:gpu.network){run_bytes+=g.source.low.size()*5ULL+g.local.low.size()*5ULL;table_bytes+=sizeof(HeaderlessClassMeta)*HL_CLASSES;}
    }
    std::cout<<"W="<<W<<" K="<<K<<" direction="<<(reverse?"reverse":"forward")
             <<" ngpu="<<ngpu<<" batches="<<batches
             <<" network_segments="<<hl.network_segments
             <<" local_cycles="<<hl.local_cycles
             <<" run_metadata_bytes="<<run_bytes
             <<" table_bytes="<<table_bytes
             <<" per_item_run_begin_bytes=0"
             <<" max_scratch_states="<<max_scratch
             <<" scalar_execution_exact=1\n";
}

void print_w28_headerless_theory(){
    constexpr Rank nonfixed=167763968ULL;
    constexpr Rank segments=117118478ULL;
    constexpr Rank bytes=5ULL*nonfixed+5ULL*segments;
    static_assert(bytes==1424412230ULL);
    std::cout<<"W=28 K=13 nonfixed_runs="<<nonfixed
             <<" network_segments="<<segments
             <<" headerless_dominant_GiB="<<double(bytes)/double(1ULL<<30)
             <<" headerless_avg_MiB_per_gpu="<<double(bytes)/8.0/double(1ULL<<20)
             <<" both_directions_avg_MiB_per_gpu="<<double(bytes)*2.0/8.0/double(1ULL<<20)
             <<" formula=5*nonfixed_runs+5*network_segments\n";
}

} // namespace

int main(int argc,char** argv){
    const int maxW=argc>1?std::atoi(argv[1]):12;
    const int ngpu=argc>2?std::atoi(argv[2]):8;
    const int batches=argc>3?std::atoi(argv[3]):8;
    if(maxW<8||maxW>14||ngpu<2||ngpu>8||batches<1||batches>64)return 2;
    for(int W=8;W<=maxW;W+=2)for(bool reverse:{false,true})verify_headerless(W,reverse,ngpu,batches);
    print_w28_headerless_theory();
    std::cout<<"ALL_OK production_p2p_headerless_segment_major=1\n";
    return 0;
}
