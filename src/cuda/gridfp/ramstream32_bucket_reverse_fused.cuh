#pragma once

#include "ramstream32_bucket_reverse_atomic.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Destination-oriented reverse/snake closure.  Reverse LOW uses the same
// inactive-HIGH L->R map as forward LOW; reverse HIGH uses the same inactive-
// LOW R->L map as forward HIGH.  Therefore the already proven factorized
// preimage walkers in ramstream32_bucket_fused.cuh can be reused directly.
// Orbit metadata remains the conflict-free ReverseBucketAtomicHost orbit
// stream; only the source-oriented closure streams are replaced here.

struct ReverseBucketFusedHost {
    std::vector<BucketFusedDst> low_dst, high_dst;
    std::vector<uint32_t> low_off, high_off;
    std::vector<uint32_t> low_local_src, low_cross_op;
    std::vector<uint32_t> high_local_src, high_cross_op;
    uint32_t low_pitch = GPU_DIRECT_MAX_BLOCK_BLOCKS + 1;
    uint32_t high_pitch = GPU_DIRECT_MAX_MAIN_BLOCKS + 1;

    size_t bytes() const {
        return (low_dst.size() + high_dst.size()) * sizeof(BucketFusedDst)
            + (low_off.size() + high_off.size()
               + low_local_src.size() + low_cross_op.size()
               + high_local_src.size() + high_cross_op.size()) * sizeof(uint32_t);
    }
};

struct ReverseBucketFusedEdge {
    uint32_t dst_locator = 0;
    uint32_t source = 0;
    uint32_t depth = 0;
};

static ReverseBucketFusedHost build_reverse_bucket_fused(
    const StorageLayout& layout, const ReverseBucketAtomicHost& src
) {
    ReverseBucketFusedHost out;
    out.low_pitch = uint32_t(layout.block_blocks.size()) + 1;
    out.high_pitch = uint32_t(layout.main_blocks.size()) + 1;
    out.low_off.resize(size_t(LOW_LUT_K) * out.low_pitch);
    out.high_off.resize(size_t(HIGH_LUT_K) * out.high_pitch);

    auto emit_side = [&](bool low) {
        int p0 = low ? 1 : LOW_LUT_K + 1;
        int p1 = low ? LOW_LUT_K : TARGET_W - 1;
        const auto& closure = low ? src.low_closure : src.high_closure;
        const auto& coff = low ? src.low_closure_off : src.high_closure_off;
        auto& dst = low ? out.low_dst : out.high_dst;
        auto& off = low ? out.low_off : out.high_off;
        auto& local = low ? out.low_local_src : out.high_local_src;
        auto& cross = low ? out.low_cross_op : out.high_cross_op;
        uint32_t pitch = low ? out.low_pitch : out.high_pitch;
        size_t spitch = size_t(src.nblocks) + 1;

        for (int p = p0; p <= p1; ++p) {
            uint32_t pi = uint32_t(p - p0);
            bool target_main = !low && p == TARGET_W - 1;
            uint32_t nt = target_main
                ? uint32_t(layout.main_blocks.size())
                : uint32_t(layout.block_blocks.size());
            std::vector<std::vector<ReverseBucketFusedEdge>> by_dst(nt);

            for (uint32_t sbid = 0; sbid < src.nblocks; ++sbid) {
                size_t oi = size_t(pi) * spitch + sbid;
                uint32_t a = coff[oi], b = coff[oi + 1];
                for (uint32_t q = a; q < b; ++q) {
                    uint64_t w = closure[q];
                    uint32_t dbid = rb_closure_block(w);
                    bool tb = rb_closure_target_block(w);
                    if (tb == target_main) {
                        std::cerr << "reverse fused target-kind mismatch p=" << p
                                  << " source_bid=" << sbid << " dbid=" << dbid
                                  << " target_main=" << target_main << " tb=" << tb << '\n';
                        std::exit(310);
                    }
                    if (dbid >= nt) {
                        std::cerr << "reverse fused destination block overflow\n";
                        std::exit(311);
                    }
                    uint32_t sl = rb_closure_src(w);
                    uint32_t depth = rb_closure_depth(w);
                    uint32_t packed = depth
                        ? bkf_cross_pack(sbid, sl, depth)
                        : bkf_src_pack(sbid, sl);
                    by_dst[dbid].push_back({rb_closure_dst(w), packed, depth});
                }
            }

            for (uint32_t dbid = 0; dbid < nt; ++dbid) {
                off[size_t(pi) * pitch + dbid] = uint32_t(dst.size());
                auto& es = by_dst[dbid];
                std::sort(es.begin(), es.end(), [](const auto& a, const auto& b) {
                    if (a.dst_locator != b.dst_locator) return a.dst_locator < b.dst_locator;
                    if ((a.depth != 0) != (b.depth != 0)) return a.depth == 0;
                    return a.source < b.source;
                });
                size_t i = 0;
                while (i < es.size()) {
                    size_t j = i + 1;
                    while (j < es.size() && es[j].dst_locator == es[i].dst_locator) ++j;
                    uint32_t lb = uint32_t(local.size()), cb = uint32_t(cross.size());
                    uint32_t lc = 0, cc = 0;
                    for (size_t k = i; k < j; ++k) {
                        if (es[k].depth) { cross.push_back(es[k].source); ++cc; }
                        else { local.push_back(es[k].source); ++lc; }
                    }
                    if (lc > 0xffffu || cc > 0xffffu) {
                        std::cerr << "reverse fused destination indegree overflow local="
                                  << lc << " cross=" << cc << '\n';
                        std::exit(312);
                    }
                    dst.push_back({es[i].dst_locator, lb, cb, lc | (cc << 16)});
                    i = j;
                }
            }
            off[size_t(pi) * pitch + nt] = uint32_t(dst.size());
            for (uint32_t dbid = nt + 1; dbid < pitch; ++dbid)
                off[size_t(pi) * pitch + dbid] = uint32_t(dst.size());
        }
    };

    emit_side(true);
    emit_side(false);
    std::cerr << "reverse_bucket_fused"
              << " low_dst=" << out.low_dst.size()
              << " low_local=" << out.low_local_src.size()
              << " low_cross=" << out.low_cross_op.size()
              << " high_dst=" << out.high_dst.size()
              << " high_local=" << out.high_local_src.size()
              << " high_cross=" << out.high_cross_op.size()
              << " mib=" << double(out.bytes()) / double(1 << 20) << '\n';
    return out;
}

static size_t reverse_bucket_orbit_bytes(const ReverseBucketAtomicHost& h) {
    return (h.low_orbit.size() + h.high_orbit.size()) * sizeof(ReverseBucketOrbitOp)
        + (h.low_orbit_off.size() + h.high_orbit_off.size()) * sizeof(uint32_t);
}

__constant__ BucketFusedDst* D_RBF_LOW_DST;
__constant__ BucketFusedDst* D_RBF_HIGH_DST;
__constant__ uint32_t* D_RBF_LOW_OFF;
__constant__ uint32_t* D_RBF_HIGH_OFF;
__constant__ uint32_t* D_RBF_LOW_LOCAL_SRC;
__constant__ uint32_t* D_RBF_LOW_CROSS_OP;
__constant__ uint32_t* D_RBF_HIGH_LOCAL_SRC;
__constant__ uint32_t* D_RBF_HIGH_CROSS_OP;
__constant__ uint32_t D_RBF_LOW_PITCH;
__constant__ uint32_t D_RBF_HIGH_PITCH;

__global__ void bucket_reverse_low_fused_closure_kernel(int p) {
    uint32_t dbid = blockIdx.z;
    if (dbid >= D_BKF_BLOCK_NBLOCKS) return;
    uint32_t pi = uint32_t(p - 1);
    size_t oi = size_t(pi) * D_RBF_LOW_PITCH + dbid;
    uint32_t a = D_RBF_LOW_OFF[oi], b = D_RBF_LOW_OFF[oi + 1];
    for (uint32_t q = a + uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
         q < b; q += uint32_t(gridDim.x) * blockDim.x) {
        BucketFusedDst rec = D_RBF_LOW_DST[q];
        uint32_t dslot = bkf_loc_owner(rec.dst_locator), dr = bkf_loc_rank(rec.dst_locator);
        BucketPhysicalBlock db = bkf_low_block(dslot, dbid);
        if (!db.valid || !db.rows || !db.cols) continue;
        uint32_t lc = rec.counts & 0xffffu, cc = rec.counts >> 16;
        for (uint32_t hr = blockIdx.y; hr < db.rows; hr += gridDim.y) {
            Count* dp = bkf_ptr(dslot, db.off + Code(hr) * db.cols + dr);
            Count sum = *dp;
            for (uint32_t e = rec.local_begin; e < rec.local_begin + lc; ++e) {
                uint32_t x = D_RBF_LOW_LOCAL_SRC[e], sl = bkf_src_locator(x);
                uint32_t ss = bkf_loc_owner(sl), sbid = bkf_src_block(x);
                BucketPhysicalBlock sb = bkf_low_main(ss, sbid);
                sum = gpu_direct_add(sum,
                    bkf_ptr(ss, sb.off + Code(hr) * sb.cols + bkf_loc_rank(sl))[0]);
            }
            if (cc) {
                uint32_t dest_code = D_BKF_HIGH_CODES[
                    D_BKF_HIGH_CODE_OFF[size_t(D_BKF_FIXED_OWNER) * D_BKF_CODE_PITCH + db.he] + hr];
                for (uint32_t e = rec.cross_begin; e < rec.cross_begin + cc; ++e) {
                    uint32_t x = D_RBF_LOW_CROSS_OP[e], sl = bkf_src_locator(x);
                    uint32_t sbid = bkf_src_block(x);
                    BucketPhysicalBlock sb = bkf_low_main(bkf_loc_owner(sl), sbid);
                    sum = gpu_direct_add(sum,
                        bkf_sum_high_preimages(dest_code, bkf_cross_depth(x), sb.he, sbid, sl));
                }
            }
            *dp = sum;
        }
    }
}

__global__ void bucket_reverse_high_fused_closure_kernel(int p) {
    bool target_main = p == TARGET_W - 1;
    uint32_t dbid = blockIdx.z;
    uint32_t nt = target_main ? D_BKF_MAIN_NBLOCKS : D_BKF_BLOCK_NBLOCKS;
    if (dbid >= nt) return;
    uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    size_t oi = size_t(pi) * D_RBF_HIGH_PITCH + dbid;
    uint32_t a = D_RBF_HIGH_OFF[oi], b = D_RBF_HIGH_OFF[oi + 1];
    for (uint32_t q = a + blockIdx.y; q < b; q += gridDim.y) {
        BucketFusedDst rec = D_RBF_HIGH_DST[q];
        uint32_t dslot = bkf_loc_owner(rec.dst_locator), dr = bkf_loc_rank(rec.dst_locator);
        BucketPhysicalBlock db = target_main ? bkf_high_main(dslot, dbid) : bkf_high_block(dslot, dbid);
        if (!db.valid || !db.rows || !db.cols) continue;
        uint32_t lc = rec.counts & 0xffffu, cc = rec.counts >> 16;
        for (uint32_t lr = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
             lr < db.cols; lr += uint32_t(gridDim.x) * blockDim.x) {
            Count* dp = bkf_ptr(dslot, db.off + Code(dr) * db.cols + lr);
            Count sum = *dp;
            for (uint32_t e = rec.local_begin; e < rec.local_begin + lc; ++e) {
                uint32_t x = D_RBF_HIGH_LOCAL_SRC[e], sl = bkf_src_locator(x);
                uint32_t ss = bkf_loc_owner(sl), sbid = bkf_src_block(x);
                BucketPhysicalBlock sb = bkf_high_main(ss, sbid);
                sum = gpu_direct_add(sum,
                    bkf_ptr(ss, sb.off + Code(bkf_loc_rank(sl)) * sb.cols + lr)[0]);
            }
            if (cc) {
                uint32_t dest_code = D_BKF_LOW_CODES[
                    D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER) * D_BKF_CODE_PITCH + db.hs] + lr];
                for (uint32_t e = rec.cross_begin; e < rec.cross_begin + cc; ++e) {
                    uint32_t x = D_RBF_HIGH_CROSS_OP[e], sl = bkf_src_locator(x);
                    uint32_t sbid = bkf_src_block(x);
                    BucketPhysicalBlock sb = bkf_high_main(bkf_loc_owner(sl), sbid);
                    sum = gpu_direct_add(sum,
                        bkf_sum_low_preimages(dest_code, bkf_cross_depth(x), sb.hs, sbid, sl));
                }
            }
            *dp = sum;
        }
    }
}

struct ReverseBucketFusedDeviceTables {
    ReverseBucketOrbitOp *low_orbit=nullptr,*high_orbit=nullptr;
    uint32_t *low_orbit_off=nullptr,*high_orbit_off=nullptr;
    BucketFusedDst *low_dst=nullptr,*high_dst=nullptr;
    uint32_t *low_off=nullptr,*high_off=nullptr;
    uint32_t *low_local=nullptr,*low_cross=nullptr,*high_local=nullptr,*high_cross=nullptr;

    template<class T>
    static void cp(T*& d, const std::vector<T>& s, const char* what) {
        if (s.empty()) return;
        ck(cudaMalloc(&d, s.size() * sizeof(T)), what);
        ck(cudaMemcpy(d, s.data(), s.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    }

    void install(const ReverseBucketAtomicHost& orbit, const ReverseBucketFusedHost& f) {
        cp(low_orbit, orbit.low_orbit, "reverse fused low orbit");
        cp(high_orbit, orbit.high_orbit, "reverse fused high orbit");
        cp(low_orbit_off, orbit.low_orbit_off, "reverse fused low orbit off");
        cp(high_orbit_off, orbit.high_orbit_off, "reverse fused high orbit off");
        cp(low_dst, f.low_dst, "reverse fused low dst");
        cp(high_dst, f.high_dst, "reverse fused high dst");
        cp(low_off, f.low_off, "reverse fused low off");
        cp(high_off, f.high_off, "reverse fused high off");
        cp(low_local, f.low_local_src, "reverse fused low local");
        cp(low_cross, f.low_cross_op, "reverse fused low cross");
        cp(high_local, f.high_local_src, "reverse fused high local");
        cp(high_cross, f.high_cross_op, "reverse fused high cross");
        ck(cudaMemcpyToSymbol(D_RB_LOW_ORBIT,&low_orbit,sizeof(low_orbit)),"reverse fused low orbit ptr");
        ck(cudaMemcpyToSymbol(D_RB_HIGH_ORBIT,&high_orbit,sizeof(high_orbit)),"reverse fused high orbit ptr");
        ck(cudaMemcpyToSymbol(D_RB_LOW_ORBIT_OFF,&low_orbit_off,sizeof(low_orbit_off)),"reverse fused low orbit off ptr");
        ck(cudaMemcpyToSymbol(D_RB_HIGH_ORBIT_OFF,&high_orbit_off,sizeof(high_orbit_off)),"reverse fused high orbit off ptr");
        uint32_t opitch=orbit.nblocks+1;
        ck(cudaMemcpyToSymbol(D_RB_PITCH,&opitch,sizeof(opitch)),"reverse fused orbit pitch");
        ck(cudaMemcpyToSymbol(D_RBF_LOW_DST,&low_dst,sizeof(low_dst)),"reverse fused low dst ptr");
        ck(cudaMemcpyToSymbol(D_RBF_HIGH_DST,&high_dst,sizeof(high_dst)),"reverse fused high dst ptr");
        ck(cudaMemcpyToSymbol(D_RBF_LOW_OFF,&low_off,sizeof(low_off)),"reverse fused low off ptr");
        ck(cudaMemcpyToSymbol(D_RBF_HIGH_OFF,&high_off,sizeof(high_off)),"reverse fused high off ptr");
        ck(cudaMemcpyToSymbol(D_RBF_LOW_LOCAL_SRC,&low_local,sizeof(low_local)),"reverse fused low local ptr");
        ck(cudaMemcpyToSymbol(D_RBF_LOW_CROSS_OP,&low_cross,sizeof(low_cross)),"reverse fused low cross ptr");
        ck(cudaMemcpyToSymbol(D_RBF_HIGH_LOCAL_SRC,&high_local,sizeof(high_local)),"reverse fused high local ptr");
        ck(cudaMemcpyToSymbol(D_RBF_HIGH_CROSS_OP,&high_cross,sizeof(high_cross)),"reverse fused high cross ptr");
        ck(cudaMemcpyToSymbol(D_RBF_LOW_PITCH,&f.low_pitch,sizeof(f.low_pitch)),"reverse fused low pitch");
        ck(cudaMemcpyToSymbol(D_RBF_HIGH_PITCH,&f.high_pitch,sizeof(f.high_pitch)),"reverse fused high pitch");
    }

    void release() {
        if(low_orbit)cudaFree(low_orbit);if(high_orbit)cudaFree(high_orbit);
        if(low_orbit_off)cudaFree(low_orbit_off);if(high_orbit_off)cudaFree(high_orbit_off);
        if(low_dst)cudaFree(low_dst);if(high_dst)cudaFree(high_dst);
        if(low_off)cudaFree(low_off);if(high_off)cudaFree(high_off);
        if(low_local)cudaFree(low_local);if(low_cross)cudaFree(low_cross);
        if(high_local)cudaFree(high_local);if(high_cross)cudaFree(high_cross);
        low_orbit=high_orbit=nullptr;low_orbit_off=high_orbit_off=nullptr;
        low_dst=high_dst=nullptr;low_off=high_off=nullptr;
        low_local=low_cross=high_local=high_cross=nullptr;
    }
};

static void bucket_launch_reverse_low_fused(
    const StorageLayout& layout,int threads=256,int grid_x=16,int grid_y=8
) {
    dim3 block(threads), og(grid_x,grid_y,unsigned(layout.main_blocks.size()));
    dim3 cg(grid_x,grid_y,unsigned(layout.block_blocks.size()));
    for(int p=1;p<=LOW_LUT_K;++p){
        bucket_reverse_low_orbit_kernel<<<og,block>>>(p);ck(cudaGetLastError(),"reverse fused low orbit");
        bucket_reverse_low_fused_closure_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"reverse fused low closure");
    }
}

static void bucket_launch_reverse_high_fused(
    const StorageLayout& layout,int threads=256,int grid_x=16,int grid_y=8
) {
    dim3 block(threads), og(grid_x,grid_y,unsigned(layout.main_blocks.size()));
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){
        bucket_reverse_high_orbit_kernel<<<og,block>>>(p);ck(cudaGetLastError(),"reverse fused high orbit");
        unsigned nt=p==TARGET_W-1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());
        dim3 cg(grid_x,grid_y,nt);
        bucket_reverse_high_fused_closure_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"reverse fused high closure");
    }
}
