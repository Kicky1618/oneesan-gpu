#pragma once

#include "ramstream32_bucket_reverse_split18.hpp"

// Build split18 directly from the legacy reverse orbit stream and fused
// closure destinations.  The old path first materialized a 32-bit attachment
// entry for every orbit, then immediately converted that entry to an 18-bit
// destination-block-local ordinal.  Here the lookup and localization are
// fused, removing that 4 B/orbit transient.
static ReverseSplit18Host build_reverse_split18_direct_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,const BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf,bool release_legacy=false
){
    validate_reverse_bucket_partner_blocks(layout,rb);
    (void)validate_bucket_orbit_closure_fusion(layout,bo,bf,rb,rf);

    const size_t legacy_ops=rb.low_orbit.size()+rb.high_orbit.size();
    const size_t legacy_orbit_bytes=reverse_bucket_orbit_bytes(rb);
    const size_t legacy_total_bytes=rb.bytes();
    ReverseSplit18Host out;out.nblocks=rb.nblocks;
    const size_t pitch=size_t(rb.nblocks)+1;
    std::vector<uint8_t> used_low(rf.low_dst.size()),used_high(rf.high_dst.size());
    uint64_t attached=0;

    auto build_side=[&](bool low,ReverseSplit18SideHost&s){
        const uint32_t steps=low?LOW_LUT_K:HIGH_LUT_K;
        const int p0=low?1:LOW_LUT_K+1;
        auto&ov=low?rb.low_orbit:rb.high_orbit;
        auto&oo=low?rb.low_orbit_off:rb.high_orbit_off;
        const auto&fd=low?rf.low_dst:rf.high_dst;
        const auto&fo=low?rf.low_off:rf.high_off;
        const uint32_t fp=low?rf.low_pitch:rf.high_pitch;
        auto&used=low?used_low:used_high;
        const char*what=low?"reverse-split18-direct-low":"reverse-split18-direct-high";

        s.nn_off.resize(size_t(steps)*pitch);
        s.nr_off.resize(size_t(steps)*pitch);
        s.nl_off.resize(size_t(steps)*pitch);

        for(uint32_t pi=0;pi<steps;++pi){
            const int p=p0+int(pi);
            const bool edge=!low&&p==TARGET_W-1;
            const uint32_t nt=low?uint32_t(layout.block_blocks.size()):
                (edge?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size()));
            std::unordered_map<BkocKey,uint32_t>dst;
            for(uint32_t dbid=0;dbid<nt;++dbid){
                const size_t doi=size_t(pi)*fp+dbid;
                const uint32_t a=fo[doi],b=fo[doi+1];
                for(uint32_t q=a;q<b;++q)bkoc_add_dst(dst,dbid,fd[q].dst_locator,q,what);
            }

            for(uint32_t bid=0;bid<rb.nblocks;++bid){
                const size_t oi=size_t(pi)*pitch+bid;
                s.nn_off[oi]=uint32_t(s.nn.size());
                s.nr_off[oi]=uint32_t(s.nr.size());
                s.nl_off[oi]=uint32_t(s.nl.size());
                const uint32_t dbid=low?uint32_t(layout.main_blocks[bid].he):
                    (edge?bid:uint32_t(layout.main_blocks[bid].hs));
                const size_t doi=size_t(pi)*fp+dbid;
                const uint32_t a=oo[oi],b=oo[oi+1];
                for(uint32_t q=a;q<b;++q){
                    const uint64_t w=ov[q];
                    const uint32_t kind=rb_orbit_kind(w);
                    uint32_t rid=BKOC_NONE;
                    if(low)rid=bkoc_lookup(dst,dbid,rb_orbit_drop(w));
                    else if(edge){if(kind==CPU_ORBIT_NN)rid=bkoc_lookup(dst,bid,rb_orbit_src(w));}
                    else rid=bkoc_lookup(dst,dbid,rb_orbit_drop(w));
                    if(rid!=BKOC_NONE){if(used[rid]++)std::exit(low?366:367);++attached;}
                    const uint32_t z=bkoc18_local_code(rid,fo,doi,what);
                    const BucketOrbitOp nw=BucketOrbitOp(w&BKOC18_FORWARD_BASE_MASK)|(uint64_t(z&0x3ffu)<<54);
                    const uint8_t hi=uint8_t(z>>10);
                    if(kind==CPU_ORBIT_NN){s.nn.push_back(nw);s.nn_hi.push_back(hi);}
                    else if(kind==CPU_ORBIT_NR){s.nr.push_back(nw);s.nr_hi.push_back(hi);}
                    else if(kind==CPU_ORBIT_NL){s.nl.push_back(nw);s.nl_hi.push_back(hi);}
                    else{std::cerr<<"reverse split direct invalid orbit kind="<<kind<<'\n';std::exit(432);}
                }
            }
            const size_t end=size_t(pi)*pitch+rb.nblocks;
            s.nn_off[end]=uint32_t(s.nn.size());
            s.nr_off[end]=uint32_t(s.nr.size());
            s.nl_off[end]=uint32_t(s.nl.size());
        }
    };

    build_side(true,out.low);
    if(release_legacy){
        std::vector<ReverseBucketOrbitOp>().swap(rb.low_orbit);
        std::vector<uint32_t>().swap(rb.low_orbit_off);
    }
    build_side(false,out.high);
    if(release_legacy){
        std::vector<ReverseBucketOrbitOp>().swap(rb.high_orbit);
        std::vector<uint32_t>().swap(rb.high_orbit_off);
    }

    for(size_t q=0;q<used_low.size();++q)if(used_low[q]!=1){
        std::cerr<<"reverse split direct LOW unattached closure q="<<q
                 <<" used="<<unsigned(used_low[q])<<'\n';std::exit(433);
    }
    for(size_t q=0;q<used_high.size();++q)if(used_high[q]!=1){
        std::cerr<<"reverse split direct HIGH unattached closure q="<<q
                 <<" used="<<unsigned(used_high[q])<<'\n';std::exit(434);
    }
    if(attached!=rf.low_dst.size()+rf.high_dst.size())std::exit(435);
    if(out.low.ops()+out.high.ops()!=legacy_ops)std::exit(436);

    if(release_legacy){
        // rf is already fully materialized, and direct closure-only device
        // tables ignore rb.  Release the remaining source-oriented closure
        // payload too; nblocks is retained for diagnostics only.
        std::vector<ReverseBucketClosureOp>().swap(rb.low_closure);
        std::vector<ReverseBucketClosureOp>().swap(rb.high_closure);
        std::vector<uint32_t>().swap(rb.low_closure_off);
        std::vector<uint32_t>().swap(rb.high_closure_off);
    }

    std::cerr<<"reverse_split18_direct low_ops="<<out.low.ops()
             <<" high_ops="<<out.high.ops()
             <<" attached="<<attached
             <<" split_mib="<<double(out.bytes())/double(1<<20)
             <<" avoided_attach_mib="<<double(4ull*legacy_ops)/double(1<<20)
             <<" legacy_orbit_mib="<<double(legacy_orbit_bytes)/double(1<<20)
             <<" legacy_total_mib="<<double(legacy_total_bytes)/double(1<<20)
             <<" release_legacy="<<int(release_legacy)<<'\n';
    return out;
}
