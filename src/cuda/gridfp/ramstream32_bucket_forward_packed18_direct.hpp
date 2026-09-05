#pragma once

#include "ramstream32_bucket_orbit_closure_packed18.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

// Build the final forward packed18 attachment directly.  The legacy packed18
// builder first materializes a uint32_t attachment for every orbit and then
// repacks it.  This path performs the same destination lookup and immediately
// stores the local ordinal in the orbit word + one-byte high sidecar, avoiding
// the transient 4 B/orbit attachment.
static BucketForwardOrbitClosureAttach18Host build_bucket_forward_orbit_closure_attach18_direct(
    const StorageLayout& layout,BucketOrbitStreamsHost& bo,const BucketFusedHost& bf
){
    BucketForwardOrbitClosureAttach18Host out;
    out.low_nn.resize(bo.low_nn.size());out.low_nr.resize(bo.low_nr.size());out.low_nl.resize(bo.low_nl.size());
    out.high_nn.resize(bo.high_nn.size());out.high_nrnl.resize(bo.high_nrnl.size());
    std::vector<uint8_t> used_low(bf.low_dst.size()),used_high(bf.high_dst.size());
    uint64_t attached=0;uint32_t maxrec=0;

    auto encode=[&](BucketOrbitOp&op,uint8_t&hi,uint32_t rid,const std::vector<uint32_t>&off,size_t doi,
                    std::vector<uint8_t>&used,const char*what){
        uint32_t z=bkoc18_local_code(rid,off,doi,what);
        op=(op&BKOC18_FORWARD_BASE_MASK)|(uint64_t(z&0x3ffu)<<54);
        hi=uint8_t(z>>10);
        if(rid!=BKOC_NONE){
            if(rid>=used.size()){std::cerr<<"forward packed18 direct rid overflow "<<what<<" rid="<<rid<<'/'<<used.size()<<'\n';std::exit(480);}
            if(used[rid]++){std::cerr<<"forward packed18 direct duplicate attachment "<<what<<" rid="<<rid<<'\n';std::exit(481);}
            ++attached;
        }
    };

    const size_t lpitch=size_t(bo.low_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){
        uint32_t pi=uint32_t(LOW_LUT_K-p);bool target_main=p==1;
        uint32_t nt=target_main?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());
        std::unordered_map<BkocKey,uint32_t> dst;
        size_t nrec=0;
        for(uint32_t dbid=0;dbid<nt;++dbid){
            uint32_t a=bf.low_off[size_t(pi)*bf.low_pitch+dbid],b=bf.low_off[size_t(pi)*bf.low_pitch+dbid+1];
            nrec+=size_t(b-a);maxrec=std::max(maxrec,b-a);
        }
        dst.reserve(nrec);
        for(uint32_t dbid=0;dbid<nt;++dbid){
            uint32_t a=bf.low_off[size_t(pi)*bf.low_pitch+dbid],b=bf.low_off[size_t(pi)*bf.low_pitch+dbid+1];
            for(uint32_t q=a;q<b;++q)bkoc_add_dst(dst,dbid,bf.low_dst[q].dst_locator,q,"forward18-direct-low");
        }
        for(uint32_t bid=0;bid<bo.low_nblocks;++bid){
            uint32_t drop_bid=uint32_t(layout.main_blocks[bid].he);
            auto scan=[&](std::vector<BucketOrbitOp>&ops,const std::vector<uint32_t>&off,std::vector<uint8_t>&hi,uint32_t kind){
                uint32_t a=off[size_t(pi)*lpitch+bid],b=off[size_t(pi)*lpitch+bid+1];
                size_t doi=size_t(pi)*bf.low_pitch+(target_main?bid:drop_bid);
                for(uint32_t q=a;q<b;++q){
                    uint32_t rid=BKOC_NONE;
                    if(target_main){if(kind==CPU_ORBIT_NN)rid=bkoc_lookup(dst,bid,bkf_orbit_src(ops[q]));}
                    else rid=bkoc_lookup(dst,drop_bid,bkf_orbit_drop(ops[q]));
                    encode(ops[q],hi[q],rid,bf.low_off,doi,used_low,"forward18-direct-low");
                }
            };
            scan(bo.low_nn,bo.low_nn_off,out.low_nn,CPU_ORBIT_NN);
            scan(bo.low_nr,bo.low_nr_off,out.low_nr,CPU_ORBIT_NR);
            scan(bo.low_nl,bo.low_nl_off,out.low_nl,CPU_ORBIT_NL);
        }
    }

    const size_t hpitch=size_t(bo.high_nblocks)+1;
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        uint32_t pi=uint32_t((TARGET_W-1)-p);
        std::unordered_map<BkocKey,uint32_t> dst;
        size_t nrec=0;
        for(uint32_t dbid=0;dbid<uint32_t(layout.block_blocks.size());++dbid){
            uint32_t a=bf.high_off[size_t(pi)*bf.high_pitch+dbid],b=bf.high_off[size_t(pi)*bf.high_pitch+dbid+1];
            nrec+=size_t(b-a);maxrec=std::max(maxrec,b-a);
        }
        dst.reserve(nrec);
        for(uint32_t dbid=0;dbid<uint32_t(layout.block_blocks.size());++dbid){
            uint32_t a=bf.high_off[size_t(pi)*bf.high_pitch+dbid],b=bf.high_off[size_t(pi)*bf.high_pitch+dbid+1];
            for(uint32_t q=a;q<b;++q)bkoc_add_dst(dst,dbid,bf.high_dst[q].dst_locator,q,"forward18-direct-high");
        }
        for(uint32_t bid=0;bid<bo.high_nblocks;++bid){
            const auto&xb=layout.main_blocks[bid];FBlock fx{};fx.he=xb.he;fx.hs=xb.hs;fx.c=xb.c;
            uint32_t drop_bid=cpu_high_orbit_drop_block(fx);size_t doi=size_t(pi)*bf.high_pitch+drop_bid;
            auto scan=[&](std::vector<BucketOrbitOp>&ops,const std::vector<uint32_t>&off,std::vector<uint8_t>&hi){
                uint32_t a=off[size_t(pi)*hpitch+bid],b=off[size_t(pi)*hpitch+bid+1];
                for(uint32_t q=a;q<b;++q){uint32_t rid=bkoc_lookup(dst,drop_bid,bkf_orbit_drop(ops[q]));encode(ops[q],hi[q],rid,bf.high_off,doi,used_high,"forward18-direct-high");}
            };
            scan(bo.high_nn,bo.high_nn_off,out.high_nn);scan(bo.high_nrnl,bo.high_nrnl_off,out.high_nrnl);
        }
    }

    for(size_t q=0;q<used_low.size();++q)if(used_low[q]!=1){std::cerr<<"forward packed18 direct LOW unattached closure q="<<q<<" used="<<unsigned(used_low[q])<<'\n';std::exit(482);}
    for(size_t q=0;q<used_high.size();++q)if(used_high[q]!=1){std::cerr<<"forward packed18 direct HIGH unattached closure q="<<q<<" used="<<unsigned(used_high[q])<<'\n';std::exit(483);}
    if(attached!=bf.low_dst.size()+bf.high_dst.size()){std::cerr<<"forward packed18 direct attached count mismatch attached="<<attached<<" expected="<<(bf.low_dst.size()+bf.high_dst.size())<<'\n';std::exit(484);}
    size_t norbit=bo.low_nn.size()+bo.low_nr.size()+bo.low_nl.size()+bo.high_nn.size()+bo.high_nrnl.size();
    std::cerr<<"bucket_forward_orbit_closure_attach18_direct attached="<<attached
             <<" max_block_records="<<maxrec
             <<" sidecar_mib="<<double(out.bytes())/double(1<<20)
             <<" avoided_attach_mib="<<double(norbit*sizeof(uint32_t))/double(1<<20)
             <<" exact_reserve=1\n";
    return out;
}
