#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <utility>
#include <vector>

// v0.7: communication-aware occupancy-block partitioning.
// Edges approximate P2P bytes saved when two logical blocks are colocated.
// Memory and a compute proxy are constrained; exact staging bytes are checked
// after partitioning before the candidate is allowed to replace legacy LPT.

struct GdpgGraph {
    std::uint32_t main_n=0,block_n=0,n=0;
    std::vector<unsigned long long> bytes,work,edge;
    std::uint32_t mn(std::uint32_t b) const { return b; }
    std::uint32_t bn(std::uint32_t b) const { return main_n+b; }
    unsigned long long& e(std::uint32_t a,std::uint32_t b){return edge[std::size_t(a)*n+b];}
    unsigned long long e(std::uint32_t a,std::uint32_t b)const{return edge[std::size_t(a)*n+b];}
};

static inline unsigned long long gdpg_bytes(const StorageBlock& b){
    return static_cast<unsigned long long>(b.rows)*b.cols*sizeof(Count);
}
static inline void gdpg_edge(GdpgGraph&g,std::uint32_t a,std::uint32_t b,unsigned long long w){
    if(a==b||!w)return;g.e(a,b)+=w;g.e(b,a)+=w;
}

static GdpgGraph build_gdpg_graph(
    const StorageLayout& layout,const LowOrbitHost& loworbit,
    const CpuHighDirectHost& highdirect,const GpuDirectGatherHost& ordinary,
    const GpuDirectCrossGatherHost& cross
){
    GdpgGraph g;g.main_n=std::uint32_t(layout.main_blocks.size());g.block_n=std::uint32_t(layout.block_blocks.size());g.n=g.main_n+g.block_n;
    g.bytes.assign(g.n,0);g.work.assign(g.n,0);g.edge.assign(std::size_t(g.n)*g.n,0);
    for(std::uint32_t i=0;i<g.main_n;++i){g.bytes[g.mn(i)]=gdpg_bytes(layout.main_blocks[i]);g.work[g.mn(i)]=g.bytes[g.mn(i)]/sizeof(Count);}
    for(std::uint32_t i=0;i<g.block_n;++i){g.bytes[g.bn(i)]=gdpg_bytes(layout.block_blocks[i]);g.work[g.bn(i)]=g.bytes[g.bn(i)]/sizeof(Count);}

    // Gather: source MAIN block -> destination logical block. Deduplicate each
    // source within one logical destination and phase. LOW p=1 CROSS refreshes
    // after local gather, so those affinities count twice.
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);
        for(std::uint32_t dbid=0;dbid<g.block_n;++dbid){
            std::vector<std::uint8_t> seen(g.main_n,0);std::size_t oi=std::size_t(pi)*ordinary.high_pitch+dbid;
            for(std::uint32_t q=ordinary.high_off[oi];q<ordinary.high_off[oi+1];++q){const auto&r=ordinary.high_dst[q];g.work[g.bn(dbid)]+=static_cast<unsigned long long>(r.edge_count)*layout.block_blocks[dbid].cols;for(std::uint32_t e=r.edge_begin;e<r.edge_begin+r.edge_count;++e)seen[gdms_src_block(ordinary.high_src[e])]=1;}
            oi=std::size_t(pi)*cross.high_pitch+dbid;
            for(std::uint32_t q=cross.high_off[oi];q<cross.high_off[oi+1];++q){const auto&r=cross.high_dst[q];g.work[g.bn(dbid)]+=2ull*r.edge_count*layout.block_blocks[dbid].cols;for(std::uint32_t e=r.edge_begin;e<r.edge_begin+r.edge_count;++e)seen[gdms_cross_block(cross.high_op[e])]=1;}
            for(std::uint32_t s=0;s<g.main_n;++s)if(seen[s])gdpg_edge(g,g.mn(s),g.bn(dbid),gdpg_bytes(layout.main_blocks[s]));
        }
    }
    for(int p=LOW_LUT_K;p>=1;--p){
        std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);bool tm=p==1;std::uint32_t dn=tm?g.main_n:g.block_n;
        for(std::uint32_t dbid=0;dbid<dn;++dbid){
            std::vector<std::uint8_t> seen(g.main_n,0),refresh(g.main_n,0);std::uint32_t dnode=tm?g.mn(dbid):g.bn(dbid);const auto&db=tm?layout.main_blocks[dbid]:layout.block_blocks[dbid];
            std::size_t oi=std::size_t(pi)*ordinary.low_pitch+dbid;
            for(std::uint32_t q=ordinary.low_off[oi];q<ordinary.low_off[oi+1];++q){const auto&r=ordinary.low_dst[q];g.work[dnode]+=static_cast<unsigned long long>(r.edge_count)*db.rows;for(std::uint32_t e=r.edge_begin;e<r.edge_begin+r.edge_count;++e)seen[gdms_src_block(ordinary.low_src[e])]=1;}
            oi=std::size_t(pi)*cross.low_pitch+dbid;
            for(std::uint32_t q=cross.low_off[oi];q<cross.low_off[oi+1];++q){const auto&r=cross.low_dst[q];g.work[dnode]+=2ull*r.edge_count*db.rows;for(std::uint32_t e=r.edge_begin;e<r.edge_begin+r.edge_count;++e){std::uint32_t s=gdms_cross_block(cross.low_op[e]);seen[s]=1;if(p==1)refresh[s]=1;}}
            for(std::uint32_t s=0;s<g.main_n;++s)if(seen[s]){auto z=gdpg_bytes(layout.main_blocks[s]);gdpg_edge(g,g.mn(s),dnode,z);if(refresh[s])gdpg_edge(g,g.mn(s),dnode,z);}
        }
    }

    // HIGH destination-owned orbit affinities.
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        std::uint32_t pi=std::uint32_t((TARGET_W-1)-p);
        for(std::uint32_t bid=0;bid<g.main_n;++bid){
            const auto&sb=layout.main_blocks[bid];if(!sb.valid||!sb.rows||!sb.cols)continue;FBlock fb{};fb.he=sb.he;fb.hs=sb.hs;fb.c=sb.c;std::uint32_t dbid=std::uint32_t(sb.hs);
            auto [na,ne]=cpu_high_direct_range(highdirect.orbit_off.nn,highdirect.nblocks,pi,bid);auto [ra,re]=cpu_high_direct_range(highdirect.orbit_off.nrnl,highdirect.nblocks,pi,bid);g.work[g.mn(bid)]+=static_cast<unsigned long long>((ne-na)+(re-ra))*sb.cols;
            if(na!=ne){std::uint32_t j=cpu_high_orbit_partner_block(bid,fb,p,true);gdpg_edge(g,g.mn(bid),g.bn(dbid),gdpg_bytes(layout.block_blocks[dbid]));gdpg_edge(g,g.mn(j),g.mn(bid),gdpg_bytes(layout.main_blocks[bid]));}
            if(ra!=re){std::uint32_t j=cpu_high_orbit_partner_block(bid,fb,p,false);gdpg_edge(g,g.mn(bid),g.mn(j),gdpg_bytes(layout.main_blocks[j]));gdpg_edge(g,g.mn(bid),g.bn(dbid),gdpg_bytes(layout.block_blocks[dbid]));gdpg_edge(g,g.bn(dbid),g.mn(bid),gdpg_bytes(layout.main_blocks[bid]));}
        }
    }

    // LOW orbit. Deduplicate ordered (destination,source) stage relations per
    // phase; graph storage is symmetric but source byte weight is preserved.
    for(int p=LOW_LUT_K;p>=1;--p){
        std::uint32_t pi=std::uint32_t(LOW_LUT_K-p);std::vector<std::uint8_t> seen(std::size_t(g.n)*g.n,0);
        auto stage=[&](std::uint32_t dst,std::uint32_t src){if(dst==src)return;std::size_t k=std::size_t(dst)*g.n+src;if(seen[k])return;seen[k]=1;gdpg_edge(g,dst,src,g.bytes[src]);};
        for(std::uint32_t bid=0;bid<g.main_n;++bid){
            const auto&sb=layout.main_blocks[bid];if(!sb.valid||!sb.rows||!sb.cols)continue;std::uint64_t cells=0;
            for(std::uint32_t lr=0;lr<sb.cols;++lr){std::uint64_t ow=loworbit.rec[std::size_t(pi)*loworbit.main_total+loworbit.main_base[bid]+lr];std::uint32_t kind=cpu_orbit_kind(ow);if(kind<CPU_ORBIT_NN||kind>CPU_ORBIT_NL)continue;++cells;std::uint32_t j=cpu_orbit_jblock(ow),d=cpu_orbit_dblock(ow);std::uint32_t x=g.mn(bid),y=g.mn(j),z=g.bn(d);if(kind==CPU_ORBIT_NN){stage(x,z);stage(y,x);}else if(p==1){stage(x,y);stage(x,z);stage(y,x);}else{stage(x,y);stage(x,z);stage(z,x);}}
            g.work[g.mn(bid)]+=cells*sb.rows;
        }
    }
    return g;
}

static GdmShardHost gdpg_pack(const StorageLayout&layout,const std::vector<std::uint8_t>&owner,int ngpu){
    std::uint32_t M=std::uint32_t(layout.main_blocks.size()),B=std::uint32_t(layout.block_blocks.size());if(owner.size()!=std::size_t(M+B))std::exit(187);GdmShardHost out;out.main_blocks.resize(M);out.block_blocks.resize(B);
    for(std::uint32_t i=0;i<M;++i){const auto&sb=layout.main_blocks[i];Code n=Code(sb.rows)*sb.cols;int d=owner[i];GdmBlock b;b.rows=sb.rows;b.cols=sb.cols;b.he=sb.he;b.hs=sb.hs;b.c=sb.c;b.valid=sb.valid;b.owner=std::uint8_t(d);b.off=out.main_elems[d];out.main_elems[d]+=n;out.main_blocks[i]=b;}
    for(std::uint32_t i=0;i<B;++i){const auto&sb=layout.block_blocks[i];Code n=Code(sb.rows)*sb.cols;int d=owner[M+i];GdmBlock b;b.rows=sb.rows;b.cols=sb.cols;b.he=sb.he;b.hs=sb.hs;b.c=sb.c;b.valid=sb.valid;b.owner=std::uint8_t(d);b.off=out.block_elems[d];out.block_elems[d]+=n;out.block_blocks[i]=b;}
    for(int d=0;d<ngpu;++d)out.total_elems[d]=out.main_elems[d]+out.block_elems[d];out.max_elems=*std::max_element(out.total_elems.begin(),out.total_elems.begin()+ngpu);out.min_elems=*std::min_element(out.total_elems.begin(),out.total_elems.begin()+ngpu);return out;
}
static std::vector<std::uint8_t> gdpg_owners(const GdmShardHost&s){std::vector<std::uint8_t>o;o.reserve(s.main_blocks.size()+s.block_blocks.size());for(const auto&b:s.main_blocks)o.push_back(b.owner);for(const auto&b:s.block_blocks)o.push_back(b.owner);return o;}
static unsigned long long gdpg_cut(const GdpgGraph&g,const std::vector<std::uint8_t>&o){unsigned long long z=0;for(std::uint32_t a=0;a<g.n;++a)for(std::uint32_t b=a+1;b<g.n;++b)if(o[a]!=o[b])z+=g.e(a,b);return z;}

static std::vector<std::uint8_t> gdpg_improve(const GdpgGraph&g,std::vector<std::uint8_t>o,int ngpu){
    unsigned long long tb=std::accumulate(g.bytes.begin(),g.bytes.end(),0ull),tw=std::accumulate(g.work.begin(),g.work.end(),0ull);unsigned long long mb=*std::max_element(g.bytes.begin(),g.bytes.end()),mw=*std::max_element(g.work.begin(),g.work.end());unsigned long long cb=std::max(mb,((tb+ngpu-1)/ngpu)*106/100),cw=std::max(mw,((tw+ngpu-1)/ngpu)*115/100);std::array<unsigned long long,GDM_MAX_GPU>lb{},lw{};for(std::uint32_t i=0;i<g.n;++i){lb[o[i]]+=g.bytes[i];lw[o[i]]+=g.work[i];}
    for(int sweep=0;sweep<12;++sweep){bool changed=false;for(std::uint32_t i=0;i<g.n;++i){int cur=o[i],best=cur;long long best_gain=0;for(int d=0;d<ngpu;++d)if(d!=cur){if(lb[d]+g.bytes[i]>cb||lw[d]+g.work[i]>cw)continue;unsigned long long oldc=0,newc=0;for(std::uint32_t j=0;j<g.n;++j)if(j!=i){auto w=g.e(i,j);if(!w)continue;if(o[j]!=cur)oldc+=w;if(o[j]!=d)newc+=w;}long long gain=oldc>=newc?static_cast<long long>(oldc-newc):-static_cast<long long>(newc-oldc);if(gain>best_gain||(gain==best_gain&&gain>0&&lb[d]<lb[best])){best_gain=gain;best=d;}}if(best!=cur&&best_gain>0){lb[cur]-=g.bytes[i];lw[cur]-=g.work[i];lb[best]+=g.bytes[i];lw[best]+=g.work[i];o[i]=std::uint8_t(best);changed=true;}}if(!changed)break;}return o;
}

static std::vector<std::uint8_t> gdpg_greedy(const GdpgGraph&g,int ngpu){
    unsigned long long tb=std::accumulate(g.bytes.begin(),g.bytes.end(),0ull),tw=std::accumulate(g.work.begin(),g.work.end(),0ull);unsigned long long mb=*std::max_element(g.bytes.begin(),g.bytes.end()),mw=*std::max_element(g.work.begin(),g.work.end());unsigned long long cb=std::max(mb,((tb+ngpu-1)/ngpu)*106/100),cw=std::max(mw,((tw+ngpu-1)/ngpu)*115/100);std::vector<std::uint32_t>ord(g.n);std::iota(ord.begin(),ord.end(),0u);std::vector<unsigned long long>deg(g.n,0);for(std::uint32_t i=0;i<g.n;++i)for(std::uint32_t j=0;j<g.n;++j)deg[i]+=g.e(i,j);std::sort(ord.begin(),ord.end(),[&](auto a,auto b){return deg[a]!=deg[b]?deg[a]>deg[b]:g.bytes[a]>g.bytes[b];});std::vector<std::uint8_t>o(g.n,0xff);std::array<unsigned long long,GDM_MAX_GPU>lb{},lw{};
    for(auto i:ord){int best=-1;unsigned long long ba=0;for(int d=0;d<ngpu;++d){if(lb[d]+g.bytes[i]>cb||lw[d]+g.work[i]>cw)continue;unsigned long long a=0;for(std::uint32_t j=0;j<g.n;++j)if(o[j]==d)a+=g.e(i,j);if(best<0||a>ba||(a==ba&&lb[d]<lb[best])){best=d;ba=a;}}if(best<0){best=0;for(int d=1;d<ngpu;++d)if(lb[d]<lb[best])best=d;}o[i]=std::uint8_t(best);lb[best]+=g.bytes[i];lw[best]+=g.work[i];}return gdpg_improve(g,std::move(o),ngpu);
}

struct GdpgCandidate{GdmShardHost shard;unsigned long long graph_cut=0;double max_to_avg=0.0;};
static GdpgCandidate build_gdpg_candidate(const StorageLayout&layout,const LowOrbitHost&loworbit,const CpuHighDirectHost&highdirect,const GpuDirectGatherHost&ordinary,const GpuDirectCrossGatherHost&cross,int ngpu){GdpgGraph g=build_gdpg_graph(layout,loworbit,highdirect,ordinary,cross);GdmShardHost legacy=build_gdm_shards(layout,ngpu);auto a=gdpg_improve(g,gdpg_owners(legacy),ngpu),b=gdpg_greedy(g,ngpu);bool use_b=gdpg_cut(g,b)<gdpg_cut(g,a);const auto&chosen=use_b?b:a;GdpgCandidate out;out.graph_cut=gdpg_cut(g,chosen);out.shard=gdpg_pack(layout,chosen,ngpu);unsigned long long total=0;for(int d=0;d<ngpu;++d)total+=out.shard.total_elems[d];double avg=ngpu?double(total)/ngpu:0.0;out.max_to_avg=avg?double(out.shard.max_elems)/avg:0.0;return out;}
