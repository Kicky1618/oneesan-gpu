#define ROW8_GAP_INTEGER_VERIFY_NO_MAIN 1
#include "row8_gap_integer_verify.cpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

struct GapSHdr {
    char magic[8];
    uint32_t version, r;
    uint32_t dims[9];
    uint64_t total_nz;
    uint64_t fnv_hash;
};
struct GapBHdr { uint32_t sym,h,h2,rows,cols,nnz; };
struct GapVHdr { uint32_t tag,sym,h,nnz; }; // tag 1=alpha, 2=beta

static uint64_t fnv64_gap(uint64_t h, void const*vp, size_t n){
    auto p=(unsigned char const*)vp;
    for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}
    return h;
}
template<class T> static void put_gap(std::ofstream&out,T const&x,uint64_t&hh){out.write((char const*)&x,sizeof(x));hh=fnv64_gap(hh,&x,sizeof(x));}
template<class T> static void putv_gap(std::ofstream&out,std::vector<T>const&x,uint64_t&hh){if(!x.empty()){out.write((char const*)x.data(),x.size()*sizeof(T));hh=fnv64_gap(hh,x.data(),x.size()*sizeof(T));}}

using GapAdj = std::vector<std::vector<uint16_t>>;

static GapAdj build_gap_adj(GapExact const&S, GapExact const&T, int sym){
    GapAdj A(S.s->dim);
    std::vector<uint16_t> touched;
    std::vector<uint16_t> count(T.s->dim);
    for(int sc=0;sc<S.s->dim;++sc){
        touched.clear();
        WVec v{{S.s->states[S.basis_raw[sc]],1}};
        auto z=wcolumn(std::move(v),8,false,sym);
        for(auto const&e:z){
            auto it=std::lower_bound(T.s->states.begin(),T.s->states.end(),e.p);
            if(it==T.s->states.end()||!(*it==e.p))throw std::runtime_error("gap cache target missing");
            int ti=it-T.s->states.begin();
            for(int tc:T.support[ti]){
                if(!count[tc]) touched.push_back((uint16_t)tc);
                uint32_t q=(uint32_t)count[tc]+e.v;
                if(q>65535)throw std::runtime_error("gap cache transition coefficient overflow");
                count[tc]=(uint16_t)q;
            }
        }
        std::sort(touched.begin(),touched.end());
        for(auto tc:touched){
            if(count[tc]!=1)throw std::runtime_error("gap cache transition is not 0/1");
            A[sc].push_back(tc);
            count[tc]=0;
        }
    }
    return A;
}

static std::vector<uint16_t> gap_initial(std::array<GapExact,9> const&G,int sym){
    int h=1+DEL_[sym];
    State z{};z.n=8;WVec v{{pack(z),1}};auto w=wcolumn(std::move(v),8,true,sym);
    std::vector<uint16_t> count(G[h].s->dim), touched;
    for(auto const&e:w){
        auto it=std::lower_bound(G[h].s->states.begin(),G[h].s->states.end(),e.p);
        if(it==G[h].s->states.end()||!(*it==e.p))throw std::runtime_error("gap initial target");
        int ti=it-G[h].s->states.begin();
        for(int q:G[h].support[ti]){
            if(!count[q])touched.push_back((uint16_t)q);
            uint32_t c=(uint32_t)count[q]+e.v;if(c>65535)throw std::runtime_error("gap initial overflow");count[q]=(uint16_t)c;
        }
    }
    std::sort(touched.begin(),touched.end());
    for(auto q:touched)if(count[q]!=1)throw std::runtime_error("gap initial not 0/1");
    return touched;
}
static std::vector<uint16_t> gap_final(GapExact const&G,int sym){
    std::vector<uint16_t> out;
    for(int q=0;q<G.s->dim;++q){WVec v{{G.s->states[G.basis_raw[q]],1}};auto x=wlast_value(std::move(v),8,sym);if(x){if(x!=1)throw std::runtime_error("gap final not 0/1");out.push_back((uint16_t)q);}}
    return out;
}

int main(int argc,char**argv){
    MODP=4294967291u;
    std::string path=argc>1?argv[1]:"work/row8_gap_cache/row8_gap01.bin";
    Vec all;int col=0;if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all))return 2;
    std::array<std::vector<Packed>,9>H;for(auto const&p:all)H[unpack(p).sp].push_back(p);
    std::array<std::unique_ptr<Space>,9>S;
    S[0]=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));
    S[1]=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));
    S[2]=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));
    for(int h=3;h<=8;++h)S[h]=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));
    std::array<GapExact,9>G;for(int h=0;h<=8;++h)G[h]=build_gap_exact(*S[h]);

    std::array<std::array<GapAdj,9>,3>A;uint64_t total=0;
    for(int a=0;a<3;++a)for(int h=0;h<=8;++h){int h2=h+DEL_[a];if(h2<0||h2>8)continue;A[a][h]=build_gap_adj(G[h],G[h2],a);for(auto const&r:A[a][h])total+=r.size();}

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path,std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("gap cache open output");
    GapSHdr hdr{{'R','8','G','A','P','0','1','\0'},1,8,{0},total,0};for(int h=0;h<9;++h)hdr.dims[h]=S[h]->dim;
    out.write((char*)&hdr,sizeof(hdr));uint64_t hh=1469598103934665603ULL;

    for(int a=0;a<3;++a){int h=1+DEL_[a];auto v=gap_initial(G,a);GapVHdr x{1u,(uint32_t)a,(uint32_t)h,(uint32_t)v.size()};put_gap(out,x,hh);putv_gap(out,v,hh);std::cout<<"alpha a="<<a<<" h="<<h<<" nz="<<v.size()<<"\n";}
    for(auto [h,a]:{std::pair<int,int>{0,0},{1,1}}){auto v=gap_final(G[h],a);GapVHdr x{2u,(uint32_t)a,(uint32_t)h,(uint32_t)v.size()};put_gap(out,x,hh);putv_gap(out,v,hh);std::cout<<"beta a="<<a<<" h="<<h<<" nz="<<v.size()<<"\n";}

    for(int a=0;a<3;++a)for(int h=0;h<=8;++h){int h2=h+DEL_[a];if(h2<0||h2>8)continue;auto const&adj=A[a][h];uint32_t nz=0;for(auto const&r:adj)nz+=r.size();GapBHdr bh{(uint32_t)a,(uint32_t)h,(uint32_t)h2,(uint32_t)S[h]->dim,(uint32_t)S[h2]->dim,nz};put_gap(out,bh,hh);std::vector<uint32_t>rp(adj.size()+1);std::vector<uint16_t>ci;ci.reserve(nz);for(int i=0;i<(int)adj.size();++i){rp[i]=ci.size();ci.insert(ci.end(),adj[i].begin(),adj[i].end());}rp[adj.size()]=ci.size();putv_gap(out,rp,hh);putv_gap(out,ci,hh);std::cout<<"block a="<<a<<" h="<<h<<" h2="<<h2<<" nz="<<nz<<"\n";}
    if(!out)throw std::runtime_error("gap cache write");out.close();
    hdr.fnv_hash=hh;std::fstream io(path,std::ios::binary|std::ios::in|std::ios::out);io.write((char*)&hdr,sizeof(hdr));io.close();
    auto sz=std::filesystem::file_size(path);
    std::cout<<"gap_cache path="<<path<<" bytes="<<sz<<" total_nz="<<total<<" hash="<<std::hex<<hh<<std::dec<<"\n";
    return 0;
}
