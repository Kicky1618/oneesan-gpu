#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

using Mate=std::uint64_t;
static constexpr int H=4,L=5,W=H+1+L,G=8;
enum V:uint8_t{N=0,R=1,LL=2};
static V get(Mate m,int p){return V((m>>(2*p))&3u);}
static Mate setv(Mate m,int p,V v){Mate z=Mate(3)<<(2*p);return(m&~z)|(Mate(v)<<(2*p));}
static int occ_mask(uint32_t code,int k){int m=0;for(int p=0;p<k;++p)if(((code>>(2*p))&3u)!=0)m|=1<<p;return m;}
static int high_end(uint32_t code,int k){int h=1;for(int p=k-1;p>=0;--p){uint32_t v=(code>>(2*p))&3u;if(v==R)--h;else if(v==LL)++h;}return h;}
static void enum_rec(int pos,int h,Mate m,std::vector<Mate>&out){
    if(pos<0){if(h==0)out.push_back(m);return;}
    enum_rec(pos-1,h,setv(m,pos,N),out);
    if(h>0)enum_rec(pos-1,h-1,setv(m,pos,R),out);
    enum_rec(pos-1,h+1,setv(m,pos,LL),out);
}
static std::vector<Mate> enum_states(int width){std::vector<Mate>v;enum_rec(width-1,1,0,v);return v;}

static std::vector<int> lpt(const std::vector<uint64_t>&w){
    std::vector<std::pair<uint64_t,int>>q;for(int i=0;i<(int)w.size();++i)q.push_back({w[i],i});
    std::sort(q.begin(),q.end(),[](auto a,auto b){return a.first!=b.first?a.first>b.first:a.second<b.second;});
    std::array<uint64_t,G>load{};std::vector<int>o(w.size());
    for(auto [z,m]:q){int g=0;for(int j=1;j<G;++j)if(load[j]<load[g])g=j;o[m]=g;load[g]+=z;}return o;
}

struct StateParts{int he,hs,c;uint32_t hc,lc;int mh,ml,oh,ol;bool blocked;};
static StateParts split(Mate m,bool blocked,const std::vector<int>&ownerH,const std::vector<int>&ownerL){
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    StateParts s{};s.blocked=blocked;s.lc=uint32_t(m)&LM;
    if(blocked){s.hc=uint32_t((m>>(2*L))&HM);s.he=high_end(s.hc,H);s.hs=s.he;s.c=-1;}
    else{s.hc=uint32_t((m>>(2*(L+1)))&HM);s.he=high_end(s.hc,H);s.c=int(get(m,L));s.hs=s.he+(s.c==LL?1:s.c==R?-1:0);}
    s.mh=occ_mask(s.hc,H);s.ml=occ_mask(s.lc,L);s.oh=ownerH[s.mh];s.ol=ownerL[s.ml];return s;
}

struct BlockShape{uint64_t off=0;uint32_t rows=0,cols=0;};
struct BucketLayout{
    std::array<std::array<std::vector<BlockShape>,G>,G> mainb,blockb;
    std::array<std::array<uint64_t,G>,G> size{};
};

static uint64_t key(int h,uint32_t code){return(uint64_t(uint32_t(h))<<32)|code;}

int main(){
    auto ms=enum_states(W),bs=enum_states(W-1);
    std::vector<uint64_t>wH(1u<<H),wL(1u<<L);
    auto count_masks=[&](Mate m,bool blocked){
        constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
        uint32_t lc=uint32_t(m)&LM;uint32_t hc=blocked?uint32_t((m>>(2*L))&HM):uint32_t((m>>(2*(L+1)))&HM);
        ++wH[occ_mask(hc,H)];++wL[occ_mask(lc,L)];
    };
    for(Mate m:ms)count_masks(m,false);for(Mate m:bs)count_masks(m,true);
    auto ownerH=lpt(wH),ownerL=lpt(wL);

    std::array<std::set<uint32_t>,H+2> highset{};
    std::array<std::set<uint32_t>,L+2> lowset{};
    auto collect=[&](Mate m,bool blocked){auto s=split(m,blocked,ownerH,ownerL);highset[s.he].insert(s.hc);lowset[s.hs].insert(s.lc);};
    for(Mate m:ms)collect(m,false);for(Mate m:bs)collect(m,true);

    std::array<std::array<uint32_t,H+2>,G> hcount{};
    std::array<std::array<uint32_t,L+2>,G> lcount{};
    std::unordered_map<uint64_t,uint32_t> hrank,lrank;
    for(int h=0;h<H+2;++h){
        std::array<std::vector<uint32_t>,G> by{};
        for(uint32_t c:highset[h])by[ownerH[occ_mask(c,H)]].push_back(c);
        for(int g=0;g<G;++g){std::sort(by[g].begin(),by[g].end(),[](uint32_t a,uint32_t b){int ma=occ_mask(a,H),mb=occ_mask(b,H);return ma!=mb?ma<mb:a<b;});hcount[g][h]=uint32_t(by[g].size());for(uint32_t r=0;r<by[g].size();++r)hrank[key(h,by[g][r])]=r;}
    }
    for(int h=0;h<L+2;++h){
        std::array<std::vector<uint32_t>,G> by{};
        for(uint32_t c:lowset[h])by[ownerL[occ_mask(c,L)]].push_back(c);
        for(int g=0;g<G;++g){std::sort(by[g].begin(),by[g].end(),[](uint32_t a,uint32_t b){int ma=occ_mask(a,L),mb=occ_mask(b,L);return ma!=mb?ma<mb:a<b;});lcount[g][h]=uint32_t(by[g].size());for(uint32_t r=0;r<by[g].size();++r)lrank[key(h,by[g][r])]=r;}
    }

    BucketLayout bl;
    for(int a=0;a<G;++a)for(int b=0;b<G;++b){
        uint64_t off=0;bl.mainb[a][b].resize(3*(H+2));
        for(int he=0;he<H+2;++he)for(int c=0;c<3;++c){int hs=he+(c==LL?1:c==R?-1:0);int bid=3*he+c;BlockShape q;q.off=off;if(hs>=0&&hs<L+2){q.rows=hcount[a][he];q.cols=lcount[b][hs];off+=uint64_t(q.rows)*q.cols;}bl.mainb[a][b][bid]=q;}
        bl.blockb[a][b].resize(H+2);
        for(int h=0;h<H+2;++h){BlockShape q{off,hcount[a][h],lcount[b][h]};off+=uint64_t(q.rows)*q.cols;bl.blockb[a][b][h]=q;}
        bl.size[a][b]=off;
    }

    std::array<std::array<std::vector<uint64_t>,G>,G> bucket;
    for(int a=0;a<G;++a)for(int b=0;b<G;++b)bucket[a][b].assign(bl.size[a][b],0);
    uint64_t marker=1;
    auto place=[&](Mate m,bool blocked){auto s=split(m,blocked,ownerH,ownerL);auto& q=blocked?bl.blockb[s.oh][s.ol][s.he]:bl.mainb[s.oh][s.ol][3*s.he+s.c];uint32_t hr=hrank.at(key(s.he,s.hc)),lr=lrank.at(key(s.hs,s.lc));if(hr>=q.rows||lr>=q.cols){std::cerr<<"rank shape overflow\n";std::exit(2);}uint64_t addr=q.off+uint64_t(hr)*q.cols+lr;if(bucket[s.oh][s.ol][addr]){std::cerr<<"bucket collision\n";std::exit(3);}bucket[s.oh][s.ol][addr]=marker++;};
    for(Mate m:ms)place(m,false);for(Mate m:bs)place(m,true);
    uint64_t total=0;for(int a=0;a<G;++a)for(int b=0;b<G;++b){total+=bl.size[a][b];for(uint64_t x:bucket[a][b])if(!x){std::cerr<<"bucket hole a="<<a<<" b="<<b<<'\n';return 4;}}
    if(total!=ms.size()+bs.size()||marker-1!=total)return 5;

    // Fixed slots: paired slots have equal capacity=max(|B_ab|,|B_ba|).
    std::array<std::array<std::vector<uint64_t>,G>,G> slots;
    for(int a=0;a<G;++a)for(int b=0;b<G;++b){uint64_t cap=std::max(bl.size[a][b],bl.size[b][a]);slots[a][b].assign(cap,0);std::copy(bucket[a][b].begin(),bucket[a][b].end(),slots[a][b].begin());}

    // Circle-method K8 schedule; each unordered GPU pair exactly once.
    std::array<int,G> p{};std::iota(p.begin(),p.end(),0);std::set<std::pair<int,int>>pairs;
    for(int r=0;r<G-1;++r){
        for(int i=0;i<G/2;++i){int a=p[i],b=p[G-1-i];if(a>b)std::swap(a,b);if(!pairs.insert({a,b}).second){std::cerr<<"duplicate pair\n";return 6;}auto& A=slots[a][b];auto& B=slots[b][a];if(A.size()!=B.size())return 7;constexpr size_t CHUNK=17;for(size_t o=0;o<A.size();o+=CHUNK){size_t n=std::min(CHUNK,A.size()-o);std::vector<uint64_t>ta(A.begin()+o,A.begin()+o+n),tb(B.begin()+o,B.begin()+o+n);std::copy(tb.begin(),tb.end(),A.begin()+o);std::copy(ta.begin(),ta.end(),B.begin()+o);}}
        int last=p[G-1];for(int i=G-1;i>=2;--i)p[i]=p[i-1];p[1]=last;
    }
    if(pairs.size()!=G*(G-1)/2)return 8;

    // After transpose B[a,b] lives on GPU b, slot a, with identical intra-bucket address.
    marker=1;
    auto verify=[&](Mate m,bool blocked){auto s=split(m,blocked,ownerH,ownerL);auto&q=blocked?bl.blockb[s.oh][s.ol][s.he]:bl.mainb[s.oh][s.ol][3*s.he+s.c];uint32_t hr=hrank.at(key(s.he,s.hc)),lr=lrank.at(key(s.hs,s.lc));uint64_t addr=q.off+uint64_t(hr)*q.cols+lr;if(slots[s.ol][s.oh][addr]!=marker){std::cerr<<"transpose value mismatch marker="<<marker<<'\n';std::exit(9);}++marker;};
    for(Mate m:ms)verify(m,false);for(Mate m:bs)verify(m,true);

    uint32_t maxhr=0,maxlr=0;for(int g=0;g<G;++g){for(int h=0;h<H+2;++h)maxhr=std::max(maxhr,hcount[g][h]);for(int h=0;h<L+2;++h)maxlr=std::max(maxlr,lcount[g][h]);}
    uint64_t padded=0;for(int a=0;a<G;++a)for(int b=0;b<G;++b)padded+=std::max(bl.size[a][b],bl.size[b][a]);
    std::cout<<"gpu-direct-bucket-layout-selftest OK W="<<W
             <<" main="<<ms.size()<<" blocked="<<bs.size()<<" total="<<total
             <<" pairs="<<pairs.size()<<" max_owner_high_rows="<<maxhr
             <<" max_owner_low_cols="<<maxlr<<" padded_states="<<padded
             <<" padding_states="<<(padded-total)<<'\n';
    return 0;
}
