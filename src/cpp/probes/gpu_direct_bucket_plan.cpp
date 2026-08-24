#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

using U64=std::uint64_t;

template<int K>
static std::vector<std::array<U64,K+2>> half_counts(int start){
    std::vector<std::array<U64,K+2>> out(size_t(1u<<K));
    for(uint32_t mask=0;mask<(1u<<K);++mask){
        std::array<U64,K+2> dp{},nd{};dp[size_t(start)]=1;
        for(int pos=0;pos<K;++pos){
            nd.fill(0);bool occ=(mask>>pos)&1u;
            for(int h=0;h<K+1;++h)if(dp[size_t(h)]){
                U64 c=dp[size_t(h)];
                if(!occ)nd[size_t(h)]+=c;
                else{if(h>0)nd[size_t(h-1)]+=c;nd[size_t(h+1)]+=c;}
            }
            dp=nd;
        }
        out[size_t(mask)]=dp;
    }
    return out;
}

static std::vector<int> lpt_owner(const std::vector<U64>&w,std::array<U64,8>&loads){
    std::vector<std::pair<U64,uint32_t>> order;order.reserve(w.size());
    for(uint32_t m=0;m<w.size();++m)order.push_back({w[m],m});
    std::sort(order.begin(),order.end(),[](auto a,auto b){return a.first!=b.first?a.first>b.first:a.second<b.second;});
    std::vector<int> owner(w.size());loads.fill(0);
    for(auto [weight,m]:order){int g=0;for(int j=1;j<8;++j)if(loads[size_t(j)]<loads[size_t(g)])g=j;owner[m]=g;loads[size_t(g)]+=weight;}
    return owner;
}

int main(){
    constexpr int H=13,L=14;
    auto A=half_counts<H>(1);          // HIGH: height 1 -> h
    auto B=half_counts<L>(0);          // reversed LOW: 0 -> start height s
    std::array<U64,H+2> At{};std::array<U64,L+2> Bt{};
    for(auto const&x:A)for(int h=0;h<H+2;++h)At[size_t(h)]+=x[size_t(h)];
    for(auto const&x:B)for(int s=0;s<L+2;++s)Bt[size_t(s)]+=x[size_t(s)];
    auto btot=[&](int s)->U64{return s>=0&&s<L+2?Bt[size_t(s)]:0;};
    auto atot=[&](int h)->U64{return h>=0&&h<H+2?At[size_t(h)]:0;};

    std::vector<U64>wH(A.size()),wL(B.size());
    for(size_t m=0;m<A.size();++m)for(int h=0;h<H+2;++h)
        wH[m]+=A[m][size_t(h)]*(2*btot(h)+btot(h-1)+btot(h+1));
    for(size_t m=0;m<B.size();++m)for(int h=0;h<H+2;++h)
        wL[m]+=atot(h)*(2*B[m][size_t(h)]+(h?B[m][size_t(h-1)]:0)+B[m][size_t(h+1)]);

    std::array<U64,8>loadH{},loadL{};auto ownerH=lpt_owner(wH,loadH);auto ownerL=lpt_owner(wL,loadL);
    std::array<std::array<U64,H+2>,8> GA{};std::array<std::array<U64,L+2>,8> GB{};
    for(size_t m=0;m<A.size();++m)for(int h=0;h<H+2;++h)GA[size_t(ownerH[m])][size_t(h)]+=A[m][size_t(h)];
    for(size_t m=0;m<B.size();++m)for(int h=0;h<L+2;++h)GB[size_t(ownerL[m])][size_t(h)]+=B[m][size_t(h)];
    auto gb=[&](int g,int h)->U64{return h>=0&&h<L+2?GB[size_t(g)][size_t(h)]:0;};

    U64 max_owner_high_rows=0,max_owner_low_cols=0;
    for(int g=0;g<8;++g){
        for(int h=0;h<H+2;++h)max_owner_high_rows=std::max(max_owner_high_rows,GA[size_t(g)][size_t(h)]);
        for(int h=0;h<L+2;++h)max_owner_low_cols=std::max(max_owner_low_cols,GB[size_t(g)][size_t(h)]);
    }

    std::array<std::array<U64,8>,8>S{};U64 total=0,diag=0;
    for(int a=0;a<8;++a)for(int b=0;b<8;++b){
        U64 z=0;for(int h=0;h<H+2;++h)z+=GA[size_t(a)][size_t(h)]*(2*gb(b,h)+gb(b,h-1)+gb(b,h+1));
        S[size_t(a)][size_t(b)]=z;total+=z;if(a==b)diag+=z;
    }
    std::array<U64,8>row{},cap{},over{};U64 mn=~U64(0),mx=0;
    for(int a=0;a<8;++a){for(int b=0;b<8;++b){row[size_t(a)]+=S[size_t(a)][size_t(b)];cap[size_t(a)]+=std::max(S[size_t(a)][size_t(b)],S[size_t(b)][size_t(a)]);mn=std::min(mn,S[size_t(a)][size_t(b)]);mx=std::max(mx,S[size_t(a)][size_t(b)]);}over[size_t(a)]=(cap[size_t(a)]-row[size_t(a)])*4;}
    U64 remote=total-diag;U64 remote_bytes=remote*4;U64 residue_bytes=remote_bytes*55;

    std::cout<<std::setprecision(15)
        <<"main_plus_blocked_states="<<total<<'\n'
        <<"diag_states="<<diag<<'\n'
        <<"remote_states_per_transpose="<<remote<<'\n'
        <<"remote_bytes_per_transpose="<<remote_bytes<<'\n'
        <<"remote_tib_per_residue="<<double(residue_bytes)/double(1ULL<<40)<<'\n'
        <<"local_fraction="<<double(diag)/double(total)<<'\n'
        <<"bucket_min_states="<<mn<<" bucket_max_states="<<mx<<'\n'
        <<"max_owner_high_rows="<<max_owner_high_rows<<'\n'
        <<"max_owner_low_cols="<<max_owner_low_cols<<'\n'
        <<"locator_bits=19\n";
    U64 max_over=0;
    for(int g=0;g<8;++g){max_over=std::max(max_over,over[size_t(g)]);std::cout<<"gpu"<<g<<"_states="<<row[size_t(g)]<<" gpu"<<g<<"_gib="<<double(row[size_t(g)]*4)/double(1ULL<<30)<<" gpu"<<g<<"_slot_overhead_bytes="<<over[size_t(g)]<<'\n';}
    std::cout<<"max_slot_overhead_bytes="<<max_over<<'\n';
    return total==520735012027ULL
        &&remote==455642447434ULL
        &&max_over==2154132ULL
        &&max_owner_high_rows==19631ULL
        &&max_owner_low_cols==30114ULL?0:2;
}
