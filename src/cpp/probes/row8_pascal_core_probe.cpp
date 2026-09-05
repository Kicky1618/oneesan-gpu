#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

static constexpr int R=8;

static void gen_rec(int pos,int needT,int bal,std::string& w,std::vector<std::string>& out){
    if(pos==R){ if(needT==0 && bal==0) out.push_back(w); return; }
    int rem=R-pos; if(needT>rem) return;
    w[pos]='N'; gen_rec(pos+1,needT,bal,w,out);
    w[pos]='U'; gen_rec(pos+1,needT,bal+1,w,out);
    w[pos]='D'; gen_rec(pos+1,needT,bal-1,w,out);
    if(needT>0 && bal==0){ w[pos]='T'; gen_rec(pos+1,needT-1,0,w,out); }
}
static std::vector<std::string> gen_words(int h){ std::string w(R,'N'); std::vector<std::string> o; gen_rec(0,h,0,w,o); return o; }

static uint32_t mask_of(std::string const&w){uint32_t m=0; for(int i=0;i<R;++i) if(w[i]!='N') m|=1u<<i; return m;}
static std::string erase_N(std::string const&w){std::string s;for(char c:w)if(c!='N')s+=c;return s;}

static uint64_t binom(int n,int k){if(k<0||k>n)return 0;uint64_t z=1;for(int i=1;i<=k;++i)z=z*(n-k+i)/i;return z;}
// A111959(j,h) = [x^(j-h)] (1-4x^2)^(-(h+1)/2).
// For d=(j-h)/2, coefficient = binom(h+2d,d) when parity matches.
// Verified by generalized binomial identity: 4^d * ((h+1)/2)_d / d! = C(h+2d,d).
static uint64_t a111959(int j,int h){if(j<h||((j-h)&1))return 0;int d=(j-h)/2;uint64_t c=1;for(int t=0;t<d;++t)c=c*2u*(h+1+2*t)/(t+1);return c;}

int main(){
    bool all=true;
    uint64_t grand=0;
    for(int h=0;h<=R;++h){
        auto ws=gen_words(h); grand+=ws.size();
        std::map<int,std::map<uint32_t,std::set<std::string>>> byj;
        for(auto const&w:ws){auto m=mask_of(w);int j=__builtin_popcount(m);byj[j][m].insert(erase_N(w));}
        uint64_t sum=0;
        for(int j=h;j<=R;++j){
            uint64_t c=a111959(j,h); if(!c) continue;
            auto it=byj.find(j); size_t nm=it==byj.end()?0:it->second.size();
            size_t minc=(size_t)-1,maxc=0; bool same=true; std::set<std::string> ref; bool first=true;
            if(it!=byj.end()) for(auto const&kv:it->second){
                auto const&s=kv.second; minc=std::min(minc,s.size());maxc=std::max(maxc,s.size());
                if(first){ref=s;first=false;} else if(s!=ref) same=false;
                sum+=s.size();
            }
            uint64_t wantMasks=binom(R,j); uint64_t want=wantMasks*c;
            bool ok=nm==wantMasks && minc==c && maxc==c && same;
            all &= ok;
            std::cout<<"h="<<h<<" j="<<j<<" masks="<<nm<<"/"<<wantMasks
                     <<" core="<<(minc==(size_t)-1?0:minc)<<".."<<maxc<<" A111959="<<c
                     <<" identical="<<same<<" block="<<want<<" exact="<<ok<<"\n";
        }
        bool hok=sum==ws.size(); all&=hok;
        std::cout<<"HEIGHT h="<<h<<" pascal_sum="<<sum<<" total="<<ws.size()<<" exact="<<hok<<"\n";
    }
    std::cout<<"TOTAL="<<grand<<" pascal_core_exact="<<all<<"\n";
    return all?0:1;
}
