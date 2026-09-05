#include "../../common/gridfp_transition.hpp"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace oneesan::gridfp;

static MateID enc(const std::string& s) {
    MateID m=0; const int W=(int)s.size();
    for(int i=0;i<W;++i){ auto v=s[i]=='N'?N:s[i]=='R'?R:L; m=mset(m,W-1-i,v); }
    return m;
}
static std::string dec(MateID m,int W){std::string s;s.reserve(W);for(int p=W-1;p>=0;--p){auto v=mget(m,p);s+=v==N?'N':v==R?'R':v==L?'L':'X';}return s;}

struct Ref { bool valid=false, blocked=false; std::string out; };
static Ref reference(std::string s,int p){
    const int W=(int)s.size(); const int a=W-1-p, b=a+1;
    char x=s[a], y=s[b]; Ref z;
    auto done=[&](bool blocked){z.valid=true;z.blocked=blocked;z.out=s;return z;};
    if(x=='N'&&y=='N'){s[a]='L';s[b]='R';return done(false);}
    if(x=='R'&&y=='N'){s[a]='N';s[b]='R';return done(false);}
    if(x=='L'&&y=='N'){s[a]='N';s[b]='L';return done(false);}
    if(x=='N'&&y=='R'){s[a]='R';s[b]='N';return done(p>1);}
    if(x=='N'&&y=='L'){s[a]='L';s[b]='N';return done(p>1);}
    if(x=='R'&&y=='L'){s[a]='N';s[b]='N';return done(p>1);}
    if(x=='L'&&y=='L'){
        // Production scans to the right in displayed order for the R matching
        // the second open L; nested L/R pairs adjust the depth.
        int depth=1, q=b+1;
        for(;q<W && depth;++q){ if(s[q]=='L')++depth; else if(s[q]=='R')--depth; }
        if(depth) return z;
        --q; s[a]=s[b]='N'; s[q]='L'; return done(p>1);
    }
    if(x=='R'&&y=='R'){
        // Symmetric matching scan to the left in displayed order.
        int depth=1, q=a-1;
        for(;q>=0 && depth;--q){ if(s[q]=='L')--depth; else if(s[q]=='R')++depth; }
        if(depth) return z;
        ++q; s[a]=s[b]='N'; s[q]='R'; return done(p>1);
    }
    return z; // LR closes locally and is intentionally invalid here.
}

static std::string ternary(std::uint64_t code,int W){std::string s(W,'N');for(int i=W-1;i>=0;--i){int d=code%3;code/=3;s[i]=d==0?'N':d==1?'R':'L';}return s;}

int main(int argc,char**argv){
    int maxW=argc>1?std::stoi(argv[1]):12;
    if(maxW<2||maxW>16)return 2;
    std::uint64_t checked=0,bad=0,valid=0,blocked=0;
    for(int W=2;W<=maxW;++W){
        std::uint64_t n=1;for(int i=0;i<W;++i)n*=3;
        for(std::uint64_t code=0;code<n;++code){
            auto s=ternary(code,W); MateID m=enc(s);
            for(int p=1;p<W;++p){
                auto ref=reference(s,p); auto got=include_horizontal(m,W,p);
                ++checked; valid+=got.valid; blocked+=got.valid&&got.blocked;
                bool ok=(got.valid==ref.valid);
                if(ok&&got.valid){
                    ok=(got.blocked==ref.blocked);
                    if(ok){MateID out=got.mate;int ow=W-(got.blocked?1:0);if(got.blocked){out=blocked_exclude(out,p-1);ow=W;}ok=dec(out,ow)==ref.out;}
                }
                if(!ok){if(bad<12){std::cerr<<"bad W="<<W<<" p="<<p<<" src="<<s<<" got_valid="<<got.valid<<" got_blocked="<<got.blocked<<" ref_valid="<<ref.valid<<" ref_blocked="<<ref.blocked<<" ref_out="<<ref.out;if(got.valid){MateID out=got.mate;int ow=W-(got.blocked?1:0);if(got.blocked){out=blocked_exclude(out,p-1);ow=W;}std::cerr<<" got_out="<<dec(out,ow);}std::cerr<<"\n";}++bad;}
            }
        }
    }
    std::cout<<"gridfp_transition_exhaustive_abi maxW="<<maxW<<" checked="<<checked<<" valid="<<valid<<" blocked="<<blocked<<" bad="<<bad<<" exact="<<(bad==0)<<"\n";
    return bad?1:0;
}
