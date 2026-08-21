#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include <iostream>
#include <vector>
#include <set>

static void clearpc(PathCounter<uint64_t>& pc){
    for(Code i=0;i<pc.mc.codeSize();++i) pc.value[i]=0;
    for(Code i=0;i<pc.wc.codeSize();++i) pc.deferred[i]=0;
}


static Mate remove_ours(Mate m, int k){
    MateID x=m.id();
    MateID lo=k?(x&((1ULL<<(2*k))-1ULL)):0ULL;
    MateID hi=x&~((1ULL<<(2*(k+1)))-1ULL);
    return Mate(lo|(hi>>2));
}

static Mate insert_symbol(Mate sm, int k, MateValue v){
    MateID m=sm.id();
    MateID lowmask=k?((MateID(1)<<(2*k))-1):0;
    MateID lo=m&lowmask, hi=m&~lowmask;
    return Mate(lo | (MateID(v)<<(2*k)) | (hi<<2));
}

static void formula_main(PathCounter<uint64_t>& pc, Mate m, Code src, int p,
                         std::set<std::pair<char,Code>>& out){
    out.insert({'M',src}); // line excluded
    auto w=m.getPair(p);
    switch(w){
    case NN: {
        m.setPair(p,LR); out.insert({'M',pc.mc.encode(m)}); break;
    }
    case NR: case NL: {
        if(p==1){ m.setPair(p,w==NR?RN:LN); out.insert({'M',pc.mc.encode(m)}); }
        else out.insert({'D',pc.wc.encode(remove_ours(m,p))});
        break;
    }
    case RN: case LN: {
        m.setPair(p,w==RN?NR:NL); out.insert({'M',pc.mc.encode(m)}); break;
    }
    case LL: {
        m.setPair(p,NN); int q=p-1,s=1;
        while(s>0){--q; switch(m.get(q)){case L:++s;break;case R:--s;break;default:break;}}
        m.set(q,L);
        if(p==1) out.insert({'M',pc.mc.encode(m)}); else out.insert({'D',pc.wc.encode(remove_ours(m,p-1))});
        break;
    }
    case RR: {
        m.setPair(p,NN); int q=p,s=1;
        while(s>0){++q; switch(m.get(q)){case L:--s;break;case R:++s;break;default:break;}}
        m.set(q,R);
        if(p==1) out.insert({'M',pc.mc.encode(m)}); else out.insert({'D',pc.wc.encode(remove_ours(m,p-1))});
        break;
    }
    case RL: {
        m.setPair(p,NN);
        if(p==1) out.insert({'M',pc.mc.encode(m)}); else out.insert({'D',pc.wc.encode(remove_ours(m,p-1))});
        break;
    }
    default: break;
    }
}

int main(){
    msg=NONE; modulus=0;
    for(int width=3;width<=8;++width){
      PathCounter<uint64_t> pc(width,width,false,false);
      for(int p=1;p<width;++p){
        int j=width-p-1; int bad=0;
        for(Code bi=0;bi<pc.mc.codeSizeL();++bi){auto const& b=pc.mc.codeTable(bi);for(Code k=0;k<b.size;++k){
          Mate m=b.mateL|b.mateR[k]; Code src=b.base+k;
          clearpc(pc);pc.value[src]=1;pc.update(j,false);
          std::set<std::pair<char,Code>> want;
          for(Code x=0;x<pc.mc.codeSize();++x) if(pc.value[x]) want.insert({'M',x});
          for(Code x=0;x<pc.wc.codeSize();++x) if(pc.deferred[x]) want.insert({'D',x});
          std::set<std::pair<char,Code>> got; formula_main(pc,m,src,p,got);
          if(got!=want){
            if(bad++<10){std::cerr<<"BAD width="<<width<<" p="<<p<<" src="<<src<<" mate="<<m<<" pair="<<m.getPair(p)<<"\n got:";for(auto z:got)std::cerr<<' '<<z.first<<z.second;std::cerr<<"\n want:";for(auto z:want)std::cerr<<' '<<z.first<<z.second;std::cerr<<"\n";}
          }
        }}
        // blocked source transform check by candidate insertion N at p
        for(Code src=0;src<pc.wc.codeSize();++src){
          clearpc(pc);pc.deferred[src]=1;pc.update(j,false);
          std::set<Code> want;for(Code x=0;x<pc.mc.codeSize();++x)if(pc.value[x])want.insert(x);
          Mate sm; // recover blocked state by scanning table
          bool found=false;
          for(Code bi=0;bi<pc.wc.codeSizeL()&&!found;++bi){auto const& b=pc.wc.codeTable(bi);for(Code k=0;k<b.size;++k)if(b.base+k==src){sm=b.mateL|b.mateR[k];found=true;break;}}
          Mate t=insert_symbol(sm,p,N); std::set<Code> got={pc.mc.encode(t)};
          if(got!=want){if(bad++<10){std::cerr<<"BAD-B width="<<width<<" p="<<p<<" src=D"<<src<<" sm="<<sm<<" got M"<<*got.begin()<<" want";for(auto x:want)std::cerr<<" M"<<x;std::cerr<<"\n";}}
        }
        std::cout<<"width="<<width<<" p="<<p<<" bad="<<bad<<"\n";
      }
    }
}
