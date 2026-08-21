#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using Code = unsigned long long;
using MateID = unsigned long long;
static constexpr int MAXW=28;
enum MateValue:uint8_t{N=0,R=1,L=2,X=3};
enum MateValuePair:uint8_t{NN=0x0,NR=0x1,NL=0x2,NX=0x3,RN=0x4,RR=0x5,RL=0x6,RX=0x7,LN=0x8,LR=0x9,LL=0xa,LX=0xb,XN=0xc,XR=0xd,XL=0xe,XX=0xf};
static Code DP[MAXW+1][MAXW+2];

static inline MateValue mget(MateID m,int k){return MateValue((m>>(2*k))&3ULL);}
static inline MateValuePair mpair(MateID m,int p){return MateValuePair((m>>(2*(p-1)))&15ULL);}
static inline MateID mset(MateID m,int k,MateValue v){MateID z=3ULL<<(2*k);return (m&~z)|(MateID(v)<<(2*k));}
static inline MateID msetpair(MateID m,int p,MateValuePair v){MateID z=15ULL<<(2*(p-1));return (m&~z)|(MateID(v)<<(2*(p-1)));}
static inline MateID mshrink(MateID m,int k){MateID mask=(1ULL<<(2*k))-1ULL;return((m&~mask)>>2)|(m&mask);}
static inline MateID minsert(MateID m,int k,MateValue v){MateID lowmask=k?((1ULL<<(2*k))-1ULL):0ULL;MateID lo=m&lowmask,hi=m&~lowmask;return lo|(MateID(v)<<(2*k))|(hi<<2);}

static void build_dp(){
  for(int h=0;h<=MAXW+1;++h)DP[0][h]=(h==0);
  for(int w=1;w<=MAXW;++w)for(int h=0;h<=MAXW;++h){
    Code x=DP[w-1][h]; if(h>0)x+=DP[w-1][h-1]; if(h<MAXW+1)x+=DP[w-1][h+1]; DP[w][h]=x;
  }
}
static MateID unrank_full(Code rank,int width){
  MateID m=0; int h=1;
  for(int pos=width-1;pos>=0;--pos){
    Code z=DP[pos][h]; if(rank<z) continue; rank-=z;
    if(h>0){z=DP[pos][h-1]; if(rank<z){m|=MateID(R)<<(2*pos); --h; continue;} rank-=z;}
    m|=MateID(L)<<(2*pos); ++h;
  }
  return m;
}
static Code rank_full(MateID m,int width){
  Code rank=0; int h=1;
  for(int pos=width-1;pos>=0;--pos){
    MateValue s=mget(m,pos);
    if(s>N) rank+=DP[pos][h];
    if(s>R&&h>0) rank+=DP[pos][h-1];
    if(s==R)--h; else if(s==L)++h;
  }
  return rank;
}

int main(int argc,char**argv){
  build_dp(); int W=argc>1?std::atoi(argv[1]):15;
  Code mn=DP[W][1], dn=DP[W-1][1];
  std::vector<MateID> main_state(mn), block_state(dn);
  for(Code i=0;i<mn;++i)main_state[i]=unrank_full(i,W);
  for(Code i=0;i<dn;++i)block_state[i]=unrank_full(i,W-1);
  std::cout<<"W="<<W<<" main="<<mn<<" block="<<dn<<"\n";
  unsigned global_main=0, global_block=0;
  for(int p=W-1;p>=1;--p){
    std::vector<uint16_t> cm(mn,1), cb(dn,0); // identity contribution to main
    for(Code i=0;i<dn;++i){
      MateID t=minsert(block_state[i],p,N);
      auto j=rank_full(t,W); if(j>=mn){std::cerr<<"bad block->main\n"; return 2;} ++cm[j];
    }
    for(Code i=0;i<mn;++i){
      MateID m=main_state[i], t=0; MateValuePair w=mpair(m,p);
      bool to_main=false,to_block=false;
      switch(w){
        case NN: t=msetpair(m,p,LR); to_main=true; break;
        case NR: case NL:
          if(p==1){t=msetpair(m,p,w==NR?RN:LN);to_main=true;}
          else {t=mshrink(m,p);to_block=true;} break;
        case RN: t=msetpair(m,p,NR);to_main=true;break;
        case LN: t=msetpair(m,p,NL);to_main=true;break;
        case LL:{t=msetpair(m,p,NN);int q=p-1,s=1;while(s){--q;auto v=mget(t,q);if(v==L)++s;else if(v==R)--s;}t=mset(t,q,L);if(p==1)to_main=true;else{t=mshrink(t,p-1);to_block=true;}break;}
        case RR:{t=msetpair(m,p,NN);int q=p,s=1;while(s){++q;auto v=mget(t,q);if(v==L)--s;else if(v==R)++s;}t=mset(t,q,R);if(p==1)to_main=true;else{t=mshrink(t,p-1);to_block=true;}break;}
        case RL:{t=msetpair(m,p,NN);if(p==1)to_main=true;else{t=mshrink(t,p-1);to_block=true;}break;}
        default: break;
      }
      if(to_main){auto j=rank_full(t,W); if(j>=mn){std::cerr<<"bad main\n";return 3;} ++cm[j];}
      if(to_block){auto j=rank_full(t,W-1); if(j>=dn){std::cerr<<"bad block\n";return 4;} ++cb[j];}
    }
    unsigned mm=*std::max_element(cm.begin(),cm.end());
    unsigned md=*std::max_element(cb.begin(),cb.end());
    global_main=std::max(global_main,mm); global_block=std::max(global_block,md);
    std::cout<<"p="<<p<<" max_main_terms="<<mm<<" max_block_terms="<<md<<"\n";
  }
  std::cout<<"global max_main_terms="<<global_main<<" max_block_terms="<<global_block<<"\n";
}
