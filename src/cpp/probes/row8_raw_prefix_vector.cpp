#define main rowr_batch_reach_embedded_main
#include "rowr_column_batch_wfa_checkpoint.cpp"
#undef main
#include <tuple>

static std::uint32_t MODP = 4294967291u;
struct WEntry { Packed p; std::uint32_t v; bool operator==(WEntry const&o) const noexcept { return p==o.p && v==o.v; } };
using WVec = std::vector<WEntry>;
static bool wentry_less(WEntry const&a,WEntry const&b){return a.p<b.p;}
static void wnorm(WVec&v){
  __gnu_parallel::sort(v.begin(),v.end(),wentry_less);
  size_t z=0;
  for(size_t i=0;i<v.size();){size_t j=i+1;std::uint64_t s=v[i].v;while(j<v.size()&&v[j].p==v[i].p){s+=v[j].v;if(s>=MODP)s-=MODP;++j;}if(s){v[z++]={v[i].p,(std::uint32_t)s};}i=j;}
  v.resize(z);
}
static void wpush(WVec&v,State s,std::uint32_t w){if(w)v.push_back({pack(std::move(s)),w});}
static WVec wexpand(WVec const&cur,int r){WVec out;out.reserve(cur.size());for(auto const&e:cur){State old=unpack(e.p);State s{};s.n=2*r+1;s.ns=old.ns;s.sp=old.sp;s.status=old.status;s.stack=old.stack;for(int y=0;y<r;++y){s.deg[y]=old.deg[y];s.comp[y]=old.comp[y];}wpush(out,s,e.v);}wnorm(out);return out;}
static WVec wphysical(WVec const&cur,int i,int j,int maxi=2,int maxj=2){WVec out;out.reserve(cur.size()*2);for(auto const&e:cur){State s=unpack(e.p);wpush(out,s,e.v);State t=s;if(add_physical(t,i,j,maxi,maxj))wpush(out,t,e.v);}wnorm(out);return out;}
static WVec wforget(WVec const&cur,int slot,int r,bool source){WVec out;out.reserve(cur.size());for(auto const&e:cur){State s=unpack(e.p);if((source&&s.deg[slot]!=1)||(!source&&s.deg[slot]!=0&&s.deg[slot]!=2))continue;auto q=s.comp[slot];if(source){if(!q||s.sp>=r)continue;for(int j=s.sp;j>0;--j)s.stack[j]=s.stack[j-1];s.stack[0]=q;++s.sp;}s.deg[slot]=0;s.comp[slot]=0;if(q){bool alive=stack_has(s,q);if(!alive)for(int i=0;i<s.n;++i)if(s.comp[i]==q){alive=true;break;}if(!alive&&!(s.status[q]&2))continue;}bool ok=true;for(int qq=1;qq<s.ns;++qq)if((s.status[qq]&2)&&!closed_consistent(s,qq)){ok=false;break;}if(ok)wpush(out,s,e.v);}wnorm(out);return out;}
static WVec wfinish(WVec const&cur,int r,int sym){WVec out;out.reserve(cur.size());int bottom=2*r;for(auto const&e:cur){State s=unpack(e.p);auto q=s.comp[bottom];bool ok=true;if(sym==0){if(s.deg[bottom]!=0)continue;}else{if(s.deg[bottom]!=1||!q)continue;if(sym==2){if(s.sp>=r)continue;s.stack[s.sp++]=q;}else{if(!s.sp)continue;auto q2=s.stack[--s.sp];if(!merge_components(s,q,q2,true))continue;q=s.comp[bottom];}}s.deg[bottom]=0;s.comp[bottom]=0;State z{};z.n=r;z.ns=s.ns;z.sp=s.sp;z.status=s.status;z.stack=s.stack;for(int y=0;y<r;++y){z.deg[y]=s.deg[r+y];z.comp[y]=s.comp[r+y];}if(q){bool alive=stack_has(z,q);if(!alive)for(int y=0;y<r;++y)if(z.comp[y]==q){alive=true;break;}if(!alive&&!(z.status[q]&2))ok=false;}if(!ok)continue;for(int qq=1;qq<z.ns;++qq)if((z.status[qq]&2)&&!closed_consistent(z,qq)){ok=false;break;}if(ok)wpush(out,z,e.v);}wnorm(out);return out;}
static WVec wcolumn(WVec cur,int r,bool source,int sym){cur=wexpand(cur,r);for(int y=0;y<r;++y){cur=wphysical(cur,y,r+y,source&&y==0?1:2,2);if(y>0){cur=wphysical(cur,y-1,y,source&&y-1==0?1:2,2);cur=wforget(cur,y-1,r,source&&y-1==0);}}cur=wphysical(cur,r-1,2*r,2,2);cur=wforget(cur,r-1,r,false);return wfinish(cur,r,sym);}
static std::array<int,9> digits9(std::uint32_t code){std::array<int,9>d{};for(int i=8;i>=0;--i){d[i]=code%3;code/=3;}return d;}
static WVec prefix_vec(std::uint32_t code){State z{};z.n=8;WVec cur{{pack(z),1}};auto d=digits9(code);for(int c=0;c<9;++c)cur=wcolumn(std::move(cur),8,c==0,d[c]);return cur;}

static std::uint32_t wlast_value(WVec cur,int r,int sym){
  // Final physical column: no outgoing horizontals, only the r vertical edges.
  for(int y=0;y<r-1;++y){cur=wphysical(cur,y,y+1,2,2);cur=wforget(cur,y,r,false);}
  // Add bottom slot at index r before the final vertical edge.
  WVec ext;ext.reserve(cur.size());
  for(auto const&e:cur){State old=unpack(e.p);State z{};z.n=r+1;z.ns=old.ns;z.sp=old.sp;z.status=old.status;z.stack=old.stack;for(int y=0;y<r;++y){z.deg[y]=old.deg[y];z.comp[y]=old.comp[y];}wpush(ext,z,e.v);}wnorm(ext);cur.swap(ext);
  cur=wphysical(cur,r-1,r,2,2);cur=wforget(cur,r-1,r,false);
  std::uint64_t ans=0;
  for(auto const&e:cur){State z=unpack(e.p);int bottom=r;auto q=z.comp[bottom];bool ok=true;
    if(sym==0){if(z.deg[bottom]!=0)continue;}
    else {if(z.deg[bottom]!=1||!q)continue;if(sym==2){if(z.sp>=r)continue;z.stack[z.sp++]=q;}else{if(!z.sp)continue;auto q2=z.stack[--z.sp];if(!merge_components(z,q,q2,true))continue;q=z.comp[bottom];}}
    z.deg[bottom]=0;z.comp[bottom]=0;
    if(q){bool alive=stack_has(z,q);if(!alive)for(int i=0;i<z.n;++i)if(z.comp[i]==q){alive=true;break;}if(!alive&&!(z.status[q]&2))ok=false;}
    if(!ok||z.sp)continue;
    for(int i=0;i<r;++i)if(z.deg[i]||z.comp[i]){ok=false;break;}
    if(!ok)continue;
    ans+=e.v;if(ans>=MODP)ans-=MODP;
  }
  return (std::uint32_t)ans;
}
static std::array<int,10> digits10(std::uint32_t code){std::array<int,10>d{};for(int i=9;i>=0;--i){d[i]=code%3;code/=3;}return d;}
static std::uint32_t full_value(std::uint32_t pc,std::uint32_t sc){auto cur=prefix_vec(pc);auto d=digits10(sc);for(int c=0;c<9;++c)cur=wcolumn(std::move(cur),8,false,d[c]);return wlast_value(std::move(cur),8,d[9]);}
#ifndef ROW8_RAW_PREFIX_NO_MAIN
int main(int argc,char**argv){
  if(argc>=2)MODP=std::strtoul(argv[1],nullptr,10);
  if(argc>=4){std::uint32_t pc=std::strtoul(argv[2],nullptr,10),sc=std::strtoul(argv[3],nullptr,10);auto t=std::chrono::steady_clock::now();auto v=prefix_vec(pc);auto val=full_value(pc,sc);std::cout<<"mod="<<MODP<<" pc="<<pc<<" sc="<<sc<<" prefix_support="<<v.size()<<" value="<<val<<" sec="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()<<"\n";return 0;}
  std::uint32_t code=argc>2?std::strtoul(argv[2],nullptr,10):1;auto t=std::chrono::steady_clock::now();auto v=prefix_vec(code);std::uint64_t sum=0;for(auto const&e:v){sum+=e.v;if(sum>=MODP)sum-=MODP;}std::cout<<"mod="<<MODP<<" code="<<code<<" support="<<v.size()<<" sum="<<sum<<" sec="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()<<"\n";return 0;
}

#endif
