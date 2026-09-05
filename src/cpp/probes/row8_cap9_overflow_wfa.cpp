#include <algorithm>
#include <parallel/algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <iterator>
#include <omp.h>
#include <stdexcept>
#include <vector>

static constexpr int MAXR=8, MAXSTACK=9, MAXV=2*MAXR+1, MAXC=MAXV+2;

struct State{
  std::array<std::uint8_t,MAXV>deg{},comp{};
  std::array<std::uint8_t,MAXC>status{};
  std::array<std::uint8_t,MAXSTACK>stack{};
  std::uint8_t n=0,ns=1,sp=0,overflow=0;
};
struct Packed{
  std::array<std::uint64_t,4>w{};
  bool operator==(Packed const&o)const noexcept{return w==o.w;}
  bool operator<(Packed const&o)const noexcept{return w<o.w;}
};
static bool stack_has(State const&s,std::uint8_t q){for(int i=0;i<s.sp;++i)if(s.stack[i]==q)return true;return false;}
static void canon(State&s){
  std::array<std::uint8_t,MAXC>rm{},st{};std::uint8_t nx=1;
  auto take=[&](std::uint8_t c){if(c&&!rm[c]){rm[c]=nx;st[nx]=s.status[c];++nx;}};
  for(int i=0;i<s.n;++i)take(s.comp[i]);for(int i=0;i<s.sp;++i)take(s.stack[i]);
  for(int i=0;i<s.n;++i)if(s.comp[i])s.comp[i]=rm[s.comp[i]];for(int i=0;i<s.sp;++i)if(s.stack[i])s.stack[i]=rm[s.stack[i]];
  s.status.fill(0);for(int q=1;q<nx;++q)s.status[q]=st[q];s.ns=nx;
  for(int i=s.n;i<MAXV;++i){s.deg[i]=0;s.comp[i]=0;}for(int i=s.sp;i<MAXSTACK;++i)s.stack[i]=0;
}
static void putbits(Packed&p,int&bit,std::uint32_t v,int n){int q=bit>>6,o=bit&63;p.w[q]|=std::uint64_t(v)<<o;if(o+n>64)p.w[q+1]|=std::uint64_t(v)>>(64-o);bit+=n;}
static std::uint32_t getbits(Packed const&p,int&bit,int n){int q=bit>>6,o=bit&63;std::uint64_t z=p.w[q]>>o;if(o+n>64)z|=p.w[q+1]<<(64-o);bit+=n;return std::uint32_t(z&((std::uint64_t(1)<<n)-1));}
static Packed pack(State x){canon(x);Packed p;int bit=0;putbits(p,bit,x.n,5);putbits(p,bit,x.ns,5);putbits(p,bit,x.sp,4);putbits(p,bit,x.overflow,1);for(int i=0;i<x.n;++i)putbits(p,bit,x.deg[i],2);for(int i=0;i<x.n;++i)putbits(p,bit,x.comp[i],5);for(int i=0;i<x.sp;++i)putbits(p,bit,x.stack[i],5);for(int q=1;q<x.ns;++q)putbits(p,bit,x.status[q],2);if(bit>256)throw std::runtime_error("pack overflow");return p;}
static State unpack(Packed const&p){State x{};int bit=0;x.n=getbits(p,bit,5);x.ns=getbits(p,bit,5);x.sp=getbits(p,bit,4);x.overflow=getbits(p,bit,1);if(x.n>MAXV||x.ns>MAXC||x.sp>MAXSTACK)throw std::runtime_error("bad pack");for(int i=0;i<x.n;++i)x.deg[i]=getbits(p,bit,2);for(int i=0;i<x.n;++i)x.comp[i]=getbits(p,bit,5);for(int i=0;i<x.sp;++i)x.stack[i]=getbits(p,bit,5);for(int q=1;q<x.ns;++q)x.status[q]=getbits(p,bit,2);return x;}
using Vec=std::vector<Packed>;
static void norm(Vec&v){std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());}
static bool closed_consistent(State const&s,std::uint8_t q){if(!(s.status[q]&2))return true;if(stack_has(s,q))return false;for(int i=0;i<s.n;++i)if(s.comp[i]==q&&s.deg[i]!=2)return false;return true;}
static bool merge_components(State&s,std::uint8_t a,std::uint8_t b,bool virt){
 if(!a||!b||(s.status[a]&2)||(s.status[b]&2))return false;
 if(a==b){if(virt){if(s.status[a]&1)return false;s.status[a]|=3;return true;}if(!(s.status[a]&1))return false;s.status[a]|=2;return true;}
 int vc=(s.status[a]&1?1:0)+(s.status[b]&1?1:0)+(virt?1:0);if(vc>1)return false;auto keep=std::min(a,b),kill=std::max(a,b);for(int i=0;i<s.n;++i)if(s.comp[i]==kill)s.comp[i]=keep;for(int i=0;i<s.sp;++i)if(s.stack[i]==kill)s.stack[i]=keep;s.status[keep]=vc?1:0;s.status[kill]=0;return true;
}
static bool add_physical(State&s,int i,int j,int maxi=2,int maxj=2){
 if(s.deg[i]>=maxi||s.deg[j]>=maxj)return false;auto a=s.comp[i],b=s.comp[j];if((a&&(s.status[a]&2))||(b&&(s.status[b]&2)))return false;++s.deg[i];++s.deg[j];
 if(!a&&!b){if(s.ns>=MAXC)return false;auto q=s.ns++;s.status[q]=0;s.comp[i]=s.comp[j]=q;return true;}if(!a||!b){auto q=a?a:b;s.comp[i]=s.comp[j]=q;return true;}return merge_components(s,a,b,false);
}
static void append(Vec&dst,State s){dst.push_back(pack(std::move(s)));}

static Vec physical_batch(Vec const&cur,int i,int j,int maxi=2,int maxj=2){
 int nt=std::max(1,omp_get_max_threads());std::vector<Vec>loc(nt);
#pragma omp parallel
 {
  int tid=omp_get_thread_num();auto&v=loc[tid];v.reserve(cur.size()*2/nt+64);
#pragma omp for schedule(static)
  for(long long k=0;k<(long long)cur.size();++k){State s=unpack(cur[(size_t)k]);append(v,s);State t=s;if(add_physical(t,i,j,maxi,maxj))append(v,t);}
 }
 Vec out;size_t z=0;for(auto const&v:loc)z+=v.size();out.reserve(z);for(auto&v:loc)out.insert(out.end(),std::make_move_iterator(v.begin()),std::make_move_iterator(v.end()));norm(out);return out;
}

static Vec expand_boundary(Vec const&cur,int r){
 int nt=std::max(1,omp_get_max_threads());std::vector<Vec>loc(nt);
#pragma omp parallel
 {
  int tid=omp_get_thread_num();auto&v=loc[tid];v.reserve(cur.size()/nt+64);
#pragma omp for schedule(static)
  for(long long k=0;k<(long long)cur.size();++k){State old=unpack(cur[(size_t)k]);if(old.n!=r)continue;State s{};s.n=2*r+1;s.ns=old.ns;s.sp=old.sp;s.overflow=old.overflow;s.status=old.status;s.stack=old.stack;for(int y=0;y<r;++y){s.deg[y]=old.deg[y];s.comp[y]=old.comp[y];}append(v,s);}
 }
 Vec out;size_t z=0;for(auto const&v:loc)z+=v.size();out.reserve(z);for(auto&v:loc)out.insert(out.end(),std::make_move_iterator(v.begin()),std::make_move_iterator(v.end()));norm(out);return out;
}

static Vec forget_slot(Vec const&cur,int slot,int r,bool source){
 int nt=std::max(1,omp_get_max_threads());std::vector<Vec>loc(nt);
#pragma omp parallel
 {
  int tid=omp_get_thread_num();auto&v=loc[tid];v.reserve(cur.size()/nt+64);
#pragma omp for schedule(static)
  for(long long k=0;k<(long long)cur.size();++k){State s=unpack(cur[(size_t)k]);
   if((source&&s.deg[slot]!=1)||(!source&&s.deg[slot]!=0&&s.deg[slot]!=2))continue;
   auto q=s.comp[slot];if(source){if(!q||s.sp>=MAXSTACK)continue;for(int j=s.sp;j>0;--j)s.stack[j]=s.stack[j-1];s.stack[0]=q;++s.sp;if(s.sp>MAXR)s.overflow=1;}
   s.deg[slot]=0;s.comp[slot]=0;
   if(q){bool alive=stack_has(s,q);if(!alive)for(int i=0;i<s.n;++i)if(s.comp[i]==q){alive=true;break;}if(!alive&&!(s.status[q]&2))continue;}
   bool ok=true;for(int qq=1;qq<s.ns;++qq)if((s.status[qq]&2)&&!closed_consistent(s,qq)){ok=false;break;}if(ok)append(v,s);
  }
 }
 Vec out;size_t z=0;for(auto const&v:loc)z+=v.size();out.reserve(z);for(auto&v:loc)out.insert(out.end(),std::make_move_iterator(v.begin()),std::make_move_iterator(v.end()));norm(out);return out;
}

static Vec finish_column(Vec const&cur,int r){
 int nt=std::max(1,omp_get_max_threads());std::vector<Vec>loc(nt);int bottom=2*r;
#pragma omp parallel
 {
  int tid=omp_get_thread_num();auto&v=loc[tid];v.reserve(cur.size()*2/nt+64);
#pragma omp for schedule(static)
  for(long long k=0;k<(long long)cur.size();++k){
   State base=unpack(cur[(size_t)k]);
   for(int sym=0;sym<3;++sym){State s=base;auto q=s.comp[bottom];bool ok=true;
    if(sym==0){if(s.deg[bottom]!=0)continue;}else{if(s.deg[bottom]!=1||!q)continue;if(sym==2){if(s.sp>=MAXSTACK)continue;s.stack[s.sp++]=q;if(s.sp>MAXR)s.overflow=1;}else{if(!s.sp)continue;auto q2=s.stack[--s.sp];if(!merge_components(s,q,q2,true))continue;q=s.comp[bottom];}}
    s.deg[bottom]=0;s.comp[bottom]=0;
    State out{};out.n=r;out.ns=s.ns;out.sp=s.sp;out.overflow=s.overflow;out.status=s.status;out.stack=s.stack;for(int y=0;y<r;++y){out.deg[y]=s.deg[r+y];out.comp[y]=s.comp[r+y];}
    if(q){bool alive=stack_has(out,q);if(!alive)for(int y=0;y<r;++y)if(out.comp[y]==q){alive=true;break;}if(!alive&&!(out.status[q]&2))ok=false;}
    if(!ok)continue;for(int qq=1;qq<out.ns;++qq)if((out.status[qq]&2)&&!closed_consistent(out,qq)){ok=false;break;}if(ok)append(v,out);
   }
  }
 }
 Vec out;size_t z=0;for(auto const&v:loc)z+=v.size();out.reserve(z);for(auto&v:loc)out.insert(out.end(),std::make_move_iterator(v.begin()),std::make_move_iterator(v.end()));norm(out);return out;
}

static Vec column_step(Vec const&boundary,int r,bool source){
 bool prof=std::getenv("ROWR_PROFILE")!=nullptr;
 auto secs=[](auto t0){return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
 auto t=std::chrono::steady_clock::now();Vec x=expand_boundary(boundary,r);if(prof)std::cerr<<"stage expand states="<<x.size()<<" s="<<secs(t)<<"\n";
 // Interleave outgoing horizontals with the old-column vertical edge whose
 // lower endpoint has just become available.  Then forget the upper old
 // vertex immediately after its last incident edge.  This is the same graph
 // as the all-horizontals-then-verticals schedule, but keeps the intermediate
 // frontier small.
 for(int y=0;y<r;++y){
   t=std::chrono::steady_clock::now();x=physical_batch(x,y,r+y,source&&y==0?1:2,2);if(prof)std::cerr<<"stage horiz y="<<y<<" states="<<x.size()<<" s="<<secs(t)<<"\n";
   if(y>0){
     t=std::chrono::steady_clock::now();x=physical_batch(x,y-1,y,source&&y-1==0?1:2,2);if(prof)std::cerr<<"stage vert y="<<(y-1)<<" states="<<x.size()<<" s="<<secs(t)<<"\n";
     t=std::chrono::steady_clock::now();x=forget_slot(x,y-1,r,source&&y-1==0);if(prof)std::cerr<<"stage forget y="<<(y-1)<<" states="<<x.size()<<" s="<<secs(t)<<"\n";
   }
 }
 t=std::chrono::steady_clock::now();x=physical_batch(x,r-1,2*r,2,2);if(prof)std::cerr<<"stage vert-last states="<<x.size()<<" s="<<secs(t)<<"\n";
 t=std::chrono::steady_clock::now();x=forget_slot(x,r-1,r,false);if(prof)std::cerr<<"stage forget-last states="<<x.size()<<" s="<<secs(t)<<"\n";
 t=std::chrono::steady_clock::now();auto out=finish_column(x,r);if(prof)std::cerr<<"stage finish states="<<out.size()<<" s="<<secs(t)<<"\n";return out;
}

static std::uint64_t payload_hash(Vec const&v){std::uint64_t h=1469598103934665603ULL;for(auto const&p:v)for(auto x:p.w)for(int b=0;b<8;++b){h^=std::uint8_t(x>>(8*b));h*=1099511628211ULL;}return h;}
struct CkHdr{char magic[8];std::uint32_t ver,r,col,reserved;std::uint64_t n,hash;};
static bool load_ck(std::string const&path,int r,int&col,Vec&v){std::ifstream in(path,std::ios::binary);if(!in)return false;CkHdr h{};in.read((char*)&h,sizeof(h));if(!in||std::string(h.magic,7)!="R9OVERF"||h.ver!=2||h.r!=(std::uint32_t)r)throw std::runtime_error("bad row-WFA checkpoint header");v.resize(h.n);if(h.n)in.read((char*)v.data(),h.n*sizeof(Packed));if(!in||payload_hash(v)!=h.hash)throw std::runtime_error("bad row-WFA checkpoint payload");if(!std::is_sorted(v.begin(),v.end())||std::adjacent_find(v.begin(),v.end())!=v.end())throw std::runtime_error("row-WFA checkpoint not normalized");col=h.col;return true;}
static void save_ck(std::string const&path,int r,int col,Vec const&v){CkHdr h{{'R','9','O','V','E','R','F','\0'},2u,(std::uint32_t)r,(std::uint32_t)col,0u,(std::uint64_t)v.size(),payload_hash(v)};std::filesystem::path p(path),tmp=p;tmp += ".tmp";{std::ofstream out(tmp,std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("open checkpoint temp");out.write((char*)&h,sizeof(h));if(!v.empty())out.write((char*)v.data(),v.size()*sizeof(Packed));out.flush();if(!out)throw std::runtime_error("write checkpoint");}std::filesystem::rename(tmp,p);}
int row8_cap9_reach_main_unused(int argc,char**argv){
 int r=argc>1?std::atoi(argv[1]):8,maxcols=argc>2?std::atoi(argv[2]):64;if(r<1||r>MAXR)return 2;
 std::string ck;if(char const*e=std::getenv("ROWR_BATCH_CHECKPOINT"))ck=e;
 Vec cur;int col=0;
 if(!ck.empty()&&load_ck(ck,r,col,cur)){std::cout<<"resume r="<<r<<" col="<<col<<" states="<<cur.size()<<" hash="<<std::hex<<payload_hash(cur)<<std::dec<<"\n";}
 else {State z;z.n=r;cur={pack(z)};cur=column_step(cur,r,true);col=1;std::cout<<"r="<<r<<" col=1 states="<<cur.size()<<"\n";if(!ck.empty())save_ck(ck,r,col,cur);}
 for(int next=col+1;next<=maxcols;++next){Vec nxt=column_step(cur,r,false);bool fixed=nxt==cur;std::cout<<"r="<<r<<" col="<<next<<" states="<<nxt.size();if(fixed)std::cout<<" fixed=1";std::cout<<"\n";std::cout.flush();cur.swap(nxt);col=next;if(!ck.empty())save_ck(ck,r,col,cur);if(fixed)return 0;}return 0;
}


static bool final_accepts_overflow(Vec cur,int r,std::uint64_t& tested){
  // Keep only histories that have already exceeded cap 8.
  Vec ov; ov.reserve(cur.size()/16+1);
  for(auto const&p:cur){auto s=unpack(p);if(s.overflow&&s.sp<=1)ov.push_back(p);} cur.swap(ov); norm(cur);
  if(cur.empty()) return false;
  for(int y=0;y<r-1;++y){cur=physical_batch(cur,y,y+1,2,2);cur=forget_slot(cur,y,r,false);if(cur.empty())return false;}
  Vec ext;ext.reserve(cur.size());
  for(auto const&p:cur){State old=unpack(p);State z{};z.n=r+1;z.ns=old.ns;z.sp=old.sp;z.overflow=old.overflow;z.status=old.status;z.stack=old.stack;for(int y=0;y<r;++y){z.deg[y]=old.deg[y];z.comp[y]=old.comp[y];}append(ext,z);}norm(ext);cur.swap(ext);
  cur=physical_batch(cur,r-1,r,2,2);cur=forget_slot(cur,r-1,r,false);
  for(auto const&p:cur){State base=unpack(p);int bottom=r;for(int sym=0;sym<3;++sym){++tested;State z=base;auto q=z.comp[bottom];bool ok=true;
    if(sym==0){if(z.deg[bottom]!=0)continue;}
    else {if(z.deg[bottom]!=1||!q)continue;if(sym==2){if(z.sp>=MAXSTACK)continue;z.stack[z.sp++]=q;if(z.sp>MAXR)z.overflow=1;}else{if(!z.sp)continue;auto q2=z.stack[--z.sp];if(!merge_components(z,q,q2,true))continue;q=z.comp[bottom];}}
    z.deg[bottom]=0;z.comp[bottom]=0;
    if(q){bool alive=stack_has(z,q);if(!alive)for(int i=0;i<z.n;++i)if(z.comp[i]==q){alive=true;break;}if(!alive&&!(z.status[q]&2))ok=false;}
    if(!ok||z.sp)continue;for(int i=0;i<r;++i)if(z.deg[i]||z.comp[i]){ok=false;break;}if(!ok)continue;
    for(int qq=1;qq<z.ns;++qq)if((z.status[qq]&2)&&!closed_consistent(z,qq)){ok=false;break;}
    if(ok&&z.overflow)return true;
  }}
  return false;
}

static void stats(Vec const&v,std::uint64_t&overflow,int&maxsp){overflow=0;maxsp=0;for(auto const&p:v){auto s=unpack(p);overflow+=!!s.overflow;maxsp=std::max(maxsp,(int)s.sp);}}

#ifndef ROW8_CAP9_NO_MAIN
int main(int argc,char**argv){
  int r=8,maxcols=argc>1?std::atoi(argv[1]):64;if(r!=8)return 2;
  std::string ck="work/formal-probes/raw_wfa_r8_cap9_overflow.ck";if(char const*e=std::getenv("ROWR_CAP9_CHECKPOINT"))ck=e;
  Vec cur;int col=0;
  auto check_final=[&](int q)->bool{std::uint64_t tested=0;bool bad=final_accepts_overflow(cur,r,tested);std::cout<<"final_check col="<<q<<" accept="<<bad<<" tested="<<tested<<"\n";std::cout.flush();return bad;};
  if(!ck.empty()&&load_ck(ck,r,col,cur)){std::uint64_t ov;int ms;stats(cur,ov,ms);std::cout<<"resume r=8 col="<<col<<" states="<<cur.size()<<" overflow="<<ov<<" maxsp="<<ms<<" hash="<<std::hex<<payload_hash(cur)<<std::dec<<"\n";if(check_final(col))return 7;}
  else {State z;z.n=r;cur={pack(z)};cur=column_step(cur,r,true);col=1;std::uint64_t ov;int ms;stats(cur,ov,ms);std::cout<<"r=8 col=1 states="<<cur.size()<<" overflow="<<ov<<" maxsp="<<ms<<"\n";if(check_final(col))return 7;if(!ck.empty())save_ck(ck,r,col,cur);}
  for(int next=col+1;next<=maxcols;++next){Vec nxt=column_step(cur,r,false);bool fixed=nxt==cur;cur.swap(nxt);col=next;std::uint64_t ov;int ms;stats(cur,ov,ms);std::cout<<"r=8 col="<<next<<" states="<<cur.size()<<" overflow="<<ov<<" maxsp="<<ms;if(fixed)std::cout<<" fixed=1";std::cout<<"\n";std::cout.flush();if(check_final(col))return 7;if(!ck.empty())save_ck(ck,r,col,cur);if(fixed){std::cout<<"cap9_all_widths_no_overflow_accept=1 fixed_col="<<col<<" states="<<cur.size()<<" overflow_states="<<ov<<" exact=1\n";return 0;}}
  return 3;
}
#endif
