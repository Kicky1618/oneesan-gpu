#define main rowr_batch_embedded_main
#include "rowr_column_batch_wfa_checkpoint.cpp"
#undef main
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Pair = std::pair<int,int>;

static void gen_matchings_rec(std::vector<int> const& pts, std::vector<Pair>& cur,
                              std::vector<std::vector<Pair>>& out) {
  if (pts.empty()) { out.push_back(cur); return; }
  int a = pts.front();
  for (size_t k=1;k<pts.size();k+=2) {
    int b=pts[k];
    std::vector<int> inside(pts.begin()+1, pts.begin()+k);
    std::vector<int> outside(pts.begin()+k+1, pts.end());
    std::vector<std::vector<Pair>> li, lo;
    std::vector<Pair> tmp;
    gen_matchings_rec(inside,tmp,li); tmp.clear(); gen_matchings_rec(outside,tmp,lo);
    for (auto const& xi: li) for (auto const& xo: lo) {
      auto save=cur.size(); cur.push_back({a,b});
      cur.insert(cur.end(),xi.begin(),xi.end()); cur.insert(cur.end(),xo.begin(),xo.end());
      out.push_back(cur);
      cur.resize(save);
    }
  }
}

static std::vector<std::vector<Pair>> gen_matchings(int npts) {
  // Simpler canonical Catalan recursion avoiding duplicate subproblem composition bugs.
  std::function<std::vector<std::vector<Pair>>(int,int)> rec = [&](int l,int r) {
    std::vector<std::vector<Pair>> res;
    if (l>=r) { res.push_back({}); return res; }
    for (int k=l+1;k<r;k+=2) {
      auto A=rec(l+1,k), B=rec(k+1,r);
      for(auto const&a:A) for(auto const&b:B){
        std::vector<Pair> z; z.reserve(1+a.size()+b.size()); z.push_back({l,k});
        z.insert(z.end(),a.begin(),a.end()); z.insert(z.end(),b.begin(),b.end()); res.push_back(std::move(z));
      }
    }
    return res;
  };
  return rec(0,npts);
}

static bool load_states(std::string const& path, Vec& v, int& r, int& col) {
  std::ifstream in(path,std::ios::binary); if(!in)return false;
  CkHdr h{}; in.read((char*)&h,sizeof(h));
  if(!in || std::string(h.magic,7)!="RWBATCH" || h.ver!=1) return false;
  r=(int)h.r; col=(int)h.col; v.resize(h.n);
  if(h.n) in.read((char*)v.data(),h.n*sizeof(Packed));
  if(!in || payload_hash(v)!=h.hash) throw std::runtime_error("bad checkpoint");
  return true;
}

static Packed make_state(int r,int h,int p,std::vector<Pair> const& pairs,uint32_t statusMask) {
  State s{}; s.n=r; s.sp=(uint8_t)h;
  int npts=p+h, narcs=npts/2;
  std::vector<uint8_t> comp(npts,0);
  s.ns=1;
  for(int e=0;e<narcs;++e){
    auto [a,b]=pairs[e]; uint8_t q=s.ns++;
    comp[a]=comp[b]=q; s.status[q]=(statusMask>>e)&1u;
  }
  // Canonical occupancy mask: first p frontier slots occupied.
  for(int i=0;i<p;++i){s.deg[i]=1;s.comp[i]=comp[i];}
  // Abstract endpoint order is frontier ascending followed by reverse(stack).
  for(int k=0;k<h;++k) s.stack[k]=comp[p+(h-1-k)];
  return pack(s);
}

static std::string sig(std::vector<Pair> const& pairs,int p){
  std::ostringstream o;
  for(size_t i=0;i<pairs.size();++i){if(i)o<<';';auto[a,b]=pairs[i];
    char typ=(b<p?'F':(a>=p?'S':'X')); o<<a<<'-'<<b<<typ;
  }
  return o.str();
}

int main(int argc,char**argv){
  std::string ck=argc>1?argv[1]:"work/formal-probes/raw_wfa_r8.ck";
  int onlyh=argc>2?std::atoi(argv[2]):-1, onlyp=argc>3?std::atoi(argv[3]):-1;
  Vec states;int r=0,col=0;if(!load_states(ck,states,r,col))return 2;
  std::cerr<<"loaded r="<<r<<" col="<<col<<" states="<<states.size()<<"\n";
  for(int h=0;h<=r;++h){ if(onlyh>=0&&h!=onlyh)continue;
    for(int p=0;p<=r;++p){ if(onlyp>=0&&p!=onlyp)continue; if((p+h)&1)continue;
      int npts=p+h;if(npts==0){std::vector<Pair> z;auto pk=make_state(r,h,p,z,0);bool ok=std::binary_search(states.begin(),states.end(),pk);std::cout<<h<<','<<p<<",EMPTY,0,"<<ok<<"\n";continue;}
      auto ms=gen_matchings(npts); size_t hit=0,tot=0;
      for(size_t mi=0;mi<ms.size();++mi){int n=(int)ms[mi].size();uint32_t lim=1u<<n;
        for(uint32_t sm=0;sm<lim;++sm){auto pk=make_state(r,h,p,ms[mi],sm);bool ok=std::binary_search(states.begin(),states.end(),pk);hit+=ok;++tot;
          std::cout<<h<<','<<p<<','<<sig(ms[mi],p)<<','<<sm<<','<<ok<<'\n';
        }
      }
      std::cerr<<"h="<<h<<" p="<<p<<" matchings="<<ms.size()<<" candidates="<<tot<<" hit="<<hit<<"\n";
    }
  }
}
