#define ROW8_RAW_PREFIX_NO_MAIN 1
#include "row8_raw_prefix_vector.cpp"
#include <map>
#include <sstream>

using Pair=std::pair<int,int>;

static State make_logical(int r,int h,std::vector<int> const& occ,
                          std::vector<Pair> const& pairs,uint32_t sm){
  int p=occ.size(); State s{}; s.n=r; s.sp=h; s.ns=1;
  std::vector<uint8_t> ep(p+h);
  for(size_t e=0;e<pairs.size();++e){auto[a,b]=pairs[e];uint8_t q=s.ns++;ep[a]=ep[b]=q;s.status[q]=(sm>>e)&1u;}
  for(int i=0;i<p;++i){s.deg[occ[i]]=1;s.comp[occ[i]]=ep[i];}
  for(int k=0;k<h;++k)s.stack[k]=ep[p+h-1-k];
  canon(s);return s;
}

static std::string skeleton_key(State s){
  int r=s.n; std::vector<int> occ; for(int i=0;i<r;++i)if(s.deg[i])occ.push_back(i);
  int p=occ.size(),h=s.sp; std::vector<uint8_t> seq;seq.reserve(p+h);
  for(int i:occ)seq.push_back(s.comp[i]);
  for(int k=h-1;k>=0;--k)seq.push_back(s.stack[k]);
  // Canonical labels in endpoint order and statuses attached to first occurrence.
  std::array<uint8_t,MAXC> rm{};uint8_t nx=1;
  std::ostringstream o;o<<"h"<<h<<"p"<<p<<":";
  for(uint8_t q:seq){if(q&&!rm[q])rm[q]=nx++;o<<int(rm[q])<<'.';}
  o<<"|";std::vector<uint8_t> inv(nx,0);for(int q=1;q<s.ns;++q)if(rm[q])inv[rm[q]]=s.status[q]&1;
  for(int q=1;q<nx;++q)o<<int(inv[q]);
  return o.str();
}

static std::map<std::string,uint32_t> compressed(WVec const&v){
  std::map<std::string,uint64_t> tmp;
  for(auto const&e:v){auto k=skeleton_key(unpack(e.p));auto &x=tmp[k];x+=e.v;x%=MODP;}
  std::map<std::string,uint32_t> z;for(auto const&[k,vv]:tmp)if(vv)z[k]=vv;return z;
}

static void run_case(int h,std::vector<Pair> pairs,uint32_t sm,std::vector<std::vector<int>> const&masks){
  std::cout<<"case h="<<h<<" p="<<(pairs.size()*2-h)<<" sm="<<sm<<"\n";
  for(int a=0;a<3;++a){std::map<std::string,uint32_t> ref;bool first=true;
    for(auto const&occ:masks){State s=make_logical(8,h,occ,pairs,sm);WVec v{{pack(s),1}};auto out=wcolumn(std::move(v),8,false,a);auto c=compressed(out);
      std::cout<<" sym="<<a<<" occ=";for(int x:occ)std::cout<<x;std::cout<<" raw="<<out.size()<<" sk="<<c.size();
      if(first){ref=c;first=false;std::cout<<" ref=1\n";}else{std::cout<<" equal="<<(c==ref)<<"\n";if(c!=ref){size_t shown=0;for(auto const&[k,vv]:c){auto it=ref.find(k);uint32_t rv=it==ref.end()?0:it->second;if(rv!=vv&&shown++<5)std::cout<<"  diff "<<k<<" ref="<<rv<<" got="<<vv<<"\n";}}}
    }
  }
}

int main(){MODP=1000000007u;
  // Four frontier endpoints + four stack endpoints, nested matching, no marks.
  run_case(4,{{0,7},{1,6},{2,5},{3,4}},0,{{0,1,2,3},{0,2,4,6},{1,3,5,7},{0,1,6,7}});
  // h=2,p=4 with mixed FF/X/SS topology.
  run_case(2,{{0,1},{2,5},{3,4}},1,{{0,1,2,3},{0,2,4,6},{1,3,5,7},{0,1,6,7}});
}
