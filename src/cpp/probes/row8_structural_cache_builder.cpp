#define ROW8_STRUCTURAL_INTEGER_NO_MAIN 1
#include "row8_structural_integer_closure.cpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

struct SHdr {
  char magic[8];
  uint32_t version, r;
  uint32_t dims[9];
  uint64_t total_nz;
  uint64_t fnv_hash;
};
struct BHdr { uint32_t sym,h,h2,rows,cols,nnz; };
struct VHdr { uint32_t tag,sym,h,nnz; }; // tag 1=alpha, 2=beta

static uint64_t fnv64(uint64_t h, void const*vp, size_t n){
  auto p=(unsigned char const*)vp;
  for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}
  return h;
}

template<class T> static void put(std::ofstream&out,T const&x,uint64_t&hh){out.write((char const*)&x,sizeof(x));hh=fnv64(hh,&x,sizeof(x));}
template<class T> static void putv(std::ofstream&out,std::vector<T>const&x,uint64_t&hh){if(!x.empty()){out.write((char const*)x.data(),x.size()*sizeof(T));hh=fnv64(hh,x.data(),x.size()*sizeof(T));}}

static std::vector<std::pair<uint16_t,int8_t>> initial_vec(Space const&S,int sym){
  State z{};z.n=8;WVec v{{pack(z),1}};auto w=wcolumn(std::move(v),8,true,sym);auto ev=evalI(S);std::vector<int64_t> c(S.dim);
  for(auto const&e:w){int i=std::lower_bound(S.states.begin(),S.states.end(),e.p)-S.states.begin();if(i==(int)S.states.size()||!(S.states[i]==e.p))throw std::runtime_error("alpha target missing");for(auto[k,a]:ev[i])c[k]+=(int64_t)e.v*a;}
  std::vector<std::pair<uint16_t,int8_t>> out;for(int i=0;i<S.dim;++i)if(c[i]){if(c[i]<-127||c[i]>127)throw std::runtime_error("alpha coeff int8");out.push_back({(uint16_t)i,(int8_t)c[i]});}return out;
}
static std::vector<std::pair<uint16_t,int8_t>> final_vec(Space const&S,int sym){
  std::vector<std::pair<uint16_t,int8_t>> out;
  for(auto const&b:S.blocks)for(int r=0;r<(int)b.rr.size();++r){int i=b.off+r,src=b.idx[b.piv[r]];WVec v{{S.states[src],1}};auto x=(int64_t)wlast_value(std::move(v),8,sym);if(x){if(x>127)throw std::runtime_error("beta coeff int8");out.push_back({(uint16_t)i,(int8_t)x});}}
  return out;
}

int main(int argc,char**argv){
  MODP=4294967291u;
  std::string path=argc>1?argv[1]:"work/row8_structural_cache/row8_structural_int_v1.bin";
  Vec all;int col=0;if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all))return 2;
  std::array<std::vector<Packed>,9>H;for(auto const&p:all)H[unpack(p).sp].push_back(p);
  std::array<std::unique_ptr<Space>,9>S;
  S[0]=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));
  S[1]=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));
  S[2]=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));
  for(int h=3;h<=8;++h)S[h]=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));

  std::array<std::array<IAdj,9>,3> QI;uint64_t total=0;int64_t maxabs=0;
  for(int h=0;h<=8;++h)for(int a=0;a<3;++a){int h2=h+DEL_[a];if(h2<0||h2>8)continue;auto te=evalI(*S[h2]);QI[a][h]=buildQI(*S[h],*S[h2],a,te);for(auto const&r:QI[a][h])for(auto [j,c]:r){++total;auto z=c<0?-c:c;maxabs=std::max(maxabs,z);if(c<-127||c>127)throw std::runtime_error("Q coeff int8");}}

  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream out(path,std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("open output");
  SHdr hdr{{'R','8','S','T','R','0','1','\0'},1,8,{0},total,0};for(int h=0;h<9;++h)hdr.dims[h]=S[h]->dim;
  out.write((char*)&hdr,sizeof(hdr)); // hash excludes header so it can be patched
  uint64_t hh=1469598103934665603ULL;

  // Three first-column vectors.  Source handling makes base height one.
  for(int a=0;a<3;++a){int h=1+DEL_[a];auto v=initial_vec(*S[h],a);VHdr x{1u,(uint32_t)a,(uint32_t)h,(uint32_t)v.size()};put(out,x,hh);for(auto [i,c]:v){put(out,i,hh);put(out,c,hh);}std::cout<<"alpha a="<<a<<" h="<<h<<" nz="<<v.size()<<"\n";}
  // Only these final cases can accept.
  for(auto [h,a]:{std::pair<int,int>{0,0},{1,1}}){auto v=final_vec(*S[h],a);VHdr x{2u,(uint32_t)a,(uint32_t)h,(uint32_t)v.size()};put(out,x,hh);for(auto [i,c]:v){put(out,i,hh);put(out,c,hh);}std::cout<<"beta a="<<a<<" h="<<h<<" nz="<<v.size()<<"\n";}

  for(int a=0;a<3;++a)for(int h=0;h<=8;++h){int h2=h+DEL_[a];if(h2<0||h2>8)continue;auto const&A=QI[a][h];uint32_t nz=0;for(auto const&r:A)nz+=r.size();BHdr b{(uint32_t)a,(uint32_t)h,(uint32_t)h2,(uint32_t)S[h]->dim,(uint32_t)S[h2]->dim,nz};put(out,b,hh);std::vector<uint32_t>rp(A.size()+1);std::vector<uint16_t>ci;std::vector<int8_t>cv;ci.reserve(nz);cv.reserve(nz);for(int i=0;i<(int)A.size();++i){rp[i]=ci.size();for(auto[j,c]:A[i]){ci.push_back((uint16_t)j);cv.push_back((int8_t)c);}}rp[A.size()]=ci.size();putv(out,rp,hh);putv(out,ci,hh);putv(out,cv,hh);std::cout<<"block a="<<a<<" h="<<h<<" h2="<<h2<<" nz="<<nz<<"\n";}
  if(!out)throw std::runtime_error("write output");out.close();
  hdr.fnv_hash=hh;std::fstream io(path,std::ios::binary|std::ios::in|std::ios::out);io.write((char*)&hdr,sizeof(hdr));io.close();
  auto sz=std::filesystem::file_size(path);
  std::cout<<"structural_cache path="<<path<<" bytes="<<sz<<" total_nz="<<total<<" maxabs="<<maxabs<<" hash="<<std::hex<<hh<<std::dec<<"\n";
  return 0;
}
