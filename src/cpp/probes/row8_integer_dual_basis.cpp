#define ROW8_H2_PARTITION_NO_MAIN 1
#include "row8_h2_relation_partition.cpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

static constexpr uint32_t REF=1000000007u;
using I64 = std::int64_t;
using Edge = std::pair<int,uint32_t>;
using Trans = std::vector<std::vector<Edge>>;

static uint32_t modinv(uint32_t a){uint64_t e=REF-2,x=a,r=1;while(e){if(e&1)r=(__uint128_t)r*x%REF;x=(__uint128_t)x*x%REF;e>>=1;}return (uint32_t)r;}

static Trans make_trans(std::vector<Packed>const&src,std::vector<Packed>const&dst,int sym){
  Trans T(src.size()); size_t ne=0;
  for(int i=0;i<(int)src.size();++i){
    WVec v{{src[i],1}}; auto z=wcolumn(std::move(v),8,false,sym);
    for(auto const&e:z){auto it=std::lower_bound(dst.begin(),dst.end(),e.p);if(it==dst.end()||!(*it==e.p))throw std::runtime_error("transition target missing");T[i].push_back({int(it-dst.begin()),e.v});++ne;}
  }
  std::cerr<<"trans sym="<<sym<<" src="<<src.size()<<" dst="<<dst.size()<<" edges="<<ne<<"\n";
  return T;
}

static std::vector<I64> pull_exact(Trans const&T,std::vector<I64>const&f,I64&globalMax){
  std::vector<I64> y(T.size());
  for(int i=0;i<(int)T.size();++i){
    __int128 z=0; for(auto [t,c]:T[i])z+=(__int128)c*f[t];
    if(z<std::numeric_limits<I64>::min()||z>std::numeric_limits<I64>::max())throw std::runtime_error("int64 overflow in pull_exact");
    y[i]=(I64)z; I64 a=y[i]<0?-y[i]:y[i]; if(a>globalMax)globalMax=a;
  }
  return y;
}

struct RedRow { int pivot=-1; std::vector<std::pair<int,uint32_t>> nz; };
struct ModReducer {
  int n; std::vector<int> at; std::vector<RedRow> rows;
  explicit ModReducer(int nn):n(nn),at(nn,-1){}
  bool independent(std::vector<I64>const&orig){
    std::vector<uint32_t> y(n);
    for(int i=0;i<n;++i){I64 a=orig[i]%(I64)REF;if(a<0)a+=REF;y[i]=(uint32_t)a;}
    for(int p=0;p<n;++p)if(y[p]&&at[p]>=0){
      uint32_t f=y[p]; auto const&r=rows[at[p]];
      for(auto [k,v]:r.nz){uint32_t sub=(uint32_t)((__uint128_t)f*v%REF);y[k]=y[k]>=sub?y[k]-sub:y[k]+REF-sub;}
    }
    int p=-1;for(int i=0;i<n;++i)if(y[i]){p=i;break;}if(p<0)return false;
    uint32_t iv=modinv(y[p]);RedRow r;r.pivot=p;r.nz.reserve(128);
    for(int i=p;i<n;++i)if(y[i])r.nz.push_back({i,(uint32_t)((__uint128_t)y[i]*iv%REF)});
    at[p]=rows.size();rows.push_back(std::move(r));return true;
  }
};

struct BasisVec { std::vector<I64> v; int kind=0, source=-1, parent=-1, depth=0; };

static void save_dense_i64(std::string path,int h,std::vector<BasisVec>const&B,int states,I64 maxabs){
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  struct H{char magic[8];uint32_t ver,h,rows,states;uint64_t maxabs;} hdr{{'R','8','D','U','A','L','I','1'},1,(uint32_t)h,(uint32_t)B.size(),(uint32_t)states,(uint64_t)maxabs};
  std::ofstream out(path,std::ios::binary|std::ios::trunc);out.write((char*)&hdr,sizeof(hdr));for(auto const&b:B)out.write((char*)b.v.data(),b.v.size()*sizeof(I64));if(!out)throw std::runtime_error("save dense");
}
static void save_recipe(std::string path,std::vector<BasisVec>const&B){std::ofstream o(path,std::ios::trunc);for(int i=0;i<(int)B.size();++i)o<<i<<' '<<B[i].kind<<' '<<B[i].source<<' '<<B[i].parent<<' '<<B[i].depth<<'\n';}

static std::vector<std::vector<I64>> load_phi2(std::vector<Packed>const&h2,std::vector<int>&goodPos,I64&mx){
  auto ws=words2();std::set<int>bad;{std::ifstream in("work/formal-probes/canonical-matrix/h2_actual_bad_indices.txt");int x;while(in>>x)bad.insert(x);}if(bad.size()!=120)throw std::runtime_error("bad120");
  goodPos.clear();for(int j=0;j<(int)ws.size();++j)if(!bad.count(j)){auto it=std::lower_bound(h2.begin(),h2.end(),ws[j].first);if(it==h2.end()||!(*it==ws[j].first))throw std::runtime_error("h2 good missing");goodPos.push_back(it-h2.begin());}if(goodPos.size()!=1308)throw std::runtime_error("good1308");
  struct HH{char m[8];uint32_t ver,mod,h,rows,states;} hh{};std::ifstream in("work/formal-probes/dual-basis/Phi_h2_extra_mod1000000007.bin",std::ios::binary);in.read((char*)&hh,sizeof(hh));if(!in||hh.mod!=REF||hh.h!=2||hh.rows!=120||hh.states!=h2.size())throw std::runtime_error("phi2 extra header");
  std::vector<std::vector<I64>> ex(120,std::vector<I64>(h2.size()));
  for(int r=0;r<120;++r)for(int i=0;i<(int)h2.size();++i){uint32_t x;in.read((char*)&x,4);I64 y=x<=REF/2?(I64)x:(I64)x-REF;if(y<-3||y>3)throw std::runtime_error("phi2 extra not small integer");ex[r][i]=y;I64 a=y<0?-y:y;if(a>mx)mx=a;}if(!in)throw std::runtime_error("phi2 extra read");
  return ex;
}

static std::vector<BasisVec> build_h1(Trans const&TL,Trans const&TN,std::vector<Packed>const&h2,std::vector<int>const&goodPos,std::vector<std::vector<I64>>const&extra,I64&mx){
  int n=TL.size();ModReducer R(n);std::vector<BasisVec>B;B.reserve(1640);std::deque<int>q;
  std::vector<std::vector<Edge>> incoming(h2.size());for(int i=0;i<n;++i)for(auto [t,c]:TL[i])incoming[t].push_back({i,c});
  auto tryadd=[&](std::vector<I64> y,int kind,int source,int parent,int depth){if(!R.independent(y))return false;int id=B.size();B.push_back({std::move(y),kind,source,parent,depth});q.push_back(id);return true;};
  int seedAdded=0;
  for(int f=0;f<1428;++f){std::vector<I64> y(n);if(f<1308){for(auto [i,c]:incoming[goodPos[f]]){y[i]+=c;if(y[i]>mx)mx=y[i];}}else y=pull_exact(TL,extra[f-1308],mx);if(tryadd(std::move(y),0,f,-1,0))++seedAdded;}
  std::map<int,int> dep;
  while(!q.empty()&&B.size()<1640){int bi=q.front();q.pop_front();auto y=pull_exact(TN,B[bi].v,mx);if(tryadd(std::move(y),1,B[bi].source,bi,B[bi].depth+1))++dep[B.back().depth];}
  std::cout<<"h1_integer seed="<<seedAdded<<" final="<<B.size()<<" maxabs="<<mx<<" depth";for(auto[d,c]:dep)std::cout<<' '<<d<<':'<<c;std::cout<<'\n';if(B.size()!=1640)throw std::runtime_error("h1 dim");return B;
}

static std::vector<BasisVec> build_h0(Trans const&TL,Trans const&TN,std::vector<BasisVec>const&H1,I64&mx){
  int n=TL.size();ModReducer R(n);std::vector<BasisVec>B;B.reserve(1107);std::deque<int>q;
  auto tryadd=[&](std::vector<I64> y,int kind,int source,int parent,int depth){if(!R.independent(y))return false;int id=B.size();B.push_back({std::move(y),kind,source,parent,depth});q.push_back(id);return true;};
  int seedAdded=0;for(int f=0;f<(int)H1.size();++f){auto y=pull_exact(TL,H1[f].v,mx);if(tryadd(std::move(y),0,f,-1,0))++seedAdded;}
  std::map<int,int>dep;while(!q.empty()&&B.size()<1107){int bi=q.front();q.pop_front();auto y=pull_exact(TN,B[bi].v,mx);if(tryadd(std::move(y),1,B[bi].source,bi,B[bi].depth+1))++dep[B.back().depth];}
  std::cout<<"h0_integer seed="<<seedAdded<<" final="<<B.size()<<" maxabs="<<mx<<" depth";for(auto[d,c]:dep)std::cout<<' '<<d<<':'<<c;std::cout<<'\n';if(B.size()!=1107)throw std::runtime_error("h0 dim");return B;
}

int main(){
  // A one-column macro has tiny integer multiplicities.  Use a large modulus so
  // WVec normalization does not change those local integer edge weights.
  MODP=4294967291u;
  Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::array<std::vector<Packed>,3>H;for(auto const&p:all){int h=unpack(p).sp;if(h<=2)H[h].push_back(p);}std::cout<<"raw h0="<<H[0].size()<<" h1="<<H[1].size()<<" h2="<<H[2].size()<<" col="<<col<<"\n";
  I64 mx=0;std::vector<int>goodPos;auto extra=load_phi2(H[2],goodPos,mx);
  auto TL1=make_trans(H[1],H[2],2),TN1=make_trans(H[1],H[1],0);
  auto B1=build_h1(TL1,TN1,H[2],goodPos,extra,mx);
  save_dense_i64("work/formal-probes/dual-basis/Phi_h1_integer.bin",1,B1,H[1].size(),mx);save_recipe("work/formal-probes/dual-basis/Phi_h1_integer.recipe",B1);
  auto TL0=make_trans(H[0],H[1],2),TN0=make_trans(H[0],H[0],0);
  auto B0=build_h0(TL0,TN0,B1,mx);
  save_dense_i64("work/formal-probes/dual-basis/Phi_h0_integer.bin",0,B0,H[0].size(),mx);save_recipe("work/formal-probes/dual-basis/Phi_h0_integer.recipe",B0);
  std::cout<<"integer_dual_basis_exact_data=1 global_maxabs="<<mx<<"\n";
}
