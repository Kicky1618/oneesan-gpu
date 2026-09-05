#define ROW8_H2_PARTITION_NO_MAIN 1
#include "row8_h2_relation_partition.cpp"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>
static constexpr uint32_t Q=1000000007u;
struct HH{char magic[8];uint32_t ver,mod,h,rows,states;};
static uint32_t invq(uint32_t a){uint64_t e=Q-2,r=1,x=a;while(e){if(e&1)r=r*x%Q;x=x*x%Q;e>>=1;}return r;}
static uint32_t omask(Packed p){State s=unpack(p);uint32_t m=0;for(int i=0;i<8;++i)if(s.deg[i])m|=1u<<i;return m;}
static Packed compress(Packed p){State s=unpack(p),t{};t.sp=s.sp;t.stack=s.stack;t.status=s.status;t.ns=s.ns;int n=0;for(int i=0;i<8;++i)if(s.deg[i]){t.deg[n]=s.deg[i];t.comp[n]=s.comp[i];++n;}t.n=n;return pack(t);}
static std::vector<std::vector<uint32_t>> rref(std::vector<std::vector<uint32_t>> B){int R=B.size(),C=R?B[0].size():0,rk=0;for(int c=0;c<C&&rk<R;++c){int p=rk;while(p<R&&!B[p][c])++p;if(p==R)continue;std::swap(B[p],B[rk]);uint32_t iv=invq(B[rk][c]);for(int j=c;j<C;++j)B[rk][j]=(uint64_t)B[rk][j]*iv%Q;for(int i=0;i<R;++i)if(i!=rk&&B[i][c]){uint32_t f=B[i][c];for(int j=c;j<C;++j)B[i][j]=(B[i][j]+Q-(uint64_t)f*B[rk][j]%Q)%Q;}++rk;}B.resize(rk);return B;}
int main(){Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::vector<Packed>h2;for(auto&p:all)if(unpack(p).sp==2)h2.push_back(p);std::ifstream in("work/formal-probes/dual-basis/Phi_h2_extra_mod1000000007.bin",std::ios::binary);HH hh{};in.read((char*)&hh,sizeof(hh));std::vector<std::vector<uint32_t>>A(hh.rows,std::vector<uint32_t>(hh.states));for(auto&r:A)in.read((char*)r.data(),r.size()*4);
 std::map<uint32_t,std::vector<int>> cols;for(int i=0;i<(int)h2.size();++i)cols[omask(h2[i])].push_back(i);
 bool ok=true;int sumrank=0,j6n=0;std::vector<Packed> refkeys;std::vector<std::vector<uint32_t>> refr; std::vector<Packed> keys8; std::vector<std::vector<uint32_t>> rref8;
 for(auto const&[m,ix0]:cols){int j=__builtin_popcount(m);if(j!=6&&j!=8)continue;std::vector<std::pair<Packed,int>> z;for(int i:ix0)z.push_back({compress(h2[i]),i});std::sort(z.begin(),z.end(),[](auto&a,auto&b){return a.first<b.first;});std::vector<Packed>keys;std::vector<int>ix;for(auto const&x:z){keys.push_back(x.first);ix.push_back(x.second);}for(size_t k=1;k<keys.size();++k)if(keys[k]==keys[k-1]){std::cerr<<"duplicate compressed state\n";return 2;}
  std::vector<std::vector<uint32_t>>B(hh.rows,std::vector<uint32_t>(ix.size()));for(int r=0;r<(int)hh.rows;++r)for(int c=0;c<(int)ix.size();++c)B[r][c]=A[r][ix[c]];auto Rr=rref(std::move(B));int rk=Rr.size();sumrank+=rk;
  std::cout<<"mask="<<m<<" j="<<j<<" rawcore="<<ix.size()<<" proj_rank="<<rk;
  if(j==6){++j6n;if(refkeys.empty()){refkeys=keys;refr=Rr;std::cout<<" reference=1";}else{bool keq=keys==refkeys,req=Rr==refr;ok&=keq&&req;std::cout<<" keys_same="<<keq<<" rref_same="<<req;}} else {keys8=keys;rref8=Rr;}
  std::cout<<"\n";
 }
 std::cout<<"j6_masks="<<j6n<<" sum_projection_ranks="<<sumrank<<" universal_j6="<<ok<<"\n";
 auto psig=[](Packed p){State s=unpack(p);std::ostringstream o;o<<"n="<<int(s.n)<<" deg=";for(int i=0;i<s.n;++i)o<<int(s.deg[i]);o<<" comp=";for(int i=0;i<s.n;++i){if(i)o<<",";o<<int(s.comp[i]);}o<<" stack=";for(int i=0;i<s.sp;++i){if(i)o<<",";o<<int(s.stack[i]);}o<<" status=";for(int q=1;q<s.ns;++q)o<<int(s.status[q]&1);return o.str();}; if(!refr.empty()){for(int r=0;r<(int)refr.size();++r){int nz=0;for(auto x:refr[r])nz+=x!=0;std::cout<<"core6_basis r="<<r<<" nz="<<nz<<"\n";for(int c=0;c<(int)refr[r].size();++c)if(refr[r][c])std::cout<<"  coeff="<<(refr[r][c]<=Q/2?(int64_t)refr[r][c]:(int64_t)refr[r][c]-Q)<<" "<<psig(refkeys[c])<<"\n";}} std::map<int,int>h8; for(auto const&row:rref8){int nz=0;for(auto x:row)nz+=x!=0;++h8[nz];}std::cout<<"core8_nz_hist";for(auto[a,b]:h8)std::cout<<" "<<a<<":"<<b;std::cout<<"\n"; for(int r=0;r<(int)rref8.size()&&r<12;++r){std::cout<<"core8_basis r="<<r<<"\n";for(int c=0;c<(int)rref8[r].size();++c)if(rref8[r][c])std::cout<<"  coeff="<<(rref8[r][c]<=Q/2?(int64_t)rref8[r][c]:(int64_t)rref8[r][c]-Q)<<" "<<psig(keys8[c])<<"\n";}
 return ok&&j6n==28&&sumrank==120?0:1;}
