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
static uint32_t mask(Packed p){State s=unpack(p);uint32_t m=0;for(int i=0;i<8;++i)if(s.deg[i])m|=1u<<i;return m;}
static Packed compress(Packed p){State s=unpack(p),t{};t.sp=s.sp;t.stack=s.stack;t.status=s.status;t.ns=s.ns;int n=0;for(int i=0;i<8;++i)if(s.deg[i]){t.deg[n]=s.deg[i];t.comp[n]=s.comp[i];++n;}t.n=n;return pack(t);}
static std::vector<std::vector<uint32_t>> rref(std::vector<std::vector<uint32_t>> B){int R=B.size(),C=R?B[0].size():0,rk=0;for(int c=0;c<C&&rk<R;++c){int p=rk;while(p<R&&!B[p][c])++p;if(p==R)continue;std::swap(B[p],B[rk]);uint32_t iv=invq(B[rk][c]);for(int j=c;j<C;++j)B[rk][j]=(uint64_t)B[rk][j]*iv%Q;for(int i=0;i<R;++i)if(i!=rk&&B[i][c]){uint32_t f=B[i][c];for(int j=c;j<C;++j)B[i][j]=(B[i][j]+Q-(uint64_t)f*B[rk][j]%Q)%Q;}++rk;}B.resize(rk);return B;}
static uint64_t a111959(int j,int h){if(j<h||((j-h)&1))return 0;int d=(j-h)/2;uint64_t c=1;for(int t=0;t<d;++t)c=c*2u*(h+1+2*t)/(t+1);return c;}
int main(){Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::vector<Packed>h2;for(auto&p:all)if(unpack(p).sp==2)h2.push_back(p);auto ws=words2();std::set<int>bad;{std::ifstream b("work/formal-probes/canonical-matrix/h2_actual_bad_indices.txt");int x;while(b>>x)bad.insert(x);}std::vector<std::vector<uint32_t>>A(1428,std::vector<uint32_t>(h2.size()));int rr=0;for(int j=0;j<(int)ws.size();++j)if(!bad.count(j)){int i=std::lower_bound(h2.begin(),h2.end(),ws[j].first)-h2.begin();A[rr++][i]=1;}if(rr!=1308)throw std::runtime_error("good");std::ifstream in("work/formal-probes/dual-basis/Phi_h2_extra_mod1000000007.bin",std::ios::binary);HH hh{};in.read((char*)&hh,sizeof(hh));for(int r=0;r<120;++r)in.read((char*)A[1308+r].data(),A[1308+r].size()*4);
 std::map<int,uint32_t> rep;for(int i=0;i<(int)h2.size();++i){int j=__builtin_popcount(mask(h2[i]));if(!rep.count(j))rep[j]=mask(h2[i]);}
 for(auto [j,m]:rep){if(!a111959(j,2))continue;std::vector<std::pair<Packed,int>>z;for(int i=0;i<(int)h2.size();++i)if(mask(h2[i])==m)z.push_back({compress(h2[i]),i});std::sort(z.begin(),z.end(),[](auto&a,auto&b){return a.first<b.first;});std::vector<std::vector<uint32_t>>B(A.size(),std::vector<uint32_t>(z.size()));for(int r=0;r<(int)A.size();++r)for(int c=0;c<(int)z.size();++c)B[r][c]=A[r][z[c].second];auto Rr=rref(std::move(B));std::map<int64_t,int>ch;int64_t mx=0;std::map<int,int>nz;for(auto const&row:Rr){int n=0;for(auto x:row)if(x){++n;int64_t y=x<=Q/2?(int64_t)x:(int64_t)x-Q;++ch[y];mx=std::max(mx,y<0?-y:y);}++nz[n];}std::cout<<"h=2 j="<<j<<" rawcore="<<z.size()<<" rank="<<Rr.size()<<" want="<<a111959(j,2)<<" maxabs="<<mx<<" vals";for(auto[v,c]:ch)std::cout<<' '<<v<<':'<<c;std::cout<<" nzHist";for(auto[n,c]:nz)std::cout<<' '<<n<<':'<<c;std::cout<<"\n";}
}
