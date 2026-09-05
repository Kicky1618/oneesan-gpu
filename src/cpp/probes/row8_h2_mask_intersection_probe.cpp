#define ROW8_H2_PARTITION_NO_MAIN 1
#include "row8_h2_relation_partition.cpp"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>
static constexpr uint32_t Q=1000000007u;
struct HH{char magic[8];uint32_t ver,mod,h,rows,states;};
static uint32_t inv(uint32_t a){uint64_t e=Q-2,r=1,x=a;while(e){if(e&1)r=r*x%Q;x=x*x%Q;e>>=1;}return r;}
static uint32_t mask(Packed p){State s=unpack(p);uint32_t m=0;for(int i=0;i<8;++i)if(s.deg[i])m|=1u<<i;return m;}
static int rank_cols(std::vector<std::vector<uint32_t>> const&A,std::vector<int> const&cols){int R=A.size(),C=cols.size(),rk=0;std::vector<std::vector<uint32_t>> B(R,std::vector<uint32_t>(C));for(int i=0;i<R;++i)for(int j=0;j<C;++j)B[i][j]=A[i][cols[j]];for(int c=0;c<C&&rk<R;++c){int p=rk;while(p<R&&!B[p][c])++p;if(p==R)continue;std::swap(B[p],B[rk]);uint32_t iv=inv(B[rk][c]);for(int j=c;j<C;++j)B[rk][j]=(uint64_t)B[rk][j]*iv%Q;for(int i=0;i<R;++i)if(i!=rk&&B[i][c]){uint32_t f=B[i][c];for(int j=c;j<C;++j)B[i][j]=(B[i][j]+Q-(uint64_t)f*B[rk][j]%Q)%Q;}++rk;}return rk;}
int main(){Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::vector<Packed>h2;for(auto&p:all)if(unpack(p).sp==2)h2.push_back(p);std::vector<uint32_t> masks(h2.size());std::set<uint32_t> um;for(size_t i=0;i<h2.size();++i){masks[i]=mask(h2[i]);um.insert(masks[i]);}
 std::ifstream in("work/formal-probes/dual-basis/Phi_h2_extra_mod1000000007.bin",std::ios::binary);HH hh{};in.read((char*)&hh,sizeof(hh));std::vector<std::vector<uint32_t>>A(hh.rows,std::vector<uint32_t>(hh.states));for(auto&r:A)in.read((char*)r.data(),r.size()*4);
 std::map<int,std::map<int,int>> hist;int sum=0,nzMasks=0;for(auto m:um){std::vector<int> outside;outside.reserve(h2.size());for(int i=0;i<(int)h2.size();++i)if(masks[i]!=m)outside.push_back(i);int rk=rank_cols(A,outside),d=hh.rows-rk;int j=__builtin_popcount(m);++hist[j][d];if(d){sum+=d;++nzMasks;std::cout<<"mask="<<m<<" j="<<j<<" outside_rank="<<rk<<" intersection="<<d<<"\n";}}
 std::cout<<"nonzero_masks="<<nzMasks<<" sum_intersections="<<sum<<" rows="<<hh.rows<<" hist";for(auto&[j,h]:hist){std::cout<<" j"<<j<<'[';for(auto[d,c]:h)std::cout<<d<<':'<<c<<',';std::cout<<']';}std::cout<<"\n";return sum==120?0:1;}
