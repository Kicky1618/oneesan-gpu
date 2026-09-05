#define main rowr_batch_embedded_main
#include "rowr_column_batch_wfa_checkpoint.cpp"
#undef main
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>
static constexpr uint32_t Q=1000000007u;
struct IH{char magic[8];uint32_t ver,h,rows,states;uint64_t maxabs;};
static uint32_t invq(uint32_t a){uint64_t e=Q-2,r=1,x=a;while(e){if(e&1)r=r*x%Q;x=x*x%Q;e>>=1;}return r;}
static uint32_t omask(Packed p){State s=unpack(p);uint32_t m=0;for(int i=0;i<8;++i)if(s.deg[i])m|=1u<<i;return m;}
static Packed compress(Packed p){State s=unpack(p),t{};t.sp=s.sp;t.stack=s.stack;t.status=s.status;t.ns=s.ns;int n=0;for(int i=0;i<8;++i)if(s.deg[i]){t.deg[n]=s.deg[i];t.comp[n]=s.comp[i];++n;}t.n=n;return pack(t);}
static uint64_t a111959(int j,int h){if(j<h||((j-h)&1))return 0;int d=(j-h)/2;uint64_t c=1;for(int t=0;t<d;++t)c=c*2u*(h+1+2*t)/(t+1);return c;}
static std::vector<std::vector<uint32_t>> rref(std::vector<std::vector<uint32_t>> B){int R=B.size(),C=R?B[0].size():0,rk=0;for(int c=0;c<C&&rk<R;++c){int p=rk;while(p<R&&!B[p][c])++p;if(p==R)continue;std::swap(B[p],B[rk]);uint32_t iv=invq(B[rk][c]);for(int j=c;j<C;++j)B[rk][j]=(uint64_t)B[rk][j]*iv%Q;for(int i=0;i<R;++i)if(i!=rk&&B[i][c]){uint32_t f=B[i][c];for(int j=c;j<C;++j)B[i][j]=(B[i][j]+Q-(uint64_t)f*B[rk][j]%Q)%Q;}++rk;}B.resize(rk);return B;}
static bool runh(int h,std::vector<Packed> const&H,std::string path,int wantRows){std::ifstream in(path,std::ios::binary);IH hh{};in.read((char*)&hh,sizeof(hh));if(!in||hh.h!=(uint32_t)h||hh.rows!=(uint32_t)wantRows||hh.states!=H.size())throw std::runtime_error("header");std::vector<std::vector<uint32_t>>A(hh.rows,std::vector<uint32_t>(hh.states));for(int r=0;r<(int)hh.rows;++r)for(int c=0;c<(int)hh.states;++c){int64_t x;in.read((char*)&x,8);x%=Q;if(x<0)x+=Q;A[r][c]=x;}
 std::map<uint32_t,std::vector<int>>cols;for(int i=0;i<(int)H.size();++i)cols[omask(H[i])].push_back(i);std::map<int,std::vector<Packed>>refKeys;std::map<int,std::vector<std::vector<uint32_t>>>refR;std::map<int,int>maskN;int sumrank=0;bool ok=true;
 for(auto const&[m,ix0]:cols){int j=__builtin_popcount(m);std::vector<std::pair<Packed,int>>z;for(int i:ix0)z.push_back({compress(H[i]),i});std::sort(z.begin(),z.end(),[](auto&a,auto&b){return a.first<b.first;});std::vector<Packed>keys;std::vector<int>ix;for(auto&x:z){keys.push_back(x.first);ix.push_back(x.second);}std::vector<std::vector<uint32_t>>B(hh.rows,std::vector<uint32_t>(ix.size()));for(int r=0;r<(int)hh.rows;++r)for(int c=0;c<(int)ix.size();++c)B[r][c]=A[r][ix[c]];auto rr=rref(std::move(B));int rk=rr.size(),want=a111959(j,h);sumrank+=rk;++maskN[j];bool lok=rk==want;if(!refKeys.count(j)){refKeys[j]=keys;refR[j]=rr; std::map<int64_t,int> ch; int64_t ma=0; for(auto const&row:rr)for(auto x:row)if(x){int64_t y=x<=Q/2?(int64_t)x:(int64_t)x-Q; ++ch[y]; ma=std::max<int64_t>(ma,y<0?-y:y);} std::cout<<" CORECOEFF[h="<<h<<",j="<<j<<"] maxabs="<<ma<<" vals="; int sh=0; for(auto [v,c]:ch){if(sh++<20)std::cout<<v<<":"<<c<<",";} }else lok &= keys==refKeys[j] && rr==refR[j];ok&=lok;std::cout<<"h="<<h<<" mask="<<m<<" j="<<j<<" rawcore="<<ix.size()<<" rank="<<rk<<" want="<<want<<" universal="<<lok<<"\n";}
 std::cout<<"SUMMARY h="<<h<<" sum_projection_ranks="<<sumrank<<" dim="<<wantRows<<" exact="<<(ok&&sumrank==wantRows)<<" masks";for(auto[j,n]:maskN)std::cout<<' '<<j<<':'<<n;std::cout<<"\n";return ok&&sumrank==wantRows;}
int main(){Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::vector<Packed>H0,H1;for(auto&p:all){int h=unpack(p).sp;if(h==0)H0.push_back(p);else if(h==1)H1.push_back(p);}bool a=runh(0,H0,"work/formal-probes/dual-basis/Phi_h0_integer.bin",1107);bool b=runh(1,H1,"work/formal-probes/dual-basis/Phi_h1_integer.bin",1640);std::cout<<"lower_pascal_factor_exact="<<(a&&b)<<"\n";return a&&b?0:1;}
