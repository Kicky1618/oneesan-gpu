#define ROW8_CANONICAL_TRIE_NO_MAIN 1
#include "row8_canonical_trie_verify.cpp"
#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
static constexpr uint32_t Q=1000000007u;
static uint32_t qpow(uint32_t a,uint64_t e){uint64_t r=1,x=a;while(e){if(e&1)r=(__uint128_t)r*x%Q;x=(__uint128_t)x*x%Q;e>>=1;}return r;}
static std::string sig8(Packed p){State s=unpack(p);std::ostringstream o;o<<"deg=";for(int i=0;i<8;++i)o<<int(s.deg[i]);o<<" comp=";for(int i=0;i<8;++i){if(i)o<<',';o<<int(s.comp[i]);}o<<" stack=";for(int i=0;i<s.sp;++i){if(i)o<<',';o<<int(s.stack[i]);}o<<" status=";for(int q=1;q<s.ns;++q)o<<int(s.status[q]&1);return o.str();}
int main(){MODP=Q;Vec all;int col=0;if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all))return 2;std::vector<Packed> hs;for(auto const&p:all)if(unpack(p).sp==4)hs.push_back(p);auto can=basis(4);int bads[5]={337,351,371,397,417};std::array<char,420>bad{};for(int x:bads)bad[x]=1;
 std::vector<std::pair<Packed,int>> goodPacked;goodPacked.reserve(415);std::vector<char> isGoodSrc(hs.size());std::vector<int> goodSrcToCan(hs.size(),-1);
 for(int j=0;j<420;++j)if(!bad[j]){auto it=std::lower_bound(hs.begin(),hs.end(),can[j]);if(it==hs.end()||!(*it==can[j]))throw std::runtime_error("canonical missing");int si=it-hs.begin();isGoodSrc[si]=1;goodSrcToCan[si]=j;goodPacked.push_back({can[j],j});}
 std::sort(goodPacked.begin(),goodPacked.end(),[](auto const&a,auto const&b){return a.first<b.first;});
 std::vector<int> nonId(hs.size(),-1), nonToSrc;for(int i=0;i<(int)hs.size();++i)if(!isGoodSrc[i]){nonId[i]=nonToSrc.size();nonToSrc.push_back(i);}int nn=nonToSrc.size();
 std::vector<std::vector<std::pair<int,uint32_t>>> pred(420);size_t edges=0,toGood=0;auto t0=std::chrono::steady_clock::now();
 for(int i=0;i<(int)hs.size();++i){WVec v{{hs[i],1}};auto z=wcolumn(std::move(v),8,false,0);edges+=z.size();if(!isGoodSrc[i])for(auto const&e:z){auto it=std::lower_bound(goodPacked.begin(),goodPacked.end(),e.p,[](auto const&a,Packed const&b){return a.first<b;});if(it!=goodPacked.end()&&it->first==e.p){pred[it->second].push_back({nonId[i],e.v});++toGood;}}if((i%8192)==0)std::cerr<<"trans "<<i<<"/"<<hs.size()<<" edges="<<edges<<"\n";}
 double build=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();std::cout<<"h4_states="<<hs.size()<<" good=415 non="<<nn<<" edges="<<edges<<" toGoodFromNon="<<toGood<<" build_s="<<build<<"\n";
 struct B{int pivot=-1,srcCan=-1;std::vector<uint32_t>v;};std::vector<B> bs;
 for(int j=0;j<420;++j)if(!bad[j]){std::vector<uint32_t> y(nn);for(auto [i,c]:pred[j])y[i]=c;for(auto const&b:bs){uint32_t f=y[b.pivot];if(!f)continue;for(int k=b.pivot;k<nn;++k)if(b.v[k])y[k]=(y[k]+Q-(__uint128_t)f*b.v[k]%Q)%Q;}int p=-1;for(int k=0;k<nn;++k)if(y[k]){p=k;break;}if(p<0)continue;uint32_t iv=qpow(y[p],Q-2);for(int k=p;k<nn;++k)if(y[k])y[k]=(__uint128_t)y[k]*iv%Q;bs.push_back({p,j,std::move(y)});std::cout<<"new rank="<<bs.size()<<" from_good_target="<<j<<" pivot_non="<<p<<" src_state="<<nonToSrc[p]<<"\n";if(bs.size()>8)break;}
 std::cout<<"dual_extra_rank="<<bs.size()<<"\n";for(size_t bi=0;bi<bs.size();++bi){auto const&b=bs[bi];size_t nz=0;std::array<size_t,5> hist{};for(auto x:b.v)if(x)++nz;std::cout<<"basis="<<bi<<" from="<<b.srcCan<<" pivot="<<b.pivot<<" nz="<<nz<<"\n";int shown=0;for(int k=0;k<nn&&shown<30;++k)if(b.v[k]){int64_t c=b.v[k]<=Q/2?b.v[k]:(int64_t)b.v[k]-Q;std::cout<<"  c="<<c<<" non="<<k<<" state="<<nonToSrc[k]<<" "<<sig8(hs[nonToSrc[k]])<<"\n";++shown;}}
 return bs.size()==5?0:1;}
