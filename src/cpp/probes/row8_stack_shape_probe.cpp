#define main rowr_checkpoint_embedded_main
#include "rowr_column_batch_wfa_checkpoint.cpp"
#undef main
#include <iostream>
#include <set>
#include <map>
int main(){Vec v;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,v);size_t n8=0,dup=0,p0=0,p0show=0;std::map<int,size_t> distinctHist;
for(auto const&p:v){State s=unpack(p);if(s.sp!=8)continue;++n8;std::set<int>st;for(int k=0;k<s.sp;++k)st.insert(s.stack[k]);if((int)st.size()!=s.sp)++dup;++distinctHist[st.size()];int occ=0;for(int i=0;i<8;++i)occ+=s.deg[i]!=0;if(occ==0){++p0;if(p0show++<8){std::cout<<"p0 stack=";for(int k=0;k<s.sp;++k)std::cout<<int(s.stack[k])<<(k+1==s.sp?" ":",");std::cout<<"ns="<<int(s.ns)<<" status=";for(int q=1;q<s.ns;++q)std::cout<<int(s.status[q]);std::cout<<"\n";}}}
std::cout<<"h8="<<n8<<" duplicate_stack="<<dup<<" p0="<<p0<<" distinctHist";for(auto[k,c]:distinctHist)std::cout<<' '<<k<<':'<<c;std::cout<<"\n";}
