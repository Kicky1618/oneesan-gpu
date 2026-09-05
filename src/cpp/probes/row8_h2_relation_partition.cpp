#define ROW8_CANONICAL_TRIE_NO_MAIN 1
#include "row8_canonical_trie_verify.cpp"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
using Sig=std::vector<std::pair<int,uint32_t>>;
static void rw2(int p,int nt,int bal,std::string&w,std::vector<std::pair<Packed,std::string>>&o){if(p==8){if(!nt&&!bal){Packed x;if(enc(w,x))o.push_back({x,w});}return;}if(nt>8-p)return;w[p]='N';rw2(p+1,nt,bal,w,o);w[p]='U';rw2(p+1,nt,bal+1,w,o);w[p]='D';rw2(p+1,nt,bal-1,w,o);if(nt&&bal==0){w[p]='T';rw2(p+1,nt-1,0,w,o);}}
static std::vector<std::pair<Packed,std::string>> words2(){std::string w(8,'N');std::vector<std::pair<Packed,std::string>>v;rw2(0,2,0,w,v);std::sort(v.begin(),v.end(),[](auto&a,auto&b){return a.first<b.first;});v.erase(std::unique(v.begin(),v.end(),[](auto&a,auto&b){return a.first==b.first;}),v.end());return v;}
#ifndef ROW8_H2_PARTITION_NO_MAIN
int main(){MODP=1000000007u;Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::vector<Packed>hs;for(auto&p:all)if(unpack(p).sp==2)hs.push_back(p);auto ws=words2();std::set<int>bad;{std::ifstream in("work/formal-probes/canonical-matrix/h2_actual_bad_indices.txt");int x;while(in>>x)bad.insert(x);}if(bad.size()!=120)throw std::runtime_error("bad120");std::vector<std::vector<std::pair<int,uint32_t>>>T(hs.size());size_t ed=0;for(int i=0;i<(int)hs.size();++i){WVec v{{hs[i],1}};auto z=wcolumn(std::move(v),8,false,0);for(auto&e:z){auto it=std::lower_bound(hs.begin(),hs.end(),e.p);if(it==hs.end()||!(*it==e.p))throw std::runtime_error("target missing");T[i].push_back({int(it-hs.begin()),e.v});++ed;}}
 std::vector<int> anchor(hs.size(),-1),cls(hs.size(),-1);int nc=0;for(int j=0;j<(int)ws.size();++j)if(!bad.count(j)){auto it=std::lower_bound(hs.begin(),hs.end(),ws[j].first);if(it==hs.end()||!(*it==ws[j].first))throw std::runtime_error("good missing");int i=it-hs.begin();anchor[i]=cls[i]=nc++;}std::cout<<"raw="<<hs.size()<<" edges="<<ed<<" initial_classes="<<nc<<"\n";
 for(int round=1;round<=10;++round){auto sig=[&](int i){Sig s;for(auto[t,c]:T[i])if(cls[t]>=0)s.push_back({cls[t],c});std::sort(s.begin(),s.end());Sig o;for(auto [q,c]:s){if(!o.empty()&&o.back().first==q)o.back().second+=c;else o.push_back({q,c});}return o;};std::map<Sig,std::vector<int>> groups;int outsideNon=0;for(int i=0;i<(int)hs.size();++i)if(anchor[i]<0){auto s=sig(i);if(!s.empty()){groups[s].push_back(i);if(cls[i]<0)++outsideNon;}}
   std::vector<int> ncls=cls;int next=1308; // anchors keep 0..1307
   // Repartition every non-anchor state with a nonzero signature by current signature.
   for(auto const&[s,v]:groups){for(int i:v)ncls[i]=next;++next;}
   int support=0;for(int x:ncls)if(x>=0)++support;int hardCov=0,hardBij=0;std::set<int>hclasses;for(int j:bad){auto it=std::lower_bound(hs.begin(),hs.end(),ws[j].first);int i=it-hs.begin();if(ncls[i]>=0){++hardCov;hclasses.insert(ncls[i]);}}
   std::map<int,int> sizes;for(int x:ncls)if(x>=1308)++sizes[x];std::map<int,int>sh;for(auto [q,z]:sizes)++sh[z];bool same=ncls==cls;cls.swap(ncls);nc=next;std::cout<<"round="<<round<<" classes="<<nc<<" extra="<<nc-1308<<" support="<<support<<" outsideNon="<<outsideNon<<" hardCov="<<hardCov<<" hardClasses="<<hclasses.size()<<" sizeHist=";for(auto[z,c]:sh)std::cout<<z<<':'<<c<<',';std::cout<<" stable="<<same<<"\n";if(same)break;}
 // Verify class lumpability and no incoming projection from outside.
 size_t outsideHit=0,classBad=0;std::map<int,Sig>rep;for(int i=0;i<(int)hs.size();++i){Sig s;for(auto[t,c]:T[i])if(cls[t]>=0)s.push_back({cls[t],c});std::sort(s.begin(),s.end());Sig o;for(auto[q,c]:s){if(!o.empty()&&o.back().first==q)o.back().second+=c;else o.push_back({q,c});}if(cls[i]<0){if(!o.empty())++outsideHit;}else if(cls[i]>=1308){auto it=rep.find(cls[i]);if(it==rep.end())rep[cls[i]]=o;else if(it->second!=o)++classBad;}}
 std::cout<<"final_classes="<<nc<<" outsideHit="<<outsideHit<<" classBad="<<classBad<<"\n";return 0;}
#endif
