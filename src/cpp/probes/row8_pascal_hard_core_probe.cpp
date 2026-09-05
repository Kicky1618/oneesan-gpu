#define ROW8_H2_PARTITION_NO_MAIN 1
#include "row8_h2_relation_partition.cpp"
#include <bit>
#include <fstream>
#include <map>
#include <set>

static uint32_t amask(std::string const&w){uint32_t m=0;for(int i=0;i<8;++i)if(w[i]!='N')m|=1u<<i;return m;}
static std::string core(std::string const&w){std::string s;for(char c:w)if(c!='N')s+=c;return s;}
static std::string bits(uint32_t m){std::string s;for(int i=0;i<8;++i)s+=((m>>i)&1)?'1':'0';return s;}
int main(){
 auto ws=words2(); std::set<int> bad; {std::ifstream in("work/formal-probes/canonical-matrix/h2_actual_bad_indices.txt");int x;while(in>>x)bad.insert(x);} 
 std::map<int,std::map<uint32_t,std::set<std::string>>> g;
 for(int i:bad){auto const&w=ws.at(i).second;auto m=amask(w);g[std::popcount(m)][m].insert(core(w));}
 std::cout<<"bad="<<bad.size()<<"\n";
 for(auto const&[j,mm]:g){std::set<std::string> uni,ref;bool same=true,first=true;size_t lo=-1,hi=0;for(auto const&[m,s]:mm){lo=std::min(lo,s.size());hi=std::max(hi,s.size());uni.insert(s.begin(),s.end());if(first){ref=s;first=false;}else if(s!=ref)same=false;}
  std::cout<<"j="<<j<<" masks="<<mm.size()<<" per_mask="<<lo<<".."<<hi<<" same="<<same<<" union="<<uni.size()<<"\n";
  if(uni.size()<=30){for(auto const&s:uni)std::cout<<"  "<<s<<"\n";}
  int shown=0;for(auto const&[m,s]:mm)if(shown++<3){std::cout<<" mask="<<bits(m)<<" n="<<s.size()<<"\n";}
 }
}
