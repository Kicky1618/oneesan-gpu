#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
struct NR{int level;uint32_t lo,hi;};
struct Z{uint32_t root=0;int vars=0;std::unordered_map<int,int> edge;std::unordered_map<uint32_t,NR> node;};
static Z load(const char*p){std::ifstream f(p);if(!f)throw std::runtime_error("open");Z z;std::string s;while(f>>s){if(s=="ONEESAN_ZDD_V1")continue;if(s[0]=='#'){std::getline(f,s);continue;}if(s=="root")f>>z.root;else if(s=="variables")f>>z.vars;else if(s=="var"){int l,e,u,v;f>>l>>e>>u>>v;z.edge[l]=e;}else if(s=="node"){uint32_t id,lo,hi;int l;f>>id>>l>>lo>>hi;z.node[id]={l,lo,hi};}else if(s=="end")break;else {std::getline(f,s);}}
return z;}
static void dfs(Z const&z,uint32_t id,uint64_t mask,std::vector<uint64_t>&out){if(id==0)return;if(id==1){out.push_back(mask);return;}auto r=z.node.at(id);dfs(z,r.lo,mask,out);int e=z.edge.at(r.level);dfs(z,r.hi,mask|(1ull<<e),out);}
static bool injective(std::vector<uint64_t>const&p,uint64_t mask){std::unordered_set<uint64_t>s;s.reserve(p.size()*2);for(auto x:p)if(!s.insert(x&mask).second)return false;return true;}
int main(int ac,char**av){if(ac<2)return 2;int trials=ac>2?atoi(av[2]):10000;auto z=load(av[1]);if(z.vars>64){std::cerr<<"vars>64\n";return 2;}std::vector<uint64_t>p;dfs(z,z.root,0,p);int E=0;for(auto const&kv:z.edge)E=std::max(E,kv.second+1);int lb=0;while((1ull<<lb)<p.size())++lb;std::cout<<"paths="<<p.size()<<" E="<<E<<" lb="<<lb<<"\n";std::mt19937_64 rng(1618);uint64_t best=E==64?~0ull:((1ull<<E)-1);int bn=E;
 for(int tr=0;tr<trials;++tr){std::vector<int>es(E);for(int i=0;i<E;++i)es[i]=i;std::shuffle(es.begin(),es.end(),rng);uint64_t m=E==64?~0ull:((1ull<<E)-1);int n=E;for(int e:es){auto q=m&~(1ull<<e);if(injective(p,q)){m=q;--n;}}if(n<bn){bn=n;best=m;std::cout<<"trial="<<tr<<" best="<<bn<<" mask=0x"<<std::hex<<best<<std::dec<<" edges=";for(int i=0;i<E;++i)if(best>>i&1)std::cout<<i<<',';std::cout<<"\n";if(bn==lb)break;}}
 // Random exact-k search below greedy best.
 for(int k=lb;k<bn;++k){int found=0;for(int tr=0;tr<trials*10;++tr){std::vector<int>es(E);for(int i=0;i<E;++i)es[i]=i;std::shuffle(es.begin(),es.end(),rng);uint64_t m=0;for(int i=0;i<k;++i)m|=1ull<<es[i];if(injective(p,m)){std::cout<<"EXACT k="<<k<<" mask=0x"<<std::hex<<m<<std::dec<<" edges=";for(int i=0;i<E;++i)if(m>>i&1)std::cout<<i<<',';std::cout<<"\n";found=1;bn=k;best=m;break;}}if(found)break;}
 std::cout<<"FINAL "<<bn<<"\n";
}
