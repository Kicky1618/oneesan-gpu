#define main cotree_original_main
#include "cotree_schedule_zdd.cpp"
#undef main
#include <fstream>
int main(int ac,char**av){int n=ac>1?std::stoi(av[1]):8;std::string ord=ac>2?av[2]:"formula2";B b(n,ord,8000000);auto pk=[](int a,int c){if(a>c)std::swap(a,c);return(uint64_t)(uint32_t)a<<32|(uint32_t)c;};std::unordered_map<uint64_t,int> eid;int q=0;for(int r=0;r<b.W;++r)for(int c=0;c<b.W;++c){if(c+1<b.W)eid[pk(b.id(r,c),b.id(r,c+1))]=q++;if(r+1<b.W)eid[pk(b.id(r,c),b.id(r+1,c))]=q++;}std::vector<int>out;out.reserve(2*b.W*(b.W-1));for(int st=0;st<b.Q;++st){auto e=b.vars[st];out.push_back(eid.at(pk(e.u,e.v)));for(int v:b.at[st]){int r=v/b.W,c=v%b.W;if(r<=0)throw std::runtime_error("bad tree event");out.push_back(eid.at(pk(v,b.id(r-1,c))));}}for(int c=b.W-1;c>=1;--c)out.push_back(eid.at(pk(b.id(0,c),b.id(0,c-1))));if((int)out.size()!=2*b.W*(b.W-1))throw std::runtime_error("size "+std::to_string(out.size()));for(int x:out)std::cout<<x<<' ';std::cout<<'\n';}
