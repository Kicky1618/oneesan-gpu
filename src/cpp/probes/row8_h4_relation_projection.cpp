#define ROW8_CANONICAL_TRIE_NO_MAIN 1
#include "row8_canonical_trie_verify.cpp"
#include <chrono>
#include <functional>
#include <iostream>
static constexpr uint32_t Q=1000000007u; static constexpr int H4=420,H5=152;
static Packed relstate(int gap,bool nested){State s{};s.n=8;s.sp=4;int q=1,sk=0;std::vector<int> f;for(int i=0;i<8;++i){if(gap<=i&&i<gap+4)f.push_back(i);else{int z=q++;s.deg[i]=1;s.comp[i]=z;s.stack[sk++]=z;}}auto add=[&](int a,int b,int st){int z=q++;s.deg[a]=s.deg[b]=1;s.comp[a]=s.comp[b]=z;s.status[z]=st;};if(!nested){add(f[0],f[1],1);add(f[2],f[3],1);}else{add(f[0],f[3],1);add(f[1],f[2],0);}s.ns=q;return pack(s);}
static uint32_t coef(WVec const&v,Packed p){auto it=std::lower_bound(v.begin(),v.end(),WEntry{p,0},[](auto const&a,auto const&b){return a.p<b.p;});return it!=v.end()&&it->p==p?it->v:0;}
struct TNode{std::array<std::unique_ptr<TNode>,3>ch;int row=-1;};
static TNode mkTrie(std::vector<uint32_t>const&codes){TNode root;for(int i=0;i<(int)codes.size();++i){auto d=d9(codes[i]);TNode*t=&root;for(int a:d){if(!t->ch[a])t->ch[a]=std::make_unique<TNode>();t=t->ch[a].get();}t->row=i;}return root;}
static void walk(TNode const&n,int depth,WVec const&v,std::function<void(int,WVec const&)>const&leaf){if(depth==9){leaf(n.row,v);return;}for(int a=0;a<3;++a)if(n.ch[a]){auto z=wcolumn(v,8,depth==0,a);walk(*n.ch[a],depth+1,z,leaf);}}
int main(){MODP=Q;auto P=loadpc();auto C=loadM(Q);auto can=basis(4);std::vector<uint32_t>A4((size_t)H4*H4);if(!loadA(4,Q,A4))throw std::runtime_error("A4 cache");std::vector<uint32_t>Arel=A4,Erel((size_t)H5*H4);int hk[5]={337,351,371,397,417};std::array<Packed,5> dis,mix;for(int g=0;g<5;++g){dis[g]=relstate(g,false);mix[g]=relstate(g,true);if(!(dis[g]==can[hk[g]]))throw std::runtime_error("hard ordering mismatch g="+std::to_string(g));}
 State z{};z.n=8;WVec init{{pack(z),1}};auto t0=std::chrono::steady_clock::now();auto tr4=mkTrie(P[4]);walk(tr4,0,init,[&](int i,WVec const&v){for(int g=0;g<5;++g)Arel[(size_t)i*H4+hk[g]]=(coef(v,dis[g])+coef(v,mix[g]))%Q;if((i&63)==0)std::cerr<<"Arel "<<i+1<<"/420\n";});double s4=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
 auto tr5=mkTrie(P[5]);t0=std::chrono::steady_clock::now();walk(tr5,0,init,[&](int i,WVec const&v){auto e=wcolumn(v,8,false,1);auto q=proj(e,can);std::copy(q.begin(),q.end(),Erel.begin()+(size_t)i*H4);for(int g=0;g<5;++g)Erel[(size_t)i*H4+hk[g]]=(coef(e,dis[g])+coef(e,mix[g]))%Q;if((i&31)==0)std::cerr<<"Erel "<<i+1<<"/152\n";});double s5=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
 int rk=rankmod(Arel,H4,Q);auto const&M=C.M[1][5];size_t bad=0;std::array<size_t,5> bh{};for(int i=0;i<H5;++i)for(int k=0;k<H4;++k){__uint128_t s=0;for(int j=0;j<H4;++j)s+=(__uint128_t)M[(size_t)i*H4+j]*Arel[(size_t)j*H4+k];uint32_t p=s%Q,g=Erel[(size_t)i*H4+k];if(p!=g){if(bad<12)std::cerr<<"bad i="<<i<<" k="<<k<<" got="<<g<<" pred="<<p<<"\n";++bad;for(int q=0;q<5;++q)if(k==hk[q])++bh[q];}}
 std::cout<<"Arel_rank="<<rk<<"/420 buildA_s="<<s4<<" buildE_s="<<s5<<" bad="<<bad<<" hard_bad=";for(auto x:bh)std::cout<<x<<',';std::cout<<" exact="<<(bad==0)<<"\n";return bad?1:0;}
