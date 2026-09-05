#define ROW8_RAW_PREFIX_NO_MAIN 1
#include "row8_raw_prefix_vector.cpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
static constexpr uint32_t MOD=1000000007u; static constexpr int N4=420,N5=152;
struct PH{size_t operator()(Packed const&p)const noexcept{uint64_t h=0x9e3779b97f4a7c15ULL;for(auto x:p.w){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;x^=x>>31;h^=x+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);}return h;}};
struct Cand{std::vector<std::pair<uint16_t,uint32_t>> a,e;};
static std::array<std::vector<uint32_t>,9> loadpc(){std::ifstream in("src/cuda/b300/row8_pivots_w19.bin",std::ios::binary);char m[8];uint32_t v,ds[9];in.read(m,8);in.read((char*)&v,4);in.read((char*)ds,36);std::array<std::vector<uint32_t>,9>p;for(int h=0;h<9;++h){p[h].resize(ds[h]);for(auto&x:p[h]){uint32_t sc;in.read((char*)&x,4);in.read((char*)&sc,4);}}return p;}
static std::vector<uint32_t> loadM(){std::string p;for(auto const&e:std::filesystem::directory_iterator("work/row8_mod_cache")){auto s=e.path().filename().string();if(s.find("row8_mod_1000000007_")==0&&s.find("_52384d4443414249.bin")!=std::string::npos){p=e.path();break;}}std::ifstream in(p,std::ios::binary);char magic[8];uint32_t ver,pm,ds[9];uint64_t fp,abi,an,bn,cnt[27],hash;in.read(magic,8);in.read((char*)&ver,4);in.read((char*)&pm,4);in.read((char*)&fp,8);in.read((char*)&abi,8);in.read((char*)ds,36);in.read((char*)&an,8);in.read((char*)&bn,8);in.read((char*)cnt,sizeof(cnt));in.read((char*)&hash,8);in.seekg((std::streamoff)(4*(an+bn)),std::ios::cur);int q=0;for(int a=0;a<3;++a)for(int h=0;h<9;++h){size_t n=cnt[q++];if(a==1&&h==5){std::vector<uint32_t>M(n);in.read((char*)M.data(),n*4);return M;}in.seekg((std::streamoff)(n*4),std::ios::cur);}throw std::runtime_error("M");}
static std::vector<uint32_t> loadA4(){std::vector<uint32_t>A((size_t)N4*N4);std::ifstream in("work/formal-probes/canonical-matrix/A_h4_mod1000000007.bin",std::ios::binary);struct H{char m[8];uint32_t v,mod,h,d;uint64_t c,hash;}x{};in.read((char*)&x,sizeof(x));in.read((char*)A.data(),A.size()*4);return A;}
struct Node{std::array<std::unique_ptr<Node>,3>ch;int row=-1;};
static std::array<int,9>d9(uint32_t x){std::array<int,9>d{};for(int i=8;i>=0;--i){d[i]=x%3;x/=3;}return d;}
static Node trie(std::vector<uint32_t>const&codes){Node root;for(int i=0;i<(int)codes.size();++i){auto d=d9(codes[i]);Node*t=&root;for(int a:d){if(!t->ch[a])t->ch[a]=std::make_unique<Node>();t=t->ch[a].get();}t->row=i;}return root;}
static void dfs_rows(Node const&n,int depth,WVec const&v,std::vector<WVec>&rows,bool extendR){if(depth==9){auto z=extendR?wcolumn(v,8,false,1):v;rows[n.row]=std::move(z);if((n.row&31)==0)std::cerr<<(extendR?"h5R ":"h4 ")<<n.row+1<<"/"<<rows.size()<<" support="<<rows[n.row].size()<<"\n";return;}for(int a=0;a<3;++a)if(n.ch[a]){auto z=wcolumn(v,8,depth==0,a);dfs_rows(*n.ch[a],depth+1,z,rows,extendR);}}
static uint32_t inv(uint32_t a){uint64_t x=a,e=MOD-2,r=1;while(e){if(e&1)r=(__uint128_t)r*x%MOD;x=(__uint128_t)x*x%MOD;e>>=1;}return r;}
struct LinBasis{std::array<std::vector<uint32_t>,N4>b;int rank=0;bool add(std::vector<uint32_t>x){for(int p=0;p<N4;++p)if(x[p]){if(!b[p].empty()){uint32_t f=x[p];for(int j=p;j<N4;++j)x[j]=(x[j]+MOD-(__uint128_t)f*b[p][j]%MOD)%MOD;}else{uint32_t iv=inv(x[p]);for(int j=p;j<N4;++j)x[j]=(__uint128_t)x[j]*iv%MOD;b[p]=std::move(x);++rank;return true;}}return false;}};
static std::string ss(Packed p){State s=unpack(p);std::string z="deg";for(int i=0;i<8;++i)z+=char('0'+s.deg[i]);z+=" c";for(int i=0;i<8;++i){z+=std::to_string(s.comp[i]);z+=',';}z+=" st";for(int q=1;q<s.ns;++q)z+=char('0'+(s.status[q]&1));z+=" sp"+std::to_string(s.sp)+":";for(int i=0;i<s.sp;++i){z+=std::to_string(s.stack[i]);z+=',';}return z;}
static int complexity(Packed p){State s=unpack(p);int c=0;for(int q=1;q<s.ns;++q)c+=3*(s.status[q]&1);for(int i=0;i<8;++i)c+=s.deg[i];return c;}
int main(){MODP=MOD;auto pc=loadpc(); auto M=loadM(); auto A4=loadA4();State z{};z.n=8;WVec init{{pack(z),1}};auto t4=trie(pc[4]);std::vector<WVec>R4(N4);dfs_rows(t4,0,init,R4,false);auto t5=trie(pc[5]);std::vector<WVec>E5(N5);dfs_rows(t5,0,init,E5,true);
 std::unordered_map<Packed,Cand,PH> C;size_t entries=0;for(int i=0;i<N4;++i)for(auto const&e:R4[i]){C[e.p].a.push_back({(uint16_t)i,e.v});++entries;}std::cerr<<"candidates="<<C.size()<<" h4entries="<<entries<<"\n";for(int i=0;i<N5;++i)for(auto const&e:E5[i]){auto it=C.find(e.p);if(it!=C.end())it->second.e.push_back({(uint16_t)i,e.v});}
 // Start with the 415 known-good canonical columns from A4.
 LinBasis B;int badk[5]={337,351,371,397,417};std::array<bool,N4>bad{};for(int x:badk)bad[x]=true;for(int k=0;k<N4;++k)if(!bad[k]){std::vector<uint32_t>x(N4);for(int i=0;i<N4;++i)x[i]=A4[(size_t)i*N4+k];if(!B.add(std::move(x)))throw std::runtime_error("canonical 415 dependent");}std::cout<<"initial_rank="<<B.rank<<"\n";
 struct Good{Packed p;Cand*c;int comp;};std::vector<Good>good;std::array<uint32_t,N5>act{},pred{};size_t checked=0,ngood=0;for(auto&[p,c]:C){act.fill(0);pred.fill(0);for(auto [i,v]:c.e)act[i]=v;for(auto [j,v]:c.a)for(int i=0;i<N5;++i)pred[i]=(pred[i]+(__uint128_t)M[(size_t)i*N4+j]*v)%MOD;bool ok=true;for(int i=0;i<N5;++i)if(act[i]!=pred[i]){ok=false;break;}++checked;if(ok){++ngood;good.push_back({p,&c,complexity(p)});}}
 std::cout<<"checked="<<checked<<" good="<<ngood<<"\n";std::sort(good.begin(),good.end(),[](auto const&a,auto const&b){if(a.comp!=b.comp)return a.comp<b.comp;return a.p<b.p;});int added=0;for(auto&g:good){std::vector<uint32_t>x(N4);for(auto [i,v]:g.c->a)x[i]=v;if(B.add(std::move(x))){std::cout<<"replacement="<<added<<" rank="<<B.rank<<" complexity="<<g.comp<<" "<<ss(g.p)<<"\n";++added;if(B.rank==N4)break;}}std::cout<<"final_rank="<<B.rank<<" replacements="<<added<<"\n";return B.rank==N4?0:1;}
