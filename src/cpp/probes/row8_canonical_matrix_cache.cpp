#define ROW8_RAW_PREFIX_NO_MAIN 1
#include "row8_raw_prefix_vector.cpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

static constexpr int CD[9]={1107,1640,1428,888,420,152,42,8,1};

static std::array<std::vector<uint32_t>,9> load_pc_only(){
 std::ifstream in("src/cuda/b300/row8_pivots_w19.bin",std::ios::binary);char m[8];uint32_t v,ds[9];in.read(m,8);in.read((char*)&v,4);in.read((char*)ds,36);
 std::array<std::vector<uint32_t>,9> p;for(int h=0;h<9;++h){p[h].resize(ds[h]);for(auto&x:p[h]){uint32_t sc;in.read((char*)&x,4);in.read((char*)&sc,4);}}return p;
}
static bool encode_word(std::string const&tok,Packed&out){if(tok.size()!=8)return false;int h=0,bal=0;for(char c:tok){if(c=='T'){if(bal)return false;++h;}else if(c=='U')++bal;else if(c=='D')--bal;else if(c!='N')return false;}if(bal)return false;State s{};s.n=8;s.sp=h;std::vector<int>tp;for(int i=0;i<8;++i)if(tok[i]=='T')tp.push_back(i);int next=1;for(int k=0;k<h;++k){int q=next++,i=tp[k];s.deg[i]=1;s.comp[i]=q;s.stack[k]=q;}std::vector<std::pair<int,int>>open;bal=0;for(int i=0;i<8;++i){char c=tok[i];if(c=='T'){if(bal||!open.empty())return false;continue;}if(c=='N')continue;int st=c=='U'?1:-1;bool op=bal==0||(bal>0&&st>0)||(bal<0&&st<0);if(op){open.push_back({i,st});bal+=st;}else{if(open.empty())return false;auto[j,sg]=open.back();open.pop_back();if(sg!=-st)return false;int q=next++;s.deg[j]=s.deg[i]=1;s.comp[j]=s.comp[i]=q;s.status[q]=sg<0;bal+=st;}}if(bal||!open.empty())return false;s.ns=next;out=pack(s);return true;}
static void gen_rec(int pos,int needT,int bal,std::string&w,std::vector<Packed>&out){if(pos==8){if(!needT&&!bal){Packed p;if(!encode_word(w,p))throw std::runtime_error("encode");out.push_back(p);}return;}if(needT>8-pos)return;w[pos]='N';gen_rec(pos+1,needT,bal,w,out);w[pos]='U';gen_rec(pos+1,needT,bal+1,w,out);w[pos]='D';gen_rec(pos+1,needT,bal-1,w,out);if(needT&&bal==0){w[pos]='T';gen_rec(pos+1,needT-1,0,w,out);}}
static std::vector<Packed> canonical(int h){std::string w(8,'N');std::vector<Packed>v;gen_rec(0,h,0,w,v);std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());if((int)v.size()!=CD[h])throw std::runtime_error("canonical size");return v;}
static std::vector<uint32_t> project(WVec const&v,std::vector<Packed>const&b){std::vector<uint32_t>z(b.size());size_t i=0,j=0;while(i<v.size()&&j<b.size()){if(v[i].p<b[j])++i;else if(b[j]<v[i].p)++j;else{z[j]=v[i].v;++i;++j;}}return z;}
struct Node{std::array<std::unique_ptr<Node>,3> ch;int row=-1;};
static std::array<int,9> digits9c(uint32_t x){std::array<int,9>d{};for(int i=8;i>=0;--i){d[i]=x%3;x/=3;}return d;}
static void dfs(Node const&n,int depth,WVec const&v,std::vector<Packed>const&basis,std::vector<uint32_t>&A,int dim,size_t&nodes){++nodes;if(depth==9){if(n.row<0)throw std::runtime_error("leaf row");auto q=project(v,basis);std::copy(q.begin(),q.end(),A.begin()+(size_t)n.row*dim);if((n.row&31)==0)std::cerr<<"leaf "<<n.row+1<<"/"<<dim<<" support="<<v.size()<<"\n";return;}for(int a=0;a<3;++a)if(n.ch[a]){auto child=wcolumn(v,8,depth==0,a);dfs(*n.ch[a],depth+1,child,basis,A,dim,nodes);}}
static uint64_t fnv(void const*p,size_t n,uint64_t h=1469598103934665603ULL){auto*b=(unsigned char const*)p;for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ULL;}return h;}
int main(int argc,char**argv){MODP=argc>1?std::strtoul(argv[1],nullptr,10):1000000007u;int h=argc>2?std::atoi(argv[2]):5;if(h<0||h>8)return 2;auto P=load_pc_only();int n=CD[h];Node root;for(int i=0;i<n;++i){auto d=digits9c(P[h][i]);Node*t=&root;for(int a:d){if(!t->ch[a])t->ch[a]=std::make_unique<Node>();t=t->ch[a].get();}if(t->row>=0)throw std::runtime_error("duplicate pc");t->row=i;}auto basis=canonical(h);std::vector<uint32_t>A((size_t)n*n);State z{};z.n=8;WVec init{{pack(z),1}};size_t nodes=0;auto t0=std::chrono::steady_clock::now();dfs(root,0,init,basis,A,n,nodes);double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();std::filesystem::create_directories("work/formal-probes/canonical-matrix");std::string path="work/formal-probes/canonical-matrix/A_h"+std::to_string(h)+"_mod"+std::to_string(MODP)+".bin";struct H{char magic[8];uint32_t ver,mod,hh,dim;uint64_t count,hash;}hdr{{'R','8','C','A','N','A','0','1'},1,MODP,(uint32_t)h,(uint32_t)n,(uint64_t)A.size(),fnv(A.data(),A.size()*4)};std::ofstream out(path,std::ios::binary|std::ios::trunc);out.write((char*)&hdr,sizeof(hdr));out.write((char*)A.data(),A.size()*4);out.close();if(!out)throw std::runtime_error("write");std::cout<<"h="<<h<<" dim="<<n<<" trie_nodes="<<nodes<<" sec="<<sec<<" hash="<<std::hex<<hdr.hash<<std::dec<<" path="<<path<<"\n";}
