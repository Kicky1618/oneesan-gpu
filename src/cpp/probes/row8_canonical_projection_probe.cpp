#define ROW8_RAW_PREFIX_NO_MAIN 1
#include "row8_raw_prefix_vector.cpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr int BD[9]={1107,1640,1428,888,420,152,42,8,1};
static constexpr int BDEL[3]={0,-1,1};
struct Cache {std::array<std::array<std::vector<uint32_t>,9>,3> M;};
static Cache load_cache_file(std::string path,uint32_t mod){
 std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cache open");char magic[8];uint32_t ver,pm,ds[9];uint64_t fp,abi,an,bn,cnt[27],hash;
 in.read(magic,8);in.read((char*)&ver,4);in.read((char*)&pm,4);in.read((char*)&fp,8);in.read((char*)&abi,8);in.read((char*)ds,36);in.read((char*)&an,8);in.read((char*)&bn,8);in.read((char*)cnt,sizeof(cnt));in.read((char*)&hash,8);
 if(std::string(magic,7)!="R8MDC02"||ver!=2||pm!=mod)throw std::runtime_error("cache header");in.seekg((std::streamoff)(4*(an+bn)),std::ios::cur);Cache z;int q=0;
 for(int a=0;a<3;++a)for(int h=0;h<9;++h){z.M[a][h].resize(cnt[q++]);if(!z.M[a][h].empty())in.read((char*)z.M[a][h].data(),z.M[a][h].size()*4);}if(!in)throw std::runtime_error("cache read");return z;
}
static std::string find_cache(uint32_t mod){for(auto const&e:std::filesystem::directory_iterator("work/row8_mod_cache")){auto s=e.path().filename().string();if(s.find("row8_mod_"+std::to_string(mod)+"_")==0&&s.find("_52384d4443414249.bin")!=std::string::npos)return e.path();}throw std::runtime_error("cache missing");}
static std::array<std::vector<uint32_t>,9> load_pc(){std::ifstream in("src/cuda/b300/row8_pivots_w19.bin",std::ios::binary);char m[8];uint32_t v,ds[9];in.read(m,8);in.read((char*)&v,4);in.read((char*)ds,36);std::array<std::vector<uint32_t>,9> p;for(int h=0;h<9;++h){p[h].resize(ds[h]);for(auto&x:p[h]){uint32_t sc;in.read((char*)&x,4);in.read((char*)&sc,4);}}return p;}

static bool encode_word(std::string const& tok,Packed&out){
 if(tok.size()!=8)return false;int h=0,bal=0;for(char c:tok){if(c=='T'){if(bal)return false;++h;}else if(c=='U')++bal;else if(c=='D')--bal;else if(c!='N')return false;}if(bal)return false;
 State s{};s.n=8;s.sp=h;std::vector<int>tp;for(int i=0;i<8;++i)if(tok[i]=='T')tp.push_back(i);int next=1;for(int k=0;k<h;++k){int q=next++,i=tp[k];s.deg[i]=1;s.comp[i]=q;s.stack[k]=q;}
 std::vector<std::pair<int,int>>open;bal=0;for(int i=0;i<8;++i){char c=tok[i];if(c=='T'){if(bal||!open.empty())return false;continue;}if(c=='N')continue;int st=c=='U'?1:-1;bool opens=bal==0||(bal>0&&st>0)||(bal<0&&st<0);if(opens){open.push_back({i,st});bal+=st;}else{if(open.empty())return false;auto[j,sg]=open.back();open.pop_back();if(sg!=-st)return false;int q=next++;s.deg[j]=s.deg[i]=1;s.comp[j]=s.comp[i]=q;s.status[q]=sg<0;bal+=st;}}
 if(bal||!open.empty())return false;s.ns=next;out=pack(s);return true;
}
static void gen_rec(int pos,int needT,int bal,std::string&w,std::vector<Packed>&out){if(pos==8){if(!needT&&!bal){Packed p;if(!encode_word(w,p))throw std::runtime_error("encode");out.push_back(p);}return;}if(needT>8-pos)return;w[pos]='N';gen_rec(pos+1,needT,bal,w,out);w[pos]='U';gen_rec(pos+1,needT,bal+1,w,out);w[pos]='D';gen_rec(pos+1,needT,bal-1,w,out);if(needT&&bal==0){w[pos]='T';gen_rec(pos+1,needT-1,0,w,out);}}
static std::vector<Packed> canonical(int h){std::string w(8,'N');std::vector<Packed>v;gen_rec(0,h,0,w,v);std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());if((int)v.size()!=BD[h])throw std::runtime_error("canonical size");return v;}
static std::vector<uint32_t> project(WVec const&v,std::vector<Packed>const&b){std::vector<uint32_t>z(b.size());size_t i=0,j=0;while(i<v.size()&&j<b.size()){if(v[i].p<b[j])++i;else if(b[j]<v[i].p)++j;else{z[j]=v[i].v;++i;++j;}}return z;}
static int rank_mod(std::vector<uint32_t>A,int n,uint32_t p){int r=0;for(int c=0;c<n&&r<n;++c){int q=r;while(q<n&&!A[(size_t)q*n+c])++q;if(q==n)continue;if(q!=r)for(int j=c;j<n;++j)std::swap(A[(size_t)q*n+j],A[(size_t)r*n+j]);uint64_t a=A[(size_t)r*n+c],e=p-2,iv=1;while(e){if(e&1)iv=(__uint128_t)iv*a%p;a=(__uint128_t)a*a%p;e>>=1;}for(int j=c;j<n;++j)A[(size_t)r*n+j]=(__uint128_t)A[(size_t)r*n+j]*iv%p;for(int i=0;i<n;++i)if(i!=r&&A[(size_t)i*n+c]){uint32_t f=A[(size_t)i*n+c];for(int j=c;j<n;++j)A[(size_t)i*n+j]=(A[(size_t)i*n+j]+p-(__uint128_t)f*A[(size_t)r*n+j]%p)%p;}++r;}return r;}

int main(int argc,char**argv){MODP=argc>1?std::strtoul(argv[1],nullptr,10):1000000007u;int h=argc>2?std::atoi(argv[2]):7;int a=argc>3?std::atoi(argv[3]):0;int h2=h+BDEL[a];if(h<0||h>8||h2<0||h2>8)return 2;auto P=load_pc();auto C=load_cache_file(find_cache(MODP),MODP);auto bs=canonical(h),bt=canonical(h2);
 std::vector<uint32_t>A((size_t)BD[h]*BD[h]),At((size_t)BD[h2]*BD[h2]);std::vector<WVec> src(BD[h]);
 auto loadA=[&](int hh,std::vector<uint32_t>&mat)->bool{std::string path="work/formal-probes/canonical-matrix/A_h"+std::to_string(hh)+"_mod"+std::to_string(MODP)+".bin";std::ifstream in(path,std::ios::binary);if(!in)return false;struct H{char magic[8];uint32_t ver,mod,h,dim;uint64_t count,hash;}x{};in.read((char*)&x,sizeof(x));if(!in||std::string(x.magic,8)!="R8CANA01"||x.ver!=1||x.mod!=MODP||x.h!=(uint32_t)hh||x.dim!=(uint32_t)BD[hh]||x.count!=mat.size())return false;in.read((char*)mat.data(),mat.size()*4);if(!in)return false;std::cerr<<"A cache hit h="<<hh<<" path="<<path<<"\n";return true;};
 auto build=[&](int hh,std::vector<Packed>const&b,std::vector<uint32_t>&mat,std::vector<WVec>*keep){if(!keep&&loadA(hh,mat))return;for(int i=0;i<BD[hh];++i){auto v=prefix_vec(P[hh][i]);auto q=project(v,b);std::copy(q.begin(),q.end(),mat.begin()+(size_t)i*BD[hh]);if(keep)(*keep)[i]=std::move(v);if((i&31)==0)std::cerr<<"A h="<<hh<<" "<<i+1<<"/"<<BD[hh]<<"\n";}};
 build(h,bs,A,&src); if(h2==h)At=A; else build(h2,bt,At,nullptr);
 std::cout<<"A_rank h="<<h<<" rank="<<rank_mod(A,BD[h],MODP)<<"/"<<BD[h]<<"\n";if(h2!=h)std::cout<<"A_rank h="<<h2<<" rank="<<rank_mod(At,BD[h2],MODP)<<"/"<<BD[h2]<<"\n";
 size_t bad=0;uint64_t hashE=1469598103934665603ULL,hashP=hashE;auto const&M=C.M[a][h];
 for(int i=0;i<BD[h];++i){auto ext=wcolumn(src[i],8,false,a);auto e=project(ext,bt);for(int k=0;k<BD[h2];++k){__uint128_t s=0;for(int j=0;j<BD[h2];++j)s+=(__uint128_t)M[(size_t)i*BD[h2]+j]*At[(size_t)j*BD[h2]+k];uint32_t pred=s%MODP;if(e[k]!=pred){if(bad<8)std::cerr<<"mismatch i="<<i<<" k="<<k<<" got="<<e[k]<<" pred="<<pred<<"\n";++bad;}hashE^=e[k];hashE*=1099511628211ULL;hashP^=pred;hashP*=1099511628211ULL;}if((i&31)==0)std::cerr<<"E "<<i+1<<"/"<<BD[h]<<"\n";}
 std::cout<<"canonical_projection h="<<h<<" sym="<<a<<" h2="<<h2<<" bad="<<bad<<" hashE="<<std::hex<<hashE<<" hashP="<<hashP<<std::dec<<" exact="<<(bad==0)<<"\n";return bad?1:0;
}
