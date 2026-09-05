#define ROW8_RAW_PREFIX_NO_MAIN 1
#include "row8_raw_prefix_vector.cpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

static constexpr int D4=420,D5=152;
struct MC{std::vector<uint32_t>M;};
static std::vector<uint32_t> loadM(){uint32_t mod=1000000007;std::string p;for(auto const&e:std::filesystem::directory_iterator("work/row8_mod_cache")){auto s=e.path().filename().string();if(s.find("row8_mod_1000000007_")==0&&s.find("_52384d4443414249.bin")!=std::string::npos){p=e.path();break;}}std::ifstream in(p,std::ios::binary);char magic[8];uint32_t ver,pm,ds[9];uint64_t fp,abi,an,bn,cnt[27],hash;in.read(magic,8);in.read((char*)&ver,4);in.read((char*)&pm,4);in.read((char*)&fp,8);in.read((char*)&abi,8);in.read((char*)ds,36);in.read((char*)&an,8);in.read((char*)&bn,8);in.read((char*)cnt,sizeof(cnt));in.read((char*)&hash,8);in.seekg((std::streamoff)(4*(an+bn)),std::ios::cur);int q=0;for(int a=0;a<3;++a)for(int h=0;h<9;++h){size_t n=cnt[q++];if(a==1&&h==5){std::vector<uint32_t>z(n);in.read((char*)z.data(),n*4);return z;}in.seekg((std::streamoff)(n*4),std::ios::cur);}throw std::runtime_error("M");}
static std::array<std::vector<uint32_t>,9> loadpc(){std::ifstream in("src/cuda/b300/row8_pivots_w19.bin",std::ios::binary);char m[8];uint32_t v,ds[9];in.read(m,8);in.read((char*)&v,4);in.read((char*)ds,36);std::array<std::vector<uint32_t>,9>p;for(int h=0;h<9;++h){p[h].resize(ds[h]);for(auto&x:p[h]){uint32_t sc;in.read((char*)&x,4);in.read((char*)&sc,4);}}return p;}
static std::vector<uint32_t> loadA4(){std::vector<uint32_t>A((size_t)D4*D4);std::ifstream in("work/formal-probes/canonical-matrix/A_h4_mod1000000007.bin",std::ios::binary);struct H{char magic[8];uint32_t ver,mod,h,dim;uint64_t count,hash;}x{};in.read((char*)&x,sizeof(x));in.read((char*)A.data(),A.size()*4);return A;}
static std::string stateSig(Packed p){State s=unpack(p);std::string z="deg=";for(int i=0;i<8;++i)z+=char('0'+s.deg[i]);z+=" comp=";for(int i=0;i<8;++i){if(i)z+=',';z+=std::to_string(s.comp[i]);}z+=" stack=";for(int i=0;i<s.sp;++i){if(i)z+=',';z+=std::to_string(s.stack[i]);}z+=" st=";for(int q=1;q<s.ns;++q)z+=char('0'+(s.status[q]&1));return z;}
int main(){MODP=1000000007u;auto P=loadpc(); auto M=loadM(); auto A=loadA4();static int badk[5]={337,351,371,397,417};constexpr int S=8;std::array<WVec,S> E;std::array<std::array<uint32_t,5>,S> target{};for(int i=0;i<S;++i){auto v=prefix_vec(P[5][i]);E[i]=wcolumn(v,8,false,1);for(int q=0;q<5;++q){__uint128_t sum=0;int k=badk[q];for(int j=0;j<D4;++j)sum+=(__uint128_t)M[(size_t)i*D4+j]*A[(size_t)j*D4+k];target[i][q]=sum%MODP;}}
 std::cout<<"targets\n";for(int q=0;q<5;++q){std::cout<<q;for(int i=0;i<S;++i)std::cout<<' '<<target[i][q];std::cout<<'\n';}
 // Intersect raw supports of the first S source rows and compare coefficient signatures.
 std::array<size_t,S> idx{};size_t found=0,scalar=0;while(true){bool end=false;for(int i=0;i<S;++i)if(idx[i]>=E[i].size())end=true;if(end)break;Packed mx=E[0][idx[0]].p;for(int i=1;i<S;++i)if(mx<E[i][idx[i]].p)mx=E[i][idx[i]].p;bool changed=false;for(int i=0;i<S;++i)while(idx[i]<E[i].size()&&E[i][idx[i]].p<mx){++idx[i];changed=true;}if(changed)continue;bool eq=true;for(int i=0;i<S;++i)if(!(E[i][idx[i]].p==mx)){eq=false;break;}if(!eq)continue;std::array<uint32_t,S> sig{};for(int i=0;i<S;++i)sig[i]=E[i][idx[i]].v;for(int q=0;q<5;++q){bool same=true;for(int i=0;i<S;++i)if(sig[i]!=target[i][q])same=false;if(same){std::cout<<"EXACT q="<<q<<' '<<stateSig(mx)<<'\n';++found;}uint32_t c=0;bool have=false,sm=true;for(int i=0;i<S;++i){if(target[i][q]){uint64_t a=target[i][q],e=MODP-2,iv=1;while(e){if(e&1)iv=(__uint128_t)iv*a%MODP;a=(__uint128_t)a*a%MODP;e>>=1;}uint32_t ci=(__uint128_t)sig[i]*iv%MODP;if(!have){c=ci;have=true;}else if(c!=ci){sm=false;break;}}else if(sig[i]){sm=false;break;}}if(sm&&have&&c){std::cout<<"SCALAR q="<<q<<" c="<<c<<' '<<stateSig(mx)<<'\n';++scalar;}}
 for(int i=0;i<S;++i)++idx[i];}
 std::cout<<"found_exact="<<found<<" scalar="<<scalar<<"\n";
}
