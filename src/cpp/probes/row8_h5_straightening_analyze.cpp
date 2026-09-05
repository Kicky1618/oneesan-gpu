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

static constexpr int D[9]={1107,1640,1428,888,420,152,42,8,1};
static constexpr int DEL[3]={0,-1,1};
struct MC{std::array<std::array<std::vector<uint32_t>,9>,3>M;};
static MC loadM(uint32_t mod){std::string p;for(auto const&e:std::filesystem::directory_iterator("work/row8_mod_cache")){auto s=e.path().filename().string();if(s.find("row8_mod_"+std::to_string(mod)+"_")==0&&s.find("_52384d4443414249.bin")!=std::string::npos){p=e.path();break;}}if(p.empty())throw std::runtime_error("M cache");std::ifstream in(p,std::ios::binary);char magic[8];uint32_t ver,pm,ds[9];uint64_t fp,abi,an,bn,cnt[27],hash;in.read(magic,8);in.read((char*)&ver,4);in.read((char*)&pm,4);in.read((char*)&fp,8);in.read((char*)&abi,8);in.read((char*)ds,36);in.read((char*)&an,8);in.read((char*)&bn,8);in.read((char*)cnt,sizeof(cnt));in.read((char*)&hash,8);in.seekg((std::streamoff)(4*(an+bn)),std::ios::cur);MC z;int q=0;for(int a=0;a<3;++a)for(int h=0;h<9;++h){z.M[a][h].resize(cnt[q++]);if(!z.M[a][h].empty())in.read((char*)z.M[a][h].data(),z.M[a][h].size()*4);}return z;}
static std::array<std::vector<uint32_t>,9> loadpc(){std::ifstream in("src/cuda/b300/row8_pivots_w19.bin",std::ios::binary);char m[8];uint32_t v,ds[9];in.read(m,8);in.read((char*)&v,4);in.read((char*)ds,36);std::array<std::vector<uint32_t>,9>p;for(int h=0;h<9;++h){p[h].resize(ds[h]);for(auto&x:p[h]){uint32_t sc;in.read((char*)&x,4);in.read((char*)&sc,4);}}return p;}

static bool enc(std::string const&t,Packed&out){if(t.size()!=8)return false;int h=0,b=0;for(char c:t){if(c=='T'){if(b)return false;++h;}else if(c=='U')++b;else if(c=='D')--b;else if(c!='N')return false;}if(b)return false;State s{};s.n=8;s.sp=h;std::vector<int>tp;for(int i=0;i<8;++i)if(t[i]=='T')tp.push_back(i);int nx=1;for(int k=0;k<h;++k){int q=nx++,i=tp[k];s.deg[i]=1;s.comp[i]=q;s.stack[k]=q;}std::vector<std::pair<int,int>>op;b=0;for(int i=0;i<8;++i){char c=t[i];if(c=='T'){if(b||!op.empty())return false;continue;}if(c=='N')continue;int st=c=='U'?1:-1;bool o=b==0||(b>0&&st>0)||(b<0&&st<0);if(o){op.push_back({i,st});b+=st;}else{auto[j,sg]=op.back();op.pop_back();int q=nx++;s.deg[j]=s.deg[i]=1;s.comp[j]=s.comp[i]=q;s.status[q]=sg<0;b+=st;}}s.ns=nx;out=pack(s);return true;}
static void rec(int pos,int nt,int bal,std::string&w,std::vector<std::pair<Packed,std::string>>&out){if(pos==8){if(!nt&&!bal){Packed p;if(enc(w,p))out.push_back({p,w});}return;}if(nt>8-pos)return;w[pos]='N';rec(pos+1,nt,bal,w,out);w[pos]='U';rec(pos+1,nt,bal+1,w,out);w[pos]='D';rec(pos+1,nt,bal-1,w,out);if(nt&&bal==0){w[pos]='T';rec(pos+1,nt-1,0,w,out);}}
static std::vector<std::pair<Packed,std::string>> basisWords(int h){std::string w(8,'N');std::vector<std::pair<Packed,std::string>>v;rec(0,h,0,w,v);std::sort(v.begin(),v.end(),[](auto&a,auto&b){return a.first<b.first;});v.erase(std::unique(v.begin(),v.end(),[](auto&a,auto&b){return a.first==b.first;}),v.end());return v;}
static std::vector<uint32_t> proj(WVec const&v,std::vector<std::pair<Packed,std::string>>const&b){std::vector<uint32_t>z(b.size());size_t i=0,j=0;while(i<v.size()&&j<b.size()){if(v[i].p<b[j].first)++i;else if(b[j].first<v[i].p)++j;else{z[j]=v[i].v;++i;++j;}}return z;}
static std::vector<uint32_t> loadA(int h,uint32_t mod){std::vector<uint32_t>A((size_t)D[h]*D[h]);std::ifstream in("work/formal-probes/canonical-matrix/A_h"+std::to_string(h)+"_mod"+std::to_string(mod)+".bin",std::ios::binary);struct H{char magic[8];uint32_t ver,mod,h,dim;uint64_t count,hash;}x{};in.read((char*)&x,sizeof(x));in.read((char*)A.data(),A.size()*4);if(!in)throw std::runtime_error("A cache");return A;}
static std::string gapPattern(std::string w){for(char&c:w)if(c=='T')c='|';return w;}
int main(){MODP=1000000007u;auto P=loadpc();auto C=loadM(MODP);auto bt=basisWords(4);auto A4=loadA(4,MODP);auto const&M=C.M[1][5];std::map<std::string,size_t>pat;std::map<size_t,size_t>hist;size_t total=0;
 for(int i=0;i<D[5];++i){auto src=prefix_vec(P[5][i]);auto ext=wcolumn(src,8,false,1);auto got=proj(ext,bt);size_t bi=0;for(int k=0;k<D[4];++k){__uint128_t s=0;for(int j=0;j<D[4];++j)s+=(__uint128_t)M[(size_t)i*D[4]+j]*A4[(size_t)j*D[4]+k];uint32_t pr=s%MODP;if(got[k]!=pr){++bi;++total;++pat[gapPattern(bt[k].second)];if(total<=40)std::cout<<"i="<<i<<" k="<<k<<" word="<<bt[k].second<<" got="<<got[k]<<" pred="<<pr<<"\n";}}++hist[bi];if((i&31)==0)std::cerr<<"row "<<i+1<<"/"<<D[5]<<" mism="<<bi<<"\n";}
 std::cout<<"total="<<total<<"\nsource_mismatch_hist";for(auto [n,c]:hist)std::cout<<" "<<n<<":"<<c;std::cout<<"\npatterns="<<pat.size()<<"\n";for(auto const&[p,c]:pat)std::cout<<c<<" "<<p<<"\n";
}
