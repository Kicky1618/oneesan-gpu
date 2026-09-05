#include <bits/stdc++.h>
#include "../cuda/o0_midpoint_tables.hpp"
using namespace std;
static constexpr uint32_t MOD=65521, Z=61640, ZI=19685;
static constexpr int KQ[4]={-1,0,0,1};
static constexpr int DQ[3]={0,1,-1};
inline uint32_t addm(uint32_t a,uint32_t b){uint32_t x=a+b;return x>=MOD?x-MOD:x;}
inline uint32_t mulm(uint32_t a,uint32_t b){return uint64_t(a)*b%MOD;}
uint64_t p3(int e){uint64_t x=1;while(e--)x*=3;return x;}
int chraw(uint64_t raw,int m){int q=0;for(int i=0;i<m;i++){q+=DQ[raw%3];raw/=3;}return q;}
uint64_t insertd(uint64_t rest,int p,int u){uint64_t s=p3(p);return rest%s+uint64_t(u)*s+(rest/s)*(3*s);}

array<array<uint16_t,12>,12> staticM(const uint16_t* C,bool rev){
 array<array<uint16_t,12>,12>M{};
 if(!rev){
  for(int kp=0;kp<4;kp++)for(int d=0;d<3;d++)for(int k=0;k<4;k++)for(int u=0;u<3;u++)
    M[kp+4*d][k+4*u]=C[(d+3*kp)*12+(k+4*u)];
 }else{
  for(int kp=0;kp<4;kp++)for(int d=0;d<3;d++)for(int k=0;k<4;k++)for(int u=0;u<3;u++)
    M[kp+4*d][k+4*u]=C[(kp+4*d)*12+(u+3*k)];
 }
 return M;
}

struct Sol {
 int n,m; uint64_t Nraw; vector<int64_t> ix; vector<uint32_t> v; array<uint64_t,5> base{};
 array<array<uint16_t,12>,12> F,R;
 Sol(int n):n(n),m(n-1),Nraw(p3(m)),ix(4*Nraw,-1),F(staticM(o0mid::C_F,false)),R(staticM(o0mid::C_R,true)){
   uint64_t cur=0;
   for(int k=0;k<4;k++){base[k]=cur;for(uint64_t raw=0;raw<Nraw;raw++)if(KQ[k]+chraw(raw,m)==1)ix[k*Nraw+raw]=cur++;}
   base[4]=cur;v.assign(cur,0);
 }
 uint64_t idx(int k,uint64_t raw)const{auto z=ix[k*Nraw+raw];if(z<0)throw runtime_error("bad idx");return z;}
 void local(int p,const array<array<uint16_t,12>,12>&M,bool bottom=false){
   uint64_t Nr=p3(m-1);
   for(uint64_t rest=0;rest<Nr;rest++){
     int qr=chraw(rest,m-1),T=1-qr,pair[4],dim=0;uint64_t ids[4];uint32_t x[4],y[4];
     for(int k=0;k<4;k++)for(int u=0;u<3;u++)if(KQ[k]+DQ[u]==T){pair[dim]=k+4*u;auto raw=insertd(rest,p,u);ids[dim]=idx(k,raw);x[dim]=v[ids[dim]];dim++;}
     if(!dim)continue;
     for(int a=0;a<dim;a++){y[a]=0;int d=pair[a]/4;if(bottom&&d!=0)continue;for(int b=0;b<dim;b++)if(M[pair[a]][pair[b]])y[a]=addm(y[a],mulm(M[pair[a]][pair[b]],x[b]));}
     for(int a=0;a<dim;a++)v[ids[a]]=y[a];
   }
 }
 void turn(const uint16_t*J){
   for(uint64_t raw=0;raw<Nraw;raw++)if(chraw(raw,m)==1){auto i1=idx(1,raw),i2=idx(2,raw);uint32_t x1=v[i1],x2=v[i2];v[i1]=addm(mulm(J[5],x1),mulm(J[6],x2));v[i2]=addm(mulm(J[9],x1),mulm(J[10],x2));}
 }
 uint32_t run(){
   for(int d0=0;d0<2;d0++){int r=d0==0?1:0;uint32_t sw=d0==0?1:ZI;for(int k=0;k<4;k++){uint32_t a=o0mid::A_F[k*9+r];if(!a)continue;auto z=idx(k,d0);v[z]=addm(v[z],mulm(sw,a));}}
   for(int p=1;p<m;p++)local(p,F);
   bool forward=true;
   for(int y=1;y+1<n;y++)if(forward){turn(o0mid::J_FR);forward=false;for(int p=m-1;p>=0;p--)local(p,R);}else{turn(o0mid::J_RF);forward=true;for(int p=0;p<m;p++)local(p,F);}
   if(forward)throw runtime_error("odd n required");turn(o0mid::J_RF);for(int p=0;p+1<m;p++)local(p,F,true);
   uint32_t ans=0;
   for(int k=0;k<4;k++)for(int u=0;u<3;u++){uint32_t tw=0;for(int r=0;r<3;r++){uint32_t ew=(r==1&&u==0)?1:(r==0&&u==1)?Z:0;if(ew)tw=addm(tw,mulm(o0mid::B_F[(3*r)*4+k],ew));}if(!tw)continue;uint64_t raw=uint64_t(u)*p3(m-1);auto z=ix[k*Nraw+raw];if(z>=0)ans=addm(ans,mulm(v[z],tw));}
   return ans;
 }
};
int main(int ac,char**av){int a=ac>1?atoi(av[1]):3,b=ac>2?atoi(av[2]):a;for(int n=a;n<=b;n+=2){auto t=chrono::steady_clock::now();Sol s(n);auto z=s.run();cerr<<"n="<<n<<" states="<<s.v.size()<<" sec="<<chrono::duration<double>(chrono::steady_clock::now()-t).count()<<"\n";cout<<n<<" "<<z<<"\n";}}
