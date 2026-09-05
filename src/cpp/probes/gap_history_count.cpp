#include <bits/stdc++.h>
using namespace std;
using U=unsigned long long;
static U ct(int n){ // central trinomial: balanced words over {-1,0,+1}
  vector<U>d(2*n+1); d[n]=1;
  for(int k=0;k<n;++k){vector<U>e(2*n+1);for(int s=-k;s<=k;++s){U z=d[n+s];if(!z)continue;e[n+s]+=z;e[n+s-1]+=z;e[n+s+1]+=z;}d.swap(e);}return d[n];
}
static U gaps(int r,int h){int m=r-h;if(m<0)return 0;vector<U>dp(m+1);dp[0]=1;for(int g=0;g<h+1;++g){vector<U>nd(m+1);for(int a=0;a<=m;++a)if(dp[a])for(int z=0;a+z<=m;++z)nd[a+z]+=dp[a]*ct(z);dp.swap(nd);}return dp[m];}
int main(int argc,char**argv){int R=argc>1?atoi(argv[1]):10;for(int r=0;r<=R;++r){U s=0;cout<<r<<":";for(int h=0;h<=r;++h){auto z=gaps(r,h);s+=z;cout<<" "<<z;}cout<<" | "<<s<<"\n";}}
