#define ROW8_CANONICAL_TRIE_NO_MAIN 1
#include "row8_canonical_trie_verify.cpp"
#include <iostream>
int main(){auto C=loadM(1000000007u);for(int h=0;h<=8;++h)for(int a=0;a<3;++a){int h2=h+DEL_[a];if(h2<0||h2>8)continue;auto const&M=C.M[a][h];size_t nz=0;for(auto x:M)nz+=x!=0;size_t total=(size_t)D_[h]*D_[h2];std::cout<<"h="<<h<<" a="<<a<<" h2="<<h2<<" nz="<<nz<<" total="<<total<<" density="<<(double)nz/total<<"\n";}}
