#define main rowr_batch_embedded_main
#include "rowr_column_batch_wfa_checkpoint.cpp"
#undef main
#include <iostream>

static Vec finish_column_sym(Vec const&cur,int r,int sym){
 int nt=std::max(1,omp_get_max_threads());std::vector<Vec>loc(nt);int bottom=2*r;
#pragma omp parallel
 {
  int tid=omp_get_thread_num();auto&v=loc[tid];v.reserve(cur.size()/nt+64);
#pragma omp for schedule(static)
  for(long long k=0;k<(long long)cur.size();++k){
   State s=unpack(cur[(size_t)k]);auto q=s.comp[bottom];bool ok=true;
   if(sym==0){if(s.deg[bottom]!=0)continue;}
   else {if(s.deg[bottom]!=1||!q)continue;if(sym==2){if(s.sp>=r)continue;s.stack[s.sp++]=q;}else{if(!s.sp)continue;auto q2=s.stack[--s.sp];if(!merge_components(s,q,q2,true))continue;q=s.comp[bottom];}}
   s.deg[bottom]=0;s.comp[bottom]=0;State out{};out.n=r;out.ns=s.ns;out.sp=s.sp;out.status=s.status;out.stack=s.stack;for(int y=0;y<r;++y){out.deg[y]=s.deg[r+y];out.comp[y]=s.comp[r+y];}
   if(q){bool alive=stack_has(out,q);if(!alive)for(int y=0;y<r;++y)if(out.comp[y]==q){alive=true;break;}if(!alive&&!(out.status[q]&2))ok=false;}
   if(!ok)continue;for(int qq=1;qq<out.ns;++qq)if((out.status[qq]&2)&&!closed_consistent(out,qq)){ok=false;break;}if(ok)append(v,out);
  }
 }
 Vec out;size_t z=0;for(auto const&v:loc)z+=v.size();out.reserve(z);for(auto&v:loc)out.insert(out.end(),std::make_move_iterator(v.begin()),std::make_move_iterator(v.end()));norm(out);return out;
}
static Vec column_step_sym(Vec const&boundary,int r,int sym){
 Vec x=expand_boundary(boundary,r);
 for(int y=0;y<r;++y){x=physical_batch(x,y,r+y,2,2);if(y>0){x=physical_batch(x,y-1,y,2,2);x=forget_slot(x,y-1,r,false);}}
 x=physical_batch(x,r-1,2*r,2,2);x=forget_slot(x,r-1,r,false);return finish_column_sym(x,r,sym);
}
static size_t last_accept_count(Vec cur,int r,int sym){
 for(int y=0;y<r-1;++y){cur=physical_batch(cur,y,y+1,2,2);cur=forget_slot(cur,y,r,false);}
 Vec ext;ext.reserve(cur.size());for(auto const&p:cur){State old=unpack(p),z{};z.n=r+1;z.ns=old.ns;z.sp=old.sp;z.status=old.status;z.stack=old.stack;for(int y=0;y<r;++y){z.deg[y]=old.deg[y];z.comp[y]=old.comp[y];}append(ext,z);}norm(ext);cur.swap(ext);
 cur=physical_batch(cur,r-1,r,2,2);cur=forget_slot(cur,r-1,r,false);
 size_t ans=0;
#pragma omp parallel for reduction(+:ans) schedule(static)
 for(long long k=0;k<(long long)cur.size();++k){State z=unpack(cur[(size_t)k]);int b=r;auto q=z.comp[b];bool ok=true;if(sym==0){if(z.deg[b]!=0)continue;}else{if(z.deg[b]!=1||!q)continue;if(sym==2){if(z.sp>=r)continue;z.stack[z.sp++]=q;}else{if(!z.sp)continue;auto q2=z.stack[--z.sp];if(!merge_components(z,q,q2,true))continue;q=z.comp[b];}}z.deg[b]=0;z.comp[b]=0;if(q){bool alive=stack_has(z,q);if(!alive)for(int i=0;i<z.n;++i)if(z.comp[i]==q){alive=true;break;}if(!alive&&!(z.status[q]&2))ok=false;}if(!ok||z.sp)continue;for(int i=0;i<r;++i)if(z.deg[i]||z.comp[i]){ok=false;break;}if(ok)++ans;}
 return ans;
}
static Packed h8_identity(){State s{};s.n=8;s.sp=8;s.ns=9;for(int i=0;i<8;++i){s.deg[i]=1;s.comp[i]=uint8_t(i+1);s.stack[i]=uint8_t(i+1);}return pack(s);}
int main(int argc,char**argv){
 std::string ck="work/formal-probes/h8_dual_nonidentity.ck";if(argc>1)ck=argv[1];
 int target=argc>2?std::atoi(argv[2]):9;Vec cur;int step=0;
 if(load_ck(ck,8,step,cur)){std::cout<<"resume step="<<step<<" states="<<cur.size()<<"\n";}
 else {Vec all;int col=0;if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all))return 2;auto id=h8_identity();for(auto const&p:all){auto s=unpack(p);if(s.sp==8 && !(p==id))cur.push_back(p);}norm(cur);save_ck(ck,8,0,cur);std::cout<<"init nonidentity h8="<<cur.size()<<"\n";}
 static int syms[10]={0,0,1,1,1,1,1,1,1,1};
 while(step<std::min(target,9)){cur=column_step_sym(cur,8,syms[step]);++step;save_ck(ck,8,step,cur);std::cout<<"step="<<step<<" sym="<<syms[step-1]<<" states="<<cur.size()<<"\n";std::cout.flush();}
 if(target>=10 && step==9){auto n=last_accept_count(cur,8,syms[9]);std::cout<<"final_accept_states="<<n<<"\n";return n?1:0;}
 return 0;
}
