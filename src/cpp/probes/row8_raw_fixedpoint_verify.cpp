#define main rowr_checkpoint_embedded_main
#include "rowr_column_batch_wfa_checkpoint.cpp"
#undef main
#include <chrono>
#include <iostream>

int main(int argc,char**argv){
  std::string path=argc>1?argv[1]:"work/formal-probes/raw_wfa_r8.ck";
  Vec cur; int col=0;
  if(!load_ck(path,8,col,cur)) throw std::runtime_error("checkpoint missing");
  auto h0=payload_hash(cur); auto t0=std::chrono::steady_clock::now();
  Vec nxt=column_step(cur,8,false);
  double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  auto h1=payload_hash(nxt); bool same=nxt==cur;
  std::cout<<"raw_fixedpoint r=8 col="<<col<<" states="<<cur.size()
           <<" input_hash="<<std::hex<<h0<<" step_hash="<<h1<<std::dec
           <<" same="<<same<<" sec="<<s<<"\n";
  return same?0:1;
}
