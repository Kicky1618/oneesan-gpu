#!/usr/bin/env python3
import pathlib,sys
if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-runtime-threads.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
old='int threads=256,maxgroups=0;auto prep0=std::chrono::steady_clock::now();'
new='int threads=256;if(const char*e=std::getenv("GRIDFP_THREADS")){char*end=nullptr;long v=std::strtol(e,&end,10);if(!end||*end||v<32||v>1024||(v&31)){std::cerr<<"GRIDFP_THREADS must be a multiple of 32 in [32,1024], got "<<e<<"\\n";return 18;}threads=int(v);}std::cerr<<"GRIDFP runtime threads="<<threads<<"\\n";int maxgroups=0;auto prep0=std::chrono::steady_clock::now();'
n=s.count(old)
if n!=1:raise SystemExit(f'runtime threads anchor expected one match got {n}')
s=s.replace(old,new,1)
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: runtime_threads_env=GRIDFP_THREADS default=256 min=32 max=1024 warp_multiple=1')
