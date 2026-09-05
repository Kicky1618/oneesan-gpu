#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

LOOP_OLD='''    auto wall0=std::chrono::steady_clock::now();int done_windows=0;
    for(int row=0;row<W;++row){'''
LOOP_NEW='''    int b300_row_limit=W;if(const char*e=std::getenv("B300_ROW_LIMIT")){char*end=nullptr;long v=std::strtol(e,&end,10);if(!end||*end||v<1||v>W){std::cerr<<"invalid B300_ROW_LIMIT="<<e<<" expected 1.."<<W<<"\\n";return 15;}b300_row_limit=int(v);}
    std::cerr<<"B300 row limit: rows="<<b300_row_limit<<"/"<<W<<" calibration="<<(b300_row_limit<W?1:0)<<"\\n";
    auto wall0=std::chrono::steady_clock::now();int done_windows=0;
    for(int row=0;row<b300_row_limit;++row){'''

PROGRESS_OLD='''        std::cerr<<"row "<<row+1<<"/"<<W<<" windows="<<done_windows<<"\\n";'''
PROGRESS_NEW='''        std::cerr<<"row "<<row+1<<"/"<<b300_row_limit<<" windows="<<done_windows<<"\\n";'''


def once(text:str,old:str,new:str,label:str)->str:
    n=text.count(old)
    if n!=1: raise SystemExit(f'{label}: expected exactly one generated-source match, got {n}')
    return text.replace(old,new,1)


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    text=a.src.read_text()
    text=once(text,LOOP_OLD,LOOP_NEW,'row-limit loop')
    text=once(text,PROGRESS_OLD,PROGRESS_NEW,'row-limit progress')
    if 'for(int row=0;row<W;++row)' in text: raise SystemExit('unbounded row loop remains after row-limit lowering')
    for required in ('B300_ROW_LIMIT','b300_row_limit=W','for(int row=0;row<b300_row_limit;++row)','calibration='):
        if required not in text: raise SystemExit(f'missing row-limit artifact: {required}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)
    print(f'lowered {a.out} row_limit_env=B300_ROW_LIMIT default_rows=W calibration_default=0')

if __name__=='__main__':main()
