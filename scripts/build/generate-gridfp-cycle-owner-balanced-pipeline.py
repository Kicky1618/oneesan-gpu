#!/usr/bin/env python3
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: generate-gridfp-cycle-owner-balanced-pipeline.py PIPELINE BALANCED_BASE OUT", file=sys.stderr)
        return 2
    pipeline = Path(sys.argv[1])
    balanced_base = Path(sys.argv[2])
    out = Path(sys.argv[3])
    s = pipeline.read_text()

    old_include = '#include "gridfp_reduced_production_p2p_cycle_owner_descriptorless_microprobe.cu"'
    new_include = f'#include "{balanced_base.name}"'
    if s.count(old_include) != 1:
        raise SystemExit("balanced pipeline generator: include marker mismatch")
    s = s.replace(old_include, new_include)

    old_tag = 'std::cout << "gridfp-p2p-cycle-owner-pipeline"'
    new_tag = 'std::cout << "gridfp-p2p-cycle-owner-balanced-pipeline"'
    if s.count(old_tag) != 1:
        raise SystemExit("balanced pipeline generator: output tag mismatch")
    s = s.replace(old_tag, new_tag)

    flag = '              << " compressed_cycle_owner_list=1"\n'
    if s.count(flag) != 1:
        raise SystemExit("balanced pipeline generator: output flag mismatch")
    s = s.replace(
        flag,
        flag +
        '              << " leader_batch_hash=1"\n'
        '              << " batch_hash_shift=12"\n')

    old_ok = 'std::cout << "ALL_OK gridfp_p2p_cycle_owner_pipeline=1\\n";'
    new_ok = 'std::cout << "ALL_OK gridfp_p2p_cycle_owner_balanced_pipeline=1\\n";'
    if s.count(old_ok) != 1:
        raise SystemExit("balanced pipeline generator: ALL_OK marker mismatch")
    s = s.replace(old_ok, new_ok)

    if balanced_base.name not in s:
        raise SystemExit("balanced pipeline generator: generated include missing")
    if 'leader_batch_hash=1' not in s:
        raise SystemExit("balanced pipeline generator: hash flag missing")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
