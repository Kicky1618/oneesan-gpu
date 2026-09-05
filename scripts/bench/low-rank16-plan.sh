#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null; then echo "$CXX not found" >&2; exit 2; fi
SRC="$ONEESAN_ROOT/src/cpp/probes/gridfp_low_rank16_plan.cpp"
BIN="${BIN:-$ONEESAN_BUILD_DIR/gridfp_low_rank16_plan}"
mkdir -p "$(dirname "$BIN")"
"$CXX" -O3 -std=c++17 "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Fq 'gridfp-low-rank16-plan OK' <<<"$out"
grep -Fq 'W=28 low_k=14 high_k=13' <<<"$out"
grep -Fq 'low_codes=1201917' <<<"$out"
grep -Fq 'l_digits=3720805 valid_rank16=3720805' <<<"$out"
grep -Fq 'owner_invariant=1 all_L_flips_legal=1' <<<"$out"
grep -Fq 'dense_bytes_per_code=28' <<<"$out"
grep -Fq 'rankstream_model=offset32+rank16_per_L' <<<"$out"
for g in 0 1 2 3 4 5 6 7; do grep -Fq "owner=$g " <<<"$out"; done
printf '%s\n' "$out" | python3 -c '
import re,sys
text=sys.stdin.read()
rows=[tuple(map(int,m)) for m in re.findall(r"owner=(\d+) low_codes=(\d+) l_digits=(\d+)",text)]
if len(rows)!=8: raise SystemExit("expected 8 owner rows")
L=14; key_bits=23; prefix_bits=9; block=32
if 3**L > 1<<key_bits: raise SystemExit("key packing bound failed")
if (block-1)*L >= 1<<prefix_bits: raise SystemExit("prefix packing bound failed")
prekey=sum(n*4 for _,n,_ in rows)
rank16=prekey+sum(n*L*2 for _,n,_ in rows)
rankstream=prekey+sum(n*4+l*2 for _,n,l in rows)
rankstream32=sum(n*4+((n+block-1)//block)*4+l*2 for _,n,l in rows)
print(f"prekey_metadata_mib_total={prekey/(1<<20):.6f}")
print(f"rank16_runtime_metadata_mib_total={rank16/(1<<20):.6f}")
print(f"rankstream_runtime_metadata_mib_total={rankstream/(1<<20):.6f}")
print(f"rankstream32_runtime_metadata_mib_total={rankstream32/(1<<20):.6f}")
print(f"rankstream32_vs_rankstream_metadata_reduction={rankstream/rankstream32:.6f}x")
print(f"rankstream32_vs_rank16_metadata_reduction={rank16/rankstream32:.6f}x")
print("rankstream32_pack_proved=1 key_bits=23 prefix_bits=9 block=32")
'
echo 'low-rank16-plan OK W=28 owner_invariant=1 all_L_flips_legal=1 rankstream32_pack_proved=1' >&2
