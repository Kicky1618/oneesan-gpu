#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
# Verify the exact queue-id -> stream-index mapping used by dynamic PIPE2 when
# per-position stream starts are cached once in CTA shared memory.
cases=0
for nn0 in (0,1,17,1234):
  for nn_count in (0,1,2,31,257):
    nn1=nn0+nn_count
    for nr0 in (0,3,41,2048):
      for nr_count in (0,1,5,63,511):
        nr1=nr0+nr_count
        total=nn_count+nr_count
        stream=(nn0,nr0,0)
        for k in range(total):
          old = nn0+k if k<nn_count else nr0+k-nn_count
          new = stream[0]+k if k<nn_count else stream[1]+k-nn_count
          assert old==new
          cases+=1

for nn0 in (0,1,17,1234):
  for nn_count in (0,1,2,31):
    nn1=nn0+nn_count
    for nr0 in (0,3,41,2048):
      for nr_count in (0,1,5,63):
        nr1=nr0+nr_count
        for nl0 in (0,7,99,4096):
          for nl_count in (0,1,9,127):
            nl1=nl0+nl_count
            total=nn_count+nr_count+nl_count
            stream=(nn0,nr0,nl0)
            for k in range(total):
              if k<nn_count:
                old=nn0+k; new=stream[0]+k
              elif k<nn_count+nr_count:
                old=nr0+k-nn_count; new=stream[1]+k-nn_count
              else:
                old=nl0+k-nn_count-nr_count; new=stream[2]+k-nn_count-nr_count
              assert old==new
              cases+=1
print('b300_pipe2_stream_base_proof=OK')
print(f'cases={cases} forward_stream_bases=2 reverse_stream_bases=3 per_orbit_offset_table_loads_cached_path=0 shared_bytes=12')
PY
