#!/bin/bash
cd /root
rm -f n24.out n24.err n24.rc
(
  export OMP_NUM_THREADS=16
  export OMP_PROC_BIND=spread
  export OMP_PLACES=cores
  ./host32 24 4294967291 4096 14 >n24.out 2>n24.err
  echo $? >n24.rc
) </dev/null >/dev/null 2>&1 &
echo $!
