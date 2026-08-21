# Grid-FP multi-GPU

Build on the target machine so nvcc chooses the installed GPU architecture:

```bash
ARCH=native OUT=oneesan_cuda_gridfp_multigpu ./build_gridfp_multigpu.sh
```

B300 x8 example, one CRT modulus:

```bash
./run_gridfp_b300x8.sh 27 2305843009213693951 /raid/gridfp/n27_p0
```

Arguments to the binary:

```text
binary n modulus target_mib_per_gpu max_window gpu_count store_directory
```

The scheduler uses the paper's transition-closed groups. A host thread owns each GPU,
and an atomic work queue assigns the largest groups first. Each device has independent
CUDA constant state and buffers, so kernels execute concurrently. CUDA peer access is
enabled automatically wherever the platform reports P2P (NVLink/NVSwitch ready).
The current data path keeps the authoritative state in the mmap external store; it does
not yet require peer copies for correctness.

For n=27 with one 64-bit residue the external files are about 3.88 TiB. Allow additional
space for filesystem/page-cache overhead and use local NVMe/RAID, not a network filesystem.
