# Multi-GPU regression results

Validated on Vast.ai on 2026-08-21 with the Grid-FP mmap multi-GPU solver.

## 2 x RTX A2000

- CUDA compute capability: 8.6
- P2P capability: both directed links enabled
- `cudaMemcpyPeerAsync` smoke test: 256 MiB x 16, 25.154 GB/s, data verification passed
- `n=16`, target 16 MiB/GPU:
  - 1 GPU: 2.611 s
  - 2 GPU: 1.461 s (1.79x)
  - final `main.bin` and `blocked.bin`: byte-identical
- `n=17`, target 8 MiB/GPU, 256 groups/window:
  - 1 GPU: 40.765 s
  - 2 GPU: 21.758 s (1.87x)
  - final state files: byte-identical

## 4 x RTX 3060

`n=17`, target 8 MiB/GPU:

- 1 GPU: 39.355 s
- 2 GPU: 23.024 s
- 4 GPU: 11.034 s (3.57x vs 1 GPU)
- final 1/2/4-GPU state files: byte-identical
- worker active times on 4 GPUs: 10.926--10.949 s

## 8 x RTX 3060

Same `n=17` regression on one 8-GPU host:

- 1 GPU: 38.905 s
- 2 GPU: 20.580 s (1.89x)
- 4 GPU: 11.004 s (3.54x)
- 8 GPU: 6.105 s (6.37x)
- 4-GPU and 8-GPU `main.bin`/`blocked.bin`: byte-identical
- main SHA256: `ab6bb24f44a559f0bc5d666fd539277b185b92e4df35c9d7d6d1cb2366a8fe48`
- blocked SHA256: `69c6e4e7348a82a17f9703a06c412ed40ee554619e8042fc4553e36c0c847233`
- 8 worker active times: 5.964--6.021 s

The RTX 3060 host exposed no CUDA P2P links; correctness and scheduler scaling do not depend on P2P. P2P is optional and intended for the later NVLink/NVSwitch redistribution optimization.

Total Vast.ai debug spend was about USD 0.044.
