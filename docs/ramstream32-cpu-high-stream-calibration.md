# CPU HIGH direct stream-weight calibration

The default cost model treats NN, NR/NL, BLOCK closure, and CROSS closure cell iterations equally. This workflow measures whether that assumption is valid on the target CPU/memory topology.

First build an exact cost plan:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-high-cost-plan.sh
./build/ramstream32_cpu_high_cost_plan_n27 27 > high-cost.tsv
```

Choose topology-diverse calibration groups. The designer normalizes the four stream-count columns plus a fixed-cost column and greedily maximizes the log determinant of the resulting Gram matrix. It also preserves rare stream-fraction extremes before applying its candidate cap.

```bash
python3 scripts/tools/design_cpu_high_stream_calibration.py high-cost.tsv \
  --out-dir work/high-stream-design \
  --samples 12 \
  --min-roundtrip-mib 64 \
  --max-roundtrip-mib 1024
```

This writes one single-group `.groups` policy per sample, `manifest.tsv`, and `validation-all.groups` containing the union of selected calibration groups.

Run the samples on the actual machine:

```bash
N=27 \
MANIFEST=work/high-stream-design/manifest.tsv \
CPU_HIGH_CPU_LIST='0-31' \
CPU_LOW_CPU_LIST='32-63' \
CPU_LOW_SCHEDULE=contiguous \
CPU_HIGH_WORKERS=32 \
CPU_HIGH_OVERLAP=0 \
REPEATS=2 \
bash scripts/bench/ramstream32-cpu-high-stream-calibration.sh
```

`CPU_LOW_SCHEDULE` accepts `dynamic`, `sticky`, or `contiguous` and defaults to `dynamic`. The harness explicitly propagates and records it. Keep the LOW schedule and LOW affinity fixed across all calibration samples: even when `CPU_HIGH_OVERLAP=0`, the LOW pass shares the same authoritative RAM and contributes to whole-run thermal/cache/NUMA state; under overlap or repeated runs those effects can change the apparent CPU HIGH coefficients.

Odd repeats execute samples in manifest order and even repeats reverse the order. Every run must produce the same residue. The optional `validation-all.groups` policy is run once after the calibration samples.

Fit non-negative per-stream costs:

```bash
python3 scripts/tools/fit_cpu_high_stream_weights.py \
  high-cost.tsv \
  work/bench_ramstream32_cpu_high_stream_calibration/stream-calibration-n27-....tsv \
  --write-args work/high-stream-design/planner-weights.txt
```

The model is

```text
CPU HIGH seconds ~= beta_NN    * NN cells
                 + beta_NRNL  * NR/NL cells
                 + beta_BLOCK * BLOCK cells
                 + beta_CROSS * CROSS cells
                 + beta_group * group executions
```

The fitter uses non-negative coordinate-descent least squares, reports each coefficient and residual, then rewrites the four cell coefficients into a common throughput plus relative planner weights:

```text
--cpu-gcell-s ...
--nn-weight ...
--nrnl-weight ...
--block-weight ...
--cross-weight ...
--group-overhead-us ...
```

The geometric mean of positive stream coefficients is used as the common reference, so the rewritten model preserves the fitted seconds exactly rather than arbitrarily forcing NN weight to one.

`validation-all.groups` is a useful holdout. Single-group calibration identifies stream ratios well, but its fixed-cost term also absorbs per-row wake/synchronization cost. The multi-group validation row exposes whether that fixed cost scales per group as assumed. Large holdout error means the model needs a separate per-row term or a different calibration design before the weights should be fed back into the production planner.

For `CPU_HIGH_OVERLAP=1`, repeat the calibration under overlap rather than reusing non-overlap weights: System-RAM contention can change relative stream costs, especially CROSS gathers. LOW schedule, affinity, CPU frequency policy, and NUMA conditions are part of that calibrated contention regime and should be kept with the resulting coefficients.
