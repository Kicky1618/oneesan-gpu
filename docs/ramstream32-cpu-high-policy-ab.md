# CPU HIGH policy A/B benchmark

Use this after `ramstream32-cpu-high-sweep.sh` has produced a calibrated non-monotone `.groups` policy. The goal is to compare the cost-model policy against the best observed size threshold on the same binary and host, rather than trusting the model prediction.

Example:

```bash
N=27 \
CPU_HIGH_OVERLAP=1 \
CPU_HIGH_CPU_LIST='0-31' \
CPU_LOW_CPU_LIST='32-63' \
CPU_LOW_SCHEDULE=contiguous \
CPU_WORKERS=32 \
CPU_HIGH_WORKERS=32 \
THRESHOLD_MIB=256 \
GROUPS_FILE=work/bench_ramstream32_cpu_high_sweep/cpu-high-cost-policy-overlap1-n27-....groups \
REPEATS=4 \
bash scripts/bench/ramstream32-cpu-high-policy-ab.sh
```

`CPU_LOW_SCHEDULE` accepts `dynamic`, `sticky`, or `contiguous` and defaults to `dynamic`. It is explicitly propagated to both HIGH-policy variants and recorded in benchmark metadata. Keep the same LOW schedule while comparing threshold versus `.groups`; changing it at the same time would confound the HIGH-policy A/B result.

The harness alternates order to reduce temperature/cache/order bias:

```text
repeat 1: threshold -> policy
repeat 2: policy -> threshold
repeat 3: threshold -> policy
repeat 4: policy -> threshold
```

Both variants use `CPU_HIGH_MODE=direct`, the same worker counts, GPU target, overlap setting, HIGH/LOW affinity lists, LOW schedule, modulus, binary, and authoritative algorithm. The threshold variant clears `CPU_HIGH_GROUPS_FILE`; the policy variant sets `CPU_HIGH_MAX_MIB=0` and uses the supplied exact group file.

Every run must produce the same residue. For the known `n=21, modulus=4294967291` case the harness also checks residue `998035516` automatically. `EXPECTED_RESIDUE` can be supplied for other known cases.

The TSV records wall time, H2D+D2H components, GPU kernel time, CPU HIGH/LOW wall time, PCIe volume removed, CPU HIGH group count, and the production selection hash. The metadata file records commit, host/GPU/driver, binary SHA-256, group-policy SHA-256, HIGH/LOW affinity, `CPU_LOW_SCHEDULE`, threshold, and all execution parameters.

The final `comparison` line reports:

```text
policy_speedup = mean_threshold_wall / mean_policy_wall
policy_saved_s = mean_threshold_wall - mean_policy_wall
```

A generated policy should only replace the simpler threshold policy after this A/B benchmark shows a repeatable wall-time improvement. If its predicted advantage disappears here, inspect the affine model residuals, LOW scheduling/NUMA placement, and overlap contention before adding more model complexity.
