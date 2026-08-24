# CPU HIGH policy A/B benchmark

Use this after `ramstream32-cpu-high-sweep.sh` has produced a calibrated non-monotone `.groups` policy. The goal is to compare the cost-model policy against the best observed size threshold on the same binary and host, rather than trusting the model prediction.

Example with domain LOW scheduling:

```bash
N=27 \
CPU_HIGH_OVERLAP=1 \
CPU_HIGH_CPU_LIST='0-31' \
CPU_LOW_CPU_LIST='32-95' \
CPU_LOW_SCHEDULE=domain \
CPU_LOW_DOMAIN_SIZE=32 \
CPU_LOW_DOMAIN_REFINE=1 \
CPU_WORKERS=64 \
CPU_HIGH_WORKERS=32 \
THRESHOLD_MIB=256 \
GROUPS_FILE=work/bench_ramstream32_cpu_high_sweep/cpu-high-cost-policy-overlap1-n27-....groups \
REPEATS=4 \
bash scripts/bench/ramstream32-cpu-high-policy-ab.sh
```

`CPU_LOW_SCHEDULE` accepts `dynamic`, `sticky`, `contiguous`, or `domain` and defaults to `dynamic`. `domain` additionally requires a positive `CPU_LOW_DOMAIN_SIZE <= CPU_WORKERS`. `CPU_LOW_DOMAIN_REFINE` defaults to `1`; set it to `0` only when intentionally comparing HIGH policies under the unrefined domain partition.

The LOW schedule, domain size, and refinement condition are propagated to both threshold and `.groups` variants and recorded in metadata. Keep all three fixed while comparing HIGH policies; changing LOW ownership at the same time would confound the HIGH-policy A/B result.

The harness alternates order to reduce temperature/cache/order bias:

```text
repeat 1: threshold -> policy
repeat 2: policy -> threshold
repeat 3: threshold -> policy
repeat 4: policy -> threshold
```

Both variants use `CPU_HIGH_MODE=direct`, the same worker counts, GPU target, overlap setting, HIGH/LOW affinity lists, LOW scheduling controls, modulus, binary, and authoritative algorithm. The threshold variant clears `CPU_HIGH_GROUPS_FILE`; the policy variant sets `CPU_HIGH_MAX_MIB=0` and uses the supplied exact group file.

Every run must produce the same residue. For the known `n=21, modulus=4294967291` case the harness also checks residue `998035516` automatically. `EXPECTED_RESIDUE` can be supplied for other known cases.

The TSV records wall time, H2D+D2H components, GPU kernel time, CPU HIGH/LOW wall time, PCIe volume removed, CPU HIGH group count, and the production selection hash. Metadata records commit, host/GPU/driver, binary SHA-256, group-policy SHA-256, HIGH/LOW affinity, `CPU_LOW_SCHEDULE`, `CPU_LOW_DOMAIN_SIZE`, `CPU_LOW_DOMAIN_REFINE`, threshold, and all other execution parameters.

The final `comparison` line reports:

```text
policy_speedup = mean_threshold_wall / mean_policy_wall
policy_saved_s = mean_threshold_wall - mean_policy_wall
```

A generated policy should only replace the simpler threshold policy after this A/B benchmark shows a repeatable wall-time improvement. If its predicted advantage disappears here, inspect the affine model residuals, LOW scheduling/refinement, NUMA placement, and overlap contention before adding more model complexity.
