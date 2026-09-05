#!/usr/bin/env python3
"""Check n=9 exact residues across frontier graph, cache and stream policies.

Pass a row6 factorized solver built with TARGET_W=10, LOW_LUT_K=5,
HIGH_LUT_K=4. This is an explicit GPU integration test, not a timing test.
"""
import argparse
from itertools import product
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    exact = 41044208702632496804
    primes = (4294967291, 4294966997)
    expected = {p: exact % p for p in primes}
    for graph_mode, cached, stream_limit, maps, pipeline in product(
            (0, 1, 2), (0, 1), (0, 18446744073709551615), (0, 1), (0, 1)):
        env = os.environ.copy()
        env.update(GRIDFP_PLAN_ONLY="0", GRIDFP_ROW6_INIT_ONLY="0",
                   GRIDFP_TRANSFER_BATCH="1", GRIDFP_PROFILE_BATCH="0",
                   GRIDFP_TRANSITION_GRAPHS=str(graph_mode),
                   GRIDFP_TRANSFER_MAP_MIB=str(maps),
                   GRIDFP_VERIFY_TRANSFER_MAPS=str(maps),
                   GRIDFP_PIPELINE_GROUPS=str(pipeline),
                   GRIDFP_CACHE_BLOCK_MATES=str(cached),
                   GRIDFP_SINGLE_STREAM_MAX_STATES=str(stream_limit))
        result = subprocess.run(
            [str(args.binary.resolve()), "9", "1", "14", "1", *map(str, primes)],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=True)
        rows = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                for line in result.stdout.splitlines() if line.startswith("backend=")]
        actual = {int(row["modulus"]): int(row["residue"]) for row in rows}
        if len(rows) != len(primes) or actual != expected:
            raise RuntimeError(f"graph={graph_mode}, cache={cached}, stream_limit={stream_limit}: {result.stdout}")
        for row in rows:
            graph_time = float(row["group_graph_sum_s"])
            if maps and (int(row["transfer_builds"]) <= 0 or int(row["transfer_hits"]) <= 0):
                raise RuntimeError(f"Transfer map sharing was not exercised: {row}")
            if graph_mode and int(row["graph_replays"]) <= 0:
                raise RuntimeError(f"Graph cache was not exercised: {row}")
            if graph_mode and not maps and int(row["residue_index"]) > 0 and int(row["graph_builds"]) != 0:
                raise RuntimeError(f"Graph cache was not reused across moduli: {row}")
            if graph_mode == 2:
                if graph_time <= 0 or abs(graph_time - float(row["active_sum_s"])) > 1e-4:
                    raise RuntimeError(f"Whole-graph phase/reset accounting: {row}")
            elif graph_time != 0:
                raise RuntimeError(f"Unexpected whole-graph accounting: {row}")
        print(f"PASS maps={maps} pipeline={pipeline} graph={graph_mode} cache={cached} stream_limit={stream_limit}: {actual}", flush=True)

if __name__ == "__main__":
    main()
