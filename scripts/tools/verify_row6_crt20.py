#!/usr/bin/env python3
"""Verify the production row-6 CRT20 coefficient table from its exact rational certificate."""
from __future__ import annotations

import argparse
import hashlib
import lzma
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "solve"))
from path_bound import PRIMES  # noqa: E402

CERTIFICATE = ROOT / "formal" / "certificates" / "row6_rational_dump.txt.xz"
CRT_HEADER = ROOT / "src" / "cuda" / "b300" / "row6_automaton_crt20_generated.hpp"
MOD_HEADER = ROOT / "src" / "cuda" / "b300" / "row6_automaton_mod1000000007.hpp"
MOD_REFERENCE = 1_000_000_007
EXPECTED_COUNTS = {"N": 47_299, "R": 28_717, "L": 41_081}
DIM = 558
CERTIFICATE_SHA256 = "58c12e5cab188c6118637266b4627a8101c0c8f1ae5f8227a385cea0cf96fb1f"
UNCOMPRESSED_CERTIFICATE_SHA256 = "a7011c70cb28d556d3155c525a12e6f4e8b23d3a6e5a154cd3f4974a8311cf80"


def _sha256_stream(stream) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def verify_certificate_hashes(path: Path = CERTIFICATE) -> None:
    with path.open("rb") as f:
        compressed = _sha256_stream(f)
    if compressed != CERTIFICATE_SHA256:
        raise ValueError(f"row6 compressed certificate SHA-256 mismatch: {compressed}")
    with lzma.open(path, "rb") as f:
        uncompressed = _sha256_stream(f)
    if uncompressed != UNCOMPRESSED_CERTIFICATE_SHA256:
        raise ValueError(f"row6 uncompressed certificate SHA-256 mismatch: {uncompressed}")


def load_certificate(path: Path = CERTIFICATE):
    verify_certificate_hashes(path)
    beta: list[int] = []
    transitions: dict[str, list[tuple[int, int, int, int]]] = {s: [] for s in "NRL"}
    with lzma.open(path, "rt") as f:
        first = f.readline().split()
        if first != ["DIM", str(DIM)]:
            raise ValueError(f"unexpected certificate dimension line: {first!r}")
        for line in f:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "B":
                i, value = int(fields[1]), int(fields[2])
                if i != len(beta):
                    raise ValueError(f"non-canonical beta index {i}, expected {len(beta)}")
                beta.append(value)
            elif fields[0] == "T" and fields[1] in transitions:
                sym = fields[1]
                transitions[sym].append(tuple(map(int, fields[2:6])))
            else:
                raise ValueError(f"invalid certificate line: {line[:120]!r}")
    if len(beta) != DIM:
        raise ValueError(f"beta length {len(beta)} != {DIM}")
    for sym, count in EXPECTED_COUNTS.items():
        if len(transitions[sym]) != count:
            raise ValueError(f"{sym} transition count {len(transitions[sym])} != {count}")
    return beta, transitions


def verify_mod_reference(beta, transitions, path: Path = MOD_HEADER) -> None:
    text = path.read_text()
    match = re.search(r"BETA\[DIM\]=\{([^}]*)\}", text)
    if not match:
        raise ValueError("BETA not found in row6 mod reference header")
    mod_beta = [int(x) for x in match.group(1).replace("u", "").split(",")]
    expected_beta = [x % MOD_REFERENCE for x in beta]
    if mod_beta != expected_beta:
        raise ValueError("row6 mod reference BETA does not match exact certificate")

    for sym in "NRL":
        off_match = re.search(rf"OFF_{sym}\[DIM\+1\]=\{{([^}}]*)\}}", text)
        tr_match = re.search(rf"TR_{sym}\[(\d+)\]=\{{(.*?)\}};", text, re.S)
        if not off_match or not tr_match:
            raise ValueError(f"missing OFF/TR table for {sym}")
        offsets = [int(x) for x in off_match.group(1).split(",")]
        mod_transitions = [
            (int(dst), int(coeff))
            for dst, coeff in re.findall(r"\{(\d+),(\d+)u\}", tr_match.group(2))
        ]
        exact = transitions[sym]
        if int(tr_match.group(1)) != len(exact) or len(mod_transitions) != len(exact):
            raise ValueError(f"{sym} reference transition length mismatch")

        expected_offsets = [0]
        expected_transitions: list[tuple[int, int]] = []
        k = 0
        for src in range(DIM):
            while k < len(exact) and exact[k][0] == src:
                _, dst, num, den = exact[k]
                if den <= 0 or math.gcd(abs(num), den) != 1:
                    raise ValueError(f"non-canonical rational coefficient at {sym}[{k}]")
                if den % MOD_REFERENCE == 0:
                    raise ValueError(f"denominator vanishes mod {MOD_REFERENCE} at {sym}[{k}]")
                coeff = (num % MOD_REFERENCE) * pow(den, -1, MOD_REFERENCE) % MOD_REFERENCE
                expected_transitions.append((dst, coeff))
                k += 1
            expected_offsets.append(k)
        if k != len(exact):
            raise ValueError(f"{sym} certificate source ordering is not monotone")
        if offsets != expected_offsets:
            raise ValueError(f"OFF_{sym} does not match exact certificate")
        if mod_transitions != expected_transitions:
            for i, (actual, expected) in enumerate(zip(mod_transitions, expected_transitions)):
                if actual != expected:
                    raise ValueError(f"TR_{sym}[{i}] mismatch: {actual} != {expected}")
            raise ValueError(f"TR_{sym} length mismatch")


def verify_crt_header(transitions, path: Path = CRT_HEADER) -> None:
    with path.open() as f:
        lines = iter(f)
        if not next(lines).startswith("#pragma once"):
            raise ValueError("unexpected CRT20 header preamble")
        if not next(lines).startswith("#include"):
            raise ValueError("unexpected CRT20 include line")
        if "row6crt" not in next(lines):
            raise ValueError("unexpected CRT20 namespace")
        if "NPRIMES=20" not in next(lines):
            raise ValueError("unexpected CRT20 prime count")
        got_primes = [int(x) for x in re.findall(r"(\d+)u", next(lines))]
        expected_primes = PRIMES[:20]
        if got_primes != expected_primes:
            raise ValueError(f"CRT20 primes do not match solver prefix: {got_primes}")
        if "prime_index" not in next(lines):
            raise ValueError("missing prime_index in CRT20 header")

        for sym in "NRL":
            count = EXPECTED_COUNTS[sym]
            header = next(lines)
            if f"CO_{sym}[NPRIMES][{count}]" not in header:
                raise ValueError(f"unexpected CO_{sym} declaration")
            exact = transitions[sym]
            denominators = {den for _, _, _, den in exact}
            for p in got_primes:
                row = next(lines).strip().rstrip(",")
                if not (row.startswith("{") and row.endswith("}")):
                    raise ValueError(f"malformed CO_{sym} row for modulus {p}")
                values = [int(x[:-1]) for x in row[1:-1].split(",")]
                if len(values) != count:
                    raise ValueError(f"CO_{sym} row length {len(values)} != {count}")
                inverse_denominator = {}
                for den in denominators:
                    if den % p == 0:
                        raise ValueError(f"denominator vanishes mod {p} in CO_{sym}")
                    inverse_denominator[den] = pow(den, -1, p)
                for i, ((_, _, num, den), actual) in enumerate(zip(exact, values)):
                    expected = (num % p) * inverse_denominator[den] % p
                    if actual != expected or not 0 <= actual < p:
                        raise ValueError(
                            f"CO_{sym}[prime={p}][{i}] mismatch: {actual} != {expected}"
                        )
            if next(lines).strip() != "};":
                raise ValueError(f"malformed CO_{sym} terminator")
        if next(lines).strip() != "}":
            raise ValueError("malformed CRT20 namespace terminator")


def verify_all() -> None:
    beta, transitions = load_certificate()
    verify_mod_reference(beta, transitions)
    verify_crt_header(transitions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    verify_all()
    print("row6 CRT20 exact-certificate verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
