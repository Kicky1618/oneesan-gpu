#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
import sys

ENTRY_RE = re.compile(r"Compiling entry function '([^']+)'")
PROP_RE = re.compile(r"Function properties for\s+(.+)$")
STACK_RE = re.compile(r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*(\d+) bytes spill loads")
USED_RE = re.compile(r"Used\s+(\d+) registers(?:,\s*(.*))?$")
BYTE_RE = re.compile(r"(\d+) bytes ([A-Za-z0-9_\[\]]+)")


def demangle(name: str) -> str:
    if not shutil.which("c++filt"):
        return name
    try:
        p = subprocess.run(["c++filt", name], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        out = p.stdout.strip()
        return out or name
    except OSError:
        return name


def emit(label, name, data):
    dname = demangle(name)
    if "bucket_" not in dname:
        return
    print("\t".join([
        label,
        dname,
        str(data.get("registers", "NA")),
        str(data.get("stack", "NA")),
        str(data.get("spill_stores", "NA")),
        str(data.get("spill_loads", "NA")),
        str(data.get("smem", "NA")),
        str(data.get("cmem0", "NA")),
    ]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--label", required=True)
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()

    if args.header:
        print("backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes")

    current = None
    records = {}
    with open(args.log, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            m = ENTRY_RE.search(line)
            if m:
                current = m.group(1)
                records.setdefault(current, {})
                continue
            m = PROP_RE.search(line)
            if m:
                current = m.group(1).strip()
                records.setdefault(current, {})
                continue
            if current is None:
                continue
            m = STACK_RE.search(line)
            if m:
                r = records[current]
                r["stack"] = int(m.group(1))
                r["spill_stores"] = int(m.group(2))
                r["spill_loads"] = int(m.group(3))
                continue
            m = USED_RE.search(line)
            if m:
                r = records[current]
                r["registers"] = int(m.group(1))
                tail = m.group(2) or ""
                for n, kind in BYTE_RE.findall(tail):
                    if kind == "smem":
                        r["smem"] = int(n)
                    elif kind == "cmem[0]":
                        r["cmem0"] = int(n)
                continue
            for n, kind in BYTE_RE.findall(line):
                r = records[current]
                if kind == "smem":
                    r["smem"] = int(n)
                elif kind == "cmem[0]":
                    r["cmem0"] = int(n)

    if not records:
        print(f"no ptxas function records found in {args.log}", file=sys.stderr)
        return 2
    for name in sorted(records, key=demangle):
        emit(args.label, name, records[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
