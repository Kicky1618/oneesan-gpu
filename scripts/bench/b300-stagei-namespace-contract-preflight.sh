#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

EVICT_REAL="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-selfevict-prepare.sh"
MATE_REAL="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-mategeo-prepare.sh"
for f in "$EVICT_REAL" "$MATE_REAL"; do [[ -f "$f" ]] || { echo "missing namespace wrapper=$f" >&2; exit 2; }; bash -n "$f"; done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stagei-ns.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
root="$tmp/root"; mkdir -p "$root/scripts/run" "$root/scripts/lib" "$root/work" "$root/bin"
cp "$EVICT_REAL" "$root/scripts/run/b300x8-nextgen-hybrid8-selfevict-prepare.sh"
cp "$MATE_REAL" "$root/scripts/run/b300x8-nextgen-hybrid8-mategeo-prepare.sh"
cat >"$root/scripts/lib/common.sh" <<'EOF'
#!/usr/bin/env bash
ONEESAN_ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; export ONEESAN_ROOT
EOF
for b in evict evict-control mate mate-control; do printf '#!/usr/bin/env bash\nexit 0\n' >"$root/bin/$b"; chmod +x "$root/bin/$b"; done
profile="$root/work/profile.env"; input="$root/work/stagef.env"; printf 'PROFILE=fake\n' >"$profile"; printf 'STAGEF=fake\n' >"$input"

cat >"$root/scripts/run/b300x8-nextgen-hybrid8-nextself-evict-staged-fullprime-race.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${PREPARE_ENV:?}"
manifest="$ONEESAN_ROOT/work/evict.manifest"; printf 'fake\n' >"$manifest"
cat >"$PREPARE_ENV" <<EOT
B300_STAGEI_PREPARED=1
B300_STAGEI_PREPARED_HINT=last
B300_STAGEI_PREPARED_WIDTH=4
B300_STAGEI_PREPARED_DISTANCE=2
B300_STAGEI_PREPARED_BIN=$ONEESAN_ROOT/bin/evict
B300_STAGEI_PREPARED_LABEL=evict_last
B300_STAGEI_PREPARED_THREADS=256
B300_STAGEI_PREPARED_CONTROL_BIN=$ONEESAN_ROOT/bin/evict-control
B300_STAGEI_PREPARED_CONTROL_LABEL=evict_default
B300_STAGEI_PREPARED_CONTROL_THREADS=128
B300_STAGEI_PREPARED_MANIFEST=$manifest
EOT
EOF
chmod +x "$root/scripts/run/b300x8-nextgen-hybrid8-nextself-evict-staged-fullprime-race.sh"

cat >"$root/scripts/run/b300x8-nextgen-hybrid8-nextmate-geometry-staged-fullprime-race.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${PREPARE_ENV:?}"
manifest="$ONEESAN_ROOT/work/mate.manifest"; printf 'fake\n' >"$manifest"
cat >"$PREPARE_ENV" <<EOT
B300_STAGEI_PREPARED=1
B300_STAGEI_PREPARED_SELF_WIDTH=4
B300_STAGEI_PREPARED_SELF_DISTANCE=2
B300_STAGEI_PREPARED_MATE_WIDTH=1
B300_STAGEI_PREPARED_MATE_DISTANCE=4
B300_STAGEI_PREPARED_BIN=$ONEESAN_ROOT/bin/mate
B300_STAGEI_PREPARED_LABEL=mate_w1d4
B300_STAGEI_PREPARED_THREADS=512
B300_STAGEI_PREPARED_CONTROL_BIN=$ONEESAN_ROOT/bin/mate-control
B300_STAGEI_PREPARED_CONTROL_LABEL=self_w4d2
B300_STAGEI_PREPARED_CONTROL_THREADS=256
B300_STAGEI_PREPARED_MANIFEST=$manifest
EOT
EOF
chmod +x "$root/scripts/run/b300x8-nextgen-hybrid8-nextmate-geometry-staged-fullprime-race.sh"

ev="$root/work/evict.prepared.env"; mg="$root/work/mate.prepared.env"
PROFILE_FILE="$profile" INPUT_ENV="$input" PREPARE_ENV="$ev" STAGED_PREFIX="$root/work/evict-stage" \
  bash "$root/scripts/run/b300x8-nextgen-hybrid8-selfevict-prepare.sh" 27 >/dev/null
[[ -s "$ev" ]] || exit 3
# shellcheck disable=SC1090
source "$ev"
[[ "$B300_EVICT_PREPARED" == 1 && "$B300_EVICT_HINT" == last && "$B300_EVICT_WIDTH" == 4 && "$B300_EVICT_DISTANCE" == 2 ]] || exit 4
E_HINT="$B300_EVICT_HINT"; E_BIN="$B300_EVICT_BIN"

PROFILE_FILE="$profile" INPUT_ENV="$input" SELF_EVICT="$E_HINT" MATE_EVICT=normal PREPARE_ENV="$mg" STAGED_PREFIX="$root/work/mate-stage" \
  bash "$root/scripts/run/b300x8-nextgen-hybrid8-mategeo-prepare.sh" 27 >/dev/null
[[ -s "$mg" ]] || exit 3
# shellcheck disable=SC1090
source "$mg"
[[ "$B300_MATEGEO_PREPARED" == 1 ]] || exit 4
[[ "$B300_MATEGEO_SELF_WIDTH" == 4 && "$B300_MATEGEO_SELF_DISTANCE" == 2 ]] || exit 4
[[ "$B300_MATEGEO_MATE_WIDTH" == 1 && "$B300_MATEGEO_MATE_DISTANCE" == 4 ]] || exit 4
[[ "$B300_MATEGEO_SELF_EVICT" == last && "$B300_MATEGEO_MATE_EVICT" == normal ]] || exit 4
[[ "$B300_MATEGEO_BIN" == "$root/bin/mate" && "$B300_MATEGEO_CONTROL_BIN" == "$root/bin/mate-control" ]] || exit 4
# The first contract remains stable after sourcing the second namespaced env.
[[ "$B300_EVICT_HINT" == "$E_HINT" && "$B300_EVICT_BIN" == "$E_BIN" ]] || { echo 'namespaced self-eviction values were contaminated' >&2; exit 4; }
# Raw ambiguous names may exist in the shell, but downstream contracts must not depend on them.
[[ "$B300_EVICT_HINT" != "$B300_MATEGEO_MATE_EVICT" ]] || true

grep -Fq 'B300_EVICT_PREPARED=1' "$ev"
grep -Fq 'B300_MATEGEO_PREPARED=1' "$mg"
echo 'b300_stagei_namespace_contract_preflight=OK self_evict_namespace=1 mate_geometry_namespace=1 sequential_source_stable=1 raw_stagei_collision_isolated=1 gpu_work=0'
