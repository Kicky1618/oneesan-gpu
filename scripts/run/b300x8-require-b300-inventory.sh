#!/usr/bin/env bash
set -euo pipefail

command -v nvidia-smi >/dev/null || {
  echo 'B300 hardware guard: nvidia-smi required' >&2
  exit 2
}

inventory="$(nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader)" || {
  echo 'B300 hardware guard: failed to query GPU inventory' >&2
  exit 2
}
mapfile -t rows < <(printf '%s\n' "$inventory" | sed '/^[[:space:]]*$/d')
if (( ${#rows[@]} != 8 )); then
  echo "B300 hardware guard: need exactly 8 visible GPUs; got ${#rows[@]}" >&2
  exit 2
fi

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

declare -A seen_uuid=()
for i in "${!rows[@]}"; do
  IFS=',' read -r index name uuid memory driver extra <<<"${rows[$i]}"
  index="$(trim "${index:-}")"
  name="$(trim "${name:-}")"
  uuid="$(trim "${uuid:-}")"
  memory="$(trim "${memory:-}")"
  driver="$(trim "${driver:-}")"
  extra="$(trim "${extra:-}")"

  if [[ ! "$index" =~ ^[0-9]+$ || -z "$uuid" || -z "$memory" || -z "$driver" || -n "$extra" ]]; then
    echo "B300 hardware guard: malformed GPU inventory row $i: ${rows[$i]}" >&2
    exit 2
  fi
  if [[ ! "$name" =~ ^NVIDIA[[:space:]]+B300([[:space:]].*)?$ ]]; then
    echo "B300 hardware guard: GPU row $i is not NVIDIA B300: name=$name" >&2
    exit 2
  fi
  if [[ -n "${seen_uuid[$uuid]+x}" ]]; then
    echo "B300 hardware guard: duplicate GPU UUID at row $i: $uuid" >&2
    exit 2
  fi
  seen_uuid[$uuid]=1
done

# Emit exactly the inventory snapshot that passed validation. The canonical
# first-pass stores this same text in its provenance metadata.
printf '%s\n' "$inventory"
