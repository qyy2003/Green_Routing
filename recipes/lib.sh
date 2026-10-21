# Shared helpers for the recipe scripts (sourced, not executed).
#
# Knobs (environment):
#   GR_ARTIFACTS_DIR   where runs go                      (default: <repo>/artifacts)
#   PARALLEL           training jobs at once              (recipe-specific default)
#   THREADS_PER_JOB    CPU threads per job (torch/OpenMP) (default 4)
#   PYTHON             interpreter for the helper scripts (default: python)
#
# Every job pins its CPU threads: eight torch processes with the default thread
# count (= all cores each) drove a 40-core box to load ~300 and slowed every job
# several-fold.

set -euo pipefail

RECIPES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$RECIPES/.." && pwd)"
: "${GR_ARTIFACTS_DIR:=$REPO/artifacts}"
export GR_ARTIFACTS_DIR
: "${PYTHON:=python}"
: "${THREADS_PER_JOB:=4}"
export OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
       OPENBLAS_NUM_THREADS="$THREADS_PER_JOB"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

need() {   # need <command> — fail early with a hint when the package isn't installed
  command -v "$1" >/dev/null 2>&1 || {
    echo "'$1' not found — activate the environment and run: pip install -e '.[dev]'" >&2
    exit 1; }
}

# pool <parallel> <logdir> <name> <command> [<name> <command> ...]
#   Runs the named commands, at most <parallel> at a time, each logging to
#   <logdir>/<name>.out (exit code in <name>.rc). Returns non-zero if any failed.
pool() {
  local par="$1" dir="$2" fail=0 name cmd
  local -a names=()
  shift 2
  mkdir -p "$dir"
  while (($#)); do
    name="$1"; cmd="$2"; shift 2
    names+=("$name")
    rm -f "$dir/$name.rc"
    while (( $(jobs -rp | wc -l) >= par )); do wait -n || true; done
    log "start $name"
    ( rc=0; bash -c "$cmd" > "$dir/$name.out" 2>&1 || rc=$?
      echo "$rc" > "$dir/$name.rc"
      if [[ "$rc" == 0 ]]; then log "done  $name"
      else log "FAIL  $name (exit $rc) — see $dir/$name.out"; fi ) &
  done
  wait
  for name in "${names[@]}"; do
    [[ "$(cat "$dir/$name.rc" 2>/dev/null)" == 0 ]] || fail=1
  done
  return "$fail"
}
