#!/usr/bin/env bash
#
# Measure arithmetic-coding DECODE time and verify the round trip is lossless.
#
# compress.py only times the encode side, so decompression time -- one of the
# metrics reviewers asked for -- is not in results.json. This script replays
# decode_and_eval.py over the ac_output/ directories an AC run already left on
# disk, so nothing has to be recompressed.
#
# Each scene writes <ac_dir>/decoded_model/decode_eval_results.json holding
# time_ac_decode plus the PSNR/SSIM/LPIPS of the decoded model. Those files are
# picked up automatically the next time extract_metrics.py runs.
#
#   ./scripts/decode_timing.sh                    # all scenes of runs/AC_timing
#   SCENES_OVERRIDE="train truck" ./scripts/decode_timing.sh
#
set -uo pipefail

RESULTS_ROOT=${RESULTS_ROOT:-/home/luisserra/pibic/runs}
METHOD_LABEL=${METHOD_LABEL:-AC_timing}
SEED=${SEED:-0}
C3DGS_DIR=${C3DGS_DIR:-/home/luisserra/pibic/c3dgs}
PYTHON_BIN=${PYTHON_BIN:-/home/luisserra/pibic/miniconda3/envs/c3dgs/bin/python}
LOG_DIR=${LOG_DIR:-$RESULTS_ROOT/$METHOD_LABEL/decode_logs}

SCENES=(bicycle bonsai counter drjohnson flowers garden kitchen \
        playroom room stump train treehill truck)
[ -n "${SCENES_OVERRIDE:-}" ] && read -r -a SCENES <<< "$SCENES_OVERRIDE"

# The evaluation step computes LPIPS, which pulls the VGG16 weights through
# torch.hub. The inherited TORCH_HOME (~/.profile sets /d01/luis/torch_cache)
# is not writable here, and the failure lands *after* the decode has run, so
# the timing measurement is lost with it. Same guard as run_experiments.sh.
TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
if ! mkdir -p "$TORCH_HOME/hub/checkpoints" 2>/dev/null; then
  echo "WARNING: TORCH_HOME '$TORCH_HOME' is not writable; falling back to $HOME/.cache/torch" >&2
  TORCH_HOME=$HOME/.cache/torch
  mkdir -p "$TORCH_HOME/hub/checkpoints" || { echo "cannot create a torch cache" >&2; exit 1; }
fi
export TORCH_HOME
echo "TORCH_HOME=$TORCH_HOME"

mkdir -p "$LOG_DIR"
cd "$C3DGS_DIR" || { echo "cannot cd to $C3DGS_DIR" >&2; exit 1; }

ok=0 fail=0 skip=0
for scene in "${SCENES[@]}"; do
  run_dir="$RESULTS_ROOT/$METHOD_LABEL/$scene/seed_$SEED"
  ac_dir="$run_dir/ac_output"
  out_json="$ac_dir/decoded_model/decode_eval_results.json"

  if [ ! -d "$ac_dir" ]; then
    echo "SKIP $scene: no $ac_dir"; skip=$((skip + 1)); continue
  fi
  if [ -s "$out_json" ]; then
    echo "SKIP $scene: already decoded"; skip=$((skip + 1)); continue
  fi

  echo "=== $scene: decoding ==="
  start=$(date +%s)
  # tqdm writes its bars with \r; translating to \n and dropping the progress
  # lines keeps the log readable instead of one multi-megabyte line.
  "$PYTHON_BIN" decode_and_eval.py \
      --ac_dir "$ac_dir" --cfg_args "$run_dir/cfg_args" 2>&1 \
    | tr '\r' '\n' | grep -Ev "(gauss/s|it/s|\?gauss)" > "$LOG_DIR/$scene.log"
  status=${PIPESTATUS[0]}
  elapsed=$(( $(date +%s) - start ))

  if [ "$status" -eq 0 ] && [ -s "$out_json" ]; then
    echo "  ok   $scene in ${elapsed}s -> $out_json"
    ok=$((ok + 1))
  else
    echo "  FAIL $scene (exit=$status) -> $LOG_DIR/$scene.log"
    fail=$((fail + 1))
  fi
done

echo "=== decode done: $ok ok, $skip skipped, $fail failed ==="
