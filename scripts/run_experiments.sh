#!/usr/bin/env bash
#
# Batch runner for the C3DGS compression experiments (SBrT article).
#
# Edit the CONFIG block below for one experimental case, give it a METHOD_LABEL,
# and run. Each case writes its own output tree under RESULTS_ROOT; the analysis
# step at the end aggregates across every case found there.
#
# The run is checkpointed: completed scene/seed combinations are recorded in
# <RESULTS_ROOT>/<METHOD_LABEL>/progress.tsv and skipped on a re-run, so an
# interrupted or partially failed batch can simply be started again.
#
#   ./scripts/run_experiments.sh
#
# ============================================================================
# CONFIG
# ============================================================================

# Each field can also be overridden for a single invocation without editing the
# file, e.g.:  METHOD_LABEL=Softmax USE_SOFTMAX=true ./scripts/run_experiments.sh
#
# --- paths -----------------------------------------------------------------
C3DGS_DIR=${C3DGS_DIR:-/home/luisserra/pibic/c3dgs}
SCRIPTS_DIR=${SCRIPTS_DIR:-/home/luisserra/pibic/scripts}
IMAGES_ROOT=${IMAGES_ROOT:-/censipam_data/luisserra/images}
MODELS_ROOT=${MODELS_ROOT:-/censipam_data/luisserra/models}
RESULTS_ROOT=${RESULTS_ROOT:-/home/luisserra/pibic/runs}
PYTHON_BIN=${PYTHON_BIN:-/home/luisserra/pibic/miniconda3/envs/c3dgs/bin/python}
# The c3dgs env has no scipy, so the significance tests would silently degrade to
# descriptive statistics only. The analysis scripts need nothing from torch, so
# run them under an interpreter that does have scipy.
ANALYSIS_PYTHON=${ANALYSIS_PYTHON:-/usr/bin/python3}

# --- this experimental case -------------------------------------------------
METHOD_LABEL=${METHOD_LABEL:-C3DGS}     # output subdir + label used in the analysis
BASELINE_LABEL=${BASELINE_LABEL:-C3DGS} # label stats_significance.py compares against

SCENES=(bicycle bonsai counter drjohnson flowers garden kitchen \
        playroom room stump train treehill truck)
SEEDS=(0 1 2 3 4)

# --- the six experiment knobs ----------------------------------------------
AUTO_CODEBOOK=${AUTO_CODEBOOK:-false}         # true|false (entropy-driven codebook size)
CODEBOOK_SIZE=${CODEBOOK_SIZE:-4096}          # used only when AUTO_CODEBOOK=false
CODEBOOK_SWEEP=()             # optional, e.g. (4 8 16 32 64 128 256 512 1024 2048)
                              #   if non-empty, loops these K instead of CODEBOOK_SIZE
SENSITIVITY_MODE=${SENSITIVITY_MODE:-energy}  # energy|abs|sq
ARITHMETIC_ENCODING=${ARITHMETIC_ENCODING:-false}  # true => AC; false => Morton DEFLATE
USE_SOFTMAX=${USE_SOFTMAX:-false}             # true => softmax; false => weighted distance

# --- execution control ------------------------------------------------------
FINETUNE_ITERATIONS=${FINETUNE_ITERATIONS:-5000}  # C3DGS default; 0 for a fast smoke test
AC_BACKEND=${AC_BACKEND:-cpu}      # cpu|gpu (only used when ARITHMETIC_ENCODING=true)
DATA_DEVICE=${DATA_DEVICE:-cuda}   # must be passed explicitly (see note below)
# Where torch caches the VGG16 weights that LPIPS needs. The inherited
# TORCH_HOME is validated below and replaced if it is not writable.
TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
RESUME=${RESUME:-true}             # skip runs already recorded as DONE
FORCE_RERUN=${FORCE_RERUN:-false}  # ignore checkpoints and redo everything
DRY_RUN=${DRY_RUN:-false}          # print the commands without running them
RUN_ANALYSIS=${RUN_ANALYSIS:-true} # after the runs, aggregate + significance + figures

# Optional space-separated overrides for the arrays above, e.g. SCENES_OVERRIDE="train truck"
[ -n "${SCENES_OVERRIDE:-}" ] && read -r -a SCENES <<< "$SCENES_OVERRIDE"
[ -n "${SEEDS_OVERRIDE:-}" ]  && read -r -a SEEDS  <<< "$SEEDS_OVERRIDE"
[ -n "${CODEBOOK_SWEEP_OVERRIDE:-}" ] && read -r -a CODEBOOK_SWEEP <<< "$CODEBOOK_SWEEP_OVERRIDE"

# ============================================================================
# END CONFIG -- no edits needed below
# ============================================================================

# Deliberately no `set -e`: a single failing scene must not abort a batch that
# may take days. Failures are recorded and the loop moves on.
set -uo pipefail

METHOD_DIR="$RESULTS_ROOT/$METHOD_LABEL"
PROGRESS="$METHOD_DIR/progress.tsv"
SUMMARY="$METHOD_DIR/run_summary.log"
FAILURES="$RESULTS_ROOT/failures.log"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$SUMMARY"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- preflight --------------------------------------------------------------
[ -x "$PYTHON_BIN" ] || die "PYTHON_BIN not executable: $PYTHON_BIN"
[ -d "$C3DGS_DIR" ]  || die "C3DGS_DIR not found: $C3DGS_DIR"

if [ "$RUN_ANALYSIS" = true ]; then
  [ -x "$ANALYSIS_PYTHON" ] || die "ANALYSIS_PYTHON not executable: $ANALYSIS_PYTHON"
  "$ANALYSIS_PYTHON" -c "import scipy, numpy, matplotlib" 2>/dev/null || \
    die "ANALYSIS_PYTHON ($ANALYSIS_PYTHON) needs scipy, numpy and matplotlib"
fi

case "$SENSITIVITY_MODE" in energy|abs|sq) ;; *)
  die "SENSITIVITY_MODE must be energy|abs|sq (got '$SENSITIVITY_MODE')" ;; esac

mkdir -p "$METHOD_DIR"
touch "$PROGRESS" "$SUMMARY"

# Fail fast on a broken environment rather than after N doomed runs. compress.py
# imports the arithmetic coder at module level, so this catches a missing
# `arithmeticcoding` (Nayuki reference implementation) up front.
if ! (cd "$C3DGS_DIR" && "$PYTHON_BIN" -c "import compress" >/dev/null 2>&1); then
  echo "ERROR: 'import compress' failed in $C3DGS_DIR. Details:" >&2
  (cd "$C3DGS_DIR" && "$PYTHON_BIN" -c "import compress" 2>&1 | tail -5) >&2
  die "fix the environment before launching a batch"
fi

# LPIPS pulls the VGG16 ImageNet weights through torch.hub on first use. If
# TORCH_HOME is unwritable the run dies *after* compressing, wasting the whole
# job, so fall back to a writable cache instead of failing 65 times over.
if ! mkdir -p "$TORCH_HOME/hub/checkpoints" 2>/dev/null; then
  echo "WARNING: TORCH_HOME '$TORCH_HOME' is not writable; falling back to $HOME/.cache/torch" >&2
  TORCH_HOME=$HOME/.cache/torch
  mkdir -p "$TORCH_HOME/hub/checkpoints" || die "cannot create a torch cache at $TORCH_HOME"
fi
export TORCH_HOME

# Every scene needs both a trained model and its source images.
missing=()
for scene in "${SCENES[@]}"; do
  [ -f "$MODELS_ROOT/$scene/cfg_args" ] || missing+=("$scene: no models/$scene/cfg_args")
  [ -d "$IMAGES_ROOT/$scene" ]          || missing+=("$scene: no images/$scene")
done
if [ ${#missing[@]} -gt 0 ]; then
  printf 'ERROR: %s\n' "${missing[@]}" >&2
  die "resolve the missing scenes above, or trim SCENES"
fi

# --- translate the knobs into compress.py flags -----------------------------
# Note: --not_sort_morton is never passed, so Morton sorting stays enabled. The
# DEFLATE baseline depends on it (compress.py rejects --no_ac without it).
# --data_device must be passed explicitly: ModelParams is built with sentinel=True,
# so any flag left off the command line becomes None, and get_combined_args only
# merges non-None values over the model's cfg_args -- which has no data_device.
# Omitting it makes Scene() die with "GroupParams has no attribute data_device".
common_flags=(
  --data_device "$DATA_DEVICE"
  --sensitivity_mode "$SENSITIVITY_MODE"
  --finetune_iterations "$FINETUNE_ITERATIONS"
)
[ "$ARITHMETIC_ENCODING" = true ] \
  && common_flags+=(--ac_backend "$AC_BACKEND") \
  || common_flags+=(--no_ac)
[ "$USE_SOFTMAX" = true ] || common_flags+=(--no_softmax)

# Which codebook configurations to iterate. "auto" means let the entropy
# criterion pick K; otherwise each entry is a literal codebook size.
if [ "$AUTO_CODEBOOK" = true ]; then
  ks=(auto)
elif [ ${#CODEBOOK_SWEEP[@]} -gt 0 ]; then
  ks=("${CODEBOOK_SWEEP[@]}")
else
  ks=("$CODEBOOK_SIZE")
fi

total=$(( ${#SCENES[@]} * ${#SEEDS[@]} * ${#ks[@]} ))
log "=== $METHOD_LABEL: $total runs (${#SCENES[@]} scenes x ${#SEEDS[@]} seeds x ${#ks[@]} K) ==="
log "sensitivity=$SENSITIVITY_MODE ac=$ARITHMETIC_ENCODING softmax=$USE_SOFTMAX \
auto_codebook=$AUTO_CODEBOOK K=${ks[*]} finetune=$FINETUNE_ITERATIONS"

cd "$C3DGS_DIR" || die "cannot cd to $C3DGS_DIR"

done_n=0 skip_n=0 fail_n=0 idx=0

for scene in "${SCENES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for k in "${ks[@]}"; do
      idx=$((idx + 1))
      run_key="${scene}:${seed}:${k}"

      # Output layout mirrors what extract_metrics.py expects: the scene name is
      # a path component and the seed is nested beneath it.
      if [ ${#ks[@]} -gt 1 ]; then
        out="$METHOD_DIR/$scene/K_$k/seed_$seed"
      else
        out="$METHOD_DIR/$scene/seed_$seed"
      fi

      # A run counts as complete only if it was recorded DONE *and* left a
      # non-empty results.json, so a truncated run is never mistaken for one.
      marker=$(printf 'DONE\t%s\t' "$run_key")
      if [ "$FORCE_RERUN" != true ] && [ "$RESUME" = true ] \
         && grep -qF -- "$marker" "$PROGRESS" && [ -s "$out/results.json" ]; then
        skip_n=$((skip_n + 1))
        log "[$idx/$total] skip $run_key (done)"
        continue
      fi

      # Clear a stale results.json so a failed retry cannot later read as complete.
      rm -f "$out/results.json"
      mkdir -p "$out"

      run_flags=("${common_flags[@]}")
      if [ "$k" = auto ]; then
        run_flags+=(--auto_codebook)
      else
        run_flags+=(--color_codebook_size "$k" --gaussian_codebook_size "$k")
      fi

      log "[$idx/$total] run $run_key -> $out"
      if [ "$DRY_RUN" = true ]; then
        printf '  %s compress.py --model_path %q --source_path %q --output_vq %q --seed %s %s\n' \
          "$PYTHON_BIN" "$MODELS_ROOT/$scene" "$IMAGES_ROOT/$scene" "$out" "$seed" "${run_flags[*]}"
        continue
      fi

      # --images is intentionally NOT passed: each model's cfg_args already names
      # the correct images/images_2/images_4 subfolder. --source_path IS passed,
      # because cfg_args still points at the original Windows training path.
      "$PYTHON_BIN" compress.py \
        --model_path  "$MODELS_ROOT/$scene" \
        --source_path "$IMAGES_ROOT/$scene" \
        --output_vq   "$out" \
        --seed "$seed" \
        "${run_flags[@]}" 2>&1 | tee "$out/run.log"
      status=${PIPESTATUS[0]}

      ts=$(date -Is)
      if [ "$status" -eq 0 ] && [ -s "$out/results.json" ]; then
        printf 'DONE\t%s\t%s\t%s\n' "$run_key" "$ts" "$out" >> "$PROGRESS"
        done_n=$((done_n + 1))
        log "[$idx/$total] ok   $run_key"
      else
        printf 'FAIL\t%s\t%s\t%s\n' "$run_key" "$ts" "$out" >> "$PROGRESS"
        printf '%s\t%s\t%s\texit=%s\t%s\n' "$ts" "$METHOD_LABEL" "$run_key" "$status" "$out/run.log" \
          >> "$FAILURES"
        fail_n=$((fail_n + 1))
        log "[$idx/$total] FAIL $run_key (exit=$status) -> $out/run.log"
      fi
    done
  done
done

log "=== $METHOD_LABEL done: $done_n ok, $skip_n skipped, $fail_n failed ==="
[ "$fail_n" -gt 0 ] && log "failures listed in $FAILURES; re-run this script to retry them"

# --- analysis ---------------------------------------------------------------
if [ "$RUN_ANALYSIS" != true ]; then
  exit 0
fi
if [ "$DRY_RUN" = true ]; then
  log "DRY_RUN: skipping analysis"
  exit 0
fi
if [ "$fail_n" -gt 0 ]; then
  log "WARNING: analysing with $fail_n failed run(s); numbers will be incomplete"
fi

# Every method subdirectory under RESULTS_ROOT becomes one labelled root, so the
# aggregate covers all cases run so far, not just this one.
roots=() labels=()
for d in "$RESULTS_ROOT"/*/; do
  [ -d "$d" ] || continue
  compgen -G "$d"'**/results.json' >/dev/null 2>&1 || \
    [ -n "$(find "$d" -name results.json -print -quit 2>/dev/null)" ] || continue
  roots+=("$d")
  labels+=("$(basename "$d")")
done

if [ ${#roots[@]} -eq 0 ]; then
  log "no results.json found under $RESULTS_ROOT; skipping analysis"
  exit 0
fi

CSV="$RESULTS_ROOT/metrics.csv"
log "aggregating ${#roots[@]} method(s): ${labels[*]}"
"$ANALYSIS_PYTHON" "$SCRIPTS_DIR/extract_metrics.py" "${roots[@]}" \
  --labels "${labels[@]}" -o "$CSV" 2>&1 | tee -a "$SUMMARY"

# Significance testing needs the baseline label to actually be present.
if printf '%s\n' "${labels[@]}" | grep -qx -- "$BASELINE_LABEL"; then
  "$ANALYSIS_PYTHON" "$SCRIPTS_DIR/stats_significance.py" "$CSV" \
    --baseline "$BASELINE_LABEL" \
    --latex "$RESULTS_ROOT/significance_table.tex" 2>&1 | tee -a "$SUMMARY"
else
  log "baseline '$BASELINE_LABEL' not among ${labels[*]}; skipping significance tests"
fi

# Only meaningful once a codebook sweep exists (needs several K per scene).
if [ ${#ks[@]} -gt 1 ] || grep -q "K_" "$PROGRESS" 2>/dev/null; then
  "$ANALYSIS_PYTHON" "$SCRIPTS_DIR/plot_entropy_figures.py" "$CSV" \
    -o "$RESULTS_ROOT/figures" 2>&1 | tee -a "$SUMMARY"
fi

log "analysis written to $CSV"
