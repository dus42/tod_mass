#!/usr/bin/env bash
# Reproduces every result in the paper. Run from this folder, with the data in data/ (see README.md).
#   ./run_all.sh          # everything
#   ./run_all.sh train    # resume from a step
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=..
mkdir -p logs results
PY="${PY:-uv run python -u}"
FROM=${1:-processing}
started=0
step() {
    local name=$1; shift
    [[ $name == "$FROM" ]] && started=1
    (( started )) || return 0
    echo "[$(date +%H:%M)] $name"
    "$@" > "logs/$name.log" 2>&1
}

step processing   $PY processing_tod.py
step features     $PY features_tod_full.py --workers 6 --overwrite
step frame        $PY all_features_tod.py
step train        $PY train_models.py
step scatter      $PY plot_test_scatter.py
step pertype_fig  $PY plot_pertype_mape.py
step pertype_tab  $PY make_pertype_table.py --out latex/table_pertype.tex
step unseen_mass  $PY experiment_unseen_types.py --target mass  --out results/results_tod_unseen_mass.csv
step unseen_ratio $PY experiment_unseen_types.py --target ratio --out results/results_tod_unseen_ratio.csv
step missing      $PY experiment_missing_descent.py --days 12 --workers 6
step repeats      $PY experiment_missing_descent.py --days 12 --workers 6 --only measured --repeats 5 --out results/results_tod_measured_repeats.csv
step masks_fig    $PY plot_mask_coverage.py
step numbers      $PY paper_numbers.py
step verify       $PY verify_pipeline.py
echo "[$(date +%H:%M)] done"
