#!/usr/bin/env bash
# Launch preprocess_ocmr.py as several parallel processes, each handling a
# disjoint slice of the 165 `fs` rows (indices 0-164 in ocmr.csv).
#
# Usage: ./run_parallel.sh <raw_dir> <target_dir> [n_workers] [threads_per_worker]
set -euo pipefail

RAW_DIR="$1"
TARGET_DIR="$2"
N_WORKERS="${3:-4}"
THREADS_PER_WORKER="${4:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"

TOTAL_FS_ROWS=165  # rows 0-164 in ocmr.csv are the fs_ files
CHUNK=$(( (TOTAL_FS_ROWS + N_WORKERS - 1) / N_WORKERS ))

mkdir -p "$TARGET_DIR/logs"

echo "Launching $N_WORKERS workers ($THREADS_PER_WORKER threads each) over $TOTAL_FS_ROWS fs rows..."

pids=()
for ((i = 0; i < N_WORKERS; i++)); do
    start=$(( i * CHUNK ))
    if (( start > TOTAL_FS_ROWS - 1 )); then
        break
    fi
    end=$(( start + CHUNK - 1 ))
    if (( end > TOTAL_FS_ROWS - 1 )); then
        end=$(( TOTAL_FS_ROWS - 1 ))
    fi

    echo "  worker $i: csv rows $start-$end -> logs/worker_${i}.log"
    python "$SCRIPT_DIR/preprocess_ocmr.py" \
        --raw_dir "$RAW_DIR" --target_dir "$TARGET_DIR" \
        --csv_query "smp=='fs'" --csv_idx_range "$start" "$end" \
        --accelerations 8 12 16 20 --mask_types gro --no_csv_persist \
        > "$TARGET_DIR/logs/worker_${i}.log" 2>&1 &
    pids+=($!)
done

echo "Launched PIDs: ${pids[*]}"
wait
echo "All workers finished. Check $TARGET_DIR/logs/ for any errors."
