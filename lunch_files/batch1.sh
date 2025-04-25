#!/usr/bin/env bash
# Batch 1: K=1 (entire) on GPU 0 and K=2 (blockwise_sequential) on GPU 1

child_pids=()
cleanup() {
  echo "\nTerminating batch 1 processes..."
  for pid in "${child_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid"
    fi
  done
  exit 1
}
trap cleanup SIGINT SIGTERM

echo "Starting Batch 1 experiments:"

# Parameters
ROUNDS=5000
BATCH_SIZE=128
REPORT_RATE=20
STEP_SIZES=(0.001 0.0001 0.00001)

# K=1 (entire) on GPU 0
for STEP in "${STEP_SIZES[@]}"; do
  echo "Launching K=1, step_size=$STEP on GPU 0 (entire)"
  python3 -m src.dol1 \
    --model cnn \
    --dataset_name mnist \
    --training_mode entire \
    --step_size $STEP \
    --batch_size $BATCH_SIZE \
    --rounds $ROUNDS \
    --K 1 \
    --cuda_core 0 \
    --report_sampling_rate $REPORT_RATE &
  child_pids+=("$!")
done

# K=2 (blockwise_sequential) on GPU 1
for STEP in "${STEP_SIZES[@]}"; do
  echo "Launching K=2, step_size=$STEP on GPU 1 (blockwise_sequential)"
  python3 -m src.dol1 \
    --model cnn \
    --dataset_name mnist \
    --training_mode blockwise_sequential \
    --step_size $STEP \
    --batch_size $BATCH_SIZE \
    --rounds $ROUNDS \
    --K 2 \
    --cuda_core 1 \
    --report_sampling_rate $REPORT_RATE &
  child_pids+=("$!")
done

echo "Launched ${#child_pids[@]} processes for Batch 1."
wait
echo "Batch 1 complete."