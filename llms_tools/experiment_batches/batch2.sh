#!/usr/bin/env bash
# Batch 2: K=5 and K=10 (blockwise_sequential) across GPUs

child_pids=()
cleanup() {
  echo "\nTerminating batch 2 processes..."
  for pid in "${child_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid"
    fi
  done
  exit 1
}
trap cleanup SIGINT SIGTERM

echo "Starting Batch 2 experiments:"

# Parameters
ROUNDS=5000
BATCH_SIZE=128
REPORT_RATE=20
STEP_SIZES=(0.001 0.0001 0.00001)

# K=5 (blockwise_sequential) on GPU 0
for STEP in "${STEP_SIZES[@]}"; do
  echo "Launching K=5, step_size=$STEP on GPU 0 (blockwise_sequential)"
  python3 -m src.dol1 \
    --model cnn \
    --dataset_name mnist \
    --training_mode blockwise_sequential \
    --step_size $STEP \
    --batch_size $BATCH_SIZE \
    --rounds $ROUNDS \
    --K 5 \
    --cuda_core 0 \
    --report_sampling_rate $REPORT_RATE &
  child_pids+=("$!")
done

# K=10 (blockwise_sequential) on GPU 1
for STEP in "${STEP_SIZES[@]}"; do
  echo "Launching K=10, step_size=$STEP on GPU 1 (blockwise_sequential)"
  python3 -m src.dol1 \
    --model cnn \
    --dataset_name mnist \
    --training_mode blockwise_sequential \
    --step_size $STEP \
    --batch_size $BATCH_SIZE \
    --rounds $ROUNDS \
    --K 10 \
    --cuda_core 1 \
    --report_sampling_rate $REPORT_RATE &
  child_pids+=("$!")
done

echo "Launched ${#child_pids[@]} processes for Batch 2."
wait
echo "Batch 2 complete."