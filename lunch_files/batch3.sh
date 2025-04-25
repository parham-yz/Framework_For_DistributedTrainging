#!/usr/bin/env bash
# Batch 3: K=20 on GPU0 and K=40 on GPU1 (blockwise_sequential)

child_pids=()
cleanup() {
  echo "\nTerminating Batch 3 processes..."
  for pid in "${child_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid"
    fi
  done
  exit 1
}
trap cleanup SIGINT SIGTERM

echo "Starting Batch 3 experiments: K=20 (GPU0), K=40 (GPU1)"

# Common parameters
ROUNDS=5000
BATCH_SIZE=128
REPORT_RATE=20
STEP_SIZES=(0.001 0.0001 0.00001)

# K=20 on GPU0
for STEP in "${STEP_SIZES[@]}"; do
  echo "Launching K=20, step_size=$STEP on GPU 0"
  python3 -m src.dol1 \
    --model cnn \
    --dataset_name mnist \
    --training_mode blockwise_sequential \
    --step_size $STEP \
    --batch_size $BATCH_SIZE \
    --rounds $ROUNDS \
    --K 20 \
    --cuda_core 0 \
    --report_sampling_rate $REPORT_RATE &
  child_pids+=("$!")
done

# K=40 on GPU1
for STEP in "${STEP_SIZES[@]}"; do
  echo "Launching K=40, step_size=$STEP on GPU 1"
  python3 -m src.dol1 \
    --model cnn \
    --dataset_name mnist \
    --training_mode blockwise_sequential \
    --step_size $STEP \
    --batch_size $BATCH_SIZE \
    --rounds $ROUNDS \
    --K 40 \
    --cuda_core 1 \
    --report_sampling_rate $REPORT_RATE &
  child_pids+=("$!")
done

echo "Launched ${#child_pids[@]} processes for Batch 3."
wait
echo "Batch 3 complete."