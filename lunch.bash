#!/bin/bash

# Define the number of instances
NUM_INSTANCES=5

# Define arrays for hyperparameters
declare -a stepsizes=("0.00005" "0.00005" "0.00005" "0.00005" "0.00005")
declare -a k_values=("0" "2" "5" "10" "20")
declare -a cuda_cores=(1 0 1 0 1)
declare -a training_modes=("entire" "blockwise_sequential" "blockwise_sequential" "blockwise_sequential" "blockwise_sequential")

# Define global variables
ROUNDS="2000"
REPORT_SAMPLING_RATE="20"
DATASET="cifar100"
BATCH_SIZE="128"

# Array to keep track of child process PIDs
child_pids=()

# Cleanup function to terminate child processes
cleanup() {
    echo "Terminating child processes..."
    for pid in "${child_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Terminating child process $pid"
            kill -TERM "$pid"
        fi
    done
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM signals and call the cleanup function
trap cleanup SIGINT SIGTERM

# Launch the instances in parallel
for ((i=0; i<$NUM_INSTANCES; i++))
do
    echo "Starting instance $i with stepsize ${stepsizes[$i]}, batch size $BATCH_SIZE, K ${k_values[$i]}, rounds $ROUNDS, dataset $DATASET, on CUDA core ${cuda_cores[$i]}, training mode ${training_modes[$i]}, report sampling rate $REPORT_SAMPLING_RATE"
    python3 dol1.py --step_size "${stepsizes[$i]}" --batch_size "$BATCH_SIZE" --K "${k_values[$i]}" --rounds "$ROUNDS" --dataset_name "$DATASET" --cuda_core "${cuda_cores[$i]}" --training_mode "${training_modes[$i]}" --report_sampling_rate "$REPORT_SAMPLING_RATE" &
    # Save the PID of the background process
    child_pids+=($!)
done

# Wait for all background processes to complete
wait

echo "All instances completed."