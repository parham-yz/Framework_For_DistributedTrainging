#!/bin/bash

# Define the hyperparameters for each instance
declare -a stepsizes=("1e-5" "1e-5" "1e-5" "1e-6" "1e-6" "1e-6" "1e-7" "1e-7")
declare -a Ks=(1 2 5 1 2 5 1 2)
declare -a cuda_cores=(0 1 2 3 4 5 6 7)

# Run the instances
for i in {0..7}
do
    python3 dol1.py ${stepsizes[$i]} ${Ks[$i]} 20000 mnist ${cuda_cores[$i]} &
done

# Wait for all background processes to finish
wait