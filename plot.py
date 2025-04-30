import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from termcolor import colored
import subprocess
import sys
import time
import threading
import re # Import the regular expression module
import shutil # Added for save_and_export_logs
from datetime import datetime # Added for save_and_export_logs
import zipfile # Added for save_and_export_logs

import json
import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import itertools # Import itertools for cycling through colors
# ... (Keep all other functions like plot_results, plot_with_variance, etc., as they are) ...


COLORS = ['blue', 'black', 'red', 'purple', 'green', 'brown', 'gray', 'olive', 'cyan']

def plot_results(K_values, step_sizes, dataset, beta=0):
    # Ensure the figures directory exists
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Use getBestStepsizes to get the best step sizes
    best_step_sizes = getBestStepsizes(K_values, step_sizes, dataset)

    plt.figure(figsize=(12, 8))
    found_data = False # Flag to check if any data was plotted

    for K, step_size in best_step_sizes.items():
        log_files = [f"saved_logs/{file}" for file in os.listdir('saved_logs') if file.startswith(f"lr={step_size}_K={K}_dataset={dataset}")]
        if log_files:
            all_loss_logs = []
            min_len = float('inf') # To handle logs of different lengths if needed
            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        log_data = json.load(f)
                        if log_data.get('loss_log'): # Check if loss_log exists and is not empty
                             loss_log = log_data['loss_log']
                             all_loss_logs.append(loss_log)
                             min_len = min(min_len, len(loss_log))
                        else:
                            print(f"Warning: 'loss_log' empty or missing in {log_file}")
                except json.JSONDecodeError:
                    print(f"Error reading JSON from {log_file}")
                except Exception as e:
                    print(f"Error processing {log_file}: {e}")


            # Truncate all logs to the minimum length found across files for this K, step_size pair
            if all_loss_logs and min_len != float('inf'):
                 truncated_loss_logs = [log[:min_len] for log in all_loss_logs]

                 # Compute running average and variance only if we have data
                 if truncated_loss_logs:
                    running_avg = []
                    running_var = []
                    initial_losses = [log[0] for log in truncated_loss_logs if len(log) > 0]
                    if not initial_losses: continue # Skip if no valid initial losses

                    avg = np.mean(initial_losses) # Initialize with mean of first point across logs
                    for i in range(min_len):
                        losses_at_i = [log[i] for log in truncated_loss_logs]
                        current_mean = np.mean(losses_at_i)
                        avg = beta * avg + (1 - beta) * current_mean
                        var = np.var(losses_at_i)
                        running_avg.append(avg)
                        running_var.append(var)

                    if running_avg: # Check if we computed anything
                        iterations = np.arange(len(running_avg)) * 200 # Assuming 200 rounds per log point
                        # Add a small epsilon to running_avg and running_var before taking log10 to avoid log(0) or log(negative)
                        log_avg = np.log10(np.maximum(running_avg, 1e-10))
                        std_dev = np.sqrt(np.maximum(running_var, 0)) # Ensure variance is non-negative
                        lower_bound = np.log10(np.maximum(running_avg - std_dev, 1e-10))
                        upper_bound = np.log10(np.maximum(running_avg + std_dev, 1e-10))

                        plt.plot(iterations, log_avg, label=f'K={K}, step_size={step_size}')
                        plt.fill_between(iterations, lower_bound, upper_bound, alpha=0.2)
                        found_data = True

    if not found_data:
        print(f"No data found to plot for dataset: {dataset}")
        plt.close() # Close the empty plot
        return

    plt.xlabel('Rounds')
    plt.ylabel('Loss ($log_{10}$ scale)') # Use LaTeX for log10
    plt.title(f'Comparison of Loss vs Rounds for Best Step Sizes (Dataset: {dataset})')
    plt.legend()
    plot_filename_base = f'comparison_loss_vs_rounds_best_ss_{dataset}'
    plt.savefig(os.path.join(figures_dir, f'{plot_filename_base}.png'))
    plt.savefig(os.path.join(figures_dir, f'{plot_filename_base}.pdf'))
    # Save figure as PDF with specific naming convention for 'svhn' dataset
    if dataset == 'svhn':
        date_str = time.strftime("%Y%m%d")
        plt.savefig(os.path.join(figures_dir, f'svhn_{date_str}.pdf'))

    plt.close()

def plot_with_variance(K_values, step_sizes, dataset):
    # Ensure the figures directory exists
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    plt.figure(figsize=(12, 8))
    found_data = False # Flag to check if any data was plotted

    epsilon = 1e-10  # Small value to avoid log of zero or negative numbers

    for K in K_values:
        for step_size in step_sizes:
            log_files = [f"saved_logs/{file}" for file in os.listdir('saved_logs') if file.startswith(f"lr={step_size}_K={K}_dataset={dataset}")]
            if len(log_files) >= 1:  # Ensure there is at least 1 log file
                all_loss_logs = []
                for log_file in log_files:
                    try:
                        with open(log_file, 'r') as f:
                            log_data = json.load(f)
                            if log_data.get('loss_log'): # Check if loss_log exists and is not empty
                                 all_loss_logs.append(log_data['loss_log'])
                            else:
                                print(f"Warning: 'loss_log' empty or missing in {log_file}")
                    except json.JSONDecodeError:
                        print(f"Error reading JSON from {log_file}")
                    except Exception as e:
                        print(f"Error processing {log_file}: {e}")


                if all_loss_logs:
                    # Pad logs with NaN if they have different lengths
                    try:
                        max_len = max(len(log) for log in all_loss_logs) if all_loss_logs else 0
                    except ValueError: # Handles case where all_loss_logs might be empty despite check
                         max_len = 0

                    if max_len > 0:
                        padded_loss_logs = [log + [np.nan] * (max_len - len(log)) for log in all_loss_logs]

                        # Compute mean and variance, ignoring NaN values
                        loss_matrix = np.array(padded_loss_logs)
                        # Check if the matrix contains only NaNs before proceeding
                        if np.all(np.isnan(loss_matrix)):
                            print(f"Warning: All loss data is NaN for K={K}, step_size={step_size}, dataset={dataset}. Skipping.")
                            continue

                        mean_loss_log = np.nanmean(loss_matrix, axis=0)
                        var_loss_log = np.nanvar(loss_matrix, axis=0)

                        # Ensure mean_loss_log is not all NaNs
                        if np.all(np.isnan(mean_loss_log)):
                             print(f"Warning: Mean loss is NaN for K={K}, step_size={step_size}, dataset={dataset}. Skipping.")
                             continue


                        # Smooth variance (optional, but was in the original code)
                        # smoothing makes less sense when variance can be NaN, handle NaNs
                        smoothed_var_loss_log = []
                        beta = 0.9  # Smoothing factor
                        smoothed_var = np.nan # Initialize as NaN

                        for var in var_loss_log:
                            if not np.isnan(var):
                                if np.isnan(smoothed_var): # First valid variance value
                                     smoothed_var = var
                                else:
                                     smoothed_var = beta * smoothed_var + (1 - beta) * var
                            smoothed_var_loss_log.append(smoothed_var) # Append NaN if var is NaN

                        var_loss_log = np.array(smoothed_var_loss_log)
                        # Replace remaining NaNs in variance with 0 for plotting sqrt
                        var_loss_log = np.nan_to_num(var_loss_log, nan=0.0)


                        iterations = np.arange(len(mean_loss_log)) * 200 # Assuming 200 rounds per log point
                        # Plot only if mean_loss_log is not empty
                        if len(mean_loss_log) > 0:
                            log_mean = np.log10(np.maximum(mean_loss_log, epsilon)) # Use maximum to avoid log(<=0)
                            # Use np.maximum to avoid taking sqrt of negative numbers due to variance
                            variance_sqrt = np.sqrt(np.maximum(var_loss_log, 0))
                            lower_bound = np.log10(np.maximum(mean_loss_log - variance_sqrt, epsilon))
                            upper_bound = np.log10(np.maximum(mean_loss_log + variance_sqrt, epsilon))

                            plt.plot(iterations, log_mean, label=f'K={K}, step_size={step_size}')
                            plt.fill_between(iterations, lower_bound, upper_bound, alpha=0.2)
                            found_data = True


    if not found_data:
        print(f"No data found to plot with variance for dataset: {dataset}")
        plt.close() # Close the empty plot
        return

    plt.xlabel('Rounds')
    plt.ylabel('Loss ($log_{10}$ scale)')
    plt.title(f'Loss vs Rounds with Variance (Dataset: {dataset})')
    plt.legend()
    plot_filename = f'comparison_loss_vs_rounds_with_variance_{dataset}.png'
    plt.savefig(os.path.join(figures_dir, plot_filename))
    plt.close()

def plot_results_for_specific_K(K, step_sizes, dataset):
    # Ensure the figures directory exists
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    plt.figure(figsize=(12, 8))
    found_data = False # Flag to check if any data was plotted

    for step_size in step_sizes:
        log_files = [f"saved_logs/{file}" for file in os.listdir('saved_logs') if file.startswith(f"lr={step_size}_K={K}_dataset={dataset}")]
        if log_files:
            all_loss_logs = []
            for log_file in log_files:
                 try:
                     with open(log_file, 'r') as f:
                         log_data = json.load(f)
                         if log_data.get('loss_log'): # Check if loss_log exists and is not empty
                              all_loss_logs.append(log_data['loss_log'])
                         else:
                             print(f"Warning: 'loss_log' empty or missing in {log_file}")
                 except json.JSONDecodeError:
                     print(f"Error reading JSON from {log_file}")
                 except Exception as e:
                     print(f"Error processing {log_file}: {e}")

            if all_loss_logs:
                # Compute average loss log, handling potential different lengths
                try:
                    max_len = max(len(log) for log in all_loss_logs) if all_loss_logs else 0
                except ValueError:
                    max_len = 0

                if max_len > 0:
                    padded_loss_logs = [log + [np.nan] * (max_len - len(log)) for log in all_loss_logs]
                    loss_matrix = np.array(padded_loss_logs)
                    if np.all(np.isnan(loss_matrix)):
                        print(f"Warning: All loss data is NaN for K={K}, step_size={step_size}, dataset={dataset}. Skipping.")
                        continue

                    avg_loss_log = np.nanmean(loss_matrix, axis=0)

                     # Ensure avg_loss_log is not all NaNs
                    if np.all(np.isnan(avg_loss_log)):
                         print(f"Warning: Average loss is NaN for K={K}, step_size={step_size}, dataset={dataset}. Skipping.")
                         continue

                    # Add a small epsilon before taking log to avoid log(0) or log(negative)
                    # Use log10 for consistency with other plots
                    log_avg = np.log10(np.maximum(avg_loss_log, 1e-10))
                    iterations = np.arange(len(log_avg)) * 200 # Assuming 200 rounds per log point

                    # Plot only if log_avg is not empty
                    if len(log_avg) > 0:
                        plt.plot(iterations, log_avg, label=f'step_size={step_size}')
                        found_data = True


    if not found_data:
        print(f"No data found to plot for K={K}, dataset: {dataset}")
        plt.close() # Close the empty plot
        return

    plt.xlabel('Rounds') # Changed from Epochs to Rounds
    plt.ylabel('Loss ($log_{10}$ scale)')
    plt.title(f'Comparison of Loss vs Rounds for K={K} (Dataset: {dataset})')
    plt.legend()
    plot_filename = f'comparison_loss_vs_rounds_K={K}_{dataset}.png'
    plt.savefig(os.path.join(figures_dir, plot_filename))
    plt.close()


def _plot_single_report_dir(report_subdir_path, subdir_name):
    """Helper function to plot reports from a single subdirectory."""
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    print(f"Processing reports in: {report_subdir_path}")

    # Use a nested defaultdict to store data by K and then by metric type
    data_by_k = defaultdict(lambda: defaultdict(list))
    found_files = False

    for filename in os.listdir(report_subdir_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(report_subdir_path, filename)
            found_files = True
            k_value = None # k_value is now read per file

            try:
                with open(filepath, 'r') as file:
                    for line in file:
                        # Extract hyperparameters first to get the K value for the current file
                        if 'Hyperparameters:' in line:
                            try:
                                # Remove potential leading/trailing whitespace before loading
                                hyper_str = line.split('Hyperparameters:', 1)[1].strip()
                                hyperparameters = json.loads(hyper_str)
                                k_value = hyperparameters.get('K')
                                if k_value is not None:
                                    print(f"Found K={k_value} in file: {filename}")
                            except json.JSONDecodeError:
                                print(f"Warning: Error decoding JSON in file: {filepath} - Line: {line.strip()}")
                                k_value = None # Reset k_value if JSON is bad
                                continue # Skip to next line

                        # Extract metrics from lines starting with '[' and containing 'Time'
                        # Process lines only if a valid k_value has been found in the current file
                        if line.startswith('[') and 'Time' in line and k_value is not None:
                            try:
                                # Use regex to find Time, Energy, and Accuracy values
                                time_match = re.search(r'Time (\d+\.?\d*):', line)
                                energy_match = re.search(r'Energy = (\d+\.?\d*)', line)
                                accuracy_match = re.search(r'Accuracy = (\d+\.?\d*)', line)

                                current_time = float(time_match.group(1)) if time_match else np.nan
                                current_energy = float(energy_match.group(1)) if energy_match else np.nan
                                # Handle accuracy potentially having '%' sign
                                if accuracy_match:
                                     acc_str = accuracy_match.group(1).replace('%', '')
                                     current_accuracy = float(acc_str)
                                else:
                                     current_accuracy = np.nan

                                # Append metrics directly to the list for the current k_value
                                data_by_k[k_value]['time'].append(current_time)
                                data_by_k[k_value]['energy'].append(current_energy)
                                data_by_k[k_value]['accuracy'].append(current_accuracy)

                            except (ValueError, AttributeError, TypeError) as e:
                                print(f"Warning: Error parsing metrics in line: {line.strip()} in file: {filepath} - {e}")
                                # Append NaN for all metrics if parsing fails for any
                                data_by_k[k_value]['time'].append(np.nan)
                                data_by_k[k_value]['energy'].append(np.nan)
                                data_by_k[k_value]['accuracy'].append(np.nan)


            except Exception as e:
                 print(f"Error reading or processing file {filepath}: {e}")

    if not found_files:
        print(f"No .txt report files found in {report_subdir_path}")
        return
    if not data_by_k:
        print(f"No valid data extracted from reports in {report_subdir_path}")
        return

    # --- Plotting ---
    plt.figure(figsize=(14, 6))
    plot_created = False

    # Define a list of colors to cycle through
    colors = COLORS
    color_cycle = itertools.cycle(colors) # Create a cycle iterator

    # Energy plot on a log scale
    plt.subplot(1, 2, 1)
    energy_plotted = False
    # Sort K values for consistent legend and color order
    sorted_k = sorted(data_by_k.keys())
    for k in sorted_k:
        vals = data_by_k[k]
        # Ensure lists are not empty before processing
        if vals['time'] and vals['energy']:
            # Convert to numpy arrays for easier filtering
            times = np.array(vals['time'])
            energies = np.array(vals['energy'])

            # Filter out NaN values for both time and energy simultaneously
            valid_indices = ~np.isnan(times) & ~np.isnan(energies) & (energies > 0) # Also ensure energy > 0 for log
            valid_times = times[valid_indices]
            valid_energy = energies[valid_indices]

            # Sort by time before plotting
            sort_indices = np.argsort(valid_times)
            valid_times_sorted = valid_times[sort_indices]
            valid_energy_sorted = valid_energy[sort_indices]

            if len(valid_times_sorted) > 0:
                 # Get the next color from the cycle
                 color = next(color_cycle)
                 plt.plot(valid_times_sorted, np.log10(valid_energy_sorted), label=f'K={k}', color=color)
                 energy_plotted = True
                 plot_created = True

    if energy_plotted:
        plt.title(f'Log-Scale Energy over Time ({subdir_name})')
        plt.xlabel('Time (s)')
        plt.ylabel('Log-Scale Energy ($log_{10}$)') # Updated label for clarity
        plt.legend()
    else:
        plt.title(f'No valid energy data ({subdir_name})')

    # Reset the color cycle for the second plot to ensure consistent colors for the same K values
    color_cycle = itertools.cycle(colors)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    accuracy_plotted = False
    for k in sorted_k: # Use the same sorted K values and order
        vals = data_by_k[k]
        if vals['time'] and vals['accuracy']:
             # Convert to numpy arrays
            times = np.array(vals['time'])
            accuracies = np.array(vals['accuracy'])

            # Filter out NaN values for both time and accuracy simultaneously
            valid_indices = ~np.isnan(times) & ~np.isnan(accuracies)
            valid_times = times[valid_indices]
            valid_accuracy = accuracies[valid_indices]

             # Sort by time before plotting
            sort_indices = np.argsort(valid_times)
            valid_times_sorted = valid_times[sort_indices]
            valid_accuracy_sorted = valid_accuracy[sort_indices]

            if len(valid_times_sorted) > 0:
                # Get the next color from the cycle
                color = next(color_cycle)
                plt.plot(valid_times_sorted, valid_accuracy_sorted, label=f'K={k}', color=color)
                accuracy_plotted = True
                plot_created = True

    if accuracy_plotted:
        plt.title(f'Accuracy over Time ({subdir_name})')
        plt.xlabel('Time (s)')
        plt.ylabel('Accuracy (%)')
        plt.legend()
    else:
         plt.title(f'No valid accuracy data ({subdir_name})')


    if plot_created:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        # Include subdirectory name in the saved figure filenames
        energy_filename = os.path.join(figures_dir, f'log_energy_over_time_{subdir_name}.png')
        accuracy_filename = os.path.join(figures_dir, f'accuracy_over_time_{subdir_name}.png')
        # Save the combined plot
        combined_filename1 = os.path.join(figures_dir, f'report_plots_{subdir_name}.png')
        combined_filename2 = os.path.join(figures_dir, f'report_plots_{subdir_name}.pdf')

        print(f"Saving combined plot to {subdir_name}")
        plt.savefig(combined_filename1)
        plt.savefig(combined_filename2)

        # If you want separate files as well:
        # plt.subplot(1, 2, 1) # Reactivate subplot
        # plt.savefig(energy_filename)
        # plt.subplot(1, 2, 2) # Reactivate subplot
        # plt.savefig(accuracy_filename)
    else:
        print(f"No data plotted for {subdir_name}.")


    plt.close() # Close the figure


def plot_reports(base_reports_dir='reports'):
    """Plots reports by iterating through subdirectories in the base reports directory."""
    if not os.path.exists(base_reports_dir):
        print(f"Base reports directory not found: {base_reports_dir}")
        return

    subdirs_found = False
    for item in os.listdir(base_reports_dir):
        item_path = os.path.join(base_reports_dir, item)
        if os.path.isdir(item_path):
            subdirs_found = True
            _plot_single_report_dir(item_path, item) # Pass subdir path and name

    if not subdirs_found:
        print(f"No subdirectories found in {base_reports_dir}. Attempting to plot files in base directory.")
        # Optionally, you could call _plot_single_report_dir on the base_reports_dir itself
        # if you expect .txt files directly within 'reports' sometimes.
        _plot_single_report_dir(base_reports_dir, os.path.basename(base_reports_dir)) # Treat base dir as a single report dir
# Assuming your report files are in subdirectories within a 'reports' directory
# Example usage:
# plot_reports('reports')

def getBestStepsizes(K_values, step_sizes, dataset):
    best_results = {}
    saved_logs_dir = 'saved_logs' # Define directory

    if not os.path.exists(saved_logs_dir):
        print(f"Directory '{saved_logs_dir}' not found. Cannot determine best step sizes.")
        return best_results

    print(f"\nDetermining best step sizes for dataset: {dataset}")
    for K in K_values:
        best_step_size = None
        best_min_loss = float('inf')
        found_logs_for_K = False # Flag for K

        for step_size in step_sizes:
            # Correctly construct the search pattern
            pattern = f"lr={step_size}_K={K}_dataset={dataset}"
            log_files = [f for f in os.listdir(saved_logs_dir) if f.startswith(pattern) and f.endswith('.json')]

            if log_files:
                found_logs_for_K = True # Found logs for this K at least once
                min_losses_for_step = []
                for log_filename in log_files:
                    log_filepath = os.path.join(saved_logs_dir, log_filename)
                    try:
                        with open(log_filepath, 'r') as f:
                            log_data = json.load(f)
                            loss_log = log_data.get('loss_log') # Use .get for safety
                            if loss_log: # Check if loss_log is not empty or None
                                # Filter out potential non-numeric values or NaNs before min
                                numeric_losses = [l for l in loss_log if isinstance(l, (int, float)) and not np.isnan(l)]
                                if numeric_losses:
                                    min_losses_for_step.append(min(numeric_losses))
                                else:
                                     print(f"Warning: No valid numeric losses found in {log_filename}")
                            else:
                                 print(f"Warning: 'loss_log' empty or missing in {log_filename}")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in {log_filepath}")
                    except Exception as e:
                        print(f"Error processing file {log_filepath}: {e}")

                if min_losses_for_step: # Check if we found any minimum losses for this step size
                    avg_min_loss = np.mean(min_losses_for_step)
                    if avg_min_loss < best_min_loss:
                        best_min_loss = avg_min_loss
                        best_step_size = step_size
                # else: # Optional: print if no valid min losses found for a specific step size
                    # print(f"  No valid min losses found for K={K}, step_size={step_size}")


        if best_step_size is not None:
            best_results[K] = best_step_size
            print(f"  For K={K}, Best Step Size: {best_step_size} (Avg Min Loss: {best_min_loss:.4e})")
        elif found_logs_for_K:
             print(f"  For K={K}, No valid min losses found across any step size.")
        # else: # Optional: print if no logs were found at all for K
             # print(f"  For K={K}, No log files found.")


    return best_results


def extract_K_and_stepsizes(directory):
    K_values = set()
    step_sizes = set()
    datasets = set()

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return sorted(list(K_values)), sorted(list(step_sizes), key=float), sorted(list(datasets))


    print(f"Extracting hyperparameters from: {directory}")
    found_files_count = 0
    skipped_files_count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            found_files_count += 1
            parts = filename.split('_')
            # Example: lr=0.01_K=10_dataset=cifar10_...json
            if len(parts) >= 3:
                try:
                    lr_part = parts[0]
                    k_part = parts[1]
                    dataset_part = parts[2]

                    if lr_part.startswith("lr=") and k_part.startswith("K=") and dataset_part.startswith("dataset="):
                         step_size = lr_part.split('=')[1]
                         K = int(k_part.split('=')[1])
                         # Handle potential extra parts in dataset name or filename
                         dataset = dataset_part.split('=')[1]
                         # Remove .json extension if present
                         if dataset.endswith('.json'):
                              dataset = dataset[:-5]

                         # Validate step_size can be converted to float
                         _ = float(step_size)

                         step_sizes.add(step_size)
                         K_values.add(K)
                         datasets.add(dataset)
                    else:
                         # print(f"Skipping file with unexpected format: {filename}")
                         skipped_files_count += 1
                         continue
                except (IndexError, ValueError) as e:
                    print(f"Skipping malformed filename: {filename} - {e}")
                    skipped_files_count += 1
                    continue
            else:
                 # print(f"Skipping file with too few parts: {filename}")
                 skipped_files_count += 1
                 continue

    print(f"Found {found_files_count} JSON files, successfully parsed {found_files_count - skipped_files_count}, skipped {skipped_files_count}.")
    # Sort K values numerically, step sizes as floats, datasets alphabetically
    sorted_K = sorted(list(K_values))
    sorted_steps = sorted(list(step_sizes), key=float)
    sorted_datasets = sorted(list(datasets))

    print(f"Extracted K values: {sorted_K}")
    print(f"Extracted Step Sizes: {sorted_steps}")
    print(f"Extracted Datasets: {sorted_datasets}")

    return sorted_K, sorted_steps, sorted_datasets

def get_free_gpu_cores(reports_folder='reports'):
    total_cores = set()
    used_cores = set()

    if not os.path.exists(reports_folder):
        print(f"Reports directory '{reports_folder}' not found.")
        # Try searching in common parent directory if it exists
        parent_dir = os.path.dirname(reports_folder)
        if os.path.exists(parent_dir):
             print(f"Searching for reports in subdirectories of '{parent_dir}'...")
             # This part might need adjustment based on where reports *actually* are
             # For now, let's just return empty if the specific folder isn't found
             return []
        else:
            return []

    print(f"Checking GPU core status in reports folder: {reports_folder}")
    files_processed = 0
    cores_found = 0

    # Walk through the reports folder and its subdirectories
    for root, dirs, files in os.walk(reports_folder):
        for filename in files:
            if filename.endswith('.txt'):
                filepath = os.path.join(root, filename)
                files_processed += 1
                status_line = None
                hyperparameters_line = None
                hyperparameters = {}
                cuda_core = None

                try:
                    with open(filepath, 'r') as file:
                        lines = file.readlines()
                        for line in lines:
                            # Use case-insensitive matching and strip whitespace
                            if line.strip().lower().startswith('status:'):
                                status_line = line.strip()
                            elif line.strip().lower().startswith('hyperparameters:'):
                                hyperparameters_line = line.strip()
                                # Attempt to parse JSON immediately if found
                                try:
                                    # Extract JSON string after 'Hyperparameters:'
                                    json_str = hyperparameters_line.split(':', 1)[1].strip()
                                    hyperparameters = json.loads(json_str)
                                    cuda_core_val = hyperparameters.get('cuda_core')
                                    # Ensure cuda_core is treated consistently (e.g., as string or int)
                                    if cuda_core_val is not None:
                                         cuda_core = str(cuda_core_val) # Store as string for consistency
                                         cores_found += 1
                                except json.JSONDecodeError:
                                    print(f"Warning: Error decoding JSON in file: {filepath} - Line: {hyperparameters_line}")
                                    hyperparameters = {} # Reset if decode fails
                                    cuda_core = None
                                except Exception as e: # Catch other potential errors during parsing
                                     print(f"Warning: Error parsing hyperparameters in file {filepath}: {e}")
                                     hyperparameters = {}
                                     cuda_core = None


                        # Process status only if we successfully found a cuda_core
                        if cuda_core is not None:
                            total_cores.add(cuda_core)
                            if status_line:
                                status = status_line.split(':', 1)[1].strip()
                                # Consider a core used if status is not explicitly 'dead' (case-insensitive)
                                if status.lower() != 'dead':
                                    used_cores.add(cuda_core)
                            else:
                                # If status line is missing, conservatively assume the core might be used
                                # print(f"Warning: Status line missing in {filepath}, assuming core {cuda_core} might be used.")
                                used_cores.add(cuda_core) # Or decide not to add to used_cores if status unknown


                except Exception as e:
                    print(f"Error reading or processing file {filepath}: {e}")

    print(f"Processed {files_processed} report files, found {cores_found} CUDA core mentions.")
    # Convert cores to integers for sorting if possible, otherwise sort as strings
    try:
        sorted_total_cores = sorted([int(c) for c in total_cores])
        sorted_used_cores = sorted([int(c) for c in used_cores])
    except ValueError:
         print("Warning: Could not convert all core IDs to integers, sorting as strings.")
         sorted_total_cores = sorted(list(total_cores))
         sorted_used_cores = sorted(list(used_cores))


    # Calculate free cores (convert back to set for difference)
    free_cores_set = set(map(str, sorted_total_cores)) - set(map(str, sorted_used_cores))

    # Sort the final list of free cores
    try:
        sorted_free_cores = sorted([int(c) for c in free_cores_set])
    except ValueError:
         sorted_free_cores = sorted(list(free_cores_set))


    print(f"Total Unique GPU Cores Found: {sorted_total_cores}")
    print(f"Currently Used GPU Cores: {sorted_used_cores}")
    print(f"Available Free GPU Cores: {sorted_free_cores}")
    return sorted_free_cores

def monitor_reports(base_reports_folder='reports'):
    if not os.path.exists(base_reports_folder):
        print(f"Base reports directory '{base_reports_folder}' not found.")
        return

    # --- Nested Helper Functions ---
    def read_report(filepath):
        """Reads a single report file."""
        if not os.path.exists(filepath):
            return None, [], {} # Return empty if file no longer exists

        status_line = None
        progress_lines = []
        hyperparameters = {}

        try:
            with open(filepath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line.lower().startswith('status:'):
                        status_line = stripped_line
                    # Assuming progress lines reliably start with '['
                    elif stripped_line.startswith('['):
                        progress_lines.append(stripped_line)
                    elif stripped_line.lower().startswith('hyperparameters:'):
                        try:
                            json_str = stripped_line.split(':', 1)[1].strip()
                            hyperparameters = json.loads(json_str)
                        except json.JSONDecodeError:
                            print(f"Warning: Error decoding JSON in file: {filepath} - Line: {stripped_line}")
                            hyperparameters = {} # Reset on error
                        except Exception as e:
                             print(f"Warning: Error parsing hyperparameters line in {filepath}: {e}")
                             hyperparameters = {}


        except Exception as e:
             print(f"Error reading file {filepath}: {e}")
             return None, [], {} # Return empty on read error

        return status_line, progress_lines, hyperparameters

    def parse_progress_line(line):
        """Parses metrics from a progress line using regex."""
        metrics = {}
        timestamp = 'N/A'

        # Extract timestamp (part within the first pair of square brackets)
        timestamp_match = re.match(r'\[(.*?)\]', line)
        if timestamp_match:
             timestamp = timestamp_match.group(1)

        # Use regex to find Time, Energy, and Accuracy values robustly
        time_match = re.search(r'Time\s+(\d+\.?\d*)', line) # More flexible spacing
        energy_match = re.search(r'Energy\s*=\s*(\d+\.?\d*)', line) # Flexible spacing around =
        accuracy_match = re.search(r'Accuracy\s*=\s*(\d+\.?\d*)', line) # Flexible spacing

        try:
            if time_match:
                 metrics['Time'] = float(time_match.group(1))
            if energy_match:
                 metrics['Energy'] = float(energy_match.group(1))
            if accuracy_match:
                 metrics['Accuracy'] = float(accuracy_match.group(1))
        except (ValueError, TypeError) as e:
             print(f"Warning: Error parsing numeric value in progress line: '{line}' - {e}")
             # Keep already parsed metrics, or decide how to handle partial parse errors

        return timestamp, metrics

    def monitor(stop_event):
        """The core monitoring loop running in a thread."""
        last_update_times = {} # Track last update time per file to highlight changes

        while not stop_event.is_set():
            alive_reports_info = []
            current_files = set() # Track files seen in this cycle

            # Walk through the directory structure
            for root, dirs, files in os.walk(base_reports_folder):
                for filename in files:
                    if filename.endswith('.txt'):
                        filepath = os.path.join(root, filename)
                        current_files.add(filepath) # Mark file as seen

                        status_line, progress_lines, hyperparameters = read_report(filepath)

                        # Check if the report status is 'alive' (case-insensitive)
                        if status_line and 'alive' in status_line.lower():
                             # Get file modification time
                             try:
                                 mtime = os.path.getmtime(filepath)
                             except OSError:
                                 mtime = 0 # File might have been deleted between list and stat

                             alive_reports_info.append({
                                 'path': filepath,
                                 'filename': os.path.relpath(filepath, base_reports_folder), # Show relative path
                                 'status': status_line.split(':', 1)[1].strip(),
                                 'progress': progress_lines,
                                 'hyperparameters': hyperparameters,
                                 'mtime': mtime
                             })

            # Sort reports perhaps by modification time or name
            alive_reports_info.sort(key=lambda x: x['mtime'], reverse=True)

            # --- Display Logic ---
            os.system('cls' if os.name == 'nt' else 'clear') # Clear console for clean display
            print(f"--- Monitoring Alive Reports ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
            if not alive_reports_info:
                print("No alive reports found.")
            else:
                print(f"Found {len(alive_reports_info)} alive reports:")

            for report in alive_reports_info:
                filename = report['filename']
                hyperparams = report['hyperparameters']
                progress_lines = report['progress']
                mtime = report['mtime']

                # Check if file is newly updated since last check
                updated_symbol = "*" if mtime > last_update_times.get(report['path'], 0) else " "
                last_update_times[report['path']] = mtime # Update last seen time

                # Format hyperparameters nicely
                bs = hyperparams.get('batch_size', 'N/A')
                step = hyperparams.get('step_size', hyperparams.get('lr', 'N/A')) # Allow 'lr' as fallback
                k_val = hyperparams.get('K', 'N/A')
                core = hyperparams.get('cuda_core', 'N/A')
                dataset = hyperparams.get('dataset', 'N/A')

                # Display basic info
                print(f"\n{updated_symbol}Report: {colored(filename, 'cyan')} [Dataset: {dataset}, K: {k_val}, Step: {step}, BS: {bs}, Core: {core}]")

                # Display last progress line
                if progress_lines:
                    last_line = progress_lines[-1]
                    timestamp, metrics = parse_progress_line(last_line)
                    metrics_str = ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
                    if not metrics_str: metrics_str = "No metrics parsed"
                    print(f"  Last Update [{timestamp}]: {metrics_str}")
                else:
                    print(colored("  No progress lines found yet.", 'yellow'))

            # --- Cleanup ---
            # Remove files from last_update_times if they no longer exist or aren't alive
            stale_files = set(last_update_times.keys()) - current_files
            for stale_file in stale_files:
                 del last_update_times[stale_file]


            # Wait before next update cycle
            try:
                stop_event.wait(5) # Wait for 5 seconds or until stop event is set
            except KeyboardInterrupt: # Allow Ctrl+C to potentially stop if main thread handles it
                 break


    # --- Main part of monitor_reports ---
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor, args=(stop_event,), daemon=True) # Use daemon thread
    monitor_thread.start()

    print("Monitoring started. Press Enter or type 'x' + Enter to stop...")
    try:
        # Wait for user input to stop
        while True:
             user_input = input().strip().lower()
             if user_input == 'x' or user_input == '':
                 print("Stopping monitor...")
                 stop_event.set() # Signal the thread to stop
                 break
             else:
                  print("Unknown command. Press Enter or type 'x' + Enter to stop.")

    except KeyboardInterrupt:
         print("\nCtrl+C detected. Stopping monitor...")
         stop_event.set() # Signal the thread to stop on Ctrl+C


    monitor_thread.join(timeout=2) # Wait briefly for thread to finish cleanly
    if monitor_thread.is_alive():
         print("Monitor thread did not exit cleanly.")
    else:
         print("Monitoring stopped.")
    # Clear the "stopping" messages etc.
    os.system('cls' if os.name == 'nt' else 'clear')


def save_and_export_logs():
    # import shutil # Already imported at the top
    # from datetime import datetime # Already imported at the top
    # import zipfile # Already imported at the top

    exports_dir = 'exports'
    saved_logs_dir = 'saved_logs'
    figures_dir = 'figures'
    reports_dir = 'reports' # Include reports dir in export

    # Create the exports directory if it doesn't exist
    if not os.path.exists(exports_dir):
        print(f"Creating directory: {exports_dir}")
        os.makedirs(exports_dir)

    # Create a timestamp for the export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = os.path.join(exports_dir, f"export_{timestamp}.zip")
    readme_filename = f"README_{timestamp}.txt" # Use unique name for temp readme

    print(f"Starting export process to {zip_filename}...")

    try:
        # --- Generate README ---
        readme_path = readme_filename # Temporary path
        original_stdout = sys.stdout
        readme_content = ""
        print(f"Generating {readme_filename}...")
        try:
            with open(readme_path, 'w') as readme_file:
                sys.stdout = readme_file # Redirect stdout
                print(f"--- Experiment Summary ({timestamp}) ---")

                # Call option_2 to list hyperparameters and missing combinations
                print("\n--- Hyperparameter Check (from saved_logs) ---")
                option_2(run_missing_prompt=False) # Call option_2 without prompting to run

                # Add GPU core status
                print("\n--- GPU Core Status (from reports) ---")
                get_free_gpu_cores() # Function already prints the info

                print(f"\n--- End of Summary ---")

            # Read the content back if needed (optional)
            # with open(readme_path, 'r') as f:
            #     readme_content = f.read()

        except Exception as e:
            print(f"Error generating README content: {e}", file=original_stdout) # Print error to original stdout
            # Ensure stdout is restored even if README generation fails
        finally:
            sys.stdout = original_stdout # IMPORTANT: Restore stdout
            print(f"Finished generating {readme_filename}.")


        # --- Create ZIP file ---
        print("Creating ZIP archive...")
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf: # Use compression
             # Add README file to the ZIP file
            if os.path.exists(readme_path):
                zipf.write(readme_path, 'README.txt') # Store as README.txt in zip
                print(f"- Added {readme_path} as README.txt")
            else:
                 print(f"Warning: {readme_path} was not created, skipping.")

            # Function to add a directory recursively to the zip
            def add_dir_to_zip(dir_to_add, arc_dir_name):
                if os.path.exists(dir_to_add):
                    print(f"- Adding directory '{dir_to_add}' as '{arc_dir_name}'...")
                    for root, dirs, files in os.walk(dir_to_add):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Create the path as it should appear inside the zip archive
                            archive_path = os.path.join(arc_dir_name, os.path.relpath(file_path, dir_to_add))
                            zipf.write(file_path, archive_path)
                    print(f"- Finished adding '{dir_to_add}'.")
                else:
                     print(f"Warning: Directory '{dir_to_add}' not found, skipping.")

            # Add directories
            add_dir_to_zip(saved_logs_dir, 'saved_logs')
            add_dir_to_zip(figures_dir, 'figures')
            add_dir_to_zip(reports_dir, 'reports') # Add reports directory


    except Exception as e:
        print(f"Error during ZIP creation: {e}")
        # Optionally remove partially created zip file on error
        if os.path.exists(zip_filename):
            try:
                os.remove(zip_filename)
                print(f"Removed partially created {zip_filename}.")
            except OSError as oe:
                 print(f"Error removing partial zip file: {oe}")

    finally:
        # Clean up the temporary README file regardless of success/failure
        if os.path.exists(readme_path):
             try:
                 os.remove(readme_path)
                 print(f"Removed temporary file: {readme_path}")
             except OSError as oe:
                  print(f"Error removing temporary file {readme_path}: {oe}")


    if os.path.exists(zip_filename):
         print(colored(f"\nExport successful! Logs, figures, reports, and README have been saved to {zip_filename}", 'green'))
    else:
         print(colored("\nExport failed.", 'red'))


# Keep option_6 as is, it depends on saved_logs, not reports structure
def option_6():
    # Extract the K values, step sizes, and datasets from the saved logs directory
    K_values, step_sizes, datasets = extract_K_and_stepsizes('saved_logs')

    if not datasets:
        print("No datasets found in saved_logs. Cannot plot with variance.")
        return
    if not K_values or not step_sizes:
        print("No K values or step sizes found in saved_logs. Cannot plot with variance.")
        return


    print("\n--- Plotting Loss with Variance (from saved_logs) ---")
    # Plot the results with variance for each dataset
    for dataset in datasets:
        print(f"Plotting for dataset: {dataset}")
        plot_with_variance(K_values, step_sizes, dataset)
    print("Plotting with variance complete. Figures saved in the 'figures' directory.")


# Modify option_7 to call the new plot_reports
def option_7():
    print("\n--- Plotting Reports (Energy/Accuracy vs Time from reports) ---")
    # Plot the reports from the reports directory and its subdirectories
    plot_reports(base_reports_dir='reports')
    print("Plotting reports complete. Figures saved in the 'figures' directory.")
    print("Note: Each subdirectory in 'reports' should have its own combined plot PNG file.")


if __name__ == "__main__":
    # import sys # Already imported

    def display_menu():
        print(colored("\n--- Analysis Tool Menu ---", 'cyan', attrs=['bold']))
        print("1. Plot Loss vs Rounds (Best Step Sizes from saved_logs)")
        print("2. List Trained Hyperparameters & Check Missing (from saved_logs)")
        print("3. Monitor Alive Reports (Real-time)")
        print("4. Check Free/Used GPU Cores (from reports)")
        print("5. Export Logs, Figures, Reports & Summary")
        print("6. Plot Loss vs Rounds with Variance (All Step Sizes from saved_logs)")
        print("7. Plot Reports (Energy/Accuracy vs Time from reports subdirs)")
        print("8. Exit")
        print("-" * 26)


    def option_1():
        print("\n--- Plotting Loss vs Rounds (Best Step Sizes) ---")
        # Extract the K values, step sizes, and datasets from the saved logs directory
        K_values, step_sizes, datasets = extract_K_and_stepsizes('saved_logs')

        if not datasets:
            print(colored("No datasets found in saved_logs. Cannot plot results.", 'yellow'))
            return
        if not K_values or not step_sizes:
             print(colored("No K values or step sizes found in saved_logs. Cannot plot results.", 'yellow'))
             return


        # Ask for beta value
        beta = 0.9 # Default beta
        while True:
            beta_input = input(f"Enter beta for running average (0 to 1, default {beta}), or press Enter: ").strip()
            if not beta_input:
                 break # Keep default
            try:
                beta_float = float(beta_input)
                if 0.0 <= beta_float <= 1.0:
                    beta = beta_float
                    break
                else:
                    print(colored("Beta value must be between 0 and 1.", 'red'))
            except ValueError:
                print(colored("Invalid input. Please enter a numerical value.", 'red'))

        print(f"Using beta = {beta}")

        # Plot the results for each dataset
        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")
            # Plot comparison using best step sizes determined by getBestStepsizes
            plot_results(K_values, step_sizes, dataset, beta=beta)

            # Plot individual K values showing different step sizes
            print(f"Plotting individual K values for dataset: {dataset}")
            for K in K_values:
                plot_results_for_specific_K(K, step_sizes, dataset)

        print(colored("\nPlotting complete. Figures saved in the 'figures' directory.", 'green'))

    # Modified option_2 to accept a flag to skip the run prompt (used by export)
    def option_2(run_missing_prompt=True):
        print("\n--- Hyperparameter Check (saved_logs) ---")
        hyperparameters = defaultdict(lambda: {'found': set(), 'all_k': set(), 'all_steps': set()})
        saved_logs_dir = 'saved_logs'

        if not os.path.exists(saved_logs_dir):
            print(colored(f"'{saved_logs_dir}' directory not found.", 'yellow'))
            return

        # Use the extraction function
        K_values_all, step_sizes_all, datasets_all = extract_K_and_stepsizes(saved_logs_dir)

        # Populate found combinations per dataset
        for filename in os.listdir(saved_logs_dir):
             if filename.endswith('.json'):
                parts = filename.split('_')
                if len(parts) >= 3:
                     try:
                         lr_part = parts[0]
                         k_part = parts[1]
                         dataset_part = parts[2]

                         if lr_part.startswith("lr=") and k_part.startswith("K=") and dataset_part.startswith("dataset="):
                             step_size = lr_part.split('=')[1]
                             K = int(k_part.split('=')[1])
                             dataset = dataset_part.split('=')[1]
                             if dataset.endswith('.json'): dataset = dataset[:-5]

                             # Validate before adding
                             _ = float(step_size)

                             hyperparameters[dataset]['found'].add((K, step_size))
                             hyperparameters[dataset]['all_k'].add(K)
                             hyperparameters[dataset]['all_steps'].add(step_size)

                     except (IndexError, ValueError):
                         # Silently ignore malformed files here as extract_K_and_stepsizes handles reporting
                         continue


        print("\nListing trained hyperparameters by dataset:")
        if not hyperparameters:
            print(colored("No valid trained hyperparameters found in saved_logs.", 'yellow'))
            return


        for dataset, data in hyperparameters.items():
            print(colored(f"\nDataset: {dataset}", 'magenta'))
            if not data['found']:
                print("  No logs found for this dataset.")
                continue

            # Sort for consistent display
            dataset_K_values = sorted(list(data['all_k']))
            dataset_step_sizes = sorted(list(data['all_steps']), key=float)
            found_combinations = data['found']

            print(f"  K values found: {dataset_K_values}")
            print(f"  Step Sizes found: {dataset_step_sizes}")
            print(f"  Total combinations found: {len(found_combinations)}")


            missing_combinations = []
            # Check all permutations of found K and step sizes within this dataset
            for K in dataset_K_values:
                for step_size in dataset_step_sizes:
                    if (K, step_size) not in found_combinations:
                        missing_combinations.append((K, step_size))

            if missing_combinations:
                print(colored("\n  Missing combinations (within found K and step ranges):", 'yellow'))
                # Sort missing for clarity
                missing_combinations.sort(key=lambda x: (x[0], float(x[1])))
                for K, step_size in missing_combinations:
                    print(colored(f"    K={K}, Step Size={step_size}", 'red'))

                # --- Prompt to run missing (only if run_missing_prompt is True) ---
                if run_missing_prompt:
                    run_training_input = input(f"  Run training for these {len(missing_combinations)} missing combinations for '{dataset}'? (yes/no): ").strip().lower()
                    if run_training_input in ['yes', 'y']:
                        print("  Checking for free GPU cores...")
                        free_cores = get_free_gpu_cores() # Get currently free cores

                        if not free_cores:
                            print(colored("  No free GPU cores available. Cannot run training.", 'red'))
                        else:
                            print(f"  Available free GPU cores: {free_cores}")
                            processes = []
                            core_idx = 0
                            print("  Starting training processes...")
                            for i, (K, step_size) in enumerate(missing_combinations):
                                if core_idx < len(free_cores):
                                    gpu_core = free_cores[core_idx]
                                    core_idx += 1 # Use next free core

                                    # --- Adjust command as needed ---
                                    # Default rounds, adjust if necessary
                                    rounds = 20000
                                    python_exe = sys.executable # Use the same python interpreter running this script
                                    script_name = "dol1.py" # Assumed script name

                                    # Construct command more robustly
                                    command_parts = [
                                        python_exe, script_name,
                                        str(step_size), str(K), str(rounds),
                                        str(dataset), str(gpu_core)
                                    ]
                                    command_str = " ".join(command_parts) # Create string for printing/shell=True
                                    print(f"    Running: {command_str}")

                                    try:
                                        # Run in background using Popen with shell=False is safer if args are simple
                                        # If script needs shell features, keep shell=True but be mindful of args
                                        process = subprocess.Popen(command_parts, shell=False,
                                                                    stdout=subprocess.DEVNULL, # Suppress output from dol1.py
                                                                    stderr=subprocess.DEVNULL) # Suppress errors too (or redirect to log)
                                        processes.append(process)
                                        print(f"      -> Started on core {gpu_core} (PID: {process.pid})")
                                        time.sleep(0.5) # Small delay between starts
                                    except FileNotFoundError:
                                        print(colored(f"    Error: Script '{script_name}' not found.", 'red'))
                                        # Stop trying to run more if script is missing
                                        break
                                    except Exception as e:
                                        print(colored(f"    Error starting process for K={K}, step_size={step_size}: {e}", 'red'))
                                        # Optionally break here too, or just report and continue
                                else:
                                    print(colored(f"    Warning: No more free GPU cores. {len(missing_combinations) - i} combinations not started.", 'yellow'))
                                    break # Stop if out of cores

                            if processes:
                                print(colored(f"  {len(processes)} training processes started in the background.", 'green'))
                            else:
                                print(colored("  No training processes were started.", 'yellow'))
                    else:
                         print("  Skipping running missing combinations.")
            else:
                 print(colored("  No missing combinations found within the detected ranges.", 'green'))


    def option_3():
        monitor_reports()

    def option_4():
        print("\n--- Checking GPU Core Status ---")
        free_cores = get_free_gpu_cores()
        # The function already prints details


    def option_5():
        print("\n--- Exporting Data ---")
        save_and_export_logs()

    # option_6 remains the same as it works on saved_logs

    # option_7 is updated to call the modified plot_reports
    def option_7():
        print("\n--- Plotting Reports (Energy/Accuracy vs Time from reports) ---")
        # Plot the reports from the reports directory and its subdirectories
        plot_reports(base_reports_dir='reports') # Calls the updated function
        print("\nPlotting reports complete. Figures saved in the 'figures' directory.")
        print("Note: Each subdirectory in 'reports' should have its own combined plot PNG file.")

    def option_8():
        print("Exiting analysis tool.")
        sys.exit()


    options = {
        "1": option_1,
        "2": option_2,
        "3": option_3,
        "4": option_4,
        "5": option_5,
        "6": option_6, # Stays the same
        "7": option_7, # Calls updated plot_reports
        "8": option_8
    }

    while True:
        display_menu()
        choice = input("Enter your choice (1-8): ").strip()

        action = options.get(choice)
        if action:
            try:
                # Pass default arguments for functions that now expect them
                if choice == "2":
                    action() # Calls option_2() which defaults run_missing_prompt=True
                elif choice == "7":
                     action() # Calls option_7() which calls plot_reports()
                else:
                    action()
            except Exception as e:
                 print(colored(f"\nAn error occurred executing option {choice}:", 'red'))
                 print(colored(f"{type(e).__name__}: {e}", 'red'))
                 import traceback
                 print(colored("Traceback:", 'red'))
                 traceback.print_exc() # Print detailed traceback for debugging
                 print(colored("Please check the error and try again.", 'red'))

        elif choice: # If user entered something, but it's not a valid option
             print(colored("Invalid choice. Please select a valid option (1-8).", 'yellow'))
        # If choice is empty (user just pressed Enter), loop again showing the menu