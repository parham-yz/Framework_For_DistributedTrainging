#!/usr/bin/env python3
"""
Tool: parse_reports.py
Purpose: Scan the `reports/` or a specified subdirectory for training runs.
Identifies unique (model, dataset) combinations, prompts user if multiple exist.
Extracts rounds reported by the early stopping mechanism for the chosen combination.
Prints a Markdown summary table (K vs step_size) of rounds to standard output,
with aligned columns and formatted K values.

Usage:
  python3 llms_tools/parse_reports.py [--experiment_dir <dir_name>]
"""
import os
import re
import json
import argparse
import sys

def get_reports_path(experiment_dir=None):
    """Return absolute path for reports dir, optionally within an experiment subdirectory."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    reports_dir = os.path.join(repo_root, 'reports')
    if experiment_dir:
        return os.path.join(reports_dir, experiment_dir)
    return reports_dir

def parse_report(path):
    """
    Parse one report file.
    Extracts model, dataset, K, step_size, and rounds from early stopping line.
    Returns (model, dataset, K, step_size, rounds) if successful, else None.
    """
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {path}: {e}", file=sys.stderr)
        return None # Return None immediately if file reading fails

    # --- Code below this line runs only if file reading was successful ---

    H = None
    # First pass: Find Hyperparameters
    for line in lines:
        if line.startswith('Hyperparameters:'):
            try:
                _, json_str = line.split(':', 1)
                H = json.loads(json_str.strip())
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {path}: {e}", file=sys.stderr)
                return None # Return None if hyperparameter JSON parsing fails
            break # Found hyperparameters, stop scanning for them

    if not H:
        # print(f"No Hyperparameters found in {path}", file=sys.stderr) # Optional debug
        return None # Return None if no hyperparameters were found

    model = H.get('model', 'N/A')
    dataset = H.get('dataset_name') or H.get('dataset', 'N/A')
    K = H.get('K', 'N/A')
    step_size = H.get('step_size', 'N/A')

    if K == 'N/A' or step_size == 'N/A':
         # print(f"Missing K or step_size in {path}", file=sys.stderr) # Optional debug
         return None # Return None if essential hyperparameters are missing

    # Second pass: Find early stopping trigger and the final rounds count
    rounds_pattern = re.compile(r'Training completed.* over (\d+) rounds')
    early_stopping_triggered = False
    rounds_at_stopping = None

    for line in lines:
        if 'Early stopping triggered' in line:
            early_stopping_triggered = True
            # Continue scanning, as the rounds count line appears after this.
            # We specifically want the rounds *when* early stopping triggered.

        m = rounds_pattern.search(line)
        if m:
            try:
                # We found a rounds count line. Capture it.
                current_rounds = int(m.group(1))
                # We want the rounds count that corresponds to the *termination* of the run.
                # In a log with early stopping, the *last* 'Training completed... over X rounds' line
                # should be the one *after* the early stopping trigger message.
                # We'll store the last one found, but only return it if early stopping was indeed triggered.
                rounds_at_stopping = current_rounds

            except ValueError:
                 print(f"Could not parse rounds value from line in {path}: {line.strip()}", file=sys.stderr)
                 # Continue scanning, maybe there's another valid rounds line


    # --- Code below this line runs after scanning all lines ---

    # Final check: Was early stopping triggered AND did we find a rounds count afterwards?
    # Note: The early_stopping_triggered flag must be True. The rounds_at_stopping must not be None.
    # If early stopping was triggered, we expect to find a final rounds count corresponding to the point of stopping.
    if early_stopping_triggered and rounds_at_stopping is not None:
        # Success! Found an early-stopped run with the final round count.
        return (model, dataset, K, step_size, rounds_at_stopping)
    else:
        # print(f"Report {path}: Early stopping not triggered or final rounds not found after trigger.", file=sys.stderr) # Optional debug
        # This report did not end due to early stopping, or the log format is unexpected.
        return None


def main():
    parser = argparse.ArgumentParser(description='Parse training reports for early stopping rounds.')
    parser.add_argument(
        '--experiment_dir',
        type=str,
        help='Optional subdirectory within reports/ to scan. If not provided, scans reports/.'
    )
    args = parser.parse_args()

    reports_dir = get_reports_path(args.experiment_dir)

    if not os.path.isdir(reports_dir):
        print(f'Error: Reports directory not found: {reports_dir}', file=sys.stderr)
        sys.exit(1)

    print(f"Scanning directory: {reports_dir} for reports...", file=sys.stderr)

    all_parsed_data = []
    found_combinations = set()

    for fn in os.listdir(reports_dir):
        if not fn.startswith('R') or not fn.endswith('.txt'):
            continue

        path = os.path.join(reports_dir, fn)
        parsed = parse_report(path)

        if parsed:
            model, dataset, K, step, rounds = parsed
            all_parsed_data.append(parsed)
            found_combinations.add((model, dataset))

    if not all_parsed_data:
        print("No relevant reports with early stopping rounds found.", file=sys.stderr)
        sys.exit(0)

    selected_model = None
    selected_dataset = None

    if len(found_combinations) > 1:
        print("\nMultiple (model, dataset) combinations found:", file=sys.stderr)
        combinations_list = sorted(list(found_combinations))
        for i, (model, dataset) in enumerate(combinations_list):
            print(f"{i+1}: Model='{model}', Dataset='{dataset}'", file=sys.stderr)

        while True:
            try:
                choice = input(f"Enter the number of the combination you want to process (1-{len(combinations_list)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(combinations_list):
                    selected_model, selected_dataset = combinations_list[choice_idx]
                    print(f"Selected: Model='{selected_model}', Dataset='{selected_dataset}'\n", file=sys.stderr)
                    break
                else:
                    print("Invalid choice. Please enter a number within the range.", file=sys.stderr)
            except ValueError:
                print("Invalid input. Please enter a number.", file=sys.stderr)
            except EOFError:
                 print("\nExiting due to input interruption.", file=sys.stderr)
                 sys.exit(1)

    elif len(found_combinations) == 1:
        selected_model, selected_dataset = list(found_combinations)[0]
        print(f"\nFound one combination: Model='{selected_model}', Dataset='{selected_dataset}'\n", file=sys.stderr)

    filtered_data = [
        (K, step, rounds) for model, dataset, K, step, rounds in all_parsed_data
        if model == selected_model and dataset == selected_dataset
    ]

    if not filtered_data:
        # This case is unlikely if found_combinations logic is correct, but good for safety.
        print(f"No reports found for selected combination (Model='{selected_model}', Dataset='{selected_dataset}') after filtering. This should not happen if selection worked correctly.", file=sys.stderr)
        sys.exit(1)

    # results[K_key] = dict mapping step_size_key -> rounds_at_stopping
    results = {}
    step_sizes = set()
    Ks = set()

    for K, step, rounds in filtered_data:
        K_key = str(K) if isinstance(K, (list, dict)) else K
        step_key = str(step) if isinstance(step, (list, dict)) else step

        results.setdefault(K_key, {})[step_key] = rounds
        step_sizes.add(step_key)
        Ks.add(K_key)

    summary = {}
    for K_key, rounds_by_step in results.items():
         if rounds_by_step:
            summary[K_key] = min(rounds_by_step.values())

    step_list = sorted(list(step_sizes))
    # K_list = sorted(list(Ks)) # Not strictly needed for data lookup after results dict is built

    # --- Start: Padding Logic for Aligned Columns ---

    # Determine maximum width for each column
    # Start with header widths
    header_cols = ['K', 'Min Rounds'] + [str(s) for s in step_list]
    column_widths = [len(col) for col in header_cols]

    # Add data widths, including the '.' for the K column
    # Iterate through sorted K_keys from results to ensure all K values in the data contribute to width
    for K_key in sorted(results.keys()):
        # Width for K column (plus '.')
        k_str = str(K_key) + '.' if isinstance(K_key, (int, float)) or (isinstance(K_key, str) and K_key.isdigit()) else str(K_key) # Add '.' if it looks like a number
        column_widths[0] = max(column_widths[0], len(k_str))

        # Width for Min Rounds column
        # Use summary.get(K_key) to avoid error if a K_key in results didn't end up in summary (e.g., due to all '-' values)
        min_rounds_val = summary.get(K_key)
        min_rounds_str = str(min_rounds_val) if min_rounds_val is not None else '-' # Represent None as '-' for width calc
        column_widths[1] = max(column_widths[1], len(min_rounds_str))


        # Widths for step size columns
        rounds_by_step = results.get(K_key, {})
        for i, s_key in enumerate(step_list):
            cell_content = str(rounds_by_step.get(s_key, '-'))
            column_widths[i + 2] = max(column_widths[i + 2], len(cell_content))


    # Add padding (e.g., 1 space on each side)
    padding = 1
    column_widths = [w + 2 * padding for w in column_widths] # Add padding to each side

    # --- End: Padding Logic for Aligned Columns ---

    # Print Markdown table to standard output
    print(f"# Model: {selected_model}, Dataset: {selected_dataset}")
    if args.experiment_dir:
        print(f"## Experiment: {args.experiment_dir}")
    print("## Rounds at Early Stopping (Lower is Better)")
    print("") # Blank line before table

    # Print header row with padding
    header_padded = [header_cols[i].center(column_widths[i]) for i in range(len(header_cols))]
    print('|' + '|'.join(header_padded) + '|')

    # Print separator line with padding
    sep_padded = ['-' * column_widths[i] for i in range(len(header_cols))]
    print('|' + '|'.join(sep_padded) + '|')

    # Print data rows with padding and formatted K
    for K_key in sorted(results.keys()): # Use sorted results.keys() to include all K values in table
        row_cells = []

        # K column with '.' and padding
        k_str = str(K_key) + '.' if isinstance(K_key, (int, float)) or (isinstance(K_key, str) and K_key.isdigit()) else str(K_key) # Add '.' if it looks like a number
        row_cells.append(k_str.ljust(column_widths[0] - padding).rjust(column_widths[0])) # Left align content, pad total width

        # Min Rounds column with padding
        min_rounds_val = summary.get(K_key)
        min_rounds_str = str(min_rounds_val) if min_rounds_val is not None else '-'
        row_cells.append(min_rounds_str.ljust(column_widths[1] - padding).rjust(column_widths[1]))

        # Step size columns with padding
        rounds_by_step = results.get(K_key, {})
        for i, s_key in enumerate(step_list):
            cell_content = str(rounds_by_step.get(s_key, '-'))
            # You can choose alignment here: .ljust(), .rjust(), .center()
            row_cells.append(cell_content.ljust(column_widths[i + 2] - padding).rjust(column_widths[i + 2])) # Or .center()

        print('|' + '|'.join(row_cells) + '|')

    print("\n'-' indicates no data available for this combination of K and step size.")
    print("\nNote: Table columns are padded for visual alignment in monospace fonts.")
    print(f"\nFinished processing reports for Model='{selected_model}', Dataset='{selected_dataset}'", file=sys.stderr)
    if args.experiment_dir:
         print(f" in directory '{args.experiment_dir}'.", file=sys.stderr)
    else:
         print(".", file=sys.stderr)


if __name__ == '__main__':
    main()