# python ./model_performance_comparator.py <leaked_plaintext_size> <number_of_runs> <policy_penalty>

import subprocess
import sys

import pandas as pd
import statistics
import re
from collections import defaultdict

# The fixed 10 test households that will be compared across all RL models.
electricity_test_households = ["MAC000248.csv",
                               "MAC000252.csv",
                               "MAC000253.csv",
                               "MAC000255.csv",
                               "MAC000256.csv",
                               "MAC000258.csv",
                               "MAC000271.csv",
                               "MAC000272.csv",
                               "MAC000274.csv",
                               "MAC004539.csv"]

water_test_households = ["e158012f-5c69-4a20-9a41-f7acde0e0ddd.csv",
                         "e363b1f3-f503-48b4-b87c-98fe07632c02.csv",
                         "e41dddd2-87dd-4d4b-bdb6-9859c34768f1.csv",
                         "e76658cf-88ea-4123-8715-0248909dd88b.csv",
                         "f12f91f7-81ca-4b7c-a5e8-3b81c4e5720b.csv",
                         "f45ff6bf-08c4-450d-bbf3-5597f66c68ba.csv",
                         "f5850315-552a-440f-9871-173212ad467f.csv",
                         "f5a28746-11f7-423f-9ae0-204a9b6d50ac.csv",
                         "faea8eb7-c134-4c8b-99ac-c2c7ddd60d8b.csv",
                         ]

def parse_ratios_string(s: str) -> dict:
    ratios_by_hh = defaultdict(list)
    if not isinstance(s, str): return {}
    pattern = re.compile(r"(H(?:[ef][\w-]+|MAC\d+)\.csv)-S\d+:(\d+\.\d+)")
    matches = pattern.findall(s)
    for household_id, ratio_str in matches:
        key = household_id[1:] if household_id.startswith('H') else household_id
        ratios_by_hh[key].append(float(ratio_str))
    return {hh: statistics.mean(r_list) for hh, r_list in ratios_by_hh.items()}

def parse_per_party_string(s: str) -> dict:
    metrics = {}
    if not isinstance(s, str): return {}
    for entry in s.split('; '):
        parts = entry.split(':', 1)
        if len(parts) == 2:
            key = parts[0][1:] if parts[0].startswith('H') else parts[0]
            try:
                metrics[key] = float(parts[1])
            except (ValueError, IndexError):
                continue
    return metrics

def parse_block_level_log(log_df: pd.DataFrame, test_households: list) -> list:
    if log_df.empty:
        return []

    run_data = log_df.iloc[0]
    results = []

    ratios = parse_ratios_string(run_data.get("Chosen Encryption Ratios", ""))
    sum_errors = parse_per_party_string(run_data.get("Per-Party Summation Errors", ""))
    dev_errors = parse_per_party_string(run_data.get("Per-Party Deviation Errors", ""))
    enc_times = parse_per_party_string(run_data.get("Per-Party Encryption Times (NS)", ""))
    dec_times = parse_per_party_string(run_data.get("Per-Party Decryption Times (NS)", ""))
    sum_ops_times = parse_per_party_string(run_data.get("Per-Party Summation Operations Times (NS)", ""))
    dev_ops_times = parse_per_party_string(run_data.get("Per-Party Deviation Operations Times (NS)", ""))

    reid_rate = run_data.get("Reidentification Rate")
    adv_reid_rate = run_data.get("Advanced Reidentification Rate")
    mem_consumption = run_data.get("Memory Consumption (MiB)")

    # Create a dictionary for each household
    for hh_id in test_households:
        if hh_id in ratios:
            results.append({
                "household_id": hh_id,
                "Encryption Ratio": ratios.get(hh_id),
                "Reidentification Rate": reid_rate,
                "Advanced Reidentification Rate": adv_reid_rate,
                "Memory Consumption (MiB)": mem_consumption,
                "Summation Error": sum_errors.get(hh_id),
                "Deviation Error": dev_errors.get(hh_id),
                "Encryption Time (NS)": enc_times.get(hh_id),
                "Decryption Time (NS)": dec_times.get(hh_id),
                "Summation Operations Time (NS)": sum_ops_times.get(hh_id),
                "Deviation Operations Time (NS)": dev_ops_times.get(hh_id),
            })
    return results


def run_single_test_and_get_results(leaked_plaintext_size: str, folder_name: str, model_name: str,
                                    dataset: str, policy_penalty: str = None) -> list:
    """Executes one test run and calls the correct parser based on the model type."""
    print(f"Executing test run for {folder_name} (size: {leaked_plaintext_size}, policy: {policy_penalty or 'N/A'})...")

    command = ["python", f"./{folder_name}/{model_name}.py", leaked_plaintext_size]
    if policy_penalty:
        command.append(policy_penalty)
    command.append("testing")

    log_filename_suffix = f"{leaked_plaintext_size}"
    if policy_penalty:
        log_filename_suffix += f"_{policy_penalty}"
    log_file = f"./{folder_name}/testing_log_{log_filename_suffix}.csv"

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Testing script failed.\n{e.stderr}")
        return []
    except subprocess.TimeoutExpired:
        print("ERROR: Testing script timed out after 30 minutes.")
        return []

    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"ERROR: Could not find log file: '{log_file}'")
        return []

    test_households = electricity_test_households if dataset == "ELECTRICITY" else water_test_households

    if "household_level" in model_name:
        results = []
        for household_id in test_households:
            row = df[df["HouseholdID"] == household_id]
            if not row.empty:
                data = row.iloc[0]
                results.append({
                    "household_id": household_id,
                    "Encryption Ratio": data.get("Selected Encryption Ratio"),
                    "Reidentification Rate": data.get("Average Reidentification Mean"),
                    "Memory Consumption (MiB)": data.get("Average Memory MiB"),
                    "Summation Error": data.get("Summation Error"),
                    "Deviation Error": data.get("Deviation Error"),
                    "Encryption Time (NS)": data.get("Encryption Time"),
                })
        return results
    else:
        return parse_block_level_log(df, test_households)


def run_analysis(all_runs_data: list):
    """Calculates mean and std dev from a list of run dictionaries."""
    if not all_runs_data:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_runs_data)
    analysis = results_df.groupby('household_id').agg(['mean', 'std']).fillna(0)
    analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]

    return analysis

# def write_results_to_csv(analysis_dict, model_complexity, dataset, leaked_plaintext_size, policy_penalty: str = None):
#     """Writes the final analysis dictionary to a CSV file."""
#     df = pd.DataFrame.from_dict(analysis_dict, orient='index')
#     if policy_penalty:
#         output_filename = f"{dataset}_{model_complexity}_{leaked_plaintext_size}_{policy_penalty}_collated_results.csv"
#     else:
#         output_filename = f"{dataset}_{model_complexity}_{leaked_plaintext_size}_collated_results.csv"
#     try:
#         df.to_csv(output_filename)
#         print(f"Successfully wrote aggregated results to {output_filename}")
#     except Exception as e:
#         print(f"ERROR: Could not write to file: {output_filename}. Reason: {e}")

def write_results_to_csv(analysis_df: pd.DataFrame, model_complexity: str, dataset: str, leaked_plaintext_size: str,
                         policy_penalty: str = None):
    """Writes the final analysis DataFrame to a CSV file."""
    filename_suffix = leaked_plaintext_size
    if policy_penalty:
        filename_suffix += f"_{policy_penalty}"

    model_name_from_folder = model_complexity.split('/')[-1]
    output_filename = f"{dataset}_{model_name_from_folder}_{filename_suffix}_collated_results.csv"

    try:
        analysis_df.to_csv(output_filename)
        print(f"\n Successfully wrote aggregated results to {output_filename}")
    except Exception as e:
        print(f"ERROR: Could not write to file: {output_filename}. Reason: {e}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python model_performance_comparator.py <leaked_plaintext_size> <number_of_runs> [policy_penalty]")
        leaked_plaintext_size = "12"
        number_of_runs = 4
        policy_penalty = "700"
    else:
        leaked_plaintext_size = sys.argv[1]
        number_of_runs = int(sys.argv[2])
        policy_penalty = sys.argv[3] if len(sys.argv) > 3 else None  # Policy penalty is optional

    folder_names = [
        # "ELECTRICITY_household_level_encryption_ratio_selector",
        # "WATER_household_level_encryption_ratio_selector",
        # "ELECTRICITY_block_level_encryption_ratio_selector",
        # "WATER_block_level_encryption_ratio_selector",
        "ELECTRICITY_block_level_encryption_ratio_selector_with_policy",
        # "WATER_block_level_encryption_ratio_selector_with_policy"
    ]

    print(f"Starting comparison of {number_of_runs} test runs for leaked plaintext size of {leaked_plaintext_size}\n")

    for folder_name in folder_names:
        model_name = "household_level_encryption_ratio_selector" if "household_level" in folder_name else \
            "block_level_encryption_ratio_selector_with_policy" if "with_policy" in folder_name else \
                "block_level_encryption_ratio_selector"
        dataset = folder_name.split("_")[0]

        all_runs_data = []

        for folder_name in folder_names:
            current_policy_penalty = None
            if "with_policy" in folder_name:
                if not policy_penalty:
                    print(f"WARNING: Skipping {folder_name} because policy_penalty is required but not provided.")
                    continue
                current_policy_penalty = policy_penalty

            model_name = "household_level_encryption_ratio_selector" if "household_level" in folder_name else \
                "block_level_encryption_ratio_selector_with_policy" if "with_policy" in folder_name else \
                    "block_level_encryption_ratio_selector"
            dataset = folder_name.split("_")[0]
            all_runs_data = []

            print(f"\n--- Processing Model: {folder_name} ---")
            for i in range(number_of_runs):
                single_run_results = run_single_test_and_get_results(
                    leaked_plaintext_size, folder_name, model_name, dataset, current_policy_penalty
                )
                if single_run_results:
                    all_runs_data.extend(single_run_results)
                else:
                    print(f"  - Run {i + 1} failed. Skipping.")

            final_analysis = run_analysis(all_runs_data)

            write_results_to_csv(final_analysis, model_name, dataset, leaked_plaintext_size, current_policy_penalty)
            print("-" * 50)


if __name__ == "__main__":
    main()