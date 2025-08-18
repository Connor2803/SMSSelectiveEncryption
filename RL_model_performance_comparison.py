# python ./RL_model_performance_comparison.py

import subprocess
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


# String parsing functions for RL V2 test .csvs
def parse_ratios_string(s: str) -> dict:
    """
    Parses the 'Chosen Encryption Ratios' string, e.g., "H...-S0:0.1; H...-S1:0.2".
    Returns a dictionary mapping household ID to its average ratio for the run.
    """
    ratios_by_hh = defaultdict(list)
    if not isinstance(s, str):
        return {}

    # Regex to find household ID and ratio value
    pattern = re.compile(r"(H(?:[ef][\w-]+|MAC\d+)\.csv)-S\d+:(\d+\.\d+)")
    matches = pattern.findall(s)

    for household_id, ratio_str in matches:
        key = household_id[1:] if household_id.startswith('H') else household_id
        ratios_by_hh[key].append(float(ratio_str))

    # Calculate the average ratio for each household
    avg_ratios = {hh: statistics.mean(r_list) for hh, r_list in ratios_by_hh.items()}
    return avg_ratios


def parse_per_party_string(s: str) -> dict:
    """
    Parses a generic 'Per-Party' metric string, e.g., "H...:123.45; H...:678.90".
    Returns a dictionary mapping household ID to its metric value.
    """
    metrics = {}
    if not isinstance(s, str):
        return {}

    entries = s.split('; ')
    for entry in entries:
        parts = entry.split(':', 1)
        if len(parts) == 2:
            # Key is like 'He158...csv', remove the leading 'H'
            key = parts[0][1:]
            try:
                value = float(parts[1])
                metrics[key] = value
            except (ValueError, IndexError):
                continue
    return metrics

def basic_printing_func(analysis_dict, model_complexity, dataset, attack_block_size):
    for data in analysis_dict.items():
        df = pd.DataFrame.from_dict(data, orient='index')
        output_filename = f"{dataset}_{model_complexity}_{attack_block_size}_collated_results.csv"
        try:
            with open(output_filename, "w") as f:
                df.to_csv(f)
                print(f"DEBUG: Wrote {output_filename}")
        except FileNotFoundError:
            print(f"ERROR: Could not write to file: {output_filename}")

def advanced_printing_func(analysis_dict, model_complexity, dataset, attack_block_size):
    for data in analysis_dict.items():
        output_filename = f"{dataset}_{model_complexity}_{attack_block_size}_collated_results.csv"
        try:
            with open(output_filename, "w") as f:
                data.to_csv(f)
                print(f"DEBUG: Wrote {output_filename}")
        except (FileNotFoundError, AttributeError) as e:
            print(f"ERROR: Could not write to file: {output_filename} due to {e}")

def test_analysis(all_runs_data):
    per_household_test_analysis = {}
    for household_id, runs in all_runs_data.items():
        if runs:
            avg_encryption_ratio = statistics.mean([r["Selected Encryption Ratio"] for r in runs])
            avg_reidentification_rate = statistics.mean([r["Ciphertext Uniqueness"] for r in runs])
            avg_memory_consumption = statistics.mean([r["Memory Consumption"] for r in runs])
            avg_summation_error = statistics.mean([r["Summation Error"] for r in runs])
            avg_deviation_error = statistics.mean([r["Deviation Error"] for r in runs])
            avg_encryption_time = statistics.mean([r["Encryption Time"] for r in runs])

            std_encryption_ratio = statistics.stdev([r["Selected Encryption Ratio"] for r in runs])
            std_reidentification_rate = statistics.stdev([r["Ciphertext Uniqueness"] for r in runs])
            std_memory_consumption = statistics.stdev((r["Memory Consumption"] for r in runs))
            std_summation_error = statistics.stdev((r["Summation Error"] for r in runs))
            std_deviation_error = statistics.stdev(((r["Deviation Error"] for r in runs)))
            std_encryption_time = statistics.stdev([float(r["Encryption Time"]) for r in runs])

            per_household_test_analysis[household_id] = {
                "Average Encryption Ratio": avg_encryption_ratio,
                "Standard Deviation Encryption Ratio": std_encryption_ratio,

                "Average Ciphertext Uniqueness": avg_reidentification_rate,
                "Standard Deviation Ciphertext Uniqueness": std_reidentification_rate,

                "Average Memory Consumption": avg_memory_consumption,
                "Standard Deviation Memory Consumption": std_memory_consumption,

                "Average Summation Error": avg_summation_error,
                "Standard Deviation Summation Error": std_summation_error,

                "Average Deviation Error": avg_deviation_error,
                "Standard Deviation Deviation Error": std_deviation_error,

                "Average Encryption Time": avg_encryption_time,
                "Standard Deviation Encryption Time": std_encryption_time,
            }

    return per_household_test_analysis

def per_household_analysis(all_runs_data):
    per_household_analysis = {}
    for household_id, runs in all_runs_data.items():
        if runs:
            avg_encryption_ratio = statistics.mean([r["Selected Encryption Ratio"] for r in runs])
            avg_reidentification_rate = statistics.mean([r["Ciphertext Uniqueness"] for r in runs])
            avg_memory_consumption = statistics.mean([r["Memory Consumption"] for r in runs])
            avg_summation_error = statistics.mean([r["Summation Error"] for r in runs])
            avg_deviation_error = statistics.mean([r["Deviation Error"] for r in runs])
            avg_encryption_time = statistics.mean([r["Encryption Time"] for r in runs])

            std_encryption_ratio = statistics.stdev([r["Selected Encryption Ratio"] for r in runs])
            std_reidentification_rate = statistics.stdev([r["Ciphertext Uniqueness"] for r in runs])
            std_memory_consumption = statistics.stdev((r["Memory Consumption"] for r in runs))
            std_summation_error = statistics.stdev((r["Summation Error"] for r in runs))
            std_deviation_error = statistics.stdev(((r["Deviation Error"] for r in runs)))
            std_encryption_time = statistics.stdev([float(r["Encryption Time"]) for r in runs])

            per_household_analysis[household_id] = {
                "Average Encryption Ratio": avg_encryption_ratio,
                "Standard Deviation Encryption Ratio": std_encryption_ratio,

                "Average Reidentification Rate": avg_reidentification_rate,
                "Standard Deviation Reidentification Rate": std_reidentification_rate,

                "Average Memory Consumption": avg_memory_consumption,
                "Standard Deviation Memory Consumption": std_memory_consumption,

                "Average Summation Error": avg_summation_error,
                "Standard Deviation Summation Error": std_summation_error,

                "Average Deviation Error": avg_deviation_error,
                "Standard Deviation Deviation Error": std_deviation_error,

                "Average Encryption Time": avg_encryption_time,
                "Standard Deviation Encryption Time": std_encryption_time,
            }

    return per_household_analysis

def per_block_analysis(all_runs_data):
    per_block_analysis = all_runs_data
    if per_block_analysis:
        results_df = pd.DataFrame(per_block_analysis)

        analysis = results_df.groupby('household_id').agg(['mean', 'std'])

        analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]

        column_rename_map = {
            'Encryption Ratio_mean': 'Average Encryption Ratio',
            'Reidentification Rate_mean': 'Average Reidentification Rate',
            'Advanced Reidentification Rate_mean': 'Average Advanced Reidentification Rate',
            'Memory Consumption (MiB)_mean': 'Average Memory Consumption',
            'Summation Error_mean': 'Average Summation Error',
            'Deviation Error_mean': 'Average Deviation Error',
            'Encryption Time (NS)_mean': 'Average Encryption Time',
            'Decryption Time (NS)_mean': 'Average Decryption Time',
            'Summation Operations Time (NS)_mean': 'Average Summation Operations Time',
            'Deviation Operations Time (NS)_mean': 'Average Deviation Operations Time',

            'Encryption Ratio_std': 'Standard Deviation Encryption Ratio',
            'Reidentification Rate_std': 'Standard Deviation Reidentification Rate',
            'Advanced Reidentification Rate_std': 'Standard Deviation Advanced Reidentification Rate',
            'Memory Consumption (MiB)_std': 'Standard Deviation Memory Consumption',
            'Summation Error_std': 'Standard Deviation Summation Error',
            'Deviation Error_std': 'Standard Deviation Deviation Error',
            'Encryption Time (NS)_std': 'Standard Deviation Encryption Time',
            'Decryption Time (NS)_std': 'Standard Deviation Decryption Time',
            'Summation Operations Time (NS)_std': 'Standard Deviation Summation Operations Time',
            'Deviation Operations Time (NS)_std': 'Standard Deviation Deviation Operations Time',
        }

        per_block_analysis = analysis.rename(columns=column_rename_map)
    else:
        per_block_analysis = pd.DataFrame()

    return per_block_analysis

def run_electricity_per_household_test(attack_block_size):
        try:
            subprocess.run(["python", "./test_model_electricity/test_model_electricity.py", attack_block_size])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: test_model_electricity.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found")
            return None

        try:
            electricity_per_household_test_df = pd.read_csv(f"./test_model_electricity/testing_log_{attack_block_size}")

        except FileNotFoundError:
            print(f"ERROR: Could not find the log file: './test_model_electricity/testing_log_{attack_block_size}.csv'")
            return None

        electricity_per_household_test_df['HouseholdID'] = pd.Categorical(electricity_per_household_test_df['HouseholdID'],
                                                          categories=electricity_test_households, ordered=True)
        electricity_per_household_test_df_sorted = electricity_per_household_test_df.sort_values('HouseholdID')

        for electricity_household_id in electricity_test_households:
            curr_electricity_household_data = electricity_per_household_test_df_sorted[
                electricity_per_household_test_df_sorted["HouseholdID"] == electricity_household_id]

            if not curr_electricity_household_data.empty:
                data_entry = {
                    "Selected Encryption Ratio": curr_electricity_household_data["Selected Encryption Ratio"].values[0],
                    "Ciphertext Uniqueness": curr_electricity_household_data["Average ASR Mean"].values[0],
                    "Memory Consumption": curr_electricity_household_data["Average Memory MiB"].values[0],
                    "Summation Error": curr_electricity_household_data["Summation Error"].values[0],
                    "Deviation Error": curr_electricity_household_data["Deviation Error"].values[0],
                    "Encryption Time": curr_electricity_household_data["Encryption Time"].values[0]
                }
                return data_entry

        return None

def run_water_per_household_test(attack_block_size):
        try:
            subprocess.run(["python", "./test_model_water/test_model_water.py", attack_block_size])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: test_model_water.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found.")
            return None

        try:
            water_per_household_test_df = pd.read_csv(f"./test_model_water/testing_log_{attack_block_size}.csv")
        except FileNotFoundError:
            print(f"ERROR: Could not find the log file: './test_model_water/testing_log_{attack_block_size}.csv'")
            return None

        water_per_household_test_df["HouseholdID"] = pd.Categorical(water_per_household_test_df["HouseholdID"],
                                                    categories=water_test_households, ordered=True)
        water_per_household_test_df_sorted = water_per_household_test_df.sort_values("HouseholdID")
        for water_household_id in water_test_households:
            curr_water_household_data = water_per_household_test_df_sorted[
                water_per_household_test_df_sorted["HouseholdID"] == water_household_id]

            if not curr_water_household_data.empty:
                data_entry = {
                    "Selected Encryption Ratio": curr_water_household_data["Selected Encryption Ratio"].values[0],
                    "Ciphertext Uniqueness": curr_water_household_data["Average ASR Mean"].values[0],
                    "Memory Consumption": curr_water_household_data["Average Memory MiB"].values[0],
                    "Summation Error": curr_water_household_data["Summation Error"].values[0],
                    "Deviation Error": curr_water_household_data["Deviation Error"].values[0],
                    "Encryption Time": curr_water_household_data["Encryption Time"].values[0]
                }
                return data_entry

        return None

def run_electricity_per_household(attack_block_size):
        try:
            subprocess.run(["python", "./ELECTRICITY_household_level_encryption_ratio_selector/household_level_encryption_ratio_selector.py", attack_block_size])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: ELECTRICITY_household_level_encryption_ratio_selector.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found")
            return None

        try:
            electricity_per_household_df = pd.read_csv(f"./ELECTRICITY_household_level_encryption_ratio_selector/testing_log_{attack_block_size}.csv")
        except FileNotFoundError:
            print(f"ERROR: Could not find the log file: './ELECTRICITY_household_level_encryption_ratio_selector/testing_log_{attack_block_size}.csv'")
            return None

        electricity_per_household_df["HouseholdID"] = pd.Categorical(electricity_per_household_df["HouseholdID"],
                                                          categories=electricity_test_households, ordered=True)
        electricity_per_household_df_sorted = electricity_per_household_df.sort_values("HouseholdID")

        for electricity_household_id in electricity_test_households:
            curr_electricity_household_data = electricity_per_household_df_sorted[
                electricity_per_household_df_sorted["HouseholdID"] == electricity_household_id]

            if not curr_electricity_household_data.empty:
                data_entry = {
                    "Selected Encryption Ratio": curr_electricity_household_data["Selected Encryption Ratio"].values[0],
                    "Reidentification Rate": curr_electricity_household_data["Average Reidentification Mean"].values[0],
                    "Memory Consumption": curr_electricity_household_data["Average Memory MiB"].values[0],
                    "Summation Error": curr_electricity_household_data["Summation Error"].values[0],
                    "Deviation Error": curr_electricity_household_data["Deviation Error"].values[0],
                    "Encryption Time": curr_electricity_household_data["Encryption Time"].values[0]
                }
                return data_entry

        return None

def run_water_per_household(attack_block_size):
        try:
            subprocess.run(["python", "./WATER_household_level_encryption_ratio_selector/household_level_encryption_ratio_selector.py", attack_block_size])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: WATER_household_level_encryption_ratio_selector.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found.")
            return None

        try:
            water_per_household_test_df = pd.read_csv(f"./WATER_household_level_encryption_ratio_selector/testing_log_{attack_block_size}.csv")
        except FileNotFoundError:
            print(f"ERROR: Could not find the log file: './WATER_household_level_encryption_ratio_selector/testing_log_{attack_block_size}.csv'")
            return None

        water_per_household_test_df["HouseholdID"] = pd.Categorical(water_per_household_test_df["HouseholdID"],
                                                      categories=water_test_households, ordered=True)
        water_per_household_test_df_sorted = water_per_household_test_df.sort_values("HouseholdID")

        for water_household_id in water_test_households:
            curr_water_household_data = water_per_household_test_df_sorted[
                water_per_household_test_df_sorted["HouseholdID"] == water_household_id]

            if not curr_water_household_data.empty:
                data_entry = {
                    "Selected Encryption Ratio": curr_water_household_data["Selected Encryption Ratio"].values[0],
                    "Reidentification Rate": curr_water_household_data["Average Reidentification Mean"].values[0],
                    "Memory Consumption": curr_water_household_data["Average Memory MiB"].values[0],
                    "Summation Error": curr_water_household_data["Summation Error"].values[0],
                    "Deviation Error": curr_water_household_data["Deviation Error"].values[0],
                    "Encryption Time": curr_water_household_data["Encryption Time"].values[0]
                }
                return data_entry

        return None

def run_electricity_per_block(attack_block_size):
        try:
            subprocess.run(["python", "./ELECTRICITY_block_level_encryption_ratio_selector/block_level_encryption_ratio_selector.py", attack_block_size])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: ELECTRICITY_block_level_encryption_ratio_selector.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found.")
            return None

        try:
            electricity_per_block_df = pd.read_csv(f"./ELECTRICITY_block_level_encryption_ratio_selector/testing_log_{attack_block_size}.csv")
            electricity_per_block_run_data = electricity_per_block_df.iloc[0]
        except FileNotFoundError:
            print(f"ERROR: Could not find the log file: './ELECTRICITY_block_level_encryption_ratio_selector/testing_log_{attack_block_size}.csv'")
            return None

        ratios = parse_ratios_string(electricity_per_block_run_data["Chosen Encryption Ratios"])
        sum_errors = parse_per_party_string(electricity_per_block_run_data["Per-Party Summation Errors"])
        dev_errors = parse_per_party_string(electricity_per_block_run_data["Per-Party Deviation Errors"])
        enc_times = parse_per_party_string(electricity_per_block_run_data["Per-Party Encryption Times (NS)"])
        dec_times = parse_per_party_string(electricity_per_block_run_data["Per-Party Decryption Times (NS)"])
        sum_ops_times = parse_per_party_string(electricity_per_block_run_data["Per-Party Summation Operations Times (NS)"])
        dev_ops_times = parse_per_party_string(electricity_per_block_run_data["Per-Party Deviation Operations Times (NS)"])

        reid_rate = electricity_per_block_run_data["Reidentification Rate"]
        adv_reid_rate = electricity_per_block_run_data["Advanced Reidentification Rate"]
        reid_duration = electricity_per_block_run_data["Reidentification Duration (NS)"]
        mem_consumption = electricity_per_block_run_data["Memory Consumption (MiB)"]

        for hh_id in electricity_test_households:
            if hh_id in ratios:
                data_entry = {
                    "household_id": hh_id,
                    "Encryption Ratio": ratios.get(hh_id),
                    "Reidentification Rate": reid_rate,
                    "Advanced Reidentification Rate": adv_reid_rate,
                    "Reidentification Duration (NS)": reid_duration,
                    "Memory Consumption (MiB)": mem_consumption,
                    "Summation Error": sum_errors.get(hh_id),
                    "Deviation Error": dev_errors.get(hh_id),
                    "Encryption Time (NS)": enc_times.get(hh_id),
                    "Decryption Time (NS)": dec_times.get(hh_id),
                    "Summation Operations Time (NS)": sum_ops_times.get(hh_id),
                    "Deviation Operations Time (NS)": dev_ops_times.get(hh_id),
                }
                return data_entry

        return None

def run_water_per_block(attack_block_size):
        try:
            subprocess.run(["python", "./WATER_block_level_encryption_ratio_selector/block_level_encryption_ratio_selector.py", attack_block_size])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: block_level_encryption_ratio_selector.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found.")
            return None

        try:
            water_per_block_df = pd.read_csv(f"./WATER_block_level_encryption_ratio_selector/testing_log_{attack_block_size}.csv")
            water_per_block_run_data = water_per_block_df.iloc[0]
        except FileNotFoundError:
            print(f"ERROR: Could not find the log file: './WATER_block_level_encryption_ratio_selector/testing_log_{attack_block_size}.csv'")
            return None

        ratios = parse_ratios_string(water_per_block_run_data["Chosen Encryption Ratios"])
        sum_errors = parse_per_party_string(water_per_block_run_data["Per-Party Summation Errors"])
        dev_errors = parse_per_party_string(water_per_block_run_data["Per-Party Deviation Errors"])
        enc_times = parse_per_party_string(water_per_block_run_data["Per-Party Encryption Times (NS)"])
        dec_times = parse_per_party_string(water_per_block_run_data["Per-Party Decryption Times (NS)"])
        sum_ops_times = parse_per_party_string(water_per_block_run_data["Per-Party Summation Operations Times (NS)"])
        dev_ops_times = parse_per_party_string(water_per_block_run_data["Per-Party Deviation Operations Times (NS)"])

        reid_rate = water_per_block_run_data["Reidentification Rate"]
        adv_reid_rate = water_per_block_run_data["Advanced Reidentification Rate"]
        reid_duration = water_per_block_run_data["Reidentification Duration (NS)"]
        mem_consumption = water_per_block_run_data["Memory Consumption (MiB)"]

        for hh_id in water_test_households:
            if hh_id in ratios:
                data_entry = {
                    "household_id": hh_id,
                    "Encryption Ratio": ratios.get(hh_id),
                    "Reidentification Rate": reid_rate,
                    "Advanced Reidentification Rate": adv_reid_rate,
                    "Reidentification Duration (NS)": reid_duration,
                    "Memory Consumption (MiB)": mem_consumption,
                    "Summation Error": sum_errors.get(hh_id),
                    "Deviation Error": dev_errors.get(hh_id),
                    "Encryption Time (NS)": enc_times.get(hh_id),
                    "Decryption Time (NS)": dec_times.get(hh_id),
                    "Summation Operations Time (NS)": sum_ops_times.get(hh_id),
                    "Deviation Operations Time (NS)": dev_ops_times.get(hh_id),
                }
                return data_entry

        return None

def run_electricity_per_block_with_policy(attack_block_size):
        try:
            subprocess.run(["python", "./ELECTRICITY_block_level_encryption_ratio_selector_with_policy/block_level_encryption_ratio_selector_with_policy.py", attack_block_size])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: ELECTRICITY_block_level_encryption_ratio_selector_with_policy.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found.")
            return None
        
        try:
            electricity_per_block_with_policy_df = pd.read_csv(
                f"ELECTRICITY_block_level_encryption_ratio_selector_with_policy/testing_log_{attack_block_size}.csv")
            electricity_per_block_with_policy_run_data = electricity_per_block_with_policy_df.iloc[0]
        except FileNotFoundError:
            print(f"ERROR: Could not find the log file: './ELECTRICITY_block_level_encryption_ratio_selector_with_policy/testing_log_{attack_block_size}.csv'")
            return None

        ratios = parse_ratios_string(electricity_per_block_with_policy_run_data["Chosen Encryption Ratios"])
        sum_errors = parse_per_party_string(electricity_per_block_with_policy_run_data["Per-Party Summation Errors"])
        dev_errors = parse_per_party_string(electricity_per_block_with_policy_run_data["Per-Party Deviation Errors"])
        enc_times = parse_per_party_string(electricity_per_block_with_policy_run_data["Per-Party Encryption Times (NS)"])
        dec_times = parse_per_party_string(electricity_per_block_with_policy_run_data["Per-Party Decryption Times (NS)"])
        sum_ops_times = parse_per_party_string(electricity_per_block_with_policy_run_data["Per-Party Summation Operations Times (NS)"])
        dev_ops_times = parse_per_party_string(electricity_per_block_with_policy_run_data["Per-Party Deviation Operations Times (NS)"])

        reid_rate = electricity_per_block_with_policy_run_data["Reidentification Rate"]
        adv_reid_rate = electricity_per_block_with_policy_run_data["Advanced Reidentification Rate"]
        reid_duration = electricity_per_block_with_policy_run_data["Reidentification Duration (NS)"]
        mem_consumption = electricity_per_block_with_policy_run_data["Memory Consumption (MiB)"]

        for hh_id in electricity_test_households:
            if hh_id in ratios:
                data_entry = {
                    "household_id": hh_id,
                    "Encryption Ratio": ratios.get(hh_id),
                    "Reidentification Rate": reid_rate,
                    "Advanced Reidentification Rate": adv_reid_rate,
                    "Reidentification Duration (NS)": reid_duration,
                    "Memory Consumption (MiB)": mem_consumption,
                    "Summation Error": sum_errors.get(hh_id),
                    "Deviation Error": dev_errors.get(hh_id),
                    "Encryption Time (NS)": enc_times.get(hh_id),
                    "Decryption Time (NS)": dec_times.get(hh_id),
                    "Summation Operations Time (NS)": sum_ops_times.get(hh_id),
                    "Deviation Operations Time (NS)": dev_ops_times.get(hh_id),
                }
                return data_entry

        return None

def run_water_per_block_with_policy(attack_block_size):
        try:
            subprocess.run(["python", "./WATER_block_level_encryption_ratio_selector_with_policy/block_level_encryption_ratio_selector_with_policy.py", attack_block_size])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: WATER_block_level_encryption_ratio_selector_with_policy.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found.")
            return None

        try:
            water_per_block_with_policy_df = pd.read_csv(f"./WATER_block_level_encryption_ratio_selector_with_policy/testing_log_{attack_block_size}.csv")
            water_per_block_with_policy_run_data = water_per_block_with_policy_df.iloc[0]
        except FileNotFoundError:
            print(f"ERROR: Could not find the log file: './WATER_block_level_encryption_ratio_selector_with_policy/testing_log_{attack_block_size}.csv'")
            return None

        ratios = parse_ratios_string(water_per_block_with_policy_run_data["Chosen Encryption Ratios"])
        sum_errors = parse_per_party_string(water_per_block_with_policy_run_data["Per-Party Summation Errors"])
        dev_errors = parse_per_party_string(water_per_block_with_policy_run_data["Per-Party Deviation Errors"])
        enc_times = parse_per_party_string(water_per_block_with_policy_run_data["Per-Party Encryption Times (NS)"])
        dec_times = parse_per_party_string(water_per_block_with_policy_run_data["Per-Party Decryption Times (NS)"])
        sum_ops_times = parse_per_party_string(water_per_block_with_policy_run_data["Per-Party Summation Operations Times (NS)"])
        dev_ops_times = parse_per_party_string(water_per_block_with_policy_run_data["Per-Party Deviation Operations Times (NS)"])

        reid_rate = water_per_block_with_policy_run_data["Reidentification Rate"]
        adv_reid_rate = water_per_block_with_policy_run_data["Advanced Reidentification Rate"]
        reid_duration = water_per_block_with_policy_run_data["Reidentification Duration (NS)"]
        mem_consumption = water_per_block_with_policy_run_data["Memory Consumption (MiB)"]

        for hh_id in water_test_households:
            if hh_id in ratios:
                data_entry = {
                    "household_id": hh_id,
                    "Encryption Ratio": ratios.get(hh_id),
                    "Reidentification Rate": reid_rate,
                    "Advanced Reidentification Rate": adv_reid_rate,
                    "Reidentification Duration (NS)": reid_duration,
                    "Memory Consumption (MiB)": mem_consumption,
                    "Summation Error": sum_errors.get(hh_id),
                    "Deviation Error": dev_errors.get(hh_id),
                    "Encryption Time (NS)": enc_times.get(hh_id),
                    "Decryption Time (NS)": dec_times.get(hh_id),
                    "Summation Operations Time (NS)": sum_ops_times.get(hh_id),
                    "Deviation Operations Time (NS)": dev_ops_times.get(hh_id),
                }
                return data_entry

        return None


def main():
    attack_block_sizes = ["3", "6", "9", "12", "24", "36", "48"]  # Where 12 refers to half a day's worth of utility readings exposed to the attacker
    number_of_runs=1

    for attack_block_size in attack_block_sizes:
        # all_electricity_per_household_test_run_data = {hh: [] for hh in electricity_test_households}
        # for i in range(number_of_runs):
        #     run_data = run_electricity_per_household_test(attack_block_size=attack_block_size)
        #     all_electricity_per_household_test_run_data.update(run_data)
        #
        # electricity_per_household_test_analysis = test_analysis(all_electricity_per_household_test_run_data)
        # basic_printing_func(analysis_dict=electricity_per_household_test_analysis, model_complexity="per_household_test", dataset="ELECTRICITY", attack_block_size=attack_block_size)

        # all_water_per_household_test_run_data = {hh: [] for hh in water_test_households}
        # for i in range(number_of_runs):
        #     run_data = run_water_per_household_test(attack_block_size=attack_block_size)
        #     all_water_per_household_test_run_data.update(run_data)
        #
        # water_per_household_test_analysis = test_analysis(all_water_per_household_test_run_data)
        # basic_printing_func(analysis_dict=water_per_household_test_analysis, model_complexity="per_household_test", dataset="WATER", attack_block_size=attack_block_size)

        # all_electricity_per_household_run_data = {hh: [] for hh in electricity_test_households}
        # for i in range(number_of_runs):
        #     run_data = run_electricity_per_household(attack_block_size=attack_block_size)
        #     all_electricity_per_household_run_data.update(run_data)
        #
        # electricity_per_household_analysis = test_analysis(all_electricity_per_household_run_data)
        # basic_printing_func(analysis_dict=electricity_per_household_analysis, model_complexity="per_household", dataset="ELECTRICITY", attack_block_size=attack_block_size)

        # all_water_per_household_run_data = {hh: [] for hh in water_test_households}
        # for i in range(number_of_runs):
        #     run_data = run_water_per_household(attack_block_size=attack_block_size)
        #     all_water_per_household_run_data.update(run_data)
        #
        # water_per_household_analysis = test_analysis(all_water_per_household_run_data)
        # basic_printing_func(analysis_dict=water_per_household_analysis, model_complexity="per_household", dataset="WATER", attack_block_size=attack_block_size)

        all_electricity_per_block_run_data = {hh: [] for hh in electricity_test_households}
        for i in range(number_of_runs):
            run_data = run_electricity_per_block(attack_block_size=attack_block_size)
            all_electricity_per_block_run_data.update(run_data)

        electricity_per_block_analysis = per_block_analysis(all_runs_data=all_electricity_per_block_run_data)
        advanced_printing_func(analysis_dict=electricity_per_block_analysis, model_complexity="per_block",
                            dataset="ELECTRICITY", attack_block_size=attack_block_size)

        # all_water_per_block_run_data = {hh: [] for hh in water_test_households}
        # for i in range(number_of_runs):
        #     run_data = run_water_per_block(attack_block_size=attack_block_size)
        #     all_water_per_block_run_data.update(run_data)
        #
        # water_per_block_analysis = per_block_analysis(all_runs_data=all_water_per_block_run_data)
        # advanced_printing_func(analysis_dict=water_per_block_analysis, model_complexity="per_block",
        #                     dataset="WATER", attack_block_size=attack_block_size)

        # all_electricity_per_block_with_policy_run_data = {hh: [] for hh in electricity_test_households}
        # for i in range(number_of_runs):
        #     run_data = run_electricity_per_block_with_policy(attack_block_size=attack_block_size)
        #     all_electricity_per_block_with_policy_run_data.update(run_data)
        #
        # electricity_per_block_with_policy_analysis = per_block_analysis(all_runs_data=all_electricity_per_block_with_policy_run_data)
        # advanced_printing_func(analysis_dict=electricity_per_block_with_policy_analysis, model_complexity="per_block_with_policy",
        #                     dataset="ELECTRICITY", attack_block_size=attack_block_size)


        # all_water_per_block_with_policy_run_data = {hh: [] for hh in water_test_households}
        # for i in range(number_of_runs):
        #     run_data = run_water_per_block_with_policy(attack_block_size=attack_block_size)
        #     all_water_per_block_with_policy_run_data.update(run_data)
        #
        # water_per_block_with_policy_analysis = per_block_analysis(all_runs_data=all_water_per_block_with_policy_run_data)
        # advanced_printing_func(analysis_dict=water_per_block_with_policy_analysis, model_complexity="per_block_with_policy",
        #                     dataset="WATER", attack_block_size=attack_block_size)


if __name__ == "__main__":
    main()