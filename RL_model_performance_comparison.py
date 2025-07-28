# python ./RL_model_performance_comparison.py
# NOTE: It takes about ~4.5 hours for this code to run on an AMD Ryzen 5 55500U processor and 16GB RAM.

import subprocess
import pandas as pd
import statistics
import re
from collections import defaultdict

# GLOBAL VARIABLES
number_of_runs = 10  # Number of times the RL model scripts are called.
attackBlockSizes = ["12", "24", "36", "48"]  # Where 12 refers to half a day's worth of utility readings exposed to the attacker.

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


for attackBlockSize in attackBlockSizes:
    # ------------------------------------------------------------------------------------------------------------------
    # ELECTRICITY MODEL V1 CODE:
    electricity_v1_run_data = {hh: [] for hh in electricity_test_households}
    for curr_run in range(number_of_runs):
        print(f"\n\n\n--- Starting Electricity Model V1, Run {curr_run + 1} with attackBlockSize:{attackBlockSize} ---")

        try:
            subprocess.run(["python", "./RL_model_V1_ELECTRICITY/RL_model_V1_ELECTRICITY.py", attackBlockSize])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: RL_model_V1_ELECTRICITY.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found, skipping run {curr_run + 1}")
            continue

        try:
            electricity_v1_df = pd.read_csv("./RL_model_V1_ELECTRICITY/V1_testing_log_ELECTRICITY.csv")

        except FileNotFoundError:
            print("ERROR: Could not find the log file: './RL_model_V1_ELECTRICITY/V1_testing_log_ELECTRICITY.csv'")
            continue

        electricity_v1_df['HouseholdID'] = pd.Categorical(electricity_v1_df['HouseholdID'],
                                                          categories=electricity_test_households, ordered=True)
        electricity_v1_df_sorted = electricity_v1_df.sort_values('HouseholdID')

        for electricity_household_id in electricity_test_households:
            curr_electricity_household_data = electricity_v1_df_sorted[
                electricity_v1_df_sorted["HouseholdID"] == electricity_household_id]

            if not curr_electricity_household_data.empty:
                data_point = {
                    "Selected Encryption Ratio": curr_electricity_household_data["Selected Encryption Ratio"].values[0],
                    "Ciphertext Uniqueness": curr_electricity_household_data["Average ASR Mean"].values[0],
                    "Memory Consumption": curr_electricity_household_data["Average Memory MiB"].values[0],
                    "Summation Error": curr_electricity_household_data["Summation Error"].values[0],
                    "Deviation Error": curr_electricity_household_data["Deviation Error"].values[0],
                    "Encryption Time": curr_electricity_household_data["Encryption Time"].values[0]
                }
                electricity_v1_run_data[electricity_household_id].append(data_point)

    electricity_v1_per_household_analysis = {}
    for household_id, runs in electricity_v1_run_data.items():
        if runs:
            avg_encryption_ratio = statistics.mean([r["Selected Encryption Ratio"] for r in runs])
            avg_ciphertext_uniqueness = statistics.mean([r["Ciphertext Uniqueness"] for r in runs])
            avg_memory_consumption = statistics.mean([r["Memory Consumption"] for r in runs])
            avg_summation_error = statistics.mean([r["Summation Error"] for r in runs])
            avg_deviation_error = statistics.mean([r["Deviation Error"] for r in runs])
            avg_encryption_time = statistics.mean([r["Encryption Time"] for r in runs])

            std_encryption_ratio = statistics.stdev([r["Selected Encryption Ratio"] for r in runs])
            std_ciphertext_uniqueness = statistics.stdev([r["Ciphertext Uniqueness"] for r in runs])
            std_memory_consumption = statistics.stdev((r["Memory Consumption"] for r in runs))
            std_summation_error = statistics.stdev((r["Summation Error"] for r in runs))
            std_deviation_error = statistics.stdev(((r["Deviation Error"] for r in runs)))
            std_encryption_time = statistics.stdev([float(r["Encryption Time"]) for r in runs])

            electricity_v1_per_household_analysis[household_id] = {
                "Average Encryption Ratio": avg_encryption_ratio,
                "Standard Deviation Encryption Ratio": std_encryption_ratio,

                "Average Ciphertext Uniqueness": avg_ciphertext_uniqueness,
                "Standard Deviation Ciphertext Uniqueness": std_ciphertext_uniqueness,

                "Average Memory Consumption": avg_memory_consumption,
                "Standard Deviation Memory Consumption": std_memory_consumption,

                "Average Summation Error": avg_summation_error,
                "Standard Deviation Summation Error": std_summation_error,

                "Average Deviation Error": avg_deviation_error,
                "Standard Deviation Deviation Error": std_deviation_error,

                "Average Encryption Time": avg_encryption_time,
                "Standard Deviation Encryption Time": std_encryption_time,
            }

            # print(f"DEBUG: Analysis for household {household_id}: {electricity_v1_per_household_analysis[household_id]}")
    # ------------------------------------------------------------------------------------------------------------------
    # ELECTRICITY MODEL V1.5 CODE:
    electricity_v1_5_run_data = {hh: [] for hh in electricity_test_households}
    for curr_run in range(number_of_runs):
        print(f"\n\n\n--- Starting Electricity Model V1.5, Run {curr_run + 1} with attackBlockSize:{attackBlockSize} ---")

        try:
            subprocess.run(["python", "./RL_model_V1-5_ELECTRICITY/RL_model_V1-5_ELECTRICITY.py", attackBlockSize])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: RL_model_V1-5_ELECTRICITY.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found, skipping run {curr_run + 1}")
            continue

        try:
            electricity_v1_5_df = pd.read_csv("./RL_model_V1-5_ELECTRICITY/V1-5_testing_log_ELECTRICITY.csv")
        except FileNotFoundError:
            print("ERROR: Could not find the log file: './RL_model_V1-5_ELECTRICITY/V1-5_testing_log_ELECTRICITY.csv'")
            continue

        electricity_v1_5_df["HouseholdID"] = pd.Categorical(electricity_v1_5_df["HouseholdID"],
                                                          categories=electricity_test_households, ordered=True)
        electricity_v1_5_df_sorted = electricity_v1_5_df.sort_values("HouseholdID")

        for electricity_household_id in electricity_test_households:
            curr_electricity_household_data = electricity_v1_5_df_sorted[
                electricity_v1_5_df_sorted["HouseholdID"] == electricity_household_id]

            if not curr_electricity_household_data.empty:
                data_point = {
                    "Selected Encryption Ratio": curr_electricity_household_data["Selected Encryption Ratio"].values[0],
                    "Reidentification Rate": curr_electricity_household_data["Average Reidentification Mean"].values[0],
                    "Memory Consumption": curr_electricity_household_data["Average Memory MiB"].values[0],
                    "Summation Error": curr_electricity_household_data["Summation Error"].values[0],
                    "Deviation Error": curr_electricity_household_data["Deviation Error"].values[0],
                    "Encryption Time": curr_electricity_household_data["Encryption Time"].values[0]
                }
                electricity_v1_5_run_data[electricity_household_id].append(data_point)
                # print(f"DEBUG: Added data point for household {electricity_v1_5_run_data[electricity_household_id]}")

    electricity_v1_5_per_household_analysis = {}
    for household_id, runs in electricity_v1_5_run_data.items():
        if runs:
            avg_encryption_ratio = statistics.mean([r["Selected Encryption Ratio"] for r in runs])
            avg_reidentification_rate = statistics.mean([r["Reidentification Rate"] for r in runs])
            avg_memory_consumption = statistics.mean([r["Memory Consumption"] for r in runs])
            avg_summation_error = statistics.mean([r["Summation Error"] for r in runs])
            avg_deviation_error = statistics.mean([r["Deviation Error"] for r in runs])
            avg_encryption_time = statistics.mean([r["Encryption Time"] for r in runs])

            std_encryption_ratio = statistics.stdev([r["Selected Encryption Ratio"] for r in runs])
            std_reidentification_rate = statistics.stdev([r["Reidentification Rate"] for r in runs])
            std_memory_consumption = statistics.stdev((r["Memory Consumption"] for r in runs))
            std_summation_error = statistics.stdev((r["Summation Error"] for r in runs))
            std_deviation_error = statistics.stdev(((r["Deviation Error"] for r in runs)))
            std_encryption_time = statistics.stdev([float(r["Encryption Time"]) for r in runs])

            electricity_v1_5_per_household_analysis[household_id] = {
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
            # print(f"DEBUG: Analysis for household {household_id}: {electricity_v1_5_per_household_analysis[household_id]}")
    # ------------------------------------------------------------------------------------------------------------------
    # WATER MODEL V1 CODE:
    water_v1_run_data = {hh: [] for hh in water_test_households}
    # MODEL V1 WATER DATA GATHERING
    for curr_run in range(number_of_runs):
        print(f"\n\n\n--- Starting Water Model V1, Run {curr_run + 1} with attackBlockSize:{attackBlockSize} ---")

        try:
            subprocess.run(["python", "./RL_model_V1_WATER/RL_model_V1_WATER.py", attackBlockSize])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: RL_model_V1_WATER.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found, skipping run {curr_run + 1}")
            continue

        try:
            water_v1_df = pd.read_csv("./RL_model_V1_WATER/V1_testing_log_WATER.csv")
        except FileNotFoundError:
            print("ERROR: Could not find the log file: './RL_model_V1_WATER/V1_testing_log_WATER.csv'")
            continue

        water_v1_df["HouseholdID"] = pd.Categorical(water_v1_df["HouseholdID"],
                                                    categories=water_test_households, ordered=True)
        water_v1_df_sorted = water_v1_df.sort_values("HouseholdID")
        for water_household_id in water_test_households:
            curr_water_household_data = water_v1_df_sorted[
                water_v1_df_sorted["HouseholdID"] == water_household_id]

            if not curr_water_household_data.empty:
                data_point = {
                    "Selected Encryption Ratio": curr_water_household_data["Selected Encryption Ratio"].values[0],
                    "Ciphertext Uniqueness": curr_water_household_data["Average ASR Mean"].values[0],
                    "Memory Consumption": curr_water_household_data["Average Memory MiB"].values[0],
                    "Summation Error": curr_water_household_data["Summation Error"].values[0],
                    "Deviation Error": curr_water_household_data["Deviation Error"].values[0],
                    "Encryption Time": curr_water_household_data["Encryption Time"].values[0]
                }
                water_v1_run_data[water_household_id].append(data_point)

    water_v1_per_household_analysis = {}
    for household_id, runs in water_v1_run_data.items():
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

            water_v1_per_household_analysis[household_id] = {
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

            # print(f"DEBUG: Analysis for household {household_id}: {water_v1_per_household_analysis[household_id]}")
    # ------------------------------------------------------------------------------------------------------------------
    # WATER MODEL V1.5 CODE:
    water_v1_5_run_data = {hh: [] for hh in water_test_households}
    for curr_run in range(number_of_runs):
        print(f"\n\n\n--- Starting Water Model V1.5, Run {curr_run + 1} with attackBlockSize:{attackBlockSize} ---")

        try:
            subprocess.run(["python", "./RL_model_V1-5_WATER/RL_model_V1-5_WATER.py", attackBlockSize])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: RL_model_V1-5_WATER.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found, skipping run {curr_run + 1}")
            continue

        try:
            water_v1_5_df = pd.read_csv("./RL_model_V1-5_WATER/V1-5_testing_log_WATER.csv")
        except FileNotFoundError:
            print("ERROR: Could not find the log file: './RL_model_V1-5_WATER/V1-5_testing_log_WATER.csv'")
            continue

        water_v1_5_df["HouseholdID"] = pd.Categorical(water_v1_5_df["HouseholdID"],
                                                      categories=water_test_households, ordered=True)
        water_v1_5_df_sorted = water_v1_5_df.sort_values("HouseholdID")

        for water_household_id in water_test_households:
            curr_water_household_data = water_v1_5_df_sorted[
                water_v1_5_df_sorted["HouseholdID"] == water_household_id]

            if not curr_water_household_data.empty:
                data_point = {
                    "Selected Encryption Ratio": curr_water_household_data["Selected Encryption Ratio"].values[0],
                    "Reidentification Rate": curr_water_household_data["Average Reidentification Mean"].values[0],
                    "Memory Consumption": curr_water_household_data["Average Memory MiB"].values[0],
                    "Summation Error": curr_water_household_data["Summation Error"].values[0],
                    "Deviation Error": curr_water_household_data["Deviation Error"].values[0],
                    "Encryption Time": curr_water_household_data["Encryption Time"].values[0]
                }
                water_v1_5_run_data[water_household_id].append(data_point)
                # print(f"DEBUG: Added data point for household {water_v1_5_run_data[water_household_id]}")

    water_v1_5_per_household_analysis = {}
    for household_id, runs in water_v1_5_run_data.items():
        if runs:
            avg_encryption_ratio = statistics.mean([r["Selected Encryption Ratio"] for r in runs])
            avg_reidentification_rate = statistics.mean([r["Reidentification Rate"] for r in runs])
            avg_memory_consumption = statistics.mean([r["Memory Consumption"] for r in runs])
            avg_summation_error = statistics.mean([r["Summation Error"] for r in runs])
            avg_deviation_error = statistics.mean([r["Deviation Error"] for r in runs])
            avg_encryption_time = statistics.mean([r["Encryption Time"] for r in runs])

            std_encryption_ratio = statistics.stdev([r["Selected Encryption Ratio"] for r in runs])
            std_reidentification_rate = statistics.stdev([r["Reidentification Rate"] for r in runs])
            std_memory_consumption = statistics.stdev((r["Memory Consumption"] for r in runs))
            std_summation_error = statistics.stdev((r["Summation Error"] for r in runs))
            std_deviation_error = statistics.stdev(((r["Deviation Error"] for r in runs)))
            std_encryption_time = statistics.stdev([float(r["Encryption Time"]) for r in runs])

            water_v1_5_per_household_analysis[household_id] = {
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
    # ------------------------------------------------------------------------------------------------------------------
    # MODEL V1 AND V1.5 RESULT PRINTING CODE:
    basic_analyses = {
        "electricity_v1": electricity_v1_per_household_analysis,
        "electricity_v1_5": electricity_v1_5_per_household_analysis,

        "water_v1": water_v1_per_household_analysis,
        "water_v1_5": water_v1_5_per_household_analysis,
    }

    for label, data in basic_analyses.items():
        df = pd.DataFrame.from_dict(data, orient='index')
        output_filename = f"{label}_{attackBlockSize}_result.txt"
        try:
            with open(output_filename, "w") as f:
                f.write(df.to_string())
                print(f"DEBUG: Wrote {output_filename}")
        except FileNotFoundError:
            print(f"ERROR: Could not write to file: {output_filename}")
    # ------------------------------------------------------------------------------------------------------------------
    # MODEL ELECTRICITY V2 CODE:
    electricity_v2_all_runs_data = []
    for curr_run in range(number_of_runs):
        print(f"\n\n\n--- Starting Electricity Model V2, Run {curr_run + 1} with attackBlockSize:{attackBlockSize} ---")
        try:
            subprocess.run(["python", "./RL_model_V2_ELECTRICITY/RL_model_V2_ELECTRICITY.py", attackBlockSize])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: RL_model_V2_ELECTRICITY.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found, skipping run {curr_run + 1}")
            continue

        try:
            electricity_v2_df = pd.read_csv("./RL_model_V2_ELECTRICITY/V2_testing_log_combined.csv")
            electricity_v2_run_data = electricity_v2_df.iloc[0]
        except FileNotFoundError:
            print("ERROR: Could not find the log file: './RL_model_V2_ELECTRICITY/V2_testing_log_combined.csv'")
            continue

        ratios = parse_ratios_string(electricity_v2_run_data["Chosen Encryption Ratios"])
        sum_errors = parse_per_party_string(electricity_v2_run_data["Per-Party Summation Errors"])
        dev_errors = parse_per_party_string(electricity_v2_run_data["Per-Party Deviation Errors"])
        enc_times = parse_per_party_string(electricity_v2_run_data["Per-Party Encryption Times (NS)"])
        dec_times = parse_per_party_string(electricity_v2_run_data["Per-Party Decryption Times (NS)"])
        sum_ops_times = parse_per_party_string(electricity_v2_run_data["Per-Party Summation Operations Times (NS)"])
        dev_ops_times = parse_per_party_string(electricity_v2_run_data["Per-Party Deviation Operations Times (NS)"])

        reid_rate = electricity_v2_run_data["Reidentification Rate"]
        reid_duration = electricity_v2_run_data["Reidentification Duration (NS)"]
        mem_consumption = electricity_v2_run_data["Memory Consumption (MiB)"]

        for hh_id in electricity_test_households:
            if hh_id in ratios:
                electricity_v2_all_runs_data.append({
                    "household_id": hh_id,
                    "Encryption Ratio": ratios.get(hh_id),
                    "Reidentification Rate": reid_rate,
                    "Reidentification Duration (NS)": reid_duration,
                    "Memory Consumption (MiB)": mem_consumption,
                    "Summation Error": sum_errors.get(hh_id),
                    "Deviation Error": dev_errors.get(hh_id),
                    "Encryption Time (NS)": enc_times.get(hh_id),
                    "Decryption Time (NS)": dec_times.get(hh_id),
                    "Summation Operations Time (NS)": sum_ops_times.get(hh_id),
                    "Deviation Operations Time (NS)": dev_ops_times.get(hh_id),
                })

    electricity_v2_per_household_analysis = {}
    if electricity_v2_all_runs_data:
        results_df = pd.DataFrame(electricity_v2_all_runs_data)

        analysis = results_df.groupby('household_id').agg(['mean', 'std'])

        analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]

        column_rename_map = {
            'Encryption Ratio_mean': 'Average Encryption Ratio',
            'Reidentification Rate_mean': 'Average Reidentification Rate',
            'Memory Consumption (MiB)_mean': 'Average Memory Consumption',
            'Summation Error_mean': 'Average Summation Error',
            'Deviation Error_mean': 'Average Deviation Error',
            'Encryption Time (NS)_mean': 'Average Encryption Time',
            'Decryption Time (NS)_mean': 'Average Decryption Time',
            'Summation Operations Time (NS)_mean': 'Average Summation Operations Time',
            'Deviation Operations Time (NS)_mean': 'Average Deviation Operations Time',

            'Encryption Ratio_std': 'Standard Deviation Encryption Ratio',
            'Reidentification Rate_std': 'Standard Deviation Reidentification Rate',
            'Memory Consumption (MiB)_std': 'Standard Deviation Memory Consumption',
            'Summation Error_std': 'Standard Deviation Summation Error',
            'Deviation Error_std': 'Standard Deviation Deviation Error',
            'Encryption Time (NS)_std': 'Standard Deviation Encryption Time',
            'Decryption Time (NS)_std': 'Standard Deviation Decryption Time',
            'Summation Operations Time (NS)_std': 'Standard Deviation Summation Operations Time',
            'Deviation Operations Time (NS)_std': 'Standard Deviation Deviation Operations Time',
        }

        electricity_v2_per_household_analysis = analysis.rename(columns=column_rename_map)
    else:
        electricity_v2_per_household_analysis = pd.DataFrame()
    # ------------------------------------------------------------------------------------------------------------------
    # WATER MODEL V2 CODE:
    water_v2_all_runs_data = []
    for curr_run in range(number_of_runs):
        print(f"\n\n\n--- Starting Water Model V2, Run {curr_run + 1} with attackBlockSize:{attackBlockSize} ---")
        try:
            subprocess.run(["python", "./RL_model_V2_WATER/RL_model_V2_WATER.py", attackBlockSize])
        except subprocess.CalledProcessError as e:
            print(f"ERROR: RL_model_V1_WATER.py program failed with CalledProcessError: {e}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"WARNING: File not found, skipping run {curr_run + 1}")
            continue

        try:
            water_v2_df = pd.read_csv("./RL_model_V2_WATER/V2_testing_log_combined.csv")
            water_v2_run_data = water_v2_df.iloc[0]
        except FileNotFoundError:
            print("ERROR: Could not find the log file: './RL_model_V2_WATER/V2_testing_log_combined.csv'")
            continue

        ratios = parse_ratios_string(water_v2_run_data["Chosen Encryption Ratios"])
        sum_errors = parse_per_party_string(water_v2_run_data["Per-Party Summation Errors"])
        dev_errors = parse_per_party_string(water_v2_run_data["Per-Party Deviation Errors"])
        enc_times = parse_per_party_string(water_v2_run_data["Per-Party Encryption Times (NS)"])
        dec_times = parse_per_party_string(water_v2_run_data["Per-Party Decryption Times (NS)"])
        sum_ops_times = parse_per_party_string(water_v2_run_data["Per-Party Summation Operations Times (NS)"])
        dev_ops_times = parse_per_party_string(water_v2_run_data["Per-Party Deviation Operations Times (NS)"])

        reid_rate = water_v2_run_data["Reidentification Rate"]
        reid_duration = water_v2_run_data["Reidentification Duration (NS)"]
        mem_consumption = water_v2_run_data["Memory Consumption (MiB)"]

        for hh_id in water_test_households:
            if hh_id in ratios:
                water_v2_all_runs_data.append({
                    "household_id": hh_id,
                    "Encryption Ratio": ratios.get(hh_id),
                    "Reidentification Rate": reid_rate,
                    "Reidentification Duration (NS)": reid_duration,
                    "Memory Consumption (MiB)": mem_consumption,
                    "Summation Error": sum_errors.get(hh_id),
                    "Deviation Error": dev_errors.get(hh_id),
                    "Encryption Time (NS)": enc_times.get(hh_id),
                    "Decryption Time (NS)": dec_times.get(hh_id),
                    "Summation Operations Time (NS)": sum_ops_times.get(hh_id),
                    "Deviation Operations Time (NS)": dev_ops_times.get(hh_id),
                })

    water_v2_per_household_analysis = {}
    if water_v2_all_runs_data:
        results_df = pd.DataFrame(water_v2_all_runs_data)

        analysis = results_df.groupby('household_id').agg(['mean', 'std'])

        analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]

        column_rename_map = {
            'Encryption Ratio_mean': 'Average Encryption Ratio',
            'Reidentification Rate_mean': 'Average Reidentification Rate',
            'Memory Consumption (MiB)_mean': 'Average Memory Consumption',
            'Summation Error_mean': 'Average Summation Error',
            'Deviation Error_mean': 'Average Deviation Error',
            'Encryption Time (NS)_mean': 'Average Encryption Time',
            'Decryption Time (NS)_mean': 'Average Decryption Time',
            'Summation Operations Time (NS)_mean': 'Average Summation Operations Time',
            'Deviation Operations Time (NS)_mean': 'Average Deviation Operations Time',

            'Encryption Ratio_std': 'Standard Deviation Encryption Ratio',
            'Reidentification Rate_std': 'Standard Deviation Reidentification Rate',
            'Memory Consumption (MiB)_std': 'Standard Deviation Memory Consumption',
            'Summation Error_std': 'Standard Deviation Summation Error',
            'Deviation Error_std': 'Standard Deviation Deviation Error',
            'Encryption Time (NS)_std': 'Standard Deviation Encryption Time',
            'Decryption Time (NS)_std': 'Standard Deviation Decryption Time',
            'Summation Operations Time (NS)_std': 'Standard Deviation Summation Operations Time',
            'Deviation Operations Time (NS)_std': 'Standard Deviation Deviation Operations Time',
        }

        water_v2_per_household_analysis = analysis.rename(columns=column_rename_map)
    else:
        water_v2_per_household_analysis = pd.DataFrame()
    # ------------------------------------------------------------------------------------------------------------------
    # MODEL V2 RESULT PRINTING CODE:
    advanced_analyses = {
        "water_v2": water_v2_per_household_analysis,
        "electricity_v2": electricity_v2_per_household_analysis,
    }

    for label, data_df in advanced_analyses.items():
        if not data_df.empty:
            output_filename = f"{label}_{attackBlockSize}_result.txt"
            try:
                with open(output_filename, "w") as f:
                    f.write(data_df.to_string())
                    print(f"DEBUG: Wrote {output_filename}")
            except FileNotFoundError:
                print(f"ERROR: Could not write to file: {output_filename}")
    # ------------------------------------------------------------------------------------------------------------------
