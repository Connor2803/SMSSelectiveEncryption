# python ./RL_model_performance_comparison.py
import subprocess
import pandas as pd
import statistics
import re
import numpy as np

# ELECTRICITY MODEL ANALYSIS
electricity_test_households = ["MAC000248.csv", "MAC000252.csv", "MAC000253.csv", "MAC000255.csv", "MAC000256.csv",
                               "MAC000258.csv", "MAC000271.csv", "MAC000272.csv", "MAC000274.csv", "MAC004539.csv"]
electricity_v1_run_data = {hh: [] for hh in electricity_test_households}

# MODEL V1 ELECTRICITY DATA GATHERING
for curr_run in range(10):
    subprocess.run(["python", "./RL_model_V1_ELECTRICITY/RL_model_V1_ELECTRICITY.py"])

    electricity_v1_df = pd.read_csv("./RL_model_V1_ELECTRICITY/V1_testing_log_ELECTRICITY.csv")
    electricity_v1_df["HouseholdID"] = pd.Categorical(electricity_v1_df["HouseholdID"],
                                                      categories=electricity_test_households, ordered=True)
    electricity_v1_df_sorted = electricity_v1_df.sort_values("HouseholdID")

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

# MODEL V1 ELECTRICITY DATA TRANSFORMATION
electricity_v1_per_household_analysis = {}
for household_id, runs in electricity_v1_run_data.items():
    if runs:
        avg_encryption_ratio = statistics.mean([r["Selected Encryption Ratio"] for r in runs])
        avg_ciphertext_uniqueness = statistics.mean([r["Ciphertext Uniqueness"] for r in runs])
        avg_memory_consumption = statistics.mean([r["Memory Consumption"] for r in runs])
        avg_summation_error = statistics.mean([r["Summation Error"] for r in runs])
        avg_deviation_error = statistics.mean([r["Deviation Error"] for r in runs])
        avg_encryption_time = statistics.mean([r["Encryption Time"] for r in runs])

        electricity_v1_per_household_analysis[household_id] = {
            "Average Encryption Ratio": avg_encryption_ratio,
            "Average Ciphertext Uniqueness": avg_ciphertext_uniqueness,
            "Average Memory Consumption": avg_memory_consumption,
            "Average Summation Error": avg_summation_error,
            "Average Deviation Error": avg_deviation_error,
            "Average Encryption Time": avg_encryption_time
        }

# MODEL V2 ELECTRICITY DATA GATHERING
electricity_v2_run_data = {hh: [] for hh in electricity_test_households}
for curr_run in range(10):
    subprocess.run(["python", "./RL_model_V2_ELECTRICITY/RL_model_V2_ELECTRICITY.py"])

    electricity_v2_df = pd.read_csv("./RL_model_V2_ELECTRICITY/V2_testing_log_combined.csv")

    reid_vals = np.array_split(electricity_v2_df["Reidentification Rate"], len(electricity_test_households))
    mem_vals = np.array_split(electricity_v2_df["Memory Consumption (MiB)"], len(electricity_test_households))
    sum_err_vals = np.array_split(electricity_v2_df["Summation Error"], len(electricity_test_households))
    dev_err_vals = np.array_split(electricity_v2_df["Deviation Error"], len(electricity_test_households))
    enc_time_vals = np.array_split(electricity_v2_df["Encryption Time"], len(electricity_test_households))

    chosen_ratios_series = electricity_v2_df[
        "Chosen Encryption Ratios"] if "Chosen Encryption Ratios" in electricity_v2_df.columns else electricity_v2_df.iloc[
                                                                                                    :, 2]

    for ratio_list_str in chosen_ratios_series:
        ratio_list_str = ratio_list_str.strip("[]").replace(""", "").replace(""", "")
        ratio_entries = ratio_list_str.split(", ")

    for ratio_entry in ratio_entries:
        # Parse the string: e.g. "HMAC000248.csv-S1:0.80"
        match = re.match(r"H(.*?)-S\d+:(\d+\.\d+)", ratio_entry)
        if match:
            household_id = match.group(1)
            encryption_ratio = float(match.group(2))

            if household_id in electricity_v2_run_data:
                index = electricity_test_households.index(household_id)
                data_point = {
                    "Selected Encryption Ratio": encryption_ratio,
                    "Reidentification Mean": float(reid_vals[index].mean()),
                    "Memory Consumption": float(mem_vals[index].mean()),
                    "Summation Error": float(sum_err_vals[index].mean()),
                    "Deviation Error": float(dev_err_vals[index].mean()),
                    "Encryption Time": float(enc_time_vals[index].mean())
                }
                electricity_v2_run_data[household_id].append(data_point)

# MODEL V2 ELECTRICITY DATA TRANSFORMATION
electricity_v2_per_household_analysis = {}
for household_id, runs in electricity_v1_run_data.items():
    if runs:
        avg_encryption_ratio = statistics.mean([r["Selected Encryption Ratio"] for r in runs])
        avg_reidentification_mean = statistics.mean([r["Reidentification Mean"] for r in runs])
        avg_memory_consumption = statistics.mean([r["Memory Consumption"] for r in runs])
        avg_summation_error = statistics.mean([r["Summation Error"] for r in runs])
        avg_deviation_error = statistics.mean([r["Deviation Error"] for r in runs])
        avg_encryption_time = statistics.mean([r["Encryption Time"] for r in runs])

        electricity_v2_per_household_analysis[household_id] = {
            "Average Encryption Ratio": avg_encryption_ratio,
            "Average Reidentification Rate": avg_reidentification_mean,
            "Average Memory Consumption": avg_memory_consumption,
            "Average Summation Error": avg_summation_error,
            "Average Deviation Error": avg_deviation_error,
            "Average Encryption Time": avg_encryption_time
        }

# WATER MODEL ANALYSIS
water_test_households = ["e158012f-5c69-4a20-9a41-f7acde0e0ddd.csv", "e363b1f3-f503-48b4-b87c-98fe07632c02.csv",
                         "e41dddd2-87dd-4d4b-bdb6-9859c34768f1.csv", "e76658cf-88ea-4123-8715-0248909dd88b.csv",
                         "f12f91f7-81ca-4b7c-a5e8-3b81c4e5720b.csv",
                         "f45ff6bf-08c4-450d-bbf3-5597f66c68ba.csv""f5850315-552a-440f-9871-173212ad467f.csv",
                         "f5a28746-11f7-423f-9ae0-204a9b6d50ac.csv", "faea8eb7-c134-4c8b-99ac-c2c7ddd60d8b.csv",
                         "fc15a343-5276-4ce0-be8c-34087ae69070.csv"]

water_v1_run_data = {hh: [] for hh in water_test_households}
# MODEL V1 WATER DATA GATHERING
for curr_run in range(10):
    subprocess.run(["python", "./RL_model_V1_WATER/RL_model_V1_WATER.py"])

    water_v1_df = pd.read_csv("./RL_model_V1_WATER/V1_testing_log_WATER.csv")
    water_v1_df["HouseholdID"] = pd.Categorical(water_v1_run_data["HouseholdID"],
                                                      categories=water_test_households, ordered=True)
    water_v1_df_sorted = water_v1_df.sort_values("HouseholdID")

    for water_household_id in water_test_households:
        curr_water_household_data = water_v1_df_sorted[
            water_v1_df_sorted["HouseholdID"] == water_household_id]
        _
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

# MODEL V1 WATER DATA TRANSFORMATION
water_v1_per_household_analysis = {}
for household_id, runs in water_v1_per_household_analysis.items():
    if runs:
        avg_encryption_ratio = statistics.mean([r["Selected Encryption Ratio"] for r in runs])
        avg_ciphertext_uniqueness = statistics.mean([r["Ciphertext Uniqueness"] for r in runs])
        avg_memory_consumption = statistics.mean([r["Memory Consumption"] for r in runs])
        avg_summation_error = statistics.mean([r["Summation Error"] for r in runs])
        avg_deviation_error = statistics.mean([r["Deviation Error"] for r in runs])
        avg_encryption_time = statistics.mean([r["Encryption Time"] for r in runs])

        water_v1_per_household_analysis[household_id] = {
            "Average Encryption Ratio": avg_encryption_ratio,
            "Average Ciphertext Uniqueness": avg_ciphertext_uniqueness,
            "Average Memory Consumption": avg_memory_consumption,
            "Average Summation Error": avg_summation_error,
            "Average Deviation Error": avg_deviation_error,
            "Average Encryption Time": avg_encryption_time
        }

# MODEL V2 WATER DATA GATHERING
water_v2_run_data = {hh: [] for hh in water_test_households}
for curr_run in range(10):
    subprocess.run(["python", "./RL_model_V2_WATER/RL_model_V2_WATER.py"])

    water_v2_df = pd.read_csv("./RL_model_V2_WATER/V2_testing_log_combined.csv")

    reid_vals = np.array_split(water_v2_df["Reidentification Rate"], len(electricity_test_households))
    mem_vals = np.array_split(water_v2_df["Memory Consumption (MiB)"], len(electricity_test_households))
    sum_err_vals = np.array_split(water_v2_df["Summation Error"], len(electricity_test_households))
    dev_err_vals = np.array_split(water_v2_df["Deviation Error"], len(electricity_test_households))
    enc_time_vals = np.array_split(water_v2_df["Encryption Time"], len(electricity_test_households))

    chosen_ratios_series = water_v2_df[
        "Chosen Encryption Ratios"] if "Chosen Encryption Ratios" in water_v2_df.columns else water_v2_df.iloc[
                                                                                                    :, 2]

    for ratio_list_str in chosen_ratios_series:
        ratio_list_str = ratio_list_str.strip("[]").replace(""", "").replace(""", "")
        ratio_entries = ratio_list_str.split(", ")

    for ratio_entry in ratio_entries:
        # Parse the string: e.g. "HMAC000248.csv-S1:0.80"
        match = re.match(r"H(.*?)-S\d+:(\d+\.\d+)", ratio_entry)
        if match:
            household_id = match.group(1)
            encryption_ratio = float(match.group(2))

            if household_id in water_v2_run_data:
                index = water_test_households.index(household_id)
                data_point = {
                    "Selected Encryption Ratio": encryption_ratio,
                    "Reidentification Mean": float(reid_vals[index].mean()),
                    "Memory Consumption": float(mem_vals[index].mean()),
                    "Summation Error": float(sum_err_vals[index].mean()),
                    "Deviation Error": float(dev_err_vals[index].mean()),
                    "Encryption Time": float(enc_time_vals[index].mean())
                }
                water_v2_run_data[household_id].append(data_point)

# MODEL V2 ELECTRICITY DATA TRANSFORMATION
water_v2_per_household_analysis = {}
for household_id, runs in water_v1_run_data.items():
    if runs:
        avg_encryption_ratio = statistics.mean([r["Selected Encryption Ratio"] for r in runs])
        avg_reidentification_mean = statistics.mean([r["Reidentification Mean"] for r in runs])
        avg_memory_consumption = statistics.mean([r["Memory Consumption"] for r in runs])
        avg_summation_error = statistics.mean([r["Summation Error"] for r in runs])
        avg_deviation_error = statistics.mean([r["Deviation Error"] for r in runs])
        avg_encryption_time = statistics.mean([r["Encryption Time"] for r in runs])

        water_v2_per_household_analysis[household_id] = {
            "Average Encryption Ratio": avg_encryption_ratio,
            "Average Reidentification Rate": avg_reidentification_mean,
            "Average Memory Consumption": avg_memory_consumption,
            "Average Summation Error": avg_summation_error,
            "Average Deviation Error": avg_deviation_error,
            "Average Encryption Time": avg_encryption_time
        }


electricity_v1_result_df = pd.DataFrame.from_dict(electricity_v1_per_household_analysis, orient="index")
electricity_v2_result_df = pd.DataFrame.from_dict(electricity_v2_per_household_analysis, orient="index")
water_v1_result_df = pd.DataFrame.from_dict(water_v1_per_household_analysis, orient="index")
water_v2_result_df = pd.DataFrame.from_dict(water_v2_per_household_analysis, orient="index")

print(electricity_v1_result_df)
print(electricity_v2_result_df)
print(water_v1_result_df)
print(water_v2_result_df)