# python ./WATER_block_level_encryption_ratio_selector/block_level_encryption_ratio_selector.py

import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter

import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.dqn.policies import MultiInputPolicy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXECUTABLE_NAME = "generate_block_level_metrics"
if sys.platform == "win32":
    EXECUTABLE_NAME += ".exe"
GO_SOURCE_PATH = os.path.join(SCRIPT_DIR, "generate_block_level_metrics.go")
GO_EXECUTABLE_PATH = os.path.join(SCRIPT_DIR, EXECUTABLE_NAME)
print(f"\nGo executable path: {GO_EXECUTABLE_PATH}")
print(f"\nGo source path: {GO_SOURCE_PATH}")

currentLeakedPlaintextSize = "12"


class Welford:
    def __init__(self):
        # Where, M is the mean, S is the sum of square differences, V is the variance,
        # Z is the standard deviation, and K is the current number of data points.
        self.reidentification_rate_M = 0
        self.advanced_reidentification_rate_M = 0
        self.reidentification_duration_M = 0
        self.memory_consumption_M = 0
        self.summation_error_M = 0
        self.deviation_error_M = 0
        self.encryption_time_M = 0
        self.decryption_time_M = 0
        self.summation_operations_time_M = 0
        self.deviation_operations_time_M = 0

        self.reidentification_rate_S = 0
        self.advanced_reidentification_rate_S = 0
        self.reidentification_duration_S = 0
        self.memory_consumption_S = 0
        self.summation_error_S = 0
        self.deviation_error_S = 0
        self.encryption_time_S = 0
        self.decryption_time_S = 0
        self.summation_operations_time_S = 0
        self.deviation_operations_time_S = 0

        self.reidentification_rate_V = 0
        self.advanced_reidentification_rate_V = 0
        self.reidentification_duration_V = 0
        self.memory_consumption_V = 0
        self.summation_error_V = 0
        self.deviation_error_V = 0
        self.encryption_time_V = 0
        self.decryption_time_V = 0
        self.summation_operations_time_V = 0
        self.deviation_operations_time_V = 0

        self.reidentification_rate_Z = 0
        self.advanced_reidentification_rate_Z = 0
        self.reidentification_duration_Z = 0
        self.memory_consumption_Z = 0
        self.summation_error_Z = 0
        self.deviation_error_Z = 0
        self.encryption_time_Z = 0
        self.decryption_time_Z = 0
        self.summation_operations_time_Z = 0
        self.deviation_operations_time_Z = 0

        self.k = 0

    def update(self, current_reidentification_rate, current_advanced_reidentification_rate,
               current_reidentification_duration, current_memory_consumption,
               current_summation_error,
               current_deviation_error, current_encryption_time, current_decryption_time,
               current_summation_operations_time,
               current_deviation_operations_time):
        self.k += 1

        old_reidentification_rate_M = self.reidentification_rate_M
        old_advanced_reidentification_rate_M = self.advanced_reidentification_rate_M
        old_reidentification_duration_M = self.reidentification_duration_M
        old_memory_consumption_M = self.memory_consumption_M
        old_summation_error_M = self.summation_error_M
        old_deviation_error_M = self.deviation_error_M
        old_encryption_time_M = self.encryption_time_M
        old_decryption_time_M = self.decryption_time_M
        old_summation_operations_time_M = self.summation_operations_time_M
        old_deviation_operations_time_M = self.deviation_operations_time_M

        self.reidentification_rate_M += (current_reidentification_rate - self.reidentification_rate_M) / self.k
        self.advanced_reidentification_rate_M += (
                                                             current_advanced_reidentification_rate - self.advanced_reidentification_rate_M) / self.k
        self.reidentification_duration_M += (
                                                    current_reidentification_duration - self.reidentification_duration_M) / self.k
        self.memory_consumption_M += (current_memory_consumption - self.memory_consumption_M) / self.k
        self.summation_error_M += (current_summation_error - self.summation_error_M) / self.k
        self.deviation_error_M += (current_deviation_error - self.deviation_error_M) / self.k
        self.encryption_time_M += (current_encryption_time - self.encryption_time_M) / self.k
        self.decryption_time_M += (current_decryption_time - self.decryption_time_M) / self.k
        self.summation_operations_time_M += (
                                                    current_summation_operations_time - self.summation_operations_time_M) / self.k
        self.deviation_operations_time_M += (
                                                    current_deviation_operations_time - self.deviation_operations_time_M) / self.k

        self.reidentification_rate_S += (current_reidentification_rate - self.reidentification_rate_M) * (
                current_reidentification_rate - old_reidentification_rate_M)
        self.advanced_reidentification_rate_S += (
                                                             current_advanced_reidentification_rate - self.advanced_reidentification_rate_M) * (
                                                             current_advanced_reidentification_rate - old_advanced_reidentification_rate_M)
        self.reidentification_duration_S += (current_reidentification_duration - self.reidentification_duration_M) * (
                current_reidentification_duration - old_reidentification_duration_M)
        self.memory_consumption_S += (current_memory_consumption - self.memory_consumption_M) * (
                current_memory_consumption - old_memory_consumption_M)
        self.summation_error_S += (current_summation_error - self.summation_error_M) * (
                current_summation_error - old_summation_error_M)
        self.deviation_error_S += (current_deviation_error - self.deviation_error_M) * (
                current_deviation_error - old_deviation_error_M)
        self.encryption_time_S += (current_encryption_time - self.encryption_time_M) * (
                current_encryption_time - old_encryption_time_M)
        self.decryption_time_S += (current_decryption_time - self.decryption_time_M) * (
                current_decryption_time - old_decryption_time_M)
        self.summation_operations_time_S += (current_summation_operations_time - self.summation_operations_time_M) * (
                current_summation_operations_time - old_summation_operations_time_M)
        self.deviation_operations_time_S += (current_deviation_operations_time - self.deviation_operations_time_M) * (
                current_deviation_operations_time - old_deviation_operations_time_M)

    def get_variance(self):
        if self.k < 2:
            return [0.0] * 9

        self.reidentification_rate_V = self.reidentification_rate_S / (self.k - 1)
        self.advanced_reidentification_rate_V = self.advanced_reidentification_rate_V / (self.k - 1)
        self.reidentification_duration_V = self.reidentification_duration_S / (self.k - 1)
        self.memory_consumption_V = self.memory_consumption_S / (self.k - 1)
        self.summation_error_V = self.summation_error_S / (self.k - 1)
        self.deviation_error_V = self.deviation_error_S / (self.k - 1)
        self.encryption_time_V = self.encryption_time_S / (self.k - 1)
        self.decryption_time_V = self.decryption_time_S / (self.k - 1)
        self.summation_operations_time_V = self.summation_operations_time_S / (self.k - 1)
        self.deviation_operations_time_V = self.deviation_operations_time_S / (self.k - 1)

        return [self.reidentification_rate_V, self.advanced_reidentification_rate_V, self.reidentification_duration_V,
                self.memory_consumption_V,
                self.summation_error_V,
                self.deviation_error_V, self.encryption_time_V, self.decryption_time_V,
                self.summation_operations_time_V, self.deviation_operations_time_V]

    def get_standardised_values(self, current_reidentification_rate, current_advanced_reidentification_rate,
                                current_reidentification_duration,
                                current_memory_consumption,
                                current_summation_error,
                                current_deviation_error, current_encryption_time, current_decryption_time,
                                current_summation_operations_time, current_deviation_operations_time):
        if self.k < 2:
            return [0] * 10

        std_devs = [math.sqrt(v) if v > 0 else 1e-8 for v in self.get_variance()]

        self.reidentification_rate_Z = (current_reidentification_rate - self.reidentification_rate_M) / std_devs[0]
        self.advanced_reidentification_rate_Z = (
                                                            current_advanced_reidentification_rate - self.advanced_reidentification_rate_M) / \
                                                std_devs[1]
        self.reidentification_duration_Z = (current_reidentification_duration - self.reidentification_duration_M) / \
                                           std_devs[2]
        self.memory_consumption_Z = (current_memory_consumption - self.memory_consumption_M) / std_devs[3]
        self.summation_error_Z = (current_summation_error - self.summation_error_M) / std_devs[4]
        self.deviation_error_Z = (current_deviation_error - self.deviation_error_M) / std_devs[5]
        self.encryption_time_Z = (current_encryption_time - self.encryption_time_M) / std_devs[6]
        self.decryption_time_Z = (current_decryption_time - self.decryption_time_M) / std_devs[7]
        self.summation_operations_time_Z = (current_summation_operations_time - self.summation_operations_time_M) / \
                                           std_devs[8]
        self.deviation_operations_time_Z = (current_deviation_operations_time - self.deviation_operations_time_M) / \
                                           std_devs[9]

        return [self.reidentification_rate_Z, self.advanced_reidentification_rate_Z, self.reidentification_duration_Z,
                self.memory_consumption_Z,
                self.summation_error_Z,
                self.deviation_error_Z, self.encryption_time_Z, self.decryption_time_Z,
                self.summation_operations_time_Z, self.deviation_operations_time_Z]


class EncryptionSelectorEnv(gym.Env):
    def __init__(self, dataset_type: str):
        super().__init__()
        self._welford = Welford()

        self._encryption_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Read the inputs CSV file generated from Water dataset.
        self._df = pd.read_csv("./WATER_block_level_encryption_ratio_selector/inputs.csv", header=0)

        # Retrieve the unique household IDs from the Water dataset.
        water_households_data_folder_path = './examples/datasets/water/households_10240'
        try:
            folder_filenames_raw = os.listdir(water_households_data_folder_path)
            folder_filenames_sorted = sorted(folder_filenames_raw)  # Sorted alphabetically.
        except FileNotFoundError:
            print(f"Error: Folder not found at {water_households_data_folder_path}. Please check the path.")
            folder_filenames_sorted = []

        unique_household_IDs_from_df = self._df["Household ID"].unique()
        ordered_unique_household_IDs = [
            filename for filename in folder_filenames_sorted if filename in unique_household_IDs_from_df
        ]
        unique_household_IDs = ordered_unique_household_IDs

        # Create permanent testing household subset for comparative performance analysis.
        permanent_testing_household_IDs = unique_household_IDs[-10:]
        unique_household_IDs = unique_household_IDs[:-10]

        # Convert string representation of list to actual list of floats.
        self._df["All Utility Readings in Section"] = self._df["All Utility Readings in Section"].apply(
            lambda x: json.loads(x))

        # Split the dataset into training, validation, and testing household IDs based on the household IDs.
        training_households_array, validation_households_array = train_test_split(unique_household_IDs,
                                                                                  test_size=(10 / 70),
                                                                                  random_state=42,
                                                                                  shuffle=True)

        training_households = training_households_array
        validation_households = validation_households_array

        self._active_households = None  # This will store the list of households for the current phase.

        # Populate training, validation, and testing dataframes based on the split household IDs.
        training_df = self._df[self._df["Household ID"].isin(training_households)]
        validation_df = self._df[self._df["Household ID"].isin(validation_households)]
        testing_df = self._df[self._df["Household ID"].isin(permanent_testing_household_IDs)]

        # Assign dataframes and household IDs as instance attributes.
        self._all_households = self._df["Household ID"]
        self._training_df = training_df
        self._validation_df = validation_df
        self._testing_df = testing_df
        self._training_households = training_households
        self._validation_households = validation_households
        self._testing_households = permanent_testing_household_IDs

        if dataset_type == 'train':
            self._active_households = self._training_households
        elif dataset_type == 'validation':
            self._active_households = self._validation_households
        elif dataset_type == 'test':
            self._active_households = self._testing_households
        else:
            raise ValueError("dataset_type must be 'train', 'validation', or 'test'")

        # Internal state to keep track of the current household and section.
        self._current_household_idx = 0
        self._current_section_idx_in_household = 0
        self._current_household_data = None
        self._household_ids_processed_in_phase = []
        self._episode_choices_for_log = []

        print(f"DEBUG: __init__ - Number of active households: {len(self._active_households)}")

        self._chosen_encryption_ratios = {}
        self._all_party_metrics = None
        self._go_metrics = None

        # Define observation space boundaries.
        max_section_level_water_usage = self._df["Per Section Utility Usage"].max()
        min_section_level_water_usage = self._df["Per Section Utility Usage"].min()

        # Calculate max and min pre-encryption section-level entropy.
        all_readings_flat = [reading for section_list in self._df["All Utility Readings in Section"] for reading in
                             section_list]  # shape: (819200) > 80 households x 10 sections x 1024 readings.
        counts = Counter(all_readings_flat)
        total = len(all_readings_flat)
        assert len(self._df["All Utility Readings in Section"]) == 800

        prob = np.array([count / total for count in
                         counts.values()])  # Calculates the relative frequency of each reading in your dataset.
        self._max_global_entropy = -np.sum(prob * np.log2(prob))  # Shannon Entropy max is log2(n).
        self._min_global_entropy = 0.0

        self._max_section_level_water_usage = max_section_level_water_usage
        self._min_section_level_water_usage = min_section_level_water_usage

        # Observations are dictionaries that describe the current state of the environment the agent.
        # NOTE: The keys here MUST match the keys in _get_observation()
        self.observation_space = gym.spaces.Dict({
            "section_level_water_usage": gym.spaces.Box(low=self._min_section_level_water_usage,
                                                        high=self._max_section_level_water_usage,
                                                        shape=(1,), dtype=np.float64),
            "section_raw_entropy": gym.spaces.Box(low=self._min_global_entropy,
                                                  high=self._max_global_entropy, shape=(1,), dtype=np.float64),

        })

        entropy_scalar = preprocessing.MinMaxScaler()
        all_readings_flat_reshaped = np.array(all_readings_flat).reshape(-1, 1)
        self._entropy_scalar = entropy_scalar.fit(all_readings_flat_reshaped)

        # Actions are discrete integers that describe the action to be taken by the agent.
        # We have 10 discrete actions, corresponding to the 10 valid encryption ratios.
        self.action_space = gym.spaces.Discrete(10)

    def _calculate_entropy(self, data):
        if not data:
            return 0.0

        frequency = {}
        for val in data:
            # Rounding here treats inputs that are very close to each other as identical
            # for the purpose of entropy calculations, converting continuous-like data
            # into discrete "bins" before calculating probabilities.
            # Using 3 decimal places as per Go code (1000 multiplier)
            rounded_val = round((val * 1000) / 1000)
            frequency[rounded_val] = frequency.get(rounded_val, 0) + 1

        total = float(len(data))
        entropy = 0.0
        for count in frequency.values():
            if count > 0:
                p = float(count) / total
                entropy -= p * math.log2(p)
        return entropy

    def _apply_encryption_ratio_and_quantisation(self, raw_section_data, enc_ratio):
        # Create a copy to avoid modifying the original dataframe data
        processed_data = list(raw_section_data)
        for i in range(len(processed_data)):
            processed_data[i] *= (1.0 - enc_ratio)

            # Apply quantisation (coarser rounding) based on the encryption ratio
            # To simulate the information loss that occurs during a privacy-preserving.
            precision = 0.0
            if enc_ratio <= 0.3:
                precision = 100.0
            elif enc_ratio <= 0.7:
                precision = 10.0
            else:  # encRatio > 0.7
                precision = 1.0

            if precision > 0:
                processed_data[i] = float(round(processed_data[i] * precision) / precision)
            else:
                processed_data[i] = round(processed_data[i])

        return processed_data

    # _get_observation() function provides the current state of the environment the agent is interacting with, i.e., independent of an encryption ratio.
    def _get_observation(self):

        current_household_id = self._active_households[self._current_household_idx]
        section_data_for_household = self._df[
            self._df["Household ID"] == current_household_id].sort_values(by="Section Number")

        current_section_row = section_data_for_household.iloc[self._current_section_idx_in_household]
        section_utility_usage = current_section_row["Per Section Utility Usage"]
        utility_readings_array = current_section_row["All Utility Readings in Section"]

        # Calculate raw entropy for the current section using the time-series array
        section_raw_entropy = self._calculate_entropy(utility_readings_array)

        return {
            "section_level_water_usage": np.array([section_utility_usage], dtype=np.float64),
            "section_raw_entropy": np.array([section_raw_entropy], dtype=np.float64)
        }

    # Returns state information about the current section and current household that the model is selecting an encryption ratio for.
    def _get_intermediate_info(self):
        current_household_id = self._active_households[self._current_household_idx]
        section_data_for_household = self._df[
            self._df["Household ID"] == current_household_id].sort_values(by="Section Number")
        current_section_row = section_data_for_household.iloc[self._current_section_idx_in_household]
        return {
            "household_id": current_household_id,
            "section_number": current_section_row["Section Number"],
            "date_range": current_section_row["Date Range"],
            "per_section_utility_usage": current_section_row["Per Section Utility Usage"],
            "all_utility_readings_in_section": current_section_row["All Utility Readings in Section"]

        }

    # Return the state information of all households and sections after the model has selected an encryption ratio for.
    def _get_terminated_global_info(self):
        if self._go_metrics is None or self._all_party_metrics is None:
            print("DEBUG: _go_metrics or _all_party_metrics is None, returning zeros.")
            return {
                "global_reidentification_rate": 0,
                "global_advanced_reidentification_rate": 0,
                "global_reidentification_duration": 0,
                "global_memory_consumption": 0,
                "global_summation_error": 0,
                "global_deviation_error": 0,
                "global_encryption_time": 0,
                "global_decryption_time": 0,
                "global_summation_operations_time": 0,
                "global_deviation_operations_time": 0,
            }

        global_reidentification_rate = self._go_metrics["globalReidentificationRate"]
        global_advanced_reidentification_rate = self._go_metrics["globalAdvancedReidentificationRate"]
        global_reidentification_duration = self._go_metrics["globalReidentificationDurationNS"]
        global_memory_consumption = self._go_metrics["globalMemoryConsumptionMiB"]

        global_summation_error = sum(p["summationError"] for p in self._all_party_metrics.values())
        global_deviation_error = sum(p["deviationError"] for p in self._all_party_metrics.values())
        global_encryption_time = sum(p["encryptionTimeNS"] for p in self._all_party_metrics.values())
        global_decryption_time = sum(p["decryptionTimeNS"] for p in self._all_party_metrics.values())
        global_summation_operations_time = sum(p["summationOpsTimeNS"] for p in self._all_party_metrics.values())
        global_deviation_operations_time = sum(p["deviationOpsTimeNS"] for p in self._all_party_metrics.values())

        return {
            "global_reidentification_rate": global_reidentification_rate,
            "global_advanced_reidentification_rate": global_advanced_reidentification_rate,
            "global_reidentification_duration": global_reidentification_duration,
            "global_memory_consumption": global_memory_consumption,
            "global_summation_error": global_summation_error,
            "global_deviation_error": global_deviation_error,
            "global_encryption_time": global_encryption_time,
            "global_decryption_time": global_decryption_time,
            "global_summation_operations_time": global_summation_operations_time,
            "global_deviation_operations_time": global_deviation_operations_time,
        }

    def step(self, action):

        # Map the action to the encryption ratio chosen by the agent.
        selected_encryption_ratio = self._encryption_ratios[action]

        info = self._get_intermediate_info()
        current_household_id = info["household_id"]
        current_section_number = info["section_number"]

        # print(f"DEBUG: Entering step method. Current household ID: {current_household_id}, Current household index: {self._current_household_idx}, section index: {self._current_section_idx_in_household}")

        raw_utility_readings_array = info["all_utility_readings_in_section"]
        section_raw_entropy = self._calculate_entropy(raw_utility_readings_array)

        encrypted_quantised_data = self._apply_encryption_ratio_and_quantisation(raw_utility_readings_array,
                                                                                 selected_encryption_ratio)
        section_remaining_entropy = self._calculate_entropy(encrypted_quantised_data)

        self._chosen_encryption_ratios[(current_household_id, current_section_number)] = {
            "ratio": selected_encryption_ratio,
            "raw_entropy": section_raw_entropy,
            "remaining_entropy": section_remaining_entropy,
            "original_utility_readings": raw_utility_readings_array,
        }
        scaled_remaining_entropy = self._entropy_scalar.transform(pd.DataFrame([[section_remaining_entropy]]))[0][0]

        intermediate_reward = scaled_remaining_entropy
        reward = intermediate_reward
        self._episode_choices_for_log.append(
            f"H{current_household_id}-S{current_section_number}:{selected_encryption_ratio:.2f}")

        # Advance to the next section.
        self._current_section_idx_in_household += 1
        terminated = False
        truncated = False

        if self._current_section_idx_in_household == 0 and self._current_household_idx > 0:
            next_household_id = self._active_households[self._current_household_idx]
            self._current_household_data = self._df[
                self._df["Household ID"] == next_household_id].sort_values(by="Section Number")
            # print(f"DEBUG: Step - Loaded data for new household: {next_household_id}")
            # print(f"DEBUG: Step - Number of sections in new household data: {len(self._current_household_data)}")

        # Check if all sections in current household are processed.
        if self._current_section_idx_in_household >= len(self._current_household_data):
            self._current_household_idx += 1  # Move to the next household.
            self._current_section_idx_in_household = 0  # Reset section index.

            # current_household_id = self._active_households[self._current_household_idx]
            # print(f"DEBUG: Reset - Loaded data for household: {current_household_id}")
            # print(f"DEBUG: Reset - Number of sections in current household data: {len(self._current_household_data)}")
            # print(f"DEBUG: Reset - Total active households: {len(self._active_households)}")

            if self._current_household_idx >= len(self._active_households):
                terminated = True  # All households in this phase are processed.
                self._household_ids_processed_in_phase.extend(self._active_households)

        # print(f"DEBUG: 'terminated' flag value before conditional block: {terminated}")

        if terminated:
            info["chosen_encryption_ratios"] = self._episode_choices_for_log
            try:
                # 1. Prepare data for the Go program
                data_for_go = [
                    {
                        "household_id": str(k[0]),
                        "section_number": int(k[1]),
                        "ratio": v["ratio"],
                        "raw_section_entropy": v["raw_entropy"],
                        "remaining_section_entropy": v["remaining_entropy"],
                        "original_utility_readings": v["original_utility_readings"],
                    }
                    for k, v in self._chosen_encryption_ratios.items()
                ]
                data_for_go_filepath = os.path.join(SCRIPT_DIR, "RL_V2_choices.json")

                # print(f"\nDEBUG: Data for Go program (first entry, total {len(data_for_go)}):")
                # for i, entry in enumerate(data_for_go[:1]):
                #     print(f"  {i}: {entry}")
                # print("-" * 50)

                with open(data_for_go_filepath, "w") as f:
                    json.dump(data_for_go, f)
                # print(f"DEBUG: RL_choices.json created at {os.path.abspath(data_for_go_filepath)}")

                # 2. Run the Go program as a subprocess.
                # print(f"\nEpisode finished. Calling Go program to calculate reward metrics...")

                go_result = subprocess.run(
                    [GO_EXECUTABLE_PATH, data_for_go_filepath, currentLeakedPlaintextSize],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=3600  # 1 hour
                )

                # print(f"Go Program stdout: {go_result.stdout}")
                # print(f"Go Program stderr: {go_result.stderr}")

                # 3. Parse the metrics from the Go stdout
                self._go_metrics = json.loads(go_result.stdout)
                self._all_party_metrics = self._go_metrics["allPartyMetrics"]

                # print(f"DEBUG: self._go_metrics after parsing: {self._go_metrics}")
                # print(f"DEBUG: self._all_party_metrics after parsing: {self._all_party_metrics}")

                global_reidentification_rate = self._go_metrics["globalReidentificationRate"]
                global_advanced_reidentification_rate = self._go_metrics["globalAdvancedReidentificationRate"]
                global_reidentification_duration = self._go_metrics["globalReidentificationDurationNS"]
                global_memory_consumption = self._go_metrics["globalMemoryConsumptionMiB"]

                global_summation_error = sum(p["summationError"] for p in self._all_party_metrics.values())
                global_deviation_error = sum(p["deviationError"] for p in self._all_party_metrics.values())
                global_encryption_time = sum(p["encryptionTimeNS"] for p in self._all_party_metrics.values())
                global_decryption_time = sum(p["decryptionTimeNS"] for p in self._all_party_metrics.values())
                global_summation_operations_time = sum(
                    p["summationOpsTimeNS"] for p in self._all_party_metrics.values())
                global_deviation_operations_time = sum(
                    p["deviationOpsTimeNS"] for p in self._all_party_metrics.values())

                self._welford.update(global_reidentification_rate, global_advanced_reidentification_rate,
                                     global_reidentification_duration,
                                     global_memory_consumption,
                                     global_summation_error, global_deviation_error, global_encryption_time,
                                     global_decryption_time, global_summation_operations_time,
                                     global_deviation_operations_time)

                z_scores = self._welford.get_standardised_values(global_reidentification_rate,
                                                                 global_advanced_reidentification_rate,
                                                                 global_reidentification_duration,
                                                                 global_memory_consumption,
                                                                 global_summation_error, global_deviation_error,
                                                                 global_encryption_time,
                                                                 global_decryption_time,
                                                                 global_summation_operations_time,
                                                                 global_deviation_operations_time)

                z_reidentification_rate, z_advanced_reidentification_rate, z_reidentification_duration, z_memory, z_sum_error, z_dev_error, z_enc_time, z_dec_time, z_sum_ops_time, z_dev_ops_time = z_scores

                # Privacy cost = reidentification success
                privacy_cost = z_reidentification_rate + z_advanced_reidentification_rate - (
                            z_reidentification_duration + z_dec_time)

                # Utility cost = errors + computation time
                utility_cost = z_sum_error + z_dev_error + z_enc_time + z_sum_ops_time + z_dev_ops_time + z_memory

                # 4. Calculate the final reward.
                reward = intermediate_reward - privacy_cost - utility_cost

                # Store global metrics in info dictionary for the callback
                info["global_metrics"] = {
                    "global_reidentification_rate": global_reidentification_rate,
                    "global_advanced_reidentification_rate": global_advanced_reidentification_rate,
                    "global_reidentification_duration": global_reidentification_duration,
                    "global_memory_consumption": global_memory_consumption,
                    "global_summation_error": global_summation_error,
                    "global_deviation_error": global_deviation_error,
                    "global_encryption_time": global_encryption_time,
                    "global_decryption_time": global_decryption_time,
                    "global_summation_operations_time": global_summation_operations_time,
                    "global_deviation_operations_time": global_deviation_operations_time,
                }

                info["terminated_household_id"] = current_household_id
                info["training_households"] = self._training_households
                per_household_metrics = {}
                if self._all_party_metrics:
                    per_household_metrics["summation_errors"] = [f"H{p['partyID']}:{p['summationError']:.6f}" for p in
                                                                 self._all_party_metrics.values()]
                    per_household_metrics["deviation_errors"] = [f"H{p['partyID']}:{p['deviationError']:.6f}" for p in
                                                                 self._all_party_metrics.values()]
                    per_household_metrics["encryption_times"] = [f"H{p['partyID']}:{p['encryptionTimeNS']}" for p in
                                                                 self._all_party_metrics.values()]
                    per_household_metrics["decryption_times"] = [f"H{p['partyID']}:{p['decryptionTimeNS']}" for p in
                                                                 self._all_party_metrics.values()]
                    per_household_metrics["summation_ops_times"] = [f"H{p['partyID']}:{p['summationOpsTimeNS']}" for p
                                                                    in self._all_party_metrics.values()]
                    per_household_metrics["deviation_ops_times"] = [f"H{p['partyID']}:{p['deviationOpsTimeNS']}" for p
                                                                    in self._all_party_metrics.values()]
                info["per_household_metrics"] = per_household_metrics


            except subprocess.CalledProcessError as e:
                print(f"ERROR: Go program failed with CalledProcessError: {e}")
                print(f"Stderr: {e.stderr}")
                info["global_metrics"] = self._get_terminated_global_info()
                self._go_metrics = None
                self._all_party_metrics = None
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to decode JSON from Go program: {e}")
                info["global_metrics"] = self._get_terminated_global_info()
                self._go_metrics = None
                self._all_party_metrics = None
            except FileNotFoundError:
                print(
                    "ERROR: Go program 'generate_metrics' not found. Make sure it's compiled and in the correct path.")
                info["global_metrics"] = self._get_terminated_global_info()
                self._go_metrics = None
                self._all_party_metrics = None
            except Exception as e:
                print(f"ERROR: An unexpected error occurred in step method: {e}")
                info["global_metrics"] = self._get_terminated_global_info()
                self._go_metrics = None
                self._all_party_metrics = None

        if not terminated:
            observation = self._get_observation()
        else:
            # StableBaselines3 expects an observation even if terminated.
            observation = self.observation_space.sample()
            if "global_metrics" not in info:
                info["global_metrics"] = self._get_terminated_global_info()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # print("DEBUG: Environment reset called.")

        self._current_household_idx = 0
        self._current_section_idx_in_household = 0
        self._household_ids_processed_in_phase = []
        self._episode_choices_for_log = []
        self._chosen_encryption_ratios = {}
        self._go_metrics = None
        self._all_party_metrics = None

        current_household_id = self._active_households[self._current_household_idx]
        self._current_household_data = self._df[
            self._df["Household ID"] == current_household_id].sort_values(by="Section Number")

        observation = self._get_observation()
        info = self._get_intermediate_info()

        return observation, info

    def render(self):
        pass


class SectionLoggingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param log_path_section: Path for section-level log (optional).
    :param log_path_global_train: Path for global training log.
    :param log_path_global_test: Path for global testing log.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, current_dataset_type: str, log_path_global_train: str, log_path_global_test_ph: str,
                 log_path_global_test_combined, verbose: int = 0):
        super().__init__(verbose)

        # self.log_path_section = log_path_section
        # self.log_file_section = None
        # self.writer_section = None

        self.current_dataset_type = current_dataset_type
        self.log_path_global_train = log_path_global_train
        self.log_path_global_test_ph = log_path_global_test_ph
        self.log_path_global_test_combined = log_path_global_test_combined
        self.log_files = {}
        self.writers = {}
        self.episode_num = 0

    def _on_training_start(self) -> None:
        """
               This method is called before the first rollout starts.
        """
        self.episode_num = 0

        # log_headers_section = [
        #     "Step",
        #     "Household ID",
        #     "Section Number",
        #     "Selected Encryption Ratio",
        #     "Intermediate Reward"
        # ]

        log_headers_train = [
            "Episode",
            "Processed Household(s)",
            "Chosen Encryption Ratios",
            "Reidentification Rate",
            "Advanced Reidentification Rate",
            "Reidentification Duration (NS)",
            "Memory Consumption (MiB)",
            "Global Summation Error",
            "Global Deviation Error",
            "Global Encryption Time (NS)",
            "Global Decryption Time (NS)",
            "Global Summation Operations Time (NS)",
            "Global Deviation Operations Time (NS)"
        ]

        log_headers_test = [
            "Episode",
            "Processed Household(s)",
            "Chosen Encryption Ratios",
            "Reidentification Rate",
            "Advanced Reidentification Rate",
            "Reidentification Duration (NS)",
            "Memory Consumption (MiB)",
            "Per-Party Summation Errors",
            "Per-Party Deviation Errors",
            "Per-Party Encryption Times (NS)",
            "Per-Party Decryption Times (NS)",
            "Per-Party Summation Operations Times (NS)",
            "Per-Party Deviation Operations Times (NS)"
        ]

        # try:
        #     if self.log_path_section:
        #         self.log_file_section = open(self.log_path_section, 'w', newline='')
        #         self.writer_section = csv.writer(self.log_file_section)
        #         self.writer_section.writerow(log_headers_section)
        #
        # except Exception as e:
        #     print(f"ERROR: Failed to open log file or initialize CSV writer: {e}")
        #     raise

        try:
            if self.log_path_global_train:
                self.log_files['train'] = open(self.log_path_global_train, 'w', newline='')
                self.writers['train'] = csv.writer(self.log_files['train'])
                self.writers['train'].writerow(log_headers_train)

            if self.log_path_global_test_ph:
                self.log_files['test_ph'] = open(self.log_path_global_test_ph, 'w', newline='')
                self.writers['test_ph'] = csv.writer(self.log_files['test_ph'])
                self.writers['test_ph'].writerow(log_headers_test)

            if self.log_path_global_test_combined:
                self.log_files['test_combined'] = open(self.log_path_global_test_combined, 'w', newline='')
                self.writers['test_combined'] = csv.writer(self.log_files['test_combined'])
                self.writers['test_combined'].writerow(log_headers_test)

        except Exception as e:
            print(f"ERROR: Failed to open log file or initialize CSV writer: {e}")
            raise

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        # if self.log_path_section and len(self.locals['infos']) > 0:
        #     if not self.locals['dones'][0]:  # Only log section data if the episode is NOT terminated in this step
        #         chosen_ratio_info = env_instance._chosen_encryption_ratios.get(
        #             (info['household_id'], info['section_number']))
        #         if chosen_ratio_info:
        #             self.writer_section.writerow([
        #                 self.n_calls,
        #                 info.get('household_id'),
        #                 info.get('section_number'),
        #                 chosen_ratio_info.get('ratio'),
        #                 self.locals['rewards'][0],  # Intermediate reward, i.e., remaining entropy.
        #             ])

        if self.locals['dones'][0]:
            self.episode_num += 1
            info = self.locals['infos'][0]
            global_info = info.get('global_metrics')
            per_household_info = info.get('per_household_metrics', {})

            writer = None
            households_to_log = "N/A"
            log_type = self.current_dataset_type

            if log_type == 'train':
                writer = self.writers.get('train')
                households_to_log = info.get('training_households')

            elif log_type == 'test_combined':
                if info.get('testing_households_in_run'):
                    writer = self.writers.get('test_combined')
                    households_to_log = info.get('testing_households_in_run')

            elif log_type == 'test_ph':
                writer = self.writers.get('test_ph')
                households_to_log = info.get('terminated_household_id')

            if writer and global_info:
                chosen_ratios = info.get("chosen_encryption_ratios", "N/A")

                if log_type == 'train':
                    metrics_values = [
                        global_info.get("global_reidentification_rate", 0),
                        global_info.get("global_advanced_reidentification_rate", 0),
                        global_info.get("global_reidentification_duration", 0),
                        global_info.get("global_memory_consumption", 0),
                        global_info.get("global_summation_error", 0),
                        global_info.get("global_deviation_error", 0),
                        global_info.get("global_encryption_time", 0),
                        global_info.get("global_decryption_time", 0),
                        global_info.get("global_summation_operations_time", 0),
                        global_info.get("global_deviation_operations_time", 0)
                    ]
                    writer.writerow([self.episode_num, households_to_log, "N/A"] + metrics_values)

                if self.current_dataset_type == 'test_combined':
                    global_metrics_values = [
                        global_info.get("global_reidentification_rate", 0),
                        global_info.get("global_advanced_reidentification_rate", 0),
                        global_info.get("global_reidentification_duration", 0),
                        global_info.get("global_memory_consumption", 0),
                    ]

                    per_party_metrics_str = [
                        "; ".join(per_household_info.get("summation_errors", [])),
                        "; ".join(per_household_info.get("deviation_errors", [])),
                        "; ".join(per_household_info.get("encryption_times", [])),
                        "; ".join(per_household_info.get("decryption_times", [])),
                        "; ".join(per_household_info.get("summation_ops_times", [])),
                        "; ".join(per_household_info.get("deviation_ops_times", [])),
                    ]

                    writer.writerow([self.episode_num, households_to_log,
                                     chosen_ratios] + global_metrics_values + per_party_metrics_str)

            elif writer:
                print(
                    f"WARNING: Global metrics not found for {self.current_dataset_type} episode {self.episode_num}. Skipping log.")
        return True

    def _on_training_end(self) -> None:
        """
               This event is triggered before exiting the `learn()` method.
        """
        # if self.log_file_section:
        #     self.log_file_section.close()

        for file in self.log_files.values():
            file.close()


class ConvergenceStoppingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback`` that stops the training phase when the moving average reward
    has stabilised.
    """

    def __init__(self, check_freq: int, window_size: int, variance_threshold: float, verbose: int = 0):
        super(ConvergenceStoppingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        self.episode_rewards = []
        self.last_mean_rewards = []

    def _on_rollout_end(self) -> None:
        """
        This is called at the end of each episode.
        """
        if self.locals.get('dones') and self.locals['dones'][-1]:
            episode_reward = np.sum(self.locals['rewards'][:-1])
            self.episode_rewards.append(episode_reward)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) >= self.window_size:
            recent_rewards = self.episode_rewards[-self.window_size:]

            mean_reward = np.mean(recent_rewards)
            self.last_mean_rewards.append(mean_reward)

            if len(self.last_mean_rewards) >= 2 and np.var(self.last_mean_rewards) < self.variance_threshold:
                if self.verbose > 0:
                    print(f"Stopping training at timestep {self.n_calls}. "
                          f"Mean reward variance over the last {self.window_size} episodes is "
                          f"{np.var(self.last_mean_rewards):.4f}, which is below the threshold of "
                          f"{self.variance_threshold}.")
                return False

        return True


def main():
    if len(sys.argv) != 2:
        print("WARNING: Not enough arguments provided! Please provide the leaked plaintext size as an argument.")
        current_leaked_plaintext_size = "12"
    else:
        current_leaked_plaintext_size = sys.argv[1]

    try:
        subprocess.run(["go", "build", "-o", GO_EXECUTABLE_PATH, GO_SOURCE_PATH], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to compile Go program: {e}")
        return

    if not os.path.exists(GO_EXECUTABLE_PATH):
        raise FileNotFoundError(f"Go executable not found at: {GO_EXECUTABLE_PATH}")

    # ----- TRAINING PHASE ------
    print("\n----- TRAINING PHASE BEGIN ------")
    env_train = EncryptionSelectorEnv(dataset_type="train")

    model = DQN(policy=MultiInputPolicy, env=env_train, verbose=1)

    logging_callback = SectionLoggingCallback(
        current_dataset_type="train",
        log_path_global_train=os.path.join(os.getcwd(), f'./WATER_block_level_encryption_ratio_selector/training_log_{current_leaked_plaintext_size}.csv'),
        log_path_global_test_ph=None,
        log_path_global_test_combined=None,
        verbose=0)

    convergence_callback = ConvergenceStoppingCallback(
        check_freq=300000,  # How many timesteps before checking convergence? [Check every 500 episodes]
        window_size=100,  # How many episodes to use to calculate the moving average reward? [Use last 100 episodes]
        variance_threshold=0.1,
    )

    combined_callbacks = CallbackList([logging_callback, convergence_callback])

    model.learn(total_timesteps=6000000,
                callback=combined_callbacks)  # 1 episode: total_timesteps = 60 testing households x 10 sections (600)
    model.save(f"./WATER_block_level_encryption_ratio_selector/DQN_Block_Level_Encryption_Ratio_Selector_{current_leaked_plaintext_size}")

    del model

    # ----- VALIDATION PHASE ------
    print("\n----- VALIDATION PHASE BEGIN ------")

    model = DQN.load(f"./WATER_block_level_encryption_ratio_selector/DQN_Block_Level_Encryption_Ratio_Selector_{current_leaked_plaintext_size}")
    env_val = EncryptionSelectorEnv(dataset_type="validation")
    env_val.reset()
    model.set_env(env_val)
    evaluate_policy(model, env_val, render=False)

    # ----- TESTING PHASE (Combined) ------
    print("\n----- TESTING PHASE (Combined) BEGIN ------")

    model = DQN.load(f"./WATER_block_level_encryption_ratio_selector/DQN_Block_Level_Encryption_Ratio_Selector_{current_leaked_plaintext_size}")
    env_test_combined = EncryptionSelectorEnv(dataset_type="test")
    model.set_env(env_test_combined)

    combined_test_callback = SectionLoggingCallback(
        current_dataset_type="test_combined",
        log_path_global_train=None,
        log_path_global_test_ph=None,
        log_path_global_test_combined=os.path.join(os.getcwd(), f'./WATER_block_level_encryption_ratio_selector/testing_log_{current_leaked_plaintext_size}.csv'),
        verbose=0
    )
    combined_test_callback.init_callback(model)
    combined_test_callback._on_training_start()

    obs, info = env_test_combined.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test_combined.step(action)
        done = terminated or truncated
        episode_reward += reward

        if done:
            info['testing_households_in_run'] = env_test_combined._testing_households

        combined_test_callback.locals = {'infos': [info], 'dones': [done], 'rewards': [reward]}
        combined_test_callback._on_step()

    combined_test_callback._on_training_end()


    # # ----- TESTING PHASE (Per-Household) ------
    # print("\n----- TESTING PHASE (Per-Household) BEGIN ------")
    # start_time_ph = time.time()
    #
    # model = DQN.load("DQN_Encryption_Ratio_Selector_V2")
    #
    # temp_env_for_ids = EncryptionSelectorEnv(dataset_type="test")
    # testing_household_ids = temp_env_for_ids._testing_households
    # del temp_env_for_ids
    #
    # ph_callback = SectionLoggingCallback(
    #     current_dataset_type="test_ph",
    #     log_path_global_train=None,
    #     log_path_global_test_ph=os.path.join(os.getcwd(), 'V2_testing_log_ph.csv'),
    #     log_path_global_test_combined=None,
    #     verbose=0
    # )
    #
    # ph_callback.init_callback(model)
    # ph_callback._on_training_start()
    #
    # for i, household_id in enumerate(testing_household_ids):
    #     print(f"\n--- Running test for Household: {household_id} ({i + 1}/{len(testing_household_ids)}) ---")
    #
    #     env_test_single = EncryptionSelectorEnv(dataset_type="test")
    #     env_test_single._active_households = [household_id]
    #
    #     model.set_env(env_test_single)
    #
    #     obs, info = env_test_single.reset()
    #     done = False
    #     while not done:
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, reward, terminated, truncated, info = env_test_single.step(action)
    #         done = terminated or truncated
    #
    #         ph_callback.locals = {'infos': [info], 'dones': [done], 'rewards': [reward]}
    #         ph_callback._on_step()
    #
    #
    # ph_callback._on_training_end()
    #
    # end_time_ph = time.time()
    # print(f"Per-household testing finished at: {time.ctime(end_time_ph)}")
    # elapsed_time_ph = end_time_ph - start_time_ph
    # print(f"Total per-household testing duration: {elapsed_time_ph:.2f} seconds")


if __name__ == "__main__":
    main()
