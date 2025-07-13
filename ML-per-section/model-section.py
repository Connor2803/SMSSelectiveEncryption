# python model.py

import csv
import json
import math
import random
import subprocess
import time
from collections import Counter

import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.dqn.policies import MultiInputPolicy


class Welford:
    def __init__(self):
        # Where, M is the mean, S is the sum of square differences, V is the variance,
        # Z is the standard deviation, and K is the current number of data points.
        self.asr_mean_M = 0
        self.asr_duration_M = 0
        self.memory_consumption_M = 0
        self.summation_error_M = 0
        self.deviation_error_M = 0
        self.encryption_time_M = 0
        self.decryption_time_M = 0
        self.summation_operations_time_M = 0
        self.deviation_operations_time_M = 0

        self.asr_mean_S = 0
        self.asr_duration_S = 0
        self.memory_consumption_S = 0
        self.summation_error_S = 0
        self.deviation_error_S = 0
        self.encryption_time_S = 0
        self.decryption_time_S = 0
        self.summation_operations_time_S = 0
        self.deviation_operations_time_S = 0

        self.asr_mean_V = 0
        self.asr_duration_V = 0
        self.memory_consumption_V = 0
        self.summation_error_V = 0
        self.deviation_error_V = 0
        self.encryption_time_V = 0
        self.decryption_time_V = 0
        self.summation_operations_time_V = 0
        self.deviation_operations_time_V = 0

        self.asr_mean_Z = 0
        self.asr_duration_Z = 0
        self.memory_consumption_Z = 0
        self.summation_error_Z = 0
        self.deviation_error_Z = 0
        self.encryption_time_Z = 0
        self.decryption_time_Z = 0
        self.summation_operations_time_Z = 0
        self.deviation_operations_time_Z = 0

        self.k = 0

    def update(self, current_asr_mean, current_asr_duration, current_memory_consumption, current_summation_error,
               current_deviation_error, current_encryption_time, current_decryption_time,
               current_summation_operations_time,
               current_deviation_operations_time):
        self.k += 1

        old_asr_mean_M = self.asr_mean_M
        old_asr_duration_M = self.asr_duration_M
        old_memory_consumption_M = self.memory_consumption_M
        old_summation_error_M = self.summation_error_M
        old_deviation_error_M = self.deviation_error_M
        old_encryption_time_M = self.encryption_time_M
        old_decryption_time_M = self.decryption_time_M
        old_summation_operations_time_M = self.summation_operations_time_M
        old_deviation_operations_time_M = self.deviation_operations_time_M

        self.asr_mean_M += (current_asr_mean - self.asr_mean_M) / self.k
        self.asr_duration_M += (current_asr_duration - self.asr_duration_M) / self.k
        self.memory_consumption_M += (current_memory_consumption - self.memory_consumption_M) / self.k
        self.summation_error_M += (current_summation_error - self.summation_error_M) / self.k
        self.deviation_error_M += (current_deviation_error - self.deviation_error_M) / self.k
        self.encryption_time_M += (current_encryption_time - self.encryption_time_M) / self.k
        self.decryption_time_M += (current_decryption_time - self.decryption_time_M) / self.k
        self.summation_operations_time_M += (
                                                    current_summation_operations_time - self.summation_operations_time_M) / self.k
        self.deviation_operations_time_M += (
                                                    current_deviation_operations_time - self.deviation_operations_time_M) / self.k

        self.asr_mean_S += (current_asr_mean - self.asr_mean_M) * (current_asr_mean - old_asr_mean_M)
        self.asr_duration_S += (current_asr_duration - self.asr_duration_M) * (
                current_asr_duration - old_asr_duration_M)
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
            return 0

        self.asr_mean_V = self.asr_mean_S / (self.k - 1)
        self.asr_duration_V = self.asr_duration_S / (self.k - 1)
        self.memory_consumption_V = self.memory_consumption_S / (self.k - 1)
        self.summation_error_V = self.summation_error_S / (self.k - 1)
        self.deviation_error_V = self.deviation_error_S / (self.k - 1)
        self.encryption_time_V = self.encryption_time_S / (self.k - 1)
        self.decryption_time_V = self.decryption_time_S / (self.k - 1)
        self.summation_operations_time_V = self.summation_operations_time_S / (self.k - 1)
        self.deviation_operations_time_V = self.deviation_operations_time_S / (self.k - 1)

        return [self.asr_mean_V, self.asr_duration_V, self.memory_consumption_V, self.summation_error_V,
                self.deviation_error_V, self.encryption_time_V, self.decryption_time_V,
                self.summation_operations_time_V, self.deviation_operations_time_V]

    def get_standardised_values(self, current_asr_mean, current_asr_duration, current_memory_consumption,
                                current_summation_error,
                                current_deviation_error, current_encryption_time, current_decryption_time,
                                current_summation_operations_time, current_deviation_operations_time):
        if self.k < 2:
            return [0] * 9

        std_devs = [math.sqrt(v) if v > 0 else 1e-8 for v in self.get_variance()]

        self.asr_mean_Z = (current_asr_mean - self.asr_mean_M) / std_devs[0]
        self.asr_duration_Z = (current_asr_duration - self.asr_duration_M) / std_devs[1]
        self.memory_consumption_Z = (current_memory_consumption - self.memory_consumption_M) / std_devs[2]
        self.summation_error_Z = (current_summation_error - self.summation_error_M) / std_devs[3]
        self.deviation_error_Z = (current_deviation_error - self.deviation_error_M) / std_devs[4]
        self.encryption_time_Z = (current_encryption_time - self.encryption_time_M) / std_devs[5]
        self.decryption_time_Z = (current_decryption_time - self.decryption_time_M) / std_devs[6]
        self.summation_operations_time_Z = (current_summation_operations_time - self.summation_operations_time_M) / \
                                           std_devs[7]
        self.deviation_operations_time_Z = (current_deviation_operations_time - self.deviation_operations_time_M) / \
                                           std_devs[8]

        return [self.asr_mean_Z, self.asr_duration_Z, self.memory_consumption_Z, self.summation_error_Z,
                self.deviation_error_Z, self.encryption_time_Z, self.decryption_time_Z,
                self.summation_operations_time_Z, self.deviation_operations_time_Z]


class EncryptionSelectorEnv(gym.Env):
    def __init__(self, dataset_type="train"):
        super().__init__()
        self._welford = Welford()

        self._encryption_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Read the inputs CSV file generated from Water dataset.
        self._df = pd.read_csv("./inputs.csv", header=0)
        self._per_party_df = pd.read_csv("../ML_database_generation-per-party/ML_metrics_WATER.csv", header=0)
        self._per_party_HE_df = pd.read_csv("../ML_database_generation-per-party/ML_party-metrics_WATER.csv", header=0)

        # Convert string representation of list to actual list of floats.
        self._df["All Utility Readings in Section"] = self._df["All Utility Readings in Section"].apply(
            lambda x: json.loads(x))

        # Split the dataset into training, validation, and testing household IDs based on the household IDs.
        training_households_array, validation_and_testing_households_array = train_test_split(self._df["Household ID"],
                                                                                  test_size=0.25,
                                                                                  random_state=42, shuffle=True)
        validation_households_array, testing_households_array = train_test_split(validation_and_testing_households_array, test_size=0.5,
                                                                     random_state=42, shuffle=True)

        training_households = training_households_array.tolist()
        validation_households = validation_households_array.tolist()
        testing_households = testing_households_array.tolist()

        self._active_households = None  # This will store the list of households for the current phase

        # Populate training, validation, and testing dataframes based on the split household IDs.
        training_df = self._df[self._df["Household ID"].isin(training_households)]
        validation_df = self._df[self._df["Household ID"].isin(validation_households)]
        testing_df = self._df[self._df["Household ID"].isin(testing_households)]

        # Assign dataframes and household IDs as instance attributes.
        self._all_households = self._df["Household ID"]
        self._training_df = training_df
        self._validation_df = validation_df
        self._testing_df = testing_df
        self._training_households = training_households
        self._validation_households = validation_households
        self._testing_households = testing_households

        if dataset_type == 'train':
            self._active_households = self._training_households
        elif dataset_type == 'validation':
            self._active_households = self._validation_households
        elif dataset_type == 'test':
            self._active_households = self._testing_households
        else:
            raise ValueError("dataset_type must be 'train', 'validation', or 'test'")

        self._current_household_idx = 0

        self._chosen_encryption_ratios = {}

        # Define observation space boundaries.
        max_section_level_water_usage = self._df["Per Section Utility Usage"].max()
        min_section_level_water_usage = self._df["Per Section Utility Usage"].min()

        # Calculate max and min pre-encryption section-level entropy.
        all_readings_flat = [reading for section_list in self._df["All Utility Readings in Section"] for reading in section_list]  # shape: (819200) > 80 households x 10 sections x 1024 readings.
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
        # We have 9 discrete actions, corresponding to the 9 encryption ratios.
        self.action_space = gym.spaces.Discrete(9)

        # Internal state to keep track of current household and section.
        self._current_household_idx = 0
        self._current_section_idx_in_household = 0
        self._current_household_data = None
        self._household_ids_processed_in_phase = []

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
        utility_readings_array = current_section_row["All Utility Readings in Section"]  # Get the actual array

        # Calculate raw entropy for the current section using the time-series array
        section_raw_entropy = self._calculate_entropy(utility_readings_array)

        return {
            "section_level_water_usage": np.array([section_utility_usage], dtype=np.float64),
            "section_raw_entropy": np.array([section_raw_entropy], dtype=np.float64)
        }

    def _get_info(self):
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

    def step(self, action):
        # Map the action to the encryption ratio chosen by the agent.
        selected_encryption_ratio = self._encryption_ratios[action]

        info = self._get_info()
        current_household_id = info["household_id"]
        current_section_number = info["section_number"]

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

        # Advance to next section.
        self._current_section_idx_in_household += 1
        terminated = False
        truncated = False

        # Check if all sections in current household are processed.
        if self._current_section_idx_in_household >= len(self._current_household_data):
            self._current_household_idx += 1  # Move to the next household.
            self._current_section_idx_in_household = 0  # Reset section index.

            if self._current_household_idx >= len(self._active_households):
                terminated = True  # All households in this phase are processed.
                self._household_ids_processed_in_phase.extend(self._active_households)
            else:
                # Load data for the next household.
                next_household_id = self._active_households[self._current_household_idx]
                self._current_household_data = self._df[
                    self._df["Household ID"] == next_household_id].sort_values(by="Section Number")

        if terminated:
            try:
                # 1. Prepare data for Go program
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
                data_for_go_filepath = "RL_choices.json"
                with open(data_for_go_filepath, "w") as f:
                    json.dump(data_for_go, f)

                # 2. Run the Go program as a subprocess.
                print("\nEpisode finished. Calling Go program to calculate reward metrics...")
                go_result = subprocess.run(
                    ["./generate_metrics", data_for_go_filepath],
                    capture_output=True,
                    text=True,
                    check=True
                )

                # 3. Parse the metrics from the Go stdout
                go_metrics = json.loads(go_result.stdout)
                all_party_metrics = go_metrics["allPartyMetrics"]

                global_asr_mean = go_metrics["GlobalASRMean"]
                global_asr_duration = go_metrics["GlobalASRDurationNS"]
                global_memory_consumption = go_metrics["GlobalMemoryConsumption"]

                global_summation_error = sum(p["summationError"] for p in all_party_metrics.values())
                global_deviation_error = sum(p["deviationError"] for p in all_party_metrics.values())
                global_encryption_time = sum(p["encryptionTimeNS"] for p in all_party_metrics.values())
                global_decryption_time = sum(p["decryptionTimeNS"] for p in all_party_metrics.values())
                global_summation_operations_time = sum(p["summationOpsTimeNS"] for p in all_party_metrics.values())
                global_deviation_operations_time = sum(p["deviationOpsTimeNS"] for p in all_party_metrics.values())

                self._welford.update(global_asr_mean, global_asr_duration, global_memory_consumption,
                                     global_summation_error, global_deviation_error, global_encryption_time,
                                     global_decryption_time, global_summation_operations_time,
                                     global_deviation_operations_time)

                z_scores = self._welford.get_standardised_values(global_asr_mean, global_asr_duration,
                                                                    global_memory_consumption,
                                                                    global_summation_error, global_deviation_error,
                                                                    global_encryption_time,
                                                                    global_decryption_time,
                                                                    global_summation_operations_time,
                                                                    global_deviation_operations_time)

                z_asr_mean, z_asr_duration, z_memory, z_sum_error, z_dev_error, z_enc_time, z_dec_time, z_sum_ops_time, z_dev_ops_time = z_scores

                # Privacy cost = ASR
                privacy_cost = z_asr_mean - (z_asr_duration + z_dec_time)

                # Utility cost = errors + computation time
                utility_cost = z_sum_error + z_dev_error + z_enc_time + z_sum_ops_time + z_dev_ops_time + z_memory

                # 4. Calculate the final reward.
                reward = intermediate_reward - privacy_cost - utility_cost

            except subprocess.CalledProcessError as e:
                print(f"Error executing Go program: {e}")
                print(f"Stderr: {e.stderr}")
                intermediate_reward -= 100
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error processing Go program output: {e}")
                intermediate_reward -= 100

        if not terminated:
            observation = self._get_observation()
        else:
            # StableBaselines3 expects an observation even if terminated.
            observation = self.observation_space.sample()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._active_households and self._current_household_idx >= len(self._active_households):
            self._current_household_idx = 0

        self._current_section_idx_in_household = 0
        self._household_ids_processed_in_phase = []
        self._chosen_encryption_ratios = {}

        current_household_id = self._active_households[self._current_household_idx]
        self._current_household_data = self._df[
            self._df["Household ID"] == current_household_id].sort_values(by="Section Number")

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def render(self):
        pass


def log_to_csv(writer, step_count, household_id, section_number, intermediate_reward, info_step):
    writer.writerow([
        step_count,
        household_id,
        section_number,
        info_step.get('selected_encryption_ratio'),
        intermediate_reward,
    ])


class SectionLoggingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.log_file = None
        self.writer = None

    def _on_training_start(self) -> None:
        """
               This method is called before the first rollout starts.
        """

        log_headers = [
            "Step",
            "Household ID",
            "Section Number",
            "Selected Encryption Ratio",
            "Intermediate Reward"
        ]

        self.log_file = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(log_headers)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            # The info dict might be from the previous step if 'dones' is True
            if not self.locals['dones'][0]:
                chosen_ratio_info = self.training_env.get_attr('_chosen_encryption_ratios')[0].get(
                    (info['household_id'], info['section_number']))
                if chosen_ratio_info:
                    self.writer.writerow([
                        self.n_calls,
                        info.get('household_id'),
                        info.get('section_number'),
                        chosen_ratio_info.get('ratio'),
                        self.locals['rewards'][0],
                    ])
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
               This event is triggered before exiting the `learn()` method.
        """
        if self.log_file:
            self.log_file.close()


def main():
    try:
        subprocess.run(["go", "build", "-o", "generate_metrics", "./generate_metrics-section.go"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to compile Go program: {e}")
        return

    # ----- TRAINING PHASE ------
    env_train = EncryptionSelectorEnv(dataset_type="train")

    model = DQN(policy=MultiInputPolicy, env=env_train, verbose=1)
    model.learn(total_timesteps=10, log_interval=4,
                callback=SectionLoggingCallback(log_path='training_log_sections.csv', verbose=0))

    start_time = time.time()
    print(f"Training started at: {time.ctime(start_time)}")

    model.save("dqn_encryption_selector")

    end_time = time.time()
    print(f"Training finished at: {time.ctime(end_time)}")
    elapsed_time = end_time - start_time
    print(f"Total training duration: {elapsed_time:.2f} seconds")

    # del model


if __name__ == "__main__":
    main()
