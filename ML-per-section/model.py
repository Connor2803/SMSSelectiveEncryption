# python model.py

from tarfile import data_filter
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import time
import csv
import math
from collections import Counter


class EncryptionSelectorEnv(gym.Env):
    def __init__(self, dataset_type="train"):
        super().__init__()

        self._encryption_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Read the inputs CSV file generated from Water dataset.
        self._df = pd.read_csv("../ML-per-section/inputs.csv", header=0)

        # Split the dataset into training, validation, and testing household IDs based on the household IDs.
        training_households, validation_and_testing_households = train_test_split(self._df["Household ID"],
                                                                                  test_size=0.25,
                                                                                  random_state=42, shuffle=True)
        validation_households, testing_households = train_test_split(validation_and_testing_households, test_size=0.5,
                                                                     random_state=42, shuffle=True)

        self._active_households_list = None  # This will store the list of households for the current phase

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

        self._chosen_encryption_ratios = {}

        max_section_level_water_usage = self._df["Per Section Utility Usage"].max()
        min_section_level_water_usage = self._df["Per Section Utility Usage"].min()

        # Calculate max and min pre-encryption section-level entropy
        all_readings = np.array(self._df[
                                    "All Utility Readings in Section"]).flatten()  # shape: (819200,) > 80 households x 10 sections x 1024 readings.
        counts = Counter(all_readings)
        total = len(all_readings)
        assert len(self._df["All Utility Readings in Section"]) == 800

        prob = np.array([count / total for count in
                         counts.values()])  # Calculates the relative frequency of each reading in your dataset.
        self._max_global_entropy = -np.sum(prob * np.log2(prob))  # Shannon Entropy max is log2(n).
        self._min_global_entropy = 0.0

        self._max_section_level_water_usage = max_section_level_water_usage
        self._min_section_level_water_usage = min_section_level_water_usage

        # Observations are dictionaries that describe the current state of the environment the agent.
        self.observation_space = gym.spaces.Dict({
            "section_level_water_usage": gym.spaces.Box(low=self._min_section_level_water_usage,
                                                        high=self._max_section_level_water_usage,
                                                        shape=(1,), dtype=np.float64),
            "global_entropy": gym.spaces.Box(low=self._min_global_entropy,
                                             high=self._max_global_entropy, shape=(1,), dtype=np.float64),

        })

        entropy_scalar = preprocessing.MinMaxScaler()
        self._entropy_scalar = entropy_scalar.fit(all_readings)

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
            rounded_val = math.round(val * 1000) / 1000
            frequency[rounded_val] = frequency.get(rounded_val, 0) + 1

        total = float(len(data))
        entropy = 0.0
        for count in frequency.values():
            if count > 0:
                p = float(count) / total
                entropy -= p * math.log2(p)
        return entropy

    def _apply_encryption_ratio_and_quantisation(self, raw_section_data, enc_ratio):
        for i in range(len(raw_section_data)):
            raw_section_data[i] *= (1.0 - enc_ratio)

            # Apply quantisation (coarser rounding) based on the encryption ratio
            # To simulate the information loss that occurs during a privacy-preserving.
            precision = 0.0
            if enc_ratio <= 0.3:
                precision = 100.0  # Keep 2 decimal places for low ratios
            elif enc_ratio <= 0.7:
                precision = 10.0  # Round to 1 decimal place for medium ratios
            else:  # encRatio > 0.7
                precision = 1.0  # Round to 0 decimal place for high ratios

            if precision > 0:  # Avoid division by zero if precision is not set
                raw_section_data[i] = float(math.round(raw_section_data[i] * precision) / precision)
            else:
                raw_section_data[i] = math.round(raw_section_data[i])  # Default to nearest integer

        return raw_section_data

    # _get_observation() function provides the current state of the environment the agent is interacting with, i.e., independent of an encryption ratio.
    def _get_observation(self):

        current_household_id = self._active_households_list[self._current_household_idx]
        section_data_for_household = self._df[
            self._df["Household ID"] == current_household_id].sort_values(by="Section Number")

        current_section_row = section_data_for_household.iloc[self._current_section_idx_in_household]
        raw_section_utility_usage_scalar = current_section_row["Per Section Utility Usage"]
        raw_utility_readings_array = current_section_row["All Utility Readings in Section"]  # Get the actual array

        # Calculate raw entropy for the current section using the time-series array
        section_raw_entropy_val = self._calculate_entropy(raw_utility_readings_array)

        return {
            "section_utility_usage": np.array([raw_section_utility_usage_scalar], dtype=np.float64),
            "section_raw_entropy": np.array([section_raw_entropy_val], dtype=np.float64)
        }

    def _get_info(self):
        current_household_id = self._active_households_list[self._current_household_idx]
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

        self._chosen_encryption_ratios[(current_household_id, current_section_number)] = selected_encryption_ratio

        scaled_remaining_entropy = self._entropy_scalar.transform(pd.DataFrame([[section_remaining_entropy]]))[0][0]

        immediate_reward = scaled_remaining_entropy

        # Advance to next section.
        self._current_section_idx_in_household += 1
        terminated = False
        truncated = False

        # Check if all sections in current household are processed.
        if self._current_section_idx_in_household >= len(self._current_household_data):
            self._current_household_idx += 1  # Move to the next household
            self._current_section_idx_in_household = 0  # Reset section index

            # Check if all households in the active set are processed
            if self._current_household_idx >= len(self._active_households_list):
                terminated = True  # All households in this phase are processed
                self._household_ids_processed_in_phase.extend(self._active_households_list)
            else:
                # Load data for the next household
                next_household_id = self._active_households_list[self._current_household_idx]
                self._current_household_data = self._df[
                    self._df["Household ID"] == next_household_id].sort_values(by="Section Number")

        if not terminated:
            observation = self._get_obs()
        else:
            # If terminated, return a dummy observation for the last step.
            # StableBaselines3 expects an observation even if terminated.
            observation = {
                "section_utility_usage": np.array([0.0], dtype=np.float64),
                "section_raw_entropy": np.array([0.0], dtype=np.float64)
            }

        info_step_details = {
            "household_id": current_household_id,
            "section_number": current_section_number,
            "selected_encryption_ratio": selected_encryption_ratio,
            "section_raw_entropy": section_raw_entropy,
            "section_remaining_entropy": section_remaining_entropy,
            "immediate_reward": immediate_reward
        }

        return observation, immediate_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_household_idx = random.choice(
            self._active_households_list)  # NOTE: Will re-vist already seen households for training.
        self._current_household_idx = 0
        self._current_section_idx_in_household = 0
        self._household_ids_processed_in_phase = []

        current_household_id = self._active_households_list[self._current_household_idx]
        self._current_household_data = self._df[
            self._df["Household ID"] == current_household_id].sort_values(by="Section Number")

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def render(self):
        pass


def log_to_csv(writer, step_count, household_id, section_number, immediate_reward, info_step):
    writer.writerow([
        step_count,
        household_id,
        section_number,
        info_step.get('selected_encryption_ratio'),
        immediate_reward,
        info_step.get('section_raw_entropy'),
        info_step.get('section_remaining_entropy'),
    ])


class SectionLoggingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
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
            "Immediate Reward (Remaining Entropy)",
            "Section Raw Entropy",
            "Section Remaining Entropy",
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

        if self.n_calls % self.check_freq == 0:
            info_step = self.locals['infos'][0]

            household_id = info_step.get('household_id')
            section_number = info_step.get('section_number')

            if household_id is not None and section_number is not None:
                immediate_reward = self.locals['immediate_reward'][0]
                log_to_csv(self.writer, self.n_calls, household_id, section_number, immediate_reward, info_step)
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
    # ----- TRAINING PHASE ------
    env_train = EncryptionSelectorEnv(dataset_type="train")

    model = DQN(policy=MultiInputPolicy, env=env_train, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4,
                callback=SectionLoggingCallback(check_freq=1, log_path='training_log_sections.csv', verbose=0))

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
