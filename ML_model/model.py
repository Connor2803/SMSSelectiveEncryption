from tarfile import data_filter
import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random


def main():
    env_train = EncryptionSelectorEnv(dataset_type="train")
    model = DQN("MlpPolicy", env_train, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("dqn_encryption_selector")

    del model

    model = DQN.load("dqn_encryption_selector")

    obs, info = env_train.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_train.step(action)
        if terminated or truncated:
            obs, info = env_train.reset()


class EncryptionSelectorEnv(gym.Env):
    def __init__(self, dataset_type="train"):
        super().__init__()

        self._encryption_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Read the ML metrics CSV file for Water dataset.
        df = pd.read_csv("../ML_database_generation/ML_metrics_WATER.csv", header=0)

        # Retrieve the unique household IDs from the Water dataset.
        unique_household_IDs = df["filename"].unique()

        # Split the dataset into training, validation, and testing household IDs based on the unique household IDs to keep each household data in tact.
        training_households, validation_and_testing_households = train_test_split(unique_household_IDs, test_size=0.6,
                                                                                  random_state=42, shuffle=True)
        validation_households, testing_households = train_test_split(validation_and_testing_households, test_size=0.5,
                                                                     random_state=42, shuffle=True)

        self._active_households = None # This will store the list of households for the current phase

        # Populate training, validation, and testing dataframes based on the split household IDs.
        training_df = df[df["filename"].isin(training_households)]
        validation_df = df[df["filename"].isin(validation_households)]
        testing_df = df[df["filename"].isin(testing_households)]

        # Assign dataframes and household IDs as instance attributes.
        self._df = df
        self._unique_households = unique_household_IDs
        self._training_df = training_df
        self._validation_df = validation_df
        self._testing_df = testing_df
        self._training_households = training_households
        self._validation_households = validation_households
        self._testing_households = testing_households

        self._active_households = None  # This will store the list of households for the current phase

        if dataset_type == 'train':
            self._active_households = self._training_households
        elif dataset_type == 'validation':
            self._active_households = self._validation_households
        elif dataset_type == 'test':
            self._active_households = self._testing_households
        else:
            raise ValueError("dataset_type must be 'train', 'validation', or 'test'")

        # Remove duplicate entries in ML metrics CSV file for Water dataset.
        df = df.drop_duplicates(subset=["filename", "section"])

        max_household_water_usage_summation = df.groupby("filename")["section_sum_usage"].sum().max()
        min_household_water_usage_summation = df.groupby("filename")["section_sum_usage"].sum().min()
        max_household_raw_entropy = df.groupby("filename")["section_raw_entropy"].sum().max()
        min_household_raw_entropy = df.groupby("filename")["section_raw_entropy"].sum().min()

        self._max_household_water_usage_summation = max_household_water_usage_summation
        self._min_household_water_usage_summation = min_household_water_usage_summation
        self._max_household_raw_entropy = max_household_raw_entropy
        self._min_household_raw_entropy = min_household_raw_entropy

        # Observations are dictionaries that describe the current state of the environment the agent.
        self.observation_space = gym.spaces.Dict({
            # Represents the range of summed household water utility usage values across all sections.
            # I.e., sum(section_sum_usage) for a household.
            "household_water_usage_summation": gym.spaces.Box(low=self._min_household_water_usage_summation,
                                                              high=self._max_household_water_usage_summation,
                                                              shape=(1,), dtype=np.float64),
            # Represents the entropy of the raw household water utility entropy values across all sections.
            # I.e., sum(section_raw_entropy) for a household.
            "household_raw_entropy": gym.spaces.Box(low=self._min_household_raw_entropy,
                                                    high=self._max_household_raw_entropy, shape=(1,), dtype=np.float64),
        })

        asr_duration_scaler = preprocessing.MinMaxScaler()
        self._asr_duration_scalar = asr_duration_scaler.fit(self._df[["asr_attack_duration"]])

        asr_mean_scaler = preprocessing.MinMaxScaler()
        self._asr_mean_scalar = asr_mean_scaler.fit(self._df[["asr_mean"]])

        remaining_entropy_scaler = preprocessing.MinMaxScaler()
        self._remaining_entropy_scalar = remaining_entropy_scaler.fit(self._df[["section_remaining_entropy"]])

        memory_scaler = preprocessing.MinMaxScaler()
        self._memory_scalar = memory_scaler.fit(self._df[["allocated_memory_MiB"]])

        # Actions are discrete integers that describe the action to be taken by the agent.
        # We have 9 discrete actions, corresponding to the 9 encryption ratios.
        self.action_space = gym.spaces.Discrete(9)

    # _get_observation() function provides the current state of the environment the agent is interacting with, i.e., independent of an encryption ratio.
    def _get_observation(self):

        household_data = self._df[self._df["filename"] == self._current_household_ID]

        unique_sections_data = household_data.drop_duplicates(subset=["section"])

        sum_usage_for_current_household = unique_sections_data["section_sum_usage"].sum()

        sum_raw_entropy_for_current_household = unique_sections_data["section_raw_entropy"].sum()

        return {
            "household_water_usage_summation": np.array([sum_usage_for_current_household], dtype=np.float64),
            "household_raw_entropy": np.array([sum_raw_entropy_for_current_household], dtype=np.float64)
        }

    def _get_info(self):
        return {}

    def step(self, action):
        # Map the action to the encryption ratio chosen by the agent.
        encryption_ratio = self._encryption_ratios[action]

        metrics_row = self._df[
            (self._df["filename"] == self._current_household_ID) &
            (self._df["encryption_ratio"] == encryption_ratio)
        ]

        current_asr_attack_duration = metrics_row["asr_attack_duration"].mean()
        current_asr_mean = metrics_row["asr_mean"].mean()
        current_remaining_entropy = metrics_row["remaining_entropy"].sum()
        current_memory = metrics_row["allocated_memory_MiB"].mean()

        scaled_current_asr_attack_duration = self._asr_duration_scalar.transform(np.array([[current_asr_attack_duration]]))[0][0]
        scaled_current_asr_mean = self._asr_mean_scalar.transform(np.array([[current_asr_mean]]))[0][0]
        scaled_current_remaining_entropy = self._remaining_entropy_scalar.transform(np.array([[current_remaining_entropy]]))[0][0]
        scaled_current_memory = self._memory_scalar.transform(np.array([[current_memory]]))[0][0]

        observation = self._get_observation()
        info = self._get_info()

        w1, w2, w3, w4 = 1, 1, 1, 1 # TODO: Fine tune later!

        reward = w1 * scaled_current_asr_attack_duration - w2 * scaled_current_asr_mean + w3 * scaled_current_remaining_entropy - w4 * scaled_current_memory

        terminated = True  # As this is a single-step episode.
        truncated = False

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_household_id = random.choice(self._active_households) # NOTE: Will re-vist already seen households for training.

        return self._get_observation(), self._get_info()

    def render(self):
        pass


if __name__ == "__main__":
    main()
