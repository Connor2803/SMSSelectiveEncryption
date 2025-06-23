from tarfile import data_filter

import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random


def main():
    env = EncryptionSelectorEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
    env.close()


class EncryptionSelectorEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Read the ML metrics CSV file for Water dataset.
        df = pd.read_csv("../ML_database_generation/ML_metrics_WATER.csv", header=0

        # Retrieve the unique household IDs from the Water dataset.
        unique_households = df["filename"].unique()

        # Split the dataset into training, validation, and testing household IDs based on the unique household IDs to keep each household data in tact.
        training_households, validation_and_testing_households = train_test_split(unique_households, test_size=0.6,
                                                                                  random_state=42, shuffle=True)
        validation_households, testing_households = train_test_split(validation_and_testing_households, test_size=0.5,
                                                                     random_state=42, shuffle=True)

        # Populate training, validation, and testing dataframes based on the split household IDs.
        training_df = df[df["filename"].isin(training_households)]
        validation_df = df[df["filename"].isin(validation_households)]
        testing_df = df[df["filename"].isin(testing_households)]

        # Assign dataframes and household IDs as instance attributes.
        self._df = df
        self._unique_households = unique_households
        self._training_df = training_df
        self._validation_df = validation_df
        self._testing_df = testing_df
        self._training_households = training_households
        self._validation_households = validation_households
        self._testing_households = testing_households

        max_household_water_usage_summation = 0
        min_household_water_usage_summation = 1000000
        max_household_raw_entropy = 0
        min_household_raw_entropy = 1000000

        for unique_household in df["filename"].unique():
            current_utility_sum = df[df["filename"] == unique_household]["section_sum_usage"].sum()
        current_raw_entropy = [df["filename"] == unique_household]["section_raw_entropy"].sum()

        if current_utility_sum > max_household_water_usage_summation:
            max_household_water_usage_summation = current_utility_sum
        if current_utility_sum < min_household_water_usage_summation:
            min_household_water_usage_summation = current_utility_sum

        if current_raw_entropy > max_household_raw_entropy:
            max_household_raw_entropy = current_raw_entropy
        if current_raw_entropy < min_household_raw_entropy:
            min_household_raw_entropy = current_raw_entropy

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

        # Actions are discrete integers that describe the action to be taken by the agent.
        # We have 9 discrete actions, corresponding to the 9 encryption ratios.
        self.action_space = gym.spaces.Discrete(9)

    # _get_observation() function provides the current state of the environment the agent is interacting with, i.e., independent of an encryption ratio.
    def _get_observation(self):

        household_data = self._df[self._df["filename"] == self._current_household_id]

        unique_sections_data = household_data.drop_duplicates(subset=["section"])

        sum_usage_for_current_household = unique_sections_data["section_sum_usage"].sum()

        sum_raw_entropy_for_current_household = unique_sections_data["section_raw_entropy"].sum()

        return {
            "household_water_usage_summation": np.array([sum_usage_for_current_household], dtype=np.float64),
            "household_raw_entropy": np.array([sum_raw_entropy_for_current_household], dtype=np.float64)
        }

    def step(self, action):
        # Map the action to the encryption ratio chosen by the agent.
        encryption_ratio = self._encryption_ratios[action]

        relevant_rows = self._df [
            self._df[["filename"] == self._current_household_id] &
            self._df[["encryption_ratio"] == encryption_ratio]
        ]

        duration = relevant_rows[["asr_attack_duration"]][0] # Measured in seconds.
        asr = relevant_rows[["asr_mean"]]
        remaining_entropy = relevant_rows[["section_remaining_entropy"]].sum()
        memory = relevant_rows[["allocated_memory_MiB"]]

        observation = self._get_observation()

        reward = w1 * duration - w2 * asr + w3 * remaining_entropy - w4 * memory

        terminated = True  # As this is a single-step episode.
        truncated = False

        return observation, reward, terminated, truncated

    def reset(self):
        self._current_household_id = self._training_households[0]
        self._encryption_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        return self._get_observation()


if __name__ == "__main__":
    main()
