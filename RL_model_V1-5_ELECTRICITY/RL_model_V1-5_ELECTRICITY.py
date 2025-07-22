# python ./RL_model_V1-5_ELECTRICITY/RL_model_V1-5_ELECTRICITY.py
import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.dqn.policies import MultiInputPolicy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import time
import csv
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXECUTABLE_NAME = "generate_metrics_V1-5"
if sys.platform == "win32":
    EXECUTABLE_NAME += ".exe"
GO_SOURCE_PATH = os.path.join(SCRIPT_DIR, "generate_metrics_V1-5.go")
GO_EXECUTABLE_PATH = os.path.join(SCRIPT_DIR, EXECUTABLE_NAME)
print(f"\nGo executable path: {GO_EXECUTABLE_PATH}")
print(f"\nGo source path: {GO_SOURCE_PATH}")

class EncryptionSelectorEnv(gym.Env):
    def __init__(self, dataset_type="train"):
        super().__init__()

        self._encryption_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Read the ML metrics CSV file for Electricity dataset.
        df = pd.read_csv("./RL_model_V1-5_ELECTRICITY/ML_metrics_ELECTRICITY_v1-5.csv", header=0)
        df_HE = pd.read_csv("./RL_model_V1-5_ELECTRICITY/ML_party_metrics_ELECTRICITY_v1-5.csv", header=0)

        # Retrieve the unique household IDs from the Electricity dataset.
        electricity_households_data_folder_path = './examples/datasets/electricity/households_10240'
        try:
            folder_filenames_raw = os.listdir(electricity_households_data_folder_path)
            folder_filenames_sorted = sorted(folder_filenames_raw)  # Sorted alphabetically.
        except FileNotFoundError:
            print(f"Error: Folder not found at {electricity_households_data_folder_path}. Please check the path.")
            folder_filenames_sorted = []

        unique_household_IDs_from_df = df["filename"].unique()
        ordered_unique_household_IDs = [
            filename for filename in folder_filenames_sorted if filename in unique_household_IDs_from_df
        ]
        unique_household_IDs = ordered_unique_household_IDs

        # Create permanent testing household subset for comparative performance analysis.
        permanent_testing_IDs = unique_household_IDs[-10:]
        unique_household_IDs = unique_household_IDs[:-10]

        # Split the dataset into training, validation, and testing household IDs based on the unique household IDs to keep each household data intact.
        training_households, validation_households = train_test_split(unique_household_IDs, test_size=(10/70),
                                                                      random_state=42, shuffle=True)

        self._active_households = None  # This will store the list of households for the current phase

        # Populate training, validation, and testing dataframes based on the split household IDs.
        training_df = df[df["filename"].isin(training_households)]
        training_df_HE = df_HE[df_HE["filename"].isin(training_households)]
        validation_df = df[df["filename"].isin(validation_households)]
        validation_df_HE = df_HE[df_HE["filename"].isin(validation_households)]
        testing_df = df[df["filename"].isin(permanent_testing_IDs)]
        testing_df_HE = df_HE[df_HE["filename"].isin(permanent_testing_IDs)]

        # Assign dataframes and household IDs as instance attributes.
        self._df = df
        self._df_HE = df_HE
        self._unique_households = unique_household_IDs
        self._training_df = training_df
        self._training_df_HE = training_df_HE
        self._validation_df = validation_df
        self._validation_df_HE = validation_df_HE
        self._testing_df = testing_df
        self._testing_df_HE = testing_df_HE
        self._training_households = training_households
        self._validation_households = validation_households
        self._testing_households = permanent_testing_IDs

        self._active_households = None  # This will store the list of households for the current phase.
        self.dataset_type = dataset_type

        if dataset_type == 'train':
            self._active_households = self._training_households
        elif dataset_type == 'validation':
            self._active_households = self._validation_households
        elif dataset_type == 'test':
            self._active_households = self._testing_households
        else:
            raise ValueError("dataset_type must be 'train', 'validation', or 'test'")

        # Remove duplicate entries in ML metrics CSV file for Electricity dataset.
        df = df.drop_duplicates(subset=["filename", "section"])

        max_household_electricity_usage_summation = df.groupby("filename")["section_sum_usage"].sum().max()
        min_household_electricity_usage_summation = df.groupby("filename")["section_sum_usage"].sum().min()
        max_household_raw_entropy = df.groupby("filename")["section_raw_entropy"].sum().max()
        min_household_raw_entropy = df.groupby("filename")["section_raw_entropy"].sum().min()

        self._max_household_electricity_usage_summation = max_household_electricity_usage_summation
        self._min_household_electricity_usage_summation = min_household_electricity_usage_summation
        self._max_household_raw_entropy = max_household_raw_entropy
        self._min_household_raw_entropy = min_household_raw_entropy

        # Observations are dictionaries that describe the current state of the environment the agent.
        self.observation_space = gym.spaces.Dict({
            # Represents the range of summed household electricity utility usage values across all sections.
            # I.e., sum(section_sum_usage) for a household.
            "household_electricity_usage_summation": gym.spaces.Box(low=self._min_household_electricity_usage_summation,
                                                              high=self._max_household_electricity_usage_summation,
                                                              shape=(1,), dtype=np.float64),
            # Represents the entropy of the raw household electricity utility entropy values across all sections.
            # I.e., sum(section_raw_entropy) for a household.
            "household_raw_entropy": gym.spaces.Box(low=self._min_household_raw_entropy,
                                                    high=self._max_household_raw_entropy, shape=(1,), dtype=np.float64),
        })

        reidentification_duration_scaler = preprocessing.MinMaxScaler()
        self._reidentification_duration_scalar = reidentification_duration_scaler.fit(self._df[["reidentification_attack_duration"]])

        reidentification_mean_scaler = preprocessing.MinMaxScaler() 
        self._reidentification_mean_scalar = reidentification_mean_scaler.fit(self._df[["reidentification_mean"]])

        remaining_entropy_scaler = preprocessing.MinMaxScaler()
        self._remaining_entropy_scalar = remaining_entropy_scaler.fit(
            self._df.groupby("filename")["section_remaining_entropy"].sum().to_frame())

        memory_scaler = preprocessing.MinMaxScaler()
        self._memory_scalar = memory_scaler.fit(self._df[["allocated_memory_MiB"]])

        summation_error_scaler = preprocessing.MinMaxScaler()
        self._summation_error_scaler = summation_error_scaler.fit(abs(self._df_HE[["summation_error"]]))

        deviation_error_scaler = preprocessing.MinMaxScaler()
        self._deviation_error_scaler = deviation_error_scaler.fit(abs(self._df_HE[["deviation_error"]]))

        encryption_time_scaler = preprocessing.MinMaxScaler()
        self._encryption_time_scaler = encryption_time_scaler.fit(self._df_HE[["encryption_time_ns"]])

        decryption_time_scaler = preprocessing.MinMaxScaler()
        self._decryption_time_scaler = decryption_time_scaler.fit(self._df_HE[["decryption_time_ns"]])

        summation_operations_time_scaler = preprocessing.MinMaxScaler()
        self._summation_operations_time_scaler = summation_operations_time_scaler.fit(
            self._df_HE[["summation_operations_time_ns"]])

        deviation_operations_time_scaler = preprocessing.MinMaxScaler()
        self._deviation_operations_time_scaler = deviation_operations_time_scaler.fit(
            self._df_HE[["deviation_operations_time_ns"]])

        # Actions are discrete integers that describe the action to be taken by the agent.
        # We have 9 discrete actions, corresponding to the 9 encryption ratios.
        self.action_space = gym.spaces.Discrete(9)

        self._current_household_ID = None

    # _get_observation() function provides the current state of the environment the agent is interacting with, i.e., independent of an encryption ratio.
    def _get_observation(self):
        if self._current_household_ID is None:
            return {
                "household_water_usage_summation": np.zeros(1, dtype=np.float64),
                "household_raw_entropy": np.zeros(1, dtype=np.float64)
            }

        household_data = self._df[self._df["filename"] == self._current_household_ID]
        unique_sections_data = household_data.drop_duplicates(subset=["section"])
        sum_usage_for_current_household = unique_sections_data["section_sum_usage"].sum()
        sum_raw_entropy_for_current_household = unique_sections_data["section_raw_entropy"].sum()

        return {
            "household_electricity_usage_summation": np.array([sum_usage_for_current_household], dtype=np.float64),
            "household_raw_entropy": np.array([sum_raw_entropy_for_current_household], dtype=np.float64)
        }

    def _get_info(self):
        return {}

    def step(self, action):
        # Map the action to the encryption ratio chosen by the agent.
        encryption_ratio = self._encryption_ratios[action]

        # Select reward parameters based on the chosen encryption ratio and household ID.
        metrics_row = self._df[
            (self._df["filename"] == self._current_household_ID) &
            (self._df["encryption_ratio"] == encryption_ratio)
            ]

        party_row = self._df_HE[
            (self._df_HE["filename"] == self._current_household_ID) &
            (self._df_HE["encryption_ratio"] == encryption_ratio)
            ]

        current_reidentification_attack_duration = metrics_row["reidentification_attack_duration"].mean() 
        current_reidentification_mean = metrics_row["reidentification_mean"].mean()
        current_remaining_entropy = metrics_row["section_remaining_entropy"].sum()
        current_memory = metrics_row["allocated_memory_MiB"].mean()

        current_summation_error = party_row["summation_error"].iloc[0]
        current_deviation_error = party_row["deviation_error"].iloc[0]
        current_encryption_time = party_row["encryption_time_ns"].iloc[0]
        current_decryption_time = party_row["decryption_time_ns"].iloc[0]
        current_summation_operations_time = party_row["summation_operations_time_ns"].iloc[0]
        current_deviation_operations_time = party_row["deviation_operations_time_ns"].iloc[0]

        # Scale the reward parameters using pre-established MinMaxScaler.
        scaled_current_reidentification_attack_duration = \
            self._reidentification_duration_scalar.transform(
                pd.DataFrame([[current_reidentification_attack_duration]], columns=["reidentification_attack_duration"]))[0][0]
        scaled_current_reidentification_mean = \
        self._reidentification_mean_scalar.transform(pd.DataFrame([[current_reidentification_mean]], columns=["reidentification_mean"]))[0][0]
        scaled_current_remaining_entropy = \
            self._remaining_entropy_scalar.transform(
                pd.DataFrame([[current_remaining_entropy]], columns=["section_remaining_entropy"]))[0][0]
        scaled_current_memory = \
        self._memory_scalar.transform(pd.DataFrame([[current_memory]], columns=["allocated_memory_MiB"]))[0][0]

        scaled_current_summation_error = \
            self._summation_error_scaler.transform(
                pd.DataFrame([[abs(current_summation_error)]], columns=["summation_error"]))[0][0]
        scaled_current_deviation_error = \
            self._deviation_error_scaler.transform(
                pd.DataFrame([[abs(current_deviation_error)]], columns=["deviation_error"]))[0][0]
        scaled_current_encryption_time = \
            self._encryption_time_scaler.transform(
                pd.DataFrame([[current_encryption_time]], columns=["encryption_time_ns"]))[0][0]
        scaled_current_decryption_time = \
            self._decryption_time_scaler.transform(
                pd.DataFrame([[current_decryption_time]], columns=["decryption_time_ns"]))[0][0]
        scaled_current_summation_operations_time = \
            self._summation_operations_time_scaler.transform(
                pd.DataFrame([[current_summation_operations_time]], columns=["summation_operations_time_ns"]))[0][0]
        scaled_current_deviation_operations_time = \
            self._deviation_operations_time_scaler.transform(
                pd.DataFrame([[current_deviation_operations_time]], columns=["deviation_operations_time_ns"]))[0][0]

        observation = self._get_observation()
        info = {
            "household_id": self._current_household_ID,
            "selected_encryption_ratio": encryption_ratio,
            "scaled_reidentification_attack_duration": scaled_current_reidentification_attack_duration,
            "scaled_reidentification_mean": scaled_current_reidentification_mean,
            "scaled_remaining_entropy": scaled_current_remaining_entropy,
            "scaled_memory": scaled_current_memory,

            "scaled_summation_error": scaled_current_summation_error,
            "scaled_deviation_error": scaled_current_deviation_error,
            "scaled_encryption_time": scaled_current_encryption_time,
            "scaled_decryption_time": scaled_current_decryption_time,
            "scaled_summation_operations_time": scaled_current_summation_operations_time,
            "scaled_deviation_operations_time": scaled_current_deviation_operations_time,

            "raw_reidentification_attack_duration": current_reidentification_attack_duration,
            "raw_reidentification_mean": scaled_current_reidentification_mean,
            "raw_remaining_entropy": current_remaining_entropy,
            "raw_memory": current_memory,
            "raw_summation_error": current_summation_error,
            "raw_deviation_error": current_deviation_error,
            "raw_encryption_time": current_encryption_time,
            "raw_decryption_time": current_decryption_time,
            "raw_summation_operations_time": current_summation_operations_time,
            "raw_deviation_operations_time": current_deviation_operations_time,
        }

        # Positive contribution: reidentification_attack_duration, decryption_time
        # Negative contribution: scaled_current_reidentification_mean, scaled_current_remaining_entropy, scaled_current_memory, summation_error, deviation_error, encryption_time, summation_operations_time, deviation_operations_time

        reward = scaled_current_reidentification_attack_duration + scaled_current_decryption_time - scaled_current_reidentification_mean - scaled_current_remaining_entropy - scaled_current_memory - scaled_current_summation_error - scaled_current_deviation_error - scaled_current_encryption_time - scaled_current_summation_operations_time - scaled_current_deviation_operations_time

        terminated = True  # As this is a single-step episode.
        truncated = False  # As this is a single-step episode.

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.dataset_type == 'train':
            self._current_household_ID = random.choice(self._active_households)
        elif self._current_household_ID is None:
            self._current_household_ID = self._active_households[0]

        return self._get_observation(), self._get_info()

    def _set_household(self, household_id):
        """
        Manually sets the current household ID for the environment.
        Used for sequential evaluation during testing/validation.
        """
        self._current_household_ID = household_id
        return self._get_observation()

    def render(self):
        pass


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, csv_writer, verbose: int = 0):
        super().__init__(verbose)
        self.csv_writer = csv_writer
        self._episode_num = 0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        # self.locals contains variables accessible in the current step.
        # self.locals['dones'] is a boolean list (for vectorized envs) indicating if an env is done, i.e., the outcome of each step: truncated or terminated.
        # self.locals['infos'] is a list of info dictionaries, one per env.
        # self.locals['rewards'] is a list of rewards for the current step.

        for idx, done in enumerate(self.locals['dones']):
            if done:
                self._episode_num += 1

                info = self.locals['infos'][idx]

                total_reward = self.locals['rewards'][idx]

                household_id = info.get("household_id", "N/A")
                selected_encryption_ratio = info.get("selected_encryption_ratio", "N/A")
                scaled_reidentification_attack_duration = info.get("scaled_reidentification_attack_duration", 0)
                scaled_reidentification_mean = info.get("scaled_reidentification_mean", 0)
                scaled_remaining_entropy = info.get("scaled_remaining_entropy", 0)
                scaled_memory = info.get("scaled_memory", 0)

                scaled_summation_error = info.get("scaled_summation_error", 0)
                scaled_deviation_error = info.get("scaled_deviation_error", 0)
                scaled_encryption_time = info.get("scaled_encryption_time", 0)
                scaled_decryption_time = info.get("scaled_decryption_time", 0)
                scaled_summation_operations_time = info.get("scaled_summation_operations_time", 0)
                scaled_deviation_operations_time = info.get("scaled_deviation_operations_time", 0)

                raw_reidentification_attack_duration = info.get("raw_reidentification_attack_duration", 0)
                raw_reidentification_mean = info.get("raw_reidentification_mean", 0)
                raw_remaining_entropy = info.get("raw_remaining_entropy", 0)
                raw_memory = info.get("raw_memory", 0)
                raw_summation_error = info.get("raw_summation_error", 0)
                raw_deviation_error = info.get("raw_deviation_error", 0)
                raw_summation_operations_time = info.get("raw_summation_operations_time", 0)
                raw_deviation_operations_time = info.get("raw_deviation_operations_time", 0)

                # Write the row to the CSV file
                self.csv_writer.writerow([
                    self._episode_num,
                    household_id,
                    total_reward,

                    # Calculated from section-level statistics.
                    selected_encryption_ratio,
                    scaled_reidentification_attack_duration,
                    scaled_reidentification_mean,
                    scaled_remaining_entropy,
                    scaled_memory,

                    # Per-party level statistics.
                    scaled_summation_error,
                    scaled_deviation_error,
                    scaled_encryption_time,
                    scaled_decryption_time,
                    scaled_summation_operations_time,
                    scaled_deviation_operations_time,

                    # Raw original values.
                    raw_reidentification_attack_duration,
                    raw_reidentification_mean,
                    raw_remaining_entropy,
                    raw_memory,
                    raw_summation_error,
                    raw_deviation_error,
                    raw_summation_operations_time,
                    raw_deviation_operations_time,

                ])
                if self.verbose > 0:
                    print(
                        f"Logged episode {self._episode_num} for Household {household_id} with reward {total_reward:.4f}")

        return True

def log_to_csv(writer, episode_num, household_id, reward, info):
    """Helper function to write a row to the CSV log file."""
    writer.writerow([
        episode_num,
        household_id,
        reward,
        info.get("selected_encryption_ratio", "N/A"),
        info.get("scaled_reidentification_attack_duration", 0),
        info.get("scaled_reidentification_mean", 0),
        info.get("scaled_remaining_entropy", 0),
        info.get("scaled_memory", 0),
        info.get("scaled_summation_error", 0),
        info.get("scaled_deviation_error", 0),
        info.get("scaled_encryption_time", 0),
        info.get("scaled_decryption_time", 0),
        info.get("scaled_summation_operations_time", 0),
        info.get("scaled_deviation_operations_time", 0),
        info.get("raw_reidentification_attack_duration", 0),
        info.get("raw_reidentification_mean", 0),
        info.get("raw_remaining_entropy", 0),
        info.get("raw_memory", 0),
        info.get("raw_summation_error", 0),
        info.get("raw_deviation_error", 0),
        info.get("raw_encryption_time", 0),
        info.get("raw_decryption_time", 0),
        info.get("raw_summation_operations_time", 0),
        info.get("raw_deviation_operations_time", 0),
    ])

def main():
    if len(sys.argv) != 2:
        print("WARNING: Not enough arguments provided! Please provide the atdSize.")
        currentAtdSize = "12"
    else:
        currentAtdSize = sys.argv[1]

    try:
        subprocess.run(["go", "build", "-o", GO_EXECUTABLE_PATH, GO_SOURCE_PATH], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to compile Go program: {e}")
        return

    if not os.path.exists(GO_EXECUTABLE_PATH):
        raise FileNotFoundError(f"Go executable not found at: {GO_EXECUTABLE_PATH}")

    print(f"Running Go metrics generator with atdSize = {currentAtdSize}...")
    try:
        run_args = [GO_EXECUTABLE_PATH, "2", currentAtdSize]
        subprocess.run(run_args, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run Go program: {e}")
        print(f"Stderr: {e.stderr}")
        return

    # ----- TRAINING PHASE ------
    env_train = EncryptionSelectorEnv(dataset_type="train")

    # Training file log creation
    log_file_name = "./RL_model_V1-5_ELECTRICITY/V1-5_training_log_ELECTRICITY.csv"
    csv_file = None
    csv_writer = None

    try:
        csv_file = open(log_file_name, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["Episode",
             "HouseholdID",
             "Total Reward",
             "Selected Encryption Ratio",
             "Scaled Average Reidentification Attack Duration",
             "Scaled Average Reidentification Mean",
             "Scaled Sum Remaining Entropy",
             "Scaled Average Memory MiB",
             "Scaled Summation Error",
             "Scaled Deviation Error",
             "Scaled Encryption Time",
             "Scaled Decryption Time",
             "Scaled Summation Operations Time",
             "Scaled Deviation Operations Time",
             "Average Reidentification Duration",
             "Average Reidentification Mean",
             "Sum Remaining Entropy",
             "Average Memory MiB",
             "Summation Error",
             "Deviation Error",
             "Encryption Time",
             "Decryption Time",
             "Summation Operations Time",
             "Deviation Operations Time",
             ])

        start_time = time.time()
        print(f"Training started at: {time.ctime(start_time)}")

        model = DQN(policy=MultiInputPolicy, env=env_train, verbose=1)
        model.learn(total_timesteps=1000, log_interval=4, callback=CustomCallback(csv_writer, verbose=0))
        model.save("./RL_model_V1-5_ELECTRICITY/DQN_Encryption_Ratio_Selector_V1-5")

        end_time = time.time()
        print(f"Training finished at: {time.ctime(end_time)}")
        elapsed_time = end_time - start_time
        print(f"Total training duration: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        if csv_file:
            csv_file.close()
            print(f"Training log saved to {log_file_name}")

    del model

    log_headers = ["Episode",
             "HouseholdID",
             "Total Reward",
             "Selected Encryption Ratio",
             "Scaled Average Reidentification Attack Duration",
             "Scaled Average Reidentification Mean",
             "Scaled Sum Remaining Entropy",
             "Scaled Average Memory MiB",
             "Scaled Summation Error",
             "Scaled Deviation Error",
             "Scaled Encryption Time",
             "Scaled Decryption Time",
             "Scaled Summation Operations Time",
             "Scaled Deviation Operations Time",
             "Average Reidentification Duration",
             "Average Reidentification Mean",
             "Sum Remaining Entropy",
             "Average Memory MiB",
             "Summation Error",
             "Deviation Error",
             "Encryption Time",
             "Decryption Time",
             "Summation Operations Time",
             "Deviation Operations Time",
             ]

    # ----- VALIDATION PHASE ------
    print("\n--- Starting Validation ---")
    model = DQN.load("./RL_model_V1-5_ELECTRICITY/DQN_Encryption_Ratio_Selector_V1-5")
    env_val = EncryptionSelectorEnv(dataset_type="validation")
    env_val.reset()
    model.set_env(env_val)

    mean_reward, std_reward = evaluate_policy(model, env_val, render=False)
    print(f"Validation mean reward: {mean_reward:.2f} +- {std_reward:.2f}")

    # num_validation_households = len(env_val._active_households)
    #
    # with open('V!_validation_log_ELECTRICITY.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(log_headers)
    #
    #     obs, info = env_val.reset()
    #     for i in range(num_validation_households):
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, terminated, truncated, info_step = env_val.step(action)
    #
    #         household_id = info_step.get('household_id')
    #         print(
    #             f"Validated Household: {household_id}, Chosen Ratio: {info_step.get('selected_encryption_ratio')}, Reward: {reward:.4f}")
    #
    #         log_to_csv(writer, i + 1, household_id, reward, info_step)
    #
    #         if terminated or truncated:
    #             obs, info = env_val.reset()

    # ----- TESTING PHASE ------
    print("\n--- Starting Testing ---")
    env_test = EncryptionSelectorEnv(dataset_type="test")
    testing_households = env_test._active_households.copy()

    with open('./RL_model_V1-5_ELECTRICITY/V1-5_testing_log_ELECTRICITY.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(log_headers)

        obs, info = env_test.reset()
        for i, household_id in enumerate(testing_households):
            obs = env_test._set_household(household_id)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info_step = env_test.step(action)

            print(
                f"Tested Household: {household_id}, Chosen Ratio: {info_step.get('selected_encryption_ratio')}, Reward: {reward:.4f}")

            log_to_csv(writer, i + 1, household_id, reward, info_step)

if __name__ == "__main__":
    main()
