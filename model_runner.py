# python ./model_runner.py

import subprocess

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


def run_electricity_per_household_test(attack_block_size):
    try:
        subprocess.run(["python", "./test_model_electricity/test_model_electricity.py", attack_block_size])
    except subprocess.CalledProcessError as e:
        print(f"ERROR: test_model_electricity.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found")
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


def run_electricity_per_household(attack_block_size):
    try:
        subprocess.run(["python",
                        "./ELECTRICITY_household_level_encryption_ratio_selector/household_level_encryption_ratio_selector.py",
                        attack_block_size])
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: ELECTRICITY_household_level_encryption_ratio_selector.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found")
        return None


def run_water_per_household(attack_block_size):
    try:
        subprocess.run(
            ["python", "./WATER_household_level_encryption_ratio_selector/household_level_encryption_ratio_selector.py",
             attack_block_size])
    except subprocess.CalledProcessError as e:
        print(f"ERROR: WATER_household_level_encryption_ratio_selector.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found.")
        return None


def run_electricity_per_block(attack_block_size):
    try:
        subprocess.run(
            ["python", "./ELECTRICITY_block_level_encryption_ratio_selector/block_level_encryption_ratio_selector.py",
             attack_block_size])
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: ELECTRICITY_block_level_encryption_ratio_selector.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found.")
        return None


def run_water_per_block(attack_block_size):
    try:
        subprocess.run(
            ["python", "./WATER_block_level_encryption_ratio_selector/block_level_encryption_ratio_selector.py",
             attack_block_size])
    except subprocess.CalledProcessError as e:
        print(f"ERROR: block_level_encryption_ratio_selector.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found.")
        return None


def run_electricity_per_block_with_policy(attack_block_size, policy_penalty):
    try:
        subprocess.run(["python",
                        "./ELECTRICITY_block_level_encryption_ratio_selector_with_policy/block_level_encryption_ratio_selector_with_policy.py",
                        attack_block_size, policy_penalty])
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: ELECTRICITY_block_level_encryption_ratio_selector_with_policy.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found.")
        return None


def run_water_per_block_with_policy(attack_block_size, policy_penalty):
    try:
        subprocess.run(["python",
                        "./WATER_block_level_encryption_ratio_selector_with_policy/block_level_encryption_ratio_selector_with_policy.py",
                        attack_block_size, policy_penalty])
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: WATER_block_level_encryption_ratio_selector_with_policy.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found.")
        return None


def main():
    attack_block_sizes = ["3", "6", "9", "12", "24", "36",
                          "48"]  # Where 12 refers to half a day's worth of utility readings exposed to the attacker
    policy_penalties = ["500", "600", "800"]

    for attack_block_size in attack_block_sizes:
        # run_electricity_per_household_test(attack_block_size=attack_block_size)
        # run_water_per_household_test(attack_block_size=attack_block_size)

        # run_electricity_per_household(attack_block_size=attack_block_size)
        # run_water_per_household(attack_block_size=attack_block_size)

        run_electricity_per_block(attack_block_size=attack_block_size)
        # run_water_per_block(attack_block_size=attack_block_size)

        # for policy_penalty in policy_penalties:
        #     run_electricity_per_block_with_policy(attack_block_size=attack_block_size, policy_penalty=policy_penalty)
        #     run_water_per_block_with_policy(attack_block_size=attack_block_size, policy_penalty=policy_penalty)


if __name__ == "__main__":
    main()
