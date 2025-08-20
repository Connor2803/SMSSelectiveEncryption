# python ./model_runner.py

import subprocess

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

        # run_electricity_per_block(attack_block_size=attack_block_size)
        run_water_per_block(attack_block_size=attack_block_size)

        # for policy_penalty in policy_penalties:
        #     run_electricity_per_block_with_policy(attack_block_size=attack_block_size, policy_penalty=policy_penalty)
        #     run_water_per_block_with_policy(attack_block_size=attack_block_size, policy_penalty=policy_penalty)


if __name__ == "__main__":
    main()
