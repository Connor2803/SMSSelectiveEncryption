# python ./model_runner.py

import subprocess

def run_electricity_per_household(leaked_plaintext_size, phase_type):
    try:
        subprocess.run(["python",
                        "./ELECTRICITY_household_level_encryption_ratio_selector/household_level_encryption_ratio_selector.py",
                        leaked_plaintext_size, phase_type])
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: ELECTRICITY_household_level_encryption_ratio_selector.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found")
        return None


def run_water_per_household(leaked_plaintext_size, phase_type):
    try:
        subprocess.run(
            ["python", "./WATER_household_level_encryption_ratio_selector/household_level_encryption_ratio_selector.py",
             leaked_plaintext_size, phase_type])
    except subprocess.CalledProcessError as e:
        print(f"ERROR: WATER_household_level_encryption_ratio_selector.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found.")
        return None


def run_electricity_per_block(leaked_plaintext_size, phase_type):
    try:
        subprocess.run(
            ["python", "./ELECTRICITY_block_level_encryption_ratio_selector/block_level_encryption_ratio_selector.py",
             leaked_plaintext_size, phase_type])
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: ELECTRICITY_block_level_encryption_ratio_selector.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found.")
        return None


def run_water_per_block(leaked_plaintext_size, phase_type):
    try:
        subprocess.run(
            ["python", "./WATER_block_level_encryption_ratio_selector/block_level_encryption_ratio_selector.py",
             leaked_plaintext_size, phase_type])
    except subprocess.CalledProcessError as e:
        print(f"ERROR: block_level_encryption_ratio_selector.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found.")
        return None


def run_electricity_per_block_with_policy(leaked_plaintext_size, policy_penalty, phase_type):
    try:
        subprocess.run(["python",
                        "./ELECTRICITY_block_level_encryption_ratio_selector_with_policy/block_level_encryption_ratio_selector_with_policy.py",
                        leaked_plaintext_size, policy_penalty, phase_type])
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: ELECTRICITY_block_level_encryption_ratio_selector_with_policy.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found.")
        return None


def run_water_per_block_with_policy(leaked_plaintext_size, policy_penalty, phase_type):
    try:
        subprocess.run(["python",
                        "./WATER_block_level_encryption_ratio_selector_with_policy/block_level_encryption_ratio_selector_with_policy.py",
                        leaked_plaintext_size, policy_penalty, phase_type])
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: WATER_block_level_encryption_ratio_selector_with_policy.py program failed with CalledProcessError: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"WARNING: File not found.")
        return None


def main():
    leaked_plaintext_sizes = ["12", "24", "48"]  # Where 12 refers to half a day's worth of utility readings exposed to the attacker
    policy_penalties = ["500", "600", "700"]
    phase_type = "training"

    for leaked_plaintext_size in leaked_plaintext_sizes:
        # run_electricity_per_household(leaked_plaintext_size=leaked_plaintext_size, phase_type=phase_type)
        # run_water_per_household(leaked_plaintext_size=leaked_plaintext_size, phase_type=phase_type)
        #
        # run_electricity_per_block(leaked_plaintext_size=leaked_plaintext_size, phase_type=phase_type)
        # run_water_per_block(leaked_plaintext_size=leaked_plaintext_size, phase_type=phase_type)

        for policy_penalty in policy_penalties:
            print(f"Running with leaked plaintext size: {leaked_plaintext_size} and policy penalty: {policy_penalty}\n")
            run_electricity_per_block_with_policy(leaked_plaintext_size=leaked_plaintext_size, policy_penalty=policy_penalty, phase_type="training")
            run_water_per_block_with_policy(leaked_plaintext_size=leaked_plaintext_size, policy_penalty=policy_penalty, phase_type="training")


if __name__ == "__main__":
    main()
