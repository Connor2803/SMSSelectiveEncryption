# python ./RL_model_V2_WATER/generate_inputs_V2_WATER.py

import csv
import os
import glob

MAX_HOUSEHOLD_UTILITY_READINGS = 10240
SECTION_SIZE = 1024


def main():
    all_household_output_data = []
    input_folder_path = "../examples/datasets/water/households_10240/"

    csv_files = glob.glob(os.path.join(input_folder_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_folder_path}. Please check the path.")
        return

    for file_path in csv_files:
        household_id = os.path.splitext(os.path.basename(file_path))[0]

        utility_readings_current_file = []
        utility_reading_dates_current_file = []

        try:
            with open(file_path, mode='r', newline='') as household_csv_file:
                csv_reader = csv.reader(household_csv_file, delimiter=',')

                rows_read_in_file = 0
                for row in csv_reader:
                    if rows_read_in_file >= MAX_HOUSEHOLD_UTILITY_READINGS:
                        break

                    try:
                        utility_readings_current_file.append(float(row[-1]))
                        utility_reading_dates_current_file.append(row[0])
                        rows_read_in_file += 1
                    except (ValueError, IndexError) as e:
                        print(f"Skipping malformed row in {file_path}: {row}. Error: {e}")
                        rows_read_in_file += 1
                        continue

            # Reverse the lists so the oldest readings are at the beginning.
            utility_readings_current_file.reverse()
            utility_reading_dates_current_file.reverse()
            num_sections = (len(utility_readings_current_file) + SECTION_SIZE - 1) // SECTION_SIZE

            for i in range(num_sections):
                start_index = i * SECTION_SIZE
                end_index = min((i + 1) * SECTION_SIZE, len(utility_readings_current_file))

                section_readings = utility_readings_current_file[start_index:end_index]
                section_dates = utility_reading_dates_current_file[start_index:end_index]

                if section_readings:
                    # Water readings are in reverse chronological order, the first date is the latest.
                    date_range = f"{section_dates[-1]} - {section_dates[0]}"  # Oldest date - Newest date.
                    total_usage_in_section = sum(section_readings)

                    all_household_output_data.append([
                        household_id,
                        i + 1, # Section number.
                        date_range,
                        total_usage_in_section,
                        section_readings
                    ])

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except Exception as e:
            print(f"An unexpected error occurred while reading {file_path}: {e}")

    output_file = "inputs_V2_WATER.csv"
    try:
        with open(output_file, 'w', newline='') as output_csv_file:
            csv_writer = csv.writer(output_csv_file)
            csv_writer.writerow(
                ["Household ID",
                 "Section Number",
                 "Date Range",
                 "Per Section Utility Usage",
                 "All Utility Readings in Section"]
            )
            csv_writer.writerows(all_household_output_data)

    except Exception as e:
        print(f"An error occurred during writing: {e}")
    finally:
        print(f"RL model inputs saved to {output_file}")


if __name__ == "__main__":
    main()