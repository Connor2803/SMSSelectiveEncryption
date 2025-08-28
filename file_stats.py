import os
import csv

# def analyse_csv_row_counts(folder_path):
#     row_counts = []
#
#     # Loop through all files in the directory
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.csv'):
#             file_path = os.path.join(folder_path, filename)
#             with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
#                 reader = csv.reader(csvfile)
#                 next(reader, None)  # Skip header row
#                 row_count = sum(1 for _ in reader)
#                 row_counts.append(row_count)
#
#     if not row_counts:
#         return None, None, None  # No CSV files found
#
#     max_rows = max(row_counts)
#     min_rows = min(row_counts)
#     avg_rows = sum(row_counts) / len(row_counts)
#
#     return max_rows, min_rows, avg_rows
#
# folder = './examples/datasets/electricity/households_10240'
# maximum, minimum, average = analyse_csv_row_counts(folder)
#
# print(f"Max rows: {maximum}")
# print(f"Min rows: {minimum}")
# print(f"Average rows: {average:.2f}")
