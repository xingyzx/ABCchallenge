import os
import re
import pandas as pd
from datetime import timedelta


def parse_timestamp_from_filename(filename):
    """
    Parse the timestamp from a filename of the form:
      user-acc_<ID>_2024-09-05T09_30_18.064+0100_<random>.csv
    Returns a Python datetime object in UTC.

    Example filename: user-acc_1716_2024-09-05T09_30_18.064+0100_7610.csv
    We'll extract: 2024-09-05T09:30:18.064+0100 (replace the first two underscores with colons).
    """
    pattern = r"user-acc_\d+_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\.\d{3}\+\d{4})_\d+\.csv"
    match = re.search(pattern, filename)
    if match:
        # e.g. 2024-09-05T09_30_18.064+0100
        timestamp_str = match.group(1)
        # 将前两个下划线替换为冒号
        # 2024-09-05T09_30_18.064+0100 -> 2024-09-05T09:30:18.064+0100
        timestamp_str = timestamp_str.replace('_', ':', 2)

        # 使用 pandas.to_datetime 解析为 UTC
        dt = pd.to_datetime(timestamp_str, errors='coerce')
        return dt
    else:
        return None


def merge_csv_files(input_dir, output_dir, time_threshold_seconds=60, min_gap_seconds=0.5):
    """
    Traverse all CSV files in `input_dir`, parse their timestamps, sort them by timestamp,
    and merge consecutive files if the timestamp difference between the last record of one file
    and the first record of the next file is <= `time_threshold_seconds` and the gap is >= `min_gap_seconds`.
    Save merged CSV files into `output_dir`, with filename reflecting the time range
    (down to the minute, including UTC offset).

    Parameters:
        input_dir (str): Directory containing original CSV files.
        output_dir (str): Directory to save merged CSV files.
        time_threshold_seconds (int): If the gap between the last record of one file
                                       and the first record of the next file is <= this value, merge them.
        min_gap_seconds (int): If the gap between the last record of one file
                               and the first record of the next file is >= this value, treat as non-continuous.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Collect all CSV files in input_dir
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    # 2. Parse timestamps and store (filename, timestamp) pairs
    file_info_list = []
    for csv_file in csv_files:
        ts = parse_timestamp_from_filename(csv_file)
        if ts is not None:
            file_info_list.append((csv_file, ts))
        else:
            print(f"Warning: Could not parse timestamp from {csv_file}, skipping.")

    # 3. Sort files by timestamp
    file_info_list.sort(key=lambda x: x[1])  # sort by second element (timestamp)

    current_batch_data = []
    earliest_dt = None  # earliest timestamp in current batch
    last_timestamp = None  # timestamp of the last record from previous CSV file

    # CSV column names (based on the structure: record, timestamp, x, y, z)
    columns = ['Record', 'Timestamp', 'X', 'Y', 'Z']

    for i, (filename, ts) in enumerate(file_info_list):
        file_path = os.path.join(input_dir, filename)

        # Load CSV with custom column names
        df = pd.read_csv(file_path, header=None, names=columns)

        # Get the timestamp of the first and last record in the current CSV
        first_record_timestamp = pd.to_datetime(df['Timestamp'].iloc[0])
        last_record_timestamp = pd.to_datetime(df['Timestamp'].iloc[-1])

        # Check if the time difference between the last record of the previous file and the first record of this file is small enough
        if last_timestamp is None:
            # This is the first CSV, start a new batch
            current_batch_data.append(df)
            earliest_dt = first_record_timestamp
            last_timestamp = last_record_timestamp
        else:
            # Compare time difference with last record's timestamp
            time_diff = (first_record_timestamp - last_timestamp).total_seconds()

            # Check if the gap is less than 5 seconds (non-continuous)
            if min_gap_seconds > time_diff > 0:
                # The CSV files are continuous, merge them
                current_batch_data.append(df)
                last_timestamp = last_record_timestamp
            else:
                # Finalize previous batch and save it
                merged_df = pd.concat(current_batch_data, ignore_index=True)

                # Generate filename based on the earliest and latest timestamps (rounded to minute)
                start_str = earliest_dt.floor('min').strftime("%Y-%m-%dT%H-%M%z")
                end_str = last_timestamp.floor('min').strftime("%Y-%m-%dT%H-%M%z")
                output_file = os.path.join(output_dir, f"{start_str}_{end_str}.csv")

                merged_df.to_csv(output_file, index=False, header=False)
                print(f"Saved merged file: {output_file} (rows={len(merged_df)})")

                # Start new batch
                current_batch_data = [df]
                earliest_dt = first_record_timestamp
                last_timestamp = last_record_timestamp

    # Process the last batch
    if current_batch_data:
        merged_df = pd.concat(current_batch_data, ignore_index=True)
        start_str = earliest_dt.floor('min').strftime("%Y-%m-%dT%H-%M%z")
        end_str = last_timestamp.floor('min').strftime("%Y-%m-%dT%H-%M%z")
        output_file = os.path.join(output_dir, f"{start_str}_{end_str}.csv")

        merged_df.to_csv(output_file, index=False, header=False)
        print(f"Saved merged file: {output_file} (rows={len(merged_df)})")


def main():
    input_dir = "TrainingDataPD25/users_timeXYZ/users/all"
    output_dir = "TrainingDataPD25/users_timeXYZ/All"
    merge_csv_files(input_dir, output_dir, time_threshold_seconds=60, min_gap_seconds=5)


if __name__ == "__main__":
    main()
