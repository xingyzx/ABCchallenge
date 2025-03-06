import os
import pandas as pd
from datetime import timedelta


def split_into_windows(input_file, output_dir, window_length_minutes=2, step_size_minutes=1):
    """
    Split a CSV file into overlapping windows of fixed length and step size.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_dir (str): Path to the output directory where the split files will be saved.
        window_length_minutes (int): Length of each window in minutes.
        step_size_minutes (int): Time step for sliding windows in minutes.
    """
    # Load the CSV file
    df = pd.read_csv(input_file,names=['ID', 'Timestamp', 'X', 'Y', 'Z'])

    # Ensure that the data has the 'Timestamp' column in datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Get the start and end time for the whole data
    start_time = df['Timestamp'].iloc[0]
    end_time = df['Timestamp'].iloc[-1]

    # Create a subfolder for the current CSV file
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    subfolder_path = os.path.join(output_dir, base_filename)
    os.makedirs(subfolder_path, exist_ok=True)

    # Create the windows and save each window to a new CSV file
    window_start = start_time
    window_end = start_time + timedelta(minutes=window_length_minutes)

    window_idx = 1  # Start window index
    while window_end-timedelta(minutes=window_length_minutes) <= end_time:
        # Filter the data within the current window
        window_df = df[(df['Timestamp'] >= window_start) & (df['Timestamp'] < window_end)]

        # Use the window start time for the filename, format as "YYYY-MM-DDTHH-MM+TZ"
        window_filename = window_start.strftime("%Y-%m-%dT%H-%M%z") + '.csv'
        window_filepath = os.path.join(subfolder_path, window_filename)

        # Save the window to a new CSV file
        window_df.to_csv(window_filepath, index=False)
        print(f"Saved window {window_idx} to {window_filepath}")

        # Move to the next window
        window_start = window_start + timedelta(minutes=step_size_minutes)
        window_end = window_start + timedelta(minutes=window_length_minutes)
        window_idx += 1


def process_all_files(input_dir, output_dir):
    """
    Process all CSV files in the input directory, splitting them into windows.

    Parameters:
        input_dir (str): Directory containing the merged CSV files.
        output_dir (str): Directory where the split files will be saved.
    """
    # Get a list of all CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for csv_file in csv_files:
        input_file = os.path.join(input_dir, csv_file)
        print(f"Processing file: {input_file}")
        split_into_windows(input_file, output_dir)


def main():
    # Define the input and output directories
    input_dir = "TrainingDataPD25/users_timeXYZ/All_merge"  # Path to the folder with merged CSV files
    output_dir = "TrainingDataPD25/users_timeXYZ/All_split"  # Path to the folder where split files will be saved

    # Process all files
    process_all_files(input_dir, output_dir)


if __name__ == "__main__":
    main()
