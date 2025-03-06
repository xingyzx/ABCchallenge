import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from datetime import timedelta
from sklearn.model_selection import train_test_split


def extract_features_from_window(window_df):
    """
    Extract features from a single window of accelerometer data.

    Parameters:
        window_df (DataFrame): DataFrame containing 'Timestamp', 'X', 'Y', 'Z'.

    Returns:
        dict: Dictionary containing extracted features.
    """
    features = {}

    # Calculate Average and Standard Deviation for each axis
    for axis in ['X', 'Y', 'Z']:
        axis_data = window_df[axis].values
        features[f'{axis}_mean'] = np.mean(axis_data)
        features[f'{axis}_std'] = np.std(axis_data)

    # Calculate Average Absolute Difference for each axis
    for axis in ['X', 'Y', 'Z']:
        axis_data = window_df[axis].values
        mean_val = np.mean(axis_data)
        abs_diff = np.abs(axis_data - mean_val)
        features[f'{axis}_aad'] = np.mean(abs_diff)

    # Calculate Average Resultant Acceleration
    x_arr = window_df['X'].values
    y_arr = window_df['Y'].values
    z_arr = window_df['Z'].values
    resultant = np.sqrt(x_arr ** 2 + y_arr ** 2 + z_arr ** 2)
    features['avg_resultant_acc'] = np.mean(resultant)

    # Calculate Time Between Peaks for each axis in milliseconds
    for axis in ['X', 'Y', 'Z']:
        axis_data = window_df[axis].values
        peak_indices, _ = find_peaks(axis_data)
        if len(peak_indices) < 2:
            features[f'{axis}_peak_interval_ms'] = 0.0
        else:
            # Get the timestamps corresponding to the peak indices
            peak_times = window_df.index[peak_indices]
            # Compute differences between successive peaks (in milliseconds)
            deltas = (peak_times[1:] - peak_times[:-1]).total_seconds() * 1000.0
            features[f'{axis}_peak_interval_ms'] = np.mean(deltas)

    # Calculate Binned Distribution for each axis (10 bins per axis -> 30 features total)
    num_bins = 10
    for axis in ['X', 'Y', 'Z']:
        axis_data = window_df[axis].values
        min_val, max_val = np.min(axis_data), np.max(axis_data)
        range_val = max_val - min_val

        if range_val == 0:
            # If the range is zero, put all data in the first bin
            dist = np.zeros(num_bins)
            dist[0] = 1.0
        else:
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)
            counts, _ = np.histogram(axis_data, bins=bin_edges)
            dist = counts / counts.sum()

        for i in range(num_bins):
            features[f'{axis}_bin_{i}'] = dist[i]

    return features


def extract_features_from_file(file_df, window_size_s=2.0, overlap=0.5):
    """
    Extract sliding-window features from a single accelerometer file.

    Parameters:
        file_df (DataFrame): DataFrame containing accelerometer data with columns ['Timestamp', 'X', 'Y', 'Z'].
        window_size_s (float): Window size in seconds (default 2 seconds).
        overlap (float): Overlap ratio between windows (default 0.5 for 50% overlap).

    Returns:
        DataFrame: Each row corresponds to features extracted from one window.
    """
    # Ensure the data is sorted by timestamp and set Timestamp as index
    if 'Timestamp' in file_df.columns:
        file_df['Timestamp'] = pd.to_datetime(file_df['Timestamp'])
        file_df = file_df.sort_values('Timestamp')
        file_df = file_df.set_index('Timestamp')
    else:
        file_df = file_df.sort_index()

    window_delta = pd.Timedelta(seconds=window_size_s)
    step_delta = pd.Timedelta(seconds=window_size_s * (1 - overlap))

    start_time = file_df.index.min()
    end_time = file_df.index.max()

    feature_list = []

    while start_time + window_delta <= end_time:
        window_end = start_time + window_delta
        # Slice the data for the current window
        window_df = file_df.loc[start_time:window_end]
        if len(window_df) > 0:
            features = extract_features_from_window(window_df)
            # Record window start and end time for reference
            # features['window_start'] = start_time
            # features['window_end'] = window_end
            feature_list.append(features)
        # Move to the next window
        start_time += step_delta

    features_df = pd.DataFrame(feature_list)
    return features_df


def process_all_files(dataloader_df, window_size_s=2.0, overlap=0.5):
    """
    Process each row from the dataloader DataFrame, extracting features for each file.

    Parameters:
        dataloader_df (DataFrame): DataFrame with two columns: 'Data' and 'Activity'.
                                   'Data' is a DataFrame of accelerometer readings from one file.
        window_size_s (float): Window size in seconds.
        overlap (float): Overlap ratio between windows.

    Returns:
        DataFrame: Each row corresponds to one window with extracted features and its associated Activity.
    """
    all_features = []
    # Iterate over each row in the dataloader DataFrame
    for idx, row in dataloader_df.iterrows():
        file_data = row['Data']
        activity = row['Activity']
        # Extract features for the current file
        features_df = extract_features_from_file(file_data, window_size_s, overlap)
        # Add the activity label to each window's features
        features_df['Activity'] = activity
        all_features.append(features_df)
    if all_features:
        # Combine all windows from all files into one DataFrame
        result_df = pd.concat(all_features, ignore_index=True)
        print("Feature extraction completed")

        train_df, test_df = train_test_split(result_df, test_size=0.3, random_state=42,
                                             stratify=result_df['Activity'])
        return train_df, test_df
    else:
        return pd.DataFrame().pd.DataFrame()


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example: Simulate dataloader output with two sample accelerometer files
    # Each 'Data' field is a DataFrame with columns ['Timestamp', 'X', 'Y', 'Z']
    time_index1 = pd.date_range("2024-09-08 23:31:16.515", periods=200, freq='20L')  # Approximately 50Hz
    data1 = pd.DataFrame({
        'Timestamp': time_index1,
        'X': np.random.randn(len(time_index1)),
        'Y': np.random.randn(len(time_index1)),
        'Z': np.random.randn(len(time_index1))
    })

    time_index2 = pd.date_range("2024-09-08 23:32:16.515", periods=250, freq='20L')
    data2 = pd.DataFrame({
        'Timestamp': time_index2,
        'X': np.random.randn(len(time_index2)),
        'Y': np.random.randn(len(time_index2)),
        'Z': np.random.randn(len(time_index2))
    })

    # Create a simulated dataloader DataFrame
    dataloader_df = pd.DataFrame({
        'Data': [data1, data2],
        'Activity': ['Activity_A', 'Activity_B']
    })

    # Extract features from all files in the dataloader
    features_result = process_all_files(dataloader_df, window_size_s=2.0, overlap=0.5)

    # Print the first few rows of the extracted features
    print(features_result.head())
