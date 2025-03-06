import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from defines import activity_map
from datetime import timedelta

matched_filenames = []
# -------------------------------
# Denoising functions using Butterworth low-pass filter
# -------------------------------
def load_data(data_root):
    """
    主数据处理函数，返回包含窗口化数据和标签的DataFrame
    """
    rows = []
    # 遍历所有子文件夹
    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        # 解析文件夹名称获取元数据
        parts = folder_name.split('_')
        num_actions = int(parts[2])  # 动作数量
        # 加载加速度数据
        for fname in os.listdir(folder_path):
            if fname.startswith("202") and fname.endswith(".csv"):
                file_path = os.path.join(folder_path, fname)
                accel_data = load_acc_data(file_path)
                if accel_data.empty:
                    continue
        # 加载活动标签数据
        activities = load_activity_labels(folder_path)
        if activities is None:
            continue

        # 时间对齐和切割处理
        process_actions(accel_data, activities, num_actions, rows)

    return pd.DataFrame(rows, columns=['Data', 'Activity'])

# -------------------------------
# Activity and accelerometer data loading functions
# -------------------------------
def load_activity_labels(folder_path):
    """
    Load activity labels from CSV file and perform timezone conversion.
    """
    for fname in os.listdir(folder_path):
        if fname.startswith("TrainActivities"):
            activities_df = pd.read_csv(os.path.join(folder_path, fname))


            # activities_df['Start Time'] = activities_df['Start Time'].dt.tz_convert('UTC')
            # activities_df['End Time'] = activities_df['End Time'].dt.tz_convert('UTC')
            # activities_df['Updated Time'] = activities_df['Updated Time'].dt.tz_convert('UTC')
            # Add 'Has Start Time' column

            return activities_df

    return None


def load_acc_data(file_path):
    """
    Load accelerometer data from a CSV file.
    Convert timestamp to datetime.

    Parameters:
        file_path (str): Path to the accelerometer CSV file.

    Returns:
        DataFrame: DataFrame with columns ['Timestamp', 'X', 'Y', 'Z'].
    """
    df = pd.read_csv(file_path, header=None, names=['ID', 'Timestamp', 'X', 'Y', 'Z'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    return df[['Timestamp', 'X', 'Y', 'Z']]


def process_actions(accel_data, activities, num_actions, rows):
    """处理单个文件夹的动作分割（带时间戳保留）"""
    # 获取总时间范围
    min_time = accel_data['Timestamp'].min()
    max_time = accel_data['Timestamp'].max()
    total_duration = max_time - min_time

    # 计算每个动作的理论持续时间
    action_duration = total_duration / num_actions

    # 创建理论时间区间
    time_slices = []
    current_start = min_time
    for _ in range(num_actions):
        time_end = current_start + action_duration
        time_slices.append((current_start, time_end))
        current_start = time_end

    # 匹配实际活动标签

    # 分割加速度数据（保留时间戳）
    for idx, (start_time, end_time) in enumerate(time_slices):
        # 获取当前时间片段的完整数据
        mask = (accel_data['Timestamp'] >= start_time) & \
               (accel_data['Timestamp'] < end_time)
        action_data = accel_data[mask]

        if len(action_data) == 0:
            continue

        # 转换为包含时间戳的numpy数组
        timed_data = action_data.to_numpy(dtype=np.object_)

        # 获取对应的活动标签
        activity_type = activities['Activity Type'][idx]
        label = activity_map[activity_type]
        rows.append({
            'Data': action_data,  # 包含时间戳的二维数组
            'Activity': label
        })

# -------------------------------
# Helper function to find matching accelerometer files for an activity
# -------------------------------



# -------------------------------
# Main data loading function that iterates over activity records
# -----------------------------

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    base_dir = "TrainingDataPD25/users_timeXYZ/All_match"  # Root directory for accelerometer data
    activity_file = 'TrainingDataPD25/TrainActivities.csv'  # Path to the activity labels file

    # Load and process the data
    data = load_data(base_dir)
    # 示例：打印前几行记录，每一行显示 Activity 类型和对应 Data 的前几行数据
    for i in range(min(3, len(data))):
        print(f"Activity: {data.loc[i, 'Activity']}")
        print(data.loc[i, 'Data'].head())
