import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser


def load_and_process_data(data_root):
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

        # 加载加速度数据（含时间戳）
        accel_data = load_accel_data(folder_path)
        if accel_data.empty:
            continue

        # 加载活动标签数据
        activities = load_activity_labels(folder_path)
        if activities is None:
            continue

        # 时间对齐和切割处理
        process_actions(accel_data, activities, num_actions, rows)

    return pd.DataFrame(rows, columns=['Data', 'Activity'])


def load_accel_data(folder_path):
    """加载并合并所有加速度数据文件（保留时间戳）"""
    accel_dfs = []

    for fname in os.listdir(folder_path):
        if fname.startswith("202") and fname.endswith(".csv"):
            file_path = os.path.join(folder_path, fname)
            # 修改列名为大写
            df = pd.read_csv(file_path, header=None,
                             names=['id', 'Timestamp', 'X', 'Y', 'Z'])
            df['Timestamp'] = df['Timestamp'].apply(parser.parse)
            accel_dfs.append(df)

    if not accel_dfs:
        return pd.DataFrame()

    full_df = pd.concat(accel_dfs).sort_values('Timestamp')
    return full_df.reset_index(drop=True)[['Timestamp', 'X', 'Y', 'Z']]


def load_activity_labels(folder_path):
    """加载活动标签映射表（时区转换优化版）"""
    for fname in os.listdir(folder_path):
        if fname.startswith("TrainActivities"):
            df = pd.read_csv(os.path.join(folder_path, fname))

            # 统一时区转换
            for col in ['Started', 'Finished']:
                df[col] = pd.to_datetime(df[col], format='%Y/%m/%d %H:%M'
                                         ).dt.tz_localize('Asia/Tokyo'  # +0900
                                                          ).dt.tz_convert('Etc/GMT-1')  # +0100
            return df
    return None


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
    activity_mapping = {}
    for _, activity in activities.iterrows():
        for idx, (slice_start, slice_end) in enumerate(time_slices):
            # 找到与时间切片重叠的活动
            if (activity['Started'] < slice_end) and (activity['Finished'] > slice_start):
                activity_mapping[idx] = activity['Activity Type']

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
        label = activity_mapping.get(idx, 'Unknown')
        rows.append({
            'Data': timed_data,  # 包含时间戳的二维数组
            'Activity': label
        })


# 使用示例
if __name__ == "__main__":
    # 创建标签映射字典（示例）
    label_map = {
        '1 (FACING camera) Sit and stand': 0,
        '2 (FACING camera) both hands SHAKING (sitting position)': 1,
        '3 Stand up from chair - both hands with SHAKING': 2,
        # ...添加其他标签映射
    }

    # 加载数据
    df = load_and_process_data("TrainingDataPD25/users_timeXYZ/All_match")

    # 转换标签为数字
    df['Activity'] = df['Activity'].map(label_map)

    # 显示数据结构
    print("处理后的数据结构示例：")
    print(df.head())

    # 保存处理结果（使用pickle保持数据类型）
    df.to_pickle("processed_activity_with_timestamp.pkl")