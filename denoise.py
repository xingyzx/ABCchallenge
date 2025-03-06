import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """巴特沃斯低通滤波器"""
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def denoise_accel_file(filepath):
    """处理单个加速度文件"""
    try:
        # 读取数据
        df = pd.read_csv(filepath, header=None, names=['id', 'timestamp', 'x', 'y', 'z'])

        # 计算采样频率
        timestamps = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f%z')
        time_diff = (timestamps - timestamps.shift(1)).dropna()
        fs = 1 / (time_diff.mean().total_seconds())  # 估算采样率

        # 滤波参数（根据人体运动特点设置）
        cutoff = 5  # 截止频率5Hz（高于正常人体运动频率）
        order = 4  # 滤波器阶数

        # 对各轴分别滤波
        for axis in ['x', 'y', 'z']:
            df[axis] = butter_lowpass_filter(df[axis].values,
                                             cutoff,
                                             fs,
                                             order)

        # 保存处理后的数据（覆盖原文件）
        df.to_csv(filepath, index=False, header=False)
        print(f"Processed: {filepath}")

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")


def batch_denoise(root_dir="TrainingDataPD25/users_timeXYZ/All_match"):
    """批量处理目录"""
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith("TrainActivities_"):
                continue  # 跳过活动标注文件

            if file.endswith(".csv"):
                filepath = os.path.join(root, file)
                denoise_accel_file(filepath)


if __name__ == "__main__":
    # 使用示例（请先备份原始数据）
    batch_denoise()

    print("去噪处理完成！")
