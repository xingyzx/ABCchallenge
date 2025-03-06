import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
import re


def convert_to_utc_and_rename(input_dir):
    """
    递归处理所有子文件夹和文件，将时间统一转换为UTC
    """
    # 遍历所有子目录和文件
    for root, dirs, files in os.walk(input_dir, topdown=False):
        # 1. 处理子文件夹名
        for dir_name in dirs:
            # 匹配时间部分的正则表达式（忽略结尾的_c等）
            dir_pattern = r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}\+\d{4})_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}\+\d{4})"
            match = re.match(dir_pattern, dir_name)
            if match:
                # 提取两个时间部分并转换为UTC
                start_time_str, end_time_str = match.groups()
                start_utc = convert_time_str_to_utc(start_time_str)
                end_utc = convert_time_str_to_utc(end_time_str)

                # 构建新文件夹名
                new_dir_name = f"{start_utc}_{end_utc}_c"
                old_dir_path = os.path.join(root, dir_name)
                new_dir_path = os.path.join(root, new_dir_name)
                os.rename(old_dir_path, new_dir_path)

        # 2. 处理CSV文件名和内容
        for file_name in files:
            if file_name.endswith('.csv'):
                # 处理文件名
                file_time_str = os.path.splitext(file_name)[0]
                utc_time_str = convert_time_str_to_utc(file_time_str)
                new_file_name = f"{utc_time_str}.csv"

                # 文件路径操作
                old_file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(root, new_file_name)
                os.rename(old_file_path, new_file_path)

                # 处理文件内容
                df = pd.read_csv(new_file_path)
                df['Timestamp'] = df['Timestamp'].apply(convert_timestamp_to_utc)
                df.to_csv(new_file_path, index=False)


def convert_time_str_to_utc(time_str):
    """
    将时间字符串（如2024-09-01T22-25+0100）转换为UTC格式
    """
    # 解析原始时间
    dt = datetime.strptime(time_str, "%Y-%m-%dT%H-%M%z")
    # 转换为UTC
    utc_dt = dt.astimezone(pytz.UTC)
    # 格式化为目标字符串（毫秒部分已去除）
    return utc_dt.strftime("%Y-%m-%dT%H-%M%z").replace("+0000", "+0000")


def convert_timestamp_to_utc(timestamp_str):
    """
    将Timestamp列的时间（如2024-09-01 22:25:07.752000+01:00）转换为UTC格式
    """
    # 解析带时区的时间
    dt = pd.to_datetime(timestamp_str, utc=True)
    # 转换为UTC并格式化（保留3位毫秒）
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "+00:00"


if __name__ == "__main__":
    input_dir = "TrainingDataPD25/users_timeXYZ/All_split"
    convert_to_utc_and_rename(input_dir)