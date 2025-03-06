import os
import shutil
import pandas as pd
from datetime import datetime, timedelta
import pytz
import re
from defines import activity_map  # 确保存在defines.py文件


def match_activities(config):
    # 读取活动数据
    activities = pd.read_csv(config['activities_path'])
    activities['start_utc'] = activities.apply(parse_activity_time_start, axis=1)
    activities['end_utc'] = activities.apply(parse_activity_time_end, axis=1)
    # # 转换时区并处理时间字段
    # activities['start_utc'] = activities.apply(
    #     lambda x: parse_activity_time(x['Started'] or x['Updated'], 540), axis=1)
    # activities['end_utc'] = activities.apply(
    #     lambda x: parse_activity_time(x['Finished'] or x['Updated'], 540), axis=1)


    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)

    unmatched_count = 0
    for idx, row in activities.iterrows():
        # 获取活动类型代码
        activity_code = activity_map.get(row['Activity Type'], -1)
        if activity_code == -1:
            continue  # 跳过未知活动类型

        # 查找匹配文件
        matched_files = find_matching_files(
            row['start_utc'],
            row['end_utc'],
            config['data_root']
        )



        # 创建目标文件夹
        folder_name = f"{row['ID']}_{activity_code}_{len(matched_files)}"
        target_dir = os.path.join(config['output_dir'], folder_name)
        os.makedirs(target_dir, exist_ok=True)

        if not matched_files:
            unmatched_count += 1
            continue

        # 复制并重命名文件
        for src_path in matched_files:
            dest_filename = generate_new_filename(src_path)
            shutil.copy2(src_path, os.path.join(target_dir, dest_filename))

    print(f"未匹配记录数: {unmatched_count}")


# def parse_activity_time(time_str, offset):
#     """将+0900时间转换为UTC"""
#     try:
#         dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M")
#         return dt.replace(tzinfo=pytz.FixedOffset(offset)).astimezone(pytz.UTC)
#     except:
#         return None

def parse_activity_time_start(row):
    """解析活动时间并转换为UTC"""
    time_str = row['Started'] if pd.notnull(row['Started']) else row['Updated']
    dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M").replace(tzinfo=pytz.FixedOffset(540))  # +0900时区
    return dt.astimezone(pytz.UTC)

def parse_activity_time_end(row):
    """解析活动时间并转换为UTC"""
    time_str = row['Finished'] if pd.notnull(row['Finished']) else row['Updated']
    dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M").replace(tzinfo=pytz.FixedOffset(540))  # +0900时区
    return dt.astimezone(pytz.UTC)


def find_matching_files(start_time, end_time, data_root):
    """递归查找匹配文件"""
    matched_files = set()

    for root, _, files in os.walk(data_root):
        for dir_name in os.listdir(root):
            dir_path = os.path.join(root, dir_name)
            if not os.path.isdir(dir_path):
                continue

            # 解析文件夹时间范围
            folder_times = parse_folder_times(dir_name)
            if not folder_times:
                continue

            # 检查时间重叠
            if (folder_times['end'] < start_time) or (folder_times['start'] > end_time):
                continue

            # 处理单个文件夹
            folder_files = process_folder(start_time, end_time, dir_path)
            matched_files.update(folder_files)

    return list(matched_files)


def parse_folder_times(dir_name):
    """解析文件夹时间范围"""
    pattern = r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}\+0000)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}\+0000)"
    match = re.match(pattern, dir_name)
    if not match:
        return None

    try:
        start = datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M%z").astimezone(pytz.UTC)
        end = datetime.strptime(match.group(2), "%Y-%m-%dT%H-%M%z").astimezone(pytz.UTC)
        return {'start': start, 'end': end}
    except:
        return None


def process_folder(start, end, dir_path):
    """处理单个文件夹的匹配逻辑"""
    valid_files = []
    for fname in os.listdir(dir_path):
        if not fname.endswith('.csv'):
            continue

        # 解析文件时间
        file_time = parse_file_time(fname)
        if not file_time:
            continue

        # 计算时间窗口
        window_start = file_time
        window_end = window_start + timedelta(minutes=2)

        # 检查时间重叠
        if window_end <= start or window_start > end:
            continue

        valid_files.append((window_start, os.path.join(dir_path, fname)))

    if not valid_files:
        return []

    # 按时间排序
    valid_files.sort(key=lambda x: x[0])

    # 处理不同时间范围的情况
    if  start.minute == end.minute:
        # 同一分钟取最早的一个
        return [valid_files[0][1]]
    else:
        # 跨分钟取所有符合条件的文件
        return [p for t, p in valid_files
                if (t >= start - timedelta(minutes=1)) and (t <= end)]


def parse_file_time(fname):
    """解析CSV文件名时间"""
    try:
        time_str = os.path.splitext(fname)[0]
        return datetime.strptime(time_str, "%Y-%m-%dT%H-%M%z").astimezone(pytz.UTC)
    except:
        return None


def generate_new_filename(src_path):
    """生成目标文件名"""
    dir_part = os.path.basename(os.path.dirname(src_path)).split('_c')[0].replace('+0000', '')
    file_part = os.path.basename(src_path).split('T')[1].replace('+0000', '')[:-4]
    return f"{dir_part}_c_{file_part}.csv"


if __name__ == "__main__":
    config = {
        'activities_path': 'TrainingDataPD25/TrainActivities.csv',
        'data_root': 'TrainingDataPD25/users_timeXYZ/All_split',
        'output_dir': 'TrainingDataPD25/users_timeXYZ/All_match'
    }
    match_activities(config)