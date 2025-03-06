import os
import csv
from datetime import datetime, timedelta
from dateutil import tz
import pytz

# 时区定义
TRAIN_TZ = tz.tzoffset(None, 9 * 3600)  # +0900
TARGET_TZ = tz.tzoffset(None, 1 * 3600)  # +0100


def parse_train_time(time_str):
    """解析训练数据时间并转换为+0100时区"""
    dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M")
    return dt.replace(tzinfo=TRAIN_TZ).astimezone(TARGET_TZ)


def parse_accel_filename(filename):
    """解析加速度文件名中的时间范围"""
    parts = os.path.splitext(filename)[0].split('_')
    try:
        start = datetime.strptime(parts[0], "%Y-%m-%dT%H-%M%z")
        end = datetime.strptime(parts[1], "%Y-%m-%dT%H-%M%z")
        return start, end
    except:
        return None, None


def match_activities():
    # 创建输出目录
    os.makedirs(target_filepath, exist_ok=True)

    # 读取训练活动数据
    activities = []
    with open(ActivitiesFile, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 处理时间字段
            started = row['Started'] or row['Updated']
            finished = row['Finished'] or row['Updated']

            activity = {
                "id": row["ID"],
                "start": parse_train_time(started),
                "end": parse_train_time(finished),
                "row": row
            }
            activities.append(activity)

    # 分组连续活动
    groups = []
    current_group = []
    prev_end = None

    for act in activities:
        if not current_group:
            current_group.append(act)
            prev_end = act["end"]
            continue

        if act["start"] == prev_end:
            current_group.append(act)
            prev_end = act["end"]
        else:
            groups.append(current_group)
            current_group = [act]
            prev_end = act["end"]
    if current_group:
        groups.append(current_group)

    # 处理每个组
    for group in groups:
        if not group:
            continue

        # 获取组时间范围
        group_start = group[0]["start"]
        group_end = group[-1]["end"]+ timedelta(minutes=1)

        # 在All_merge中查找匹配的加速度文件
        matched_files = []
        for filename in os.listdir("TrainingDataPD25/users_timeXYZ/All_merge"):
            if not filename.endswith(".csv"):
                continue

            file_start, file_end = parse_accel_filename(filename)
            if not file_start or not file_end:
                continue

            # 检查时间重叠
            if (group_start <= file_end) and (group_end >= file_start):
                matched_files.append((
                    os.path.join("TrainingDataPD25/users_timeXYZ/All_merge", filename),
                    file_start,
                    file_end
                ))

        # 生成输出目录
        group_folder = (
            f"{group_start.strftime('%Y-%m-%dT%H-%M%z')}_"
            f"{group_end.strftime('%Y-%m-%dT%H-%M%z')}_"
            f"{len(group)}_"
            f"{len(matched_files)}"
        )
        group_path = os.path.join(target_filepath, group_folder)
        os.makedirs(group_path, exist_ok=True)

        # 保存活动数据
        activity_file = os.path.join(group_path, f"TrainActivities_{group[0]['id']}.csv")
        with open(activity_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(group[0]["row"].keys())  # 写入标题
            for act in group:
                writer.writerow(act["row"].values())

        # 保存加速度数据
        for i, (filepath, _, _) in enumerate(matched_files, 1):
            accel_data = []
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        timestamp = datetime.strptime(row[1], "%Y-%m-%dT%H:%M:%S.%f%z")
                        if group_start <= timestamp <= group_end:
                            accel_data.append(row)
                    except:
                        continue

            # 生成加速度文件名
            accel_filename = (
                f"{group_start.strftime('%Y-%m-%dT%H-%M%z')}_"
                f"{group_end.strftime('%Y-%m-%dT%H-%M%z')}_{i}.csv"
            )
            accel_path = os.path.join(group_path, accel_filename)

            with open(accel_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(accel_data)


if __name__ == "__main__":
    ActivitiesFile = "TestActivities-20240920.csv"
    target_filepath = 'TrainingDataPD25/users_timeXYZ/All_match_test'
    match_activities()