import os
import pandas as pd
from datetime import datetime
import pytz


def convert_timezone_in_file(input_path):
    # 解析文件名
    dir_path, filename = os.path.split(input_path)
    base_name, ext = os.path.splitext(filename)

    # 转换文件名中的时区
    parts = base_name.split('_')
    new_parts = []
    for part in parts[:-1]:  # 最后部分是_130不需要处理
        if '+' in part:
            dt_str, tz = part.split('+')
            dt = datetime.strptime(dt_str, "%Y-%m-%dT%H-%M")
            # 创建带时区的datetime对象
            dt = pytz.timezone('Asia/Tokyo').localize(dt)  # +0900时区
            # 转换为+0100时区
            dt = dt.astimezone(pytz.timezone('Europe/London'))  # +0100时区
            # 重新格式化为字符串
            new_part = dt.strftime("%Y-%m-%dT%H-%M") + "+0100"
            new_parts.append(new_part)
        else:
            new_parts.append(part)
    new_filename = "_".join(new_parts + [parts[-1]]) + ext

    # 创建新文件路径
    output_path = os.path.join(dir_path, new_filename)

    # 转换文件内容
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        # 处理标题行
        header = f_in.readline().strip()
        if not header.startswith("ID,Timestamp,X,Y,Z"):
            raise ValueError("文件头格式不符合预期")
        f_out.write(header + "\n")

        # 处理数据行
        for line_num, line in enumerate(f_in, 2):  # 从第2行开始计数
            line = line.strip()
            if not line:  # 跳过空行
                continue

            parts = line.split(',')
            # 数据校验
            if len(parts) < 5:
                print(f"警告：第{line_num}行字段不足，已跳过 | 内容: {line}")
                continue

            try:
                # 解析原始时间
                dt = datetime.strptime(parts[1], "%Y-%m-%dT%H:%M:%S.%f%z")
            except ValueError as e:
                print(f"错误：第{line_num}行时间格式无效 | 内容: {parts[1]} | 错误: {str(e)}")
                continue

            # 转换为+0100时区
            dt = dt.astimezone(pytz.timezone('Europe/London'))

            # 格式化为新字符串（保留3位毫秒）
            new_timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0100"

            # 重构行数据
            new_line = f"{parts[0]},{new_timestamp},{','.join(parts[2:5])}\n"  # 只取前5个字段
            f_out.write(new_line)

    print(f"文件已转换并保存至：{output_path}")


# 使用示例
input_file = "TrainingDataPD25/users_timeXYZ/All_merge/2024-09-06T21-07+0900_2024-09-06T21-11+0900_130.csv"
convert_timezone_in_file(input_file)