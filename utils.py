from pathlib import Path

# 指定根文件夹路径
folder_path = Path('TrainingDataPD25/users_timeXYZ/users')  # 替换为你的文件夹路径

# 初始化计数器
total_csv_count = 0

# 遍历文件夹中的每个子文件夹
for subfolder in folder_path.iterdir():
    if subfolder.is_dir():
        # 获取该子文件夹中的所有.csv文件
        csv_files = list(subfolder.glob('*.csv'))

        # 输出每个子文件夹的名称及其 CSV 文件个数
        print(f"子文件夹 '{subfolder.name}' 中的 CSV 文件个数: {len(csv_files)}")

        # 累加到总计数器
        total_csv_count += len(csv_files)

# 输出总数
print(f"所有子文件夹中的 CSV 文件总数: {total_csv_count}")
