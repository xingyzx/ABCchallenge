import numpy as np
import pandas as pd
import os
import glob
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from defines import activity_map


# 1. 数据加载和预处理
def load_and_preprocess_data(base_path, activity_map, time_window_size=20):
    # 读取训练数据路径
    data_dirs = glob.glob(os.path.join(base_path, '*/All_match/*'))

    # 定义数据存储变量
    x_data = []
    y_data = []

    # 遍历所有子文件夹
    for data_dir in data_dirs:
        # 读取加速度数据
        acc_file = glob.glob(os.path.join(data_dir, "*.csv"))[0]  # 假设每个子文件夹只有一个加速度数据文件
        acc_data = pd.read_csv(acc_file, header=None, names=['ID', 'Time', 'X', 'Y', 'Z'])

        # 读取活动数据
        activity_file = glob.glob(os.path.join(data_dir, "TrainActivities.csv"))[0]
        activities = pd.read_csv(activity_file)

        # 对于每个动作进行平均分割
        for _, row in activities.iterrows():
            activity = row['Activity Type']
            activity_id = activity_map[activity]

            # 获取该动作的起始和结束时间
            start_time = pd.to_datetime(row['Started'])
            end_time = pd.to_datetime(row['Finished'])
            action_duration = (end_time - start_time).seconds

            # 切割加速度数据为时间段
            acc_data_segment = acc_data[(acc_data['Time'] >= start_time) & (acc_data['Time'] <= end_time)]

            # 分割为平均片段
            num_segments = int(np.ceil(action_duration / time_window_size))  # 每个片段的时间大小为20s
            segment_size = len(acc_data_segment) // num_segments

            for i in range(num_segments):
                segment = acc_data_segment.iloc[i * segment_size:(i + 1) * segment_size]

                # 提取特征
                x_data.append(segment[['X', 'Y', 'Z']].values)
                y_data.append(activity_id)

    # 转换为NumPy数组并返回
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


# 2. CNN-LSTM模型设计
def build_cnn_lstm_model(input_shape, num_classes):
    model = Sequential()

    # 1D卷积层：提取加速度数据中的时序特征
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # LSTM层：捕捉时间序列的长期依赖性
    model.add(LSTM(128, return_sequences=False))

    # 输出层：进行分类
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# 3. 模型训练和评估
def train_and_evaluate(base_path, activity_map):
    # 加载数据
    x_data, y_data = load_and_preprocess_data(base_path, activity_map)

    # 标签转换为独热编码
    y_data = to_categorical(y_data)

    # 切分数据集：80%训练，20%测试
    split_index = int(0.8 * len(x_data))
    x_train, x_test = x_data[:split_index], x_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]

    # 定义模型
    input_shape = (x_train.shape[1], x_train.shape[2])  # 这里x_train.shape[1]是时间步长，x_train.shape[2]是特征数
    model = build_cnn_lstm_model(input_shape, len(activity_map))

    # 训练模型
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    # 评估模型
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')


# 执行训练
base_path = '/path/to/your/data'  # 这里需要替换为你的数据路径
train_and_evaluate(base_path, activity_map)
