from data_loader import load_data
from feature_extractor import process_all_files
from model_XGBoost import train_XGBoost_model
from model_SVM import train_SVM_model
from evaluation import evaluate_all_models
import pandas as pd
from defines import activity_map

base_dir = 'TrainingDataPD25/users_timeXYZ/users'  # Root directory for accelerometer data
activity_file = 'TrainingDataPD25/TrainActivities.csv'  # Path to the activity labels file
process_data= True
train_model = True
def main():
    # Load and process the data
    if process_data:
        data = load_data("TrainingDataPD25/users_timeXYZ/All_match")

        train_data,test_data = process_all_files(data)
        train_data.to_csv("train_features.csv", index=False)
        test_data.to_csv("test_features.csv", index=False)
    else:
        train_data = pd.read_csv("train_features.csv")
        test_data = pd.read_csv("test_features.csv")
    if train_model:
        train_XGBoost_model(train_data,test_data)
        #train_SVM_model(train_data)
    evaluate_all_models(test_data)




if __name__ == "__main__":
    main()
