# Parkinson's and Alzheimer's Disease Activity Recognition Using Accelerometer Data

## Project Structure

```
.
|-- README.md               // Project overview and instructions
|-- TrainingDataPD25/      // Training data directory
|-- CNN_LSTM.py            // CNN-LSTM model implementation
|-- data_loader.py         // Data loading and preprocessing script
|-- data_loader_2.py       // Alternative data loading script
|-- data_loader_old.py     // Legacy data loading script
|-- defines.py             // Constants and configuration definitions
|-- denoise.py            // Data denoising script
|-- evaluation.py          // Model evaluation script
|-- feature_extractor.py   // Feature extraction script
|-- main.py                // Main script to run the pipeline
|-- main_2.py             // Secondary main script
|-- matchCSV.py           // CSV file matching script
|-- matchCSV_old.py       // Legacy CSV file matching script
|-- mergeCSV.py           // CSV file merging script
|-- model_CNN_LSTM.py     // CNN-LSTM model definition
|-- model_SVM.py          // SVM model implementation
|-- model_XGBoost.py      // XGBoost model implementation
|-- processed_activity_data.pkl // Processed activity data file
|-- splitCSV.py           // CSV file splitting script
|-- svm_model.pkl         // Trained SVM model file
|-- test_features.csv     // Test features data file
|-- TestActivities-20240920.csv // Test activities labels file
```

## README

### Project Overview

This repository contains code and data for recognizing normal and unusual activities related to Parkinson's and Alzheimer's diseases using accelerometer data. The project includes three models: XGBoost, SVM, and CNN-LSTM.

### Data Structure

The data is stored in the `TrainingDataPD25` directory. The `data_loader.py` and `data_loader_2.py` scripts are used to load and preprocess the data. The `mergeCSV.py` script is used to merge CSV files, while the `splitCSV.py` script is used to split them into smaller chunks for pro
