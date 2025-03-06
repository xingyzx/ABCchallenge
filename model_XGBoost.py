import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report


def train_XGBoost_model(train_data, test_data):

    # Split the dataset into training and testing sets
    X_train = train_data.drop('Activity', axis=1)
    y_train = train_data['Activity']
    X_test = test_data.drop('Activity', axis=1)
    y_test = test_data['Activity']

    # Convert to DMatrix (XGBoost's internal data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set parameters for XGBoost model
    params = {
        'objective': 'multi:softmax',  # Multi-class classification
        'num_class': 10,  # Set number of activity classes (update this to match your data)
        'eval_metric': 'mlogloss',  # Multi-class log loss (alternative to merror)
        'max_depth': 6,  # Maximum depth of the tree
        'eta': 0.1,  # Learning rate (typically a small value like 0.1 or 0.01)
        'subsample': 0.8,  # Subsample ratio (typically 0.8 or 0.9)
        'colsample_bytree': 0.8,  # Subsample ratio of features
        'silent': 1,  # Silent mode (don't print messages)
        'n_jobs': -1  # Use all CPU cores
    }

    # Train the model with early stopping
    num_round = 300  # Number of boosting rounds (iterations)
    bst = xgb.train(params, dtrain, num_round, evals=[(dtest, 'eval')], early_stopping_rounds=10)
    # Optionally, save the model
    bst.save_model('xgboost_model.json')