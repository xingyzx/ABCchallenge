import pandas as pd
from sklearn.svm import SVC
import joblib


def train_SVM_model(train_data):
    # Split features and labels from the training data
    X_train = train_data.drop('Activity', axis=1)
    y_train = train_data['Activity']

    # Create an SVM classifier with RBF kernel
    svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

    # Train the SVM classifier on the training data
    svm_clf.fit(X_train, y_train)

    # Save the trained SVM model to a file using joblib
    joblib.dump(svm_clf, 'svm_model.pkl')
    print("SVM model trained and saved as 'svm_model.pkl'.")

    return svm_clf
