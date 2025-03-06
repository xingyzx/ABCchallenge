# evaluation.py

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt


def load_models():
    """
    Load models from file.
    """
    models = {}
    # Load XGBoost model from file
    try:
        xgb_model = xgb.Booster()
        xgb_model.load_model('xgboost_model.json')
        models['xgboost'] = xgb_model
        print("XGBoost model loaded successfully.")
    except Exception as e:
        print("Error loading XGBoost model:", e)

    # Load SVM model from file using joblib
    try:
        svm_model = joblib.load('svm_model.pkl')
        models['svm'] = svm_model
        print("SVM model loaded successfully.")
    except Exception as e:
        print("Error loading SVM model:", e)

    return models


def evaluate_single_model(model, test_data, model_type='xgboost'):
    """
    Evaluate a single model on test_data and generate an 11x11 evaluation table.
    The table layout:
      - Rows 0-9: Actual classes (0-9), columns 0-9: predicted classes (0-9), as integers.
      - Last row: Precision for each predicted class, shown as a percentage (two decimals).
      - Last column: Recall for each actual class, shown as a percentage (two decimals).
      - Bottom-right cell: Macro F1-score, shown as "F1: xx.xx%".
    """
    # Split features and labels
    X_test = test_data.drop('Activity', axis=1)
    y_test = test_data['Activity']

    # Predict according to model type
    if model_type.lower() == 'xgboost':
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
    elif model_type.lower() == 'svm':
        y_pred = model.predict(X_test)
    else:
        raise ValueError("Unsupported model_type. Use 'xgboost' or 'svm'.")

    # Compute overall accuracy and print classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Compute confusion matrix (assume classes are 0-9)
    cm = confusion_matrix(y_test, y_pred, labels=range(10))

    # Compute per-class precision and recall (as percentages)
    precisions = []
    recalls = []
    f1_scores = []
    for i in range(10):
        col_sum = cm[:, i].sum()
        row_sum = cm[i, :].sum()
        p = (cm[i, i] / col_sum * 100) if col_sum > 0 else 0.0
        r = (cm[i, i] / row_sum * 100) if row_sum > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        f1_scores.append(f1)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)

    # Build an extended 11x11 table with object dtype to allow mixed types (integers and formatted strings)
    table = np.empty((11, 11), dtype=object)

    # Fill first 10x10 with confusion matrix (as integers)
    for i in range(10):
        for j in range(10):
            table[i, j] = int(cm[i, j])

    # Fill last row (row index 10) with precision (formatted as "xx.xx%")
    for j in range(10):
        table[10, j] = f"{precisions[j]:.2f}%"

    # Fill last column (column index 10) with recall (formatted as "xx.xx%")
    for i in range(10):
        table[i, 10] = f"{recalls[i]:.2f}%"

    # Set bottom-right cell as macro F1-score formatted string
    table[10, 10] = f"F1: {macro_f1:.2f}%"

    # Build DataFrame for the extended table
    classes = [str(i) for i in range(10)]
    index_labels = classes + ["Precision"]
    col_labels = classes + ["Recall"]
    df_cm = pd.DataFrame(table, index=index_labels, columns=col_labels)

    return df_cm, accuracy, macro_precision, macro_recall, macro_f1


def plot_evaluation_table(df, model_type, overall_accuracy, macro_precision, macro_recall, macro_f1):
    """
    Plot the evaluation table using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    title = (f"{model_type.upper()} Evaluation Table\n"
             f"Accuracy: {overall_accuracy:.4f}, Macro Precision: {macro_precision:.2f}%, "
             f"Macro Recall: {macro_recall:.2f}%, Macro F1: {macro_f1:.2f}%")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def evaluate_all_models(test_data):
    """
    Evaluate multiple models (XGBoost and SVM) on the given test_data.
    Models are loaded from stored files.

    Parameters:
        test_data (DataFrame): DataFrame containing features and numeric label column 'Activity'.

    Returns:
        results (dict): A dictionary with keys as model types ('xgboost', 'svm') and values as a tuple:
                        (evaluation table DataFrame, overall accuracy, macro_precision, macro_recall, macro_f1)
    """
    results = {}
    models = load_models()

    for model_type, model in models.items():
        print(f"\nEvaluating model: {model_type}")
        df_cm, acc, macro_prec, macro_rec, macro_f1 = evaluate_single_model(model, test_data, model_type=model_type)
        print("Extended Evaluation Table (11x11):")
        print(df_cm)
        print(f"Overall Accuracy for {model_type}: {acc:.4f}")
        print(f"Macro Precision: {macro_prec:.2f}%, Macro Recall: {macro_rec:.2f}%, Macro F1: {macro_f1:.2f}%")
        # Plot the evaluation table
        plot_evaluation_table(df_cm, model_type, acc, macro_prec, macro_rec, macro_f1)
        results[model_type] = (df_cm, acc, macro_prec, macro_rec, macro_f1)

    return results


