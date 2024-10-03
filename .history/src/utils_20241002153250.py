import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix

def load_data(filepath):
    """
    Loads a CSV file and returns a Pandas DataFrame.
    
    :param filepath: Path to the CSV file
    :return: Pandas DataFrame containing the data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    return pd.read_csv(filepath)

def save_model(model, filepath):
    """
    Saves a trained model to a file.
    
    :param model: The trained model object
    :param filepath: Path where the model should be saved
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Loads a model from a file.
    
    :param filepath: Path to the saved model file
    :return: Loaded model object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    return joblib.load(filepath)

def generate_classification_report(y_true, y_pred):
    """
    Generates a classification report and confusion matrix.
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Classification report as a string
    """
    report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", conf_matrix)
    return report, conf_matrix

def save_results(results, filepath):
    """
    Saves model evaluation results (e.g., accuracy, classification report) to a CSV file.
    
    :param results: Dictionary containing results
    :param filepath: Path where the results CSV should be saved
    """
    results_df = pd.DataFrame([results])
    results_df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
