import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath)
    # Preprocessing steps here
    return data

def preprocess_data(data):
    # Data cleaning and feature selection
    # Example: handling missing values, encoding categorical data
    return processed_data

if __name__ == "__main__":
    data = load_data('data/dataset.csv')
    processed_data = preprocess_data(data)
