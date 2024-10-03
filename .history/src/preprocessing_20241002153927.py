import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Example preprocessing: handle missing values, drop unnecessary columns
    data.fillna(0, inplace=True)
    return data

if __name__ == "__main__":
    data = load_data('../data/dataset.csv')
    processed_data = preprocess_data(data)
    print(processed_data.head())
