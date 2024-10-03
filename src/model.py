from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_data, preprocess_data

def build_model(data):
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test

if __name__ == "__main__":
    data = load_data('../data/dataset.csv')
    data = preprocess_data(data)
    model, X_test, y_test = build_model(data)
    print("Model trained!")
