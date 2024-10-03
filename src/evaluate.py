from sklearn.metrics import accuracy_score
from model import build_model
from preprocessing import load_data, preprocess_data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return accuracy

if __name__ == "__main__":
    data = load_data('../data/dataset.csv')
    data = preprocess_data(data)
    model, X_test, y_test = build_model(data)
    evaluate_model(model, X_test, y_test)
