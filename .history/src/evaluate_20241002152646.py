from sklearn.metrics import accuracy_score
from model import build_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    model, X_test, y_test = build_model('data/dataset.csv')
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Model accuracy: {accuracy}')
