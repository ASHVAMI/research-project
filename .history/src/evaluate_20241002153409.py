from sklearn.metrics import accuracy_score
from model import build_model
from utils import load_data, save_model, load_model, generate_classification_reportS

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    model, X_test, y_test = build_model('data/dataset.csv')
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Model accuracy: {accuracy}')

# Example in model.py
data = load_data('data/dataset.csv')
# After training the model
save_model(model, 'output/model.pkl')

# Example in evaluate.py
report, conf_matrix = generate_classification_report(y_test, y_pred)
save_results({'accuracy': accuracy}, 'output/results.csv')