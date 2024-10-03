import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_correlation(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig('../output/visualizations/correlation_heatmap.png')

if __name__ == "__main__":
    data = pd.read_csv('../data/dataset.csv')
    visualize_correlation(data)
    print("Visualization saved!")
