import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_data(data):
    sns.heatmap(data.corr(), annot=True)
    plt.savefig('output/visualizations/correlation_heatmap.png')

if __name__ == "__main__":
    data = pd.read_csv('data/dataset.csv')
    visualize_data(data)
