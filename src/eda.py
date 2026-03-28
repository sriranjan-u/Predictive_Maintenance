import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df_path):
    # Load data
    df = pd.read_csv(df_path)
    os.makedirs('Predictive_Maintenance/plots', exist_ok=True)
    
    # 1. Target Distribution (Class Balance)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Engine Condition', data=df, palette='viridis')
    plt.title('Distribution of Engine Health (0: Normal, 1: Faulty)')
    plt.savefig('Predictive_Maintenance/plots/target_distribution.png')
    plt.close()

    # 2. Correlation Heatmap (Identifying Thermal Coupling)
    plt.figure(figsize=(10, 8))
    # Select only numeric columns
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('Predictive_Maintenance/plots/correlation_heatmap.png')
    plt.close()

    # 3. Boxplots (Visualizing Outliers for the Report)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[['Lub oil pressure', 'Fuel pressure', 'Coolant pressure']])
    plt.title('Pressure Feature Ranges (Identifying Sensor Spikes)')
    plt.savefig('Predictive_Maintenance/plots/pressure_outliers.png')
    plt.close()

    print("EDA Complete. Plots saved to Predictive_Maintenance/plots/")

if __name__ == "__main__":
    perform_eda('Predictive_Maintenance/data/engine_data.csv')
