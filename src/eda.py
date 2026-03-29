
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import display

def perform_eda(df_path):
    # Load data
    df = pd.read_csv(df_path)

    # Create plots directory
    plot_dir = 'Predictive_Maintenance/plots'
    os.makedirs(plot_dir, exist_ok=True)

    sns.set(style="whitegrid")

    # ===============================
    # 1. DATA OVERVIEW
    # ===============================
    print("\n===== 1. DATA OVERVIEW =====")
    print("Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nSummary Statistics:\n", df.describe())

    # ===============================
    # 2. TARGET DISTRIBUTION
    # ===============================
    print("\n===== 2. TARGET DISTRIBUTION =====")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='Engine Condition', data=df, ax=ax)
    ax.set_title('Figure 1: Engine Condition Distribution (0: Normal, 1: Faulty)')
    fig.savefig(f'{plot_dir}/01_target_distribution.png')
    plt.close(fig)

    # ===============================
    # 3. UNIVARIATE ANALYSIS
    # ===============================
    print("\n===== 3. UNIVARIATE ANALYSIS =====")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        fig.savefig(f'{plot_dir}/02_hist_{col}.png')
        plt.close(fig)

    # ===============================
    # 4. BIVARIATE ANALYSIS
    # ===============================
    print("\n===== 4. BIVARIATE ANALYSIS =====")

    for col in numeric_cols:
        if col != 'Engine Condition':
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(x='Engine Condition', y=col, data=df, ax=ax)
            ax.set_title(f'{col} vs Engine Condition')
            fig.savefig(f'{plot_dir}/03_box_{col}.png')
            plt.close(fig)

    # ===============================
    # 5. MULTIVARIATE ANALYSIS
    # ===============================
    print("\n===== 5. MULTIVARIATE ANALYSIS =====")

    fig, ax = plt.subplots(figsize=(10,8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    fig.savefig(f'{plot_dir}/04_correlation_heatmap.png')
    plt.close(fig)

    # ===============================
    # 6. OUTLIER ANALYSIS
    # ===============================
    print("\n===== 6. OUTLIER ANALYSIS =====")

    pressure_cols = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure']
    existing_cols = [col for col in pressure_cols if col in df.columns]

    if existing_cols:
        fig, ax = plt.subplots(figsize=(12,6))
        sns.boxplot(data=df[existing_cols], ax=ax)
        ax.set_title('Pressure Feature Outliers')
        fig.savefig(f'{plot_dir}/05_pressure_outliers.png')
        plt.close(fig)

    # ===============================
    # 7. CLASS-WISE ANALYSIS
    # ===============================
    print("\n===== 7. CLASS-WISE FEATURE ANALYSIS =====")

    if 'Engine Condition' in df.columns:
        class_mean = df.groupby('Engine Condition').mean()
        print(class_mean)
        class_mean.to_csv(f'{plot_dir}/classwise_means.csv')
    print(f"\nEDA Complete. Plots saved to {plot_dir}/")


if __name__ == "__main__":
    perform_eda('Predictive_Maintenance/data/engine_data.csv')
