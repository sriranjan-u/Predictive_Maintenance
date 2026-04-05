import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

def perform_eda(df_path):
    print(f"\n===== STARTING MACRO EDA ON: {df_path} =====")

    # Load data
    df = pd.read_csv(df_path)

    # Create a dedicated eda plots directory
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # ===============================
    # DATA OVERVIEW
    # ===============================
    print("\n===== 1. DATA OVERVIEW =====")
    print("Shape:", df.shape)
    print("\nSummary Statistics:\n", df.describe())

    # ===============================
    # TARGET DISTRIBUTION
    # ===============================
    print("\n===== 2. GENERATING TARGET DISTRIBUTION =====")
    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(x='Engine Condition', data=df, ax=ax, hue='Engine Condition', palette='Set2', legend=False)

    ax.set_title('Engine Condition Distribution (0: Healthy, 1: Faulty)')
    fig.savefig(f'{plot_dir}/01_target_distribution.png', bbox_inches='tight')
    plt.close(fig)

    # ===============================
    # CORRELATION HEATMAP
    # ===============================
    print("\n===== 3. GENERATING CORRELATION HEATMAP =====")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    fig, ax = plt.subplots(figsize=(10,8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Heatmap')
    fig.savefig(f'{plot_dir}/02_correlation_heatmap.png', bbox_inches='tight')
    plt.close(fig)

    # ===============================
    # MULTIVARIATE PAIRPLOT (NO UNIVARIATE DIAGONALS)
    # ===============================
    print("\n===== 4. GENERATING MULTIVARIATE PAIRPLOT =====")

    df_sample = df.sample(n=min(1000, len(df)), random_state=42) if len(df) > 1000 else df

    g = sns.pairplot(df_sample, hue='Engine Condition', corner=True, palette='Set1',
                     plot_kws={'alpha':0.6}, diag_kind=None)
    g.fig.suptitle('Multivariate Feature Interactions', y=1.02, fontsize=16)
    g.savefig(f'{plot_dir}/03_multivariate_pairplot.png', bbox_inches='tight')
    plt.close(g.fig)

    # ===============================
    # CLASS-WISE ANALYSIS & INDIVIDUAL PLOTS
    # ===============================
    print("\n===== 5. CALCULATING & PLOTTING CLASS-WISE MEANS =====")
    if 'Engine Condition' in df.columns:
        # Calculate and save the CSV
        class_mean = df.groupby('Engine Condition').mean()
        class_mean.to_csv(f'{plot_dir}/04_classwise_means.csv')

        # Generate Separate Subplots for Each Sensor
        features = class_mean.columns
        num_features = len(features)

        # Create a 2-row by 3-column grid (adjusts automatically based on feature count)
        cols = 3
        rows = math.ceil(num_features / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle('Average Sensor Values: Healthy (0) vs. Faulty (1) Engines', fontsize=16, y=1.02)

        axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

        for i, feature in enumerate(features):
            sns.barplot(
                x=class_mean.index,
                y=class_mean[feature],
                ax=axes[i],
                hue=class_mean.index,
                palette='Set1',
                legend=False
            )
            axes[i].set_title(f'{feature}')
            axes[i].set_ylabel('Mean Value')
            axes[i].set_xlabel('Engine Condition')

        # Hide any unused subplots (e.g., if you have 5 features in a 6-slot grid)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        # Save as a new file name to clearly differentiate it from the old grouped barplot
        fig.savefig(f'{plot_dir}/05_classwise_means_subplots.png', bbox_inches='tight')
        plt.close(fig)

    print(f"\nEDA Complete. Assets saved to {plot_dir}/")

if __name__ == "__main__":
    perform_eda('data/processed_engine_data.csv')
