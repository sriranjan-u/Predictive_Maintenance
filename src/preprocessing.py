import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from huggingface_hub import HfApi

def clean_and_register_data(repo_id):
    print("===== STARTING DATA PREPARATION & CLEANING =====")
    
    # 1. Load from Hub
    url = f"https://huggingface.co/datasets/{repo_id}/raw/main/engine_data.csv"
    try:
        df = pd.read_csv(url)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Keep a copy of the raw data for 'Before' comparisons
    df_raw = df.copy()

    # ==========================================
    # 2. IQR OUTLIER CAPPING (THE CLEANING PHASE)
    # ==========================================
    sensor_cols = ['Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp', 'Engine rpm']
    
    print(f"Applying IQR Capping to {len(sensor_cols)} sensor columns to neutralize extreme outliers...")
    
    for col in sensor_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
    print("IQR Capping successfully applied.")

    # ==========================================
    # 3. VISUALIZATION: Before & After + Scatter
    # ==========================================
    print("Generating and displaying Before/After and Scatter plots...")
    plot_dir = 'Predictive_Maintenance/plots/'
    os.makedirs(plot_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Before and After Boxplots & Histograms
    for col in sensor_cols:
        if col in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Outlier Management: {col} (Before vs After IQR Capping)', fontsize=14, y=1.02)

            # Before Capping
            sns.boxplot(x=df_raw[col], ax=axes[0, 0], color='lightcoral')
            axes[0, 0].set_title('Boxplot (Before)')
            sns.histplot(df_raw[col], kde=True, ax=axes[0, 1], color='lightcoral')
            axes[0, 1].set_title('Histogram (Before)')

            # After Capping
            sns.boxplot(x=df[col], ax=axes[1, 0], color='lightgreen')
            axes[1, 0].set_title('Boxplot (After)')
            sns.histplot(df[col], kde=True, ax=axes[1, 1], color='lightgreen')
            axes[1, 1].set_title('Histogram (After)')

            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{col}_outlier_treatment.png', dpi=300, bbox_inches='tight')
            plt.close(fig) # Prevent duplicate inline rendering if running locally

    # ==========================================
    # NEW: Side-by-Side Scatter Plot Comparison
    # ==========================================
    if all(c in df.columns for c in ['Lub oil pressure', 'Coolant pressure', 'Engine Condition']):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Feature Interaction: Lub Oil Pressure vs Coolant Pressure', fontsize=16, y=1.05)

        # Before (Raw Data)
        sns.scatterplot(data=df_raw, x='Lub oil pressure', y='Coolant pressure', hue='Engine Condition', alpha=0.6, palette='Set1', ax=axes[0])
        axes[0].set_title('Before Capping: Raw Data (Extreme Outliers Visible)')

        # After (Cleaned Data)
        sns.scatterplot(data=df, x='Lub oil pressure', y='Coolant pressure', hue='Engine Condition', alpha=0.6, palette='Set1', ax=axes[1])
        axes[1].set_title('After Capping: Clean Data Clusters')

        plt.tight_layout()
        plt.savefig(f'{plot_dir}/scatter_lub_vs_coolant_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()  # Display this specific comparison directly in Colab

    print(f"All plots successfully generated and saved to {plot_dir}/")

    # ==========================================
    # 4. Save Full Processed Dataset & Split
    # ==========================================
    print("Saving the full processed dataset and splitting into Train/Test...")
    os.makedirs('Predictive_Maintenance/data', exist_ok=True)
    
    # Save the full processed dataset
    processed_path = 'Predictive_Maintenance/data/processed_engine_data.csv'
    df.to_csv(processed_path, index=False)

    # Train-Test Split
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Engine Condition'])

    train_path = 'Predictive_Maintenance/data/train.csv'
    test_path = 'Predictive_Maintenance/data/test.csv'
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print("Train and Test splits saved locally.")

    # ==========================================
    # 5. Cloud Upload (Hugging Face)
    # ==========================================
    token = os.getenv("HF_TOKEN")
    if token:
        print("Initiating upload to Hugging Face...")
        api = HfApi(token=token)
        
        # Upload Full Processed Data
        api.upload_file(path_or_fileobj=processed_path, path_in_repo="processed_engine_data.csv", repo_id=repo_id, repo_type="dataset")
        # Upload Train Data
        api.upload_file(path_or_fileobj=train_path, path_in_repo="train.csv", repo_id=repo_id, repo_type="dataset")
        # Upload Test Data
        api.upload_file(path_or_fileobj=test_path, path_in_repo="test.csv", repo_id=repo_id, repo_type="dataset")
        
        print(f"Full processed dataset, Train, and Test sets uploaded to HF Dataset: {repo_id}")
    else:
        print("HF_TOKEN not found. Cloud upload skipped.")

if __name__ == "__main__":
    REPO_ID = "Sriranjan/Predictive_Maintenance_Data"
    clean_and_register_data(REPO_ID)
