import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_metrics():
    print("Generating model performance visualizations...")
    
    # Load the comparison data
    report_path = "Predictive_Maintenance/reports/model_comparison.csv"
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Could not find {report_path}. Run training first.")
        
    df = pd.read_csv(report_path)
    
    # Create the output directory
    plot_dir = 'Predictive_Maintenance/plots/evaluation'
    os.makedirs(plot_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # We will focus on Recall and F1-Score as they are the most critical for this business problem
    metrics_to_plot = ['Recall', 'F1']
    
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort values so the best models appear at the top
        df_sorted = df.sort_values(by=metric, ascending=False)
        
        # Create a grouped bar chart
        sns.barplot(
            data=df_sorted, 
            x=metric, 
            y='Model', 
            hue='Type', 
            palette=['#1f77b4', '#ff7f0e'], # Blue for baseline, Orange for tuned
            alpha=0.9
        )
        
        ax.set_title(f'Model Comparison: {metric} Score (Baseline vs. Tuned)', fontsize=16, pad=15)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_ylabel('Algorithm', fontsize=12)
        
        # Add a vertical line to highlight the best score
        best_score = df[metric].max()
        ax.axvline(best_score, color='red', linestyle='--', alpha=0.5, label=f'Best {metric}: {best_score:.4f}')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        save_path = f"{plot_dir}/01_{metric.lower()}_comparison.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        
    print(f"Visualizations saved to {plot_dir}/")

if __name__ == "__main__":
    plot_metrics()
