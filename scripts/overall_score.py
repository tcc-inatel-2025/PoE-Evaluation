from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
from scipy.stats import spearmanr

# Configuration
SUMMARY_GLOB = "../results/summary/*_summary.csv"
OUTPUT_DIR = Path("../results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics to compare against overall_score (in order of importance)
PREFERRED_METRICS = [
    "functional_correctness",
    "efficiency_score",
    "avg_cyclomatic_complexity",
    "max_cyclomatic_complexity",
    "style_score",
    "loc_score",
]

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


def load_summary_files(pattern: str) -> pd.DataFrame:
    """Load all matching CSVs and attach a 'model' column (derived from filename).
    Returns a concatenated DataFrame (may be empty).
    """
    files = glob.glob(pattern)
    print(f"Found {len(files)} summary files")
    dfs = []
    for f in files:
        model_name = Path(f).stem.replace("_summary", "")
        try:
            df_temp = pd.read_csv(f)
        except Exception as e:
            print(f"WARNING: failed to read '{f}': {e}")
            continue
        df_temp["model"] = model_name
        dfs.append(df_temp)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def sanitize_series(s: pd.Series) -> pd.Series:
    """Attempt to coerce series to numeric and drop NaNs for plotting."""
    s = pd.to_numeric(s, errors="coerce")
    return s


def plot_overall_score_distribution(df: pd.DataFrame, output_root: Path):
    """Create box plots and violin plots showing overall_score distribution by model.
    Saves files to output_root/distribution_plots.png
    """
    if df.empty:
        print("No data available for distribution plots.")
        return
    
    # Clean the data
    df_clean = df.copy()
    df_clean["overall_score"] = sanitize_series(df_clean["overall_score"])
    df_clean = df_clean.dropna(subset=["overall_score"])
    
    if df_clean.empty:
        print("No valid overall_score data for distribution plots.")
        return
    
    models = df_clean["model"].unique()
    print(f"Creating distribution plots for models: {models}")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    box_data = [df_clean[df_clean["model"] == model]["overall_score"].values for model in models]
    bp = ax1.boxplot(box_data, labels=models, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title("Overall Score Distribution by Model (Box Plot)")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Overall Score")
    ax1.tick_params(axis='x', rotation=45)
    
    # Violin plot
    violin_data = []
    positions = []
    for i, model in enumerate(models):
        model_data = df_clean[df_clean["model"] == model]["overall_score"].values
        violin_data.append(model_data)
        positions.append(i + 1)
    
    parts = ax2.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(models, rotation=45)
    ax2.set_title("Overall Score Distribution by Model (Violin Plot)")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Overall Score")
    
    plt.tight_layout()
    
    # Save the figure
    out_file = output_root / "distribution_plots.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved distribution plots -> {out_file}")


def plot_rank_concordance(df: pd.DataFrame, output_root: Path, metrics: list):
    """Create rank concordance plots comparing overall_score rankings vs other metrics.
    Shows how well overall_score correlates with individual metrics in terms of model rankings.
    Saves files to output_root/rank_concordance_plots.png
    """
    if df.empty:
        print("No data available for rank concordance plots.")
        return
    
    # Clean the data
    df_clean = df.copy()
    df_clean["overall_score"] = sanitize_series(df_clean["overall_score"])
    df_clean = df_clean.dropna(subset=["overall_score"])
    
    if df_clean.empty:
        print("No valid overall_score data for rank concordance plots.")
        return
    
    # Calculate mean scores per model for ranking
    model_stats = df_clean.groupby("model").agg({
        "overall_score": "mean",
        **{metric: "mean" for metric in metrics if metric in df_clean.columns}
    }).reset_index()
    
    available_metrics = [m for m in metrics if m in model_stats.columns]
    if not available_metrics:
        print("No available metrics for rank concordance analysis.")
        return
    
    print(f"Creating rank concordance plots for metrics: {available_metrics}")
    
    # Calculate number of subplots needed
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        # Calculate rankings (1 = best, higher = worse)
        overall_ranks = model_stats["overall_score"].rank(ascending=False, method="average")
        metric_ranks = model_stats[metric].rank(ascending=False, method="average")
        
        # Create scatter plot
        ax.scatter(overall_ranks, metric_ranks, s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add model labels
        for _, row in model_stats.iterrows():
            ax.annotate(row["model"], 
                       (overall_ranks[row.name], metric_ranks[row.name]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        # Add perfect correlation line
        min_rank = min(overall_ranks.min(), metric_ranks.min())
        max_rank = max(overall_ranks.max(), metric_ranks.max())
        ax.plot([min_rank, max_rank], [min_rank, max_rank], 'r--', alpha=0.5, linewidth=2)
        
        # Calculate and display Spearman correlation
        correlation, p_value = spearmanr(overall_ranks, metric_ranks)
        
        ax.set_xlabel("Overall Score Rank")
        ax.set_ylabel(f"{metric} Rank")
        ax.set_title(f"Rank Concordance: Overall Score vs {metric}\n(Spearman ρ = {correlation:.3f}, p = {p_value:.3f})")
        ax.grid(True, alpha=0.3)
        
        # Set equal axis limits for better comparison
        ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
        ax.set_ylim(min_rank - 0.5, max_rank + 0.5)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure
    out_file = output_root / "rank_concordance_plots.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved rank concordance plots -> {out_file}")


def plot_delta_improvement(df: pd.DataFrame, output_root: Path, metrics: list):
    """Create delta/improvement plots showing nuances captured by overall_score vs other metrics.
    Shows where overall_score provides different insights compared to individual metrics.
    Saves files to output_root/delta_improvement_plots.png
    """
    if df.empty:
        print("No data available for delta improvement plots.")
        return
    
    # Clean the data
    df_clean = df.copy()
    df_clean["overall_score"] = sanitize_series(df_clean["overall_score"])
    df_clean = df_clean.dropna(subset=["overall_score"])
    
    if df_clean.empty:
        print("No valid overall_score data for delta improvement plots.")
        return
    
    # Calculate mean scores per model
    model_stats = df_clean.groupby("model").agg({
        "overall_score": "mean",
        **{metric: "mean" for metric in metrics if metric in df_clean.columns}
    }).reset_index()
    
    available_metrics = [m for m in metrics if m in model_stats.columns]
    if not available_metrics:
        print("No available metrics for delta improvement analysis.")
        return
    
    print(f"Creating delta improvement plots for metrics: {available_metrics}")
    
    # Normalize scores to 0-1 scale for fair comparison
    model_stats_norm = model_stats.copy()
    model_stats_norm["overall_score"] = (model_stats_norm["overall_score"] - model_stats_norm["overall_score"].min()) / (model_stats_norm["overall_score"].max() - model_stats_norm["overall_score"].min())
    
    for metric in available_metrics:
        if metric in model_stats_norm.columns:
            min_val = model_stats_norm[metric].min()
            max_val = model_stats_norm[metric].max()
            if max_val > min_val:  # Avoid division by zero
                model_stats_norm[metric] = (model_stats_norm[metric] - min_val) / (max_val - min_val)
    
    # Calculate number of subplots needed
    n_metrics = len(available_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        # Calculate deltas (overall_score - metric_score)
        deltas = model_stats_norm["overall_score"] - model_stats_norm[metric]
        
        # Create bar plot of deltas
        bars = ax.bar(range(len(model_stats)), deltas, 
                     color=['green' if d > 0 else 'red' for d in deltas], 
                     alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for j, (bar, delta) in enumerate(zip(bars, deltas)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                   f'{delta:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=8, fontweight='bold')
        
        ax.set_xlabel("Models")
        ax.set_ylabel("Delta (Overall Score - " + metric + ")")
        ax.set_title(f"Delta Analysis: Overall Score vs {metric}\n(Green: Overall better, Red: Metric better)")
        ax.set_xticks(range(len(model_stats)))
        ax.set_xticklabels(model_stats["model"], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add summary statistics
        positive_deltas = deltas[deltas > 0].count()
        negative_deltas = deltas[deltas < 0].count()
        mean_delta = deltas.mean()
        
        ax.text(0.02, 0.98, f'Positive deltas: {positive_deltas}/{len(deltas)}\nMean delta: {mean_delta:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure
    out_file = output_root / "delta_improvement_plots.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved delta improvement plots -> {out_file}")


def plot_per_model_scatter(df: pd.DataFrame, output_root: Path, metrics: list):
    """For each unique model in df, create a multi-row scatter plot of overall_score vs each available metric.
    Saves files to output_root/<model>/scatter_overall_vs_metrics.png
    """
    if df.empty:
        print("No data available to plot.")
        return

    models = df["model"].unique()
    print(f"Generating plots for models: {models}")

    for model in models:
        df_model = df[df["model"] == model].copy()
        if df_model.empty:
            print(f"Skipping empty model: {model}")
            continue

        # Check available metrics for this model
        available_metrics = [m for m in metrics if m in df_model.columns]
        # Also ensure overall_score exists
        if "overall_score" not in df_model.columns:
            print(f"Skipping model '{model}': 'overall_score' column missing")
            continue

        if not available_metrics:
            print(f"Skipping model '{model}': none of the preferred metrics found in columns: {df_model.columns.tolist()}")
            continue

        # sanitize
        df_model["overall_score"] = sanitize_series(df_model["overall_score"])
        # drop rows where overall_score is NaN
        df_model = df_model.dropna(subset=["overall_score"]) 
        if df_model.empty:
            print(f"Skipping model '{model}': all overall_score values are NaN after coercion")
            continue

        num_metrics = len(available_metrics)
        # Create a vertical grid of scatterplots (one column)
        fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(10, 3 * num_metrics), squeeze=False)

        for i, metric in enumerate(available_metrics):
            ax = axes[i, 0]
            # Coerce metric to numeric
            series_metric = sanitize_series(df_model[metric])
            # Align lengths and drop rows where metric is NaN
            plot_df = pd.DataFrame({"overall_score": df_model["overall_score"], metric: series_metric}).dropna()
            if plot_df.empty:
                ax.set_visible(False)
                continue

            # Scatter plot: jitter x slightly if it's discrete, alpha for density
            x = plot_df[metric].values
            y = plot_df["overall_score"].values

            # If metric is integer-like with few unique values, add jitter on x
            if np.issubdtype(plot_df[metric].dtype, np.number) and len(np.unique(np.round(x, 6))) < 15:
                jitter = (np.random.rand(len(x)) - 0.5) * (np.ptp(x) * 0.02 + 1e-6)
                x = x + jitter

            ax.scatter(x, y, alpha=0.7, edgecolors='none')
            ax.set_xlabel(metric)
            ax.set_ylabel("overall_score")
            ax.set_title(f"{model}: overall_score vs {metric}")

            # Add a simple linear trend line if there are enough points
            if len(x) >= 3:
                try:
                    m, b = np.polyfit(x, y, 1)
                    xs = np.linspace(np.min(x), np.max(x), 100)
                    ax.plot(xs, m * xs + b, linestyle='--', linewidth=1)
                except Exception:
                    # If polyfit fails, silently skip trendline
                    pass

        plt.tight_layout()

        # Save per-model figure
        model_out_dir = output_root / model
        model_out_dir.mkdir(parents=True, exist_ok=True)
        out_file = model_out_dir / "scatter_overall_vs_metrics.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved scatter plot for model '{model}' -> {out_file}")


if __name__ == "__main__":
    df_all = load_summary_files(SUMMARY_GLOB)
    
    # Create distribution plots (box plots and violin plots)
    plot_overall_score_distribution(df_all, OUTPUT_DIR)
    
    # Create rank concordance plots
    plot_rank_concordance(df_all, OUTPUT_DIR, PREFERRED_METRICS)
    
    # Create delta improvement plots
    plot_delta_improvement(df_all, OUTPUT_DIR, PREFERRED_METRICS)
    
    # Create per-model scatter plots
    plot_per_model_scatter(df_all, OUTPUT_DIR, PREFERRED_METRICS)
    
    print("Done.")
