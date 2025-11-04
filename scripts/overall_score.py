from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
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

# Theme
sns.set_theme(style="whitegrid", palette="Set3", font_scale=3, font="sans-serif", rc={
    "figure.figsize": (10, 5),
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})


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
            df_temp["model"] = model_name
            dfs.append(df_temp)
        except Exception as e:
            print(f"WARNING: failed to read '{f}': {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


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
    """
    Create rank concordance plots comparing overall_score rankings vs other metrics.
    Saves one file per metric and also a combined file with all metrics.
    """
    if df.empty:
        print("No data available for rank concordance plots.")
        return

    df_clean = df.copy()
    df_clean["overall_score"] = sanitize_series(df_clean["overall_score"])
    df_clean = df_clean.dropna(subset=["overall_score"])
    if df_clean.empty:
        print("No valid overall_score data for rank concordance plots.")
        return

    # Mean per model
    model_stats = df_clean.groupby("model").agg({
        "overall_score": "mean",
        **{m: "mean" for m in metrics if m in df_clean.columns}
    }).reset_index()

    available_metrics = [m for m in metrics if m in model_stats.columns]
    if not available_metrics:
        print("No available metrics for rank concordance analysis.")
        return

    print(f"Creating rank concordance plots for metrics: {available_metrics}")

    # Prepare colors: one color per model (consistent across plots)
    models = model_stats["model"].tolist()
    model_palette = sns.color_palette("Set3", n_colors=len(models))
    model_color_map = {m: model_palette[i] for i, m in enumerate(models)}

    overall_ranks = model_stats["overall_score"].rank(ascending=False, method="average")

    # Combined figure setup
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig_combined, axes_combined = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_combined = np.array(axes_combined).reshape(-1)

    for i, metric in enumerate(available_metrics):
        metric_ranks = model_stats[metric].rank(ascending=False, method="average")

        # Individual figure
        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_rank_concordance_core(ax, model_stats, overall_ranks, metric_ranks, metric, model_color_map)
        rank_dir = output_root / "rank_concordance" 
        rank_dir.mkdir(parents=True, exist_ok=True)
        out_file_ind = rank_dir / f"rank_concordance_{metric}.png"
        fig.savefig(out_file_ind, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved rank concordance plot -> {out_file_ind}")

        # Populate combined subplot
        axc = axes_combined[i]
        _plot_rank_concordance_core(axc, model_stats, overall_ranks, metric_ranks, metric, model_color_map)

    # Hide remaining axes in combined
    for j in range(len(available_metrics), len(axes_combined)):
        axes_combined[j].set_visible(False)

    plt.tight_layout()
    out_file_comb = output_root / "rank_concordance_plots.png"
    fig_combined.savefig(out_file_comb, dpi=300, bbox_inches="tight")
    plt.close(fig_combined)
    print(f"Saved combined rank concordance plots -> {out_file_comb}")


def _plot_rank_concordance_core(ax, model_stats, overall_ranks, metric_ranks, metric, model_color_map):
    """Core plotting routine for a rank concordance axis."""
    # Scatter with model-specific colors
    xs = overall_ranks.values
    ys = metric_ranks.values
    colors = [model_color_map[m] for m in model_stats["model"]]
    ax.scatter(xs, ys, s=90, alpha=0.9, edgecolors="black", linewidth=0.6, c=colors)

    # Annotate each point with model name
    for idx, row in model_stats.iterrows():
        ax.annotate(row["model"], (xs[idx], ys[idx]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, alpha=0.85)

    # Perfect correlation line
    min_rank = min(xs.min(), ys.min())
    max_rank = max(xs.max(), ys.max())
    ax.plot([min_rank, max_rank], [min_rank, max_rank], "r--", alpha=0.35, linewidth=1.5)

    # Spearman
    corr, p_val = spearmanr(xs, ys)

    # Formatting
    ax.set_xlabel("Overall Score Rank", fontsize=10)
    ax.set_ylabel(f"{metric} Rank", fontsize=10)
    ax.set_title(f"Rank Concordance: Overall vs {metric}\nρ={corr:.3f}, p={p_val:.3f}", fontsize=11, fontweight="semibold")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(min_rank - 0.5, max_rank + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#fbfbfb")


def plot_delta_improvement(df: pd.DataFrame, output_root: Path, metrics: list):
    """
    Create delta/improvement plots showing where overall_score differs from each metric.
    Saves one file per metric and a combined figure.
    """
    if df.empty:
        print("No data available for delta improvement plots.")
        return

    df_clean = df.copy()
    df_clean["overall_score"] = sanitize_series(df_clean["overall_score"])
    df_clean = df_clean.dropna(subset=["overall_score"])
    if df_clean.empty:
        print("No valid overall_score data for delta improvement plots.")
        return

    model_stats = df_clean.groupby("model").agg({
        "overall_score": "mean",
        **{m: "mean" for m in metrics if m in df_clean.columns}
    }).reset_index()

    available_metrics = [m for m in metrics if m in model_stats.columns]
    if not available_metrics:
        print("No available metrics for delta improvement analysis.")
        return

    print(f"Creating delta improvement plots for metrics: {available_metrics}")

    # Color palette per model
    models = model_stats["model"].tolist()
    model_palette = sns.color_palette("Set3", n_colors=len(models))
    model_color_map = {m: model_palette[i] for i, m in enumerate(models)}

    # Normalize scores (0-1) per column to keep comparisons fair
    def _normalize_series(s):
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min())
        return s - s.min()

    model_stats_norm = model_stats.copy()
    model_stats_norm["overall_score"] = _normalize_series(model_stats_norm["overall_score"])
    for m in available_metrics:
        model_stats_norm[m] = _normalize_series(model_stats_norm[m])

    # Combined figure setup
    n_metrics = len(available_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig_combined, axes_combined = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes_combined = np.array(axes_combined).reshape(-1)

    for i, metric in enumerate(available_metrics):
        # Individual
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_delta_core(ax, model_stats_norm, metric, model_color_map)
        rank_dir = output_root / "delta_improvement" 
        rank_dir.mkdir(parents=True, exist_ok=True)
        out_file_ind = rank_dir / f"delta_improvement_{metric}.png"
        fig.savefig(out_file_ind, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved delta improvement plot -> {out_file_ind}")

        # Combined
        _plot_delta_core(axes_combined[i], model_stats_norm, metric, model_color_map)

    # Hide extra combined axes
    for j in range(len(available_metrics), len(axes_combined)):
        axes_combined[j].set_visible(False)

    plt.tight_layout()
    out_file_comb = output_root / "delta_improvement_plots.png"
    fig_combined.savefig(out_file_comb, dpi=300, bbox_inches="tight")
    plt.close(fig_combined)
    print(f"Saved combined delta improvement plots -> {out_file_comb}")


def _plot_delta_core(ax, model_stats_norm, metric, model_color_map):
    """Core plotting for delta bars for a given metric on axis ax."""
    models = model_stats_norm["model"].tolist()
    deltas = (model_stats_norm["overall_score"] - model_stats_norm[metric]).values

    # Colors: use green/red but keep alpha consistent; we still have palette for marker consistency
    # Use Set3 palette colors for consistency (greenish/red tones)
    palette = sns.color_palette("Set3")
    positive_color = palette[1]  # green pastel
    negative_color = palette[0]  # red pastel
    
    bar_colors = [positive_color if d > 0 else negative_color for d in deltas] 
    bars = ax.bar(models, deltas, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.4)

    # Value labels
    for bar, delta in zip(bars, deltas):
        h = bar.get_height()
        y_offset = 0.01 if h >= 0 else -0.01
        va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, h + y_offset, f"{delta:.3f}",
                ha="center", va=va, fontsize=8, fontweight="semibold")

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(f"Δ Improvement: Overall vs {metric}", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Δ (Overall - Metric)")
    ax.set_xticklabels(models, rotation=45, ha="right")
    mean_delta = deltas.mean()
    positives = (deltas > 0).sum()
    ax.text(0.02, 0.98, f"Mean Δ: {mean_delta:.3f}\nPositive: {positives}/{len(deltas)}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_facecolor("#fbfbfb")


def plot_per_model_scatter(df: pd.DataFrame, output_root: Path, metrics: list):
    """
    For each model, create a vertical stack of scatter plots (overall_score vs each metric).
    Saves one image per model in output_root/<model>/scatter_overall_vs_metrics.png
    """
    if df.empty:
        print("No data available to plot per-model scatter.")
        return

    models = sorted(df["model"].unique())
    print(f"Generating plots for models: {models}")

    for model in models:
        df_model = df[df["model"] == model].copy()
        if df_model.empty:
            continue

        available_metrics = [m for m in metrics if m in df_model.columns]
        if "overall_score" not in df_model.columns or not available_metrics:
            print(f"Skipping model '{model}': insufficient columns")
            continue

        df_model["overall_score"] = sanitize_series(df_model["overall_score"])
        df_model = df_model.dropna(subset=["overall_score"])
        if df_model.empty:
            print(f"Skipping model '{model}': no valid overall_score values")
            continue

        n = len(available_metrics)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 3 * n), squeeze=False)

        for i, metric in enumerate(available_metrics):
            ax = axes[i, 0]
            series_metric = sanitize_series(df_model[metric])
            plot_df = pd.DataFrame({"overall_score": df_model["overall_score"], metric: series_metric}).dropna()
            if plot_df.empty:
                ax.set_visible(False)
                continue

            x = plot_df[metric].values
            y = plot_df["overall_score"].values

            # Jitter if discrete
            if np.issubdtype(plot_df[metric].dtype, np.number) and len(np.unique(np.round(x, 6))) < 15:
                jitter = (np.random.rand(len(x)) - 0.5) * (np.ptp(x) * 0.02 + 1e-6)
                x = x + jitter

            ax.scatter(x, y, alpha=0.75, edgecolors="none")
            ax.set_xlabel(metric)
            ax.set_ylabel("overall_score")
            ax.set_title(f"{model}: overall_score vs {metric}", fontsize=10)

            # Trend line if enough points
            if len(x) >= 3:
                try:
                    m, b = np.polyfit(x, y, 1)
                    xs = np.linspace(np.min(x), np.max(x), 100)
                    ax.plot(xs, m * xs + b, linestyle="--", linewidth=1)
                except Exception:
                    pass

            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        model_out_dir = output_root / model
        model_out_dir.mkdir(parents=True, exist_ok=True)
        out_file = model_out_dir / "scatter_overall_vs_metrics.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved scatter plot for model '{model}' -> {out_file}")


if __name__ == "__main__":
    df_all = load_summary_files(SUMMARY_GLOB)

    # Distribution: box + violin combined
    plot_overall_score_distribution(df_all, OUTPUT_DIR)

    # Rank concordance: individual + combined
    plot_rank_concordance(df_all, OUTPUT_DIR, PREFERRED_METRICS)

    # Delta improvement: individual + combined
    plot_delta_improvement(df_all, OUTPUT_DIR, PREFERRED_METRICS)

    # Per-model scatter plots
    plot_per_model_scatter(df_all, OUTPUT_DIR, PREFERRED_METRICS)

    print("Done.")