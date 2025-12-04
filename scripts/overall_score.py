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

#theme
sns.set_theme(style="whitegrid", rc={
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.figsize": (10, 5.5),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "grid.alpha": 0.25,
})

# Beautiful red → green divergent palette (12 steps)
divergent_cmap = sns.diverging_palette(
    h_neg=355,   # red
    h_pos=130,   # green
    s=90, l=55,
    sep=10,
    n=12,
    center="light"
)

sns.set_palette(divergent_cmap)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=divergent_cmap)


def load_summary_files(pattern: str) -> pd.DataFrame:
    """Load all matching CSVs and attach a 'model' column (derived from filename)."""
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
    return pd.to_numeric(s, errors="coerce")

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
    bp = ax1.boxplot(box_data, labels=models, patch_artist=True, showfliers=False)
    
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
    fig.savefig(out_file, dpi=200, bbox_inches="tight")  # Reduced DPI for memory
    plt.close(fig)
    print(f"Saved distribution plots -> {out_file}")
    

def plot_rank_concordance(df: pd.DataFrame, output_root: Path, metrics: list):
    """
    Create rank concordance plots comparing overall_score rankings vs other metrics at task level.
    Memory-optimized version that processes one metric at a time.
    """
    import gc
    
    if df.empty:
        print("No data available for rank concordance plots.")
        return

    # Minimal initial processing
    df_clean = df.copy()
    df_clean["overall_score"] = sanitize_series(df_clean["overall_score"])
    df_clean = df_clean.dropna(subset=["overall_score"])
    if df_clean.empty:
        print("No valid overall_score data for rank concordance plots.")
        return

    available_metrics = [m for m in metrics if m in df_clean.columns]
    if not available_metrics:
        print("No available metrics for rank concordance analysis.")
        return

    print(f"Creating task-level rank concordance plots for metrics: {available_metrics}")

    # Color map per model (compute once)
    models = sorted(df_clean["model"].unique())
    model_palette = sns.color_palette("Set3", n_colors=len(models))
    model_color_map = {m: model_palette[i] for i, m in enumerate(models)}

    rank_dir = output_root / "rank_concordance"
    rank_dir.mkdir(parents=True, exist_ok=True)

    # === 1. Rank Concordance Plots (individual + combined) ===
    # Process metrics one at a time to save memory
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig_combined, axes_combined = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes_combined = np.array(axes_combined).ravel()

    plot_idx = 0
    for i, metric in enumerate(available_metrics):
        print(f"  Processing {metric} ({i+1}/{n_metrics})...", end="")
        
        # Sanitize only this metric
        df_clean[metric] = sanitize_series(df_clean[metric])
        
        # Filter to valid data (where both overall_score and metric are available)
        valid_mask = ~(df_clean["overall_score"].isna() | df_clean[metric].isna())
        valid_data = df_clean.loc[valid_mask, ["model", "overall_score", metric]].copy()
        
        if len(valid_data) < 2:
            print(f" skipped (insufficient data, n={len(valid_data)})")
            continue

        # Calculate Spearman correlation at task level (use numpy arrays directly)
        overall_values = valid_data["overall_score"].values
        metric_values = valid_data[metric].values
        corr, p_val = spearmanr(overall_values, metric_values)

        # Individual plot - use lower DPI and smaller point size
        fig, ax = plt.subplots(figsize=(7, 6))
        _plot_rank_concordance_task_level(ax, valid_data, metric, model_color_map, corr, p_val)
        out_file_ind = rank_dir / f"rank_concordance_{metric}.png"
        fig.savefig(out_file_ind, dpi=200, bbox_inches="tight")  # Reduced DPI
        plt.close(fig)
        del fig, ax
        print(f" saved (n={len(valid_data)}, ρ={corr:.3f})")

        # Combined subplot - reuse the same data
        if plot_idx < len(axes_combined):
            axc = axes_combined[plot_idx]
            _plot_rank_concordance_task_level(axc, valid_data, metric, model_color_map, corr, p_val)
            axc.set_title(metric.replace("_", " ").title(), fontsize=11, pad=10)
            plot_idx += 1

        # Free memory
        del valid_data
        gc.collect()

    # Hide unused subplots
    for j in range(plot_idx, len(axes_combined)):
        axes_combined[j].set_visible(False)

    plt.tight_layout()
    combined_path = output_root / "rank_concordance_plots.png"
    fig_combined.savefig(combined_path, dpi=200, bbox_inches="tight")  # Reduced DPI
    plt.close(fig_combined)
    del fig_combined
    print(f"   Saved combined: {combined_path.name}")
    gc.collect()

    # === 2. Inter-metric Spearman Correlation Heatmap (Task-Level) ===
    print("   Generating inter-metric Spearman correlation matrix (task-level)...")
    
    # Get task-level data for all metrics (only rows where all metrics are available)
    # Process in chunks to avoid memory issues
    task_data = df_clean[["overall_score"] + available_metrics].copy()
    task_data = task_data.dropna()
    
    if len(task_data) < 2:
        print("   Warning: Insufficient data for inter-metric correlation matrix.")
    else:
        # Use only the metric columns for correlation (exclude overall_score from matrix)
        corr_data = task_data[available_metrics]
        
        # Calculate correlation matrix
        spearman_matrix = corr_data.corr(method='spearman')

        # Create heatmap
        fig_heatmap = plt.figure(figsize=(max(7, len(available_metrics) * 0.95), max(6, len(available_metrics) * 0.85)))
        mask = np.triu(np.ones_like(spearman_matrix, dtype=bool), k=1)

        sns.heatmap(spearman_matrix,
                    annot=True,
                    fmt=".3f",
                    cmap="coolwarm",
                    center=0,
                    vmin=-1, vmax=1,
                    square=True,
                    linewidths=0.7,
                    linecolor='gray',
                    cbar_kws={"shrink": 0.8},
                    mask=mask,
                    annot_kws={"size": 10, "weight": "bold"})

        plt.title(f"Spearman Rank Correlation Between Individual Metrics (Task-Level, n={len(task_data)})", 
                  fontsize=14, fontweight="bold", pad=20)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        heatmap_path = rank_dir / "inter_metric_spearman_correlation.png"
        plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")  # Reduced DPI
        plt.close(fig_heatmap)
        del fig_heatmap
        print(f"   Saved correlation heatmap → {heatmap_path.name}")

        # Save CSV tables
        csv_path = rank_dir / "inter_metric_spearman_correlation.csv"
        spearman_matrix.round(4).to_csv(csv_path)
        print(f"   Saved correlation table → {csv_path.name}")

        # Calculate p-values (this can be memory intensive, so do it carefully)
        try:
            # Use numpy arrays directly to save memory
            corr_array = corr_data.values
            _, pval = spearmanr(corr_array)
            pval_df = pd.DataFrame(pval, index=available_metrics, columns=available_metrics)
            pval_csv = rank_dir / "inter_metric_spearman_pvalues.csv"
            pval_df.round(4).to_csv(pval_csv)
            print(f"   Saved p-values → {pval_csv.name}")
            del corr_array, pval, pval_df
        except Exception as e:
            print(f"   Warning: Could not calculate p-values: {e}")

        # Free memory
        del task_data, corr_data, spearman_matrix
        gc.collect()

    # Final cleanup
    del df_clean
    gc.collect()
    print("All rank concordance + inter-metric correlation analysis complete!\n")


def _plot_rank_concordance_task_level(ax, task_data, metric, model_color_map, corr, p_val):
    """Core plotting routine for task-level rank concordance (memory-optimized)."""
    # Convert to numpy arrays for efficiency
    model_col = task_data["model"].values
    overall_values = task_data["overall_score"].values
    metric_values = task_data[metric].values
    
    # Sample data if too many points to reduce memory usage
    max_points = 5000
    if len(overall_values) > max_points:
        # Random sample
        np.random.seed(42)  # For reproducibility
        sample_idx = np.random.choice(len(overall_values), max_points, replace=False)
        model_col = model_col[sample_idx]
        overall_values = overall_values[sample_idx]
        metric_values = metric_values[sample_idx]
    
    # Scatter plot colored by model (use numpy for efficiency)
    models = np.unique(model_col)
    point_size = max(20, min(50, 5000 // len(overall_values)))  # Adaptive point size
    
    for model in models:
        model_mask = model_col == model
        model_overall = overall_values[model_mask]
        model_metric = metric_values[model_mask]
        if len(model_overall) > 0:
            ax.scatter(model_overall, model_metric, 
                      s=point_size, alpha=0.5, edgecolors="none",  # No edge to save memory
                      c=[model_color_map[model]], label=model)
    
    ax.set_xlabel("Overall Score", fontsize=10)
    ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=10)
    n_display = len(task_data) if len(overall_values) == len(task_data) else f"{len(overall_values)}/{len(task_data)}"
    ax.set_title(f"Task-Level: Overall vs {metric.replace('_', ' ').title()}\nρ = {corr:.3f} (p = {p_val:.3f}), n = {n_display}",
                 fontsize=11, fontweight="semibold")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_facecolor("#fbfbfb")
    ax.legend(loc="best", fontsize=8, ncol=1)

def _plot_rank_concordance_core(ax, model_stats, overall_ranks, metric_ranks, metric, model_color_map):
    """Core plotting routine for a rank concordance axis (model-level, deprecated)."""
    xs = overall_ranks.values
    ys = metric_ranks.values
    colors = [model_color_map[m] for m in model_stats["model"]]
    ax.scatter(xs, ys, s=90, alpha=0.9, edgecolors="black", linewidth=0.6, c=colors)

    for idx, row in model_stats.iterrows():
        ax.annotate(row["model"], (xs[idx], ys[idx]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, alpha=0.85)

    min_rank = min(xs.min(), ys.min())
    max_rank = max(xs.max(), ys.max())
    ax.plot([min_rank, max_rank], [min_rank, max_rank], "r--", alpha=0.35, linewidth=1.5)

    corr, p_val = spearmanr(xs, ys)
    ax.set_xlabel("Overall Score Rank", fontsize=10)
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} Rank", fontsize=10)
    ax.set_title(f"Overall vs {metric.replace('_', ' ').title()}\nρ = {corr:.3f} (p = {p_val:.3f})",
                 fontsize=11, fontweight="semibold")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(min_rank - 0.5, max_rank + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#fbfbfb")


def plot_delta_improvement(df: pd.DataFrame, output_root: Path, metrics: list):
    """
    Create delta/improvement plots showing where overall_score differs from each metric.
    Memory-optimized version.
    """
    import gc
    
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

    rank_dir = output_root / "delta_improvement" 
    rank_dir.mkdir(parents=True, exist_ok=True)

    for i, metric in enumerate(available_metrics):
        # Individual
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_delta_core(ax, model_stats_norm, metric, model_color_map)
        out_file_ind = rank_dir / f"delta_improvement_{metric}.png"
        fig.savefig(out_file_ind, dpi=200, bbox_inches="tight")  # Reduced DPI
        plt.close(fig)
        del fig, ax
        print(f"Saved delta improvement plot -> {out_file_ind}")

        # Combined
        _plot_delta_core(axes_combined[i], model_stats_norm, metric, model_color_map)

    # Hide extra combined axes
    for j in range(len(available_metrics), len(axes_combined)):
        axes_combined[j].set_visible(False)

    plt.tight_layout()
    out_file_comb = output_root / "delta_improvement_plots.png"
    fig_combined.savefig(out_file_comb, dpi=200, bbox_inches="tight")  # Reduced DPI
    plt.close(fig_combined)
    del fig_combined
    print(f"Saved combined delta improvement plots -> {out_file_comb}")
    
    # Cleanup
    del df_clean, model_stats, model_stats_norm
    gc.collect()


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
    ax.set_xticks(range(len(models)))
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
    Memory-efficient version: processes one model at a time and aggressively frees memory.
    """
    import gc
    
    if df.empty:
        print("No data available to plot per-model scatter.")
        return

    models = sorted(df["model"].unique())
    print(f"Generating per-model scatter plots for {len(models)} models...")

    for idx, model in enumerate(models, 1):
        print(f"  [{idx}/{len(models)}] Processing model: {model}", end="")

        # Filter and copy only what we need
        df_model = df[df["model"] == model][["overall_score"] + [m for m in metrics if m in df.columns]].copy()
        if df_model.empty:
            print(" -> skipped (empty)")
            continue

        available_metrics = [m for m in metrics if m in df_model.columns]
        if "overall_score" not in df_model.columns or not available_metrics:
            print(" -> skipped (missing columns)")
            continue

        df_model["overall_score"] = sanitize_series(df_model["overall_score"])
        df_model = df_model.dropna(subset=["overall_score"])
        if df_model.empty:
            print(" -> skipped (no valid scores)")
            continue

        n = len(available_metrics)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(9, 2.7 * n), squeeze=False)
        fig.suptitle(f"{model} — Overall Score vs Individual Metrics", fontsize=14, y=0.98)

        any_plotted = False
        overall_scores = df_model["overall_score"].values  # Extract once
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i, 0]
            series_metric = sanitize_series(df_model[metric])
            
            # Use numpy arrays directly instead of DataFrame
            valid_mask = ~(pd.isna(overall_scores) | pd.isna(series_metric.values))
            if valid_mask.sum() == 0:
                ax.set_visible(False)
                continue

            any_plotted = True
            x = series_metric.values[valid_mask]
            y = overall_scores[valid_mask]

            # Sample if too many points
            max_points = 2000
            if len(x) > max_points:
                np.random.seed(42)
                sample_idx = np.random.choice(len(x), max_points, replace=False)
                x = x[sample_idx]
                y = y[sample_idx]

            # Light jitter only if needed
            if len(np.unique(np.round(x, 5))) < 20:
                jitter = np.random.uniform(-0.02, 0.02, size=len(x)) * (x.ptp() or 1)
                x = x + jitter

            point_size = max(15, min(30, 2000 // len(x)))  # Adaptive point size
            ax.scatter(x, y, alpha=0.5, s=point_size, edgecolor="none", color=sns.color_palette("Set3")[i % 12])
            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_ylabel("Overall Score")
            ax.grid(True, alpha=0.25)

            # Trend line
            if len(x) >= 3:
                try:
                    m, b = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 50)  # Fewer points for line
                    ax.plot(x_line, m * x_line + b, "--", color="red", linewidth=1.0, alpha=0.8)
                except:
                    pass

        if not any_plotted:
            plt.close(fig)
            del fig, axes
            print(" -> no valid data")
            continue

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        model_out_dir = output_root / "per_model_scatters" / model
        model_out_dir.mkdir(parents=True, exist_ok=True)
        out_file = model_out_dir / "scatter_overall_vs_metrics.png"

        fig.savefig(out_file, dpi=150, bbox_inches="tight")  # Further reduced DPI
        plt.close(fig)
        del fig, axes, df_model
        print(f" -> saved")

        # Explicit garbage collection after each model
        gc.collect()

    print("All per-model scatter plots completed safely!")

def plot_metrics_heatmap(df: pd.DataFrame, metrics: list = None, output_dir: Path = OUTPUT_DIR):
    """
    Heatmap that inherits the current global divergent palette (red → green).
    overall_score = always first column.
    """
    if df.empty:
        print("No data to plot heatmap.")
        return

    if metrics is None:
        metrics = PREFERRED_METRICS

    # 1. Desired column order – overall_score always first
    desired_columns = ["overall_score"]
    for m in metrics:
        if m != "overall_score" and m in df.columns:
            desired_columns.append(m)

    available_cols = [c for c in desired_columns if c in df.columns]
    if not available_cols:
        print("None of the requested metrics found.")
        return

    # 2. One row per model (mean if multiple runs)
    heatmap_data = df.pivot_table(
        index="model",
        values=available_cols,
        aggfunc="mean"
    )[available_cols]  # enforce order

    # Sort models by overall_score descending
    if "overall_score" in heatmap_data.columns:
        heatmap_data = heatmap_data.sort_values(by="overall_score", ascending=False)

    # 3. Normalize 0–1 (invert low-is-better metrics)
    normalized = heatmap_data.copy()
    for col in normalized.columns:
        mn, mx = normalized[col].min(), normalized[col].max()
        if mx == mn:
            normalized[col] = 0.5
            continue
        if "cyclomatic_complexity" in col.lower() or col == "loc_score":
            normalized[col] = (mx - normalized[col]) / (mx - mn)   # lower → better
        else:
            normalized[col] = (normalized[col] - mn) / (mx - mn)

    # 4. Colormap – reuse the global divergent palette as a continuous cmap
    try:
        # `divergent_cmap` is the global 12-color red→green palette defined at the top.
        cmap = sns.color_palette(divergent_cmap, as_cmap=True)
    except Exception:
        # Very safe fallback if something goes wrong
        cmap = "RdYlGn"

    # 5. Plot
    plt.figure(figsize=(len(available_cols) * 1.3 + 2, max(4.5, len(heatmap_data) * 0.65)))

    ax = sns.heatmap(
        normalized,
        annot=heatmap_data.round(3),
        fmt="",
        cmap=cmap,
        center=0.5,
        linewidths=1,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Normalized Performance (0 = worst → 1 = best)"},
        annot_kws={"size": 10.5, "weight": "bold"}
    )

    # Pretty labels
    label_mapping = {
        "overall_score": "Overall Score",
        "functional_correctness": "Func. Correctness",
        "efficiency_score": "Efficiency",
        "avg_cyclomatic_complexity": "Avg Cyclomatic",
        "max_cyclomatic_complexity": "Max Cyclomatic",
        "style_score": "Code Style",
        "loc_score": "LOC Score",
    }
    pretty = [label_mapping.get(c, c.replace("_", " ").title()) for c in available_cols]
    ax.set_xticklabels(pretty, rotation=45, ha="right")

    plt.title("Model Performance Heatmap\n(Green = Best | Red = Worst | Overall Score = Leftmost)",
              fontsize=15, fontweight="bold", pad=25)
    plt.ylabel("Model", fontsize=12)
    plt.xlabel("")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = output_dir / "model_metrics_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")  # Reduced DPI for memory
    print(f"Heatmap saved → {out_path}")
    plt.close()

if __name__ == "__main__":
    import gc
    
    df_all = load_summary_files(SUMMARY_GLOB)

    print("=" * 80)
    print("Starting plot generation...")
    print("=" * 80)
    
    plot_overall_score_distribution(df_all, OUTPUT_DIR)
    gc.collect()
    
    plot_rank_concordance(df_all, OUTPUT_DIR, PREFERRED_METRICS)
    gc.collect()
    
    plot_delta_improvement(df_all, OUTPUT_DIR, PREFERRED_METRICS)
    gc.collect()
    
    plot_per_model_scatter(df_all, OUTPUT_DIR, PREFERRED_METRICS)
    gc.collect()
    
    plot_metrics_heatmap(df_all, PREFERRED_METRICS, OUTPUT_DIR)
    gc.collect()

    print("=" * 80)
    print("All plots generated successfully!")
    print("=" * 80)
