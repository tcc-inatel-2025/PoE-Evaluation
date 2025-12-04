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

    fig.suptitle(
        "Distribution of Overall PoE Score Across Code Generation Models",
        fontsize=15,
        fontweight="bold",
        y=1.02
    )
    
    # Box plot
    box_data = [df_clean[df_clean["model"] == model]["overall_score"].values for model in models]
    bp = ax1.boxplot(box_data, labels=models, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title("Overall PoE Score per Model (HumanEval Tasks)")
    ax1.set_xlabel("Code Generation Model")
    ax1.set_ylabel("Overall PoE Score")
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
    
    ax2.set_title("Score Density per Model (Violin Plot)")
    ax2.set_xlabel("Code Generation Model")
    ax2.set_ylabel("Overall PoE Score")
    
    plt.tight_layout()
    
    # Save the figure
    out_file = output_root / "distribution_plots.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved distribution plots -> {out_file}")
    

def plot_rank_concordance(df: pd.DataFrame, output_root: Path, metrics: list):
    """
    Create rank concordance plots comparing overall_score rankings vs other metrics.
    Also generates inter-metric Spearman correlation matrix and heatmap.
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

    # Color map per model
    models = model_stats["model"].tolist()
    model_palette = sns.color_palette("Set3", n_colors=len(models))
    model_color_map = {m: model_palette[i] for i, m in enumerate(models)}

    overall_ranks = model_stats["overall_score"].rank(ascending=False, method="average")

    # === 1. Rank Concordance Plots (individual + combined) ===
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig_combined, axes_combined = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes_combined = np.array(axes_combined).ravel()

    rank_dir = output_root / "rank_concordance"
    rank_dir.mkdir(parents=True, exist_ok=True)

    for i, metric in enumerate(available_metrics):
        metric_ranks = model_stats[metric].rank(ascending=False, method="average")

        # Individual plot
        fig, ax = plt.subplots(figsize=(7, 6))
        _plot_rank_concordance_core(ax, model_stats, overall_ranks, metric_ranks, metric, model_color_map)
        out_file_ind = rank_dir / f"rank_concordance_{metric}.png"
        fig.savefig(out_file_ind, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {out_file_ind.name}")

        # Combined subplot
        axc = axes_combined[i]
        _plot_rank_concordance_core(axc, model_stats, overall_ranks, metric_ranks, metric, model_color_map)
        axc.set_title(metric.replace("_", " ").title(), fontsize=11, pad=10)

    # Hide unused subplots
    for j in range(i + 1, len(axes_combined)):
        axes_combined[j].set_visible(False)

    plt.tight_layout()
    combined_path = output_root / "rank_concordance_plots.png"
    fig_combined.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig_combined)
    print(f"   Saved combined: {combined_path.name}")

    # === 2. NEW: Inter-metric Spearman Correlation Heatmap ===
    print("   Generating inter-metric Spearman correlation matrix...")
    
    corr_data = model_stats[available_metrics]
    spearman_matrix = corr_data.corr(method='spearman')

    plt.figure(figsize=(max(7, len(available_metrics) * 0.95), max(6, len(available_metrics) * 0.85)))
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

    plt.title(
    "Spearman Rank Correlation Between Evaluation Metrics\n"
    "(Higher |ρ| indicates stronger monotonic relationship)",
    fontsize=14,
    fontweight="bold",
    pad=20
)
    
    plt.xlabel("Metric", fontsize=11)
    plt.ylabel("Metric", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    heatmap_path = rank_dir / "inter_metric_spearman_correlation.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved correlation heatmap → {heatmap_path.name}")

    # Save CSV tables (perfect for appendix)
    csv_path = rank_dir / "inter_metric_spearman_correlation.csv"
    spearman_matrix.round(4).to_csv(csv_path)
    print(f"   Saved correlation table → {csv_path.name}")

    # Optional: p-values
    _, pval = spearmanr(corr_data)
    pval_df = pd.DataFrame(pval, index=available_metrics, columns=available_metrics)
    pval_csv = rank_dir / "inter_metric_spearman_pvalues.csv"
    pval_df.round(4).to_csv(pval_csv)
    print(f"   Saved p-values → {pval_csv.name}")

    print("All rank concordance + inter-metric correlation analysis complete!\n")


def _plot_rank_concordance_core(ax, model_stats, overall_ranks, metric_ranks, metric, model_color_map):
    """Core plotting routine for a rank concordance axis."""
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

    metric_label_map = {
        "functional_correctness": "Functional Correctness",
        "efficiency_score": "Efficiency Score",
        "avg_cyclomatic_complexity": "Avg. Cyclomatic Complexity",
        "max_cyclomatic_complexity": "Max Cyclomatic Complexity",
        "style_score": "Style Score",
        "loc_score": "LOC Score"
    }
    
    pretty_metric = metric_label_map.get(metric, metric.replace("_", " ").title())

    ax.set_xlabel("Overall Score Rank (1 = best)", fontsize=10)
    ax.set_ylabel(f"{pretty_metric} Rank (1 = best)", fontsize=10)

    ax.set_title(
        f"Rank Concordance: Overall Score vs {pretty_metric}\n"
        f"Spearman ρ = {corr:.3f} (p = {p_val:.3f})",
        fontsize=11,
        fontweight="semibold"
    )
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
    bars = ax.bar(models, deltas, color=bar_colors, alpha=0.85,
                  edgecolor="black", linewidth=0.4)
    # Value labels
    for bar, delta in zip(bars, deltas):
        h = bar.get_height()
        y_offset = 0.01 if h >= 0 else -0.01
        va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, h + y_offset,
                f"{delta:.3f}",
                ha="center", va=va, fontsize=8, fontweight="semibold")

    ax.axhline(0, color="black", linewidth=0.7)

    metric_label_map = {
        "functional_correctness": "Functional Correctness",
        "efficiency_score": "Efficiency Score",
        "avg_cyclomatic_complexity": "Avg. Cyclomatic Complexity",
        "max_cyclomatic_complexity": "Max Cyclomatic Complexity",
        "style_score": "Style Score",
        "loc_score": "LOC Score"
    }
    pretty_metric = metric_label_map.get(metric, metric.replace("_", " ").title())

    ax.set_title(
        f"Difference Between Normalized Overall Score and {pretty_metric}\n"
        f"(Δ = Overall − {pretty_metric}; positive = Overall > Metric)",
        fontsize=11,
        fontweight="semibold"
    )
    ax.set_ylabel("Δ (Normalized Overall − Metric)", fontsize=10)

    ax.set_xticklabels(models, rotation=45, ha="right")

    mean_delta = np.nanmean(deltas)
    positives = np.sum(deltas > 0)

    ax.text(
        0.02, 0.98,
        f"Mean Δ: {mean_delta:.3f}\nPositive bars: {positives}/{len(deltas)}",
        transform=ax.transAxes,
        va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_facecolor("#fbfbfb")


def plot_per_model_scatter(df: pd.DataFrame, output_root: Path, metrics: list):
    """
    Memory-efficient version: processes one model at a time and aggressively frees memory.
    """
    if df.empty:
        print("No data available to plot per-model scatter.")
        return

    models = sorted(df["model"].unique())
    print(f"Generating per-model scatter plots for {len(models)} models...")

    for idx, model in enumerate(models, 1):
        print(f"  [{idx}/{len(models)}] Processing model: {model}", end="")

        df_model = df[df["model"] == model].copy()
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
        fig.suptitle(
            f"{model} — Relationship Between Overall Score and Individual Metrics",
            fontsize=14,
            y=0.98
    )

        any_plotted = False
        for i, metric in enumerate(available_metrics):
            ax = axes[i, 0]

            metric_label_map = {
                "functional_correctness": "Functional Correctness",
                "efficiency_score": "Efficiency Score",
                "avg_cyclomatic_complexity": "Avg. Cyclomatic Complexity",
                "max_cyclomatic_complexity": "Max Cyclomatic Complexity",
                "style_score": "Style Score",
                "loc_score": "LOC Score"
            }
            pretty_metric = metric_label_map.get(metric, metric.replace("_", " ").title())

            series_metric = sanitize_series(df_model[metric])
            plot_df = pd.DataFrame({
                "overall_score": df_model["overall_score"],
                metric: series_metric
            }).dropna()

            if plot_df.empty:
                ax.set_visible(False)
                continue

            any_plotted = True
            x = plot_df[metric].values
            y = plot_df["overall_score"].values

            # Light jitter only if needed
            if len(np.unique(np.round(x, 5))) < 20:
                jitter = np.random.uniform(-0.02, 0.02, size=len(x)) * (x.ptp() or 1)
                x = x + jitter

            ax.scatter(
                x, y,
                alpha=0.65,
                s=30,
                edgecolor="none",
                color=sns.color_palette("Set3")[i % 12]
            )

            ax.set_xlabel(pretty_metric, fontsize=10)
            ax.set_ylabel("Overall PoE Score", fontsize=10)
            ax.set_title(f"{pretty_metric} vs Overall PoE Score", fontsize=11)
            ax.grid(True, alpha=0.25)

            # Trend line
            if len(x) >= 3:
                try:
                    m, b = np.polyfit(x, y, 1)
                    xx = np.linspace(x.min(), x.max(), 100)
                    ax.plot(xx, m * xx + b, "--", color="red", linewidth=1.2, alpha=0.8)
                except:
                    pass


        if not any_plotted:
            plt.close(fig)
            print(" -> no valid data")
            continue

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        model_out_dir = output_root / "per_model_scatters" / model
        model_out_dir.mkdir(parents=True, exist_ok=True)
        out_file = model_out_dir / "scatter_overall_vs_metrics.png"

        fig.savefig(out_file, dpi=200, bbox_inches="tight")  # Reduced DPI = huge memory savings
        plt.close(fig)  # Critical: free memory immediately
        print(f" -> saved ({len(df_model)} points)")

        # Explicit garbage collection every few models
        if idx % 5 == 0:
            import gc
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

    plt.title(
        "Normalized Performance of Code Generation Models Across Evaluation Metrics\n"
        "(Green = Best, Red = Worst; Overall Score on the Left)",
        fontsize=15,
        fontweight="bold",
        pad=25
    )
    plt.ylabel("Code Generation Model", fontsize=12)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = output_dir / "model_metrics_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved → {out_path}")
    plt.close()

if __name__ == "__main__":
    df_all = load_summary_files(SUMMARY_GLOB)

    plot_overall_score_distribution(df_all, OUTPUT_DIR)
    plot_rank_concordance(df_all, OUTPUT_DIR, PREFERRED_METRICS)
    plot_delta_improvement(df_all, OUTPUT_DIR, PREFERRED_METRICS)
    plot_per_model_scatter(df_all, OUTPUT_DIR, PREFERRED_METRICS)
    plot_metrics_heatmap(df_all, PREFERRED_METRICS, OUTPUT_DIR)

    print("All plots generated successfully!")