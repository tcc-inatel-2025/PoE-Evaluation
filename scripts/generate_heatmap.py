from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Configuration
SUMMARY_GLOB = "../results/summary/*_summary.csv"
OUTPUT_DIR = Path("../results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics to include in the heatmap
HEATMAP_METRICS = [
    "overall_score",
    "functional_correctness",
    "efficiency_score",
    "avg_cyclomatic_complexity",
    "style_score",
    "loc_score",
]

sns.set_theme(style="white")

def load_summary_files(pattern: str) -> pd.DataFrame:
    """Load all matching CSVs and attach a 'model' column (derived from filename).
    Returns a concatenated DataFrame (may be empty).
    """
    files = glob.glob(pattern)
    print(f"Found {len(files)} summary files to generate heatmap.")
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

def plot_model_comparison_heatmap(df: pd.DataFrame, output_root: Path):
    """
    Creates and saves a heatmap to compare the mean scores of all models across key metrics.
    """
    if df.empty:
        print("No data available to generate heatmap.")
        return

    # Ensure required columns exist
    available_metrics = [m for m in HEATMAP_METRICS if m in df.columns]
    if not available_metrics:
        print("None of the specified heatmap metrics are available in the data.")
        return

    # Calculate the mean of the available metrics for each model
    model_stats = df.groupby("model")[available_metrics].mean()

    print("Generating model comparison heatmap with the following data:")
    print(model_stats)

    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(
        model_stats,
        annot=True,       # Display the values in the cells
        fmt=".3f",        # Format values to three decimal places
        cmap="viridis",   # Use a color-blind friendly colormap
        linewidths=.5,
        linecolor='black'
    )
    
    plt.title("Model Performance Comparison Heatmap", fontsize=16, pad=20)
    plt.xlabel("Evaluation Metrics", fontsize=12)
    plt.ylabel("Models", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Save the figure
    out_file = output_root / "model_comparison_heatmap.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully saved model comparison heatmap -> {out_file}")


def main():
    """
    Main function to load data and generate the heatmap plot.
    """
    df_all = load_summary_files(SUMMARY_GLOB)
    plot_model_comparison_heatmap(df_all, OUTPUT_DIR)
    print("Heatmap generation process complete.")

if __name__ == "__main__":
    main()
