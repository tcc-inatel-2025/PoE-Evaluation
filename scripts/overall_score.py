import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Read all summary CSV files from results/summary/
summary_files = glob.glob("../results/summary/*_summary.csv")
print(f"Found {len(summary_files)} summary files: {summary_files}")

# Combine all CSV files into a single DataFrame
dataframes = []
for file in summary_files:
    model_name = os.path.basename(file).replace('_summary.csv', '')
    df_temp = pd.read_csv(file)
    df_temp['model'] = model_name  # Add model column for identification
    dataframes.append(df_temp)

df = pd.concat(dataframes, ignore_index=True)
print(f"Combined dataset shape: {df.shape}")
print(f"Models included: {df['model'].unique()}")

# Ensure plots directory exists
os.makedirs("results/plots", exist_ok=True)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Overall score distribution
plt.figure()
sns.histplot(df["overall_score"], bins=20, kde=True, color="skyblue")
plt.title("Distribuição do Overall Score (PoE)")
plt.xlabel("Overall Score")
plt.ylabel("Número de funções")
plt.savefig("results/plots/overall_score_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

metrics = [
    "avg_cyclomatic_complexity", "max_cyclomatic_complexity",
    "style_score", "efficiency_score", "loc_score",
    "functional_correctness"
]

mean_values = df[metrics].mean().sort_values(ascending=False)

plt.figure()
sns.barplot(x=mean_values.values, y=mean_values.index, palette="viridis")
plt.title("Média das métricas por categoria")
plt.xlabel("Score médio (0-1)")
plt.ylabel("Métricas")
plt.savefig("results/plots/metrics_mean.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
sns.heatmap(df[metrics + ["overall_score"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlação entre métricas e Overall Score")
plt.savefig("results/plots/metrics_correlation.png", dpi=300, bbox_inches='tight')
plt.show()

# Additional analysis: Model comparison
plt.figure(figsize=(14, 8))
model_means = df.groupby('model')[metrics + ['overall_score']].mean()
sns.heatmap(model_means.T, annot=True, cmap="RdYlBu_r", fmt=".3f", cbar_kws={'label': 'Score médio'})
plt.title("Comparação de métricas por modelo")
plt.xlabel("Modelo")
plt.ylabel("Métricas")
plt.xticks(rotation=45)
plt.savefig("results/plots/model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# Model performance summary
print("\n=== Resumo de Performance por Modelo ===")
model_summary = df.groupby('model')['overall_score'].agg(['mean', 'std', 'count']).round(3)
print(model_summary)
