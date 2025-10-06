import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_file = "evaluation_summary.csv"
df = pd.read_csv(csv_file)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

plt.figure()
sns.histplot(df["overall_score"], bins=20, kde=True, color="skyblue")
plt.title("Distribuição do Overall Score (PoE)")
plt.xlabel("Overall Score")
plt.ylabel("Número de funções")
plt.savefig("overall_score_distribution.png")
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
plt.savefig("metrics_mean.png")
plt.show()

plt.figure()
sns.heatmap(df[metrics + ["overall_score"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlação entre métricas e Overall Score")
plt.savefig("metrics_correlation.png")
plt.show()
