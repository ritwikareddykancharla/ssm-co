import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.25)

LOGS = {
    "TSP10": "logs/tsp10_dim128.log",
    "TSP20": "logs/tsp20_dim128.log",
    # "TSP50": "logs/tsp50_dim256.log",
    # "TSP100": "logs/tsp100_dim256.log",
}

pattern = re.compile(
    r"step (\d+).*?tour ([0-9.]+)"
)

rows = []

for label, path in LOGS.items():
    with open(path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                tour = float(match.group(2))
                rows.append({
                    "problem": label,
                    "step": step,
                    "tour": tour
                })

df = pd.DataFrame(rows)

# -------- PLOT 1: CONVERGENCE --------
plt.figure(figsize=(7.5, 5.5))
sns.lineplot(
    data=df,
    x="step",
    y="tour",
    hue="problem",
    linewidth=2.5
)
plt.xlabel("Training Steps")
plt.ylabel("Average Tour Length")
plt.title("Training Convergence Across Problem Sizes")
plt.tight_layout()
plt.savefig("convergence.pdf")
plt.show()

# -------- PLOT 2: FINAL COST vs N --------
final_df = (
    df.sort_values("step")
      .groupby("problem")
      .tail(5)
      .groupby("problem")["tour"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(6.5, 5))
sns.pointplot(
    data=final_df,
    x="problem",
    y="tour",
    scale=1.2
)
plt.ylabel("Final Greedy Tour Length")
plt.xlabel("Problem Size")
plt.title("Scaling of Solution Quality")
plt.tight_layout()
plt.savefig("scaling.pdf")
plt.show()
