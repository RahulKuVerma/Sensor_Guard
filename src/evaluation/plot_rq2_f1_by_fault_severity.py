import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("results/metrics/rq2_controlled_sensor_fault_experiment/rq2_final_results_table.csv")
out_path = csv_path.parent / "rq2_f1_by_fault_severity.png"

df = pd.read_csv(csv_path)

required = {"Subset", "Scenario", "FaultType", "Severity", "F1"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing expected CSV columns: {sorted(missing)}")

df["F1"] = pd.to_numeric(df["F1"], errors="raise")
df["Severity"] = pd.to_numeric(df["Severity"], errors="coerce")

subsets = ["FD001", "FD002", "FD003", "FD004"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

for ax, subset in zip(axes, subsets):
    part = df[df["Subset"] == subset].copy()
    clean = part[part["Scenario"].str.lower() == "clean"]

    if clean.empty:
        ax.set_title(f"{subset} (clean baseline unavailable)")
    else:
        baseline = clean["F1"].mean()
        ax.axhline(
            baseline, linestyle="--",
            label=f"Clean baseline ({baseline:.3f})"
        )
        ax.set_title(subset)

    faults = part[part["Scenario"].str.lower() != "clean"]
    for fault_type, group in faults.groupby("FaultType", dropna=False):
        group = group.dropna(subset=["Severity"]).sort_values("Severity")
        if group.empty:
            continue
        label = str(fault_type) if pd.notna(fault_type) else "Unknown fault"
        ax.plot(group["Severity"], group["F1"], marker="o", label=label)

    ax.set_xlabel("Fault severity")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

fig.suptitle("SensorGuard: F1 under Controlled Sensor Degradation", fontsize=14)
fig.tight_layout()
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Figure saved: {out_path}")
print(f"Conditions plotted: {len(df)}")
print(f"Subsets: {', '.join(sorted(df['Subset'].unique()))}")
