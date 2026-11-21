"""Report version of the necessity/propensity figure, Qwen models only (no reference-model).

Same data and layout as necessity_propensity/experiment.py's draw_figure, which is left
untouched: clean accuracy by question type (left) and where each clean trace first
states its final answer (right). Colours match the rest of the report.
Reads logs/np_summary.csv and logs/np_trace_labels.csv only.

Run: python analysis/necessity_figure.py
Output: short_research_report.assets/necessity_propensity.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "short_research_report.assets" / "necessity_propensity.png"
MODELS = {"Qwen3-1.7B_vllm": ("Qwen3-1.7B", "#2a78d6"), "Qwen3-4B_vllm": ("Qwen3-4B", "#eb6834")}
TYPES = ["recall", "single_step", "multi_step"]
EARLY = 0.25   # same threshold as the experiment: a first statement in the first quarter

summary = pd.read_csv(ROOT / "logs" / "np_summary.csv")
acc = summary[(summary["section"] == "necessity") & (summary["metric"] == "clean accuracy (sample 0)")]
acc = acc.set_index(["model", "reasoning_type"])
traces = pd.read_csv(ROOT / "logs" / "np_trace_labels.csv")
traces = traces[(traces["condition"] == "clean") & traces["commit_position"].notna()]

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
width = 0.36
for i, (model, (name, color)) in enumerate(MODELS.items()):
    rows = [acc.loc[(model, t)] for t in TYPES]
    xs = [j + (i - 0.5) * width for j in range(len(TYPES))]
    vals = [r["value"] for r in rows]
    err = [[v - r["ci_low"] for v, r in zip(vals, rows)], [r["ci_high"] - v for v, r in zip(vals, rows)]]
    a.bar(xs, vals, width, yerr=err, capsize=3, color=color, label=name)
    for x, v in zip(xs, vals):
        a.text(x, 0.03, f"{v:.0%}", ha="center", color="white", fontsize=8)
a.set_xticks(range(len(TYPES)), [f"{t.replace('_', '-')}\n(n={int(acc.loc[(next(iter(MODELS)), t), 'n'])})"
                                  for t in TYPES])
a.set_ylim(0, 1.05)
a.set_ylabel("Clean accuracy (sample 0)")
a.set_title("Accuracy by the reasoning the question needs", fontsize=10)

for i, (model, (name, color)) in enumerate(MODELS.items()):
    data = [traces.loc[(traces["model"] == model) & (traces["reasoning_type"] == t), "commit_position"].tolist()
            for t in TYPES]
    pos = [j + (i - 0.5) * 0.36 for j in range(len(TYPES))]
    bp = b.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=False,
                   medianprops={"color": "#1b2430", "lw": 1.6})
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for j, d in enumerate(data):
        b.scatter([pos[j]] * len(d), d, s=6, color="#333333", alpha=0.35, zorder=3)
b.axhline(EARLY, color="#bbbbbb", lw=0.8, ls="--")
b.set_xticks(range(len(TYPES)), [t.replace("_", "-") for t in TYPES])
b.set_ylim(-0.02, 1.02)
b.set_ylabel("Where the final answer is first stated\n(0 = start of trace, 1 = end)")
b.set_title("Where the final answer is first stated (clean traces)", fontsize=10)
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MODELS.values()],
           [n for n, _ in MODELS.values()], fontsize=8, frameon=False, loc="upper center", ncol=2)
fig.tight_layout(rect=(0, 0, 1, 0.93))   # legend sits above both panels, clear of the data
fig.savefig(OUT, dpi=150)
print("Wrote", OUT)
