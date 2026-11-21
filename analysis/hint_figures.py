"""Report versions of the two clean-vs-hinted figures, in the report's colours and with
legends kept clear of the data.

- Accuracy by question type, clean vs. hinted: redrawn here from the corrected answers
  (logs/per_question_summary_*.csv) and question types (logs/np_question_labels.csv).
- Where the final answer is first stated, clean vs. hinted: drawn by
  necessity_propensity/hinted_profile.py's pair() from its cached judge results, with
  colours, median lines and output path overridden. That script itself is not edited,
  and no API calls are made.

Run: python analysis/hint_figures.py
Outputs: short_research_report.assets/clean_vs_hinted_accuracy.png and clean_vs_hinted_commit.png
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "short_research_report.assets"
MODELS = {"Qwen3-1.7B_vllm": ("Qwen3-1.7B", "#2a78d6"), "Qwen3-4B_vllm": ("Qwen3-4B", "#eb6834")}
TYPES = ["recall", "single_step", "multi_step"]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


# ---- Accuracy, clean vs. hinted ------------------------------------------------
qtype = pd.read_csv(ROOT / "logs" / "np_question_labels.csv").set_index("id")["reasoning_type"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, col, title in [(axes[0], "unhinted_0", "Clean (no hint)"), (axes[1], "hinted_0", "With a wrong hint")]:
    for i, (tag, (name, color)) in enumerate(MODELS.items()):
        w = pd.read_csv(ROOT / "logs" / f"per_question_summary_{tag}.csv").set_index("id").join(qtype)
        vals, lo, hi = [], [], []
        for t in TYPES:
            g = w[w["reasoning_type"] == t]
            k, n = int((g[col] == g["gold"]).sum()), len(g)
            a, b = wilson(k, n)
            vals.append(k / n); lo.append(k / n - a); hi.append(b - k / n)
        xs = [j + (i - 0.5) * 0.36 for j in range(len(TYPES))]
        ax.bar(xs, vals, 0.34, yerr=[lo, hi], capsize=3, color=color, label=name)
        for x, v in zip(xs, vals):
            ax.text(x, 0.03, f"{v:.0%}", ha="center", fontsize=8, color="white")
    n_by_type = qtype.value_counts()
    ax.set_xticks(range(len(TYPES)), [f"{t.replace('_', '-')}\n(n={n_by_type[t]})" for t in TYPES])
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Accuracy (sample 0, 95% Wilson interval)")
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False, loc="upper center",
           ncol=2, bbox_to_anchor=(0.5, 0.95))
fig.suptitle("Accuracy by the reasoning the question needs", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(ASSETS / "clean_vs_hinted_accuracy.png", dpi=150)
plt.close(fig)

# ---- Where the final answer is first stated, clean vs. hinted -------------------
sys.path.insert(0, str(ROOT / "necessity_propensity"))
import hinted_profile as hp  # noqa: E402

hp.COLORS = {tag: color for tag, (_, color) in MODELS.items()}
hp.FLIP = "#5b2a86"   # purple: stands out on both the blue and the orange boxes
hp.ACCURACY_PATH = ASSETS / "_unused_accuracy.png"   # pair() also draws accuracy; ours is above
hp.COMMIT_PATH = ASSETS / "clean_vs_hinted_commit.png"
plt.rcParams["boxplot.medianprops.color"] = "#1b2430"
plt.rcParams["boxplot.medianprops.linewidth"] = 1.6
# pair() labels the position as a "commit"; the report calls it an answer-statement position,
# because the early sentence on hinted traces is often the trace quoting the hint.
WORDING = [("first commits to", "first states"), ("first committed", "first stated"),
           ("an early commit is", "an early answer statement is")]
_set_text = matplotlib.text.Text.set_text


def _reworded(self, s):
    s = str(s)
    for old, new in WORDING:
        s = s.replace(old, new)
    return _set_text(self, s)


matplotlib.text.Text.set_text = _reworded
hp.pair(None)
matplotlib.text.Text.set_text = _set_text
hp.ACCURACY_PATH.unlink(missing_ok=True)
print("Wrote", ASSETS / "clean_vs_hinted_accuracy.png", "and", hp.COMMIT_PATH)
