"""Raw-count disclosure figure for the report, short_research_report.html (replaces the normalized-score chart).

For each hint type and in total: flips to the hint, split into disclosed and not disclosed
by the B1 judge, with flips / eligible questions as the label.
Sources: logs/per_question_summary_*.csv (corrected answers), logs/judge_*.jsonl (last
record per id is the active verdict).

Run: python analysis/disclosure_counts_figure.py
Output: short_research_report.assets/disclosure-counts.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "short_research_report.assets" / "disclosure-counts.png"
MODELS = {"Qwen3-1.7B_vllm": ("Qwen3-1.7B", "#2a78d6", "#a9c9ef"),
          "Qwen3-4B_vllm": ("Qwen3-4B", "#eb6834", "#f6c2a7")}
TYPES = ["unethical_information", "consistency", "sycophancy", "metadata", "visual_pattern",
         "grader_hacking", "ALL"]

counts = {}
for tag in MODELS:
    w = pd.read_csv(ROOT / "logs" / f"per_question_summary_{tag}.csv").set_index("id")
    judge = {j["id"]: j for j in map(json.loads, open(ROOT / "logs" / f"judge_{tag}.jsonl", encoding="utf-8"))}
    w["disclosed"] = [bool(f) and judge[i]["verbalized"] for i, f in zip(w.index, w["flipped"])]
    for t in TYPES:
        g = w if t == "ALL" else w[w["hint_type"] == t]
        counts[tag, t] = (int(g["flipped"].sum()), int(g["disclosed"].sum()), int(g["eligible"].sum()))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "ytick.left": False})
fig, ax = plt.subplots(figsize=(10, 6.2))
h = 0.36
ys = []
for g, t in enumerate(TYPES):
    base = g + (0.6 if t == "ALL" else 0)
    ys.append(base)
    for i, (tag, (name, dark, light)) in enumerate(MODELS.items()):
        flips, disc, elig = counts[tag, t]
        y = base + (i - 0.5) * h
        ax.barh(y, disc, h * 0.9, color=dark, zorder=2)
        ax.barh(y, flips - disc, h * 0.9, left=disc, color=light, zorder=2)
        ax.text(flips + 0.3, y, f"{disc} of {flips} disclosed  ·  {flips}/{elig} flipped",
                va="center", fontsize=9.5)
ax.axhline(len(TYPES) - 1 - 0.1, color="#c3c2b7", lw=0.8)
ax.set_yticks(ys, ["all hint types" if t == "ALL" else t.replace("_", " ") for t in TYPES])
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.invert_yaxis()
ax.set_xlim(0, 33)
ax.set_xlabel("flips to the hinted wrong answer")
ax.grid(axis="x", color="#e8e7e3", zorder=0)
handles = []
for name, dark, light in MODELS.values():
    handles += [plt.Rectangle((0, 0), 1, 1, color=dark, label=f"{name}: disclosed"),
                plt.Rectangle((0, 0), 1, 1, color=light, label=f"{name}: not disclosed")]
fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.955))
fig.suptitle("Most flips disclosed the hint (raw counts, LLM judge)", fontweight="bold", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.935))
fig.savefig(OUT, dpi=200)
for k, v in counts.items():
    print(k, v)
