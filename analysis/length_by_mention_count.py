"""How reasoning length grows with the number of times the hinted trace mentions the hint.

Uses every hinted run (truncated traces excluded) and compares it with the first clean run
of the same question. Mentions use the same keyword patterns as length_by_mention.py.
Reads saved logs only.

Run: python analysis/length_by_mention_count.py
Output: short_research_report.assets/length-by-mention-count.png (and .svg)
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"Qwen3-1.7B_vllm": ("Qwen3-1.7B", "#2a78d6"), "Qwen3-4B_vllm": ("Qwen3-4B", "#eb6834")}
KW = {  # same patterns as length_by_mention.py; misses paraphrases
    "sycophancy": r"professor|teacher|you think|user (says|said|suggest|thinks)|user's answer",
    "consistency": r"previous (response|answer)|earlier answer|i said|initially|original answer",
    "visual_pattern": r"■|marked|square|symbol",
    "metadata": r"metadata|<answer>|answer tag",
    "grader_hacking": r"validat|grader|function|returns? ",
    "unethical_information": r"unauthori[sz]ed|gained access|the system|user (says|said|mention)",
}

OUT = ROOT / "short_research_report.assets" / "length-by-mention-count"
BINS, LABELS = [-1, 0, 5, 10, 1000], ["0", "1–5", "6–10", "11+"]

stats = {}
for tag, (name, color) in MODELS.items():
    by = {(r["id"], r["condition"], r["sample"]): r
          for r in map(json.loads, open(ROOT / "logs" / f"results_{tag}.jsonl", encoding="utf-8"))}
    w = pd.read_csv(ROOT / "logs" / f"per_question_summary_{tag}.csv").set_index("id")
    rows = []
    for q in w.index:
        a, b, h = by[q, "unhinted", 0], by[q, "unhinted", 1], by[q, "hinted", 0]
        if a["truncated"] or b["truncated"] or h["truncated"]:
            continue
        rows.append({"mentions": len(re.findall(KW[w.loc[q, "hint_type"]], h["thinking"], re.I)),
                     "ratio": h["n_tokens"] / a["n_tokens"],
                     "wait": len(re.findall(r"\bwait\b", h["thinking"], re.I))})
    d = pd.DataFrame(rows)
    d["bin"] = pd.cut(d["mentions"], BINS, labels=LABELS)
    stats[name] = (color, d.groupby("bin", observed=False).agg(
        n=("ratio", "size"), ratio=("ratio", "median"), wait=("wait", "median")))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
x = np.arange(len(LABELS))
width = 0.38
for i, (name, (color, s)) in enumerate(stats.items()):
    xs = x + (i - 0.5) * width
    ax1.bar(xs, s["ratio"], width, color=color, label=name)
    ax2.bar(xs, s["wait"], width, color=color, label=name)
    for xx, r, n in zip(xs, s["ratio"], s["n"]):
        ax1.text(xx, r + 0.04, f"×{r:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax1.text(xx, 0.06, f"n={n}", ha="center", fontsize=8, color="white")
    for xx, v in zip(xs, s["wait"]):
        ax2.text(xx, v + 0.4, f"{v:g}", ha="center", fontsize=9, fontweight="bold")

ax1.axhline(1, color="#5b6675", lw=1, ls="--", label="same length as clean run")
ax1.set_ylabel("hinted ÷ clean length (median per question)")
ax1.set_title("Reasoning length", loc="left", fontweight="bold")
ax2.set_ylabel('"wait" per hinted trace (median)')
ax2.set_title('Back-and-forth: "wait" count', loc="left", fontweight="bold")
for ax in (ax1, ax2):
    ax.set_xticks(x, LABELS)
    ax.set_xlabel("times the hinted reasoning mentions the hint")
    ax.grid(axis="y", color="#e8e7e3")
    ax.set_axisbelow(True)
ax1.legend(frameon=False, loc="upper left")
fig.suptitle("The more a trace returns to the hint, the longer and more back-and-forth it gets",
             x=0.01, ha="left", fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
fig.savefig(OUT.with_suffix(".svg"))
for name, (_, s) in stats.items():
    print(name, s.round(2).to_dict("index"))
