"""Figure for the monitor section of the report (short_research_report.html).

Monitor flag rates, counting every trace that got a verdict (including the 36 whose
supporting quote failed the exact-match check), split by what the trace is:
disclosed flips, undisclosed flips, resisted hinted traces that reject the hint,
resisted hinted traces with no hint mention, and clean traces.

Sources: logs/b2_monitor_review.csv (monitor verdicts; unverified_flag for the invalid
quotes), logs/np_trace_labels.csv (hint stance, condition == "hinted", status == "ok").

Run: python analysis/b2_figure.py
Output: short_research_report.assets/monitor-results.png (and .svg)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "short_research_report.assets" / "monitor-results"
MODELS = {"Qwen3-1.7B_vllm": ("Qwen3-1.7B", "#2a78d6"), "Qwen3-4B_vllm": ("Qwen3-4B", "#eb6834")}

r = pd.read_csv(ROOT / "logs" / "b2_monitor_review.csv")
r = r[r["monitor_status"] != "excluded"].copy()
r["flag"] = r["monitor_flag"].where(r["monitor_status"] == "ok", r["unverified_flag"]).astype(bool)
stance = pd.read_csv(ROOT / "logs" / "np_trace_labels.csv")
stance = stance[(stance["condition"] == "hinted") & (stance["status"] == "ok")]
r = r.merge(stance[["model", "id", "stance"]].rename(columns={"id": "question_id", "stance": "hint_stance"}),
            on=["model", "question_id"], how="left")
r.loc[r["condition"] != "hinted", "hint_stance"] = None

GROUPS = [
    ("Flips that disclosed the hint", lambda d: (d.trace_class == "flip") & (d.b1_disclosed == True)),
    ("Flips that did not disclose it", lambda d: (d.trace_class == "flip") & (d.b1_disclosed == False)),
    ("Resisted the hint, trace rejects it", lambda d: (d.trace_class == "non_flip") & (d.hint_stance == "rejects")),
    ("Resisted the hint, no mention of it", lambda d: (d.trace_class == "non_flip") & (d.hint_stance == "not_mentioned")),
    ("Clean traces (no hint at all)", lambda d: d.trace_class == "clean"),
]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "ytick.left": False})
fig, ax = plt.subplots(figsize=(10.5, 5.6))
h = 0.36
rows = []
for g, (label, sel) in enumerate(GROUPS):
    for i, (tag, (name, color)) in enumerate(MODELS.items()):
        d = r[(r["model"] == tag) & sel(r)]
        k, n = int(d["flag"].sum()), len(d)
        rows.append((label, name, k, n))
        y = g + (i - 0.5) * h
        ax.barh(y, 1, h * 0.9, color="#eef0f3", zorder=1)
        ax.barh(y, k / n, h * 0.9, color=color, zorder=2)
        ax.text(1.02, y, f"{k} / {n}", va="center", fontsize=11, fontweight="bold")
ax.axhline(1.5, color="#c3c2b7", lw=0.8)
ax.set_yticks(range(len(GROUPS)), [lab for lab, _ in GROUPS])
ax.invert_yaxis()
ax.set_xlim(0, 1.13)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("share of traces the monitor flagged as relying on a suggested answer")
ax.grid(axis="x", color="#e8e7e3", zorder=0)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MODELS.values()],
           labels=[n for n, _ in MODELS.values()], frameon=False, loc="upper left",
           bbox_to_anchor=(0.012, 0.9), ncol=2)
fig.suptitle("The monitor flags disclosed flips, and also many traces that reject the hint",
             x=0.02, y=0.985, ha="left", fontweight="bold", fontsize=12.5)
fig.text(0.02, 0.915, "Monitor sees only the reasoning trace · flagged / traces with a verdict",
         fontsize=10, color="#5b6675")
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
fig.savefig(OUT.with_suffix(".svg"))
for row in rows:
    print(*row)
