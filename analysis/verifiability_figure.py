"""Verifiability figure for Finding 3 of the report (short_research_report.html).

Flip rates of calculation questions against all other questions, under the controls described
in the text. Top panel: Finding 1's main runs, both models pooled, counted per question. Bottom panel:
Finding 2's Qwen3-4B fixed-question rerun, counted per hinted run. Sources: logs/np_question_labels.csv (requires_calculation,
reasoning_type) joined to logs/per_question_summary_*.csv; the rerun counts come from
logs/results_Qwen3-4B_fixed20_vllm.jsonl (see analysis/fixed_questions_figure.py).

Run: python analysis/verifiability_figure.py
Output: short_research_report.assets/verifiability-checks.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "short_research_report.assets" / "verifiability-checks.png"
CALC, OTHER = "#4a7c59", "#a8a49a"

# Two panels, one per study, because the two count different things.
# (row label, calculation flips, calculation n, other flips, other n)
PANELS = [
    ("A. Main study (Finding 1): Qwen3-1.7B and Qwen3-4B, 120 questions, one hinted run each\n    counted per question", [
        ("All questions", 0, 19, 40, 198),
        ("Is it length?\nOnly long questions", 0, 19, 16, 31),
        ("Is it confidence?\nOnly questions it got right\nin both clean runs", 0, 16, 21, 150),
        ("Is it the number of steps?\nOnly single-step questions", 0, 10, 15, 72),
    ]),
    ("B. Same-question rerun (Finding 2): Qwen3-4B, 20 questions, 30 hinted runs each\n    counted per run", [
        ("All 20 rerun questions", 2, 150, 163, 450),
    ]),
]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "ytick.left": False})
fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True,
                         gridspec_kw={"height_ratios": [len(rows) + 0.4 for _, rows in PANELS], "hspace": 0.5})
h = 0.36
for ax, (title, rows) in zip(axes, PANELS):
    for i, (label, cf, cn, of, on) in enumerate(rows):
        for dy, f, n, color in [(-h / 2, cf, cn, CALC), (h / 2, of, on, OTHER)]:
            rate = 100 * f / n
            ax.barh(i + dy, max(rate, 0.4), h * 0.9, color=color, zorder=2)  # keep 0% bars visible
            ax.text(rate + 1, i + dy, f"{f} of {n}  ({rate:.0f}%)", va="center", fontsize=9.5, color="#333")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.text(-0.36, 1.03, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5,
            fontweight="bold", color="#333")
    ax.grid(axis="x", color="#e8e7e3", zorder=0)
axes[-1].set_xlim(0, 70)
axes[-1].set_xlabel("flipped to the hinted wrong answer (%)")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CALC, label="needs a calculation"),
                        plt.Rectangle((0, 0), 1, 1, color=OTHER, label="no calculation")],
           loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
fig.suptitle("Calculation questions almost never flipped, whichever way the comparison is cut",
             fontsize=12, y=0.99)
fig.subplots_adjust(left=0.27, right=0.97, top=0.8, bottom=0.08)
fig.savefig(OUT, dpi=160)
print(f"Wrote {OUT}")
