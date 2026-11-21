"""Same-question rerun figure for Finding 2 of the report (short_research_report.html).

Qwen3-4B, 20 fixed questions, each run 10 times without a hint and 5 times under each of the
six hints. Each cell counts hinted runs that gave the hinted letter (of 5); the first two columns
count clean runs that already gave that letter (of 5): the plain prompt, which is the control for
five hints, and the worked-example prompt, which is the control for the visual-pattern hint. Rows are sorted by total flips;
calculation questions (logs/np_question_labels.csv, via the source question) are marked.

Run: python analysis/fixed_questions_figure.py
Output: short_research_report.assets/fixed-questions.png
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
LOG = ROOT / "logs" / "results_Qwen3-4B_fixed20_vllm.jsonl"
OUT = ROOT / "short_research_report.assets" / "fixed-questions.png"
TYPES = ["unethical_information", "consistency", "sycophancy", "visual_pattern",
         "grader_hacking", "metadata"]
RX = re.compile(r"Answer\**:\**\s*\(?([ABCD])\b\)?|\\boxed\{\s*(?:\\(?:text|mathrm)\{\s*)?([ABCD])\s*\}")


def parse(c):
    h = RX.findall(c or "")
    return (h[-1][0] or h[-1][1]) if h else None


df = pd.DataFrame(map(json.loads, open(LOG, encoding="utf-8")))
df["q"] = df["id"].str.extract(r"(q\d+)")[0]
df["to_hint"] = df["content"].map(parse) == df["hint"]
sel = pd.read_csv(ROOT / "fixed_questions" / "selected_questions.csv").set_index("question_id")
calc = sel["source_id"].map(pd.read_csv(ROOT / "logs" / "np_question_labels.csv")
                            .set_index("id")["requires_calculation"])

hinted = df[df["condition"] == "hinted"].pivot_table(index="q", columns="hint_type",
                                                     values="to_hint", aggfunc="sum")[TYPES]
clean = df[df["condition"] == "unhinted"].pivot_table(index="q", columns="hint_type",
                                                      values="to_hint", aggfunc="sum")[["sycophancy", "visual_pattern"]]
order = hinted.sum(1).sort_values(ascending=False, kind="stable").index
hinted, clean = hinted.loc[order], clean.loc[order]
SHORT = {"high_school_government_and_politics": "gov. & politics", "professional_": "",
         "high_school_": "", "college_": "", "medical_": "", "_": " "}


def short(subject):
    for a, b in SHORT.items():
        subject = subject.replace(a, b)
    return subject


labels = [f"{q} {short(sel.loc[q, 'subject'])}" + ("  [calc]" if calc[q] else "") for q in order]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("orange", ["#fdf3ec", "#eb6834", "#7a2e0e"])
plt.rcParams.update({"font.size": 10.5})
fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 8), sharey=True,
                             gridspec_kw={"width_ratios": [2, 6], "wspace": 0.04})
for ax, data, top, cols in [(a0, clean.values, 5, ["no hint\nplain", "no hint\nworked ex."]),
                            (a1, hinted.values, 5, [t.replace("_", "\n") for t in TYPES])]:
    ax.imshow(data / top, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(data):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=9.5,
                color="white" if v / top > 0.55 else "#2b2b2b")
    ax.set_xticks(range(len(cols)), cols, fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(order)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
a0.set_yticks(range(len(order)), labels)
for t, q in zip(a0.get_yticklabels(), order):
    if calc[q]:
        t.set_fontweight("bold")
fig.suptitle("Qwen3-4B, same 20 questions under every hint: runs that gave the hinted answer (of 5 per hint)",
             fontsize=11.5, y=0.975)
fig.text(0.5, 0.01, "Bold rows [calc] need a calculation. Rows sorted by total flips. "
         "First two columns: clean runs that already gave the hinted answer.", ha="center", fontsize=9, color="#555")
fig.subplots_adjust(left=0.23, right=0.99, top=0.885, bottom=0.05)
fig.savefig(OUT, dpi=160)
print(f"Wrote {OUT}")
