"""The trace-profile judge on hinted traces, for a figure matching the clean-trace one.

Same frozen `trace_profile` prompt as experiment.py, applied to hinted sample-0 traces with
their own final answers. Results share logs/np_judge_results.jsonl (distinct cache keys).

  python necessity_propensity/hinted_profile.py run --dry-run
  python necessity_propensity/hinted_profile.py run       # resumable
  python necessity_propensity/hinted_profile.py figure    # analysis/necessity_propensity_hinted.png
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as e  # noqa: E402

b2 = e.b2
FIGURE_PATH = e.ROOT / "analysis" / "necessity_propensity_hinted.png"


def hinted_units(spec):
    data, tpl = e.load_dataset(), spec["tasks"]["trace_profile"]["template"]
    units = []
    for tag in e.QWEN:
        for line, r in b2.read_log(tag):
            if r["condition"] != "hinted":
                continue
            reason = b2.exclusion_reason(r)
            trace = None if reason else r["thinking"].strip()
            final = b2.parse_answer(r["content"])
            reason = reason or (None if final else "no final answer")
            text = None if reason else tpl.format(question=e.format_question(data[r["id"]]),
                                                  trace=trace, final=final)
            units.append({"task": "trace_profile", "uid": f"{tag}|{r['key']}", "id": r["id"], "model": tag,
                          "line": line, "trace": trace, "final": final, "excluded": reason, "text": text,
                          "input_sha256": b2.sha256(text) if text else None})
    return units


def run(args):
    spec, hashes = e.load_prompts()
    units = hinted_units(spec)
    cfg = b2.load_config()
    todo = e.plan(units, hashes, cfg["model"], b2.read_results(e.RESULTS_PATH))
    print(f"Hinted traces: {len(units)}, excluded {sum(bool(u['excluded']) for u in units)}, "
          f"planned requests now: {len(todo)} (about {sum(len(u['text']) for _, u in todo) // 4:,} input tokens)")
    if args.dry_run:
        print("Dry run: no API calls made.")
        return
    client, counts = b2.make_client(), Counter()
    with ThreadPoolExecutor(max_workers=6) as pool, open(e.RESULTS_PATH, "a", encoding="utf-8") as f:
        futures = [pool.submit(e.judge_unit, client, cfg, spec, hashes, k, u) for k, u in todo]
        for fut in as_completed(futures):
            records = fut.result()
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            counts[records[-1]["status"]] += 1
    print(f"Run complete: {dict(counts)}. Errors are retried on the next run.")


def figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec, hashes = e.load_prompts()
    units = hinted_units(spec)
    res = e.unit_results(spec, units, hashes, b2.load_config()["model"], b2.read_results(e.RESULTS_PATH))
    data = e.load_dataset()
    qtype = {r["id"]: r["reasoning_type"] for r in b2.csv.DictReader(open(e.QUESTIONS_CSV, encoding="utf-8"))}
    flips = {tag: {q for q, v in b2.question_labels(tag).items() if v["flipped"]} for tag in e.QWEN}
    answers = {tag: {q: v["hinted_0"] for q, v in b2.question_labels(tag).items()} for tag in e.QWEN}
    answers["reference-model"] = {r["id"]: r["answer"] for _, r in b2.read_jsonl(e.LOGS / "results_reference-model.jsonl")
                             if r["condition"] == "hinted"}
    colors = {"Qwen3-1.7B_vllm": "#7aa6d8", "Qwen3-4B_vllm": "#2a62b8", "reference-model": "#9a9a9a"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.26
    for i, model in enumerate(e.QWEN + ["reference-model"]):
        vals, errs = [], [[], []]
        for t in e.TYPES:
            qs = [q for q in qtype if qtype[q] == t]
            k = sum(answers[model][q] == data[q]["correct_answer"] for q in qs)
            lo, hi = b2.wilson(k, len(qs))
            vals.append(k / len(qs))
            errs[0].append(k / len(qs) - lo)
            errs[1].append(hi - k / len(qs))
        label = model.split("_")[0] + (" (no visible CoT)" if model == "reference-model" else " (CoT)")
        a.bar([j + (i - 1) * width for j in range(3)], vals, width, yerr=errs, capsize=3,
              color=colors[model], label=label)
    n = Counter(qtype.values())
    a.set_xticks(range(3), [f"{e.dash(t)}\n(n={n[t]})" for t in e.TYPES])
    a.set_ylim(0, 1.05)
    a.set_ylabel("Hinted accuracy")
    a.set_title("Accuracy with a wrong hint, by reasoning the question needs", fontsize=10)
    a.legend(fontsize=8, loc="lower left", frameon=False)
    for i, model in enumerate(e.QWEN):
        pos = [j + (i - 0.5) * 0.36 for j in range(3)]
        groups = [[(u["id"], res[u["uid"]][2].get("commit_position")) for u in units
                   if u["model"] == model and res[u["uid"]][0] == "ok" and qtype[u["id"]] == t
                   and res[u["uid"]][2].get("commit_position") is not None] for t in e.TYPES]
        bp = b.boxplot([[p for _, p in g] for g in groups], positions=pos, widths=0.3, patch_artist=True,
                       showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[model])
        for j, g in enumerate(groups):
            other = [p for q, p in g if q not in flips[model]]
            flipped = [p for q, p in g if q in flips[model]]
            b.scatter([pos[j]] * len(other), other, s=6, color="#333333", alpha=0.35, zorder=3)
            b.scatter([pos[j]] * len(flipped), flipped, s=22, color="#d6453d", edgecolor="white",
                      linewidth=0.5, zorder=4)
    b.axhline(e.EARLY, color="#bbbbbb", lw=0.8, ls="--")
    b.set_xticks(range(3), [e.dash(t) for t in e.TYPES])
    b.set_ylim(-0.02, 1.02)
    b.set_ylabel("Where the final answer is first committed\n(0 = start of trace, 1 = end)")
    b.set_title("Where the final answer is first committed (hinted traces)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m]) for m in e.QWEN] + [
        plt.Line2D([], [], marker="o", ls="", color="#d6453d", markersize=5)]
    b.legend(handles, [m.split("_")[0] for m in e.QWEN] + ["flipped to the hint"], fontsize=8,
             framealpha=1, edgecolor="white", loc="upper left")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    status = Counter(res[u["uid"]][0] for u in units)
    print(f"Wrote {FIGURE_PATH.relative_to(e.ROOT)}; hinted trace profiles: {dict(status)}")


ACCURACY_PATH = e.ROOT / "analysis" / "clean_vs_hinted_accuracy.png"
COMMIT_PATH = e.ROOT / "analysis" / "clean_vs_hinted_commit.png"
COLORS = {"Qwen3-4B_vllm": "#2a62b8", "Qwen3-1.7B_vllm": "#7aa6d8"}
FLIP = "#d6453d"


def pair(args):
    """Image 1: accuracy, clean vs. hinted. Image 2: commit position, clean vs. hinted,
    with the questions that flipped marked in both panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec, hashes = e.load_prompts()
    model_id = b2.load_config()["model"]
    results = b2.read_results(e.RESULTS_PATH)
    clean = [u for u in e.build_units(spec) if u["task"] == "trace_profile"]
    hinted = hinted_units(spec)
    res = e.unit_results(spec, clean + hinted, hashes, model_id, results)
    data = e.load_dataset()
    qtype = {r["id"]: r["reasoning_type"] for r in b2.csv.DictReader(open(e.QUESTIONS_CSV, encoding="utf-8"))}
    labels = {tag: b2.question_labels(tag) for tag in e.QWEN}
    flips = {tag: {q for q, v in labels[tag].items() if v["flipped"]} for tag in e.QWEN}
    n = Counter(qtype.values())
    xt = [f"{e.dash(t)}\n(n={n[t]})" for t in e.TYPES]
    order = ["Qwen3-4B_vllm", "Qwen3-1.7B_vllm"]

    # Image 1: accuracy
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, field, title in [(axes[0], "clean_0", "Clean (no hint)"), (axes[1], "hinted_0", "With a wrong hint")]:
        for i, tag in enumerate(order):
            vals, lo_err, hi_err = [], [], []
            for t in e.TYPES:
                qs = [q for q in qtype if qtype[q] == t]
                k = sum(labels[tag][q][field] == data[q]["correct_answer"] for q in qs)
                lo, hi = b2.wilson(k, len(qs))
                vals.append(k / len(qs)); lo_err.append(k / len(qs) - lo); hi_err.append(hi - k / len(qs))
            xs = [j + (i - 0.5) * 0.36 for j in range(3)]
            ax.bar(xs, vals, 0.34, yerr=[lo_err, hi_err], capsize=3, color=COLORS[tag], label=tag.split("_")[0])
            for x, v in zip(xs, vals):
                ax.text(x, 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=8, color="white")
        ax.set_xticks(range(3), xt)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy (sample 0, 95% Wilson interval)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Accuracy by the reasoning the question needs", fontsize=11)
    fig.tight_layout()
    fig.savefig(ACCURACY_PATH, dpi=150)
    plt.close(fig)

    # Image 2: commit position, flipped questions marked in both panels
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    counts = {}
    for ax, units, title in [(axes[0], clean, "Clean (no hint)"),
                             (axes[1], hinted, "With a wrong hint")]:
        for i, tag in enumerate(order):
            pos = [j + (i - 0.5) * 0.36 for j in range(3)]
            groups = [[(u["id"], res[u["uid"]][2]["commit_position"]) for u in units
                       if u["model"] == tag and qtype[u["id"]] == t and res[u["uid"]][0] == "ok"
                       and res[u["uid"]][2].get("commit_position") is not None] for t in e.TYPES]
            counts[title, tag] = sum(len(g) for g in groups)
            bp = ax.boxplot([[p for _, p in g] or [float("nan")] for g in groups], positions=pos, widths=0.3,
                            patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(COLORS[tag])
            for j, g in enumerate(groups):
                rest = [p for q, p in g if q not in flips[tag]]
                flipped = [p for q, p in g if q in flips[tag]]
                ax.scatter([pos[j]] * len(rest), rest, s=6, color="#333333", alpha=0.35, zorder=3)
                ax.scatter([pos[j]] * len(flipped), flipped, s=24, color=FLIP, edgecolor="white",
                           linewidth=0.5, zorder=4)
        ax.axhline(e.EARLY, color="#bbbbbb", lw=0.8, ls="--")
        ax.set_xticks(range(3), [e.dash(t) for t in e.TYPES])
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Where the final answer is first committed\n(0 = start of trace, 1 = end)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[t]) for t in order] + [
        plt.Line2D([], [], marker="o", ls="", color=FLIP, markersize=5)]
    fig.legend(handles, [t.split("_")[0] for t in order] + ["question that flipped to the hint"],
               fontsize=8, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Where each trace first commits to its final answer", fontsize=11)
    fig.text(0.5, 0.01, "Right panel: on hinted traces an early commit is often the trace quoting the hint "
             "(\"The teacher says D, but I need to verify\"), not a decision.", ha="center", fontsize=8,
             color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(COMMIT_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {ACCURACY_PATH.relative_to(e.ROOT)} and {COMMIT_PATH.relative_to(e.ROOT)}; "
          f"scored traces with a commit position: " + ", ".join(f"{k[0]} {k[1].split('_')[0]} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run").add_argument("--dry-run", action="store_true")
    sub.add_parser("figure")
    sub.add_parser("pair")
    args = parser.parse_args()
    {"run": run, "figure": figure, "pair": pair}[args.command](args)
