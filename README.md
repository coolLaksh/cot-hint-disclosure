# Small reasoning models hide hint-following more when the hint is their own past answer

I plant a wrong answer as a hint in MMLU questions and check whether Qwen3-1.7B and Qwen3-4B follow it, whether their chain of thought (CoT) admits doing so, and whether a monitor reading only the CoT can tell. It follows the hint-injection design of Chen et al. (2025), *Reasoning Models Don't Always Say What They Think*.

**[Read the report](https://coollaksh.github.io/cot-hint-disclosure/short_research_report.html)** — a single self-contained page with every figure embedded. (The [source file](short_research_report.html) renders as HTML only through that link; GitHub shows it as code.)

## Findings

**1. Most flips were disclosed.** When the models switched to the hinted wrong answer, the CoT acknowledged the hint in 14 of 20 flips (Qwen3-1.7B) and 16 of 20 (Qwen3-4B). Accuracy fell from 82 to 71 of 120, and from 99 to 83 of 120.

**2. The question decided whether a hint worked; the hint decided whether it was mentioned.** In a rerun of 20 questions under all six hints (800 runs, Qwen3-4B), five questions never followed any hint and five accounted for most flips. Consistency hints, where the wrong answer is prefilled as the model's own earlier reply, were followed silently most often: 12 of 34 flips never referred to the hint, against 8 of 131 under the other five hints. Calculation questions resisted hints: 0 of 19 flipped in the main study and 2 of 150 hinted runs in the rerun (exploratory).

**3. A CoT-only monitor caught what the CoT admitted, and little else.** In the rerun it flagged 125 of 139 flips that name the hint but only 3 of 20 silent ones, and it flagged 46 of 100 clean traces whose prompts contain worked examples. In the main study it caught 29 of 30 disclosed flips and 1 of 10 undisclosed.

Samples are small and the judges and the monitor all use one model (gpt-5.6-luna), so their errors may be correlated. The rerun's silent-flip result rests mostly on two law questions. The report's Limitations section has the rest.

## Repo map

| Path | Contents |
| --- | --- |
| `short_research_report.html` | The report; `short_research_report.assets/` has its figures |
| `faithfulness_mmlu_hints.jsonl` | Main dataset: 120 questions, clean and hinted prompts |
| `fixed_questions/` | Rerun dataset: 20 questions × 6 hints, and its build script |
| `dataset_creation.ipynb`, `run_llms.ipynb` | Build the dataset; generate runs with vLLM (needs a GPU) |
| `analyze.ipynb`, `analyze_combined.ipynb` | Parsing, flips, the disclosure judge, scores |
| `b2_monitor/` | The CoT-only monitor: frozen prompt, runner, scoring, tests |
| `necessity_propensity/` | Labeling jobs for question type, trace profiles, hint stance |
| `analysis/` | Figure scripts, reports, and `human_review/` with my 20 blind labels |
| `logs/` | Every generation, judge and monitor verdict, and summary table |

## Reproduce

Python 3.10 with `pandas`, `matplotlib`, `openai` and `python-dotenv`. Run from the repo root. Every judge and monitor result is saved, so these reproduce the reported numbers with no API calls:

```bash
python analysis/fixed_questions_figure.py      # and the other analysis/*_figure.py scripts
python b2_monitor/fixed_run.py score
python -m unittest discover -s b2_monitor
python -m unittest discover -s necessity_propensity
```

New runs need a GPU (`run_llms.ipynb`). To rerun the judges or monitor, put `AZURE_APIM_ENDPOINT`, `AZURE-APIM-SUBSCRIPTION-KEY` and `AZURE_DEPLOYMENT` in a `.env` (git-ignored). `monitor.py score` matches saved verdicts by a cache key that includes the deployment name, so it needs `AZURE_DEPLOYMENT=gpt-5.6-luna` set even though it makes no API calls.

## License

MIT ([LICENSE](LICENSE)). Questions come from MMLU (Hendrycks et al., 2021, [arXiv:2009.03300](https://arxiv.org/abs/2009.03300)), also MIT. Outputs were generated with Qwen3-1.7B and Qwen3-4B (Qwen Team, 2025).
