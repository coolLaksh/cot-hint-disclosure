# Fixed-question hint run

The same 20 questions under all six hint types, so differences between hint types are no longer mixed with differences between questions.

## Files

- `faithfulness_mmlu_fixed20.jsonl`: 120 records (20 questions × 6 hint types). Same schema as `faithfulness_mmlu_hints.jsonl`, plus:
  - `question_id` (`q00`–`q19`): joins the six hint types of one question.
  - `source_id`: the question's id in the original dataset.
  - `selection_reason`: why the question was picked.
  - `unhinted_prompt`: `plain`, or `fewshot` for visual_pattern, whose control prompt includes the unmarked worked examples.
  - `run_unhinted`: `true` on one record per distinct unhinted prompt (the sycophancy and visual_pattern records).
- `selected_questions.csv`: the 20 questions, their clean-run accuracy and reasoning length.
- `build_fixed_dataset.py`: rebuilds both files from the saved logs and the cached MMLU.

## How the questions were picked

Only the saved **clean** runs were used, with the corrected answer parser. Hinted runs were not looked at. Truncated runs were excluded.

- **10 unstable for 1.7B:** Qwen3-4B right in both clean runs, Qwen3-1.7B right in one of two.
- **10 long reasoning:** both models right in both clean runs, with the longest mean clean reasoning.

Each question keeps its original wrong hint letter, and all six hint types point to that same letter. The unhinted prompts are identical to the ones the models already saw.

## Colab settings

In `run_llms.ipynb`, cell 1:

```python
DATA_PATH = f"{OUT_DIR}/faithfulness_mmlu_fixed20.jsonl"
N_UNHINTED = 5
N_HINTED = 5
OUT_PATH = f"{OUT_DIR}/results_{MODEL_NAME.split('/')[-1]}_fixed20_vllm.jsonl"   # new file, do not mix with the first run
```

In cell 5, change the job loop's first line so identical unhinted prompts run only once per question:

```python
    for cond, n in (("unhinted", N_UNHINTED if r.get("run_unhinted", True) else 0), ("hinted", N_HINTED)):
```

Per model this gives 40 × 5 = 200 unhinted and 120 × 5 = 600 hinted generations: **800 in total**. The first run had 360.

Smoke test first with `LIMIT_PER_TYPE = 1`. It takes question `q00` under all six hints: 10 unhinted + 30 hinted generations.

## Analysis note

For the five plain-prompt hint types, the unhinted samples are stored on the `sycophancy_qNN` record. Join on `question_id` and `unhinted_prompt`, not on `id`.
