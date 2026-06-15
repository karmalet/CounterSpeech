#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CounterSpeech generation — unified entry point for the A/B/C conditions.

Reads a pilot hate-speech CSV (columns: id, source, domain, hate_text) and
generates counterspeech under one condition, writing a UNIFIED output schema so
the three conditions can later be concatenated and compared on the SAME inputs:

    A  zero_shot   no guidance, no examples                     (TODO)
    B  strategy    one response per strategy (3 strategies)     (implemented)
    C  rag         retrieval-augmented w/ linguistics knowledge (TODO)

Condition B is the strategy-controlled arm. A and C are scaffolded below so a
future session can fill them in without touching the shared machinery
(cs_gen_common.py) or this dispatch.

Example (condition B):
    python src/generate_counterspeech.py \
      --input pilot_data/05_pilot_300.csv \
      --condition strategy \
      --out-csv outputs/pilot_B_strategy.csv \
      --out-jsonl outputs/pilot_B_strategy.jsonl \
      --model Qwen/Qwen3-32B \
      --base-url http://localhost:8000/v1 \
      --concurrency 8 --resume

Local serving note (ASUS GX10 / GB10): Qwen3 is a reasoning model; its <think>
trace is huge and very slow here (~3 tok/s decode), so thinking is DISABLED by
default (see cs_gen_common.call_chat_completion). Use --concurrency to let vLLM
batch requests.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from prompts_strategy import STRATEGIES, GEN_SYSTEM_PROMPT, GEN_USER_PROMPT_TEMPLATE
from cs_gen_common import generate_json_field, run_concurrent_to_files

# condition key -> the value written into the output `condition` column
CONDITION_LABELS = {
    "zero_shot": "A_zero_shot",
    "strategy": "B_strategy_controlled",
    "rag": "C_rag_assisted",
}

OUTPUT_FIELDS = [
    "pair_id",
    "id",
    "source",
    "domain",
    "condition",
    "strategy",      # populated for B; empty for A; empty for C
    "hate_text",
    "counter_text",
    "parse_ok",
    "model",
]


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\r\n", "\n").replace("\r", "\n")


def find_column(df, candidates):
    cols = list(df.columns)
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# --------------------------------------------------------------------------- #
# Per-condition task builders.
# Each returns a list of task tuples plus a `work(task, args)` closure that
# produces one unified output row dict (must include 'parse_ok').
# --------------------------------------------------------------------------- #
def build_strategy_tasks(rows, condition_label):
    """Condition B: one (hate, strategy) task per strategy -> 3 rows per hate."""
    tasks = []
    for orig_id, source, domain, hate_text in rows:
        for strat in STRATEGIES:
            tasks.append((f"{orig_id}_{strat['key']}", orig_id, source, domain, hate_text, strat))

    def work(task, args):
        pair_id, orig_id, source, domain, hate_text, strat = task
        user_prompt = GEN_USER_PROMPT_TEMPLATE.format(
            strategy_name_zh=strat["name_zh"],
            strategy_name_en=strat["name"],
            strategy_instruction=strat["instruction"],
            hate_text=hate_text,
        )
        gen = generate_json_field(GEN_SYSTEM_PROMPT, user_prompt, "counterspeech", args)
        return {
            "pair_id": pair_id, "id": orig_id, "source": source, "domain": domain,
            "condition": condition_label, "strategy": strat["name"],
            "hate_text": hate_text, "counter_text": gen["value"],
            "parse_ok": gen["parse_ok"], "model": args.model,
            "raw_model_output": gen["raw_model_output"], "error": gen["error"],
        }

    return tasks, work


def build_zero_shot_tasks(rows, condition_label):
    """Condition A (TODO): one task per hate, minimal prompt, no strategy.
    See prompts_strategy.py for a ready ZERO_SHOT prompt to wire in here."""
    raise NotImplementedError(
        "Condition A (zero_shot) is not implemented yet. Wire the ZERO_SHOT prompt "
        "from prompts_strategy.py here (1 task per hate, strategy column empty)."
    )


def build_rag_tasks(rows, condition_label):
    """Condition C (TODO): retrieve linguistics-knowledge chunks per hate and
    inject into the prompt. Needs an embedding model (e.g. BGE-M3) + vector index
    over the repo's linguistics PDFs. Build the index separately, then retrieve
    top-k here and format into the user prompt."""
    raise NotImplementedError(
        "Condition C (rag) is not implemented yet. Build a BGE-M3 + FAISS index "
        "over the linguistics PDFs, retrieve top-k per hate, and inject as context."
    )


BUILDERS = {
    "zero_shot": build_zero_shot_tasks,
    "strategy": build_strategy_tasks,
    "rag": build_rag_tasks,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pilot hate-speech CSV (id, source, domain, hate_text)")
    parser.add_argument("--condition", required=True, choices=list(BUILDERS.keys()))
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-jsonl", default="")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--domains", default="", help="Comma-separated domains to keep (default: all)")
    parser.add_argument("--sample-size", type=int, default=0, help="0 = all rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Qwen3 <think> (default OFF; slow on GX10)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true", help="Skip pair_ids already in out CSV")
    args = parser.parse_args()

    condition_label = CONDITION_LABELS[args.condition]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else None
    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    hate_col = find_column(df, ["hate_text", "hatespeech", "hate_speech", "text"])
    id_col = find_column(df, ["id", "pair_id", "hate_id"])
    src_col = find_column(df, ["source", "src"])
    dom_col = find_column(df, ["domain", "group", "area"])
    if hate_col is None:
        raise ValueError(f"No hate_text column. Available: {list(df.columns)}")

    if args.domains and dom_col:
        keep = {d.strip() for d in args.domains.split(",") if d.strip()}
        before = len(df)
        df = df[df[dom_col].isin(keep)].copy()
        print(f"Filtered domains {sorted(keep)}: {len(df)}/{before} rows")

    if args.sample_size and args.sample_size > 0:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=args.seed)

    rows = []
    for idx, row in df.iterrows():
        hate_text = normalize_text(row.get(hate_col, ""))
        if not hate_text:
            continue
        orig_id = str(row.get(id_col, f"ROW_{idx + 1:05d}")) if id_col else f"ROW_{idx + 1:05d}"
        rows.append((orig_id, row.get(src_col, "") if src_col else "",
                     row.get(dom_col, "") if dom_col else "", hate_text))

    tasks, work = BUILDERS[args.condition](rows, condition_label)

    # Resume: drop already-done pair_ids.
    if args.resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(prev["pair_id"].astype(str)) if "pair_id" in prev.columns else set()
        tasks = [t for t in tasks if t[0] not in done]
        print(f"Resume: {len(done)} already done")

    print(f"Condition: {condition_label}  Inputs: {len(rows)}  To generate: {len(tasks)}")
    print(f"Model: {args.model}  Endpoint: {args.base_url}  Output: {out_csv}")

    n_ok = run_concurrent_to_files(
        tasks=tasks,
        work_fn=lambda t: work(t, args),
        fieldnames=OUTPUT_FIELDS,
        out_csv=out_csv,
        out_jsonl=out_jsonl,
        concurrency=args.concurrency,
        resume=args.resume,
        desc=args.condition,
    )
    print(f"Done. parse_ok: {n_ok}/{len(tasks)}")


if __name__ == "__main__":
    main()
