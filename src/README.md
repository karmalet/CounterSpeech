# `src/` — CounterSpeech generation (conditions A / B / C)

Generation modules for the A/B/C study. All conditions read the **same** pilot
input and write a **unified output schema**, so their CSVs can be concatenated
and compared on identical hate-speech inputs.

| Condition | Method | Status |
|---|---|---|
| **A** `zero_shot` | No guidance, no examples | ⛔ scaffolded (TODO) |
| **B** `strategy` | One response per strategy (Logical Refutation / Positive Reinterpretation / Fact-Based Response) | ✅ implemented |
| **C** `rag` | Retrieval-augmented with linguistics knowledge | ⛔ scaffolded (TODO) |

## Files

- `cs_gen_common.py` — shared request/parse/concurrency machinery (OpenAI-compatible
  endpoint call, JSON-field extraction, concurrent runner that writes CSV+JSONL).
  Condition-agnostic — A and C reuse it as-is.
- `prompts_strategy.py` — the 3 counterspeech strategy definitions + generation
  prompts (condition B). Also holds a ready `ZEROSHOT_*` prompt for condition A.
  Strategy `name` values match the labeling step's `VALID_STRATEGIES`.
- `generate_counterspeech.py` — unified entry point; `--condition {zero_shot,strategy,rag}`
  dispatches to per-condition task builders. Only `strategy` is wired up; A and C
  raise `NotImplementedError` with a pointer to what to add.

## Input / output

Input CSV columns: `id, source, domain, hate_text` (e.g. `pilot_data/05_pilot_300.csv`,
249 rows: race 102 / gender 69 / region 68 / lgbt 10).

Unified output columns:
`pair_id, id, source, domain, condition, strategy, hate_text, counter_text, parse_ok, model`
(JSONL also keeps `raw_model_output`, `error`). `strategy` is populated for B only.

## Run condition B

```bash
# Local serving (ASUS GX10 / GB10): vllm is in the project venv.
#   vllm serve Qwen/Qwen3-32B --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.85
python src/generate_counterspeech.py \
  --input pilot_data/05_pilot_300.csv \
  --condition strategy \
  --out-csv outputs/pilot_B_strategy.csv \
  --out-jsonl outputs/pilot_B_strategy.jsonl \
  --model Qwen/Qwen3-32B \
  --base-url http://localhost:8000/v1 \
  --concurrency 8 --resume
```

249 inputs × 3 strategies = 747 generations (~40 min at concurrency 8).

### Serving notes (important on GX10)
- Qwen3 is a **reasoning model**; its `<think>` trace is huge and decode is
  ~3 tok/s here, so thinking is **disabled by default** (sending
  `chat_template_kwargs={"enable_thinking": false}`). `--enable-thinking` re-enables.
- Use `--gpu-memory-utilization 0.85` (system already holds ~12 GiB of unified memory).
- `--concurrency` lets vLLM batch requests (1→8 gave ~5.7× here). `--resume` skips
  already-written `pair_id`s.

## Adding A and C (next session)

- **A (zero_shot):** implement `build_zero_shot_tasks` — 1 task per hate, use
  `ZEROSHOT_SYSTEM_PROMPT` / `ZEROSHOT_USER_TEMPLATE`, leave `strategy` empty.
- **C (rag):** build a BGE-M3 + FAISS index over the linguistics PDFs in the repo
  root (`44. …比较.pdf`, `45. …統計硏究.pdf`, `4장counterspeech.pdf`), retrieve top-k
  per hate, inject as context into a RAG prompt, then implement `build_rag_tasks`.
