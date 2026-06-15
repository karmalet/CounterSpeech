# -*- coding: utf-8 -*-
"""
Shared helpers for CounterSpeech generation across conditions (A/B/C).

Kept deliberately small and dependency-light so each condition script
(b_generate_pilot.py now; zero-shot / RAG later) can import the same
request/parse/concurrency machinery.
"""

import csv as _csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import requests
from tqdm import tqdm


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from model output."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def call_chat_completion(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    api_key: str = "EMPTY",
    enable_thinking: bool = False,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint (e.g. local vLLM)."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Qwen3 is a reasoning model; its <think> trace is huge and, at this box's
        # ~3 tok/s decode, blows past the client timeout. Disable it for this task.
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def generate_json_field(
    system_prompt: str,
    user_prompt: str,
    field: str,
    args,
) -> Dict[str, Any]:
    """Call the model with retries and pull one string field out of its JSON reply.

    Returns {value, raw_model_output, parse_ok, error}. `args` must carry
    base_url, model, temperature, max_tokens, timeout, api_key, enable_thinking,
    retries, retry_sleep.
    """
    raw_output = ""
    parsed = None
    error = ""

    for attempt in range(args.retries + 1):
        try:
            raw_output = call_chat_completion(
                base_url=args.base_url,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                api_key=args.api_key,
                enable_thinking=args.enable_thinking,
            )
            parsed = extract_json_object(raw_output)
            if parsed is not None and str(parsed.get(field, "")).strip():
                break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt < args.retries:
            time.sleep(args.retry_sleep)

    value = ""
    parse_ok = False
    if parsed is not None:
        value = str(parsed.get(field, "")).strip()
        parse_ok = bool(value)

    return {
        "value": value,
        "raw_model_output": raw_output,
        "parse_ok": parse_ok,
        "error": error,
    }


def run_concurrent_to_files(
    tasks: List[Any],
    work_fn: Callable[[Any], Dict[str, Any]],
    fieldnames: List[str],
    out_csv,
    out_jsonl,
    concurrency: int,
    resume: bool,
    extra_jsonl_fn: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None,
    desc: str = "Generating",
) -> int:
    """Run work_fn over tasks concurrently, writing each result row to CSV (and
    optional JSONL) under a lock. work_fn returns the CSV row dict (must include
    a 'parse_ok' key). extra_jsonl_fn(row, task) may add trace-only fields.

    Returns the count of parse_ok rows.
    """
    csv_mode = "a" if (resume and out_csv.exists()) else "w"
    jsonl_mode = "a" if (resume and out_jsonl and out_jsonl.exists()) else "w"
    write_header = not (resume and out_csv.exists())

    csv_f = out_csv.open(csv_mode, encoding="utf-8-sig", newline="")
    writer = _csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction="ignore")
    if write_header:
        writer.writeheader()
    jsonl_f = out_jsonl.open(jsonl_mode, encoding="utf-8") if out_jsonl else None

    write_lock = threading.Lock()

    def wrapped(task):
        row = work_fn(task)
        with write_lock:
            writer.writerow(row)
            csv_f.flush()
            if jsonl_f is not None:
                trace = dict(row)
                if extra_jsonl_fn is not None:
                    trace.update(extra_jsonl_fn(row, task))
                jsonl_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                jsonl_f.flush()
        return bool(row.get("parse_ok"))

    n_ok = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(wrapped, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                if fut.result():
                    n_ok += 1
    finally:
        csv_f.close()
        if jsonl_f is not None:
            jsonl_f.close()
    return n_ok
