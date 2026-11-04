"""Run R2R generation and export token-level hardware traces."""

from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

import r2r.models.sglang_patch.sgl_engine_patcher  # noqa: F401
from r2r.models.dynamic_sglang_selector import DynamicSimpleSGLangSelector
from r2r.utils.config import MODEL_DICT


DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain the concept of machine learning in simple terms.",
]


def parse_args():
    p = argparse.ArgumentParser(description="Export R2R token traces for KVNAND simulation.")
    p.add_argument("--router-path", required=True)
    p.add_argument("--model-config", help="Path to an R2R model config JSON/YAML file.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompts-jsonl")
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--tp-size", type=int, default=2)
    p.add_argument("--neural-threshold", type=float)
    p.add_argument("--disable-cuda-graph", action="store_true")
    return p.parse_args()


def load_prompts(path: str | None, limit: int) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS[:limit]
    prompts = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            prompts.append(str(row.get("prompt") or row.get("question") or row["text"]))
    return prompts


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    strategy_kwargs = {"model_path": args.router_path}
    if args.neural_threshold is not None:
        strategy_kwargs["threshold"] = args.neural_threshold

    sglang_kwargs = {
        "dtype": "bfloat16",
        "tp_size": args.tp_size,
    }
    if args.disable_cuda_graph:
        sglang_kwargs["disable_cuda_graph"] = True

    model_config = MODEL_DICT
    if args.model_config:
        with Path(args.model_config).open("r", encoding="utf-8") as f:
            model_config = json.load(f)

    engine = DynamicSimpleSGLangSelector(
        model_config=model_config,
        device="cuda",
        dtype=torch.bfloat16,
        switching_strategy="neural",
        strategy_kwargs=strategy_kwargs,
        is_record=True,
        sglang_kwargs=sglang_kwargs,
    )
    tokenizer = engine.tokenizer
    prompts = load_prompts(args.prompts_jsonl, args.limit)
    prompt_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    batch_input_tokens = tokenizer.batch_encode_plus(prompt_texts, return_tensors=None)["input_ids"]

    generated_texts, recorders = engine.generate(
        batch_input_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        record_generation=True,
        print_tokens=False,
    )

    token_rows = []
    response_rows = []
    for sample_id, recorder in enumerate(recorders):
        df = recorder.to_dataframe()
        if len(df):
            df.insert(0, "sample_id", sample_id)
            df.insert(1, "prompt", prompts[sample_id])
            token_rows.extend(df.to_dict("records"))
        stats = recorder.get_statistics()
        response_rows.append(
            {
                "sample_id": sample_id,
                "prompt": prompts[sample_id],
                "generated_text": generated_texts[sample_id],
                **stats,
            }
        )

    token_path = out / "r2r_tokens.jsonl"
    response_path = out / "r2r_responses.jsonl"
    with token_path.open("w", encoding="utf-8") as f:
        for row in token_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with response_path.open("w", encoding="utf-8") as f:
        for row in response_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(response_rows).to_csv(out / "r2r_summary.csv", index=False)
    print(f"Wrote {token_path}")
    print(f"Wrote {response_path}")
    print(f"Wrote {out / 'r2r_summary.csv'}")
    engine.shutdown()


if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29589")
    os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    mp.set_start_method("spawn", force=True)
    main()
