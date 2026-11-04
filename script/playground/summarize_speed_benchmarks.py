"""Summarize existing R2R speed benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize_file(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for mode, value in data.items():
        if value is None:
            rows.append({"file": path.name, "mode": mode, "status": "failed"})
            continue
        if isinstance(value, dict):
            rows.append(
                {
                    "file": path.name,
                    "mode": mode,
                    "status": "ok",
                    "num_requests": int(value.get("batch_size", 0)),
                    "total_output_tokens": float(value.get("total_output_tokens", 0)),
                    "total_time_s": float(value.get("total_time_s", 0)),
                    "tokens_per_s": float(value.get("tokens_per_s", 0)),
                    "avg_tokens_per_request": float(value.get("avg_tokens_per_request", 0)),
                }
            )
        else:
            total_tokens = sum(float(r.get("output_tokens", 0)) for r in value)
            total_time = sum(float(r.get("time_s", 0)) for r in value)
            speeds = [float(r.get("tokens_per_s", 0)) for r in value]
            rows.append(
                {
                    "file": path.name,
                    "mode": mode,
                    "status": "ok",
                    "num_requests": len(value),
                    "total_output_tokens": total_tokens,
                    "total_time_s": total_time,
                    "tokens_per_s": total_tokens / total_time if total_time > 0 else 0.0,
                    "mean_per_request_tokens_per_s": sum(speeds) / len(speeds) if speeds else 0.0,
                    "min_tokens_per_s": min(speeds) if speeds else 0.0,
                    "max_tokens_per_s": max(speeds) if speeds else 0.0,
                }
            )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-glob", default="speed_benchmark_*.json")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path(".").glob(args.input_glob)):
        rows.extend(summarize_file(path))
    df = pd.DataFrame(rows)
    df.to_csv(out / "r2r_speed_benchmark_summary.csv", index=False)
    ok = df[df["status"] == "ok"].copy()
    if len(ok):
        plt.figure(figsize=(7, 4))
        plot_df = ok.sort_values(["mode", "file"])
        plt.bar(range(len(plot_df)), plot_df["tokens_per_s"])
        plt.xticks(range(len(plot_df)), plot_df["mode"] + "\n" + plot_df["file"].str[16:29], rotation=45, ha="right")
        plt.ylabel("Tokens/s")
        plt.title("Existing R2R Benchmark Runs")
        plt.tight_layout()
        plt.savefig(out / "r2r_speed_benchmark_tokens_per_s.png", dpi=180)
    print(df.to_string(index=False))
    print(f"Wrote {out / 'r2r_speed_benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
