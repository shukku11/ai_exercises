import csv
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI


# Load the root .env so this script works regardless of current working directory.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


@dataclass
class RunResult:
    target_input_tokens: int
    run_number: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    status: str
    error: str = ""


def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    divider = "-+-".join("-" * w for w in widths)
    output = [fmt_row(headers), divider]
    output.extend(fmt_row(row) for row in rows)
    return "\n".join(output)


def build_dummy_text(target_tokens: int) -> str:
    # Approximate tokenizer behavior with a rough ratio. Actual usage is logged from API response.
    approx_chars = target_tokens * 4
    chunk = (
        "This is benchmark filler text for context window measurement. "
        "It contains repeated neutral sentences to simulate payload growth. "
    )
    repeated = (chunk * ((approx_chars // len(chunk)) + 1))[:approx_chars]
    return repeated


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    price_input_per_1k: float,
    price_output_per_1k: float,
) -> float:
    return (prompt_tokens / 1000.0) * price_input_per_1k + (
        completion_tokens / 1000.0
    ) * price_output_per_1k


def main() -> None:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    # Adjust these for your experiment profile.
    target_input_tokens = [1000, 4000, 16000]
    runs_per_size = 5
    max_output_tokens = 120
    temperature = 0

    # Set these in .env to get meaningful cost estimates.
    # Example values (replace with your actual Azure pricing):
    price_input_per_1k = get_env_float("AZURE_PRICE_INPUT_PER_1K", 0.0)
    price_output_per_1k = get_env_float("AZURE_PRICE_OUTPUT_PER_1K", 0.0)

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-12-01-preview",
    )

    results: list[RunResult] = []
    print("Starting context window experiment...\n")

    for size in target_input_tokens:
        print(f"Running target size ~{size} tokens")
        prompt_payload = build_dummy_text(size)

        for run_idx in range(1, runs_per_size + 1):
            messages = [
                {
                    "role": "system",
                    "content": "You are a concise assistant. Summarize the input in 3 bullet points.",
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize the following payload in 3 bullet points:\n\n"
                        f"{prompt_payload}"
                    ),
                },
            ]

            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                )
                latency_ms = (time.perf_counter() - started) * 1000

                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                total_tokens = response.usage.total_tokens if response.usage else 0
                cost = estimate_cost_usd(
                    prompt_tokens,
                    completion_tokens,
                    price_input_per_1k,
                    price_output_per_1k,
                )

                results.append(
                    RunResult(
                        target_input_tokens=size,
                        run_number=run_idx,
                        latency_ms=latency_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        estimated_cost_usd=cost,
                        status="ok",
                    )
                )
                print(
                    f"  Run {run_idx}: {latency_ms:.0f} ms, "
                    f"prompt={prompt_tokens}, completion={completion_tokens}, cost=${cost:.6f}"
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                results.append(
                    RunResult(
                        target_input_tokens=size,
                        run_number=run_idx,
                        latency_ms=latency_ms,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        estimated_cost_usd=0.0,
                        status="error",
                        error=str(exc),
                    )
                )
                print(f"  Run {run_idx}: ERROR after {latency_ms:.0f} ms -> {exc}")

        print()

    out_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"metrics_{timestamp}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "target_input_tokens",
                "run_number",
                "latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "estimated_cost_usd",
                "status",
                "error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.target_input_tokens,
                    r.run_number,
                    f"{r.latency_ms:.3f}",
                    r.prompt_tokens,
                    r.completion_tokens,
                    r.total_tokens,
                    f"{r.estimated_cost_usd:.8f}",
                    r.status,
                    r.error,
                ]
            )

    grouped: dict[int, list[RunResult]] = defaultdict(list)
    for r in results:
        grouped[r.target_input_tokens].append(r)

    summary_rows: list[list[str]] = []
    for size in sorted(grouped):
        ok_runs = [r for r in grouped[size] if r.status == "ok"]
        success_rate = len(ok_runs) / len(grouped[size]) if grouped[size] else 0.0
        if ok_runs:
            avg_latency = statistics.mean(r.latency_ms for r in ok_runs)
            p95_latency = p95([r.latency_ms for r in ok_runs])
            avg_prompt = statistics.mean(r.prompt_tokens for r in ok_runs)
            avg_cost = statistics.mean(r.estimated_cost_usd for r in ok_runs)
            summary_rows.append(
                [
                    str(size),
                    f"{avg_prompt:.0f}",
                    f"{avg_latency:.0f}",
                    f"{p95_latency:.0f}",
                    f"${avg_cost:.6f}",
                    f"{success_rate * 100:.0f}%",
                ]
            )
        else:
            summary_rows.append(
                [str(size), "n/a", "n/a", "n/a", "n/a", f"{success_rate * 100:.0f}%"]
            )

    print("Summary (choose your sweet spot where latency + cost fit product goals):")
    print(
        format_table(
            [
                "target_in",
                "avg_prompt",
                "avg_latency_ms",
                "p95_latency_ms",
                "avg_cost",
                "success_rate",
            ],
            summary_rows,
        )
    )
    print(f"\nSaved raw metrics to: {csv_path}")


if __name__ == "__main__":
    main()