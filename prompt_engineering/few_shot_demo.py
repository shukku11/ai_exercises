from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI


SYSTEM_PROMPT = (
    "You are a support ticket classification system.\n\n"
    "You must classify each ticket into exactly ONE category:\n\n"
    "- Billing\n"
    "- Technical Issue\n"
    "- Account Management\n"
    "- Feature Request\n\n"
    "Rules:\n"
    "- Output ONLY the category name\n"
    "- No explanation\n"
    "- No extra text"
)

def load_tickets(dataset_file: str) -> list[dict[str, str]]:
    data_path = Path(__file__).resolve().with_name(dataset_file)
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_client() -> tuple[AzureOpenAI, str]:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    endpoint = os.getenv("ENDPOINT_URL", os.getenv("AZURE_OPENAI_ENDPOINT", "https://prompt-engineering-11.openai.azure.com/"))
    deployment = os.getenv("DEPLOYMENT_NAME", os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"))
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )
    return client, deployment


def user_ticket_wrapper(ticket_text: str) -> str:
    return f"Ticket:\n{ticket_text}"


def select_support_examples(
    tickets: list[dict[str, str]], per_label: int = 1
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = {}

    for ticket in tickets:
        label = ticket["label"]
        current = counts.get(label, 0)
        if current < per_label:
            selected.append(ticket)
            counts[label] = current + 1

    return selected


def few_shot_prompt_template(
    ticket_text: str, labels: list[str], support_examples: list[dict[str, str]]
) -> str:
    lines = [
        "Task: Classify this ticket into one label from the allowed set.",
        "Examples:",
    ]
    for ex in support_examples:
        lines.append(user_ticket_wrapper(ex["text"]))
        lines.append(f"Label: {ex['label']}")
    lines.append(user_ticket_wrapper(ticket_text))
    lines.append("Return only the label.")
    return "\n".join(lines)


def classify_few_shot(
    client: AzureOpenAI,
    deployment: str,
    ticket_text: str,
    labels: list[str],
    support_examples: list[dict[str, str]],
) -> str:
    user_prompt = few_shot_prompt_template(ticket_text, labels, support_examples)

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=20,
        temperature=0,
    )
    raw = (response.choices[0].message.content or "").strip()

    for label in labels:
        if raw.lower() == label.lower():
            return label
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot ticket classification demo")
    parser.add_argument(
        "--dataset",
        default="tickets.json",
        help="Dataset file under prompt_engineering (default: tickets.json)",
    )
    args = parser.parse_args()

    tickets = load_tickets(args.dataset)
    labels = sorted({t["label"] for t in tickets})
    support_examples = select_support_examples(tickets, per_label=1)
    client, deployment = init_client()

    print("=== FEW-SHOT PROMPT TEMPLATE ===")
    print(few_shot_prompt_template("<ticket text here>", labels, support_examples))
    print()

    print("=== SUPPORT EXAMPLES USED ===")
    for idx, ex in enumerate(support_examples, start=1):
        print(f"{idx:02d}. label={ex['label']} | text={ex['text']}")
    print()

    correct = 0
    print("=== PER-TICKET RESULTS ===")
    for idx, ticket in enumerate(tickets, start=1):
        predicted = classify_few_shot(
            client,
            deployment,
            ticket["text"],
            labels,
            support_examples,
        )
        actual = ticket["label"]
        ok = predicted == actual
        if ok:
            correct += 1

        print(f"{idx:02d}. text={ticket['text']}")
        print(f"    predicted={predicted} | actual={actual} | correct={ok}")

    total = len(tickets)
    accuracy = (correct / total) if total else 0.0

    print()
    print("=== SUMMARY ===")
    print(f"Dataset: {args.dataset}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
