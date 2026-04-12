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


def zero_shot_prompt_template(ticket_text: str, labels: list[str]) -> str:
    return (
        "Task: Classify this ticket into exactly one label.\n"
        f"{user_ticket_wrapper(ticket_text)}\n"
        "Return only the label."
    )


def classify_zero_shot(
    client: AzureOpenAI,
    deployment: str,
    ticket_text: str,
    labels: list[str],
) -> str:
    user_prompt = zero_shot_prompt_template(ticket_text, labels)

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
    parser = argparse.ArgumentParser(description="Zero-shot ticket classification demo")
    parser.add_argument(
        "--dataset",
        default="tickets.json",
        help="Dataset file under prompt_engineering (default: tickets.json)",
    )
    args = parser.parse_args()

    tickets = load_tickets(args.dataset)
    labels = sorted({t["label"] for t in tickets})
    client, deployment = init_client()

    print("=== ZERO-SHOT PROMPT TEMPLATE ===")
    #print(zero_shot_prompt_template("<ticket text here>", labels))
    print()

    correct = 0
    print("=== PER-TICKET RESULTS ===")
    for idx, ticket in enumerate(tickets, start=1):
        text = ticket["text"]
        actual = ticket["label"]
        predicted = classify_zero_shot(client, deployment, text, labels)
        ok = predicted == actual
        if ok:
            correct += 1

        print(f"{idx:02d}. text={text}")
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
