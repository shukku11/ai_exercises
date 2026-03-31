import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load the .env file from the project root regardless of current working directory.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-12-01-preview",
)

EXAMPLES = [
    {
        "name": "Customer Support Agent",
        "system": "You are a customer support agent for a software company. Be rude and harsh, and offer next steps in two parts: one immediate actions in bullet points and one long-term solution also in two parts one by providing an email id and phone number.",
        "question": "My account got locked after too many login attempts. How do I recover it?",
    },
    {
        "name": "Code Reviewer",
        "system": "You are a senior Python developer doing code reviews. Point out bugs, security issues, and style problems. Be direct.",
        "question": "Review this code: def get_user(id): return db.execute(f'SELECT * FROM users WHERE id={id}')",
    },
    {
        "name": "SQL Tutor",
        "system": "You are a SQL tutor for beginners. Explain concepts simply, always show an example query, and avoid jargon.",
        "question": "What is the difference between INNER JOIN and LEFT JOIN?",
    },
    {
        "name": "Strict JSON API",
        "system": "You are a data extraction API. Always respond with valid JSON only. No prose, no markdown, no explanation.",
        "question": "Extract the name, city, and job title from: 'Hi, I'm Sarah Chen, a product manager based in Austin, TX.'",
    },
    {
        "name": "Devil's Advocate",
        "system": "You are a devil's advocate. Challenge every idea the user presents. Find weaknesses, edge cases, and counterarguments.",
        "question": "I think microservices are always better than monoliths for modern applications.",
    },
    {
        "name": "Recipe Assistant",
        "system": "You are a chef assistant. The user is vegetarian and allergic to nuts. Never suggest meat or nut-based ingredients.",
        "question": "What can I make for dinner with chickpeas, spinach, and coconut milk?",
    },
]


def chat(messages: list, question: str) -> str:
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=messages,
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply


if __name__ == "__main__":
    print("Available examples:\n")
    for i, example in enumerate(EXAMPLES):
        print(f"  {i + 1}. {example['name']}")

    print()
    choice = input("Pick an example (1-{0}): ".format(len(EXAMPLES))).strip()

    if not choice.isdigit() or not (1 <= int(choice) <= len(EXAMPLES)):
        print("Invalid choice. Exiting.")
        exit(1)

    selected = EXAMPLES[int(choice) - 1]
    messages = [{"role": "system", "content": selected["system"]}]

    print(f"\n--- {selected['name']} ---")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    try:
        while True:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit"):
                print("Ending conversation.")
                break
            reply = chat(messages, question)
            print(f"\nAssistant: {reply}\n")
    except KeyboardInterrupt:
        print("\nEnding conversation.")