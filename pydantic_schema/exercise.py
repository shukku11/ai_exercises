import os
import argparse
import sys
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AzureOpenAI

load_dotenv()

# 1. Define the Schema
class JobPosting(BaseModel):
    title: str
    company: str
    location: str
    salary_range: str | None  # Python 3.10+ syntax for Optional
    skill: list[str] = Field(default_factory=list)
    

# 2. Setup Client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-08-01-preview", # Required for Structured Outputs
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# 3. The Run
RAW_TEXT_VARIATIONS = [
    # Good examples
    "We are looking for a Senior Dev at TechCorp in Lafayette. Pay is 120k-150k.",
    "Hiring: Data Analyst at Insight Labs in Austin. Compensation: $85,000 to $105,000. Skills: SQL, Python, Tableau.",
    "Nexa Systems needs a Cloud Engineer in Seattle. Salary range: 140k-170k. Skills: Azure, Terraform, Kubernetes.",
    # Bad/noisy examples
    "Hello team, lunch is at 1 PM tomorrow. Please RSVP by EOD.",
    "Urgent!!! $$$ Best opportunity ever, DM me now!!!",
    "Company: ??? Role: ??? Location: Mars Salary: infinite",
    "",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pydantic schema extraction demo")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index from RAW_TEXT_VARIATIONS (default: 0)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom raw text input. If provided, this overrides --sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_index = len(RAW_TEXT_VARIATIONS) - 1

    if args.text is not None:
        raw_text = args.text
        print("Using custom text from --text")
    else:
        sample_index = args.sample
        if sample_index < 0 or sample_index > max_index:
            raise ValueError(f"--sample must be between 0 and {max_index}")
        raw_text = RAW_TEXT_VARIATIONS[sample_index]
        print(f"Using sample #{sample_index} of {max_index}")

    try:
        completion = client.beta.chat.completions.parse(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract job fields only from the user text. "
                        "Do not guess or infer missing facts. "
                    ),
                },
                {"role": "user", "content": raw_text},
            ],
            response_format=JobPosting,
        )
    except Exception as exc:
        print(f"Request failed: {exc}")
        return

    message = completion.choices[0].message
    job = message.parsed

    if job is None:
        print("No structured result returned for this input.")
        if getattr(message, "refusal", None):
            print(f"Refusal: {message.refusal}")
        if getattr(message, "content", None):
            print(f"Raw output: {message.content}")
        return

    # 4. Access the data as a Python Object (not a string!)
    print(f"Company: {job.company}")
    print(f"Title: {job.title}")
    print(f"Location: {job.location}")
    print(f"Salary Range: {job.salary_range}")
    print(f"Skills: {', '.join(job.skill)}")


def pause_if_debugging() -> None:
    if sys.gettrace() is not None:
        try:
            input("\nDebug run finished. Press Enter to close... ")
        except EOFError:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        pause_if_debugging()