import os
import time
import argparse
import logging
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load .env from the project root regardless of the working directory.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-12-01-preview",
)

# The deployed model name must support vision (e.g. gpt-4o, gpt-4-turbo).
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

STYLE_HINTS = {
    "formal":  "Use professional, precise language.",
    "funny":   "Use playful, humorous language.",
    "neutral": "Use clear, neutral language.",
}

# ---------------------------------------------------------------------------
# Classic chat wrapper  (Exercise 1 — base client)
# ---------------------------------------------------------------------------

def chat(messages: list[dict], system_template: str | None = None) -> str:
    """Base chat wrapper: prepends an optional system template, calls Azure,
    and centrally logs prompt / completion / token counts / latency."""
    full_messages = []
    if system_template:
        full_messages.append({"role": "system", "content": system_template})
    full_messages.extend(messages)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=full_messages,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    reply = response.choices[0].message.content
    usage = response.usage
    logger.info(
        "prompt_tokens=%d  completion_tokens=%d  total_tokens=%d  latency_ms=%.1f",
        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, latency_ms,
    )
    logger.debug("prompt=%s", full_messages)
    logger.debug("completion=%s", reply)
    return reply

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_image_to_base64(image_path: str) -> str:
    """Read a local image file and return its Base64-encoded content."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_image_message(image_source: str, prompt: str) -> list[dict]:
    """
    Build a messages list for a vision request.

    image_source can be:
      - A local file path  (e.g. 'photo.jpg')
      - A public URL       (e.g. 'https://example.com/photo.jpg')
    """
    if image_source.startswith("http://") or image_source.startswith("https://"):
        image_content = {"type": "image_url", "image_url": {"url": image_source}}
    else:
        path = Path(image_source)
        suffix = path.suffix.lstrip(".").lower() or "jpeg"
        mime = f"image/{suffix}"
        b64 = encode_image_to_base64(image_source)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }

    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": prompt},
            ],
        }
    ]


def caption_image(image_source: str, prompt: str | None = None, style: str = "neutral") -> str:
    """Send an image (URL or local path) to the vision model.
    Returns a caption and 3 tags:
        Caption: <text>
        Tags: tag1, tag2, tag3
    """
    style_hint = STYLE_HINTS.get(style, STYLE_HINTS["neutral"])
    effective_prompt = prompt or (
        f"{style_hint} "
        "Respond in exactly two lines:\n"
        "Caption: <one-sentence description>\n"
        "Tags: <tag1>, <tag2>, <tag3>"
    )
    messages = build_image_message(image_source, effective_prompt)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        max_tokens=256,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    reply = response.choices[0].message.content
    usage = response.usage
    logger.info(
        "caption  prompt_tokens=%d  completion_tokens=%d  latency_ms=%.1f",
        usage.prompt_tokens, usage.completion_tokens, latency_ms,
    )
    return reply


def chat_about_image(conversation: list[dict]) -> str:
    """
    Continue a multi-turn conversation that started from an image.

    conversation: list of {"role": "user"|"assistant", "content": str} dicts.
    The first user turn should already contain the image (built via build_image_message).
    Add subsequent turns to the same list to maintain context.
    """
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=conversation,
        max_tokens=1024,
    )

    reply = response.choices[0].message.content

    # Append assistant reply so callers can keep passing the same list.
    conversation.append({"role": "assistant", "content": reply})

    return reply


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Chat Captioner")
    parser.add_argument("--image", help="Path to a local image file to caption.")
    parser.add_argument(
        "--style",
        choices=list(STYLE_HINTS.keys()),
        default="neutral",
        help="Caption style: formal | funny | neutral (default: neutral).",
    )
    args = parser.parse_args()

    sample_url = "https://www.gstatic.com/webp/gallery/1.png"

    # --- Exercise 1: classic chat wrapper ---
    print("=== Classic chat wrapper ===")
    reply = chat(
        messages=[{"role": "user", "content": "In one sentence, what is Azure OpenAI?"}],
        system_template="You are a concise technical assistant.",
    )
    print(reply)

    # --- Exercise 2: multimodal captioner ---
    image_source = args.image if args.image else sample_url
    print(f"\n=== Multimodal caption ({args.style} style) ===")
    print(caption_image(image_source, style=args.style))

    # --- Multi-turn vision chat ---
    print("\n=== Multi-turn vision chat ===")
    conversation = build_image_message(image_source, "What objects are visible in this image?")
    print(f"Turn 1 → {chat_about_image(conversation)}")
    conversation.append({"role": "user", "content": "What colours are most prominent?"})
    print(f"Turn 2 → {chat_about_image(conversation)}")


if __name__ == "__main__":
    main()

    
