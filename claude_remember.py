"""Let Claude add memories to memory.json via the same embedding pipeline the bot uses."""

import asyncio
import sys

import yaml
from openai import AsyncOpenAI

from semantic_memory import add_memory, load_memories


def load_embedding_client():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    embedding_model_full = cfg.get("embedding_model")
    if not embedding_model_full:
        print("No embedding_model configured in config.yaml")
        sys.exit(1)

    provider, model = embedding_model_full.split("/", 1)
    provider_config = cfg["providers"][provider]
    client = AsyncOpenAI(
        base_url=provider_config["base_url"],
        api_key=provider_config.get("api_key", "sk-no-key-required"),
    )
    return client, model


async def main():
    client, model = load_embedding_client()

    memories = [
        "Hayae (Spring) is the developer and sole user of this Discord bot. He iterates on it conversationally with Claude Code (me).",
        "The bot has a context bridge system: when a context_gap_minutes break is detected, up to context_bridge_tokens (default 1000) worth of messages from before the gap are prepended with a separator, giving the model continuity without a full history load.",
        "Per-turn logging was added: every LLM response and every memory sweep writes a .log file to logs/<date>/<time>_<label>.log capturing the full model input and output.",
        "Semantic memory retrieval now logs near-miss candidates (top 3 below threshold) to the terminal, so Hayae can tell when memories are close but not surfacing.",
        "The memory.json file with embedded vectors is the likely first scalability bottleneck — at ~1000 entries it'd be ~20MB loaded on every retrieval. SQLite would be the natural next step if that becomes an issue.",
    ]

    print(f"Adding {len(memories)} memories...\n")
    for text in memories:
        memory_id = await add_memory(text, client, model)
        print(f"  [{memory_id}] {text[:80]}")

    total = len(load_memories())
    print(f"\nDone. memory.json now has {total} entries.")


if __name__ == "__main__":
    asyncio.run(main())
