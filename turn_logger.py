"""Write per-turn log files capturing everything the model saw and produced."""

from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs"

_collision_counter: dict[str, int] = {}


def _log_path(label: str) -> Path:
    """Return logs/<YYYY-MM-DD>/<HH-MM-SS>_<label>.log, creating dirs as needed."""
    now = datetime.now()
    day_dir = LOGS_DIR / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    base = f"{now.strftime('%H-%M-%S')}_{label}"
    count = _collision_counter.get(base, 0)
    _collision_counter[base] = count + 1
    filename = f"{base}.log" if count == 0 else f"{base}_{count}.log"
    return day_dir / filename


def log_message_turn(
    model: str,
    system_prompt: str | None,
    messages: list[dict],
    response_text: str,
) -> Path:
    """Log a full message turn (system prompt + messages + response)."""
    label = model.replace("/", "_").replace(":", "_")
    path = _log_path(label)

    lines: list[str] = []
    lines.append(f"=== MODEL: {model} ===\n")

    if system_prompt:
        lines.append("=== SYSTEM PROMPT ===")
        lines.append(system_prompt)
        lines.append("")

    lines.append("=== MESSAGES ===")
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            image_count = sum(1 for p in content if p.get("type") in ("image_url", "image"))
            content = "\n".join(text_parts)
            if image_count:
                content += f"\n[{image_count} image(s) attached]"
        lines.append(f"[{role}]: {content}")
    lines.append("")

    lines.append("=== RESPONSE ===")
    lines.append(response_text)

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def log_sweep_turn(
    model: str,
    prompt: str,
    output: str,
) -> Path:
    """Log a memory sweep turn (full prompt + model output)."""
    path = _log_path("sweep")

    lines: list[str] = []
    lines.append(f"=== MODEL: {model} ===\n")
    lines.append("=== SWEEP PROMPT ===")
    lines.append(prompt)
    lines.append("")
    lines.append("=== SWEEP OUTPUT ===")
    lines.append(output)

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
