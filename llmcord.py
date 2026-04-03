import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
import logging
import platform
from typing import Any, Literal, Optional

from PIL import Image

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
import openai
from openai import AsyncOpenAI
import yaml

from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from memory import check_and_run_memory_sweep, collect_since_last_sweep, run_memory_sweep
from semantic_memory import load_core_memory, retrieve_memories
from turn_logger import log_message_turn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGES = 500

SUPPORTED_IMAGE_TYPES = ("image/jpeg", "image/png")


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def message_token_estimate(msg_dict: dict) -> int:
    """Estimate tokens for an OpenAI message dict."""
    content = msg_dict["content"]
    if isinstance(content, str):
        return estimate_tokens(content)
    # list of content parts (text + images)
    return sum(estimate_tokens(p["text"]) for p in content if p.get("type") == "text")


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

def _ensure_jpeg_or_png(content_type: str, data: bytes, max_size: int = 1024) -> tuple[str, bytes]:
    """Convert image to JPEG if it's not already JPEG or PNG, and cap resolution."""
    img = Image.open(io.BytesIO(data))
    resized = max(img.size) > max_size
    if resized:
        img.thumbnail((max_size, max_size))
    if not resized and content_type in SUPPORTED_IMAGE_TYPES:
        return content_type, data
    fmt = "PNG" if content_type == "image/png" else "JPEG"
    buf = io.BytesIO()
    if fmt == "JPEG":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return f"image/{fmt.lower()}", buf.getvalue()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes: dict[int, "MsgNode"] = {}
last_task_time = 0
active_channels: set[int] = set()
channel_locks: dict[int, asyncio.Lock] = {}
session_injected_ids: dict[int, set[str]] = {}  # channel_id -> set of memory IDs injected this session
show_memories_in_chat: dict[int, bool] = {}  # channel_id -> whether to show recalled memories in chat


intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


# ---------------------------------------------------------------------------
# MCP Client
# ---------------------------------------------------------------------------

class MCPClient:
    """Manages persistent connection to local MCP server."""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.tools: list[dict] = []
        self.initialized = asyncio.Event()
        self._error: Optional[str] = None

    async def start(self, server_params: StdioServerParameters) -> None:
        """Start and maintain MCP connection (run as background task)."""
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()

                    # List available tools
                    tools_result = await session.list_tools()
                    self.tools = self._convert_tools_to_anthropic(tools_result.tools)

                    logging.info(f"MCP client connected with {len(self.tools)} tools: {[t['name'] for t in self.tools]}")
                    self.initialized.set()

                    # Keep connection alive
                    await asyncio.Event().wait()
        except Exception as e:
            self._error = str(e)
            logging.exception("MCP client failed to start")
            self.initialized.set()  # Unblock waiters even on error

    def _convert_tools_to_anthropic(self, mcp_tools) -> list[dict]:
        """Convert MCP tool definitions to Anthropic format."""
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in mcp_tools
        ]

    async def call_tool(self, name: str, tool_input: dict) -> str:
        """Call an MCP tool and return the result as string."""
        if not self.session:
            return "Error: MCP session not available"

        try:
            result = await self.session.call_tool(name, tool_input)

            # Convert result to string
            if result.content:
                parts = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        parts.append(item.text)
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            return "Tool executed successfully (no output)"
        except Exception as e:
            logging.exception("Error calling MCP tool %s", name)
            return f"Error calling tool: {str(e)}"

    def is_ready(self) -> bool:
        """Check if MCP client is ready to use."""
        return self.session is not None and self._error is None


mcp_client = MCPClient()


@dataclass
class MsgNode:
    role: Literal["user", "assistant"] = "assistant"

    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)
    image_descriptions: list[str] = field(default_factory=list)

    has_bad_attachments: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ---------------------------------------------------------------------------
# Helper: populate a MsgNode from a Discord message
# ---------------------------------------------------------------------------

async def populate_node(node: MsgNode, msg: discord.Message) -> None:
    """Fill in a MsgNode's text, images, role, and attachment info (idempotent)."""
    if node.text is not None:
        return

    cleaned_content = msg.content.removeprefix(discord_bot.user.mention).lstrip()

    good_attachments = [
        att for att in msg.attachments
        if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))
    ]
    attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

    node.role = "assistant" if msg.author == discord_bot.user else "user"

    node.text = "\n".join(
        ([cleaned_content] if cleaned_content else [])
        + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in msg.embeds]
        + [component.content for component in msg.components if component.type == discord.ComponentType.text_display]
        + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
    )

    node.images = [
        dict(type="image_url", image_url=dict(url=f"data:{mime};base64,{b64encode(data).decode('utf-8')}"))
        for att, resp in zip(good_attachments, attachment_responses)
        if att.content_type.startswith("image")
        for mime, data in [_ensure_jpeg_or_png(att.content_type, resp.content)]
    ]

    if node.role == "assistant" and node.text:
        while node.text.startswith("-# "):
            node.text = node.text.split("\n", 1)[-1] if "\n" in node.text else ""

    if node.role == "user" and (node.text or node.images):
        node.text = f"{msg.author.display_name} (<@{msg.author.id}>): {node.text}"

    if msg.reactions:
        reaction_strs = [f"{r.emoji} x{r.count}" for r in msg.reactions]
        node.text += f"\n[Reactions: {', '.join(reaction_strs)}]"

    node.has_bad_attachments = len(msg.attachments) > len(good_attachments)


def build_message_content(
    node: MsgNode,
    max_text: int,
    max_images: int,
    nodes_needing_descriptions: list[MsgNode],
) -> tuple[dict | None, set[str]]:
    """Build an OpenAI message dict from a populated MsgNode. Returns (message_dict, warnings)."""
    warnings: set[str] = set()

    if node.images[:max_images] and node.image_descriptions:
        img_notes = "\n".join(f"[Image: {desc}]" for desc in node.image_descriptions[:max_images])
        text_part = node.text[:max_text]
        content = f"{text_part}\n{img_notes}" if text_part else img_notes
    elif node.images[:max_images]:
        content = [dict(type="text", text=node.text[:max_text])] + node.images[:max_images]
        nodes_needing_descriptions.append(node)
    else:
        content = node.text[:max_text]

    if len(node.text) > max_text:
        warnings.add(f"⚠️ Max {max_text:,} characters per message")
    if len(node.images) > max_images:
        warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
    if node.has_bad_attachments:
        warnings.add("⚠️ Unsupported attachments")

    if content != "":
        return dict(content=content, role=node.role), warnings
    return None, warnings



# ---------------------------------------------------------------------------
# Permission checking
# ---------------------------------------------------------------------------

def check_permissions(new_msg: discord.Message, cfg: dict) -> bool:
    """Return True if the user/channel is allowed to interact with the bot."""
    is_dm = new_msg.channel.type == discord.ChannelType.private

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    allow_dms = cfg.get("allow_dms", True)
    permissions = cfg["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    return not is_bad_user and not is_bad_channel


# ---------------------------------------------------------------------------
# Interjection gate
# ---------------------------------------------------------------------------

async def check_interjection(new_msg: discord.Message, cfg: dict) -> bool:
    """Decide whether the bot should interject in a conversation it was not mentioned in."""
    if new_msg.channel.id in active_channels:
        return False

    interjection_model_name = cfg.get("interjection_model")
    if not interjection_model_name:
        return False

    gate_msgs = []
    async for msg in new_msg.channel.history(limit=10):
        if msg.type in (discord.MessageType.default, discord.MessageType.reply):
            gate_msgs.append(msg)

    gate_context = "\n".join(
        f"{'[Assistant]' if m.author == discord_bot.user else f'[{m.author.display_name}]'}: {m.content}"
        for m in reversed(gate_msgs)
    )

    ij_provider, ij_model = interjection_model_name.removesuffix(":vision").split("/", 1)
    ij_provider_config = cfg["providers"][ij_provider]
    ij_client = AsyncOpenAI(
        base_url=ij_provider_config["base_url"],
        api_key=ij_provider_config.get("api_key", "sk-no-key-required"),
    )

    gate_system = cfg.get("interjection_prompt", "").strip()
    if not gate_system:
        logging.warning("No interjection_prompt in config, skipping interjection check.")
        return False
    gate_system = gate_system.replace("{bot_name}", discord_bot.user.display_name)

    try:
        gate_resp = await ij_client.chat.completions.create(
            model=ij_model,
            messages=[
                dict(role="system", content=gate_system),
                dict(role="user", content=gate_context),
            ],
            max_tokens=3,
        )
        if not gate_resp.choices or not gate_resp.choices[0].message.content:
            logging.warning("Interjection gate returned empty response")
            return False
        should_interject = gate_resp.choices[0].message.content.strip().upper().startswith("YES")
        if should_interject:
            logging.info(f"Interjection approved for message {new_msg.id}")
        return should_interject
    except openai.RateLimitError as e:
        logging.warning("Interjection gate rate-limited: %s", e)
        return False
    except Exception:
        logging.exception("Interjection gate check failed")
        return False


# ---------------------------------------------------------------------------
# Message chain builders
# ---------------------------------------------------------------------------

async def _scan_context_messages(
    channel: discord.abc.Messageable,
    context_gap_minutes: int,
    max_context_tokens: int,
    context_bridge_tokens: int,
    before: discord.Message | None = None,
    start_time: datetime | None = None,
    skip_msg_ids: set[int] | None = None,
) -> tuple[list[discord.Message], list[discord.Message], int, int, float]:
    """Scan channel history for context messages with gap detection and token budget.

    Returns (recent_msgs, bridge_msgs, recent_tokens, bridge_tokens, gap_minutes).
    """
    recent_msgs: list[discord.Message] = []
    bridge_msgs: list[discord.Message] = []
    recent_tokens = 0
    bridge_tokens = 0
    gap_found = False
    gap_minutes = 0.0
    prev_msg_time = start_time or datetime.now(timezone.utc)

    history_kwargs: dict[str, Any] = dict(limit=MAX_MESSAGES)
    if before is not None:
        history_kwargs["before"] = before

    async for msg in channel.history(**history_kwargs):
        if msg.type not in (discord.MessageType.default, discord.MessageType.reply):
            continue
        if skip_msg_ids and msg.id in skip_msg_ids:
            continue

        if not gap_found:
            time_gap = (prev_msg_time - msg.created_at).total_seconds() / 60
            if recent_msgs and time_gap > context_gap_minutes:
                gap_found = True
                gap_minutes = time_gap
            else:
                msg_tokens = estimate_tokens(msg.content)
                if recent_tokens + msg_tokens > max_context_tokens:
                    break
                recent_tokens += msg_tokens
                recent_msgs.append(msg)
                prev_msg_time = msg.created_at
                continue

        # Collecting bridge messages (gap was found)
        if context_bridge_tokens <= 0:
            break
        msg_tokens = estimate_tokens(msg.content)
        if bridge_tokens + msg_tokens > context_bridge_tokens:
            break
        bridge_tokens += msg_tokens
        bridge_msgs.append(msg)

    return recent_msgs, bridge_msgs, recent_tokens, bridge_tokens, gap_minutes


async def _build_chain_common(
    new_msg: discord.Message,
    cfg: dict,
    max_text: int,
    max_images: int,
    max_context_tokens: int,
    nodes_needing_descriptions: list[MsgNode],
    skip_msg_ids: set[int] | None = None,
) -> tuple[list[dict], set[str], datetime, list[discord.Message], list[discord.Message]]:
    """Shared history scan with gap detection and context bridge.

    Returns (messages, warnings, earliest, recent_msgs, bridge_msgs) where
    messages contains the built bridge separator and message dicts, and
    recent/bridge lists are exposed for callers that need the raw Discord messages.
    """
    messages: list[dict] = []
    warnings: set[str] = set()

    context_gap_minutes = cfg.get("context_gap_minutes", 120)
    context_bridge_tokens = cfg.get("context_bridge_tokens", 1000)

    recent_msgs, bridge_msgs, _, _, gap_minutes = await _scan_context_messages(
        channel=new_msg.channel,
        context_gap_minutes=context_gap_minutes,
        max_context_tokens=max_context_tokens,
        context_bridge_tokens=context_bridge_tokens,
        before=new_msg,
        start_time=new_msg.created_at,
        skip_msg_ids=skip_msg_ids,
    )

    # Build bridge messages (oldest first)
    for msg in reversed(bridge_msgs):
        node = msg_nodes.setdefault(msg.id, MsgNode())
        async with node.lock:
            await populate_node(node, msg)
        result, w = build_message_content(node, max_text, max_images, nodes_needing_descriptions)
        warnings |= w
        if result:
            messages.append(result)

    # Insert separator between bridge and current context
    if bridge_msgs:
        gap_label = f"{int(gap_minutes)} minutes" if gap_minutes < 120 else f"{gap_minutes / 60:.1f} hours"
        messages.append(dict(
            role="user",
            content=f"--- Earlier context (before a gap of {gap_label}) ---",
        ))

    # Build current context messages
    for msg in reversed(recent_msgs):
        node = msg_nodes.setdefault(msg.id, MsgNode())
        async with node.lock:
            await populate_node(node, msg)
        result, w = build_message_content(node, max_text, max_images, nodes_needing_descriptions)
        warnings |= w
        if result:
            messages.append(result)

    # Always include the current message
    node = msg_nodes.setdefault(new_msg.id, MsgNode())
    async with node.lock:
        await populate_node(node, new_msg)
    result, w = build_message_content(node, max_text, max_images, nodes_needing_descriptions)
    warnings |= w
    if result:
        messages.append(result)

    earliest = bridge_msgs[-1].created_at if bridge_msgs else (recent_msgs[-1].created_at if recent_msgs else new_msg.created_at)

    return messages, warnings, earliest, recent_msgs, bridge_msgs


async def build_chain_thread(
    new_msg: discord.Message,
    cfg: dict,
    max_text: int,
    max_images: int,
    max_context_tokens: int,
    nodes_needing_descriptions: list[MsgNode],
) -> tuple[list[dict], set[str], datetime]:
    """Build the message chain from a thread's history with gap/token cutoffs."""
    # Fetch thread starter for public threads (lives in parent channel)
    starter_msg: discord.Message | None = None
    try:
        if new_msg.channel.type == discord.ChannelType.public_thread and new_msg.channel.parent.type == discord.ChannelType.text:
            starter_msg = new_msg.channel.starter_message or await new_msg.channel.parent.fetch_message(new_msg.channel.id)
    except (discord.NotFound, discord.HTTPException):
        logging.exception("Error fetching thread starter message")

    skip_ids = {starter_msg.id} if starter_msg else None
    messages, warnings, earliest, _, _ = await _build_chain_common(
        new_msg, cfg, max_text, max_images, max_context_tokens, nodes_needing_descriptions,
        skip_msg_ids=skip_ids,
    )

    # Prepend thread starter at the top
    if starter_msg:
        node = msg_nodes.setdefault(starter_msg.id, MsgNode())
        async with node.lock:
            await populate_node(node, starter_msg)
        result, w = build_message_content(node, max_text, max_images, nodes_needing_descriptions)
        warnings |= w
        if result:
            messages.insert(0, result)
        earliest = starter_msg.created_at

    return messages, warnings, earliest


async def build_chain_simple(
    new_msg: discord.Message,
    cfg: dict,
    max_text: int,
    max_images: int,
    max_context_tokens: int,
    nodes_needing_descriptions: list[MsgNode],
) -> tuple[list[dict], set[str], datetime]:
    """Build the message chain from DM or channel history with gap/token cutoffs."""
    messages, warnings, earliest, _, _ = await _build_chain_common(
        new_msg, cfg, max_text, max_images, max_context_tokens, nodes_needing_descriptions,
    )
    return messages, warnings, earliest


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

def create_provider_client(cfg: dict, provider_slash_model: str) -> tuple[AsyncOpenAI, str, dict]:
    """Create an OpenAI client from a provider/model string. Returns (client, model, api_kwargs)."""
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
    provider_config = cfg["providers"][provider]

    client = AsyncOpenAI(
        base_url=provider_config["base_url"],
        api_key=provider_config.get("api_key", "sk-no-key-required"),
    )

    model_parameters = cfg["models"].get(provider_slash_model) or {}
    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | model_parameters or None
    # Strip custom keys that shouldn't be sent to the API
    _custom_keys = {"cost_per_million_input_tokens", "cost_per_million_output_tokens"}
    if extra_body:
        extra_body = {k: v for k, v in extra_body.items() if k not in _custom_keys} or None

    return client, model, dict(
        provider=provider,
        provider_config=provider_config,
        model_parameters=model_parameters,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
    )


def create_embedding_client(cfg: dict) -> tuple[AsyncOpenAI | None, str | None]:
    """Create an embedding client from config. Returns (client, model) or (None, None)."""
    embedding_model_full = cfg.get("embedding_model")
    if not embedding_model_full:
        return None, None

    emb_provider, emb_model = embedding_model_full.split("/", 1)
    emb_provider_config = cfg["providers"][emb_provider]
    emb_client = AsyncOpenAI(
        base_url=emb_provider_config["base_url"],
        api_key=emb_provider_config.get("api_key", "sk-no-key-required"),
    )
    return emb_client, emb_model


def build_system_prompt(cfg: dict) -> str | None:
    """Build the system prompt with date/time and memory injected."""
    system_prompt = cfg.get("system_prompt")
    if not system_prompt:
        return None

    now = datetime.now().astimezone()
    system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()

    core_memory = load_core_memory()
    if core_memory.strip():
        system_prompt += (
            "\n\nYou have a persistent memory system. Below are your core memories — enduring identity-level facts. "
            "Use them naturally — don't reference the system itself, don't announce what you remember unless it's relevant. "
            "Just let the context inform how you respond.\n\n"
            "Additional memories may be attached to individual messages based on relevance. "
            "If any memory seems outdated or contradicted by the conversation, note it internally — "
            "it will be reviewed at the end of the session.\n\n"
            f"<core_memory>\n{core_memory}</core_memory>"
        )

    return system_prompt


# ---------------------------------------------------------------------------
# Response streaming
# ---------------------------------------------------------------------------

async def stream_response(
    new_msg: discord.Message,
    openai_client: AsyncOpenAI,
    openai_kwargs: dict,
    user_warnings: set[str],
    context_info: str,
    use_plain_responses: bool,
) -> tuple[list[discord.Message], str]:
    """Stream the LLM response, sending/editing Discord messages. Returns (response_msgs, full_text)."""
    global last_task_time

    curr_content = finish_reason = None
    response_msgs: list[discord.Message] = []
    response_contents: list[str] = []

    if use_plain_responses:
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        fields = [dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]
        fields.append(dict(name=context_info, value="", inline=False))
        embed = discord.Embed.from_dict(dict(fields=fields))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    active_channels.add(new_msg.channel.id)
    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason is not None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason is None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason is None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason is not None or msg_split_incoming
                    is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                            # Clear fields after the first message so subsequent splits don't repeat them
                            embed.clear_fields()
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for i, content in enumerate(response_contents):
                    if i == 0:
                        content = f"-# {context_info}\n{content}"
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception as e:
        logging.exception("Error while generating response")
        if not response_msgs:
            error_type = type(e).__name__
            error_brief = str(e).split("\n")[0][:200] if str(e) else "Unknown error"
            error_text = f"⚠️ **{error_type}**: {error_brief}"
            try:
                if use_plain_responses:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=error_text)))
                else:
                    error_embed = discord.Embed(description=error_text, color=EMBED_COLOR_INCOMPLETE)
                    await reply_helper(embed=error_embed, silent=True)
            except Exception:
                logging.exception("Error while sending error message")

    active_channels.discard(new_msg.channel.id)
    return response_msgs, "".join(response_contents)


# ---------------------------------------------------------------------------
# Image description caching
# ---------------------------------------------------------------------------

async def cache_image_descriptions(
    nodes: list[MsgNode],
    openai_client: AsyncOpenAI,
    model: str,
    max_images: int,
    **api_kwargs,
) -> None:
    """Generate and cache text descriptions for image attachments."""
    for node in nodes:
        if node.image_descriptions:
            continue
        descriptions = []
        for img in node.images[:max_images]:
            try:
                desc_resp = await openai_client.chat.completions.create(
                    model=model,
                    messages=[dict(role="user", content=[
                        dict(type="text", text="Describe this image in one brief sentence."),
                        img,
                    ])],
                    max_tokens=100,
                    extra_headers=api_kwargs.get("extra_headers"),
                    extra_query=api_kwargs.get("extra_query"),
                    extra_body=api_kwargs.get("extra_body"),
                )
                descriptions.append(desc_resp.choices[0].message.content.strip())
            except Exception:
                logging.exception("Error generating image description")
                descriptions.append("(image description unavailable)")
        node.image_descriptions = descriptions
        logging.info(f"Cached {len(descriptions)} image description(s) for message node")


# ---------------------------------------------------------------------------
# Node cache cleanup
# ---------------------------------------------------------------------------

async def cleanup_old_nodes() -> None:
    """Evict the oldest MsgNodes if the cache exceeds MAX_MESSAGES."""
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


# ---------------------------------------------------------------------------
# Discord commands and events
# ---------------------------------------------------------------------------

@discord_bot.tree.command(name="info", description="Estimate the number of tokens in the current chat context")
async def info_command(interaction: discord.Interaction) -> None:
    cfg = await asyncio.to_thread(get_config)
    max_context_tokens = cfg.get("max_context_tokens", 10000)
    context_gap_minutes = cfg.get("context_gap_minutes", 120)
    context_bridge_tokens = cfg.get("context_bridge_tokens", 1000)

    # --- System prompt + core memory ---
    system_prompt = build_system_prompt(cfg)
    system_tokens = estimate_tokens(system_prompt) if system_prompt else 0

    # --- MCP tool definitions ---
    tools = mcp_client.tools if mcp_client.is_ready() else []
    tool_tokens = estimate_tokens(json.dumps(tools)) if tools else 0

    # --- Message budget after reserving for system/tools ---
    reserved_tokens = system_tokens + tool_tokens
    message_budget = max(0, max_context_tokens - reserved_tokens)

    # --- Scan messages using shared logic ---
    channel = interaction.channel
    recent_msgs, bridge_msgs, recent_tokens, bridge_tokens, _ = await _scan_context_messages(
        channel=channel,
        context_gap_minutes=context_gap_minutes,
        max_context_tokens=message_budget,
        context_bridge_tokens=context_bridge_tokens,
    )

    msg_count = len(recent_msgs)
    bridge_count = len(bridge_msgs)
    message_tokens = recent_tokens + bridge_tokens
    total_tokens = reserved_tokens + message_tokens

    earliest_time = (
        bridge_msgs[-1].created_at if bridge_msgs
        else recent_msgs[-1].created_at if recent_msgs
        else datetime.now(timezone.utc)
    )
    earliest_ts = int(earliest_time.timestamp())

    # --- Format output ---
    lines = ["**Context estimate:**"]
    lines.append(f"Messages: {msg_count} (~{int(recent_tokens):,} tokens)")
    if bridge_count:
        lines.append(f"Bridge messages: {bridge_count} (~{int(bridge_tokens):,} tokens)")
    lines.append(f"System prompt: ~{system_tokens:,} tokens")
    if tool_tokens:
        lines.append(f"MCP tools ({len(tools)}): ~{tool_tokens:,} tokens")
    lines.append(f"**Total: ~{int(total_tokens):,} / {max_context_tokens:,} tokens**")
    lines.append(f"Earliest message: <t:{earliest_ts}:R>")

    # --- Cost estimate (input only, output unknown) ---
    model_params = cfg["models"].get(curr_model) or {}
    cost_in = model_params.get("cost_per_million_input_tokens")
    cost_out = model_params.get("cost_per_million_output_tokens")
    if cost_in is not None and cost_out is not None:
        input_cost = int(total_tokens) * float(cost_in) / 1_000_000
        lines.append(f"Input cost estimate: ~${input_cost:.4f} (${cost_in}/M in, ${cost_out}/M out)")

    await interaction.response.send_message("\n".join(lines), ephemeral=True)


@discord_bot.tree.command(name="memories", description="Toggle whether recalled memories are shown in chat")
async def memories_command(interaction: discord.Interaction) -> None:
    channel_id = interaction.channel_id
    current = show_memories_in_chat.get(channel_id, True)
    show_memories_in_chat[channel_id] = not current
    state = "shown" if not current else "hidden"
    await interaction.response.send_message(
        f"Recalled memories will now be **{state}** in chat.",
        ephemeral=True,
    )


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.tree.command(name="sweep", description="Manually trigger a memory sweep on messages since the last sweep")
async def sweep_command(interaction: discord.Interaction) -> None:
    cfg = await asyncio.to_thread(get_config)

    admin_ids = cfg["permissions"]["users"]["admin_ids"]
    sweep_ids = cfg["permissions"]["users"].get("sweep_ids", [])
    if interaction.user.id not in admin_ids and interaction.user.id not in sweep_ids:
        await interaction.response.send_message("You don't have permission to do that.", ephemeral=True)
        return

    await interaction.response.send_message("🧠 Manual sweep starting...", ephemeral=True)

    # Set up LLM client
    openai_client, model, api_kwargs = create_provider_client(cfg, curr_model)
    extra_headers = api_kwargs["extra_headers"]
    extra_query = api_kwargs["extra_query"]
    extra_body = api_kwargs["extra_body"]

    # Set up embedding client
    emb_client, emb_model = create_embedding_client(cfg)

    # Collect and sweep
    session_msgs = await collect_since_last_sweep(interaction.channel, discord_bot.user)
    channel_injected = session_injected_ids.get(interaction.channel.id, set())

    await run_memory_sweep(
        interaction.channel, session_msgs, openai_client, model,
        injected_ids=channel_injected,
        embedding_model=emb_model,
        embedding_client=emb_client,
        extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body,
    )

    # Clear injected IDs for this channel since we just swept
    session_injected_ids.pop(interaction.channel.id, None)


def _check_provider_api_keys(cfg: dict) -> None:
    """Warn on startup if any actively-used remote provider is missing an API key."""
    models_to_check: list[tuple[str, str]] = []  # (label, provider/model)

    models_to_check.append(("model", curr_model))
    if ij := cfg.get("interjection_model"):
        models_to_check.append(("interjection_model", ij))
    if emb := cfg.get("embedding_model"):
        models_to_check.append(("embedding_model", emb))

    for label, provider_slash_model in models_to_check:
        provider = provider_slash_model.removesuffix(":vision").split("/", 0 + 1)[0]
        provider_config = cfg.get("providers", {}).get(provider)
        if provider_config is None:
            logging.warning(f"Config '{label}' references unknown provider '{provider}'")
            continue
        base_url = provider_config.get("base_url", "")
        is_local = "localhost" in base_url or "127.0.0.1" in base_url
        has_key = bool(provider_config.get("api_key"))
        if not is_local and not has_key:
            YELLOW = "\033[93m"
            CYAN = "\033[96m"
            BOLD = "\033[1m"
            RESET = "\033[0m"
            feature_hint = ""
            if label == "interjection_model":
                feature_hint = f"\n  {YELLOW}Interjections (spontaneous replies) will {BOLD}not work{RESET}{YELLOW} without this key.{RESET}"
            elif label == "embedding_model":
                feature_hint = f"\n  {YELLOW}Memory (semantic recall & sweep) will {BOLD}not work{RESET}{YELLOW} without this key.{RESET}"
            print(f"\n{YELLOW}{'=' * 60}")
            print(f"  {BOLD}WARNING: '{label}' provider '{provider}' has no api_key{RESET}{YELLOW}")
            print(f"{'=' * 60}{RESET}")
            print(f"\n  {YELLOW}Set the API key in {CYAN}config.yaml{RESET}{YELLOW} under:{RESET}")
            print(f"  {CYAN}providers → {provider} → api_key{RESET}{feature_hint}")
            print(f"\n{YELLOW}{'=' * 60}{RESET}\n")


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        invite_url = f"https://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot"
        GREEN = "\033[92m"
        CYAN = "\033[96m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        print(f"\n{GREEN}{'=' * 60}")
        print(f"  {BOLD}BOT INVITE URL{RESET}{GREEN}")
        print(f"{'=' * 60}{RESET}")
        print(f"\n  {CYAN}{BOLD}{invite_url}{RESET}\n")
        print(f"{GREEN}{'=' * 60}{RESET}\n")

    _check_provider_api_keys(config)

    try:
        await discord_bot.tree.sync()
    except discord.errors.DiscordServerError:
        logging.warning("Failed to sync slash commands (Discord server error). Commands will sync on next successful startup.")

    # Start MCP client
    mcp_cfg = config.get("mcp", {})
    if mcp_cfg.get("enabled", False):
        logging.info("Starting MCP client...")
        mcp_server_params = StdioServerParameters(
            command=mcp_cfg.get("command", "python"),
            args=mcp_cfg.get("args", []),
            env=mcp_cfg.get("env"),
        )
        asyncio.create_task(mcp_client.start(mcp_server_params))
        await mcp_client.initialized.wait()

        if mcp_client.is_ready():
            logging.info("MCP client ready!")
        else:
            logging.warning("MCP client failed to initialize: %s", mcp_client._error)
    else:
        logging.info("MCP disabled in config")


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    is_dm = new_msg.channel.type == discord.ChannelType.private
    is_thread = new_msg.channel.type in (discord.ChannelType.public_thread, discord.ChannelType.private_thread)

    if new_msg.author.bot or new_msg.content.startswith("."):
        return

    is_mentioned = is_dm or is_thread or discord_bot.user in new_msg.mentions

    cfg = await asyncio.to_thread(get_config)

    if not check_permissions(new_msg, cfg):
        return

    if not is_mentioned and not await check_interjection(new_msg, cfg):
        return

    # --- Acquire per-channel lock to serialize responses ---
    lock = channel_locks.setdefault(new_msg.channel.id, asyncio.Lock())
    async with lock:
        # --- Provider / model setup ---
        provider_slash_model = curr_model
        openai_client, model, api_kwargs = create_provider_client(cfg, provider_slash_model)
        provider = api_kwargs["provider"]
        provider_config = api_kwargs["provider_config"]
        model_parameters = api_kwargs["model_parameters"]
        extra_headers = api_kwargs["extra_headers"]
        extra_query = api_kwargs["extra_query"]
        extra_body = api_kwargs["extra_body"]

        accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
        max_text = cfg.get("max_text", 100000)
        max_images = cfg.get("max_images", 5) if accept_images else 0
        max_context_tokens = cfg.get("max_context_tokens", 10000)

        # --- Reserve token budget for system prompt, tools, and memory ---
        system_prompt_preview = build_system_prompt(cfg)
        system_tokens = estimate_tokens(system_prompt_preview) if system_prompt_preview else 0

        tools_preview = mcp_client.tools if mcp_client.is_ready() else []
        tool_tokens = estimate_tokens(json.dumps(tools_preview)) if tools_preview else 0

        reserved_tokens = system_tokens + tool_tokens
        message_budget = max(0, max_context_tokens - reserved_tokens)

        # --- Build message chain ---
        nodes_needing_descriptions: list[MsgNode] = []

        if is_thread:
            messages, user_warnings, earliest_msg_time = await build_chain_thread(
                new_msg, cfg, max_text, max_images, message_budget, nodes_needing_descriptions,
            )
        else:
            messages, user_warnings, earliest_msg_time = await build_chain_simple(
                new_msg, cfg, max_text, max_images, message_budget, nodes_needing_descriptions,
            )

        logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

        # --- Embedding client setup ---
        emb_client, emb_model = create_embedding_client(cfg)

        # --- Memory sweep ---
        context_gap_minutes = cfg.get("context_gap_minutes", 120)
        channel_injected = session_injected_ids.get(new_msg.channel.id, set())
        new_session = await check_and_run_memory_sweep(
            new_msg, discord_bot.user, openai_client, model,
            injected_ids=channel_injected,
            embedding_model=emb_model,
            embedding_client=emb_client,
            gap_minutes=context_gap_minutes,
            extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body,
        )
        if new_session:
            session_injected_ids.pop(new_msg.channel.id, None)

        # --- Semantic memory retrieval ---
        if emb_client and new_msg.content.strip():
            try:
                channel_injected = session_injected_ids.setdefault(new_msg.channel.id, set())
                retrieved = await retrieve_memories(
                    new_msg.content, emb_client, emb_model,
                    exclude_ids=channel_injected,
                )

                if retrieved:
                    memory_block = "\n".join(
                        f"[Memory {m['id']}: {m['text']}]" for m in retrieved
                    )
                    # Append to the last user message in the chain
                    for msg_dict in reversed(messages):
                        if msg_dict["role"] == "user":
                            if isinstance(msg_dict["content"], str):
                                msg_dict["content"] += f"\n\n{memory_block}"
                            elif isinstance(msg_dict["content"], list):
                                for part in msg_dict["content"]:
                                    if part.get("type") == "text":
                                        part["text"] += f"\n\n{memory_block}"
                                        break
                            break

                    channel_injected.update(m["id"] for m in retrieved)
                    logging.info(f"Injected {len(retrieved)} memories: {[m['id'] for m in retrieved]}")

                    # Show recalled memories in chat unless hidden
                    if show_memories_in_chat.get(new_msg.channel.id, True):
                        recall_lines = [f"-# `{m['id']}` ({m['score']:.2f}) {m['text'][:80]}" for m in retrieved]
                        await new_msg.channel.send(
                            f"-# 🔍 Recalled {len(retrieved)} memories:\n" + "\n".join(recall_lines)
                        )

            except Exception:
                logging.exception("Semantic memory retrieval failed")

        # --- System prompt & message ordering ---
        ordered_messages = messages
        system_prompt = build_system_prompt(cfg)

        earliest_timestamp = int(earliest_msg_time.timestamp())
        context_info = f"🌞 Earliest message: <t:{earliest_timestamp}:t>"
        use_plain_responses = cfg.get("use_plain_responses", False)

        # --- Stream response ---
        # Use Anthropic SDK for Anthropic provider, OpenAI SDK for others
        if provider == "anthropic":
            # Use Anthropic SDK with MCP tools
            anthropic_client = AsyncAnthropic(api_key=provider_config.get("api_key"))

            # Get MCP tools if available
            tools = mcp_client.tools if mcp_client.is_ready() else []

            # Extract model parameters
            max_tokens = model_parameters.get("max_tokens", 4096) if model_parameters else 4096
            temperature = model_parameters.get("temperature", 1.0) if model_parameters else 1.0

            response_msgs, full_response = await stream_response_anthropic(
                new_msg, anthropic_client, model, ordered_messages, system_prompt,
                tools, max_tokens, temperature, user_warnings, context_info, use_plain_responses,
            )
        else:
            # Use OpenAI SDK for other providers
            if system_prompt:
                ordered_messages.insert(0, dict(role="system", content=system_prompt))

            openai_kwargs = dict(model=model, messages=ordered_messages, stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

            response_msgs, full_response = await stream_response(
                new_msg, openai_client, openai_kwargs, user_warnings, context_info, use_plain_responses,
            )

        # --- Estimate cost and update first response message ---
        cost_input_rate = model_parameters.get("cost_per_million_input_tokens") if model_parameters else None
        cost_output_rate = model_parameters.get("cost_per_million_output_tokens") if model_parameters else None
        if cost_input_rate is not None and cost_output_rate is not None and response_msgs:
            input_tokens = sum(estimate_tokens(json.dumps(m.get("content", ""))) for m in ordered_messages)
            input_tokens += estimate_tokens(system_prompt or "")
            output_tokens = estimate_tokens(full_response)
            input_cost = input_tokens * float(cost_input_rate) / 1_000_000
            output_cost = output_tokens * float(cost_output_rate) / 1_000_000
            total_cost = input_cost + output_cost
            cost_info = f"💰 ~${total_cost:.4f} ({input_tokens:,}in + {output_tokens:,}out)"
            try:
                first_msg = response_msgs[0]
                if use_plain_responses:
                    # Plain response: replace context_info line with cost appended
                    updated_context = f"-# {context_info}  ·  {cost_info}"
                    old_prefix = f"-# {context_info}"
                    # Reconstruct content: the first message starts with "-# context_info\n" + response text
                    new_content = f"{updated_context}\n{full_response[:4000 - len(updated_context) - 1]}"
                    await first_msg.edit(view=LayoutView().add_item(TextDisplay(content=new_content)))
                else:
                    # Embed response: add cost as an embed field
                    if first_msg.embeds:
                        embed = first_msg.embeds[0]
                        embed.set_field_at(len(embed.fields) - 1, name=f"{context_info}  ·  {cost_info}", value="", inline=False)
                        await first_msg.edit(embed=embed)
            except Exception:
                logging.exception("Failed to add cost estimate to response")

        # --- Log the full turn ---
        try:
            log_message_turn(
                model=provider_slash_model,
                system_prompt=system_prompt,
                messages=ordered_messages,
                response_text=full_response,
            )
        except Exception:
            logging.exception("Failed to write turn log")

        # --- Finalize response nodes ---
        for response_msg in response_msgs:
            msg_nodes[response_msg.id].text = full_response
            msg_nodes[response_msg.id].lock.release()

        # --- Cache image descriptions ---
        await cache_image_descriptions(
            nodes_needing_descriptions, openai_client, model, max_images,
            extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body,
        )

    # --- Cleanup old nodes ---
    await cleanup_old_nodes()


# ---------------------------------------------------------------------------
# Anthropic message format conversion
# ---------------------------------------------------------------------------

def convert_messages_for_anthropic(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages to Anthropic format.

    Handles:
    - image_url blocks -> Anthropic image blocks with base64 source
    - Merging consecutive same-role messages (Anthropic requires alternation)
    - Skipping system messages (handled separately by Anthropic API)
    """
    result: list[dict] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            continue

        # Convert content
        if isinstance(content, str):
            anthropic_content = content
        elif isinstance(content, list):
            anthropic_content = []
            for part in content:
                if part.get("type") == "text":
                    anthropic_content.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        anthropic_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        })
            if not anthropic_content:
                continue
        else:
            continue

        # Merge with previous message if same role
        if result and result[-1]["role"] == role:
            prev = result[-1]["content"]
            if isinstance(prev, str) and isinstance(anthropic_content, str):
                result[-1]["content"] = prev + "\n" + anthropic_content
            else:
                prev_list = [{"type": "text", "text": prev}] if isinstance(prev, str) else prev
                new_list = [{"type": "text", "text": anthropic_content}] if isinstance(anthropic_content, str) else anthropic_content
                result[-1]["content"] = prev_list + new_list
        else:
            result.append({"role": role, "content": anthropic_content})

    # Anthropic requires first message to be "user"
    if result and result[0]["role"] != "user":
        result.insert(0, {"role": "user", "content": "(conversation context)"})

    return result


# ---------------------------------------------------------------------------
# Anthropic streaming with tool support
# ---------------------------------------------------------------------------

async def stream_response_anthropic(
    new_msg: discord.Message,
    anthropic_client: AsyncAnthropic,
    model: str,
    messages: list[dict],
    system: Optional[str],
    tools: list[dict],
    max_tokens: int,
    temperature: float,
    user_warnings: set[str],
    context_info: str,
    use_plain_responses: bool,
) -> tuple[list[discord.Message], str]:
    """Stream response from Anthropic with MCP tool support. Returns (response_msgs, full_text)."""
    global last_task_time

    response_msgs: list[discord.Message] = []
    response_contents: list[str] = []

    if use_plain_responses:
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        fields = [dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]
        fields.append(dict(name=context_info, value="", inline=False))
        embed = discord.Embed.from_dict(dict(fields=fields))
        first_embed_sent = False

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    active_channels.add(new_msg.channel.id)

    try:
        # Convert messages to Anthropic format
        conversation_messages = convert_messages_for_anthropic(messages)

        iteration = 0
        max_iterations = 10

        while iteration < max_iterations:
            iteration += 1
            tool_uses: list[dict] = []
            iteration_text = ""  # Text produced in this iteration only

            # Build API kwargs, only including system/tools when present
            create_kwargs: dict[str, Any] = dict(
                model=model,
                messages=conversation_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            if system:
                create_kwargs["system"] = system
            if tools:
                create_kwargs["tools"] = tools

            async with new_msg.channel.typing():
                stream = await anthropic_client.messages.create(**create_kwargs)

                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            tool_uses.append({
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input_json": "",
                            })

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            chunk = event.delta.text
                            iteration_text += chunk

                            # Append chunk to response_contents
                            if not response_contents:
                                response_contents.append("")

                            # Start new Discord message if current one would overflow
                            if len(response_contents[-1]) + len(chunk) > max_message_length:
                                # Finalize current message
                                if not use_plain_responses and response_msgs:
                                    embed.description = response_contents[-1]
                                    embed.color = EMBED_COLOR_COMPLETE
                                    await response_msgs[-1].edit(embed=embed)

                                response_contents.append("")
                                # Force new Discord message on next update
                                start_new_msg = True
                            else:
                                start_new_msg = not response_contents[-1] and not response_msgs

                            response_contents[-1] += chunk

                            # Update Discord embed periodically
                            if not use_plain_responses:
                                time_delta = datetime.now().timestamp() - last_task_time

                                if time_delta >= EDIT_DELAY_SECONDS or start_new_msg:
                                    embed.description = response_contents[-1] + STREAMING_INDICATOR
                                    embed.color = EMBED_COLOR_INCOMPLETE

                                    if start_new_msg or len(response_msgs) < len(response_contents):
                                        await reply_helper(embed=embed, silent=True)
                                        if not first_embed_sent:
                                            embed.clear_fields()
                                            first_embed_sent = True
                                    else:
                                        await response_msgs[-1].edit(embed=embed)

                                    last_task_time = datetime.now().timestamp()

                        elif event.delta.type == "input_json_delta":
                            if tool_uses:
                                tool_uses[-1]["input_json"] += event.delta.partial_json

            # Parse accumulated JSON inputs for tool calls
            for tool_use in tool_uses:
                raw = tool_use.pop("input_json")
                try:
                    tool_use["input"] = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    tool_use["input"] = {}

            # Finalize current text in embed (mark complete or in-progress)
            if not use_plain_responses and response_contents and response_contents[-1] and response_msgs:
                embed.description = response_contents[-1]
                embed.color = EMBED_COLOR_COMPLETE if not tool_uses else EMBED_COLOR_INCOMPLETE
                await response_msgs[-1].edit(embed=embed)

            # If no tool uses, we're done
            if not tool_uses:
                break

            # Execute MCP tools
            logging.info(f"Executing {len(tool_uses)} MCP tools: {[t['name'] for t in tool_uses]}")

            # Build assistant message with text + tool_use blocks
            assistant_content = []
            if iteration_text:
                assistant_content.append({"type": "text", "text": iteration_text})
            for tool_use in tool_uses:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tool_use["id"],
                    "name": tool_use["name"],
                    "input": tool_use["input"],
                })

            conversation_messages.append({"role": "assistant", "content": assistant_content})

            # Call MCP tools and collect results
            tool_results = []
            for tool_use in tool_uses:
                result = await mcp_client.call_tool(tool_use["name"], tool_use["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": result,
                })

            conversation_messages.append({"role": "user", "content": tool_results})

        # Send plain response messages if needed
        if use_plain_responses and response_contents:
            for i, content in enumerate(response_contents):
                if i == 0:
                    content = f"-# {context_info}\n{content}"
                await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception as e:
        logging.exception("Error while generating Anthropic response")
        if not response_msgs:
            error_type = type(e).__name__
            error_brief = str(e).split("\n")[0][:200] if str(e) else "Unknown error"
            error_text = f"⚠️ **{error_type}**: {error_brief}"
            try:
                if use_plain_responses:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=error_text)))
                else:
                    error_embed = discord.Embed(description=error_text, color=EMBED_COLOR_INCOMPLETE)
                    await reply_helper(embed=error_embed, silent=True)
            except Exception:
                logging.exception("Error while sending error message")

    active_channels.discard(new_msg.channel.id)
    full_text = "".join(response_contents)
    return response_msgs, full_text


def _missing_config_error(field: str, hint: str) -> None:
    """Print a coloured, user-friendly error and exit."""
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    copy_cmd = "copy config-example.yaml config.yaml" if platform.system() == "Windows" else "cp config-example.yaml config.yaml"
    print(f"\n{RED}{'=' * 60}")
    print(f"  {BOLD}ERROR: '{field}' is not set in config.yaml!{RESET}{RED}")
    print(f"{'=' * 60}{RESET}\n")
    print(f"  {YELLOW}{hint}{RESET}\n")
    print(f"  {YELLOW}If you haven't already, duplicate the example config:{RESET}\n")
    print(f"    {CYAN}{copy_cmd}{RESET}\n")
    print(f"  {YELLOW}Then fill in the required values in {BOLD}config.yaml{RESET}{YELLOW}.{RESET}")
    print(f"{RED}{'=' * 60}{RESET}\n")
    raise SystemExit(1)


async def main() -> None:
    if not config.get("bot_token"):
        _missing_config_error("bot_token", "A Discord bot token is required. You can find it in the Bot tab\n  of the Discord Developer Portal (https://discord.com/developers).")
    if not config.get("client_id"):
        _missing_config_error("client_id", "A Discord application ID is required. You can find it under\n  General Information in the Discord Developer Portal\n  (https://discord.com/developers).")
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except discord.errors.PrivilegedIntentsRequired:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    print(f"\n{RED}{'=' * 60}")
    print(f"  {BOLD}ERROR: Privileged intents are not enabled!{RESET}{RED}")
    print(f"{'=' * 60}{RESET}\n")
    print(f"  {YELLOW}This bot requires the {BOLD}Message Content{RESET}{YELLOW} intent to be")
    print(f"  enabled in the Discord Developer Portal:{RESET}\n")
    print(f"  {CYAN}1.{RESET} {YELLOW}Go to {CYAN}https://discord.com/developers/applications/{RESET}")
    print(f"  {CYAN}2.{RESET} {YELLOW}Select your application{RESET}")
    print(f"  {CYAN}3.{RESET} {YELLOW}Navigate to the {BOLD}Bot{RESET}{YELLOW} tab{RESET}")
    print(f"  {CYAN}4.{RESET} {YELLOW}Under {BOLD}Privileged Gateway Intents{RESET}{YELLOW}, enable:")
    print(f"     {BOLD}{CYAN}Message Content Intent{RESET}\n")
    print(f"{RED}{'=' * 60}{RESET}\n")
except KeyboardInterrupt:
    pass
