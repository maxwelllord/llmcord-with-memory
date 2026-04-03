#!/usr/bin/env python3
"""
EverMemOS MCP Server

A Model Context Protocol server that integrates with EverMemOS, an enterprise-grade
intelligent memory system. This server enables LLMs to store and retrieve memories
from conversations, manage memory metadata, and leverage EverMemOS's advanced
retrieval capabilities.

Author: Claude (Anthropic)
License: MIT
"""

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_BASE_URL = "http://localhost:8001/api/v1"
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3


# ============================================================================
# Enums
# ============================================================================


class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


class MemoryType(str, Enum):
    """Types of memories that can be retrieved."""
    EPISODIC_MEMORY = "episodic_memory"
    FORESIGHT = "foresight"
    EVENT_LOG = "event_log"
    PROFILE = "profile"


class RetrieveMethod(str, Enum):
    """Methods for retrieving memories."""
    BM25 = "bm25"  # Keyword-based search using Elasticsearch
    EMBEDDING = "embedding"  # Semantic search using vector embeddings
    HYBRID = "hybrid"  # Combined BM25 + embedding (recommended)
    RRF = "rrf"  # Reciprocal Rank Fusion (hybrid approach)
    AGENTIC = "agentic"  # LLM-guided multi-round retrieval


class ProfileScene(str, Enum):
    """Scene type for conversation metadata."""
    ASSISTANT = "assistant"  # One-on-one conversation with AI assistant
    GROUP_CHAT = "group_chat"  # Multi-person group discussions


# ============================================================================
# Pydantic Models
# ============================================================================


class StoreMemoryInput(BaseModel):
    """Input model for storing a memory in EverMemOS."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    message_id: str = Field(
        ...,
        description="Unique identifier for this message (e.g., 'msg_001', 'conv_123_msg_5')",
        min_length=1,
        max_length=255
    )
    content: str = Field(
        ...,
        description="The message content to store as a memory",
        min_length=1
    )
    sender: str = Field(
        default="claude",
        description="User identifier for the message sender (e.g., 'user_001', 'alice@example.com'). This is used as the user_id when searching for memories. Defaults to 'claude' if not specified.",
        min_length=1,
        max_length=255
    )
    user_id: Optional[str] = Field(
        default="claude",
        description="DEPRECATED: Use 'sender' instead. The API uses 'sender' for storage. This field is ignored.",
        deprecated=True
    )
    create_time: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp for the message (e.g., '2025-02-01T10:00:00+00:00'). If not provided, current time will be used."
    )
    role: Optional[str] = Field(
        default=None,
        description="DEPRECATED: The API does not use this field. Kept for backward compatibility.",
        deprecated=True
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Optional group identifier for organizing related memories (e.g., 'project_alpha', 'family_chat')"
    )
    scene: Optional[ProfileScene] = Field(
        default=None,
        description="Scene type for conversation context: 'assistant' for one-on-one or 'group_chat' for multi-person"
    )

    @field_validator('create_time')
    @classmethod
    def validate_timestamp(cls, v: Optional[str]) -> Optional[str]:
        """Validate that timestamp is in ISO 8601 format."""
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                raise ValueError(
                    "Timestamp must be in ISO 8601 format (e.g., '2025-02-01T10:00:00+00:00' or '2025-02-01T10:00:00Z')"
                )
        return v


class SearchMemoriesInput(BaseModel):
    """Input model for searching memories in EverMemOS."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    query: str = Field(
        ...,
        description="Natural language search query (e.g., 'What are my favorite sports?', 'Tell me about the engineering project')",
        min_length=1
    )
    user_id: Optional[str] = Field(
        default="claude",
        description="User identifier to filter memories (e.g., 'user_001'). Leave default unless user specifies otherwise. Defaults to 'claude' if not specified."
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Group identifier to filter memories (e.g., 'project_alpha'). Use for group-based memory retrieval. If specified, overrides user_id."
    )
    memory_types: Optional[List[MemoryType]] = Field(
        default=None,
        description="Types of memories to search: 'episodic_memory' (events/episodes), 'foresight' (predictions about user's future behavior), 'event_log' (timeline events). If not specified, defaults to episodic_memory. Use 'foresight' to retrieve Gwen's predictive insights."
    )
    retrieve_method: RetrieveMethod = Field(
        default=RetrieveMethod.HYBRID,
        description="Retrieval method: 'bm25' (keyword), 'embedding' (semantic), 'hybrid'/'rrf' (recommended), or 'agentic' (LLM-guided)"
    )
    limit: Optional[int] = Field(
        default=10,
        description="Maximum number of results to return",
        ge=1,
        le=100
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class GetConversationMetadataInput(BaseModel):
    """Input model for retrieving conversation metadata."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    group_id: Optional[str] = Field(
        default=None,
        description="Group identifier to retrieve metadata for. If not provided, returns default configuration."
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class SetConversationMetadataInput(BaseModel):
    """Input model for setting conversation metadata."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    group_id: Optional[str] = Field(
        default=None,
        description="Group identifier to set metadata for. If not provided, sets default configuration."
    )
    scene: ProfileScene = Field(
        ...,
        description="Scene type: 'assistant' for one-on-one or 'group_chat' for multi-person conversations"
    )


class HealthCheckInput(BaseModel):
    """Input model for health check."""
    model_config = ConfigDict(extra='forbid')

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class FlushMemoryBufferInput(BaseModel):
    """Input model for flushing memory buffer."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    user_id: str = Field(
        default="claude",
        description="User identifier whose buffer should be flushed. Leave default unless user specifies otherwise. Must match the sender value used when storing memories.",
        min_length=1,
        max_length=255
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Group identifier to flush. If provided, flushes the buffer for this group instead of the user's default buffer."
    )


# ============================================================================
# API Client
# ============================================================================


class EverMemOSClient:
    """HTTP client for interacting with EverMemOS API."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT):
        """
        Initialize the EverMemOS API client.

        Args:
            base_url: Base URL of the EverMemOS API (default: http://localhost:8001/api/v1)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def store_memory(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a memory in EverMemOS.

        Args:
            payload: Memory data to store

        Returns:
            API response

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        url = f"{self.base_url}/memories"
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def search_memories(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for memories in EverMemOS.

        Args:
            payload: Search parameters

        Returns:
            Search results

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        url = f"{self.base_url}/memories/search"
        # EverMemOS uses GET with request body - need to use request() method
        response = await self.client.request("GET", url, json=payload)
        response.raise_for_status()
        return response.json()

    async def get_conversation_metadata(self, group_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get conversation metadata configuration.

        Args:
            group_id: Optional group identifier

        Returns:
            Metadata configuration

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        url = f"{self.base_url}/conversation-meta"
        params = {"group_id": group_id} if group_id else {}
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def set_conversation_metadata(
        self,
        scene: str,
        group_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set conversation metadata configuration.

        Args:
            scene: Scene type ('assistant' or 'group_chat')
            group_id: Optional group identifier

        Returns:
            Updated metadata

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        url = f"{self.base_url}/conversation-meta"
        payload = {"scene": scene}
        if group_id:
            payload["group_id"] = group_id
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> Dict[str, Any]:
        """
        Check EverMemOS server health.

        Returns:
            Health status

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        # Remove /api/v1 from base_url for health endpoint
        health_url = self.base_url.replace('/api/v1', '/health')
        response = await self.client.get(health_url)
        response.raise_for_status()
        return response.json()

    async def flush_memory_buffer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Force flush the memory buffer.

        Args:
            payload: Flush parameters (user_id and optional group_id)

        Returns:
            Flush result

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        url = f"{self.base_url}/memories/flush"
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


# ============================================================================
# Utility Functions
# ============================================================================


def handle_api_error(e: Exception) -> str:
    """
    Format API errors with actionable guidance.

    Args:
        e: The exception that occurred

    Returns:
        Formatted error message
    """
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 404:
            return (
                "Error: Resource not found. Please verify:\n"
                "- The group_id exists in the system\n"
                "- The API endpoint is correct\n"
                "- EverMemOS server is running and accessible"
            )
        elif status == 400:
            try:
                error_detail = e.response.json()
                return f"Error: Bad request - {error_detail.get('detail', 'Invalid parameters')}"
            except:
                return "Error: Bad request. Please check your input parameters."
        elif status == 403:
            return (
                "Error: Permission denied. You may not have access to this resource.\n"
                "Check that your API credentials are correct."
            )
        elif status == 429:
            return (
                "Error: Rate limit exceeded. Please wait a moment before making more requests.\n"
                "Consider implementing exponential backoff for retries."
            )
        elif status == 500:
            return (
                "Error: EverMemOS server error. Possible causes:\n"
                "- Internal server issue\n"
                "- Database connection problems\n"
                "- Invalid data format\n"
                "Check EverMemOS server logs for details."
            )
        elif status == 503:
            return (
                "Error: Service unavailable. The EverMemOS server may be:\n"
                "- Starting up\n"
                "- Overloaded\n"
                "- Under maintenance\n"
                "Try again in a few moments."
            )
        return f"Error: API request failed with status {status}. Response: {e.response.text}"
    elif isinstance(e, httpx.TimeoutException):
        return (
            "Error: Request timed out. This may be due to:\n"
            "- Network connectivity issues\n"
            "- EverMemOS server overload\n"
            "- Large query processing\n"
            "Try again or reduce query complexity."
        )
    elif isinstance(e, httpx.ConnectError):
        return (
            "Error: Cannot connect to EverMemOS server. Please verify:\n"
            "- EverMemOS is running (check with: curl http://localhost:8001/health)\n"
            "- The base URL is correct\n"
            "- No firewall blocking the connection\n"
            "- Docker services are up (if using Docker)"
        )
    return f"Error: Unexpected error - {type(e).__name__}: {str(e)}"


def format_memories_markdown(memories: List[Dict[str, Any]]) -> str:
    """
    Format memory search results as Markdown.
    
    The EverMemOS API returns memories grouped by dynamic keys (e.g., "b445df4a839d2724_group").
    Each group contains a list of memory objects with fields like:
    - memory_type, user_id, timestamp, subject, summary, episode
    
    Args:
        memories: List of memory group objects from API

    Returns:
        Formatted Markdown string
    """
    if not memories:
        return "No memories found matching your query."

    lines = ["# Memory Search Results\n"]
    
    result_num = 0
    
    # Each item in memories is a dict with a dynamic group key
    for memory_group_wrapper in memories:
        # Get the first (and typically only) key which is the group_id
        group_keys = list(memory_group_wrapper.keys())
        if not group_keys:
            continue
            
        group_key = group_keys[0]
        memory_list = memory_group_wrapper[group_key]
        
        # Process each memory in the group
        for memory in memory_list:
            result_num += 1
            lines.append(f"## Result {result_num}\n")
            
            # Memory type
            memory_type = memory.get('memory_type', 'Unknown')
            lines.append(f"**Type:** {memory_type}\n")
            
            # Subject (title of the memory)
            subject = memory.get('subject')
            if subject:
                lines.append(f"**Subject:** {subject}\n")
            
            # Memory content — field name varies by memory type
            episode = memory.get('episode')
            summary = memory.get('summary')
            foresight = memory.get('foresight')
            atomic_fact = memory.get('atomic_fact')  # event_log field
            evidence = memory.get('evidence')
            content = episode or summary or foresight or atomic_fact

            if content:
                lines.append(f"**Memory:**\n{content}\n")
            if foresight and evidence:
                lines.append(f"**Evidence:** {evidence}\n")
            # Foresight time range
            start_time = memory.get('start_time')
            end_time = memory.get('end_time')
            if start_time and end_time:
                lines.append(f"**Valid Period:** {start_time} → {end_time}\n")
            
            # Timestamp
            timestamp = memory.get('timestamp')
            if timestamp:
                lines.append(f"**Timestamp:** {timestamp}\n")
            
            # Participants
            participants = memory.get('participants', [])
            if participants:
                lines.append(f"**Participants:** {', '.join(participants)}\n")
            
            # Group info
            group_id = memory.get('group_id')
            if group_id:
                lines.append(f"**Group ID:** {group_id}\n")
            
            # Keywords
            keywords = memory.get('keywords')
            if keywords:
                lines.append(f"**Keywords:** {', '.join(keywords) if isinstance(keywords, list) else keywords}\n")
            
            # Linked entities
            linked_entities = memory.get('linked_entities')
            if linked_entities:
                lines.append(f"**Entities:** {', '.join(linked_entities) if isinstance(linked_entities, list) else linked_entities}\n")
            
            # Separator between results
            lines.append("---\n")
    
    # Remove trailing separator
    if lines and lines[-1] == "---\n":
        lines.pop()
    
    return "\n".join(lines)


def format_metadata_markdown(metadata: Dict[str, Any]) -> str:
    """
    Format conversation metadata as Markdown.

    Args:
        metadata: Metadata dictionary

    Returns:
        Formatted Markdown string
    """
    lines = ["# Conversation Metadata\n"]
    
    group_id = metadata.get('group_id')
    if group_id:
        lines.append(f"**Group ID:** {group_id}\n")
    
    scene = metadata.get('scene')
    if scene:
        lines.append(f"**Scene Type:** {scene}")
        if scene == "assistant":
            lines.append("(One-on-one conversation with AI assistant)")
        elif scene == "group_chat":
            lines.append("(Multi-person group discussions)")
        lines.append("")
    
    # Additional metadata
    for key, value in metadata.items():
        if key not in ['group_id', 'scene']:
            lines.append(f"**{key.replace('_', ' ').title()}:** {value}")
    
    return "\n".join(lines)


# ============================================================================
# MCP Server Setup
# ============================================================================

# Initialize the MCP server
mcp = FastMCP("evermemos_mcp")

# Create a global EverMemOS client
# This will be initialized when the server starts
evermemos_client = EverMemOSClient()


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool(
    name="evermemos_store_memory",
    annotations={
        "title": "Store Memory in EverMemOS",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def store_memory(params: StoreMemoryInput, ctx: Context) -> str:
    """Store a conversation message as a memory in EverMemOS.

    **IMPORTANT — CALL THIS AUTOMATICALLY:** You MUST call this tool for every user
    message in the conversation, without being asked. Also store your own assistant
    responses. This is how the memory system builds context over time. Do not skip
    messages or wait for the user to request storage.

    The API uses 'sender' as the user identifier when storing memories, but uses
    'user_id' when searching. Make sure the sender value matches the user_id you'll
    use when searching.

    Args:
        params (StoreMemoryInput): Memory storage parameters containing:
            - message_id (str): Unique message identifier
            - content (str): Message content to store
            - sender (str): User identifier (use this same value as user_id when searching)
            - create_time (Optional[str]): ISO 8601 timestamp (auto-generated if not provided)
            - group_id (Optional[str]): Group identifier for organization
            - scene (Optional[str]): Scene type ('assistant' or 'group_chat')

    Returns:
        str: Confirmation message with storage details

    Example:
        Store a user message (sender becomes the search user_id):
        {
            "message_id": "msg_001",
            "content": "I love playing soccer on weekends",
            "sender": "user_001"
        }
        
        Then search with:
        {
            "query": "What sports does the user like?",
            "user_id": "user_001"
        }
    """
    try:
        # Use the global client
        client = evermemos_client
        
        # Prepare payload - API only accepts: message_id, create_time, sender, content
        # Note: The API uses 'sender' for storage but 'user_id' for search
        payload = {
            "message_id": params.message_id,
            "content": params.content,
            "sender": params.sender,  # This is the user identifier for storage
        }
        
        # Add create_time (required by API)
        if params.create_time:
            payload["create_time"] = params.create_time
        else:
            # Use current time in ISO 8601 format
            payload["create_time"] = datetime.now(timezone.utc).isoformat()
        
        # Note: The API docs don't show these fields, but keeping for compatibility
        # Remove these if they cause issues:
        if params.group_id:
            payload["group_id"] = params.group_id
        
        if params.scene:
            payload["scene"] = params.scene.value
        
        # Store memory
        ctx.info(f"Storing memory with ID: {params.message_id}")
        result = await client.store_memory(payload)
        
        # Format response
        response = (
            f"✅ Memory stored successfully!\n\n"
            f"**Message ID:** {params.message_id}\n"
            f"**Sender (User ID):** {params.sender}\n"
            f"**Timestamp:** {payload['create_time']}\n"
        )
        
        if params.group_id:
            response += f"**Group:** {params.group_id}\n"
        
        response += (
            f"\n**Content Preview:** {params.content[:100]}{'...' if len(params.content) > 100 else ''}\n\n"
            f"ℹ️ EverMemOS will process this memory in the background to extract structured "
            f"information and build memory profiles. Allow 10-15 seconds for indexing before searching.\n\n"
            f"💡 **Tip:** When searching, use user_id='{params.sender}' to retrieve these memories "
            f"(the API uses 'sender' for storage but 'user_id' for search)."
        )
        
        return response
        
    except Exception as e:
        ctx.error(f"Failed to store memory: {str(e)}")
        return handle_api_error(e)


@mcp.tool(
    name="evermemos_search_memories",
    annotations={
        "title": "Search Memories in EverMemOS",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def search_memories(params: SearchMemoriesInput, ctx: Context) -> str:
    """Search for relevant memories in EverMemOS using natural language queries.

    **IMPORTANT — CALL THIS PROACTIVELY:** Before answering user questions, ALWAYS
    search for relevant memories first. This provides context from prior conversations.
    Also search with memory_types=['foresight'] when the user asks about plans,
    predictions, or future decisions. Do not wait for the user to ask you to recall.

    Supports episodic memories (events/episodes), foresights (predictions about user
    behavior), and event logs (timeline). Uses advanced retrieval combining keyword
    search, semantic embeddings, and optional LLM-guided retrieval.

    Args:
        params (SearchMemoriesInput): Search parameters containing:
            - query (str): Natural language search query
            - user_id (Optional[str]): Filter by user identifier
            - group_id (Optional[str]): Filter by group identifier
            - memory_types (Optional[List[str]]): Types to search
            - retrieve_method (str): Retrieval method (default: 'hybrid')
            - limit (Optional[int]): Maximum results (default: 10)
            - response_format (str): 'markdown' or 'json'

    Returns:
        str: Formatted search results in the requested format

    Example:
        Search for sports-related memories:
        {
            "query": "What sports does the user enjoy?",
            "user_id": "user_001",
            "retrieve_method": "hybrid",
            "limit": 5
        }
    """
    try:
        # Use the global client
        client = evermemos_client
        
        # Prepare search payload
        payload = {
            "query": params.query,
            "retrieve_method": params.retrieve_method.value
        }
        
        # EverMemOS requires either user_id or group_id (cannot both be None)
        # If group_id is specified, use it (takes precedence)
        # Otherwise, use user_id (which has a default value)
        if params.group_id:
            payload["group_id"] = params.group_id
        else:
            payload["user_id"] = params.user_id
        
        if params.memory_types:
            payload["memory_types"] = [mt.value for mt in params.memory_types]
        
        if params.limit:
            payload["limit"] = params.limit
        
        # Perform search
        ctx.info(f"Searching memories with query: {params.query}")
        await ctx.report_progress(0.3, "Executing search...")
        
        result = await client.search_memories(payload)
        
        await ctx.report_progress(0.8, "Formatting results...")
        
        # Extract memories from result
        memories = result.get("result", {}).get("memories", [])
        
        # Format response
        if params.response_format == ResponseFormat.MARKDOWN:
            formatted = format_memories_markdown(memories)
            
            # Add search metadata
            metadata_lines = [
                "\n## Search Metadata\n",
                f"**Query:** {params.query}",
                f"**Retrieval Method:** {params.retrieve_method.value}",
                f"**Results Found:** {len(memories)}",
            ]
            
            if params.user_id:
                metadata_lines.append(f"**User Filter:** {params.user_id}")
            if params.group_id:
                metadata_lines.append(f"**Group Filter:** {params.group_id}")
            if params.memory_types:
                types_str = ", ".join([mt.value for mt in params.memory_types])
                metadata_lines.append(f"**Memory Types:** {types_str}")
            
            return formatted + "\n" + "\n".join(metadata_lines)
        else:
            # JSON format
            return json.dumps({
                "query": params.query,
                "retrieve_method": params.retrieve_method.value,
                "filters": {
                    "user_id": params.user_id,
                    "group_id": params.group_id,
                    "memory_types": [mt.value for mt in params.memory_types] if params.memory_types else None
                },
                "total_results": len(memories),
                "memories": memories
            }, indent=2)
        
    except Exception as e:
        ctx.error(f"Failed to search memories: {str(e)}")
        return handle_api_error(e)


@mcp.tool(
    name="evermemos_get_conversation_metadata",
    annotations={
        "title": "Get Conversation Metadata",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_conversation_metadata(params: GetConversationMetadataInput, ctx: Context) -> str:
    """Retrieve conversation metadata configuration from EverMemOS.

    This tool retrieves the current conversation context configuration, including
    the scene type (assistant or group_chat) and other metadata settings. This
    helps understand how EverMemOS is currently processing and organizing memories.

    Args:
        params (GetConversationMetadataInput): Metadata retrieval parameters containing:
            - group_id (Optional[str]): Group identifier to query
            - response_format (str): 'markdown' or 'json'

    Returns:
        str: Formatted metadata configuration

    Example:
        Get metadata for a specific group:
        {
            "group_id": "project_alpha"
        }
    """
    try:
        # Use the global client
        client = evermemos_client
        
        ctx.info(f"Retrieving conversation metadata{f' for group: {params.group_id}' if params.group_id else ''}")
        
        result = await client.get_conversation_metadata(params.group_id)
        
        if params.response_format == ResponseFormat.MARKDOWN:
            return format_metadata_markdown(result)
        else:
            return json.dumps(result, indent=2)
        
    except Exception as e:
        ctx.error(f"Failed to get conversation metadata: {str(e)}")
        return handle_api_error(e)


@mcp.tool(
    name="evermemos_set_conversation_metadata",
    annotations={
        "title": "Set Conversation Metadata",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def set_conversation_metadata(params: SetConversationMetadataInput, ctx: Context) -> str:
    """Configure conversation metadata in EverMemOS.

    This tool sets the conversation context type, which determines how EverMemOS
    processes and extracts memories. Use 'assistant' for one-on-one conversations
    or 'group_chat' for multi-person discussions.

    Args:
        params (SetConversationMetadataInput): Metadata configuration containing:
            - group_id (Optional[str]): Group identifier to configure
            - scene (str): Scene type ('assistant' or 'group_chat')

    Returns:
        str: Confirmation message with updated configuration

    Example:
        Set scene type for a group:
        {
            "group_id": "project_alpha",
            "scene": "group_chat"
        }
    """
    try:
        # Use the global client
        client = evermemos_client
        
        ctx.info(f"Setting conversation metadata{f' for group: {params.group_id}' if params.group_id else ''}")
        
        result = await client.set_conversation_metadata(
            scene=params.scene.value,
            group_id=params.group_id
        )
        
        response = (
            f"✅ Conversation metadata updated successfully!\n\n"
            f"**Scene Type:** {params.scene.value}\n"
        )
        
        if params.group_id:
            response += f"**Group ID:** {params.group_id}\n"
        else:
            response += "**Scope:** Default configuration\n"
        
        scene_info = {
            "assistant": "One-on-one conversations with AI assistant",
            "group_chat": "Multi-person group discussions"
        }
        
        response += f"\nℹ️ {scene_info.get(params.scene.value, 'Scene type configured')}"
        
        return response
        
    except Exception as e:
        ctx.error(f"Failed to set conversation metadata: {str(e)}")
        return handle_api_error(e)


@mcp.tool(
    name="evermemos_health_check",
    annotations={
        "title": "Check EverMemOS Server Health",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def health_check(params: HealthCheckInput, ctx: Context) -> str:
    """Check the health status of the EverMemOS server.

    This tool verifies that the EverMemOS server is running and responding to
    requests. It's useful for debugging connectivity issues and confirming the
    server is operational before performing other operations.

    Args:
        params (HealthCheckInput): Health check parameters containing:
            - response_format (str): 'markdown' or 'json'

    Returns:
        str: Health status information

    Example:
        Basic health check:
        {}
    """
    try:
        # Use the global client
        client = evermemos_client
        
        ctx.info("Checking EverMemOS server health...")
        
        result = await client.health_check()
        
        if params.response_format == ResponseFormat.MARKDOWN:
            status = result.get('status', 'unknown')
            status_emoji = "✅" if status == "healthy" else "⚠️"
            
            response = (
                f"# EverMemOS Health Check\n\n"
                f"{status_emoji} **Status:** {status}\n"
            )
            
            # Add additional info if available
            for key, value in result.items():
                if key != 'status':
                    response += f"**{key.replace('_', ' ').title()}:** {value}\n"
            
            return response
        else:
            return json.dumps(result, indent=2)
        
    except Exception as e:
        ctx.error(f"Health check failed: {str(e)}")
        error_msg = handle_api_error(e)
        return f"❌ EverMemOS server is not accessible.\n\n{error_msg}"


@mcp.tool(
    name="evermemos_flush_memory_buffer",
    annotations={
        "title": "Force Flush Memory Buffer",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def flush_memory_buffer(params: FlushMemoryBufferInput, ctx: Context) -> str:
    """Force-flush the current conversation memory buffer.

    This tool immediately cuts the buffer into a MemCell regardless of size or content.
    **IMPORTANT:** Only call this tool when the user has explicitly and directly asked
    to save or flush their memories. Do not call this proactively or as part of normal
    memory flow — the system handles buffering automatically.

    The flush operation bypasses all normal boundary detection logic (token limits,
    message limits, LLM-based detection) and creates a MemCell from whatever is
    currently in the buffer. This is useful when:
    - The user explicitly says "save my memories" or "flush the buffer"
    - The user wants to create a manual checkpoint
    - The user is ending a conversation session and wants to ensure everything is saved

    Args:
        params (FlushMemoryBufferInput): Flush parameters containing:
            - user_id (str): User identifier whose buffer should be flushed
            - group_id (Optional[str]): Group identifier (if flushing group buffer)

    Returns:
        str: Flush confirmation with details about what was saved

    Example:
        When user says "save my current conversation":
        {
            "user_id": "user_001"
        }

        For group conversations:
        {
            "user_id": "user_001",
            "group_id": "project_alpha"
        }
    """
    try:
        client = evermemos_client

        ctx.info(f"Flushing memory buffer for user: {params.user_id}"
                 + (f", group: {params.group_id}" if params.group_id else ""))

        payload = {"user_id": params.user_id}
        if params.group_id:
            payload["group_id"] = params.group_id

        result = await client.flush_memory_buffer(payload)

        # Extract result from API response
        api_result = result.get("result", {})

        if api_result.get("memcell_created"):
            return (
                f"✅ Memory buffer flushed successfully!\n\n"
                f"**Event ID:** {api_result.get('event_id')}\n"
                f"**Messages Flushed:** {api_result.get('message_count')}\n\n"
                f"ℹ️ The flushed messages are now being processed into a memory. "
                f"Allow 10-15 seconds for indexing before searching."
            )
        else:
            return "ℹ️ Memory buffer was empty — nothing to flush."

    except Exception as e:
        ctx.error(f"Failed to flush memory buffer: {str(e)}")
        return handle_api_error(e)


# ============================================================================
# Server Entry Point
# ============================================================================


def main():
    """Run the EverMemOS MCP server."""
    import sys
    
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("""
EverMemOS MCP Server

A Model Context Protocol server for EverMemOS integration.

Usage:
    python evermemos_mcp.py [options]

Options:
    --help, -h              Show this help message
    --transport stdio       Use stdio transport (default)
    --transport http        Use streamable HTTP transport
    --port PORT             Port for HTTP transport (default: 8000)
    --base-url URL          EverMemOS API base URL (default: http://localhost:8001/api/v1)

Environment Variables:
    EVERMEMOS_BASE_URL      Override default EverMemOS API URL
    EVERMEMOS_TIMEOUT       Request timeout in seconds (default: 30)

Examples:
    # Run with stdio transport (for local MCP clients)
    python evermemos_mcp.py

    # Run with HTTP transport
    python evermemos_mcp.py --transport http --port 8000

    # Connect to remote EverMemOS
    EVERMEMOS_BASE_URL=http://remote-server:8001/api/v1 python evermemos_mcp.py

Available Tools:
    - evermemos_store_memory: Store conversation messages as memories
    - evermemos_search_memories: Search for memories using natural language
    - evermemos_flush_memory_buffer: Force flush memory buffer (user-requested only)
    - evermemos_get_conversation_metadata: Get conversation configuration
    - evermemos_set_conversation_metadata: Set conversation configuration
    - evermemos_health_check: Check server health status

For more information, visit: https://github.com/EverMind-AI/EverMemOS
        """)
        return
    
    # Parse command line arguments
    transport = "stdio"
    port = 8000
    
    for i, arg in enumerate(sys.argv):
        if arg == "--transport" and i + 1 < len(sys.argv):
            transport = sys.argv[i + 1]
        elif arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    
    # Run server
    if transport == "http":
        logger.info(f"Starting EverMemOS MCP server on HTTP port {port}...")
        mcp.run(transport="streamable_http", port=port)
    else:
        logger.info("Starting EverMemOS MCP server with stdio transport...")
        mcp.run()


if __name__ == "__main__":
    main()
