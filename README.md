<h1>
LLMcord with memory
</h1>

<h3>
A discord bot with a simple but effective memory system.
</h3>

![Example user interaction](https://github.com/user-attachments/assets/eb8d1712-6d67-4450-a029-4c6826f450a7)

## Features

**Self-interjection** allows LLMs to lean into the conversation more naturally.

**Automatic memory retrieval** injects stored memories into the chat when they're semantically relevant.

**Sliding chat context** set a token budget and the context window automatically adjusts.

## How the Memory System Works

The bot has a two-tiered memory system.

### Core Memory vs Semantic Memory

- **Core memory** (`core_memory.md`) This is always loaded. Holds identity-level facts — who the bot is, key relationships, and permanent details.
- **Semantic memories** (`memory.json`)  These are retrieved dynamically when a message is sent to chat. Individual facts, preferences, events, and context stored with embeddings.

### Automatic Memory Sweeps

The bot tracks **sessions** based on a configurable inactivity gap (default 2 hours). When a new message comes in after 2 hours of silence, the bot automatically runs a **memory sweep** on the previous session:

1. It collects all messages from the previous session.
2. It sends them to the LLM along with all existing memories.
3. The LLM decides what to **add**, **update**, or **delete** — keeping the memory store lean and accurate.
4. Core memory is rewritten to stay under 15 lines, and stale semantic memories are pruned.
5. Deleted memories are flagged as deleted in memory.json. You can restore these memories by

You can also trigger a sweep manually at any time with the `/sweep` command.

### Memory Retrieval


![Example retrieval](https://github.com/user-attachments/assets/12ed3f59-770e-4b13-83ca-7312c85b2afb)

When you send a message, the bot embeds your message and compares it against stored semantic memories. The top 3 most relevant results (above a similarity threshold of 0.4) are injected into the conversation so the LLM can use them naturally. Already-recalled memories won't be injected again in the same session.

When memories are recalled these are displayed in chat, along with their relevance score.

### Memory Viewing

You can view memories in a list, or a semantic map, by opening 'memory_viewer.html'. Just drag and drop memory.json in.

![Memory viewer](https://github.com/user-attachments/assets/897437f5-1ba8-489f-8d33-51984c3a5fdf)

### Setup

See below.

## Slash Commands

| Command | Description |
| --- | --- |
| **/model** | View or switch the current LLM model. Shows the active model and provides autocomplete for all configured models. Switching requires admin permissions. |
| **/sweep** | Manually trigger a memory sweep on all messages since the last sweep. Useful if you want the bot to process and remember the current conversation without waiting for the automatic 2-hour session gap. Requires admin or sweep permissions. |
| **/info** | Shows a context estimate for the current channel — how many messages are in the sliding window, approximate token count, and how far back the context reaches. |

## How to use

1. Clone the repo:
   ```bash
   git clone https://github.com/jakobdylanc/llmcord
   cd llmcord
   ```

2. Create a copy of "config-example.yaml" named "config.yaml" and set it up:

### Discord Bot Configuration

3. Go to the [Discord Developer Portal](https://discord.com/developers/applications). Create a 'New Application'. The name of the application does not matter.

4. Within your new application: Overview -> General Information, Copy your application's 'Application ID' and set it as the 'client id' in config.yaml.

5. Now within Overview -> Bot:
   - Click 'Reset Token'. Copy the new token and set it as the 'bot_token' in config.yaml.
   - Change the Bot's display name to whatever you would like to call the bot. This gets used as {bot_name} in prompts in config.yaml.
   - Enable 'Message Content Intent'.

### Model Configuration

7. Copy your API Key from your LLM provider into config.yaml

### Memory and Interjection (OPTIONAL)

8. The default config uses [OpenRouter](https://openrouter.ai/) for interjection and embedding — just paste your OpenRouter API key into the providers section and memory + interjection will work out of the box.

### Run

   **No Docker:**
   ```bash
   python -m pip install -U -r requirements.txt
   python llmcord.py
   ```

   **With Docker:**
   ```bash
   docker compose up
   ```

### Invite the Bot (First Launch Only)

9. Use the link from the terminal to invite the bot to the desired servers. (Ctrl+Click or copy and paste into your browser)

![Bot link](https://github.com/user-attachments/assets/c56e492b-d0b6-494c-87be-eb63aa314ea6)

### Discord settings:

| Setting | Description |
| --- | --- |
| **bot_token** | Create a new Discord bot at [discord.com/developers/applications](https://discord.com/developers/applications) and generate a token under the "Bot" tab. Also enable "MESSAGE CONTENT INTENT". |
| **client_id** | This is your Application ID from Overview -> General Information. OR Client ID from Overview -> OAuth2. They're the same ID. |
| **status_message** | Set a custom message that displays on the bot's Discord profile.<br /><br />**Max 128 characters.** |
| **max_text** | The maximum amount of text allowed in a single message, including text from file attachments.<br /><br />Default: `100,000` |
| **max_images** | The maximum number of image attachments allowed in a single message.<br /><br />Default: `5`<br /><br />**Only applicable when using a vision model.** |
| **max_messages** | The maximum number of messages allowed in a reply chain. When exceeded, the oldest messages are dropped.<br /><br />Default: `25` |
| **use_plain_responses** | When set to `true` the bot will use plaintext responses instead of embeds. Plaintext responses have a shorter character limit so the bot's messages may split more often.<br /><br />Default: `false`<br /><br />**Also disables streamed responses and warning messages.** |
| **allow_dms** | Set to `false` to disable direct message access.<br /><br />Default: `true` |
| **permissions** | Configure access permissions for `users`, `roles` and `channels`, each with a list of `allowed_ids` and `blocked_ids`.<br /><br />Control which `users` are admins with `admin_ids`. Admins can change the model with `/model` and DM the bot even if `allow_dms` is `false`.<br /><br />**Leave `allowed_ids` empty to allow ALL in that category.**<br /><br />**Role and channel permissions do not affect DMs.**<br /><br />**You can use [category](https://support.discord.com/hc/en-us/articles/115001580171-Channel-Categories-101) IDs to control channel permissions in groups.** |

### LLM settings:

| Setting | Description |
| --- | --- |
| **providers** | Add the LLM providers you want to use, each with a `base_url` and optional `api_key` entry. Popular providers (`openai`, `openrouter`, `ollama`, etc.) are already included.<br /><br />**Only supports OpenAI compatible APIs.**<br /><br />**Some providers may need `extra_headers` / `extra_query` / `extra_body` entries for extra HTTP data. See the included `azure-openai` provider for an example.** |
| **models** | Add the models you want to use in `<provider>/<model>: <parameters>` format (examples are included). When you run `/model` these models will show up as autocomplete suggestions.<br /><br />**Refer to each provider's documentation for supported parameters.**<br /><br />**The first model in your `models` list will be the default model at startup.**<br /><br />**Some vision models may need `:vision` added to the end of their name to enable image support.** |
| **interjection_model** | A small model used to decide if the bot should respond without being @mentioned. Uses `provider/model` format.<br /><br />**Omit or leave blank to only respond when @mentioned (default).** |
| **embedding_model** | The model used for semantic memory retrieval. Uses `provider/model` format.<br /><br />**Omit or leave blank to disable semantic memory (only core memory will be used).** |
| **system_prompt** | Write anything you want to customize the bot's behavior!<br /><br />**Leave blank for no system prompt.**<br /><br />**You can use the `{date}` and `{time}` tags in your system prompt to insert the current date and time, based on your host computer's time zone.**<br /><br />**It is recommended to include something like `"User messages are prefixed with their Discord ID as <@ID>. Use this format to mention users."` in your system prompt to help the bot understand the user message format.** |
| **sweep_prompt** | The prompt sent to the LLM during memory sweeps. Configurable primarily to adjust personality or core memory length.<br /><br />**A sensible default is included — most users won't need to change this.** |

### Context settings:

| Setting | Description |
| --- | --- |
| **context_gap_minutes** | Minutes of inactivity before a context gap is inserted. Messages before the gap are loaded as "bridge" context.<br /><br />Default: `120` |
| **max_context_tokens** | Token budget for recent messages in the sliding context window.<br /><br />Default: `10,000` |
| **context_bridge_tokens** | Token budget for older messages loaded before a context gap, giving the bot some history from before the silence.<br /><br />Default: `1,000` |

## Debugging

### Turn Logger (`turn_logger.py`)

Every LLM call (both chat responses and memory sweeps) is logged to the `logs/` directory, organised by date. Each log file captures the full system prompt, message history, and model response for that turn so you can see exactly what models see and said each call.

### Endpoint Tester (`test_endpoints.py`)

A quick sanity check that your configured model endpoints are reachable. Run it with:

```bash
python test_endpoints.py
```

It sends a minimal request to each configured model (main and interjection) and reports whether the endpoint responded successfully.
