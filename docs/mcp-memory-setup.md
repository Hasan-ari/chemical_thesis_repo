# MCP Memory Setup

This project uses two memory layers:

- `docs/learning-log/`: human-readable daily Markdown notes.
- `.mcp-memory/memory.jsonl`: private MCP Memory storage for agent-readable facts.

Do not commit `.mcp-memory/`. It is ignored by git.

## Why This Split

Markdown notes are for learning and review. They should explain what we learned,
what changed, and what we will do next.

MCP Memory is for compact facts the agent should remember, such as:

- the user prefers ELI10 analogies before formulas;
- the current model target is `conditions + time -> full trajectory`;
- Colab should copy zipped data to `/content` before training.

MCP Memory does not automatically read Markdown files. It stores memory in a
JSONL file selected by `MEMORY_FILE_PATH`.

## Codex Config

Codex MCP servers are configured in `~/.codex/config.toml`, not inside this repo.
Add this block there and restart Codex:

```toml
[mcp_servers.memory]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-memory"]

[mcp_servers.memory.env]
MEMORY_FILE_PATH = "/Users/macbook/chemical_thesis/chemical_thesis_repo/.mcp-memory/memory.jsonl"
```

The first run needs network access because `npx` downloads
`@modelcontextprotocol/server-memory`.

## JSON MCP Clients

Some MCP clients use JSON config instead of Codex TOML. For those clients:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {
        "MEMORY_FILE_PATH": "/Users/macbook/chemical_thesis/chemical_thesis_repo/.mcp-memory/memory.jsonl"
      }
    }
  }
}
```

## First Memory Facts To Store

After the MCP server is connected, store these facts:

- The user wants simple English and ELI10 analogies while coding.
- The thesis pipeline target is `Input.txt conditions + time_d -> full Output.txt trajectory`.
- The older model used sliding windows over real output rows.
- The new pipeline should be Colab-friendly, config-driven, and tracked with SQLite plus CSV.
