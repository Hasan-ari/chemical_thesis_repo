# MCP Memory Setup

This project uses two memory layers:

- `docs/learning-log/`: local-only human-readable daily Markdown notes.
- `.mcp-memory/memory.jsonl`: private MCP Memory storage for agent-readable facts.

Do not commit `.mcp-memory/` or `docs/learning-log/`. Both are ignored by git.

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

## Learning Memory Schema

Use MCP Memory for compact facts that should affect future sessions.

Core entities:

- `student_understanding_profile`: how the student learns best.
- `learning_memory_protocol`: when and how to save learning progress.
- `learning_session_<timestamp>`: one session or major slice, using
  `YYYY-MM-DDTHH:MM:SS+03:00`.
- `concept_<slug>`: one important concept, such as `concept_tensor` or
  `concept_learning_rate`.

Core relations:

- `student_understanding_profile` `uses` `learning_memory_protocol`
- `learning_session_<timestamp>` `covered` `concept_<slug>`
- `concept_<slug>` `supports` the relevant code or workflow slice when useful

Keep MCP Memory short. Long explanations, formulas, examples, and study notes
belong in `docs/learning-log/`.
