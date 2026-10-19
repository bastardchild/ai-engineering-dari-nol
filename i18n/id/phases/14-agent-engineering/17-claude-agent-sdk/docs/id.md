#  Harness sebagai sebuah Library — Subagents dan Sesi Store

> SEBUAH harness kamu bisa import: bawaan tool, subagents untuk context isolasi, hooks, W3C trace propagation, sesi persistence. Claude Agen SDK adalah reference contoh — library form dari Claude Kode harness — dan Claude Managed Agen adalah hosted alternative untuk long-berjalan async kerja.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan difference antara Anthropic Klien SDK (raw API) dan Claude Agen SDK (harness bentuk).
- Deskripsikan subagents — parallelization dan context isolasi — dan ketika ke reach untuk mereka.
- Name Python SDK's sesi store surface (`append`, `load`, `list_sessions`, `delete`, `list_subkeys`) dan peran dari `--session-mirror`.
- Implementasikan sebuah stdlib harness dengan bawaan tool, subagent spawning dengan isolated context, lifecycle hooks, dan sebuah sesi store.

## Masalahnya

SEBUAH raw LLM API mendapatkan kamu satu round-trip. SEBUAH produksi agen membutuhkan tool execution, MCP servers, lifecycle hooks, subagent spawning, sesi persistence, trace propagation. Claude Agen SDK ships ini bentuk sebagai sebuah library — sama harness Claude Kode memakai, exposed untuk custom agen.

## Konsepnya

### Klien SDK vs Agen SDK

- **Client SDK (`anthropic`).** Raw Messages API. Kamu own loop, tool, state.
- **Agent SDK (`claude-agent-sdk`).** Bawaan tool execution, MCP connections, hooks, subagent spawning, sesi store. Claude Kode loop sebagai sebuah library.

### Bawaan tool

 SDK ships 10+ tool out dari box: file baca/tulis, shell, grep, glob, web fetch, more. Custom tool register via standard tool-schema interface.

### Subagents

Dua purposes documented oleh Anthropic:

1. **Parallelization.** Jalankan independent kerja concurrently. "Find tes file untuk setiap dari ini 20 modules" adalah 20 paralel subagent tugas.
2. **Context isolation.** Subagents pakai mereka own context window; hanya hasil kembalikan ke orchestrator. orchestrator's budget adalah preserved.

Python SDK recent additions: `list_subagents()`, `get_subagent_messages()` untuk reading subagent transcripts.

### Sesi store

Protokol parity dengan TypeScript:

- `append(session_id, message)` — Tambahkan sebuah giliran.
- `load(session_id)` — restore conversation.
- `list_sessions()` — enumerate.
- `delete(session_id)` — dengan cascade ke subagent sesi.
- `list_subkeys(session_id)` — list subagent keys.

`--session-mirror` (CLI flag) mirrors transkrip ke sebuah eksternal file sebagai itu streams, untuk debugging.

### Hooks

Lifecycle hooks kamu bisa register:

- `PreToolUse`, `PostToolUse` — gate atau audit pemanggilan tool.
- `SessionStart`, `SessionEnd` — set up dan tear down.
- `UserPromptSubmit` — bertindak pada user input sebelum model sees itu.
- `PreCompact` — jalankan sebelum context compaction.
- `Stop` — cleanup pada agen exit.
- `Notification` — side-kanal alerts.

Hooks adalah bagaimana pro-workflow (Fase 14 curriculum reference) dan similar sistem add cross-cutting perilaku.

### W3C trace context

OTel spans active pada caller propagate menjadi CLI subprocess via W3C trace context headers. seluruh multi-process trace menunjukkan up sebagai satu trace di kamu backend.

### Claude Managed Agen

 hosted alternative (beta header `managed-agents-2026-04-01`). Long-berjalan async kerja, bawaan prompt caching, bawaan compaction. Trade kontrol untuk managed infrastructure.

### Di mana ini pola goes salah

- **Subagent over-spawn.** Spawning 100 subagents untuk 100 tiny tugas. Overhead dominates. Batch instead.
- **Hook creep.** Setiap tim adds hooks; startup time balloons. Review hooks quarterly.
- **Session bloat.** Sesi accumulate; size yang terus bertambah. Pakai `list_sessions` + expiry kebijakan.

```figure
ae-subagent-isolation
```

## Bangun Sendiri

`code/main.py` implements SDK bentuk di stdlib:

- `Tool`, `ToolRegistry` dengan bawaan `read_file`, `write_file`, `list_dir`.
- `Subagent` — private context, isolated jalankan, hasil returned.
- `SessionStore` — append, load, list, delete, list_subkeys.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- SEBUAH demo: utama agen spawns 3 subagents di paralel (setiap isolated), aggregates hasil, persists sesi.

Jalankan:

```
python3 code/main.py
```

 trace menunjukkan subagent context isolasi (orchestrator context size stays bounded), hook execution, dan sesi persistence.

## Gunakan

- **Claude Agent SDK** untuk Claude-pertama products itu inginkan Claude Kode harness bentuk.
- **Claude Managed Agents** untuk hosted long-berjalan async kerja.
- **OpenAI Agents SDK** (Pelajaran 16) untuk OpenAI-pertama counterparts.
- **LangGraph + custom tools** jika kamu inginkan graf-shaped state machine instead.

## Rilis

`outputs/skill-claude-agent-scaffold.md` scaffolds sebuah Claude Agen SDK app dengan subagents, hooks, sesi store, MCP server attachment, dan W3C trace propagation.

## Latihan

1. Tambahkan sebuah subagent spawner itu batches 20 tugas menjadi groups dari 5 paralel subagents. Measure orchestrator context size vs satu-per-tugas.
2. Implementasikan sebuah `PreToolUse` hook itu rate-limits `write_file` memanggil (5 per minute per sesi). Trace perilaku.
3. Wire `list_subkeys` ke render sebuah subagent tree. Apa melakukan deep nesting look like?
4. Port toy ke nyata `claude-agent-sdk` Python package. Apa changes tentang tool registration?
5. Baca Claude Managed Agen docs. Ketika akan kamu switch dari mandiri-hosted ke managed?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent SDK | "Claude Code as a library" | Harness shape: tools, MCP, hooks, subagents, session store |
| Subagent | "Child agent" | Separate context, own budget; results bubble up |
| Session store | "Conversation DB" | Persist, load, list, delete turns with subagent cascade |
| Hook | "Lifecycle callback" | Pre/post tool, session, prompt submit, compact, stop |
| W3C trace context | "Cross-process trace" | Parent span propagates into CLI subprocess |
| Managed Agents | "Hosted harness" | Anthropic-hosted long-running async work |
| `--session-mirror` | "Transcript mirror" | Writes session turns to an external file as they stream |
| MCP server | "Tool surface" | External tool/resource source attached to the agent |

## Bacaan Lanjutan

- [Claude Agen SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — library form dari Claude Kode
- [Anthropic, Building agen dengan Claude Agen SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — produksi polas
- [Claude Managed Agen overview](https://platform.claude.com/docs/en/managed-agents/overview) — hosted alternative
- [OpenAI Agen SDK](https://openai.github.io/openai-agents-python/) — counterpart
