#  Aktor Model untuk Agen — Async Messages dan Typed Runtimes

> Agen sebagai actors: async message exchange, event-driven handlers, fault isolasi, natural concurrency. AutoGen v0.4 (Microsoft Riset, Jan 2025) redesigned agen orkestrasi di sekitar ini model; framework adalah now di maintenance mode, dengan Microsoft Agen Framework (public preview Oct 2025) sebagai miliknya produksi successor.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Deskripsikan aktor model: agen sebagai actors, messages sebagai hanya IPC, kegagalan isolasi per aktor.
- Name AutoGen v0.4's tiga API layers — Core, AgentChat, Extensions — dan apa setiap adalah untuk.
- Jelaskan mengapa decoupling message delivery dari handling memberi fault isolasi dan natural concurrency.
- Implementasikan sebuah stdlib aktor runtime di Python dan port sebuah dua-agen kode-review flow ke itu.

## Masalahnya

Sebagian besar agen framework adalah synchronous: satu agen produces, satu agen consumes, di sebuah panggil stack. Kegagalan crash stack. Concurrency adalah bolted pada. Distribution membutuhkan rewriting.

AutoGen v0.4's jawaban: aktor model. Setiap agen adalah sebuah aktor dengan sebuah private inbox. Messages adalah hanya interaction. runtime decouples delivery dari handling. Kegagalan isolate ke satu aktor. Concurrency adalah native. Distribution adalah just berbeda transport.

## Konsepnya

### Actors

Sebuah aktor memiliki:

- SEBUAH private state (tidak pernah directly touched dari outside).
- Sebuah inbox (message queue).
- SEBUAH handler: `receive(message) -> effects` di mana effects bisa menjadi "reply," "kirim ke other aktor," "spawn baru aktor," "update state," "berhenti mandiri."

Dua actors cannot share memori. Mereka bisa hanya kirim messages.

### Tiga API layers

AutoGen v0.4 splits miliknya surface menjadi tiga:

1. **Core.** Low-level aktor framework. `AgentRuntime`, `Agent`, `Message`, `Topic`. Async message exchange, event-driven.
2. **AgentChat.** Tugas-driven high-level API (Gantiment untuk v0.2's ConversableAgent). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **Extensions.** Integrations — OpenAI, Anthropic, Azure, tool, memori.

### Mengapa decoupling matters

Di v0.2 model, pemanggilan `agent_a.chat(agent_b)` synchronously blocks agent_a sampai agent_b mengembalikan. Di v0.4, `send(agent_b, msg)` puts message di agent_b's inbox dan mengembalikan. runtime delivers later. Tiga consequences:

- **Fault isolation.** Agen B crashing melakukan tidak crash Agen SEBUAH — runtime catches kegagalan di B's handler dan memutuskan apa ke lakukan (log, retry, dead-letter).
- **Natural concurrency.** Banyak messages di flight pada once; actors process mereka inbox concurrently.
- **Distribution-ready.** Inbox + transport adalah sama abstraction whether aktor adalah di-process atau pada another host.

### Topologies

- **RoundRobinGroupChat.** Agen ambil giliran di sebuah fixed rotation.
- **SelectorGroupChat.** SEBUAH selector agen picks yang goes berikut based pada conversation context.
- **Magentic-One.** Reference multi-agen tim untuk web browsing, kode execution, file handling. Built pada AgentChat.

### Observabilitas

OpenTelemetry dukung adalah built di. Setiap message meng-emit sebuah span; pemanggilan tool carry `gen_ai.*` attributes per 2026 OTel GenAI semantic conventions (Pelajaran 23).

### Status: maintenance mode

Early 2026: AutoGen v0.7.x adalah stable untuk riset dan prototyping. Microsoft memiliki shifted active development ke Microsoft Agen Framework, produksi successor (public preview Oct 1 2025; 1.0 GA adalah targeted untuk berakhir dari Q1 2026). AutoGen polas port forward cleanly — aktor model adalah durable ide.

```figure
actor-mailbox
```

## Bangun Sendiri

`code/main.py` implements sebuah stdlib aktor runtime:

- `Message` — typed payload dengan `sender`, `recipient`, `topic`, `body`.
- `Actor` — abstract dengan `receive(message, runtime)`.
- `Runtime` — event loop dengan sebuah shared queue, delivery, kegagalan isolasi.
- SEBUAH dua-aktor demo: `ReviewerAgent` reviews kode, `ChecklistAgent` berjalan sebuah checklist; mereka exchange messages sampai consensus.

Jalankan:

```
python3 code/main.py
```

 trace menunjukkan message delivery, sebuah simulated kegagalan di satu aktor itu melakukan tidak crash other, dan convergence pada sebuah shared verdict.

## Gunakan

- **AutoGen v0.4/v0.7** (maintenance) — stable untuk riset, prototyping, multi-agen polas.
- **Microsoft Agent Framework** — produksi successor (public preview Oct 2025); sama aktor-model ideas di sebuah refreshed API.
- **LangGraph swarm topology** (Pelajaran 13) — similar pola via shared-tool handoffs.
- **Custom actor runtime** — ketika kamu butuhkan specific transport (NATS, RabbitMQ, gRPC).

## Rilis

`outputs/skill-actor-runtime.md` menghasilkan sebuah minimal aktor runtime plus sebuah tim templat (RoundRobin atau Selector) untuk sebuah given multi-agen tugas.

## Latihan

1. Tambahkan sebuah dead-letter queue: ketika sebuah handler raises, park failing message untuk manusia inspection. Bagaimana sering melakukan DLQ dapatkan hit di kamu toy?
2. Implementasikan `SelectorGroupChat`: sebuah selector aktor picks yang processes berikut message based pada conversation state.
3. Add distributed transport: Tukar di-process queue untuk sebuah JSON-atas-HTTP server so actors bisa jalankan di terpisah processes.
4. Wire sebuah OTel span per message (atau sebuah tidak-op stand-di). Emit `gen_ai.agent.name`, `gen_ai.operation.name` per Pelajaran 23.
5. Baca AutoGen v0.4's architecture post. Port kamu toy ke nyata `autogen_core` API. Apa melakukan kamu skip itu matters di produksi?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Actor | "Agent" | Private state + inbox + handler; no shared memory |
| Message | "Event" | Typed payload; the only way actors interact |
| Inbox | "Mailbox" | Per-actor queue of pending messages |
| Runtime | "Agent host" | Event loop that routes messages and isolates failures |
| Topic | "Channel" | Named publish-subscribe route between actors |
| Fault isolation | "Let it crash" | One actor failing does not crash others |
| RoundRobinGroupChat | "Fixed-rotation team" | Agents take turns in order |
| SelectorGroupChat | "Context-routed team" | Selector picks who goes next |
| Magentic-One | "Reference team" | Multi-agent squad for web + code + files |

## Bacaan Lanjutan

- [AutoGen v0.4, Microsoft Riset](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — redesign post
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — graf-shaped alternative
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — spans AutoGen meng-emit oleh default
