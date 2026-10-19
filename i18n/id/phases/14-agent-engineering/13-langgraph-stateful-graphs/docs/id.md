# Stateful Graf Orkestrasi — Durable Execution dan Checkpoints

> Agen adalah sebuah state machine; node adalah functions; edges adalah transitions; state adalah checkpointed setelah setiap node. Resume dari apa pun kegagalan pada terakhir successful checkpoint. LangGraph adalah 2026 reference untuk ini model dari low-level stateful orkestrasi.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Deskripsikan LangGraph's core model: state machine dengan typed state, function node, conditional edges, dan post-node checkpoints.
- Name empat kapabilitas docs highlight: durable execution, streaming, manusia-di--loop, comprehensive memori.
- Jelaskan tiga orkestrasi topologies LangGraph mendukung: supervisor, peer-ke-peer (swarm), hierarchical (nested subgraphs).
- Implementasikan sebuah stdlib state graf dengan typed state, conditional edges, dan sebuah checkpoint/resume cycle.

## Masalahnya

Agen dan workflow share sebuah masalah: ketika sebuah 40-langkah jalankan gagal pada langkah 38, kamu inginkan ke resume dari langkah 38, tidak mulai atas. Kedua-class state model leave operators hacking retries di sekitar sebuah library itu assumes fresh berjalan.

LangGraph's design jawaban: state adalah sebuah pertama-class typed object, mutations adalah eksplisit, dan checkpoints persist setelah setiap node. Resume adalah sebuah `load_state(session_id)` panggil.

## Konsepnya

###  graf

SEBUAH graf adalah defined oleh:

- **State type.** SEBUAH typed dict (atau Pydantic model) itu setiap node membaca dan mutates.
- **Nodes.** Pure functions `(state) -> state_update`. Updates adalah merged menjadi state setelah kembalikan.
- **Edges.** Conditional atau direct transitions antara node.
- **Entry and exit.** `START` dan `END` sentinel node mark boundary.

Contoh: sebuah agen dengan `classify`, `refund`, `bug`, `sales`, `done` node — sebuah routing workflow sebagai sebuah graf.

### Durable execution

Setelah setiap node mengembalikan, runtime serializes state dan menulis itu ke sebuah checkpointer (SQLite, Postgres, Redis, custom). Pada kegagalan pada langkah N, runtime bisa `resume(session_id)` dan pick up dari langkah N+1 dengan exact state.

 LangGraph docs explicitly highlight produksi users di mana ini matters: Klarna, Uber, J.P. Morgan. claim isn't graf bentuk; itu's itu graf bentuk plus checkpointing membuat recovery cheap.

### Streaming

Setiap node bisa yield partial output. graf streams per-node-delta events ke caller so UIs update sebagai graf berjalan.

### Manusia-di--loop

Inspect dan modify state antara node. Implementations: pause sebelum sebuah kritis node, surface state ke sebuah manusia, accept modifications, resume. checkpointer membuat ini easy karena state adalah sudah serialized.

### Memori

Short-term (di dalam sebuah jalankan — conversation history di state) dan long-term (lintas berjalan — persistent via checkpointer plus sebuah terpisah long-term store). LangGraph integrates dengan eksternal memori sistem (Mem0, custom) via tool.

### Tiga topologies

1. **Supervisor.** Pusat router LLM dispatches ke specialist subagents. `create_supervisor()` di `langgraph-supervisor` (though LangChain tim di 2026 recommends doing ini melalui pemanggilan tool directly untuk more context kontrol).
2. **Swarm / peer-to-peer.** Agen hand off directly via sebuah shared tool surface. Tidak pusat router.
3. **Hierarchical.** Supervisors managing sub-supervisors, implemented sebagai nested subgraphs.

### Di mana ini pola goes salah

- **Checkpoints too small.** Hanya checkpointing conversation giliran leaves tool state dan memori menulis unrecoverable. Penuh state harus serialize.
- **Non-deterministic nodes.** Resume assumes node inputs produce sama state update. Random seeds, wall-clock, eksternal APIs harus menjadi captured.
- **Over-use of conditional edges.** SEBUAH graf dengan setiap edge conditional adalah sebuah state machine itu cannot menjadi reasoned tentang. Prefer linear chains dengan occasional branches.

```figure
langgraph-state
```

## Bangun Sendiri

`code/main.py` implements sebuah stdlib stateful graf:

- `State` — sebuah typed dict dengan `messages`, `step`, `route`, `output`, `human_approval`.
- `Node` — callable taking state dan returning sebuah update dict.
- `StateGraph` — node + edges + conditional edges + jalankan + resume.
- `SQLiteCheckpointer` (di-memori fake) — serializes state setelah setiap node; `load(session_id)` restores.
- SEBUAH demo graf: classify -> branch(refund / bug / sales) -> manusia gate -> kirim.

Jalankan:

```
python3 code/main.py
```

 trace menunjukkan pertama jalankan failing pada manusia gate, persistence, maka resume producing final output.

## Gunakan

- **LangGraph** — reference, produksi-ready. Pakai `create_react_agent`, `create_supervisor`, atau build kamu own graf.
- **AutoGen v0.4** (Pelajaran 14) — aktor model alternative untuk high-concurrency scenarios.
- **Claude Agent SDK** (Pelajaran 17) — managed harness dengan bawaan sesi store.
- **Custom** — ketika kamu butuhkan exact kontrol atas state bentuk atau checkpointer backend.

## Rilis

`outputs/skill-state-graph.md` menghasilkan sebuah LangGraph-shaped state graf di apa pun target runtime dengan checkpointing dan resume wired di.

## Latihan

1. Tambahkan sebuah conditional edge dari `classify` ke `end` ketika classification confidence adalah di bawah sebuah threshold. Resume jalankan setelah sebuah manusia sets `route` manually.
2. Tukar SQLite-like fake untuk sebuah nyata SQLite checkpointer. Measure per-langkah serialization overhead.
3. Implementasikan paralel edges: dua node jalankan concurrently, merge oleh sebuah custom reducer. Apa melakukan immutable state buy here?
4. Baca `langgraph-supervisor` reference. Port toy ke `create_supervisor`. Bandingkan trace shapes.
5. Add streaming: setiap node yields partial state sementara itu berjalan. Print deltas sebagai mereka arrive.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State graph | "Agent as state machine" | Typed state + nodes + edges + reducers |
| Checkpointer | "Persistence backend" | Serializes state after every node; enables resume |
| Reducer | "State merger" | Function that combines current state with a node's update |
| Conditional edge | "Branch" | Edge chosen by a function of state |
| Subgraph | "Nested graph" | A graph used as a node inside another graph |
| Durable execution | "Resume from failure" | Restart at the last successful node with exact state |
| Supervisor | "Router LLM" | Central dispatcher for specialist subagents |
| Swarm | "P2P agents" | Agents hand off via shared tools; no central router |

## Bacaan Lanjutan

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — reference docs
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — supervisor pola API
- [AutoGen v0.4, Microsoft Riset](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — aktor-model alternative
- [Claude Agen SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — sesi store dan subagents
