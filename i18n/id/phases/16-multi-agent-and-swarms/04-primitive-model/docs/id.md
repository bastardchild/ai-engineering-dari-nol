# Model Primitif Multi-Agent

> Empat primitif, tidak lebih — agent, handoff, shared state, orchestrator — membentang di ruang desain empat dimensi, dan framework multi-agent utama yang dirilis pada 2026 (AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Microsoft Agent Framework) adalah titik-titik di dalamnya. Pelajaran ini membangunnya dari nol, menjalankan sistem main-main pada keempatnya, lalu memetakan setiap framework utama ke sumbu yang sama sehingga kamu bisa membaca rilis baru mana pun dalam satu paragraf.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## Masalahnya

Setiap enam bulan, sebuah framework multi-agent baru dirilis. AutoGen pada 2023. CrewAI pada 2024. LangGraph dan OpenAI Swarm pada 2024. Google ADK pada April 2025. Microsoft Agent Framework RC pada Februari 2026. Setiap siaran pers mengklaim sebagai "abstraksi yang tepat."

Jika kamu mencoba mempelajarinya satu per satu, kamu akan kelelahan. API-nya terlihat berbeda. Dokumentasinya tidak setuju tentang apa itu "agent." Satu framework menyebut shared memory-nya "blackboard," yang lain menyebutnya "message pool," yang ketiga menyebutnya "StateGraph." Kamu mulai menduga bidang ini hanya sedang berputar-putar.

Tidak. Di balik pemasarannya, keempat primitif itu stabil. Pelajari sekali, baca setiap framework baru dalam satu paragraf.

## Konsepnya

### Keempat primitif

1. **Agent** — sebuah system prompt ditambah daftar tool. Stateless; setiap run dimulai dari system prompt-nya dan riwayat pesan saat ini.
2. **Handoff** — transfer terstruktur dari satu agent ke agent lain. Secara mekanis, sebuah tool call yang mengembalikan agent baru atau graph edge yang mengikuti sebuah kondisi.
3. **Shared state** — struktur data apa pun yang bisa dibaca lebih dari satu agent (terkadang ditulis). Message pool, blackboard, key-value store, vector memory.
4. **Orchestrator** — siapa pun yang memutuskan siapa yang berbicara berikutnya. Pilihan: graph eksplisit (deterministik), speaker-selector LLM (lembut), handoff call dari pembicara terakhir (OpenAI Swarm), atau scheduler di atas antrean (arsitektur swarm).

Itulah seluruh ruang desain. Setiap framework memilih default untuk setiap sumbu; sisanya adalah sintaks permukaan.

### Cara setiap framework 2026 memetakannya

| Framework | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | tool returns Agent | caller's problem | the LLM's next handoff call |
| AutoGen v0.4 / AG2 | `ConversableAgent` | speaker-selector on GroupChat | message pool | selector function (LLM or round-robin) |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task outputs chained | manager LLM or static order |
| LangGraph | node function | graph edge + condition | `StateGraph` reducer | the graph, deterministic |
| Microsoft Agent Framework | agent + orchestration patterns | pattern-specific | thread / context | pattern-specific |
| Google ADK | agent + A2A card | A2A task | A2A artifacts | host decides |

Perbedaan permukaan terlihat besar. Di baliknya: empat knob yang sama.

### Mengapa ini penting

Setelah kamu melihat primitif-primitifnya, perbandingan framework menjadi checklist singkat:

- Apakah orchestrator mempercayai LLM untuk merutekan (Swarm) atau mengunci routing dalam kode (LangGraph)?
- Apakah shared state berupa full-history (GroupChat) atau projected (StateGraph reducer)?
- Apakah agent bisa memodifikasi prompt satu sama lain (manager CrewAI) atau hanya melakukan handoff (Swarm)?

Tiga pertanyaan itu menjawab 80% dari framework mana yang cocok untuk masalah tertentu. Kamu berhenti berburu "framework multi-agent terbaik" dan mulai mendesain untuk sumbu yang benar-benar kamu pedulikan.

### Wawasan stateless

Setiap primitif kecuali shared state bersifat stateless. Agent adalah fungsi dari (prompt, tools). Handoff adalah function call. Orchestrator adalah scheduler. **Satu-satunya hal stateful di sistem adalah shared state.** Di sanalah semua bug menarik tinggal: memory poisoning (Lesson 15), pengurutan pesan, versioning, write contention.

Framework yang menyembunyikan shared state (Swarm) mendorong masalahnya ke pemanggil. Framework yang menyalurkannya (checkpoint LangGraph, pool AutoGen) membuatnya dapat diinspeksi tetapi memindahkan biaya koordinasi ke implementasi shared state.

### Anatomi satu primitif

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

Tanpa memori. Tanpa state. Dua agent dengan system prompt dan tools yang sama dapat dipertukarkan. Segala sesuatu yang terlihat seperti per-agent state sebenarnya berada di shared state atau protokol handoff.

#### Handoff

```
Handoff = (from_agent, to_agent, reason, payload)
```

Tiga implementasi mendominasi:

- **Function return** — tool mengembalikan agent berikutnya. Ini adalah pola OpenAI Swarm. Agent membawa routing di dalam tool schema mereka.
- **Graph edge** — LangGraph. Edge bersifat deklaratif. LLM menghasilkan nilai; sebuah kondisi memilih node berikutnya.
- **Speaker selection** — AutoGen GroupChat. Sebuah fungsi selector (terkadang itu sendiri adalah panggilan LLM) membaca pool dan memilih siapa yang berbicara berikutnya.

#### Shared state

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

Setidaknya, sebuah daftar pesan. Sering lebih: artefak terstruktur (keluaran Task CrewAI), konten bertipe (reducer LangGraph), memori eksternal (MCP, vector DB).

Dua topologi: **full pool** (setiap agent melihat setiap pesan) dan **projected** (agent melihat tampilan berlingkup peran). Full pool sederhana dan menskalakan dengan buruk. Projected pool menskalakan tetapi memerlukan desain schema di awal.

#### Orchestrator

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

Empat varian:

- **Statis** — graph ditetapkan pada waktu build (LangGraph deterministik, CrewAI Sequential).
- **Dipilih LLM** — sebuah LLM membaca pool dan memilih pembicara berikutnya (AutoGen, CrewAI Hierarchical).
- **Didorong handoff** — agent saat ini memutuskan dengan memanggil tool handoff (Swarm).
- **Didorong antrean** — worker menarik dari antrean bersama; tanpa next-speaker eksplisit (arsitektur swarm, Matrix).

### Apa yang berubah di antara framework

Setelah primitif-primitifnya tetap, keputusan desain yang tersisa adalah:

- **Strategi memori** — checkpointing sementara vs tahan lama (checkpointer LangGraph).
- **Batas keselamatan** — siapa yang bisa menyetujui sebuah handoff (human-in-the-loop).
- **Akuntansi biaya** — anggaran token per-agent.
- **Observabilitas** — melacak handoff, memersistensikan state untuk replay.

Semuanya dapat diimplementasikan di atas primitif-primitifnya. Tidak ada yang merupakan primitif baru.

```figure
a5-primitive-radar
```

## Bangun

`code/main.py` mengimplementasikan keempat primitif dalam ~150 baris Python stdlib. Tanpa LLM nyata — setiap agent adalah kebijakan ber-skrip agar fokus tetap pada struktur koordinasi.

File ini mengekspor:

- `Agent` — sebuah dataclass dari nama, system prompt, tool, fungsi kebijakan.
- `Handoff` — fungsi yang mengembalikan agent baru.
- `SharedState` — message pool yang aman-thread.
- `Orchestrator` — tiga varian: `StaticOrchestrator`, `HandoffOrchestrator`, `LLMSelectorOrchestrator` (disimulasikan).

Demo menjalankan pipeline tiga-agent yang sama (research → write → review) melalui ketiga tipe orchestrator dan mencetak message pool di akhir. Kamu bisa melihat bahwa keluarannya hanya berbeda dalam *siapa yang memilih berikutnya*; agent dan shared state identik di semua run.

Jalankan:

```
python3 code/main.py
```

Keluaran yang diharapkan: tiga run orchestrator, satu per pola. Masing-masing mencetak message pool akhir. Run yang didorong handoff mencapai lebih sedikit agent jika researcher memutuskan ia sudah selesai lebih awal — itu adalah tradeoff routing-LLM dalam miniatur.

## Gunakan

`outputs/skill-primitive-mapper.md` adalah skill yang membaca codebase multi-agent mana pun atau dokumen framework dan mengembalikan pemetaan empat-primitif. Jalankannya pada rilis framework baru untuk mendapat pemahaman satu paragraf sebelum membaca dokumentasi secara mendalam.

## Rilis

Sebelum mengadopsi framework baru, tulis pemetaan primitifnya. Jika kamu tidak bisa, dokumentasinya tidak lengkap atau framework-nya sedang menemukan primitif kelima (jarang — periksa varian shared state yang belum pernah kamu lihat).

Kunci pemetaan itu di dokumen arsitektur-mu. Saat anggota tim baru bergabung, kirimkan pemetaan itu sebelum dokumentasi API. Saat versi framework berubah, diff pemetaannya, bukan changelog-nya.

## Latihan

1. Jalankan `code/main.py` tiga kali dengan kebijakan agent yang berbeda. Amati bagaimana pilihan orchestrator mengubah agent mana yang berjalan.
2. Implementasikan tipe orchestrator keempat: yang didorong antrean di mana agent mem-poll shared state untuk pekerjaan. Deadlock apa yang bisa terjadi, dan bagaimana kamu mendeteksinya?
3. Ambil quickstart LangGraph (https://docs.langchain.com/oss/python/langgraph/workflows-agents) dan tulis ulang sebagai keempat primitif. Abstraksi LangGraph mana yang memetakan 1:1 dan mana yang merupakan pembungkus kenyamanan?
4. Baca cookbook OpenAI Swarm (https://developers.openai.com/cookbook/examples/orchestrating_agents). Identifikasi primitif mana dari keempatnya yang membuat Swarm paling ergonomis, dan mana yang ia dorong ke pemanggil.
5. Temukan satu framework di tabel ini yang menyembunyikan shared state sepenuhnya. Jelaskan apa yang rusak ketika agent perlu berkoordinasi di antara handoff tanpa membaca ulang riwayat.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "An LLM with tools" | A `(system_prompt, tools, model)` triple. Stateless. |
| Handoff | "Transfer of control" | A structured call that names the next agent and optional payload. Three implementations: function return, graph edge, speaker selection. |
| Shared state | "Memory" / "context" | The only stateful part of a multi-agent system. Message pool or blackboard. |
| Orchestrator | "Coordinator" | Whoever decides who runs next. Static graph, LLM selector, handoff-driven, or queue-driven. |
| Primitive | "Abstraction" | One of the four axes every framework parameterizes. Not a framework feature. |
| Message pool | "Shared chat history" | Full-history shared state. Easy to reason about, scales badly. |
| Projected state | "Scoped view" | Role-specific view into shared state. Scales, requires schema design. |
| Speaker selection | "Who talks next" | Orchestrator pattern where a function (often an LLM) picks the next agent from a group. |

## Bacaan Lanjutan

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — the clearest articulation of handoff-driven orchestration
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat + speaker selection is the reference for LLM-selected orchestration
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — graph-edge orchestration and reducer-based shared state
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — role-goal-backstory agents, Sequential / Hierarchical processes
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — the live AutoGen v0.2 line after Microsoft moved v0.4 into maintenance
