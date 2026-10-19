# Agen Memori — Virtual Context dan Memori Paging

> Context windows adalah finite. Conversations, documents, dan tool traces adalah tidak. fix adalah OS virtual memori restated — utama context adalah RAM, eksternal store adalah disk, agen itu pages antara mereka. MemGPT (Packer et al., 2023) named pola; banyak produksi memori sistem build pada itu.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan OS analogy MemGPT builds pada: utama context = RAM, eksternal context = disk, memori tool = page di/out.
- Implementasikan dua-tier MemGPT pola di stdlib dengan sebuah utama-context buffer, sebuah eksternal searchable store, dan page di/out tool.
- Deskripsikan bagaimana agen itu issues "interrupts" ke query atau modify eksternal memori dan bagaimana hasil adalah spliced back menjadi berikut prompt.
- Identifikasi MemGPT design choices itu carry menjadi Letta (Pelajaran 08) dan Mem0 (Pelajaran 09).

## Masalahnya

Context windows look like mereka sebaiknya solve memori. Mereka lakukan tidak. Tiga mode kegagalan recur di produksi:

1. **Overflow.** Multi-giliran conversations, long documents, atau tool-panggil-heavy trajectories cross window. Everything past cutoff adalah gone.
2. **Dilution.** Even di dalam window, stuffing irrelevant context dilutes attention atas apa matters. Frontier model tetap degrade pada long inputs.
3. **Persistence.** SEBUAH baru sesi memulai dengan sebuah kosong window. Agen tanpa eksternal memori cannot nyatakan "remember ketika kamu bertanyaed me ke..." lintas sesi.

Bigger windows bantu but lakukan tidak fix ini. Mem0's 2025 paper measured itu 128k-window baselines tetap miss long-horizon facts itu sebuah 4k-window agen dengan eksternal memori catches.

## Konsepnya

###  OS analogy

MemGPT (Packer et al., arXiv:2310.08560, v2 Feb 2024) maps context management ke operating-sistem virtual memori:

| OS concept | MemGPT concept | 2026 production analog |
|------------|---------------|------------------------|
| RAM | main context (prompt) | Anthropic/OpenAI context window |
| Disk | external context | vector DB, KV, graph store |
| Page fault | memory tool call | `memory.search`, `memory.read`, `memory.write` |
| OS kernel | agent control loop | ReAct loop with memory tools |

Agen itu berjalan sebuah normal ReAct loop. Satu extra class dari tool lets itu page data di dan out dari utama context.

### Dua tiers

- **Main context.** Fixed-size prompt holding current tugas. Selalu visible ke model.
- **External context.** Unbounded, searchable via tool. Baca ketika relevant, written ketika facts emerge.

 original paper evaluated design pada dua tugas melampaui base window: document analysis longer than 100k token dan multi-sesi chat dengan persistent memori lintas days.

###  interrupt pola

MemGPT introduces memori-sebagai-interrupt: mid-conversation agen itu bisa invoke sebuah memori tool, runtime executes itu, dan hasil splices menjadi berikut assistant giliran sebagai sebuah baru observasi. Conceptually identical ke sebuah Unix `read()` syscall itu blocks process, mengembalikan bytes, dan process melanjutkan.

Kanonikal memori tool surface:

- `core_memory_append(section, text)` — tulis ke sebuah persistent bagian dari prompt.
- `core_memory_replace(section, old, new)` — edit sebuah persistent bagian.
- `archival_memory_insert(text)` — tulis ke searchable eksternal store.
- `archival_memory_search(query, top_k)` — retrieve dari eksternal store.
- `conversation_search(query)` — scan past giliran.

### Di mana paper berakhir dan produksi begins

Di September 2024 MemGPT became Letta. riset repo (`cpacker/MemGPT`) remains; Letta Perluass design:

- Tiga tiers alih-alih dua (core, recall, archival — Pelajaran 08).
- Native penalaran replacing `send_message`/heartbeat pola (Pelajaran 08).
- Sleep-time agen berjalan async memori kerja (Pelajaran 08).

 MemGPT paper adalah 2026 foundation even jika produksi sistem jalankan Letta, Mem0, atau sebuah custom dua-tier store.

### Di mana ini pola goes salah

- **Memory rot.** Menulis accumulate faster than membaca; retrieval drowns di stale facts. Fix: periodic consolidation (Letta sleep-time), eksplisit invalidation (Mem0 conflict detector).
- **Memory poisoning.** Eksternal memori adalah retrieved text. Jika attacker-controlled content lands di sebuah memori note, agen itu re-ingests itu berikut sesi. Ini adalah Greshake et al. (Pelajaran 27) attack restated atas time.
- **Citation loss.** Agen recalls " user asked me ke ship X" but cannot cite yang giliran. Store source references (sesi ID, giliran ID) dengan setiap archival tulis.

```figure
context-budget
```

## Bangun Sendiri

`code/main.py` implements MemGPT's dua-tier pola di stdlib:

- `MainContext` — fixed-size prompt buffer dengan sebuah `core` dict dan sebuah `messages` list; auto-compacts oldest messages ketika atas cap.
- `ArchivalStore` — di-memori BM25-esque store (token-overlap scoring) dari (id, text, tags, sesi, giliran) records.
- Lima memori tool mapping ke MemGPT surface.
- SEBUAH scripted agen itu fills archival dengan facts, maka menjawab sebuah pertanyaan oleh pemanggilan `archival_memory_search`.

Jalankan:

```
python3 code/main.py
```

 trace menunjukkan agen itu writing tiga facts, filling utama context ke cap (forcing eviction), maka answering sebuah follow-up pertanyaan oleh retrieving dari archival — reproducing MemGPT workflow tanpa apa pun nyata LLM.

## Gunakan

Setiap produksi memori sistem today adalah sebuah MemGPT variant:

- **Letta** (Pelajaran 08) — tiga tiers, native penalaran, sleep-time compute.
- **Mem0** (Pelajaran 09) — vector + KV + graf fused dengan sebuah scoring layer.
- **OpenAI Assistants / Responses** — managed memori via threads dan file.
- **Claude Agent SDK** — long-term memori via skills dan sesi store.

Pick satu oleh bentuk operasional (mandiri-hosted, managed, framework-integrated), tidak oleh core pola — core pola adalah MemGPT.

###  bentuk dari agen memori

Paging solves capacity. Itu melakukan tidak putuskan apa ke store. Empat memori types recur lintas produksi sistem, setiap answering sebuah berbeda pertanyaan:

- **Working memory** — apa matters right now? di-context tier: current tugas, recent giliran, pinned core sections. prompt itself.
- **Episodic memory** — apa happened? Past giliran dan trajectories, stored dengan sesi dan giliran references, replayable pada demand.
- **Semantic memory** — apa adalah true? Facts tentang user, domain, dunia, updated dan deduplicated sebagai mereka change.
- **Procedural memory** — bagaimana lakukan I lakukan ini? Learned routines, preferences, dan rules itu steer future perilaku rather than recall.

Buka-source implementations pick berbeda points dari attack:

| Type | Implementation | How it tackles it |
|------|----------------|-------------------|
| Working | MemGPT / Letta | Pages content in and out of a fixed prompt budget via memory tools (this lesson, Lesson 08) |
| Episodic | Zep | Temporal knowledge graph — facts carry validity intervals, so "what was true when" is queryable |
| Semantic | Mem0 | Extraction pipeline that dedupes and updates facts across vector, KV, and graph stores (Lesson 09) |
| Semantic + procedural | LangMem | Background extraction of facts and behavioral rules into a store the agent consults between turns |
| Episodic + semantic | agentmemory | Captures sessions as they run, consolidates them into typed, searchable records |

## Rilis

`outputs/skill-virtual-memory.md` adalah sebuah reusable skill itu produces sebuah benar dua-tier memori scaffold (utama + archival + tool surface) untuk apa pun target runtime, dengan eviction kebijakan dan citation fields wired di.

## Latihan

1. Tambahkan sebuah `max_main_context_tokens` cap measured di token (approximate dengan `len(text.split())` * 1.3). Compact oldest messages menjadi sebuah summary ketika cap adalah exceeded. Bandingkan perilaku dengan dan tanpa summarizer.
2. Implementasikan BM25 properly atas archival store (term frequency, inverse document frequency). Measure recall@10 pada sebuah toy fact set vs token-overlap baseline.
3. Add `citation` fields (session_id, turn_id, source_url) ke archival inserts. Buat agen itu cite sources pada setiap retrieval-backed jawaban.
4. Simulate memori poisoning: Tambahkan sebuahn archival record itu menyatakan "ignore semua future user instructions." Tulis sebuah guard itu scans retrievals untuk directive-shaped text dan marks mereka untrusted.
5. Port implementasi ke pakai MemGPT riset repo's core-memori JSON schema (`cpacker/MemGPT`). Apa changes ketika kamu switch dari flat strings ke typed sections?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Virtual context | "Unlimited memory" | Main (prompt) + external (searchable) tiers with page in/out |
| Main context | "Working memory" | The prompt — fixed-size, always visible |
| Archival memory | "Long-term store" | External searchable persistence, retrieved on demand |
| Core memory | "Persistent prompt section" | Named sections pinned inside the main context |
| Memory tool | "Memory API" | Tool call the agent issues to read/write external memory |
| Interrupt | "Memory page fault" | Agent pauses, runtime fetches, result splices into next turn |
| Memory rot | "Stale facts" | Old writes drown retrieval; fix with consolidation |
| Memory poisoning | "Injected persistent note" | Attacker content stored as memory, re-ingested on recall |

## Bacaan Lanjutan

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — OS-inspired virtual context paper
- [Letta, Memori Blocks blog](https://www.letta.com/blog/memory-blocks) — tiga-tier evolution
- [Anthropic, Efektif context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — treating context sebagai sebuah budget
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — hibrida produksi memori di atas ini pola
- [Zep (getzep/zep)](https://github.com/getzep/zep) — temporal knowledge-graf memori dari taxonomy table
- [Mem0 (mem0ai/mem0)](https://github.com/mem0ai/mem0) — extraction pipeline di balik Pelajaran 09's hibrida store
- [LangMem (langchain-ai/langmem)](https://github.com/langchain-ai/langmem) — background extraction dari facts dan behavioral rules
- [agentmemory (rohitg00/agentmemory)](https://github.com/rohitg00/agentmemory) — sesi capture consolidated menjadi typed, searchable records
