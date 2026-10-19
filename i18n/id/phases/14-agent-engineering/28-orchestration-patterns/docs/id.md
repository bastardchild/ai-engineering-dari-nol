# Orkestrasi Polas: Supervisor, Swarm, Hierarchical

> Empat orkestrasi polas recur lintas 2026 framework: supervisor-worker, swarm / peer-ke-peer, hierarchical, debat. Anthropic's guidance: "Itu's tentang building right sistem untuk kamu membutuhkan." Mulai sederhana; add topology hanya ketika sebuah tunggal agen plus lima workflow polas adalah insufficient.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Name empat recurring orkestrasi polas dan ketika setiap fits.
- Deskripsikan 2026 LangChain recommendation: tool-panggil-based supervision vs supervisor libraries.
- Jelaskan Anthropic's "build right sistem" aturan dan bagaimana itu gates topology choice.
- Implementasikan sebuahll empat di stdlib terhadap sebuah umum scripted LLM.

## Masalahnya

Teams reach untuk "multi-agen" sebelum mereka butuhkan itu. Empat polas recur lintas framework; setelah kamu bisa name mereka, kamu bisa pick right satu — atau skip topology entirely.

## Konsepnya

### Supervisor-worker

- SEBUAH pusat routing LLM dispatches ke specialist agen.
- Memutuskan: loop back ke mandiri, hand off ke specialist, terminate.
- Specialists lakukan tidak talk ke setiap other; semua routing goes melalui supervisor.

Framework: LangGraph `create_supervisor`, Anthropic orchestrator-workers, CrewAI Hierarchical Process.

**2026 LangChain recommendation:** lakukan supervision melalui direct pemanggilan tool rather than `create_supervisor`. Memberi finer context engineering kontrol — kamu putuskan exactly apa setiap specialist sees.

### Swarm / peer-ke-peer

- Agen hand off directly via sebuah shared tool surface.
- Tidak pusat router.
- Lower latency than supervisor (fewer hops).
- Harder ke reason tentang (tidak tunggal inti dari kontrol).

Framework: LangGraph swarm topology, OpenAI Agen SDK handoffs (ketika semua agen bisa hand off ke semua others).

### Hierarchical

- Supervisors managing sub-supervisors managing workers.
- Implemented sebagai nested subgraphs di LangGraph; nested kru di CrewAI.
- Scales ke besar agen populations pada cost dari operasional complexity.

Ketika kamu butuhkan itu: ketika sebuah tunggal supervisor's context budget cannot hold descriptions dari semua specialists.

### Debat

- Paralel proposers + iterative cross-critique (Pelajaran 25).
- Tidak really orkestrasi — more verifikasi — but menunjukkan up sebagai sebuah topology choice di framework.

### Otonom kru vs deterministik flow

CrewAI formalizes dua deployment modes:

- **Flow** untuk deterministik event-driven automation (recommended starting inti untuk produksi).
- **Crew** untuk otonom peran-based collaboration.

Ini adalah orthogonal ke empat polas di atas but maps ke topology: Flow adalah typically supervisor atau hierarchical; Kru adalah typically supervisor dengan sebuah LLM router.

### Anthropic's guidance

"Keberhasilan di LLM space isn't tentang building sebagian besar sophisticated sistem. Itu's tentang building right sistem untuk kamu membutuhkan."

Decision order:

1. Tunggal agen + workflow polas (Pelajaran 12) — mulai here.
2. Supervisor-worker — ketika kamu punya 2-4 specialists.
3. Swarm — ketika latency matters more than penalaran clarity.
4. Hierarchical — hanya ketika supervisor context budget gagal.
5. Debat — ketika accuracy matters more than cost.

### Di mana ini pola goes salah

- **Topology-first thinking.** "Kita butuhkan multi-agen" sebelum identifying apa masalah multi-agen solves.
- **Bouncing handoffs in swarm.** SEBUAH -> B -> SEBUAH -> B. Pakai hop counters.
- **Fake hierarchy.** Tiga layers karena "enterprise"; dua actual teams. Collapse.

```figure
orchestration-pattern
```

## Bangun Sendiri

`code/main.py` implements semua empat polas di stdlib terhadap sebuah scripted LLM:

- `Supervisor` — pusat router.
- `Swarm` — peer-ke-peer dengan direct handoffs.
- `Hierarchical` — supervisors dari supervisors.
- `Debate` — paralel proposers + critique.

Setiap pola menangani sama tiga-intent tugas (refund / bug / sales). Trace shapes differ.

Jalankan:

```
python3 code/main.py
```

Output: per-pola trace + op count. Supervisor adalah cleanest; swarm adalah shortest; hierarchical adalah deepest; debat adalah sebagian besar expensive.

## Gunakan

- **LangGraph** untuk supervisor dan hierarchical (nested subgraphs).
- **OpenAI Agents SDK** untuk handoffs-sebagai-tool (supervisor-shaped).
- **CrewAI Flow** untuk produksi deterministik.
- **Custom** untuk debat atau ketika kamu inginkan exact kontrol.

## Rilis

`outputs/skill-orchestration-picker.md` picks sebuah topology dan implements itu.

## Latihan

1. Convert sebuah supervisor-worker ke sebuah swarm oleh removing router. Apa breaks? Apa meningkatkan?
2. Tambahkan sebuah hop counter ke swarm: refuse setelah 3 handoffs. Melakukan itu catch SEBUAH->B->SEBUAH bouncing?
3. Build sebuah dua-level hierarchical sistem untuk sebuah 12-specialist domain. Di mana melakukan context budget gagal tanpa nesting?
4. Profile empat polas pada sebuah produksi-shaped workload. Yang wins pada yang metrik (latency, cost, accuracy, debuggability)?
5. Baca Anthropic's "Building Efektif Agen" post. Map setiap dari kamu produksi flow ke satu dari empat. Apa pun itu don't map cleanly?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Supervisor-worker | "Router + specialists" | Central LLM dispatches to specialists; they don't talk to each other |
| Swarm | "Peer-to-peer" | Direct handoffs via shared tools; no central router |
| Hierarchical | "Supervisors of supervisors" | Nested subgraphs for large populations |
| Debate | "Proposer + critique" | Parallel proposers, cross-critique (Lesson 25) |
| Tool-call-based supervision | "Supervisor without a library" | Implement supervisor as direct tool calls for context control |
| Crew | "Autonomous team" | CrewAI's role-based collaboration mode |
| Flow | "Deterministic workflow" | CrewAI's event-driven production mode |

## Bacaan Lanjutan

- [Anthropic, Building Efektif Agen](https://www.anthropic.com/research/building-effective-agents) — lima polas + agen vs workflow
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor, swarm, hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Kru vs Flow
- [Du et al., Society dari Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debat pola
