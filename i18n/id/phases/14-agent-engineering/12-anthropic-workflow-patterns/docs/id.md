# Anthropic's Workflow Polas: Sederhana Atas Complex

> Schluntz dan Zhang (Anthropic, Dec 2024) distinguish workflow (predefined paths) dari agen (dynamic tool-pakai). Lima workflow polas cover sebagian besar cases. Mulai dengan direct API memanggil. Tambahkan sebuahgents hanya ketika langkah cannot menjadi predicted.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Name Anthropic's lima workflow polas: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer.
- Jelaskan agen itu-vs-workflow distinction dan engineering cost dari setiap.
- Identifikasi ketika ke pick sebuah workflow atas sebuah agen (dan vice versa).
- Implementasikan sebuahll lima polas di stdlib terhadap sebuah scripted LLM.

## Masalahnya

Teams reach untuk multi-agen framework untuk problems itu inginkan sebuah tunggal function panggil. cost adalah nyata: framework add layers itu obscure prompts, hide alur kontrol, dan invite premature complexity. Schluntz dan Zhang's Dec 2024 post adalah sebagian besar-cited industry pushback: mulai sederhana, add complexity hanya ketika itu earns miliknya cost.

## Konsepnya

### Workflow vs agen

- **Workflow.** LLMs dan tool orchestrated melalui predefined kode paths. Engineers own graf.
- **Agent.** LLMs dynamically direct mereka own tool dan ambil mereka own langkah. model owns graf.

Keduanya memiliki mereka place. Workflow adalah cheaper, faster, dan easier ke debug. Agen unlock buka-ended problems but buat mode kegagalan harder ke reason tentang.

###  augmented LLM

Foundation untuk semua lima polas: satu LLM dengan tiga kapabilitas wired di — pencarian (retrieval), tool (aksi), memori (persistence). Apa pun API panggil bisa pakai ini.

###  lima polas

1. **Prompt chaining.** Output dari panggil 1 adalah input ke panggil 2. Pakai ketika sebuah tugas memiliki sebuah clean linear decomposition. Optional programmatic gates antara langkah.

2. **Routing.** SEBUAH classifier LLM picks yang downstream LLM atau tool ke invoke. Pakai ketika categorically berbeda inputs butuhkan berbeda handling (tier-1 dukung vs refund vs bug vs sales).

3. **Parallelization.** Jalankan N LLM memanggil concurrently, aggregate hasil. Dua shapes: sectioning (berbeda chunks) dan voting (sama prompt, N berjalan, majority/synthesis).

4. **Orchestrator-workers.** Sebuah orchestrator LLM dynamically memutuskan yang workers (juga LLMs) ke jalankan dan synthesizes mereka output. Similar ke agen loop but orchestrator melakukan tidak loop indefinitely.

5. **Evaluator-optimizer.** Satu LLM proposes sebuah jawaban, another LLM evaluates itu. Iterate sampai evaluator passes. Ini adalah Mandiri-Refine (Pelajaran 05) generalized.

### Di mana workflow beat agen

- **Predictable tasks.** Jika kamu bisa enumerate langkah, kamu sebaiknya.
- **Cost-bound tasks.** Workflow memiliki bounded langkah counts; agen bisa spiral.
- **Compliance-bound tasks.** Auditors inginkan ke baca graf, tidak infer itu dari trajectories.

### Di mana agen beat workflow

- **Open-ended research.** Ketika berikut langkah depends pada apa terakhir langkah returned.
- **Variable-length tasks.** Minutes ke hours dari kerja di mana langkah count adalah unknown.
- **Novel domains.** Ketika kamu don't yet know right workflow — exploration pertama, codify later.

###  context-engineering companion

"Efektif context engineering untuk AI agen" (Anthropic 2025) formalizes adjacent discipline: 200k window adalah sebuah budget, tidak sebuah container. Apa ke sertakan, ketika ke compact, ketika ke let context grow. Covered di detail di Fase 14 pelajaran pada context compression (Fase 14 earlier pelajaran 06 di ini curriculum sebelum renumber).

```figure
workflow-chain
```

## Bangun Sendiri

`code/main.py` implements semua lima workflow polas terhadap sebuah `ScriptedLLM`:

- `prompt_chain(input, steps)` — sequential.
- `route(input, classifier, handlers)` — classification + dispatch.
- `parallel_vote(prompt, n, aggregator)` — N berjalan, aggregate.
- `orchestrator_workers(task, workers)` — orchestrator picks workers.
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — loop sampai pass.

Jalankan:

```
python3 code/main.py
```

Setiap pola prints miliknya trace. Total lines dari kode per pola adalah ~10-15; cost dari sebuah framework adalah measured di thousands.

## Gunakan

- Direct API memanggil untuk sebagian besar tugas.
- Framework hanya ketika pola genuinely membutuhkan durable state (LangGraph), aktor-model concurrency (AutoGen v0.4), atau peran templating (CrewAI).
- Reach untuk Claude Agen SDK ketika kamu inginkan Claude Kode harness bentuk tanpa rebuilding itu.

## Rilis

`outputs/skill-workflow-picker.md` picks right pola untuk sebuah given tugas description, termasuk decision rationale dan refactor path ke sebuah agen jika workflow fall short.

## Latihan

1. Implementasikan routing dengan sebuah confidence threshold. Di bawah threshold -> escalate ke manusia. Di mana melakukan threshold land untuk sebuah tier-1 dukung pakai case?
2. Tambahkan sebuah timeout ke `parallel_vote`. Apa happens ketika satu panggil hangs? Bagaimana lakukan kamu aggregate dengan missing votes?
3. Giliran `evaluator_optimizer` menjadi sebuah bandit: keep top-2 outputs lintas iterations so sebuah late baik hasil doesn't dapatkan overwritten oleh sebuah late buruk satu.
4. Combine prompt chaining dengan routing: sebuah router picks satu dari tiga chains. Measure token cost vs sebuah tunggal besar-prompt alternative.
5. Pick satu dari kamu produksi features. Draw workflow graf. Count langkah. Akan sebuah agen actually menjadi better here?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Workflow | "Predefined flow" | Engineer-owned graph of LLM and tool calls |
| Agent | "Autonomous AI" | Model-owned graph; dynamic tool direction |
| Augmented LLM | "LLM with tools" | LLM + search + tools + memory; the atomic unit |
| Prompt chaining | "Sequential calls" | Output of call N is input to call N+1 |
| Routing | "Classifier dispatch" | Pick which chain/model handles the input |
| Parallelization | "Fan out" | N concurrent calls; aggregate by sectioning or voting |
| Orchestrator-workers | "Dispatcher agent" | Orchestrator LLM picks specialist LLMs dynamically |
| Evaluator-optimizer | "Proposer + judge" | Iterate until evaluator passes; Self-Refine generalized |

## Bacaan Lanjutan

- [Anthropic, Building Efektif Agen (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — lima workflow polas
- [Anthropic, Efektif context engineering untuk AI agen](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — companion discipline
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — ketika stateful graphs earn mereka cost
- [OpenAI Agen SDK](https://openai.github.io/openai-agents-python/) — orchestrator-workers pola, productized
