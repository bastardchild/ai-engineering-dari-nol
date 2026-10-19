# Perencanaan dengan HTN dan Evolutionary Pencarian

> Symbolic perencanaan menangani cases di mana rencana adalah provably benar. Evolutionary kode pencarian menangani cases di mana fitness function adalah machine-checkable. ChatHTN (2025) dan AlphaEvolve (2025) tunjukkan apa setiap unlocks ketika paired dengan sebuah LLM.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan Hierarchical Tugas Networks: tugas, methods, operators, preconditions, effects.
- Deskripsikan ChatHTN's hibrida loop — symbolic pencarian dengan LLM fallback decomposition.
- Jelaskan AlphaEvolve's evolutionary loop dan mengapa itu hanya bekerja dengan sebuah programmatic evaluator.
- Implementasikan sebuah toy HTN planner plus sebuah toy evolutionary pencarian di stdlib.

## Masalahnya

ReWOO (Pelajaran 02), Rencana-dan-Execute, dan ReAct cover sebagian besar agen perencanaan. Dua cases mereka don't cover well:

1. **Plans with provable correctness.** Scheduling, flight pathing, compliance workflow — rencana harus menjadi sound oleh construction. SEBUAH fluent LLM rencana itu kadang hallucinates sebuah langkah adalah unacceptable.
2. **Optimizations with a machine-checkable fitness function.** Matrix multiplication, scheduling heuristics, compiler passes — tujuan adalah tidak "sebuah benar rencana" but " best rencana."

HTN perencanaan dan AlphaEvolve solve dua berbeda problems. Keduanya pakai LLMs sebagai amplifiers, tidak Gantiments.

## Konsepnya

### Hierarchical Tugas Networks

Sebuah HTN adalah:

- **Tasks** — compound (ke menjadi decomposed) dan primitive (directly executable).
- **Methods** — ways ke decompose sebuah compound tugas menjadi subtasks, dengan preconditions.
- **Operators** — primitive aksi dengan preconditions dan effects.
- **State** — sebuah set dari facts.

Perencanaan: given sebuah tujuan tugas dan sebuah initial state, find sebuah decomposition menjadi primitive operators yang preconditions adalah satisfied di sequence.

HTN adalah older than LLMs dan tetap reference untuk provably-benar rencana.

### ChatHTN (Gopalakrishnan et al., 2025)

ChatHTN (arXiv:2505.11814) interleaves symbolic HTN dengan LLM queries:

1. Try ke decompose current compound tugas dengan existing methods.
2. Jika tidak metode applies, tanya LLM: "bagaimana akan kamu decompose `task` di state `s`?"
3. Translate LLM response menjadi candidate subtasks.
4. Validate terhadap operator schema; reject invalid decompositions.
5. Recurse.

 paper's pusat claim: setiap rencana produced adalah provably sound karena LLM suggestions hanya enter sebagai candidate decompositions, tidak pernah sebagai direct rencana edits. symbolic layer owns correctness; LLM expands metode library.

Online metode pembelajaran (OpenReview `gwYEDY9j2x`, 2025 follow-up) adds sebuah learner itu generalizes LLM-produced decompositions oleh regression — cutting LLM query frequency up ke 75%.

### AlphaEvolve (Novikov et al., 2025)

AlphaEvolve (arXiv:2506.13131, DeepMind, June 2025) adalah sebuah berbeda beast: evolutionary kode pencarian orchestrated oleh sebuah Gemini 2.0 Flash/Pro ensemble.

Loop:

1. Mulai dengan sebuah seed program + sebuah programmatic evaluator (mengembalikan sebuah fitness score).
2. Ensemble dari LLMs proposes mutations.
3. Jalankan mutations melalui evaluator.
4. Keep best; mutate again.

Published wins:

- Pertama improvement atas Strassen untuk 4x4 complex matrix multiplication di 56 years (48 scalar multiplications).
- 0.7% recovered Google compute via sebuah Borg scheduling heuristic.
- 32% FlashAttention speedup pada sebuah frontier workload.

 hard constraint: fitness function harus menjadi machine-checkable. Evolutionary pencarian atas prose menjawab melakukan tidak converge.

### Ketika ke pakai yang

| Problem class | Use | Why |
|---------------|-----|-----|
| Scheduling with hard constraints | HTN + ChatHTN | Provable soundness |
| Compiler optimization | AlphaEvolve | Machine-checkable fitness |
| Multi-step task execution | ReAct / ReWOO | LLM in the loop, no formal guarantees |
| Code improvement with tests | AlphaEvolve | Tests are the evaluator |
| Policy-bound automation | HTN | Preconditions encode policy |

### Di mana ini pola goes salah

- **HTN without operators.** Tanpa precondition/effect schemas soundness claim collapses. ChatHTN's "LLM suggests decomposition" membutuhkan schema ke reject invalid moves.
- **AlphaEvolve without a real evaluator.** "Tanya LLM jika kode adalah better" adalah tidak sebuah fitness function. evaluator harus menjadi deterministik dan cepat.
- **Over-engineering.** Sebagian besar agen tugas don't butuhkan either. Reach untuk ReAct atau ReWOO pertama.

```figure
htn-tree-expand
```

## Bangun Sendiri

`code/main.py` implements dua toys:

- SEBUAH stdlib HTN planner dengan operators, methods, preconditions, effects, dan sebuah `LLMFallback` itu kicks di ketika tidak metode matches sebuah compound tugas. "LLM" adalah sebuah scripted decomposer so planner berjalan offline.
- SEBUAH stdlib evolutionary pencarian atas arithmetic programs: grow expressions yang output minimizes `|f(x) - target|` atas sebuah tes set. Evaluator adalah deterministik.

Jalankan:

```
python3 code/main.py
```

 trace menunjukkan HTN planner decomposing sebuah compound tugas (dengan sebuah mid-rencana LLM fallback) dan evolutionary loop converging pada sebuah target expression.

## Gunakan

- **HTN planners** — `pyhop`, `SHOP3`, atau build kamu own untuk domain-specific kebijakan enforcement.
- **ChatHTN** — riset kode; pola (symbolic + LLM fallback) ports cleanly ke apa pun HTN planner.
- **AlphaEvolve** — DeepMind paper; pola (ensemble + evaluator) adalah reproducible. OpenEvolve dan similar buka-source forks adalah emerging.
- **Agent frameworks** — none ship pertama-class HTN atau AlphaEvolve yet. Build itu sebagai sebuah subagent atau sebuah background worker.

## Rilis

`outputs/skill-hybrid-planner.md` menghasilkan sebuah hibrida planner scaffold (HTN atau evolutionary) dengan LLM peran explicitly scoped.

## Latihan

1. Perluas HTN planner dengan backtracking: ketika sebuah operator's postcondition gagal pada runtime, roll back dan try berikut metode.
2. Tambahkan sebuah LLM-metode cache ke ChatHTN: ketika LLM decomposes tugas `T` di state pola `P`, store hasil. Re-check metode library pertama pada berikut panggil.
3. Tukar evolutionary pencarian evaluator ke sebuah nyata tes suite. Evolve sebuah sort function itu passes 20 tes cases; report generations ke convergence.
4. Baca AlphaEvolve's evaluator design notes. Design sebuah evaluator untuk sebuah domain kamu care tentang (SQL query optimization, tes-suite minimization, deployment YAML).
5. Combine: pakai HTN ke decompose sebuah compound tugas menjadi subtasks, maka pakai evolutionary pencarian pada setiap subtask's primitive operator. Di mana melakukan itu shine, di mana melakukan itu atas-engineer?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| HTN | "Hierarchical planner" | Task decomposition with operators, preconditions, effects |
| Method | "Decomposition rule" | Way to break a compound task into subtasks |
| Operator | "Primitive action" | Concrete step with precondition and effect |
| ChatHTN | "LLM + HTN" | Symbolic planner asks LLM when no method matches |
| AlphaEvolve | "Evolutionary code search" | Ensemble LLMs mutate code; deterministic evaluator selects |
| Fitness function | "Evaluator" | Deterministic, machine-checkable score over outputs |
| Online method learning | "Cached LLM decomposition" | Store + generalize LLM plans to cut query cost |

## Bacaan Lanjutan

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — symbolic + LLM hibrida planner
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — evolutionary kode pencarian dengan LLM mutations
- [Anthropic, Building Efektif Agen](https://www.anthropic.com/research/building-effective-agents) — ketika ke reach untuk sebuah planner vs sebuah sederhana loop
