# Eval-Driven Agen Development

> Anthropic's guidance: "mulai dengan sederhana prompts, optimize mereka dengan comprehensive evaluasi, dan add multi-langkah agentic sistem hanya ketika needed." Evaluasi adalah tidak terakhir langkah. Itu's outer loop itu drives setiap other choice di Fase 14.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** All of Phase 14.
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Name tiga evaluasi layers — static benchmark, custom offline, online produksi — dan apa setiap adalah untuk.
- Jelaskan evaluator-optimizer tight loop.
- Deskripsikan 2026 best practice: eval live berikut ke kode, jalankan di CI, gate PRs.
- Connect setiap Fase 14 pelajaran ke eval case itu menghasilkan.

## Masalahnya

Agen pass demos. Mereka gagal di produksi di ways demos cannot predict. Benchmark jawaban "adalah ini model broadly capable?" tidak "adalah ini agen shipping right patches untuk my product?" jawaban: evaluasi pada tiga layers, berjalan continuously, dengan setiap guardrail dan learned aturan mapped ke sebuah eval case.

## Konsepnya

### Tiga evaluasi layers

1. **Static benchmarks** — SWE-bench Verified untuk kode (Pelajaran 19), WebArena/OSWorld untuk browsing / desktop (Pelajaran 20), GAIA untuk generalist (Pelajaran 19), BFCL V4 untuk tool pakai (Pelajaran 06). Pakai untuk cross-model comparison dan regression gating. Contamination adalah nyata: SWE-bench+ found 32.67% solution leakage. Selalu report Verified / +-audited scores.

2. **Custom offline evals** — kamu product's bentuk:
   - LLM-sebagai-judge (Langfuse, Phoenix, Opik — Pelajaran 24).
   - Execution-based (jalankan patch, check tes).
   - Trajectory-based (bandingkan aksi sequences terhadap gold; OSWorld-Manusia menunjukkan top agen 1.4-2.7x atas gold).

3. **Online evals** — produksi:
   - Sesi replays (Langfuse).
   - Guardrail-triggered alerts (Pelajaran 16, 21).
   - Per-langkah cost / latency tracking (Pelajaran 23 OTel spans).

### Evaluator-optimizer (Anthropic)

 tight loop:

1. Proposer menghasilkan output.
2. Evaluator judges.
3. Refine sampai evaluator passes.

Ini adalah Mandiri-Refine (Pelajaran 05) generalized. Apa pun agen flow kamu care tentang bisa wrap di evaluator-optimizer untuk reliability.

### 2026 best practice

- Eval live berikut ke kode.
- Jalankan di CI pada setiap PR.
- Gate merge pada eval scores (e.g. "tidak regression > 5% vs utama").
- Setiap guardrail maps ke sebuah eval case.
- Setiap learned aturan (Reflexion, pro-workflow pelajari-aturan) maps ke sebuah kegagalan case.

### Tying Fase 14 together

Setiap pelajaran di Fase 14 menghasilkan eval cases:

| Lesson | Eval case it generates |
|--------|------------------------|
| 01 Agent Loop | Budget-exhausted, infinite-loop guard |
| 02 ReWOO | Planner replans correctly when a tool fails |
| 03 Reflexion | Learned reflections apply on retry |
| 05 Self-Refine/CRITIC | Judge passes refined output |
| 06 Tool Use | Argument coercion works; unknown tools rejected |
| 07-10 Memory | Retrieval citations match sources; stale facts invalidate |
| 12 Workflow Patterns | Each pattern produces correct output |
| 13 LangGraph | Resume reproduces state exactly |
| 14 AutoGen Actors | DLQ catches crashed handlers |
| 16 OpenAI Agents SDK | Guardrail trips on the right inputs |
| 17 Claude Agent SDK | Subagent results return to orchestrator |
| 19-20 Benchmarks | SWE-bench Verified score, WebArena success rate, OSWorld efficiency |
| 21 Computer Use | Per-step safety catches injected DOM |
| 23 OTel | Spans emit required attributes |
| 26 Failure Modes | Detectors tag known failures |
| 27 Prompt Injection | PVE refuses poisoned retrievals |
| 28 Orchestration | Supervisor routes to the right specialist |
| 29 Runtime Shapes | DLQ handles N% failure |

Jika kamu eval suite memiliki cases untuk setiap, kamu punya covered Fase 14.

### Di mana eval-driven development gagal

- **No baseline.** Eval tanpa sebuah terakhir-known-baik adalah unreadable. Store baselines.
- **LLM-judge without grounding.** Judges hallucinate too. CRITIC pola (Pelajaran 05) — judge grounds pada eksternal tool.
- **Over-fitting to evals.** Optimizing untuk eval diverges dari produksi usefulness. Rotate cases.
- **Flaky evals.** Non-deterministik cases cause false alarms. Pin seeds, snapshot state.

```figure
ae-eval-three-layers
```

## Bangun Sendiri

`code/main.py` adalah sebuah stdlib eval harness:

- Case registry dengan categories (benchmark, custom, online).
- SEBUAH scripted agen di bawah tes.
- Evaluator-optimizer loop: propose, judge, refine sampai pass atau max rounds.
- CI gate: aggregate pass rate + regression terhadap baseline.

Jalankan:

```
python3 code/main.py
```

Output: per-case pass/gagal, regression flag, CI gate verdict.

## Gunakan

- Tulis eval cases di sama repo sebagai kamu agen kode.
- Jalankan mereka pada setiap PR via CI.
- Gagal build pada regression.
- Lacak pass rate atas time.
- Tie setiap produksi kegagalan ke sebuah baru case.

## Rilis

`outputs/skill-eval-suite.md` builds sebuah tiga-layer eval suite untuk sebuah agen product dengan CI gates dan regression tracking.

## Latihan

1. Ambil satu dari kamu produksi kegagalan. Tulis sebuah eval case itu reproduces itu. Melakukan kamu agen pass itu now?
2. Build sebuah LLM-judge rubric untuk kamu domain dengan tiga dimensions (factual, tone, scope). Score 50 sesi.
3. Wire eval suite menjadi CI. Gagal build pada >=5% regression.
4. Tambahkan sebuah trajectory-efficiency metrik: bagaimana banyak langkah melakukan agen itu ambil vs sebuah gold trajectory?
5. Map setiap Fase 14 pelajaran ke sebuah eval case di kamu suite. Apa pun missing? Itu's sebuah gap ke tutup.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Static benchmark | "Off-the-shelf eval" | SWE-bench, GAIA, AgentBench, WebArena, OSWorld |
| Custom offline eval | "Domain eval" | LLM-as-judge / exec / trajectory on your product shape |
| Online eval | "Production eval" | Session replay, guardrail alerts, cost/latency tracking |
| Evaluator-optimizer | "Propose-judge-refine" | Iterate until judge passes |
| CI gate | "Merge blocker" | Fail the build on eval regression |
| Baseline | "Last-known-good" | Reference score to detect regression |
| Trajectory efficiency | "Steps over gold" | Agent step count divided by human expert minimum |

## Bacaan Lanjutan

- [Anthropic, Building Efektif Agen](https://www.anthropic.com/research/building-effective-agents) — "mulai sederhana, optimize dengan eval"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated benchmark
- [Berkeley Function Pemanggilan Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — tool-pakai benchmark
- [Langfuse docs](https://langfuse.com/) — eval + sesi replay dalam praktik
