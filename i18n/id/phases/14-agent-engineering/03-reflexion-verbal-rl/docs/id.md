# Reflexion: Verbal Reinforcement Pembelajaran

> Gradient-based RL membutuhkan thousands dari trials dan sebuah GPU cluster ke fix sebuah mode kegagalan. Reflexion (Shinn et al., NeurIPS 2023) melakukan itu di natural language: setelah setiap failed trial, agen itu menulis sebuah reflection, stores itu di episodic memori, dan conditions berikut trial pada itu memori. Ini adalah pola di balik Letta's sleep-time compute, Claude Kode's CLAUDE.md learnings, dan pro-workflow's pelajari-aturan.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Name tiga komponen dari Reflexion (Aktor, Evaluator, Mandiri-Reflector) dan peran dari episodic memori.
- Implementasikan sebuah stdlib Reflexion loop dengan binary evaluator, reflection buffer, dan fresh re-attempts.
- Choose antara scalar, heuristic, dan mandiri-evaluated feedback sources untuk sebuah given tugas.
- Jelaskan mengapa verbal reinforcement catches error itu gradient-based RL akan butuhkan thousands dari trials ke fix.

## Masalahnya

Sebuah agen gagal sebuah tugas. Di standard RL kamu akan jalankan thousands more trials, compute gradients, update weights. Expensive, lambat, dan sebagian besar produksi agen lakukan tidak memiliki sebuah training budget untuk setiap kegagalan.

Reflexion (Shinn et al., arXiv:2303.11366) bertanya sebuah berbeda pertanyaan: apa jika agen itu just thought tentang mengapa itu failed dan tried again dengan itu thought di miliknya prompt? Tidak weight updates. Tidak gradient. Just natural language stored antara trials.

 hasil: pada ALFWorld itu beats ReAct dan other non-fine-tuned baselines. Pada HotpotQA itu meningkatkan atas ReAct. Pada kode generation (HumanEval/MBPP) itu sets terkini pada time. Semua tanpa sebuah tunggal gradient langkah.

## Konsepnya

###  tiga komponen

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

Plus satu data structure:

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

Satu trial berjalan Aktor. Evaluator scores itu. Jika score adalah low, Mandiri-Reflector produces sebuah reflection ("I picked salah tool karena I misread pertanyaan sebagai asking tentang X ketika itu adalah asking tentang Y"). reflection goes menjadi episodic memori. Berikut trial memulai fresh but sees reflection.

### Tiga evaluator types

1. **Scalar** — sebuah eksternal binary signal. ALFWorld succeeds atau gagal. HumanEval tes pass atau gagal. Simplest, highest-signal.
2. **Heuristic** — predefined kegagalan signatures. "Jika agen itu produced sama aksi twice di sebuah row, mark sebagai stuck." "Jika trajectory exceeds 50 langkah, mark sebagai inefficient."
3. **Self-evaluated** — LLM scores miliknya own trajectory. Needed ketika tidak ground truth adalah available. Weaker signal; pairs well dengan tool-grounded verifikasi (Pelajaran 05 — CRITIC).

 2026 default adalah sebuah mix: scalar ketika available, mandiri-eval ketika tidak, heuristics sebagai keamanan rails.

### Mengapa ini generalizes

Reflexion adalah tidak sebuah baru algorithm so much sebagai sebuah named pola. Almost setiap produksi "mandiri-healing" agen berjalan beberapa variant:

- Letta's sleep-time compute (Pelajaran 08): sebuah terpisah agen reflects pada past conversations dan menulis ke memori blocks.
- Claude Kode's `CLAUDE.md` / "save memori" pola: reflections captured sebagai learnings, prepended ke future sesi.
- pro-workflow's `/learn-rule` command: corrections captured sebagai eksplisit rules.
- LangGraph's reflection node: sebuah node itu scores output dan routes ke refine jika needed.

Semua derive dari sama insight: natural language adalah sebuah rich-enough medium ke carry "apa I learned dari kegagalan" antara berjalan.

### Ketika itu bekerja dan ketika itu melakukan tidak

Reflexion bekerja ketika:

- There adalah sebuah clear kegagalan signal (tes kegagalan, tool error, salah jawaban).
-  tugas class adalah reproducible ( sama type dari pertanyaan bisa menjadi asked again).
-  reflection memiliki room ke tingkatkan pada trajectory (enough aksi budget).

Reflexion melakukan tidak bantu ketika:

- Agen itu sudah succeeds pada pertama try.
-  kegagalan adalah eksternal (network down, tool broken) — reflection pada " network adalah down" melakukan tidak bantu future berjalan.
-  reflection giliran menjadi superstition — storing sebuah narrative tentang sebuah satu-off flaky jalankan.

2026 pitfall: memori rot. Reflections accumulate; beberapa adalah obsolete atau salah; re-berjalan dapatkan slower sebagai episodic buffer yang terus bertambah. Mitigation: periodic compaction (Pelajaran 06), TTL pada reflections, atau sebuah terpisah sleep-time cleanup agen (Letta).

```figure
react-trace
```

## Bangun Sendiri

`code/main.py` implements Reflexion pada sebuah toy puzzle: produce sebuah 3-element list itu sums ke sebuah target. Aktor meng-emit candidate lists; Evaluator checks sum; Mandiri-Reflector menulis sebuah line tentang apa went salah. reflection goes menjadi episodic memori untuk berikut trial.

Komponen:

- `Actor` — sebuah scripted kebijakan itu meningkatkan ketika itu sees reflections.
- `Evaluator.binary()` — pass/gagal pada target sum.
- `SelfReflector` — menghasilkan sebuah satu-line diagnosis dari kegagalan.
- `EpisodicMemory` — sebuah bounded list dengan TTL semantics.

Jalankan:

```
python3 code/main.py
```

 trace menunjukkan tiga trials. Trial 1 gagal, sebuah reflection adalah stored, trial 2 sees reflection dan meningkatkan but tetap gagal, trial 3 succeeds. Bandingkan dengan sebuah baseline jalankan (tidak reflection) — itu stays stuck pada trial 1's jawaban.

## Gunakan

LangGraph ships reflection sebagai sebuah node pola. Claude Kode's `/memory` command dan pro-workflow's `/learn-rule` externalize episodic buffer sebagai sebuah markdown file. Letta's sleep-time compute berjalan Mandiri-Reflector pada downtime so primary agen stays latency-bound. OpenAI Agen SDK melakukan tidak ship Reflexion directly; kamu build itu dengan sebuah custom Guardrail itu rejects trajectories oleh score dan sebuah memori `Session` itu survives lintas berjalan.

## Rilis

`outputs/skill-reflexion-buffer.md` membuat dan maintains sebuah episodic buffer dengan reflection capture, TTL, dan deduplication. Given sebuah tugas class dan sebuah kegagalan, itu meng-emit sebuah reflection itu actually membantu berikut trial (tidak sebuah generic "menjadi more careful").

## Latihan

1. Switch dari binary ke scalar evaluator itu mengembalikan sebuah distance metrik (bagaimana far dari target). Melakukan itu converge faster?
2. Tambahkan sebuah TTL dari 10 trials ke reflections. Lakukan older reflections hurt atau bantu setelah itu inti?
3. Implementasikan heuristic evaluator: mark trial sebagai stuck jika sama aksi repeats. Bagaimana ini interact dengan Mandiri-Reflector?
4. Jalankan Reflexion dengan sebuah adversarial Aktor itu ignores reflections. Apa adalah minimum reflection prompt engineering itu forces Aktor ke notice mereka?
5. Baca Bagian 4 dari Reflexion paper pada AlfWorld. Reproduce 130% keberhasilan-rate improvement conceptually: apa adalah kunci delta vs vanilla ReAct?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Reflexion | "Self-correction" | Shinn et al. 2023 — Actor, Evaluator, Self-Reflector plus episodic memory |
| Verbal reinforcement | "Learning without gradients" | Natural-language reflection prepended to the next trial's prompt |
| Episodic memory | "Per-task reflections" | Bounded buffer of prior reflections for one task class |
| Scalar evaluator | "Binary success signal" | Pass/fail or numeric score from ground truth |
| Heuristic evaluator | "Pattern-based detector" | Predefined failure signatures (e.g. stuck-loop, too-many-steps) |
| Self-evaluator | "LLM-as-judge on own trace" | Lower-signal fallback when no ground truth — pair with tool-grounded verification |
| Memory rot | "Stale reflections" | Episodic buffer fills with obsolete entries; fix with compaction/TTL |
| Sleep-time reflection | "Async self-reflection" | Run Self-Reflector off the hot path so primary agent stays fast |

## Bacaan Lanjutan

- [Shinn et al., Reflexion: Language Agen dengan Verbal Reinforcement Pembelajaran (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — paper kanonikal
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — async reflection di produksi
- [Anthropic, Efektif context engineering untuk AI agen](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — managing episodic buffer sebagai bagian dari context
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — reflection node pola
