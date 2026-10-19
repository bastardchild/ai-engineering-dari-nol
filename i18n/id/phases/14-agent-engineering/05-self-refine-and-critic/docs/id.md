# Mandiri-Refine dan CRITIC: Iterative Output Improvement

> Mandiri-Refine (Madaan et al., 2023) memakai satu LLM di tiga roles — hasilkan, feedback, refine — di sebuah loop. Average gain: +20 absolute pada 7 tugas. CRITIC (Gou et al., 2023) hardens feedback langkah oleh routing verifikasi melalui eksternal tool. Di 2026 ini pola ships di setiap framework sebagai "evaluator-optimizer" (Anthropic) atau sebuah guardrail loop (OpenAI Agen SDK).

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- State Mandiri-Refine's tiga prompts (hasilkan, feedback, refine) dan jelaskan mengapa history matters untuk refine prompt.
- Jelaskan CRITIC's kritis insight: LLMs adalah unreliable pada mandiri-verifikasi tanpa eksternal grounding.
- Implementasikan sebuah stdlib Mandiri-Refine loop dengan history dan sebuah optional eksternal verifier.
- Map ini pola ke Anthropic's "evaluator-optimizer" workflow dan OpenAI Agen SDK's output guardrail.

## Masalahnya

Sebuah agen produces sebuah jawaban itu adalah almost right. Maybe sebuah line dari kode memiliki sebuah syntax error. Maybe sebuah summary adalah too long. Maybe sebuah rencana misses sebuah edge case. Apa kamu inginkan adalah: agen itu critiques miliknya own output, maka fixes itu.

Mandiri-Refine menunjukkan ini bekerja dengan sebuah tunggal model, tidak training data, tidak RL. But there adalah sebuah catch: LLMs adalah buruk pada mandiri-verifikasi pada hard facts. CRITIC names fix — route verifikasi langkah melalui eksternal tool (pencarian, kode interpreter, calculator, tes runner).

Together ini dua papers definisikan 2026 default untuk iterative improvement: hasilkan, verifikasi (externally ketika possible), refine, berhenti ketika verifier passes.

## Konsepnya

### Mandiri-Refine (Madaan et al., NeurIPS 2023)

Satu LLM, tiga roles:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

Kunci detail: `refine` sees penuh history — semua prior outputs dan critiques — so itu melakukan tidak repeat mistakes. paper ablates ini: drop history dan quality drops sharply.

Headline: +20 absolute improvement averaged lintas 7 tugas (math, kode, acronym, dialog) termasuk GPT-4. Tidak training, tidak eksternal tool, tunggal model.

### CRITIC (Gou et al., arXiv:2305.11738, v4 Feb 2024)

Mandiri-Refine's weakness: feedback langkah adalah sebuah LLM scoring itself. Untuk factual claims ini adalah unreliable (sebuah hallucination sering looks convincing ke model itu produced itu). CRITIC Gantis `feedback(task, output)` dengan `verify(task, output, tools)` di mana `tools` mencakup:

- SEBUAH pencarian engine untuk factual claims.
- SEBUAH kode interpreter untuk kode correctness.
- SEBUAH calculator untuk arithmetic.
- Domain-specific verifiers (unit tes, type checkers, linters).

 verifier produces sebuah structured critique grounded di tool hasil. refiner maka conditions pada ini critique.

Headline: CRITIC outperforms Mandiri-Refine pada factual tugas karena critique adalah grounded. Pada tugas tanpa eksternal verifiers (creative writing, formatting), CRITIC mengurangi ke Mandiri-Refine.

###  kondisi berhenti

Dua umum shapes:

1. **Verifier passes.** Eksternal tes mengembalikan keberhasilan. Preferred ketika available (unit tes, type checker, guardrail assertion).
2. **No feedback issued.** Model menyatakan " output adalah fine." Cheaper but unreliable; pair dengan sebuah max-iteration cap.

2026 default: combine mereka. "Berhenti jika verifier passes ATAU model menyatakan fine DAN iterations >= 2 ATAU iterations >= max_iterations."

### Evaluator-Optimizer (Anthropic, 2024)

Anthropic's Dec 2024 post names ini sebagai satu dari lima workflow polas. Dua roles:

- Evaluator: scores output dan produces sebuah critique.
- Optimizer: revises output given critique.

Loop sampai evaluator passes. Ini adalah Mandiri-Refine/CRITIC di Anthropic's framing. kritis engineering detail Anthropic adds: evaluator dan optimizer prompts sebaiknya menjadi substantially berbeda so model melakukan tidak just rubber-stamp.

### OpenAI Agen SDK output guardrail

OpenAI Agen SDK ships ini pola sebagai "output guardrail." SEBUAH guardrail adalah sebuah validator itu berjalan pada final output dari sebuah agen. Jika pemicu guardrail (raises `OutputGuardrailTripwireTriggered`), output adalah rejected dan agen itu bisa retry. Guardrail bisa panggil tool (CRITIC-style) atau menjadi pure functions (Mandiri-Refine-style).

### 2026 pitfalls

- **Rubber-stamp loops.** Sama model doing generation dan critique dengan sama prompt style converges pada "looks baik ke me." Pakai structurally berbeda prompts, atau sebuah smaller cheap model untuk critique.
- **Over-refinement.** Setiap refine pass adds latency dan token. Budget 1-3 passes; setelah itu, escalate ke manusia review.
- **CRITIC on trivial tasks.** Jika there adalah tidak eksternal verifier, CRITIC degenerates ke Mandiri-Refine; lakukan tidak pay latency untuk sebuah stub verifier.

```figure
self-refine
```

## Bangun Sendiri

`code/main.py` implements Mandiri-Refine dan CRITIC pada sebuah toy tugas: produce sebuah short bullet list given sebuah topic. verifier checks format (3 bullets, setiap di bawah 60 chars). CRITIC adds sebuah eksternal "fact verifier" itu penalizes known hallucinations.

Komponen:

- `generate` — scripted producer.
- `feedback` — LLM-style mandiri-critique.
- `verify_external` — CRITIC-style grounded verifier.
- `refine` — rewrites output given history.
- Kondisi berhenti — verifier passes atau max 4 iterations.

Jalankan:

```
python3 code/main.py
```

Bandingkan Mandiri-Refine vs CRITIC berjalan. CRITIC catches sebuah factual error Mandiri-Refine missed karena eksternal verifier memiliki grounding mandiri-kritik melakukan tidak.

## Gunakan

Anthropic's evaluator-optimizer adalah ini pola di Claude-friendly language. OpenAI Agen SDK's output guardrail adalah CRITIC-shaped (guardrail bisa panggil tool). LangGraph ships sebuah reflection node itu membaca like Mandiri-Refine. Google's Gemini 2.5 Komputer Pakai adds sebuah per-langkah keamanan evaluator itu adalah sebuah CRITIC variant: setiap aksi adalah verified sebelum commit.

## Rilis

`outputs/skill-refine-loop.md` configures sebuah evaluator-optimizer loop given tugas bentuk, verifier availability, dan iteration budget. Meng-emit prompts untuk generator, evaluator/verifier, dan optimizer, plus sebuah berhenti kebijakan.

## Latihan

1. Jalankan toy dengan max_iterations=1. Melakukan CRITIC tetap bantu?
2. Ganti eksternal verifier dengan sebuah noisy satu (random 30% false positives). Apa melakukan loop lakukan? Ini adalah 2026 reality dari sebagian besar guardrail stacks.
3. Implementasikan sebuah "generator-kritik pada berbeda model" variant: besar model menghasilkan, kecil model critiques. Melakukan itu beat sama-model?
4. Baca CRITIC Bagian 3 (arXiv:2305.11738 v4). Name tiga verifikasi-tool categories dan beri sebuah contoh untuk setiap.
5. Map OpenAI Agen SDK's `output_guardrails` ke CRITIC's verifier peran. Apa melakukan SDK dapatkan salah, dan apa melakukan itu dapatkan right?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Self-Refine | "LLM that fixes itself" | Generate -> feedback -> refine loop in one model, with history |
| CRITIC | "Tool-grounded verification" | Replace feedback with an external verifier (search, code, calc, tests) |
| Evaluator-Optimizer | "Anthropic workflow pattern" | Two roles — evaluator scores, optimizer revises — looped to convergence |
| Output guardrail | "Post-hoc check" | OpenAI Agents SDK validator that runs after an agent produces output |
| Verify step | "Critique phase" | The load-bearing decision: grounded or self-rated |
| Refine history | "What the model already tried" | Prior outputs + critiques prepended to refine prompt; drop and quality collapses |
| Rubber-stamp loop | "Self-agreement failure" | Same-prompt critique returns "looks good"; fix with structurally different prompts |
| Stop condition | "Convergence test" | Verifier passes OR no feedback AND iteration cap; never single-condition |

## Bacaan Lanjutan

- [Madaan et al., Mandiri-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — paper kanonikal
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — tool-grounded verifikasi
- [Anthropic, Building Efektif Agen](https://www.anthropic.com/research/building-effective-agents) — evaluator-optimizer workflow pola
- [OpenAI Agen SDK docs](https://openai.github.io/openai-agents-python/) — output guardrail sebagai CRITIC-shaped verifiers
