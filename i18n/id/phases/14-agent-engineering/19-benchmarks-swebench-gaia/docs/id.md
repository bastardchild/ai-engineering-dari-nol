# Benchmark: SWE-bench, GAIA, AgentBench

> Tiga benchmark anchor agen evaluasi di 2026. SWE-bench tes kode patching. GAIA tes generalist tool pakai. AgentBench tes multi-environment penalaran. Know mereka composition, mereka contamination story, dan apa mereka lakukan tidak measure.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 06 (Tool Use)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Name SWE-bench's tes harness (FAIL_TO_PASS) dan jelaskan mengapa itu gates pada unit tes.
- Jelaskan mengapa SWE-bench Verified (OpenAI, 500 tugas) exists dan apa itu removes.
- Deskripsikan GAIA's design: sederhana untuk humans, hard untuk AI; tiga difficulty levels.
- Name AgentBench's eight environments dan miliknya primary blocker untuk buka-source LLMs.
- Summarize SWE-bench+ contamination finding dan miliknya implications.

## Masalahnya

Leaderboards tell kamu yang model wins pada satu benchmark. Mereka lakukan tidak tell kamu:

- Whether benchmark adalah contaminated (solutions di training data, tes leakage).
- Whether benchmark measures apa kamu care tentang (kode vs browsing vs generalist).
- Whether evaluator adalah robust (AST matching, state checks, manusia review).

Know tiga anchoring benchmark dan mereka mode kegagalan sebelum kamu quote sebuah number.

## Konsepnya

### SWE-bench (Jimenez et al., ICLR 2024 oral)

- 2,294 nyata GitHub issues dari 12 popular Python repo.
- Agen mendapatkan: codebase pada pre-fix commit + natural-language issue description.
- Agen produces: sebuah patch.
- Evaluator: apply patch, jalankan repo's tes suite. patch harus flip FAIL_TO_PASS tes (previously failing, now passing) tanpa breaking PASS_TO_PASS tes.

SWE-agen (Yang et al., 2024) hit 12.5% pada release oleh emphasizing agen-komputer interfaces (file editor commands, pencarian syntax model understands).

### SWE-bench Verified

OpenAI, Aug 2024. Manusia-curated 500-tugas subset. Removes ambiguous issues, unreliable tes, dan tugas di mana fix adalah unclear. Primary benchmark untuk "melakukan kamu agen ship nyata patches?"

### Contamination

- Atas 94% dari SWE-bench issues predate sebagian besar model cutoffs.
- **SWE-bench+** found 32.67% dari successful patches leaked solutions di issue text (model saw fix di description), dan 31.08% adalah suspicious due ke weak tes coverage.
- Verified adalah cleaner but tidak contamination-free.

Practical implication: sebuah model itu scores 50% pada SWE-bench dapat score 35% pada SWE-bench+. Selalu report keduanya jika kamu claim SWE-bench performa.

### GAIA (Mialon et al., Nov 2023)

- 466 questions; 300 retained untuk private leaderboard pada huggingface.co/gaia-benchmark.
- Design philosophy: "conceptually sederhana untuk humans (92%) but hard untuk AI (GPT-4 dengan plugins: 15%)."
- Tes penalaran, multi-modality, web, tool pakai.
- Tiga difficulty levels; Level 3 membutuhkan long tool chains lintas modalities.

GAIA adalah apa kamu jalankan ke measure "generalist kapabilitas." Lakukan tidak confuse dengan kode-specific benchmark.

### AgentBench (Liu et al., ICLR 2024)

- 8 environments lintas kode (Bash, DB, KG), games (Alfworld, LTP), web (WebShop, Mind2Web), dan buka-ended generation.
- Multi-giliran, ~4k-13k giliran per split.
- Primary finding: long-term penalaran, decision-making, dan instruction following adalah blockers untuk OSS LLMs catching up ke commercial.

### Apa ini lakukan tidak measure

- Dunia-nyata operasional cost (token, wall-clock).
- Keamanan perilaku di adversarial conditions.
- Performa pada kamu domain (pakai kamu own eval, Pelajaran 30).
- Tail kegagalan (benchmark average; produksi operators care tentang worst 1%).

### Di mana benchmarking goes salah

- **Single-number fixation.** SWE-bench 50% tells kamu less than P50/P75/P95 cost + langkah distribution.
- **Contaminated claims.** Reporting SWE-bench tanpa mentioning Verified atau SWE-bench+ adalah misleading.
- **Benchmark-as-development-target.** Optimizing untuk benchmark diverges dari produksi usefulness.

```figure
ae-swebench-gate
```

## Bangun Sendiri

`code/main.py` implements sebuah toy SWE-bench-like harness:

- Synthetic bug-fix tugas (3 tugas).
- SEBUAH scripted "agen" itu proposes patches.
- SEBUAH tes runner itu checks FAIL_TO_PASS (bug now fixed) dan PASS_TO_PASS (nothing broken).
- SEBUAH GAIA-style difficulty classifier based pada pertanyaan decomposition depth.

Jalankan:

```
python3 code/main.py
```

 output menunjukkan resolution rate per tugas + per difficulty dan membuat evaluator rules concrete.

## Gunakan

- **SWE-bench Verified** untuk kode agen. Selalu report Verified scores.
- **GAIA** untuk generalist agen. Pakai private leaderboard split.
- **AgentBench** untuk multi-environment comparison.
- **Custom evals** (Pelajaran 30) untuk kamu product's actual bentuk.

## Rilis

`outputs/skill-benchmark-harness.md` builds sebuah SWE-bench-style harness untuk apa pun codebase-tugas pair dengan FAIL_TO_PASS / PASS_TO_PASS gating.

## Latihan

1. Port toy harness ke jalankan pada sebuah nyata repo (pick satu dari milikmu). Tulis 3 FAIL_TO_PASS tes untuk known bugs.
2. Tambahkan sebuah langkah-count metrik. Pada kamu 3 tugas, bagaimana banyak agen langkah per resolution?
3. Baca SWE-bench+ paper. Implementasikan sebuah solution-leakage check (pola-match issue text terhadap diff).
4. Download sebuah GAIA pertanyaan dari public split. Trace apa sebuah GPT-4-class agen akan lakukan. Apa tool melakukan itu butuhkan?
5. Baca AgentBench's per-environment breakdown. Yang environment mirrors kamu product surface? Apa melakukan "SOTA" look like there?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SWE-bench | "Code agent benchmark" | 2,294 GitHub issues; patch must flip FAIL_TO_PASS tests |
| SWE-bench Verified | "Clean SWE-bench" | 500 human-curated tasks, OpenAI |
| FAIL_TO_PASS | "Fix gate" | Tests previously failing that must pass after the patch |
| PASS_TO_PASS | "No-regression gate" | Tests that were passing and must still pass |
| GAIA | "Generalist benchmark" | 466 human-easy / AI-hard multi-tool questions |
| AgentBench | "Multi-env benchmark" | 8 environments; long-horizon multi-turn |
| Contamination | "Training-set leak" | Benchmark tasks present in model training |
| SWE-bench+ | "Contamination audit" | 32.67% solution leakage found in successful SWE-bench patches |

## Bacaan Lanjutan

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — original benchmark
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated subset
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — generalist benchmark
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — multi-environment suite
