# Multi-agen Debat dan Collaboration

> Du et al. (ICML 2024, "Society dari Minds") jalankan N model instances itu independently propose menjawab, maka iteratively critique setiap other atas R rounds ke converge. Meningkatkan factuality, aturan-following, penalaran. Sparse topology beats penuh mesh pada token cost.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 05 (Self-Refine and CRITIC)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan debat protokol: N proposers, R rounds, converge pada sebuah shared jawaban.
- Deskripsikan mengapa debat meningkatkan factuality, aturan-following, dan penalaran.
- Jelaskan sparse topology: tidak setiap debater membutuhkan ke see setiap other.
- Implementasikan sebuah stdlib debat atas sebuah scripted LLM dengan penuh-mesh dan sparse variants; measure token cost vs accuracy.

## Masalahnya

Mandiri-Refine (Pelajaran 05) adalah satu model critiquing itself — risiko groupthink. CRITIC (Pelajaran 05) grounds critique di eksternal tool — tidak selalu available. Debat introduces sebuah third mode: multiple instances, cross-critique, convergence oleh disagreement.

## Konsepnya

### Society dari Minds (Du et al., ICML 2024)

- N model instances independently propose menjawab ke sama pertanyaan.
- Atas R rounds, setiap model membaca others' proposals dan critiques mereka.
- Model update mereka menjawab based pada critiques.
- Setelah R rounds, kembalikan convergent jawaban.

Original experiments used N=3, R=2 due ke cost. Accuracy meningkatkan dengan more agen dan more rounds pada hard problems (MMLU, GSM8K, Chess Move Validity, biography generation).

Cross-model combinations beat tunggal-model debates: ChatGPT + Bard together > either alone.

### Sparse topology

"Improving Multi-agen Debat dengan Sparse Communication Topology" (arXiv:2406.11776, 2024-2025) showed penuh-mesh debat adalah tidak selalu optimal. Sparse topologies (star, ring, hub-dan-spoke) bisa match accuracy pada lower token cost. Setiap debater sees hanya sebuah subset dari peers.

Implications:

- Penuh mesh N=5, R=3 = 5 × 3 = 15 proposals, setiap reading 4 peers = 60 critique ops.
- Star N=5, R=3 (satu hub + 4 spokes) = 15 proposals, spokes baca hanya hub = 12 critique ops.

### Ketika debat membantu

- **Factuality.** N independent proposals, cross-check mengurangi hallucination.
- **Rule-following.** Chess move validity — satu model misses sebuah aturan, others catch itu.
- **Open-ended reasoning.** Multiple framings narrow di pada right jawaban.

### Ketika debat hurts

- **Latency-sensitive UX.** N × R serial rounds adalah latency kamu dapat tidak memiliki.
- **Cost-sensitive scale.** N × R token per pertanyaan.
- **Simple factual lookups.** Satu lookup adalah cheaper than lima debates.

### 2026 practical instantiations

- **Anthropic orchestrator-workers** (Pelajaran 12) — satu variant dari debat dengan sebuah synthesis langkah.
- **LangGraph supervisor** (Pelajaran 13) — pusat router + specialist agen bisa implementasikan debat sebagai sebuah node.
- **OpenAI Agents SDK** (Pelajaran 16) — agen handoff back dan forth untuk iterative critique.
- **Multi-agent evals** — pair debat + evaluator-optimizer untuk eval signal.

### Di mana ini pola goes salah

- **Convergence collapse.** Semua agen converge pada pertama salah jawaban. Mitigate dengan required disagreement rounds.
- **Hub failure.** Di sebuah star topology, sebuah buruk hub corrupts everyone. Rotate atau pakai multiple hubs.
- **Prompt homogenization.** Semua agen pakai sama prompt; mereka produce sama menjawab. Pakai diverse prompts dan/atau model.

```figure
debate-converge
```

## Bangun Sendiri

`code/main.py` implements stdlib debat:

- `Debater` class (scripted LLM dengan per-debater opinion drift).
- `FullMeshDebate` dan `SparseDebate` runners.
- Tiga questions: satu factual, satu aturan-based, satu penalaran.
- Metrik: convergent jawaban, rounds ke convergence, total critique ops.

Jalankan:

```
python3 code/main.py
```

Output: per-protokol accuracy dan cost; sparse matches penuh mesh pada 2/3 questions pada lower cost.

## Gunakan

- **Anthropic orchestrator-workers** untuk sederhana 2-3-worker debates.
- **LangGraph** untuk stateful multi-round debat dengan checkpointing.
- **Custom** untuk riset atau specialized correctness guarantees.

## Rilis

`outputs/skill-debate.md` scaffolds sebuah multi-agen debat dengan configurable topology, N, R, dan sebuah convergence aturan.

## Latihan

1. Implementasikan sebuah "forced disagreement" aturan: di round 1, setiap debater harus produce sebuah distinct proposal. Measure effect pada convergence speed.
2. Tambahkan sebuah confidence-weighted aggregation: debaters kembalikan (jawaban, confidence); aggregator weights oleh confidence. Melakukan itu bantu?
3. Swap satu "agen" untuk sebuah berbeda scripted LLM dengan berbeda opinions. Melakukan heterogeneity tingkatkan accuracy?
4. Measure token cost untuk penuh mesh vs sparse pada kamu 3 questions. Plot cost vs accuracy.
5. Baca Society dari Minds paper. Port kamu toy ke N=5, R=3. Apa breaks? Apa mendapatkan better?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Debate | "Multi-agent critique" | N proposers, R rounds of cross-critique, converge |
| Full mesh | "Everyone reads everyone" | Every debater reads every peer each round |
| Sparse topology | "Limited peer view" | Debaters read only a subset of peers |
| Hub-and-spoke | "Star topology" | One central debater, N-1 spokes read only the hub |
| Convergence | "Agreement" | Debaters converge on a shared answer |
| Society of Minds | "Du et al. debate paper" | ICML 2024 multi-agent debate method |

## Bacaan Lanjutan

- [Du et al., Society dari Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — kanonikal multi-agen debat
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — sparse topology hasil
- [Anthropic, Building Efektif Agen](https://www.anthropic.com/research/building-effective-agents) — orchestrator-workers sebagai sebuah debat variant
- [Madaan et al., Mandiri-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — tunggal-model mandiri-critique counterpart
