# Tree dari Thought dan LATS: Deliberate Pencarian

> SEBUAH tunggal chain-dari-thought trajectory memiliki tidak room ke backtrack. ToT (Yao et al., 2023) giliran penalaran menjadi sebuah tree dengan mandiri-evaluasi pada setiap node. LATS (Zhou et al., 2024) unifies ToT dengan ReAct dan Reflexion di bawah Monte Carlo Tree Pencarian. Game dari 24 goes dari 4% (CoT) ke 74% (ToT); LATS hits 92.7% pass@1 pada HumanEval.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Frame penalaran sebagai pencarian: node adalah "thought," edges adalah "expansions," value adalah "bagaimana promising."
- Implementasikan sebuah stdlib ToT-style BFS tree pencarian dengan mandiri-evaluasi scoring.
- Perluas ke sebuah toy LATS MCTS loop dengan select / expand / simulate / backpropagate.
- Putuskan ketika pencarian adalah worth token multiplier (Game dari 24, kode generation) dan ketika sebuah tunggal trajectory adalah enough (sederhana Q&SEBUAH).

## Masalahnya

Chain-dari-thought adalah sebuah linear walk. Jika pertama langkah adalah salah, setiap subsequent langkah bekerja pada sebuah buruk premise. Pada Game dari 24 (pakai empat digits dengan + − × ÷ ke buat 24), GPT-4 CoT hits 4% accuracy. model picks salah subexpression early dan cannot pulihkan.

Apa penalaran membutuhkan adalah ability ke propose multiple candidates, evaluate mereka, pick promising ones, dan backtrack ketika dead berakhir appear. Itu adalah pencarian. Tree dari Thought dan LATS adalah dua kanonikal formulations.

## Konsepnya

### Tree dari Thought (Yao et al., NeurIPS 2023)

Setiap node adalah sebuah coherent intermediate langkah ("sebuah thought"). Setiap node bisa expand ke K child thought. LLM mandiri-evaluates setiap node dengan sebuah scoring prompt. Pencarian explores tree — BFS, DFS, atau beam.

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

Mandiri-evaluasi adalah penopang beban piece. paper menunjukkan tiga variants: `sure / likely / impossible` classification, `1..10` numeric score, dan vote di antara candidates. Semua tiga beat CoT substantially pada Game dari 24 (4% -> 74% dengan GPT-4).

### LATS (Zhou et al., ICML 2024)

LATS unifies ToT, ReAct, dan Reflexion di bawah MCTS. LLM plays tiga roles:

- **Policy**: propose candidate berikut aksi (ReAct-style).
- **Value function**: score sebuah partial trajectory (ToT-style mandiri-eval).
- **Self-reflector**: pada kegagalan, tulis sebuah natural-language reflection (Reflexion-style) dan pakai itu ke reseed future rollouts.

Environment feedback (observasi) mixes menjadi value function so pencarian adalah informed oleh nyata tool hasil, tidak just model opinions. Hasil pada paper time: HumanEval pass@1 92.7% dengan GPT-4 (SOTA), WebShop average 75.9 dengan GPT-3.5 (approaching gradient-based fine-tuning).

### MCTS, minimally

Empat fase per iteration:

1. **Select** — walk dari root ke sebuah leaf memakai UCT (upper confidence bound untuk trees).
2. **Expand** — hasilkan K children via kebijakan.
3. **Simulate** — rollout dari sebuah child memakai kebijakan, score leaf dengan value function (atau environment reward).
4. **Backpropagate** — update visit counts dan value estimates up path.

UCT formula: `Q(s, a) + c * sqrt(ln N(s) / N(s, a))`. Pertama term adalah exploitation; kedua adalah exploration. Tune `c` per tugas.

###  cost reality

Pencarian explodes token. ToT pada Game dari 24 memakai 100–1000x token dari CoT. LATS adalah similar. Ini adalah tidak free; reserve pencarian untuk:

- Tugas di mana sebuah tunggal trajectory adalah demonstrably insufficient (Game dari 24, complex kode).
- Tugas di mana wall-clock adalah less penting than correctness.
- Tugas dengan sebuah cheap, reliable value function (unit tes untuk kode, eksplisit target untuk math).

Jika kamu tugas memiliki sebuah tunggal right jawaban dan sebuah noisy evaluator, pencarian sering membuat hal worse — itu finds sebuah "baik-scoring" salah jawaban.

### 2026 positioning

Sebagian besar produksi agen lakukan tidak jalankan LATS. Mereka jalankan ReAct dengan tool-grounded verifikasi (CRITIC, Pelajaran 05). Pencarian menunjukkan up di specialized niches:

- Coding agen itu jalankan tes sebagai value function (HumanEval-style).
- Deep-riset agen itu explore multiple query paths.
- Perencanaan-heavy workflow di dalam LangGraph subgraphs.

AlphaEvolve (Pelajaran 11) adalah 2025 extreme: evolutionary pencarian atas kode, machine-checkable fitness, frontier gains (pertama 4x4 matmul improvement di 56 years).

```figure
tree-of-thoughts
```

## Bangun Sendiri

`code/main.py` implements:

- SEBUAH tiny ToT BFS pada sebuah stylized "pick arithmetic ops" tugas.
- SEBUAH toy LATS MCTS loop pada sama tugas (Select / Expand / Simulate / Backpropagate) dengan UCT selection.
- SEBUAH value function itu composes sebuah symbolic score plus sebuah mandiri-eval score.

Jalankan:

```
python3 code/main.py
```

 trace menunjukkan ToT expanding tiga candidates per node dengan BFS, dibandingkan dengan LATS converging pada best rollout via MCTS. Token counts printed untuk keduanya.

## Gunakan

LangGraph ships ToT-style exploration sebagai subgraph polas; LangChain tim's blog pada LATS (Dapat 2024) adalah reference tutorial. LlamaIndex ships sebuah `TreeOfThoughts` agen. Untuk sebagian besar 2026 produksi agen ini pola lives di balik sebuah `if task_complexity > threshold: use_search()` gate — see evaluator-optimizer pola di Pelajaran 05.

## Rilis

`outputs/skill-search-policy.md` selects antara linear ReAct, ToT, LATS, dan evolutionary pencarian given tugas bentuk, budget, dan evaluator fidelity.

## Latihan

1. Jalankan toy LATS dengan UCT c=0.1 vs c=2.0. Apa yang berubah di trace?
2. Tukar value function untuk sebuah noisier scorer (add random jitter). Melakukan MCTS tetap find best leaf? Apa adalah minimum signal-ke-noise itu tolerates?
3. Implementasikan beam-pencarian ToT (keep top-k pada setiap level) dan bandingkan ke BFS. Yang adalah better pada sebuah tight token budget?
4. Baca LATS Bagian 5.1. Reproduce HumanEval trajectory count: bagaimana banyak rollouts melakukan itu ambil ke hit reported pass@1?
5. Baca LATS paper's discussion pada "ketika LATS membantu less." Tulis sebuah satu-paragraph decision aturan mapping tugas bentuk ke pencarian strategi.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tree of Thoughts | "Branching CoT" | Yao et al. — tree of thought nodes with self-evaluation |
| LATS | "MCTS for LLMs" | Zhou et al. — unifies ToT + ReAct + Reflexion under MCTS |
| UCT | "Upper confidence bound" | Select formula balancing exploitation (Q) and exploration (ln N / n) |
| Value function | "How good is this state" | Prompted LLM score or environment reward; feeds backprop |
| Policy | "Action proposer" | ReAct-style generator; emits candidate next thoughts/actions |
| Rollout | "Simulated trajectory" | Walk from a node to a leaf using policy, score with value |
| Backpropagate | "Update ancestors" | Push the leaf's reward up the path, updating visit counts and Q |
| Search cost | "Token explosion" | 100-1000x CoT on Game of 24; budget before you adopt |

## Bacaan Lanjutan

- [Yao et al., Tree dari Thought (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — paper kanonikal
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — MCTS dengan Reflexion feedback
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — subgraph polas untuk pencarian
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — evolutionary pencarian dengan programmatic evaluators
