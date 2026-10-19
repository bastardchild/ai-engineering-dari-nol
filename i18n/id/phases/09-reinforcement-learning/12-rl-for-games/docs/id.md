# RL for Games — AlphaZero, MuZero, dan Era Penalaran LLM

> 1992: TD-Gammon mengalahkan juara manusia di backgammon dengan TD murni. 2016: AlphaGo mengalahkan Lee Sedol. 2017: AlphaZero mendominasi catur, shogi, dan Go dari nol. 2024: DeepSeek-R1 membuktikan resep yang sama, dengan GRPO menggantikan PPO, bekerja untuk penalaran. Game adalah benchmark yang mendorong setiap terobosan di phase ini.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120 minutes

## Masalahnya

Game memiliki segala yang diinginkan RL. Reward bersih (menang/kalah). Episode tak terbatas (self-play reset). Simulasi sempurna (game-nya *adalah* simulator-nya). Ruang aksi diskret atau kontinu kecil. Struktur multi-agent yang memaksa ketahanan adversarial.

Dan game adalah tempat setiap terobosan RL utama diuji. TD-Gammon (backgammon, 1992). Atari-DQN (2013). AlphaGo (2016). AlphaZero (2017). OpenAI Five (Dota 2, 2019). AlphaStar (StarCraft II, 2019). MuZero (model yang dipelajari, 2019). AlphaTensor ( perkalian matriks, 2022). AlphaDev (algoritma sorting, 2023). DeepSeek-R1 (penalaran matematika, 2025) — demonstrasi terbaru bahwa teknik game-RL bekerja pada teks.

Capstone ini menyurvei tiga arsitektur bersejarah — AlphaZero, MuZero, dan GRPO — melalui satu lensa pemersatu: **self-play + search + policy improvement**. Masing-masing menggeneralisasi yang sebelumnya; GRPO khususnya adalah resep AlphaZero yang diterapkan pada penalaran LLM, dengan token sebagai aksi dan verifikasi matematika sebagai sinyal menang.

## Konsepnya

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**Loop pemersatu.**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).** Silver et al. Diberi game (catur, shogi, Go) dengan aturan yang diketahui:

- Policy-value network: satu tower `f_θ(s) → (p, v)`. `p` adalah prior atas langkah legal. `v` adalah hasil game yang diharapkan.
- Monte Carlo Tree Search (MCTS): pada setiap langkah, ekspansi pohon kemungkinan lanjutan. Gunakan `(p, v)` sebagai prior + bootstrap. Pilih node dengan UCB (PUCT): `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- Self-play: bermain game agen-vs-agen. Pada langkah `t`, distribusi kunjungan MCTS `π_t` menjadi target training policy.
- Loss: `L = (v - z)² - π · log p + c · ||θ||²`. `z` adalah hasil game (+1 / 0 / -1).

Nol pengetahuan manusia. Nol heuristik buatan-tangan. Satu resep yang menguasai catur, shogi, dan Go setelah beberapa puluh juta game self-play masing-masing.

**MuZero (2019).** Schrittwieser et al. Menghapus persyaratan bahwa aturan diketahui.

- Alih-alih environment tetap, pelajari *latent dynamics model* `(h, g, f)`:
  - `h(s)`: encode observasi ke latent state.
  - `g(s_latent, a)`: prediksi latent state berikutnya + reward.
  - `f(s_latent)`: prediksi policy prior + value.
- MCTS berjalan di *ruang latent yang dipelajari*. Search yang sama, loop training yang sama.
- Bekerja pada Go, catur, shogi *dan* Atari — satu algoritma, tanpa pengetahuan aturan.

**Stochastic MuZero (2022).** Menambahkan dinamika stokastik dan chance nodes; meluas ke game kelas backgammon.

**Muesli, Gumbel MuZero (2022-2024).** Peningkatan pada sample efficiency dan pencarian deterministik.

**GRPO (2024-2025).** Resep DeepSeek-R1. Loop berbentuk-AlphaZero yang sama, diterapkan pada penalaran bahasa-model:

- "Game": jawab masalah matematika / coding / penalaran. "Menang" = verifier (test case lulus, jawaban numerik cocok) mengembalikan 1.
- Policy: LLM-nya. Aksi: token. State: prompt + response-sejauh-ini.
- Tanpa critic (V_φ ala PPO). Sebaliknya, untuk setiap prompt, sample `G` completions dari policy. Hitung reward untuk masing-masing. Gunakan **group-relative advantage** `A_i = (r_i - mean_r) / std_r` sebagai sinyal untuk update gaya REINFORCE.
- KL penalty ke policy reference untuk mencegah drift (seperti RLHF).
- Loss lengkap:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

Tanpa reward model, tanpa critic, tanpa MCTS. Baseline relatif-grup menggantikan ketiganya. Menyamai atau melampaui kualitas PPO-RLHF di benchmark penalaran pada sebagian kecil compute.

**Resep R1 seluruhnya.** DeepSeek-R1 (DeepSeek 2025) adalah dua model dalam satu paper:

- **R1-Zero.** Mulai dari base model DeepSeek-V3. Tanpa SFT. Terapkan GRPO langsung dengan dua komponen reward: *accuracy reward* (berbasis-aturan — apakah jawaban akhir ter-parse ke angka yang benar / apakah kode lulus unit tests) dan *format reward* (apakah completion membungkus chain-of-thought-nya dalam tag `<think>…`). Selama ribuan steps, panjang respons rata-rata tumbuh dari ~100 ke ~10.000 tokens dan skor benchmark matematika naik ke tingkat mendekati o1-preview. Model belajar mengejak dari nol. Kekurangannya: chain of thought-nya sering tidak terbaca, mencampur bahasa, dan kekurangan polesan gaya.
- **R1.** Perbaiki masalah keterbacaan R1-Zero dengan pipeline empat tahap:
  1. **Cold-start SFT.** Kumpulkan beberapa ribu demonstrasi panjang-CoT dengan formatting bersih. Supervised-finetune base model padanya. Ini memberikan titik awal yang terbaca.
  2. **GRPO berorientasi-penalaran.** Terapkan GRPO dengan accuracy+format rewards ditambah reward *konsistensi-bahasa* untuk mencegah code-switching.
  3. **Rejection sampling + SFT ronde 2.** Sample ~600K trajectory penalaran dari checkpoint RL, simpan hanya yang memiliki jawaban akhir benar dan CoT terbaca, dan gabungkan dengan ~200K contoh SFT non-penalaran (menulis, QA, self-cognition). Fine-tune base lagi.
  4. **GRPO spektrum-penuh.** Satu ronde RL lagi yang mencakup penalaran (reward berbasis-aturan) dan alignment umum umum (helpfulness/harmlessness berbasis-preferensi).

Hasilnya menyamai o1 di AIME dan MATH-500 dengan bobot terbuka, dan cukup kecil untuk didistilasi. Paper yang sama juga merilis enam model padat terdistilasi (Qwen-1.5B hingga Llama-70B) dengan SFT pada traces penalaran R1 — tanpa RL di student. Distilasi dari teacher RL yang kuat secara konsisten mengalahkan RL dari nol pada skala student.

**Mengapa GRPO alih-alih PPO untuk penalaran.** Tiga alasan di paper DeepSeekMath (Feb 2024): (1) tidak ada value network untuk dilatih, memangkas memori setengah; (2) baseline kelompok secara alami menangani reward akhir-trajectory yang sparse yang dihasilkan tugas penalaran; (3) normalisasi per-prompt membuat advantages dapat dibandingkan antar masalah dengan kesulitan yang sangat berbeda, yang tidak bisa dilakukan critic tunggal PPO.

**Tanpa search vs ber-search.** Game telah bercabang:

- *Game informasi-perfek dengan horizon panjang* (Go, catur): masih ber-search. AlphaZero / MuZero mendominasi.
- *Penalaran LLM*: belum ada MCTS di produksi; GRPO pada rollouts penuh, best-of-N untuk compute inference. Process reward models (PRMs) mengisyaratkan pencarian tingkat-langkah ditambahkan kembali.

```figure
f3-selfplay-ladder
```

## Bangun Sendiri

Kode di `code/main.py` mengimplementasikan **GRPO dalam miniatur** — bandit dengan beberapa kelompok sampel. Algoritmanya sama seperti pada LLM; hanya policy dan environment-nya yang lebih sederhana. Ia mengajarkan *loss* dan *group-relative advantage*, yang merupakan inovasi 2025.

### Langkah 1: environment verifier mungil

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

Di GRPO sungguhan verifier menjalankan unit tests atau memeriksa kesetaraan matematika.

### Langkah 2: policy: softmax atas K token jawaban per prompt

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

Ekivalen dengan output layer-terakhir dari LLM yang mengkondisikan pada prompt.

### Langkah 3: sampling kelompok dan group-relative advantage

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Group-relative advantage adalah trik DeepSeek 2024. Tanpa critic. "Baseline" adalah mean kelompok, dan normalisasi menggunakan std kelompok.

### Langkah 4: bandingkan dengan baseline REINFORCE (tanpa value)

Setup yang sama, compute yang sama, REINFORCE polos. GRPO konvergen lebih cepat dan lebih stabil.

### Langkah 5: amati entropy dan KL

Diagnostik yang sama dengan RLHF: mean KL ke reference, entropy policy, reward dari waktu ke waktu. Setelah ini stabil, training selesai.

## Jebakan

- **Reward hacking via permainan verifier.** GRPO mewarisi risiko RLHF: jika verifier salah atau bisa dieksploitasi, LLM akan menemukan eksploitasinya. Verifier yang tangguh (banyak test case, bukti formal) penting.
- **Ukuran kelompok terlalu kecil.** Varians baseline kelompok berbanding lurus dengan `1/√G`. Di bawah `G = 4`, sinyal advantage berisik; pilihan standar adalah `G = 8` hingga `64`.
- **Bias panjang.** Completion LLM dengan panjang berbeda memiliki log-probabilities berbeda. Normalisasi dengan jumlah token, atau gunakan log-prob tingkat-sekuens, atau potong ke panjang maksimal.
- **Siklus self-play murni.** Training gaya AlphaZero bisa tersangkut dalam loop dominasi pada game general-sum. Dimitigasi dengan pool lawan beragam (league play, Lesson 10).
- **Ketidakcocokan search-policy.** AlphaZero melatih policy untuk meniru output search. Jika policy net terlalu kecil untuk merepresentasikan distribusi search, training mandek.
- **Lantai compute.** MuZero / AlphaZero membutuhkan compute masif. Satu ablation sering ratusan GPU-jam. Demonstrasi miniatur ada (mis., AlphaZero pada Connect Four) untuk pembelajaran.
- **Cakupan verifier.** Unit tests yang lulus untuk solusi yang bug memperkuat bug-nya. Rancang verifier yang menangkap edge cases.

## Gunakan

Lanskap game-RL 2026, berdasarkan domain:

| Domain | Dominant method |
|--------|-----------------|
| Two-player zero-sum board games (Go, chess, shogi) | AlphaZero / MuZero / KataGo |
| Imperfect info card games (poker) | CFR + deep learning (DeepStack, Libratus, Pluribus) |
| Atari / pixel games | Muesli / MuZero / IMPALA-PPO |
| Large multiplayer strategy (Dota, StarCraft) | PPO + self-play + league (OpenAI Five, AlphaStar) |
| LLM math/code reasoning | GRPO (DeepSeek-R1, Qwen-RL, open replications) |
| LLM alignment | DPO / RLHF-PPO (not GRPO; verifier is preference not verifiable) |
| Robotics | PPO + DR (not game-RL, but uses same policy-gradient tools) |
| Combinatorial problems | AlphaZero variants (AlphaTensor, AlphaDev) |

*Resepnya* — self-play, peningkatan yang di-augmentasi-search, distilasi policy — membentang di teks, piksel, dan kontrol fisik. GRPO adalah instansi termuda; lebih banyak akan datang.

## Rilis

Simpan sebagai `outputs/skill-game-rl-designer.md`:

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## Latihan

1. **Mudah.** Implementasikan bandit GRPO di `code/main.py`. Latih pada 2 prompts × 4 token jawaban masing-masing. Konvergen dalam < 1.000 updates dengan `G=8`.
2. **Sedang.** Colokkan PPO (ber-clip) dan REINFORCE vanilla. Bandingkan sample efficiency dan varians reward dengan GRPO pada bandit yang sama.
3. **Sulit.** Perluas ke "reasoning chain" panjang-2: agen memancarkan dua token dan verifier memberi reward ke pasangan itu. Ukur bagaimana GRPO menangani credit assignment di sekuens dua-langkah. (Petunjuk: hitung group advantage per *sekuens penuh*, propagasikan ke kedua posisi token.)

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MCTS | "Tree search with learned net" | Monte Carlo Tree Search; UCB1/PUCT selection with learned `(p, v)` priors. |
| AlphaZero | "Self-play + MCTS" | Policy-value net trained to match MCTS visits and game outcome. |
| MuZero | "Learned-model AlphaZero" | Same loop but in latent space via learned dynamics. |
| GRPO | "Critic-free PPO" | Group Relative Policy Optimization; REINFORCE with group-mean baseline + KL. |
| PUCT | "AlphaZero's UCB" | `Q + c · p · √N / (1 + N_a)` — balances value estimate with prior. |
| Self-play | "Agent vs past self" | Standard for zero-sum; symmetric training signal. |
| League play | "Population-based self-play" | Past + current + exploiters sampled as opponents. |
| Verifier reward | "Verifiable RL" | Reward comes from a deterministic checker (tests pass, answer matches). |
| Process reward | "PRM" | Scores each reasoning step, not just the final answer. |

## Bacaan Lanjutan

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — the paper that introduced GRPO and the group-relative baseline.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — the full four-stage R1 recipe plus the R1-Zero ablation.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + deep-learning at scale.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — the paper that started it all.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — the production reference for applying GRPO with custom reward functions.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — open replication of the R1 recipe at multiple scales.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — the textbook framing for self-play, search, and "designed reward" that R1 instantiates at LLM scale.
