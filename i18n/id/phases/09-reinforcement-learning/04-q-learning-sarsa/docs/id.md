# Temporal Difference — Q-Learning & SARSA

> Monte Carlo menunggu sampai episode selesai. TD melakukan update setiap langkah dengan mem-bootstrap estimasi nilai berikutnya. Q-learning bersifat off-policy dan optimis; SARSA bersifat on-policy dan hati-hati. Keduanya hanya satu baris kode. Keduanya menjadi dasar setiap metode deep-RL di fase ini.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75 minutes

## Masalahnya

Monte Carlo berfungsi tetapi punya dua tuntutan mahal. Ia butuh episode yang berhenti, dan ia hanya melakukan update setelah return akhir masuk. Jika episodemu 1.000 langkah, MC menunggu 1.000 langkah sebelum mengupdate apa pun. Variansinya tinggi, biasnya rendah, dan lambat dalam praktik.

Dynamic programming punya profil yang berkebalikan — backup ber-bootstrap dengan varians nol — tetapi membutuhkan model yang diketahui.

Temporal difference (TD) learning membagi jarak di antaranya. Dari satu transisi `(s, a, r, s')`, bentuk target satu langkah `r + γ V(s')` dan dorong `V(s)` ke arahnya. Tanpa model. Tanpa episode lengkap. Ada bias karena memakai `V` aproksimatif di sisi kanan, tetapi variansnya jauh lebih rendah daripada MC dan update-nya online sejak langkah pertama.

Inilah titik tumpu tempat seluruh RL modern — DQN, A2C, PPO, SAC — berputar. Sisa Fase 9 adalah lapisan aproksimasi fungsi dan trik yang dibangun di atas update TD satu langkah yang akan kamu tulis di pelajaran ini.

## Konsepnya

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**Update TD(0) untuk V:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

Kuantitas dalam kurung adalah TD error `δ = r + γ V(s') - V(s)`. Ini adalah analog online dari `G_t - V(s_t)` di MC. Konvergensi mensyaratkan `α` memenuhi Robbins-Monro (`Σ α = ∞`, `Σ α² < ∞`) dan semua state dikunjungi tanpa batas.

**Q-learning.** Metode TD off-policy untuk kontrol:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` mengasumsikan kebijakan *greedy* akan diikuti sejak `s'` ke depan, terlepas dari aksi apa yang sebenarnya diambil agen. Pemisahan itu membuat Q-learning mempelajari `Q*` sementara agen menjelajah lewat ε-greedy. Mnih et al. (2015) mengubahnya menjadi deep Q-learning di Atari (Pelajaran 05).

**SARSA.** Metode TD on-policy:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

Namanya berasal dari tuple `(s, a, r, s', a')`. SARSA memakai aksi `a'` yang *benar-benar* diambil agen berikutnya, bukan `argmax` yang greedy. Konvergen ke `Q^π` untuk kebijakan ε-greedy `π` apa pun yang sedang berjalan, yang pada batas `ε → 0` menjadi `Q*`.

**Perbedaan cliff-walking.** Pada tugas cliff-walking klasik (jatuh dari tebing = reward -100), Q-learning mempelajari jalur optimal di tepi tebing tetapi sesekali mengambil penalti saat menjelajah. SARSA mempelajari jalur yang lebih aman selangkah dari tebing karena memasukkan noise eksplorasi ke dalam nilai Q-nya. Dengan training, keduanya mencapai optimal pada `ε → 0`. Dalam praktik ini penting: saat eksplorasi benar-benar terjadi saat deployment, perilaku SARSA lebih konservatif.

**Expected SARSA.** Ganti `Q(s', a')` dengan nilai harapannya di bawah `π`:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

Variansnya lebih rendah daripada SARSA (tanpa sampling `a'`), target on-policy yang sama. Sering jadi default di buku teks modern.

**n-step TD dan TD(λ).** Menginterpolasi antara TD(0) dan MC dengan menunggu `n` langkah sebelum melakukan bootstrap. `n=1` adalah TD, `n=∞` adalah MC. TD(λ) merata-ratakan seluruh `n` dengan bobot geometrik `(1-λ)λ^{n-1}`. Sebagian besar deep-RL memakai `n` antara 3 dan 20.

```figure
qlearning-gridworld
```

## Bangun Sendiri

### Langkah 1: SARSA pada kebijakan ε-greedy

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Delapan baris. Satu-satunya perbedaan dari Q-learning ada di baris target.

### Langkah 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` memisahkan target dari perilaku. Simbol tunggal itulah perbedaan antara on-policy dan off-policy.

### Langkah 3: kurva pembelajaran

Lacak return rata-rata per 100 episode. Q-learning konvergen lebih cepat pada GridWorld deterministik sederhana; SARSA lebih konservatif pada cliff-walking. Pada GridWorld 4×4 di `code/main.py`, keduanya nyaris-optimal setelah ~2.000 episode dengan `α=0.1, ε=0.1`.

### Langkah 4: bandingkan dengan kebenaran DP

Jalankan value iteration (Pelajaran 02) untuk mendapatkan `Q*`. Periksa `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`. Agen TD tabular yang sehat berada dalam jangkauan `~0.5` pada GridWorld 4×4 setelah 10.000 episode.

## Jebakan

- **Nilai Q awal itu penting.** Inisialisasi optimis (`Q = 0` untuk tugas ber-reward negatif) mendorong eksplorasi. Inisialisasi pesimis bisa menjebak kebijakan greedy selamanya.
- **Jadwal α.** `α` konstan sudah cukup untuk masalah non-stasioner. `α_n = 1/n` yang meluruh memberikan konvergensi dalam teori tetapi terlalu lambat dalam praktik — kunci `α` di `[0.05, 0.3]` dan pantau kurva pembelajarannya.
- **Jadwal ε.** Mulai tinggi (`ε=1.0`), luruhkan ke `ε=0.05`. "GLIE" (greedy pada batas dengan eksplorasi tak terbatas) adalah syarat konvergensi.
- **Bias maksimum di Q-learning.** Operator `max` bias ke atas saat `Q` bising. Mengarah ke overestimation — Double Q-learning milik Hasselt (dipakai DDQN di Pelajaran 05) memperbaikinya dengan dua tabel Q.
- **Episode tanpa terminal.** TD bisa belajar tanpa terminal, tetapi kamu perlu membatasi langkah atau menangani bootstrap dengan benar di batas itu. Standar: perlakukan batas sebagai non-terminal, terus melakukan bootstrap.
- **Hashing state.** Jika state berupa tuple/tensor, pakai kunci yang bisa di-hash (tuple, bukan list; tuple dari float yang sudah dibulatkan, bukan mentah).

## Gunakan

Lanskap TD 2026:

| Task | Method | Reason |
|------|--------|--------|
| Small tabular environments | Q-learning | Learns optimal policy directly. |
| On-policy safety-critical | SARSA / Expected SARSA | Conservative during exploration. |
| High-dimensional state | DQN (Phase 9 · 05) | Neural-net Q-function with replay and target net. |
| Continuous actions | SAC / TD3 (Phase 9 · 07) | TD update on a Q-network; policy net emits actions. |
| LLM RL (reward-model-based) | PPO / GRPO (Phase 9 · 08, 12) | Actor-critic with TD-style advantage via GAE. |
| Offline RL | CQL / IQL (Phase 9 · 08) | Q-learning with conservative regularization. |

Sembilan puluh persen dari "RL" yang kamu baca di makalah 2026 adalah semacam elaborasi dari Q-learning atau SARSA. Pahami update tabular itu sampai ke jari-jarimu sebelum membaca lebih dalam.

## Rilis

Simpan sebagai `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Latihan

1. **Mudah.** Implementasikan Q-learning dan SARSA pada GridWorld 4×4. Plot kurva pembelajaran (return rata-rata per 100 episode) selama 2.000 episode. Siapa yang konvergen lebih cepat?
2. **Sedang.** Bangun lingkungan cliff-walking (4×12, baris terakhir adalah tebing dengan reward -100 dan reset ke awal). Bandingkan kebijakan akhir Q-learning dan SARSA. Ambil tangkapan layar jalur masing-masing. Mana yang lebih dekat ke tebing?
3. **Sulit.** Implementasikan Double Q-learning. Pada GridWorld ber-reward-noisy (noise Gaussian σ=5 ditambahkan ke reward per langkah), tunjukkan bahwa Q-learning menaksir `V*(0,0)` terlalu tinggi secara bermakna sementara Double Q-learning tidak.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| TD error | "The update signal" | `δ = r + γ V(s') - V(s)`, the bootstrapped residual. |
| TD(0) | "One-step TD" | Update after every transition using only the next state's estimate. |
| Q-learning | "Off-policy RL 101" | TD update with `max` over next-state actions; learns `Q*` regardless of behavior policy. |
| SARSA | "On-policy Q-learning" | TD update using the actual next action; learns `Q^π` for current ε-greedy π. |
| Expected SARSA | "The low-variance SARSA" | Replace sampled `a'` with its expectation under π. |
| GLIE | "Correct exploration schedule" | Greedy in the Limit with Infinite Exploration; needed for Q-learning convergence. |
| Bootstrapping | "Using current estimate in the target" | What distinguishes TD from MC. Source of bias but massive variance reduction. |
| Maximization bias | "Q-learning overestimates" | `max` over noisy estimates is upward-biased; fixed by Double Q-learning. |

## Bacaan Lanjutan

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — the original paper and convergence proof.
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, Expected SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — fix for maximization bias.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — expected SARSA motivation.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — the paper that coined SARSA (then called "modified connectionist Q-learning").
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — generalizes TD(0) to TD(n), the path from Q-learning to eligibility traces and, later, GAE in PPO.
