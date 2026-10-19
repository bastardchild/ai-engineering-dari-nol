# Metode Monte Carlo — Belajar dari Episode Lengkap

> Pemrograman dinamis butuh model. Monte Carlo butuh tak lebih dari episode. Jalankan policy, saksikan return, rata-ratakan. Ide paling sederhana dalam RL — dan yang membuka segala hal downstream.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 minutes

## Masalahnya

Pemrograman dinamis elegan, tapi ia mengasumsikan kamu bisa menanyakan `P(s' | s, a)` untuk setiap state dan action. Hampir tak ada di dunia nyata yang bekerja seperti itu. Robot tidak bisa menghitung analitis distribusi atas piksel kamera setelah torsi sendi. Algoritma pricing tidak bisa mengintegralkan atas setiap reaksi pelanggan yang mungkin. LLM tidak bisa mengenumerasi semua kelanjutan yang mungkin setelah token.

Kamu butuh metode yang hanya butuh kemampuan *menyampel* dari environment. Jalankan policy. Dapatkan trajektori `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`. Pakai untuk mengestimasi value. Itulah Monte Carlo.

Pergeseran dari DP ke MC penting secara filosofis: kita bergerak dari *model diketahui + backup eksak* ke *rollout tersampel + return rata-rata*. Variansi melonjak, tapi aplikabilitas meledak. Setiap algoritma RL setelah pelajaran ini — TD, Q-learning, REINFORCE, PPO, GRPO — adalah estimator Monte Carlo pada hatinya, kadang dengan bootstrap berlapis di atasnya.

## Konsepnya

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**Ide inti, dalam satu baris:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` dengan `G^{(i)}(s)` adalah return teramati setelah kunjungan ke `s` di bawah policy `π`.

**First-visit vs every-visit MC.** Diberikan episode yang mengunjungi state `s` berkali-kali, MC first-visit hanya menghitung return dari kunjungan pertama; MC every-visit menghitung semua kunjungan. Keduanya tak bias dalam limit. First-visit lebih sederhana dianalisis (sampel iid). Every-visit memakai lebih banyak data per episode dan tipikalnya konvergen lebih cepat dalam praktik.

**Rata-rata inkremental.** Alih-alih menyimpan semua return, update rata-rata berjalan:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Susun ulang: `V_new = V_old + α · (target - V_old)` dengan `α = 1/n`. Tukar `1/n` dengan step-size konstan `α ∈ (0, 1)` dan kamu dapat estimator MC non-stasioner yang melacak perubahan `π`. Langkah itu adalah seluruh lompatan dari MC ke TD ke setiap algoritma RL modern.

**Eksplorasi kini menjadi problem.** DP menyentuh setiap state lewat enumerasi. MC hanya melihat state yang dikunjungi policy. Kalau `π` deterministik, seluruh wilayah ruang state tidak pernah tersampel, dan estimasi value-nya tetap nol selamanya. Tiga perbaikan, dalam urutan historis:

1. **Exploring start.** Mulai setiap episode dari pasangan (s, a) acak. Menjamin cakupan; tak realistis dalam praktik (kamu tidak bisa "me-reset" robot ke state sembarang).
2. **ε-greedy.** Bertindak greedy thd. Q kini, tapi dengan probabilitas `ε` pilih action acak. Semua pasangan state-action tersampel secara asimtotik.
3. **MC off-policy.** Kumpulkan data di bawah policy perilaku `μ`, pelajari soal policy target `π` via importance sampling. Variansi tinggi, tapi ia jembatan ke metode replay-buffer seperti DQN.

**Kontrol Monte Carlo.** Evaluasi → perbaiki → evaluasi, persis seperti iterasi policy, tapi evaluasi berbasis-sampling:

1. Jalankan `π`, dapatkan episode.
2. Update `Q(s, a)` dari return teramati.
3. Buat `π` ε-greedy thd. `Q`.
4. Ulangi.

Konvergen ke `Q*` dan `π*` dengan probabilitas 1 di bawah kondisi ringan (setiap pasangan dikunjungi takhingga sering, `α` memenuhi Robbins-Monro).

```figure
epsilon-greedy
```

## Bangun Sendiri

### Langkah 1: rollout → daftar (s, a, r)

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

Tanpa model, hanya `env.reset()` dan `env.step(s, a)`. Interface sama dengan environment gym tapi dilucuti.

### Langkah 2: hitung return (sapuan balik)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Satu pass, `O(T)`. Rekurensi mundur `G_t = r_{t+1} + γ G_{t+1}` menghindari penjumlahan ulang.

### Langkah 3: evaluasi MC first-visit

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Tiga baris melakukan kerja: tandai state sebagai terlihat pada kunjungan pertama, naikkan cacahan, update rata-rata berjalan.

### Langkah 4: kontrol MC ε-greedy (on-policy)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Langkah 5: bandingkan dengan standar-emas DP

Estimasi MC-mu atas `V^π` seharusnya sepakat dengan hasil DP dari Pelajaran 02 saat episode → ∞. Dalam praktik: 50.000 episode pada GridWorld 4×4 membawamu dalam `~0.1` dari jawaban DP.

## Jebakan

- **Episode takhingga.** MC mensyaratkan episode *berakhir*. Kalau policy-mu bisa loop selamanya, batasi `max_steps` dan perlakukan batas sebagai kegagalan implisit. GridWorld dengan policy acak rutin timeout — itu normal, pastikan kamu menghitungnya dengan benar.
- **Variansi.** MC memakai return penuh. Pada episode panjang, variansi besar — satu reward sial di akhir menggeser `V(s_0)` sebesar yang sama. Metode TD (Pelajaran 04) memotong ini dengan bootstrap.
- **Cakupan state.** MC greedy pada Q segar dengan tie hanya akan mencoba satu action. Kamu *harus* bereksplorasi (ε-greedy, exploring start, UCB).
- **Policy non-stasioner.** Kalau `π` berubah (seperti dalam kontrol MC), return lama berasal dari policy berbeda. MC α-konstan menangani ini; MC rata-rata-sampel tidak.
- **Importance sampling off-policy.** Bobot `π(a|s)/μ(a|s)` berlipat sepanjang trajektori. Variansi meledak seiring horizon. Batasi dengan IS tertimbang per-keputusan atau beralih ke TD.

## Gunakan

Peran 2026 metode Monte Carlo:

| Use case | Why MC |
|----------|--------|
| Short-horizon games (blackjack, poker) | Episodes terminate naturally; returns are clean. |
| Offline evaluation of a logged policy | Average discounted returns over stored trajectories. |
| Monte Carlo Tree Search (AlphaZero) | MC rollouts from tree leaves guide selection. |
| LLM RL evaluation | Compute average reward over sampled completions for a given policy. |
| Baseline estimation in PPO | The advantage target `A_t = G_t - V(s_t)` uses an MC `G_t`. |
| Teaching RL | Simplest algorithm that actually works — strip bootstrapping to see the core. |

Algoritma deep-RL modern (PPO, SAC) menginterpolasikan antara MC murni (return penuh) dan TD murni (bootstrap satu-langkah) via return `n`-langkah atau GAE. Kedua endpoint adalah instans estimator yang sama.

## Rilis

Simpan sebagai `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Latihan

1. **Mudah.** Implementasikan evaluasi MC first-visit atas policy seragam-acak pada GridWorld 4×4. Jalankan 10.000 episode. Plot `V(0,0)` sebagai fungsi jumlah episode terhadap jawaban DP.
2. **Sedang.** Implementasikan kontrol MC ε-greedy dengan `ε ∈ {0.01, 0.1, 0.3}`. Bandingkan return rata-rata setelah 20.000 episode. Seperti apa kurvanya? Di mana trade-off bias-varians hidup?
3. **Sulit.** Implementasikan MC *off-policy* dengan importance sampling: kumpulkan data di bawah policy seragam-acak `μ`, estimasi `V^π` untuk policy optimal deterministik `π`. Bandingkan IS polos vs IS per-keputusan vs IS berbobot. Mana variansi terendah?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Monte Carlo | "Random sampling" | Estimate expectations by averaging over iid samples from the distribution. |
| Return `G_t` | "Future reward" | Sum of discounted rewards from step `t` to episode end: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| First-visit MC | "Count each state once" | Only the first visit in an episode contributes to the value estimate. |
| Every-visit MC | "Use all visits" | Every visit contributes; slightly biased but more sample-efficient. |
| ε-greedy | "Exploration noise" | Pick greedy action with prob `1-ε`; random action with prob `ε`. |
| Importance sampling | "Correcting for sampling from the wrong distribution" | Reweight returns by `π(a\|s)/μ(a\|s)` products to estimate `V^π` from `μ` data. |
| On-policy | "Learn from my own data" | Target policy = behavior policy. Vanilla MC, PPO, SARSA. |
| Off-policy | "Learn from someone else's data" | Target policy ≠ behavior policy. Importance-sampled MC, Q-learning, DQN. |

## Bacaan Lanjutan

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — perlakuan kanonis.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — analisis first-visit vs every-visit.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — MC off-policy dan kontrol variansi.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — estimator IS modern variansi-rendah.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — demonstrasi empiris skala-besar pertama bahwa self-play MC/TD konvergen ke permainan superhuman; prekursor konseptual untuk setiap pelajaran di paruh kedua fase ini.
