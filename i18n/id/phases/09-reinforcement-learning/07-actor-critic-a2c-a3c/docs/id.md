# Actor-Critic — A2C dan A3C

> REINFORCE itu bising. Tambahkan kritikus yang mempelajari `V̂(s)`, kurangkan dari return, dan kamu mendapatkan advantage yang punya harapan sama tetapi varians jauh lebih rendah. Itulah actor-critic. A2C menjalankannya secara sinkron; A3C menjalankannya di lintasan-lintasan. Keduanya adalah model mental untuk setiap metode deep-RL modern.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75 minutes

## Masalahnya

REINFORCE polos berfungsi, tetapi variansnya mengerikan. Return Monte Carlo `G_t` bisa bergeser sebesar faktor 10 antar episode. Mengalikan noise itu dengan `∇ log π` dan merata-ratakannya menghasilkan estimator gradien yang butuh ribuan episode untuk menggeser kebijakan sejauh yang bisa kamu capai dengan jauh lebih sedikit update DQN.

Variansnya berasal dari pemakaian return mentah. Jika kamu mengurangkan baseline `b(s_t)` — fungsi state apa pun, termasuk nilai yang dipelajari — harapannya tidak berubah dan variansnya turun. Baseline paling praktis adalah `V̂(s_t)`. Kini kuantitas yang mengalikan `∇ log π` adalah *advantage*:

`A(s, a) = G - V̂(s)`

Sebuah aksi baik jika menghasilkan return di atas rata-rata; buruk jika di bawah. REINFORCE dengan kritikus yang dipelajari adalah *actor-critic*. Kritikus memberi aktor seorang guru ber-varians rendah. Inilah setiap metode deep-policy setelah 2015 (A2C, A3C, PPO, SAC, IMPALA).

## Konsepnya

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**Dua jaringan, satu loss bersama:**

- **Actor** `π_θ(a | s)`: kebijakannya. Disampel untuk bertindak. Dilatih dengan policy gradient.
- **Critic** `V_φ(s)`: mengecek return yang diharapkan dari sebuah state. Dilatih untuk meminimalkan `(V_φ(s) - target)²`.

**Advantage-nya.** Dua bentuk standar:

- *MC advantage:* `A_t = G_t - V_φ(s_t)`. Tidak bias, varians lebih tinggi.
- *TD advantage:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. Bias (memakai `V_φ`), varians jauh lebih rendah. Juga disebut *TD residual* `δ_t`.

**n-step advantage.** Menginterpolasi di antara keduanya:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` adalah TD murni. `n = ∞` adalah MC. Sebagian besar implementasi memakai `n = 5` untuk Atari, `n = 2048` untuk PPO di MuJoCo.

**Generalized Advantage Estimation (GAE).** Schulman et al. (2016) mengusulkan rata-rata berbobot eksponensial atas seluruh n-step advantage:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

dengan `λ ∈ [0, 1]`. `λ = 0` adalah TD (varians rendah, bias tinggi). `λ = 1` adalah MC (varians tinggi, tidak bias). `λ = 0.95` adalah default 2026 — sesuaikan sampai dial bias/varians berada di posisi yang kamu inginkan.

**A2C: synchronous advantage actor-critic.** Kumpulkan `T` langkah di `N` lingkungan paralel. Hitung advantage untuk setiap langkah. Update aktor dan kritikus pada batch gabungan. Ulangi. Kakak yang lebih sederhana dan lebih mudah diskalakan dari A3C.

**A3C: asynchronous advantage actor-critic.** Mnih et al. (2016). Lahirkan `N` thread pekerja, masing-masing menjalankan satu lingkungan. Setiap pekerja menghitung gradien secara lokal atas rollout-nya sendiri, lalu menerapkannya secara asinkron ke server parameter bersama. Tanpa replay buffer — pekerja menghilangkan korelasi dengan menjalankan trajektori yang berbeda. A3C membuktikan bahwa kamu bisa melatih di CPU berskala besar. Pada 2026, A2C berbasis GPU (lingkungan paralel ter-batch) mendominasi karena GPU menginginkan batch besar.

**Loss gabungannya.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

Tiga istilah: loss policy-gradient, regresi nilai, bonus entropi. `c_v ~ 0.5`, `c_e ~ 0.01` adalah titik awal kanonik.

```figure
actor-critic
```

## Bangun Sendiri

### Langkah 1: seorang kritikus

Kritikus linear `V_φ(s) = w · features(s)` diupdate dengan MSE:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

Di lingkungan tabular, kritikus konvergen dalam beberapa ratus episode. Di Atari, ganti kritikus linear dengan trunk CNN bersama + kepala nilai.

### Langkah 2: n-step advantage

Diberikan rollout berpanjang `T` dan `V(s_T)` ber-bootstrap akhir:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` adalah target kritikus. `advantages` adalah yang mengalikan `∇ log π`.

### Langkah 3: update gabungan

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy, satu rollout per update, learning rate terpisah untuk aktor dan kritikus.

### Langkah 4: paralelisasi (A3C vs A2C)

- **A3C:** jalankan `N` thread. Masing-masing menjalankan lingkungan dan forward pass-nya sendiri. Sesekali dorong update gradien ke master bersama. Tanpa kunci di master — balapan itu tidak apa-apa, hanya menambah noise.
- **A2C:** jalankan `N` instance lingkungan dalam satu proses, tumpuk observasi menjadi batch `[N, obs_dim]`, forward pass ter-batch, backward pass ter-batch. Utilisasi GPU lebih tinggi, deterministik, lebih mudah direasoningkan. Default pada 2026.

Kode mainan kita satu-thread agar jelas; menulisnya ulang menjadi A2C ter-batch hanya tiga baris numpy.

## Jebakan

- **Bias kritikus sebelum gradien aktor.** Jika kritikusnya acak, baseline-nya tidak informatif dan kamu melatih pada noise murni. Panaskan kritikus selama beberapa ratus langkah sebelum menyalakan policy gradient, atau pakai learning rate aktor yang lambat.
- **Advantage normalization.** Normalkan advantage ke mean-nol/std-satu per batch. Menstabilkan training secara masif dengan biaya nyaris nol.
- **Trunk bersama.** Pakai ekstraktor fitur bersama untuk aktor dan kritikus pada input gambar. Kepala terpisah. Fitur bersama itu menumpang gratis di kedua loss.
- **Kontrak on-policy.** A2C memakai ulang data tepat untuk satu update. Lebih dari itu dan gradiennya bias (koreksi importance-sampling adalah yang ditambahkan PPO).
- **Runtuhan entropi.** Tanpa `c_e > 0`, kebijakan menjadi nyaris-deterministik dalam beberapa ratus update dan berhenti menjelajah.
- **Skala reward.** Magnitudo advantage bergantung pada skala reward. Normalkan reward (mis. dibagi running-std) agar magnitudo gradien konsisten antar tugas.

## Gunakan

A2C/A3C jarang menjadi pilihan akhir pada 2026 tetapi mereka adalah arsitektur yang disempurnakan semua hal belakangan:

| Method | Relation to A2C |
|--------|----------------|
| PPO | A2C + clipped importance ratio for multi-epoch updates |
| IMPALA | A3C + V-trace off-policy correction |
| SAC (Phase 9 · 07) | Off-policy A2C with a soft-value critic (next lesson) |
| GRPO (Phase 9 · 12) | A2C without the critic — group-relative advantage |
| DPO | A2C collapsed into a preference-ranking loss, no sampling |
| AlphaStar / OpenAI Five | A2C with league training + imitation pre-training |

Jika kamu melihat "advantage" di makalah 2026, pikirkan actor-critic.

## Rilis

Simpan sebagai `outputs/skill-actor-critic-trainer.md`:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## Latihan

1. **Mudah.** Latih actor-critic dengan MC advantage (`G_t - V(s_t)`) pada GridWorld 4×4. Bandingkan efisiensi sampel dengan REINFORCE-plus-baseline-mean-berjalan dari Pelajaran 06.
2. **Sedang.** Beralih ke advantage TD-residual (`r + γ V(s') - V(s)`). Ukur varians dari batch advantage. Seberapa besar penurunannya?
3. **Sulit.** Implementasikan GAE(λ). Sweep `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`. Plot return akhir versus efisiensi sampel. Di mana titik manis bias/varians untuk tugas ini?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Actor | "The policy net" | `π_θ(a\|s)`, updated by policy gradient. |
| Critic | "The value net" | `V_φ(s)`, updated by MSE regression to returns / TD targets. |
| Advantage | "How much better than average" | `A(s, a) = Q(s, a) - V(s)` or its estimators. Multiplier for `∇ log π`. |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`; one-step advantage estimate. |
| GAE | "The interpolation knob" | Exponentially weighted sum of n-step advantages, parameterized by `λ`. |
| A2C | "Synchronous actor-critic" | Batched across envs; one gradient step per rollout. |
| A3C | "Async actor-critic" | Worker threads push gradients to a shared param server. Original paper; less common in 2026. |
| Bootstrap | "Use V at the horizon" | Truncate the rollout, add `γ^n V(s_{t+n})` to close the sum. |

## Bacaan Lanjutan

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, the original async actor-critic paper.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — foundations; pair this with Ch. 9 on function approximation when the critic is a neural net.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — scalable distributed actor-critic with V-trace off-policy correction.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — production A2C/PPO implementations worth reading.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — the foundational convergence result for the two-timescale actor-critic decomposition.
