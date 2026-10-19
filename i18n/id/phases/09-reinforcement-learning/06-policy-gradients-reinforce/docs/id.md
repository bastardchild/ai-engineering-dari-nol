# Policy Gradient — REINFORCE dari Nol

> Berhenti mengecek nilai. Parameterisasi kebijakan secara langsung, hitung gradien dari return yang diharapkan, naik ke atas. Williams (1992) menulisnya dalam satu teorema. Itulah alasan PPO, GRPO, dan setiap loop LLM RL ada.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 minutes

## Masalahnya

Q-learning dan DQN memparameterisasi fungsi *nilai*. Kamu memilih aksi lewat `argmax Q`. Itu baik untuk aksi diskrit dan state diskrit. Ia rusak saat aksi bersifat kontinu (mana mungkin `argmax` atas torsi 10 dimensi?) atau saat kamu menginginkan kebijakan stokastik (`argmax` secara konstruksi deterministik).

Policy gradient memparameterisasi *kebijakan* sebagai gantinya. `π_θ(a | s)` adalah jaringan saraf yang mengeluarkan distribusi atas aksi. Ambil sampel darinya untuk bertindak. Hitung gradien dari return yang diharapkan terhadap `θ`. Naik ke atas. Tanpa `argmax`. Tanpa rekursi Bellman. Hanya gradient ascent pada `J(θ) = E_{π_θ}[G]`.

Teorema REINFORCE (Williams 1992) memberitahumu bahwa gradien ini bisa dihitung: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. Jalankan satu episode. Hitung return-nya. Kalikan dengan `∇ log π_θ(a | s)` di setiap langkah. Rata-ratakan. Gradient-ascent. Selesai.

Setiap algoritma LLM-RL pada 2026 — PPO, DPO, GRPO — adalah penyempurnaan dari REINFORCE. Memahaminya sampai ke jari-jarimu adalah prasyarat untuk sisa fase ini, serta untuk Fase 10 · 07 (implementasi RLHF) dan Fase 10 · 08 (DPO).

## Konsepnya

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**Teorema policy gradient.** Untuk kebijakan `π_θ` apa pun yang diparameterisasi oleh `θ`:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

di mana `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` adalah return terdiskon dari langkah `t`. Harapan diambil atas seluruh trajektori `τ` yang disampel dari `π_θ`.

**Buktinya pendek.** Turunkan `J(θ) = Σ_τ P(τ; θ) G(τ)` di bawah harapan. Pakai `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (trik turunan log). Faktorkan `log P(τ; θ) = Σ log π_θ(a_t | s_t) + lingkaran lingkungan yang tidak bergantung pada θ`. Istilah lingkungan lenyap. Dua baris aljabar memberimu teoremanya.

**Trik penurunan varian.** REINFORCE polos punya varian yang kejam — return-nya bising, `∇ log π` bising, hasil kalinya sangat bising. Dua perbaikan standar:

1. **Pengurangan baseline.** Ganti `G_t` dengan `G_t - b(s_t)` untuk baseline `b(s_t)` apa pun yang tidak bergantung pada `a_t`. Tidak bias karena `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Pilihan tipikal: `b(s_t) = V̂(s_t)` yang dipelajari oleh kritikus → actor-critic (Pelajaran 07).
2. **Reward-to-go.** Ganti `Σ_t G_t · ∇ log π_θ(a_t | s_t)` dengan `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`. Hanya return masa depan yang penting bagi aksi tertentu — reward lampau memberikan noise ber-mean nol.

Digabung, kamu dapat:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

yaitu REINFORCE dengan baseline — leluhur langsung dari A2C (Pelajaran 07) dan PPO (Pelajaran 08).

**Parameterisasi kebijakan softmax.** Untuk aksi diskrit, pilihan standarnya:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

di mana `f_θ` adalah jaringan saraf apa pun yang mengeluarkan skor per aksi. Gradiennya punya bentuk yang rapi:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

yaitu, skor aksi yang diambil dikurangi nilai harapannya di bawah kebijakan.

**Kebijakan Gaussian untuk aksi kontinu.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)` punya bentuk tertutup. Itu saja yang dibutuhkan SAC di Fase 9 · 07.

```figure
policy-gradient-landscape
```

## Bangun Sendiri

### Langkah 1: jaringan kebijakan softmax

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Pakai kebijakan linear (satu vektor bobot per aksi) untuk lingkungan tabular. Untuk Atari, tukar dengan CNN dan pertahankan kepala softmax-nya.

### Langkah 2: sampling dan log-probability

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Langkah 3: rollout dengan log-probs yang ditangkap

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Langkah 4: update REINFORCE

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

Gradien `∇ log π(a|s) = e_a - π(·|s)` (onehot dari `a` dikurangi probabilitas) adalah jantung dari policy gradient softmax. Bakar itu sampai jadi ingatan otot.

### Langkah 5: baseline

Mean berjalan dari `G` atas episode terbaru sudah cukup sebagai penurunan varian untuk menjalankan GridWorld 4×4; butuh ~500 episode untuk konvergen. Tingkatkan baseline menjadi `V̂(s)` yang dipelajari dan kamu mendapatkan actor-critic.

## Jebakan

- **Gradien meledak.** Return bisa sangat besar. Selalu normalkan `G` ke `~N(0, 1)` di seluruh batch sebelum mengalikannya dengan `∇ log π`.
- **Runtuhan entropi.** Kebijakan konvergen ke aksi yang nyaris-deterministik terlalu dini, berhenti menjelajah, macet. Perbaikannya: tambahkan bonus entropi `β · H(π(·|s))` ke objektif.
- **Varians tinggi.** REINFORCE polos butuh ribuan episode. Baseline kritikus (Pelajaran 07) atau trust region TRPO/PPO (Pelajaran 08) adalah perbaikan standarnya.
- **Inefisiensi sampel.** On-policy berarti kamu membuang setiap transisi setelah satu update. Koreksi off-policy lewat importance sampling mengembalikan data, dengan biaya varians (rasio PPO adalah bobot IS yang di-clip).
- **Gradien non-stasioner.** Gradien yang sama dari 100 episode lalu memakai `π` lama. Metode on-policy melakukan update setiap beberapa rollout karena alasan ini.
- **Penugasan kredit.** Tanpa reward-to-go, reward lampau memberikan noise. Selalu pakai reward-to-go.

## Gunakan

Pada 2026, REINFORCE jarang dijalankan secara langsung tetapi rumus gradiennya ada di mana-mana:

| Use case | Derived method |
|----------|---------------|
| Continuous control | PPO / SAC with Gaussian policy |
| LLM RLHF | PPO with KL penalty, running on token-level policy |
| LLM reasoning (DeepSeek) | GRPO — REINFORCE with group-relative baseline, no critic |
| Multi-agent | Centralized-critic REINFORCE (MADDPG, COMA) |
| Discrete action robotics | A2C, A3C, PPO |
| Preference-only settings | DPO — REINFORCE rewritten as a preference-likelihood loss, no sampling |

Saat kamu membaca `loss = -advantage * log_prob` di skrip training 2026, itulah REINFORCE dengan baseline. Makalah utuh (DPO, GRPO, RLOO) adalah trik penurunan varian di atas baris tunggal ini.

## Rilis

Simpan sebagai `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Latihan

1. **Mudah.** Implementasikan REINFORCE pada GridWorld 4×4 dengan kebijakan softmax linear. Latih selama 1.000 episode tanpa baseline. Plot kurva pembelajarannya; ukur varians (std dari return).
2. **Sedang.** Tambahkan baseline mean berjalan. Latih lagi. Bandingkan efisiensi sampel dan varians dengan run polos. Seberapa besar baseline mengurangi langkah menuju konvergensi?
3. **Sulit.** Tambahkan bonus entropi `β · H(π)`. Sweep `β ∈ {0, 0.01, 0.1, 1.0}`. Plot return akhir dan entropi kebijakan. Di mana titik manisnya pada tugas ini?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy gradient | "Train the policy directly" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; derived from the log-derivative trick. |
| REINFORCE | "The original PG algorithm" | Williams (1992); Monte Carlo returns multiplied by log-policy gradient. |
| Log-derivative trick | "Score function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; makes gradients of expectations tractable. |
| Baseline | "Variance reduction" | Any `b(s)` subtracted from `G`; unbiased because `E[b · ∇ log π] = 0`. |
| Reward-to-go | "Only future returns count" | `G_t^{from t}` instead of the full `G_0`; correct and lower-variance. |
| Entropy bonus | "Encourage exploration" | `+β · H(π(·\|s))` term keeps the policy from collapsing. |
| On-policy | "Train on what you just saw" | Gradient expectation is w.r.t. the current policy — cannot reuse old data directly. |
| Advantage | "How much better than average" | `A(s, a) = G(s, a) - V(s)`; the signed quantity REINFORCE-with-baseline multiplies. |

## Bacaan Lanjutan

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — the original REINFORCE paper.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — the modern policy-gradient theorem with function approximation.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — textbook presentation.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — clear pedagogical exposition with PyTorch code.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — variance-reduction and the natural-gradient view that connects REINFORCE to the trust-region family (TRPO, PPO).
