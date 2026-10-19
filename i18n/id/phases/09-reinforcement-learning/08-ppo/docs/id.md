# Proximal Policy Optimization (PPO)

> A2C membuang setiap rollout setelah satu update. PPO membungkus policy gradient dalam rasio importance ber-clip sehingga kamu bisa melakukan 10+ epoch pada data yang sama tanpa policy meledak. Schulman et al. (2017). Masih algoritma policy-gradient default di 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 minutes

## Masalahnya

A2C (Lesson 07) adalah on-policy: gradien `E_{π_θ}[A · ∇ log π_θ]` membutuhkan data yang di-sample dari `π_θ` *saat ini*. Lakukan satu update, dan `π_θ` berubah; data yang kamu gunakan kini menjadi off-policy. Gunakan ulang dan gradienmu akan bias.

Rollout mahal. Di Atari, satu rollout melintasi 8 envs × 128 steps = 1024 transisi dan belasan detik waktu environment. Membuangnya setelah satu gradient step adalah pemborosan.

Trust Region Policy Optimization (TRPO, Schulman 2015) adalah perbaikan pertama: batasi setiap update sehingga KL divergence antara policy lama dan baru tetap di bawah `δ`. Secara teori bersih, tetapi membutuhkan conjugate-gradient solve per update. Tidak ada yang menjalankan TRPO di 2026.

PPO (Schulman et al. 2017) menggantikan batas trust-region keras dengan objective ber-clip sederhana. Satu baris kode tambahan. Sepuluh epoch per rollout. Tanpa conjugate gradients. Jaminan teori yang cukup baik. Sembilan tahun kemudian ia masih algoritma policy-gradient default untuk semuanya dari MuJoCo hingga RLHF.

## Konsepnya

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**Rasio importance.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

Ini adalah likelihood ratio dari policy baru vs policy yang mengumpulkan data. `r_t = 1` berarti tidak ada perubahan. `r_t = 2` berarti policy baru dua kali lebih mungkin mengambil `a_t` dibanding yang lama.

**Surrogate yang di-clip.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Dua istilah:

- Jika advantage `A_t > 0` dan rasio mencoba tumbuh melewati `1 + ε`, clip meratakan gradien — jangan mendorong aksi yang baik lebih jauh dari `+ε` di atas probabilitas lama.
- Jika advantage `A_t < 0` dan rasio mencoba tumbuh melewati `1 - ε` (artinya kita akan membuat aksi yang buruk lebih mungkin dibanding reduksi ber-clip-nya), clip membatasi gradien — jangan mendorong aksi yang buruk di bawah `-ε`.

`min` menangani arah lain: jika rasio sudah bergerak ke arah yang *menguntungkan*, kamu tetap mendapat gradien (tanpa clipping di sisi yang akan menyakitimu).

Tipikal `ε = 0.2`. Plot objective sebagai fungsi dari `r_t`: fungsi piecewise-linear dengan atap datar di "sisi baik" dan lantai datar di "sisi buruk."

**Loss PPO lengkap.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

Struktur actor-critic sama dengan A2C. Tiga koefisien, biasanya `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`.

**Loop training.**

1. Kumpulkan `N × T` transisi di `N` envs paralel selama `T` steps masing-masing.
2. Hitung advantages (GAE), bekukan sebagai konstanta.
3. Bekukan `π_{θ_old}` sebagai snapshot dari `π_θ` saat ini.
4. Selama `K` epochs, untuk setiap minibatch dari `(s, a, A, V_target, log π_old(a|s))`:
   - Hitung `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`.
   - Terapkan `L^{CLIP}` + value loss + entropy.
   - Gradient step.
5. Buang rollout. Kembali ke step 1.

`K = 10` dan minibatches of 64 adalah set hyperparameter standar. PPO tangguh: angka pastinya jarang penting dalam rentang ±50%.

**Varian KL-penalty.** Paper asli mengusulkan alternatif dengan KL penalty adaptif: `L = L^{PG} - β · KL(π_θ || π_old)` dengan `β` disesuaikan berdasarkan KL yang diamati. Versi clipping menjadi dominan; varian KL bertahan di RLHF (di mana KL ke reference policy adalah constraint terpisah yang selalu kamu inginkan).

```figure
ppo-clip
```

## Bangun Sendiri

### Langkah 1: rekam `log π_old(a | s)` saat rollout

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

Snapshot diambil sekali, saat rollout. Ia tidak berubah selama epoch update.

### Langkah 2: hitung advantages GAE (Lesson 07)

Sama dengan A2C. Normalisasi di seluruh batch.

### Langkah 3: update clipped surrogate

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

Pola "clipped → zero gradient" adalah inti PPO. Jika policy baru sudah menyimpang terlalu jauh ke arah yang menguntungkan, update berhenti.

### Langkah 4: value dan entropy

Tambahkan MSE standar ke target critic dan bonus entropy pada actor, sama seperti A2C.

### Langkah 5: diagnostik

Tiga hal yang dipantau setiap update:

- **Mean KL** `E[log π_old - log π_θ]`. Harus tetap di `[0, 0.02]`. Jika meledak melewati `0.1`, kurangi `K_EPOCHS` atau `LR`.
- **Clip fraction** — fraksi sampel yang rasio-nya di luar `[1-ε, 1+ε]`. Harus `~0.1-0.3`. Jika `~0`, clip tidak pernah terpicu → naikkan `LR` atau `K_EPOCHS`. Jika `~0.5+`, kamu over-fit pada rollout → turunkan.
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`. Metrik kualitas critic. Harus naik menuju 1 saat critic belajar.

## Jebakan

- **Koefisien clip salah set.** `ε = 0.2` adalah standar de-facto. Ke `0.1` membuat update terlalu takut-takut; `0.3+` mengundang ketidakstabilan.
- **Terlalu banyak epochs.** `K > 20` sering menstabilkan karena policy menyimpang jauh dari `π_old`. Batasi epochs, terutama untuk jaringan besar.
- **Tanpa normalisasi reward.** Skala reward besar menggerogoti rentang clip. Normalisasi rewards (running std) sebelum menghitung advantages.
- **Lupa normalisasi advantage.** Normalisasi zero-mean/unit-std per-batch adalah standar. Melewatkannya merusak PPO di sebagian besar benchmarks.
- **Learning rate tidak di-decay.** PPO diuntungkan oleh linear LR decay ke nol. LR konstan sering lebih buruk.
- **Kesalahan matematika rasio importance.** Selalu `exp(log_new - log_old)` untuk stabilitas numerik, bukan `new / old`.
- **Tanda gradien salah.** Maksimalkan surrogate = *minimalkan* `-L^{CLIP}`. Tanda terbalik adalah bug PPO paling umum.

## Gunakan

PPO adalah algoritma RL default 2026 di jumlah domain yang mengejutkan:

| Use case | PPO variant |
|----------|-------------|
| MuJoCo / robotics control | PPO with Gaussian policy, GAE(0.95) |
| Atari / discrete games | PPO with categorical policy, rolling 128-step rollouts |
| RLHF for LLMs | PPO with KL penalty to reference model, reward from RM at end of response |
| Large-scale game agents | IMPALA + PPO (AlphaStar, OpenAI Five) |
| Reasoning LLMs | GRPO (Lesson 12) — PPO variant without critic |
| Preference-only data | DPO — closed-form collapsing of PPO+KL, no online sampling |

*Loss shape* PPO — clipped surrogate + value + entropy — adalah kerangka untuk DPO, GRPO, dan hampir setiap pipeline RLHF.

## Rilis

Simpan sebagai `outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Latihan

1. **Mudah.** Jalankan PPO pada 4×4 GridWorld dengan `ε=0.2, K=4`. Bandingkan sample efficiency dengan A2C (satu epoch per rollout) pada langkah env yang sama.
2. **Sedang.** Sweep `K ∈ {1, 4, 10, 30}`. Plot return vs langkah env dan lacak mean KL per update. Pada `K` berapa KL meledak di tugas ini?
3. **Sulit.** Ganti clipped surrogate dengan KL penalty adaptif (`β` digandakan jika `KL > 2·target`, di-setengah jika `KL < target/2`). Bandingkan return akhir, stabilitas, dan bebas-clip-nya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Importance ratio | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; deviation from the policy that collected the data. |
| Clipped surrogate | "PPO's main trick" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; flat gradient past the clip on beneficial side. |
| Trust region | "TRPO / PPO intent" | Limit each update's KL to guarantee monotone improvement. |
| KL penalty | "Soft trust region" | Alternative PPO: `L - β · KL(π_θ \|\| π_old)`. Adaptive `β`. |
| Clip fraction | "How often clipping triggers" | Diagnostic — should be 0.1-0.3; outside means mistuned. |
| Multi-epoch training | "Data reuse" | K epochs on each rollout; variance cost traded for sample efficiency. |
| On-policy-ish | "Mostly on-policy" | PPO is nominally on-policy but K>1 epochs uses slightly-off-policy data safely. |
| PPO-KL | "The other PPO" | KL-penalty variant; used in RLHF where KL-to-reference is already a constraint. |

## Bacaan Lanjutan

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — the paper.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, PPO's predecessor.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — every PPO hyperparameter ablated.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; the PPO-in-RLHF recipe.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — clean modern exposition with PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — reference single-file PPO used by many papers.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — the production recipe for PPO on language models; read alongside Lesson 09 (RLHF).
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — the "37 code-level optimizations" paper; which PPO tricks are load-bearing and which are folklore.
