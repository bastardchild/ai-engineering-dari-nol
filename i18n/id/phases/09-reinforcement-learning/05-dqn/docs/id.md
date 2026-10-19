# Deep Q-Networks (DQN)

> 2013: Mnih melatih satu jaringan Q-learning pada piksel mentah, mengalahkan setiap agen RL klasik pada tujuh game Atari. 2015: diperluas ke 49 game, diterbitkan di Nature, memicu era deep-RL. DQN adalah Q-learning plus tiga trik yang membuat aproksimasi fungsi stabil.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 minutes

## Masalahnya

Q-learning tabular membutuhkan satu nilai Q terpisah untuk setiap pasangan (state, aksi). Papan catur punya ~10⁴³ state. Satu frame Atari adalah 210×160×3 = 100.800 fitur. RL tabular mati di ribuan state, apalagi miliaran.

Perbaikannya jelas setelah kejadian: ganti tabel Q dengan jaringan saraf, `Q(s, a; θ)`. Tetapi "jelas setelah kejadian" itu memakan puluhan tahun. Aproksimasi fungsi naif dengan Q-learning berdivergensi di bawah "trikad mematikan" — aproksimasi fungsi + bootstrapping + pembelajaran off-policy. Mnih et al. (2013, 2015) mengidentifikasi tiga trik rekayasa yang menstabilkan pembelajaran:

1. **Experience replay** menghilangkan korelasi antar transisi.
2. **Target network** membekukan target bootstrap.
3. **Reward clipping** menormalkan magnitudo gradien.

DQN di Atari adalah pertama kalinya satu arsitektur dengan satu set hyperparameter memecahkan puluhan masalah kontrol dari piksel mentah. Segala sesuatu tentang "deep-RL" yang dibangun sejak itu — DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 — ditumpuk di atas dasar tiga-trik ini.

## Konsepnya

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**Objektifnya.** DQN meminimalkan loss TD satu langkah pada fungsi Q saraf:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = jaringan online, diupdate setiap langkah oleh gradient descent. `θ^-` = jaringan target, disalin secara periodik dari `θ` (setiap ~10.000 langkah). `D` = buffer replay dari transisi masa lalu.

**Tiga trik itu, berdasarkan urutan pentingnya:**

**Experience replay.** Ring buffer berisi `~10⁶` transisi. Setiap langkah training mengambil sampel minibatch secara acak seragam. Ini memutus korelasi temporal (frame berurutan hampir identik), memungkinkan jaringan belajar dari transisi ber-reward langka berkali-kali, dan menghilangkan korelasi antar update gradien berurutan. Tanpanya, TD on-policy dengan jaringan saraf berdivergensi di Atari.

**Target network.** Memakai jaringan yang sama `Q(·; θ)` di kedua sisi persamaan Bellman membuat target bergerak setiap update — "mengejar ekor sendiri." Perbaikannya: simpan jaringan kedua `Q(·; θ^-)` dengan bobot beku. Setiap `C` langkah, salin `θ → θ^-`. Ini menstabilkan target regresi untuk ribuan langkah gradien sekaligus. Update lunak `θ^- ← τ θ + (1-τ) θ^-` (dipakai di DDPG, SAC) adalah varian yang lebih halus.

**Reward clipping.** Magnitudo reward Atari bervariasi dari 1 sampai 1000+. Clipping ke `{-1, 0, +1}` mencegah satu game mana pun mendominasi gradien. Salah ketika magnitudo reward itu penting; baik untuk Atari di mana hanya tanda yang penting.

**Double DQN.** Hasselt (2016) memperbaiki bias maksimum: pakai jaringan online untuk *memilih* aksi, jaringan target untuk *mengevaluasi* aksi itu.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

Pengganti langsung, konsisten lebih baik. Pakai secara default.

**Peningkatan lain (Rainbow, 2017):** replay berprioritas (ambil sampel transisi ber-TD-error tinggi lebih sering), arsitektur dueling (kepala `V(s)` dan advantage terpisah), jaringan noisy (eksplorasi yang dipelajari), return n-step, Q distribusional (C51/QR-DQN), bootstrapping multi-langkah. Masing-masing menambahkan beberapa persen; keuntungannya kira-kira aditif.

```figure
f3-dqn-stability
```

## Bangun Sendiri

Kode di sini murni stdlib tanpa numpy — kita memakai MLP satu-hidden-layer buatan tangan pada GridWorld kontinu kecil, sehingga setiap langkah training berjalan dalam mikrodetik. Algoritmanya identik dengan DQN Atari berskala besar.

### Langkah 1: replay buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Kapasitas ~50.000 untuk Atari; 5.000 sudah cukup untuk lingkungan mainan kita.

### Langkah 2: Q-network kecil (MLP manual)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Forward pass: linear → ReLU → linear. Itu seluruh jaringannya.

### Langkah 3: update DQN

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

Bentuknya adalah Q-learning dari Pelajaran 04 dengan dua perbedaan: (a) kita melakukan backprop melalui `Q(·; θ)` yang dapat diturunkan alih-alih mengindeks tabel, (b) target memakai `Q(·; θ^-)`.

### Langkah 4: loop luar

Untuk setiap episode, bertindak ε-greedy pada `Q(·; θ)`, dorong transisi ke buffer, ambil sampel minibatch, ambil satu langkah gradien, sinkronkan secara periodik `θ^- ← θ`. Polanya:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

Pada GridWorld kecil kita dengan state one-hot 16-dim, agen mempelajari kebijakan nyaris-optimal dalam ~500 episode. Di Atari, skalakan ini ke 200M frame dan tambahkan ekstraktor fitur CNN.

## Jebakan

- **Trikad mematikan.** Aproksimasi fungsi + off-policy + bootstrapping bisa berdivergensi. DQN meredakannya dengan target net + replay; jangan menghapus salah satunya.
- **Eksplorasi.** ε harus meluruh, biasanya dari 1.0 ke 0.01 selama ~10% pertama training. Tanpa eksplorasi awal yang cukup, Q-net konvergen ke satu basin lokal.
- **Overestimation.** `max` atas Q yang bising bias ke atas. Selalu pakai Double DQN di produksi.
- **Skala reward.** Clip atau normalkan reward; magnitudo gradien sebanding dengan magnitudo reward.
- **Coldstart replay buffer.** Jangan training sampai buffer punya beberapa ribu transisi. Gradien awal pada ~20 sampel akan overfit.
- **Frekuensi sinkron target.** Terlalu sering ≈ tak ada target net; terlalu jarang ≈ target basi. DQN Atari memakai 10.000 langkah env. Aturan praktis: sinkronkan setiap ~1/100 dari horizon training.
- **Pra-pemrosesan observasi.** DQN Atari menumpuk 4 frame agar state bersifat Markov. Lingkungan mana pun dengan informasi kebutuhan frame-stacking atau state rekuren.

## Gunakan

Pada 2026, DQN jarang menjadi state-of-the-art tetapi tetap merupakan algoritma off-policy rujukan:

| Task | Method of choice | Why not DQN? |
|------|------------------|--------------|
| Discrete-action Atari-like | Rainbow DQN or Muesli | Same framework, more tricks. |
| Continuous control | SAC / TD3 (Phase 9 · 07) | DQN has no policy network. |
| On-policy / high-throughput | PPO (Phase 9 · 08) | No replay buffer; easier to scale. |
| Offline RL | CQL / IQL / Decision Transformer | Conservative Q targets, no bootstrapping blowups. |
| Large discrete action spaces (recommender) | DQN with action embedding, or IMPALA | Fine; decoration matters. |
| LLM RL | PPO / GRPO | Sequence-level, not step-level; different loss. |

Pelajarannya masih terbawa. Replay dan target network muncul di SAC, TD3, DDPG, SAC-X, buffer self-play AlphaZero, dan setiap metode offline RL. Reward clipping hidup sebagai advantage normalization di PPO. Arsitekturnya adalah cetak birunya.

## Rilis

Simpan sebagai `outputs/skill-dqn-trainer.md`:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Plot kurva return per episode. Berapa episode sampai mean berjalan melebihi -10?
2. **Sedang.** Nonaktifkan target network (pakai jaringan online untuk kedua sisi target Bellman). Ukur ketidakstabilan training — apakah return berfluktuasi atau berdivergensi?
3. **Sulit.** Tambahkan Double DQN: pakai jaringan online untuk memilih `argmax a'`, jaringan target untuk mengevaluasi. Bandingkan bias `Q(s_0, best_a)` versus `V*(s_0)` yang sebenarnya setelah 1.000 episode dengan versus tanpa Double DQN pada GridWorld ber-reward-noisy.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| DQN | "Deep Q-learning" | Q-learning with a neural Q-function, replay buffer, and target network. |
| Experience replay | "Shuffled transitions" | Ring buffer sampled uniformly each gradient step; decorrelates data. |
| Target network | "Frozen bootstrap" | Periodic copy of Q used in the Bellman target; stabilizes training. |
| Deadly triad | "Why RL diverges" | Function approximation + bootstrapping + off-policy = no convergence guarantee. |
| Double DQN | "Fix for maximization bias" | Online net selects action, target net evaluates it. |
| Dueling DQN | "V and A heads" | Decompose Q = V + A - mean(A); same output, better gradient flow. |
| Rainbow | "All the tricks" | DDQN + PER + dueling + n-step + noisy + distributional in one. |
| PER | "Prioritized Replay" | Sample transitions proportional to TD-error magnitude. |

## Bacaan Lanjutan

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — the 2013 NeurIPS workshop paper that kicked off deep RL.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — the Nature paper, 49-game DQN.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — the stacked-tricks paper.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — the textbook treatment of the "deadly triad" (function approximation + bootstrapping + off-policy) that DQN's target network and replay buffer are designed to tame.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — reference single-file DQN used in ablation studies; good to read alongside this lesson's from-scratch version.
