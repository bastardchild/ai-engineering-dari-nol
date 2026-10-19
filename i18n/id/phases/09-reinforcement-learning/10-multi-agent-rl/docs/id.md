# Multi-Agent RL

> RL asumsi-agen tunggal menganggap environment bersifat stasioner. Letakkan dua agen pembelajar di dunia yang sama dan asumsi itu patah: setiap agen adalah bagian dari environment agen lainnya, dan keduanya berubah. Multi-agent RL adalah kumpulan trik untuk membuat pembelajaran konvergen ketika asumsi Markov tidak lagi berlaku.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45 minutes

## Masalahnya

Robot yang belajar menavigasi ruangan adalah masalah RL asen-tunggal. Tim sepak bola bukan. AlphaStar vs lawan StarCraft bukan. Marketplace agen yang menawar bukan. Dua mobil yang bernegosiasi di perempatan empat arah bukan. Masalah dunia-nyata banyak-saling-bukan.

Di setiap setting multi-agent, dari perspektif satu agen, agen lainnya *adalah* bagian dari environment. Saat mereka belajar dan mengubah perilaku mereka, environment menjadi non-stationary. Sifat Markov — "state berikutnya hanya bergantung pada state saat ini dan aksiku" — dilanggar karena state berikutnya juga bergantung pada apa yang dipilih agen *lain*, dan policy mereka adalah target yang bergerak.

Ini merusak bukti konvergensi tabel (jaminan Q-learning mengasumsikan environment stasioner). Ini juga merusak deep RL naif: agen mengejar satu sama lain dalam loop, tidak pernah konvergen ke policy yang stabil. Kamu membutuhkan teknik khusus multi-agent: centralized training / decentralized execution, counterfactual baselines, league play, self-play.

Aplikasi 2026: robot swarms, penyebaran lalu lintas, armada kendaraan otonom, simulator pasar, sistem LLM multi-agent (Phase 16), dan setiap game dengan lebih dari satu pemain cerdas.

## Konsepnya

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**Formalisme: Markov Game.** Generalisasi dari MDP: states `S`, joint action `a = (a_1, …, a_n)`, transition `P(s' | s, a)`, dan rewards per-agen `R_i(s, a, s')`. Setiap agen `i` memaksimalkan return-nya sendiri di bawah policy `π_i`. Jika rewards identik, itu **fully cooperative**. Jika zero-sum, itu **adversarial**. Jika campuran, itu **general-sum**.

**Tantangan inti:**

- **Non-stationarity.** `P(s' | s, a_i)` dari sudut pandang agen `i` bergantung pada `π_{-i}` yang berubah.
- **Credit assignment.** Dengan reward bersama, agen mana yang menyebabkannya?
- **Koordinasi eksplorasi.** Agen harus mengeksplorasi strategi komplementer, bukan mengeksplorasi state yang sama secara redundan.
- **Skalabilitas.** Ruang aksi joint tumbuh eksponensial dalam `n`.
- **Partial observability.** Setiap agen hanya melihat observasinya sendiri; state global tersembunyi.

**Empat rezim dominan:**

**1. Independent Q-learning / independent PPO (IQL, IPPO).** Setiap agen belajar Q atau policy sendiri, memperlakukan lainnya sebagai bagian dari environment. Sederhana, kadang berhasil (terutama dengan experience replay yang bertindak sebagai trik smoothing agent-modeling). Konvergensi teoretis: tidak ada. Dalam praktik: baik untuk tugas longgar-terhubung, buruk untuk yang erat-terhubung.

**2. Centralized training, decentralized execution (CTDE).** Paradigma modern paling umum. Setiap agen memiliki *policy*-nya sendiri `π_i` yang mengkondisikan pada observasi lokal `o_i` — decentralized execution standar saat deployment. Selama *training*, centralized critic `Q(s, a_1, …, a_n)` mengkondisikan pada state global penuh dan joint action. Contoh:
- **MADDPG** (Lowe et al. 2017): DDPG dengan centralized critic per agen.
- **COMA** (Foerster et al. 2017): counterfactual baseline — tanya "seperti apa reward-ku jika aku mengambil aksi `a'`?" — mengisolasi kontribusiku.
- **MAPPO** / **IPPO** dengan shared critic (Yu et al. 2022): PPO dengan centralized value function. Dominan di 2026 untuk MARL kooperatif.
- **QMIX** (Rashid et al. 2018): value decomposition — `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` dengan monotonic mixing.

**3. Self-play.** Dua salinan agen yang sama bermain melawan satu sama lain. Policy lawan *adalah* policy-ku dari snapshot masa lalu. AlphaGo / AlphaZero / MuZero. OpenAI Five. Paling baik untuk game zero-sum; sinyal training-nya simetris.

**4. League play.** Perpanjangan self-play ke environment general-sum / adversarial: simpan populasi policy masa lalu dan saat ini, sample lawan dari liga, latih melawan mereka. Tambahkan exploaters (spesialis mengalahkan yang terbaik saat ini) dan main exploaters (spesialis mengalahkan exploaters). AlphaStar (StarCraft II). Dibutuhkan ketika game mengakui siklus strategi "batu-kertas-gunting".

**Komunikasi.** Izinkan agen mengirim pesan yang dipelajari `m_i` satu sama lain. Berhasil di setting kooperatif. Foerster et al. (2016) menunjukkan bahwa komunikasi inter-agen yang dapat dibedakan dapat dilatih end-to-end. Sistem multi-agent berbasis LLM hari ini (Phase 16) pada dasarnya berkomunikasi dalam bahasa alami.

```figure
f3-marl-orbit
```

## Bangun Sendiri

Pelajaran ini menggunakan GridWorld 6×6 dengan dua agen kooperatif. Mereka mulai di sudut berlawanan dan harus mencapai tujuan bersama. Reward bersama: `-1` per langkah sementara salah satu agen masih bergerak, `+10` ketika keduanya tiba. Lihat `code/main.py`.

### Langkah 1: env multi-agent

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

Ruang aksi *joint* adalah `|A|² = 16`. State global adalah dua posisi.

### Langkah 2: independent Q-learning

Setiap agen menjalankan Q-table-nya sendiri yang di-key pada joint state. Pada setiap langkah: keduanya memilih aksi ε-greedy, mengumpulkan joint transisi, masing-masing meng-update Q-nya dengan reward bersama.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Berhasil pada tugas ini karena rewards padat dan selaras. Gagal pada tugas erat-terhubung (mis., di mana satu agen harus *menunggu* yang lain).

### Langkah 3: Q terpusat dengan update decomposed-value

Gunakan satu Q atas joint actions `Q(s, a_1, a_2)`. Update dari reward bersama. Desentralisasi saat eksekusi dengan marginalisasi: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. Menukar ruang aksi joint eksponensial dengan tampilan global yang *benar*.

### Langkah 4: self-play sederhana (adversarial 2-agen)

Agen yang sama, dua peran. Latih agen A melawan agen B; setelah `K` episodes, salin bobot A ke B. Training simetris, kemajuan konsisten. Resep AlphaZero dalam miniatur.

## Jebakan

- **Replay non-stationary.** Experience replay dengan agen independen lebih buruk daripada asen-tunggal karena transisi lama dihasilkan oleh lawan yang kini usang. Perbaikan: relabel atau beri bobot berdasarkan kebaruan.
- **Ambiguitas credit assignment.** Reward bersama setelah episode panjang; tidak ada cara jelas mengatakan agen mana yang berkontribusi. Perbaikan: counterfactual baselines (COMA), atau reward shaping per agen.
- **Policy drift / mengejar.** Best response setiap agen berubah dengan setiap update keduanya. Perbaikan: centralized critic, learning rate lambat, atau bekukan-satu-per-satu.
- **Reward hacking via koordinasi.** Agen menemukan eksploitasi terkoordinasi yang tidak diantisipasi perancang. Agen lelang konvergen ke penawaran nol. Perbaikan: desain reward yang hati-hati, batasan perilaku.
- **Redundansi eksplorasi.** Kedua agen mengeksplorasi pasangan state-action yang sama. Perbaikan: bonus entropy per-agen, atau role-conditioning.
- **Siklus liga.** Self-play murni bisa tersangkut dalam siklus dominasi. Perbaikan: league play dengan lawan beragam.
- **Eksplosi sampel.** `n` agen × ruang state × joint actions. Aproksimasi dengan function approximation; ruang aksi yang terfaktoris (satu policy output head per agen).

## Gunakan

Peta aplikasi MARL 2026:

| Domain | Method | Notes |
|--------|--------|-------|
| Cooperative navigation / manipulation | MAPPO / QMIX | CTDE; shared critic + decentralized actors. |
| Two-player games (chess, Go, poker) | Self-play with MCTS (AlphaZero) | Zero-sum; symmetric training. |
| Complex multiplayer (Dota, StarCraft) | League play + imitation pretraining | OpenAI Five, AlphaStar. |
| Autonomous-vehicle fleets | CTDE MAPPO / PPO with attention | Partial obs; variable team sizes. |
| Auction markets | Game-theoretic equilibrium + RL | Mean-field RL when `n` → ∞. |
| LLM multi-agent systems (Phase 16) | Natural-language comm + role conditioning | RL loop at the agent-planning layer. |

Di 2026, area pertumbuhan terbesar MARL adalah berbasis LLM: swarms dari agen bahasa-model yang bernegosiasi, berdebat, membangun software. RL-nya muncul sebagai optimasi preferensi pada output *trajectory-level*, bukan token-level (Phase 16 · 03).

## Rilis

Simpan sebagai `outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Latihan

1. **Mudah.** Latih independent Q-learning pada GridWorld kooperatif 2-agen. Berapa episodes sampai return rata-rata > 0? Plot kurva belajar joint.
2. **Sedang.** Tambahkan tugas "koordinasi": tujuan tercapai hanya ketika kedua agen melangkah ke atasnya pada giliran yang sama. Apakah independent Q masih konvergen? Apa yang patah?
3. **Sulit.** Implementasikan centralized critic untuk training gaya MAPPO dan bandingkan kecepatan konvergensi dengan independent PPO pada tugas koordinasi.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Markov game | "Multi-agent MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; each agent has its own reward. |
| CTDE | "Centralized training, decentralized execution" | Joint critic at training time; each agent's policy uses only local obs. |
| IPPO | "Independent PPO" | Each agent runs PPO separately. Simple baseline; often underrated. |
| MAPPO | "Multi-agent PPO" | PPO with a centralized value function conditioned on global state. |
| QMIX | "Monotonic value decomposition" | `Q_tot = f_monotone(Q_1, …, Q_n)` allows decentralized argmax. |
| COMA | "Counterfactual multi-agent" | Advantage = my Q minus expected Q marginalizing over my action. |
| Self-play | "Agent vs past self" | Single agent, two roles; standard for zero-sum games. |
| League play | "Population training" | Cache past policies, sample opponents from the pool; handles strategy cycles. |

## Bacaan Lanjutan

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — CTDE with a centralized critic.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — counterfactual baselines for credit assignment.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — value decomposition with monotonicity.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO is surprisingly strong for MARL.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — league play at scale.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — pure self-play in zero-sum games.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — includes the textbook's short treatment of multi-agent settings and the non-stationarity problem that CTDE is designed to solve.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — survey covering cooperative, competitive, and mixed MARL with convergence results.
