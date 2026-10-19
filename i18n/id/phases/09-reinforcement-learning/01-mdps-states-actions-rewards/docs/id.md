# MDP, State, Action & Reward

> Markov Decision Process adalah lima hal: state, action, transisi, reward, diskon. Segala hal dalam RL — Q-learning, PPO, DPO, GRPO — mengoptimasi di atas bentuk ini. Pelajari sekali, baca sisa reinforcement learning secara gratis.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 minutes

## Masalahnya

Kamu sedang menulis bot catur. Atau perencana inventaris. Atau agen trading. Atau loop PPO yang melatih model reasoning. Empat domain berbeda, satu fakta mengejutkan: keempatnya runtuh menjadi objek matematis yang sama.

Supervised learning memberimu pasangan `(x, y)` dan memintamu memfit fungsi. Reinforcement learning tidak memberimu label — hanya stream state, action yang kamu ambil, dan reward skalar. Apakah langkah itu memenangkan permainan? Apakah keputusan restock menghemat uang? Apakah trade menghasilkan profit? Apakah token yang baru dihasilkan LLM mengarah ke reward lebih tinggi dari juri?

Kamu tidak bisa belajar dari stream ini sampai kamu memformalkannya. "Apa yang kulihat," "apa yang kulakukan," "apa yang terjadi berikutnya," "seberapa bagus itu" — masing-masing harus menjadi objek yang bisa kamu nalari. Formalisasi itu adalah Markov Decision Process. Setiap algoritma RL di fase ini, termasuk loop RLHF dan GRPO di akhir, mengoptimasi di atas bentuk ini.

## Konsepnya

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**Kelima objek.**

- **State** `S`. Segala yang dibutuhkan agen untuk memutuskan. Dalam GridWorld, selnya. Dalam catur, papannya. Dalam LLM, context window plus memori apa pun.
- **Action** `A`. Pilihannya. Bergerak atas/bawah/kiri/kanan. Mainkan langkah. Emisikan token.
- **Transisi** `P(s' | s, a)`. Diberikan state `s` dan action `a`, distribusi atas state berikutnya. Deterministik dalam catur, stokastik dalam inventaris, nyaris-deterministik dalam decoding LLM.
- **Reward** `R(s, a, s')`. Sinyal skalar. Menang = +1, kalah = -1. Pendapatan minus biaya. Suku rasio log-likelihood dalam GRPO.
- **Diskon** `γ ∈ [0, 1)`. Seberapa besar reward masa depan dihitung vs kini. `γ = 0.99` membeli horizon ~100 langkah; `γ = 0.9` membeli ~10.

**Properti Markov** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. Masa depan hanya bergantung pada state kini. Kalau tidak, representasi state tidak lengkap — bukan kegagalan metode, kegagalan state.

**Policy dan return.** Policy `π(a | s)` memetakan state ke distribusi action. Return `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` adalah jumlah diskon reward masa depan. Value `V^π(s) = E[G_t | s_t = s]` adalah return harapan mulai dari `s` di bawah policy `π`. Q-value `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` adalah return harapan mulai dengan action spesifik. Setiap algoritma RL mengestimasi salah satu dari keduanya, lalu memperbaiki `π` dengan sesuai.

**Persamaan Bellman.** Persamaan titik-tetap yang dipakai segala hal di fase ini:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

Ini memecah return harapan menjadi "reward langkah ini" plus "value diskon tempat kamu mendarat." Rekursif. Setiap algoritma di Fase 9 mengiterasikan persamaan ini sampai konvergen (pemrograman dinamis), menyampel darinya (Monte Carlo), atau bootstrap satu langkah (beda temporal).

```figure
discount-horizon
```

## Bangun Sendiri

### Langkah 1: MDP deterministik mungil

GridWorld 4×4. Agen mulai kiri-atas, terminal di kanan-bawah, reward -1 per langkah, action `{up, down, left, right}`. Lihat `code/main.py`.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

Lima baris. Itulah seluruh environment. Transisi deterministik, penalti langkah konstan, state terminal menyerap.

### Langkah 2: gelar policy

Policy adalah fungsi dari state ke distribusi action. Paling sederhana: acak seragam.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

Jalankan policy acak 1000 kali. Return rata-rata sekitar -60 sampai -80 untuk papan 4×4 ini. Return optimal adalah -6 (jalur garis-lurus bawah-kanan). Menutup kesenjangan itu adalah segalanya di Fase 9.

### Langkah 3: hitung `V^π` eksak via persamaan Bellman

Untuk MDP kecil persamaan Bellman adalah sistem linear. Enumerasi state, terapkan ekspektasi, iterasikan sampai value berhenti berubah.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

Ini adalah evaluasi policy iteratif. Ia adalah algoritma pertama dalam Sutton & Barto dan fondasi teoretis setiap metode RL berikutnya.

### Langkah 4: `γ` adalah hiperparameter bermakna-fisik

Horizon efektif kira-kira `1 / (1 - γ)`. `γ = 0.9` → 10 langkah. `γ = 0.99` → 100 langkah. `γ = 0.999` → 1000 langkah.

Terlalu rendah dan agen bertindak miopik. Terlalu tinggi dan atribusi kredit menjadi berisik, karena banyak langkah awal berbagi tanggung jawab atas reward masa-depan yang jauh. RLHF LLM tipikal memakai `γ = 1` karena episode pendek dan terbatas. Task kontrol memakai `0.95–0.99`. Game strategi horizon-panjang memakai `0.999`.

## Jebakan

- **State non-Markovian.** Kalau kamu butuh tiga observasi terakhir untuk memutuskan, "state" bukan hanya observasi kini. Perbaiki: tumpuk frame (DQN pada Atari menumpuk 4) atau pakai state rekuren (LSTM/GRU atas observasi).
- **Reward sparse.** Reward hanya-menang membuat pembelajaran nyaris mustahil di ruang state besar. Bentuk reward (sinyal intermediate) atau bootstrap dengan imitasi (Fase 9 · 09).
- **Reward hacking.** Mengoptimasi reward proksi sering menghasilkan perilaku patologis. Agen balap-perahu OpenAI berputar-putar mengumpulkan powerup selamanya alih-alih menyelesaikan balapan. Selalu definisikan reward dari outcome target, bukan proksinya.
- **Salah-spesifikasi diskon.** `γ = 1` pada task horizon-takhingga membuat setiap value takhingga. Selalu batasi dengan horizon terbatas atau `γ < 1`.
- **Skala reward.** Reward {+100, -100} vs {+1, -1} memberi policy optimal identik tapi magnitudo gradien sangat berbeda. Normalkan ke `[-1, 1]`-ish sebelum mencolokkan ke PPO/DQN.

## Gunakan

Stack 2026 mereduksi setiap pipeline RL menjadi MDP sebelum menyentuh kode:

| Situation | State | Action | Reward | γ |
|-----------|-------|--------|--------|---|
| Control (locomotion, manipulation) | Joint angles + velocities | Continuous torques | Task-specific shaped | 0.99 |
| Games (chess, Go, poker) | Board + history | Legal move | Win=+1 / loss=-1 | 1.0 (finite) |
| Inventory / pricing | Stock + demand | Order qty | Revenue - cost | 0.95 |
| RLHF for LLMs | Context tokens | Next token | Reward-model score at end | 1.0 (episode ~200 tokens) |
| GRPO for reasoning | Prompt + partial response | Next token | Verifier 0/1 at end | 1.0 |

Tulis kelima tuple sebelum menulis loop training apa pun. Sebagian besar laporan bug "RL tidak bekerja" berakar pada formulasi MDP yang rusak di atas kertas.

## Rilis

Simpan sebagai `outputs/skill-mdp-modeler.md`:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Latihan

1. **Mudah.** Implementasikan GridWorld 4×4 dan rollout policy-acak di `code/main.py`. Jalankan 10.000 episode. Laporkan mean dan std return. Bandingkan dengan return optimal (-6).
2. **Sedang.** Jalankan `policy_evaluation` dengan `γ ∈ {0.5, 0.9, 0.99}` untuk policy seragam-acak. Cetak `V` sebagai grid 4×4 untuk masing-masing. Jelaskan mengapa value state dekat terminal tumbuh lebih cepat dengan `γ` lebih besar.
3. **Sulit.** Buat GridWorld stokastik: setiap action meleset ke arah bersebelahan dengan probabilitas `p = 0.1`. Evaluasi ulang policy seragam. Apakah `V[start]` membaik atau memburuk? Mengapa?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MDP | "Reinforcement learning setup" | Tuple `(S, A, P, R, γ)` satisfying the Markov property. |
| State | "What the agent sees" | Sufficient statistic for future dynamics under the chosen policy class. |
| Policy | "Agent's behavior" | Conditional distribution `π(a \| s)` or deterministic map `s → a`. |
| Return | "Total reward" | Discounted sum `Σ γ^t r_t` from the current step. |
| Value | "How good a state is" | Expected return under `π` starting from `s`. |
| Q-value | "How good an action is" | Expected return under `π` starting from `s` with first action `a`. |
| Bellman equation | "Dynamic programming recursion" | Fixed-point decomposition of value / Q into one-step reward plus discounted successor value. |
| Discount `γ` | "Future vs present" | Geometric weight on far-future reward; effective horizon `~1/(1-γ)`. |

## Bacaan Lanjutan

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — textbook-nya. Bab 3 mencakup MDP dan persamaan Bellman; Bab 1 memotivasi hipotesis reward yang mendasari setiap pelajaran berikutnya.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — asal-usul persamaan Bellman.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — primer MDP ringkas dari sudut deep-RL.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — referensi operations-research tentang MDP dan metode solusi eksak.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://cs.brown.edu/media/filer_public/d1/a6/d1a6f66a-289a-4b81-9596-417114843489/littman.pdf) — derivasi MDP terbersih sebagai spesialisasi pemrograman-dinamis.
