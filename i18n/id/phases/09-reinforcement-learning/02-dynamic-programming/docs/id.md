# Pemrograman Dinamis — Iterasi Policy & Iterasi Value

> Pemrograman dinamis adalah RL dengan mencontek. Kamu sudah tahu fungsi transisi dan reward; kamu hanya mengiterasikan persamaan Bellman sampai `V` atau `π` berhenti bergerak. Ia adalah benchmark yang coba didekati setiap metode berbasis-sampling.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 minutes

## Masalahnya

Kamu punya MDP dengan model yang diketahui: kamu bisa menanyakan `P(s' | s, a)` dan `R(s, a, s')` untuk pasangan state-action apa pun. Manajer inventaris tahu distribusi demand. Board game punya transisi deterministik. Gridworld adalah empat baris Python. Kamu punya *model*.

RL bebas-model (Q-learning, PPO, REINFORCE) diciptakan untuk kasus di mana kamu tidak punya model — kamu hanya bisa menyampel dari environment. Tapi ketika kamu punya, ada metode yang lebih cepat dan lebih baik: pemrograman dinamis. Bellman merancangnya pada 1957. Mereka tetap mendefinisikan kebenaran: ketika orang berkata "policy optimal untuk MDP ini," maksudnya policy yang akan dikembalikan DP.

Kamu membutuhkannya pada 2026 untuk tiga alasan. Pertama, setiap environment tabular dalam riset RL (GridWorld, FrozenLake, CliffWalking) dipecahkan dengan DP untuk menghasilkan policy standar-emas. Kedua, value eksak membuatmu bisa *mendebug* metode sampling: kalau estimasi Q-learning untuk `V*(s_0)` tidak sepakat dengan jawaban DP sebesar 30%, Q-learning-mu punya bug. Ketiga, metode RL offline modern dan perencana (MCTS, search AlphaZero, RL berbasis-model di Fase 9 · 10) semuanya mengiterasikan backup Bellman atas model terpelajari atau yang diberikan.

## Konsepnya

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**Dua algoritma, keduanya iterasi titik-tetap pada Bellman.**

**Iterasi policy.** Mengalihkan dua langkah sampai policy berhenti berubah.

1. *Evaluasi:* diberikan policy `π`, hitung `V^π` dengan menerapkan berulang `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` sampai konvergen.
2. *Perbaikan:* diberikan `V^π`, buat `π` greedy thd. `V^π`: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

Konvergensi dijamin karena (a) setiap langkah perbaikan membiarkan `π` sama atau menaikkan `V^π` secara ketat untuk sebagian state, (b) ruang policy deterministik terbatas. Biasanya konvergen dalam ~5–20 iterasi luar bahkan untuk ruang state besar.

**Iterasi value.** Meruntuhkan evaluasi dan perbaikan menjadi satu sapuan. Terapkan persamaan *optimalitas* Bellman:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Ulangi sampai `max_s |V_{new}(s) - V(s)| < ε`. Ekstrak policy di akhir dengan mengambil action greedy. Jauh lebih cepat per iterasi — tanpa loop evaluasi dalam — tapi tipikalnya butuh lebih banyak iterasi untuk konvergen.

**Iterasi policy tergeneralisasi (GPI).** Pembingkaian pemersatu. Fungsi value dan policy terkunci dalam loop perbaikan dua-arah; metode apa pun yang mendorong keduanya menuju konsistensi mutual (iterasi value asinkron, iterasi policy termodifikasi, Q-learning, actor-critic, PPO) adalah instans GPI.

**Mengapa `γ < 1` penting.** Operator Bellman adalah kontraksi-`γ` dalam sup-norm: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Kontraksi mengimplikasikan titik-tetap unik dan konvergensi geometris. Buang `γ < 1` dan kamu kehilangan jaminan — kamu butuh horizon terbatas atau state terminal menyerap.

```figure
value-iteration-gamma
```

## Bangun Sendiri

### Langkah 1: bangun model MDP GridWorld

Pakai GridWorld 4×4 yang sama dari Pelajaran 01. Kita tambahkan varian stokastik: dengan probabilitas `0.1` agen meleset ke arah tegak-lurus acak.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` mengembalikan daftar `(s', r, p)`. Inilah seluruh model.

### Langkah 2: evaluasi policy

Diberikan policy `π(s) = {action: prob}`, iterasikan persamaan Bellman sampai `V` berhenti bergerak:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Langkah 3: perbaikan policy

Ganti `π` dengan policy greedy thd. `V`. Kalau `π` tidak berubah, kembalikan — kita di optimum.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Langkah 4: jahit keduanya

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Konvergensi tipikal pada 4×4: 4–6 iterasi luar. Mengeluarkan `V*(0,0) ≈ -6` dan policy yang menurunkan jumlah langkah secara ketat.

### Langkah 5: iterasi value (versi satu-loop)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

Titik-tetap sama, baris kode lebih sedikit.

## Jebakan

- **Lupa menangani terminal.** Kalau kamu menerapkan Bellman ke state menyerap, ia tetap mengambil "action terbaik" yang tidak mengubah apa-apa. Jaga dengan `if s == terminal: V[s] = 0`.
- **Konvergensi sup-norm vs L2.** Pakai `max |V_new - V|`, bukan rata-rata. Jaminan teoretisnya pada sup-norm.
- **Update in-place vs sinkron.** Mengupdate `V[s]` in-place (Gauss-Seidel) konvergen lebih cepat daripada dict `V_new` terpisah (Jacobi). Kode produksi memakai in-place.
- **Tie policy.** Kalau dua action punya Q-value sama, `argmax` bisa memecah tie berbeda tiap iterasi, menyebabkan cek "policy stabil" berosilasi. Pakai pemecah-tie stabil (action pertama dalam urutan tetap).
- **Ledakan ruang-state.** DP adalah `O(|S| · |A|)` per sapuan. Bekerja sampai ~10⁷ state. Melampauinya, kamu butuh aproksimasi fungsi (Fase 9 · 05 ke atas).

## Gunakan

Pada 2026, DP adalah baseline kebenaran dan loop dalam perencana:

| Use case | Method |
|----------|--------|
| Solve a small tabular MDP exactly | Value iteration (simpler) or policy iteration (fewer outer steps) |
| Verify a Q-learning / PPO implementation | Compare to DP-optimal V* on a toy environment |
| Model-based RL (Phase 9 · 10) | Bellman backup on a learned transition model |
| Planning in AlphaZero / MuZero | Monte Carlo Tree Search = async Bellman backup |
| Offline RL (CQL, IQL) | Conservative Q-iteration — DP with a penalty on OOD actions |

Setiap kali seseorang berkata "fungsi value optimal," maksudnya "titik-tetap DP." Ketika kamu melihat `V*` atau `Q*` dalam makalah, bayangkan loop ini.

## Rilis

Simpan sebagai `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Latihan

1. **Mudah.** Jalankan iterasi value pada GridWorld 4×4 dengan `γ ∈ {0.9, 0.99}`. Berapa sapuan sampai `max |ΔV| < 1e-6`? Cetak `V*` sebagai grid 4×4.
2. **Sedang.** Bandingkan iterasi policy vs iterasi value pada GridWorld *stokastik* (probabilitas slip `0.1`). Hitung: sapuan, waktu wall-clock, `V*(0,0)` final. Mana konvergen lebih cepat dalam iterasi? Dalam wall-clock?
3. **Sulit.** Bangun iterasi policy termodifikasi: dalam langkah evaluasi, jalankan hanya `k` sapuan alih-alih sampai konvergen. Plot galat `V*(0,0)` vs `k` untuk `k ∈ {1, 2, 5, 10, 50}`. Apa yang dikatakan kurva tentang trade-off evaluasi/perbaikan?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy iteration | "DP algorithm" | Alternating evaluation (`V^π`) and improvement (greedy `π` w.r.t. `V^π`) until the policy stops changing. |
| Value iteration | "Faster DP" | Bellman optimality backup applied in one sweep; converges to `V*` geometrically. |
| Bellman operator | "The recursion" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; a `γ`-contraction in sup-norm. |
| Contraction | "Why DP converges" | Any operator `T` with `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` has a unique fixed point. |
| GPI | "Everything is DP" | Generalized Policy Iteration: any method driving `V` and `π` to mutual consistency. |
| Synchronous update | "Jacobi-style" | Use old `V` throughout a sweep; cleanly analyzable but slower. |
| In-place update | "Gauss-Seidel-style" | Use `V` as it's being updated; converges faster in practice. |

## Bacaan Lanjutan

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — presentasi kanonis iterasi policy dan iterasi value.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook_athena.html) — perlakuan rigor argumen contraction-mapping.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — iterasi policy termodifikasi dan analisis konvergensinya.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — makalah iterasi policy orisinal.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — jembatan dari DP ke DP-aproksimasi / deep RL yang dipakai setiap pelajaran berikutnya.
