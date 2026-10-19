# Optimasi Swarm untuk LLM (PSO, ACO)

> Optimasi bio-inspired sedang kembali ke LLM. **LMPSO** (arXiv:2504.09247) memakai PSO di mana kecepatan setiap partikel adalah sebuah prompt dan LLM menghasilkan kandidat berikutnya; bekerja baik pada output sekuens-terstruktur (ekspresi matematika, program). **Model Swarms** (arXiv:2410.11163) memperlakukan setiap expert LLM sebagai sebuah partikel PSO pada sebuah manifold bobot-model dan melaporkan **rata-rata gain 13.3%** di atas 12 baseline pada 9 dataset hanya dengan 200 instance. **SwarmPrompt** (ICAART 2025) meng-hibridisasi PSO + Grey Wolf untuk optimasi prompt. **AMRO-S** (arXiv:2603.12933) adalah spesialis feromon terinspirasi ACO untuk routing multi-agent LLM — **speedup 4.7x**, bukti routing yang bisa diinterpretasi, pembaruan asinkron bergerbang-kualitas yang memisahkan inferensi dari pembelajaran. Pelajaran ini mengimplementasikan PSO pada ruang parameter prompt dan ACO pada routing agent, mengukur mengapa algoritma klasik ini cocok untuk era LLM, dan kapan tidak.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 minutes

## Masalahnya

Kamu punya sebuah prompt yang menilai 62% pada eval tugas-mu. Kamu ingin memperbaikinya. Langkah naifnya adalah penyetelan manual tanpa gradien, yang menskalakan dengan buruk. Reinforcement learning membutuhkan sinyal reward dan cukup banyak rollouts untuk dilatih. Backprop melalui prompt tidak benar-benar mungkin — prompt itu sebuah string diskrit, bukan sebuah parameter yang bisa diturunkan gradiennya.

Optimasi bio-inspired klasik — PSO untuk ruang pencarian kontinu, ACO untuk pemilihan jalur — dirancang tepat untuk rezim ini: tanpa gradien, berbasis populasi, murah per evaluasi. Pasangkan mereka dengan LLM untuk langkah pencarian tanpa-gradien, dan kamu mendapatkan sebuah optimizer yang mengejutkan praktisnya.

Pola yang sama berlaku untuk *routing* agent dalam sistem multi-agent. Jejak feromon bergaya ACO mencatat agent mana yang bekerja terbaik pada tipe-tugas mana, membiarkan router mengeksploitasi jejak itu, dan meredakan feromon agar jalur bisa ditemukan-ulang.

## Konsepnya

### Pengingat PSO (Kennedy & Eberhart 1995)

Particle Swarm Optimization: populasi partikel dalam sebuah ruang pencarian kontinu. Setiap partikel punya posisi `x_i` dan kecepatan `v_i`. Setiap iterasi:

```
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

Di mana `p_best` adalah best milik partikel itu sendiri, `g_best` adalah best milik swarm, `w, c1, c2` adalah bobot inertia + kognitif + sosial, `r1, r2` adalah faktor acak.

### PSO pada output LLM — LMPSO

arXiv:2504.09247 mengadaptasi PSO untuk output terstruktur hasil-LLM (ekspresi matematika, program). Setiap partikel adalah sebuah kandidat output. Kecepatannya adalah sebuah *prompt* yang menjelaskan cara memodifikasi output saat ini ke arah best personal/global. LLM menghasilkan output baru dari prompt kecepatan itu. "Inertia" dari kecepatan berupa prompt seperti "buat perubahan inkremental kecil."

Ini bekerja baik ketika:
- Output-nya terstruktur (bisa di-parse, bisa dievaluasi).
- Fitness-nya otomatis (jalankan tes, evaluasi aritmetika).
- Populasinya kecil (~10-30 partikel) sehingga total panggilan LLM tetap terkendali.

Ini tidak bekerja baik ketika fitness membutuhkan tinjauan manusia — biaya per-iterasinya menjadi terlalu mahal.

### Model Swarms

arXiv:2410.11163 mengangkat PSO dari lapisan output ke lapisan *model*. Setiap "partikel" adalah sebuah expert LLM (parameternya). Swarm menggeser parameter ke arah best kolektif melalui sebuah pembaruan tanpa-gradien. Dilaporkan: rata-rata gain 13.3% di atas 12 baseline pada 9 dataset, dengan hanya 200 instance per iterasi.

Wawasan kuncinya adalah model expert LLM sudah berdekatan dalam sebuah manifold parameter bersama (bobot adapter, delta LoRA). PSO pada subspasiah berdimensi-rendah ini murah dan efektif.

### Pengingat ACO (Dorigo 1992)

Ant Colony Optimization: semut menjelajahi sebuah graph; setiap jalur punya jejak feromon. Probabilitas gerak semut diberi bobot oleh kekuatan feromon. Semut yang menyelesaikan tugas menyetor feromon sebanding dengan kualitas solusi. Feromon mereda seiring waktu.

### AMRO-S — ACO untuk routing agent

arXiv:2603.12933 memakai ACO untuk routing multi-agent. Setiap tipe-tugas adalah sebuah "tujuan"; setiap agent adalah sebuah jalur yang mungkin. Feromon memperkuat jalur yang menghasilkan output yang baik. Kontribusi utamanya:

- **Bukti routing yang bisa diinterpretasi.** Kekuatan feromon adalah sinyal yang bisa dibaca manusia.
- **Pembaruan asinkron bergerbang-kualitas.** Feromon hanya diperbarui setelah pemeriksaan kualitas lolos, memisahkan inferensi dari pembelajaran.
- **Speedup 4.7x** pada benchmark routing multi-agent.

Gerbang kualitasnya penting: tanpanya, agent yang cepat-tapi-salah menumpuk feromon, dan sistem mengunci diri ke jalur yang buruk.

### Kapan memakai PSO / ACO untuk LLM

**Pakai PSO ketika:**
- Ruang pencariannya kontinu atau dipetakan ke parameter kontinu (embedding prompt, bobot LoRA, parameter generasi numerik).
- Fitness-nya murah dan otomatis.
- Populasinya bisa kecil (10-30).

**Pakai ACO ketika:**
- Kamu punya masalah routing atau pemilihan jalur.
- Keputusan menguat seiring waktu (tipe-tugas yang sama datang kembali).
- Kamu membutuhkan bukti yang bisa diinterpretasi untuk keputusan routing.

**Jangan pakai keduanya ketika:**
- Fitness membutuhkan tinjauan manusia (terlalu mahal per iterasi).
- Ruang pencariannya diskrit dan kombinatorial dengan cara yang tidak dicakup PSO (gunakan algoritma genetik saja).
- Keputusan real-time membutuhkan latensi yang ketat (PSO/ACO konvergen lambat dibanding heuristik single-pass).

### Mengapa bio-inspired tetap menang

Metode berbasis gradien membutuhkan sinyal yang bisa diturunkan gradiennya. Output LLM dan keputusan routing bukan sesuatu yang trivial bisa diturunkan gradiennya. Metode pseudo-gradien (router yang dilatih-reinforcement, penyetel prompt ala DPO) bekerja tetapi membutuhkan training yang mahal.

PSO dan ACO hanya membutuhkan sebuah fungsi *evaluator*. Jika kamu bisa menilai sebuah kandidat output atau sebuah keputusan routing, kamu bisa mengoptimalkan di atas ruang itu. Itu membuat ambang kemampuan-berlakunya jauh lebih rendah.

### Batasan praktis

- **Anggaran populasi.** N partikel × T iterasi × biaya per-eval. Untuk evaluasi LLM pada ~$0.02 / panggilan, sebuah PSO 20 partikel yang berjalan 50 iterasi berbiaya ~$20. Rencanakan sesuai itu.
- **Eksplorasi vs eksploitasi.** Laju reda feromon dan inertia PSO saling bertukar; reda terlalu cepat → lupa solusi; terlalu lambat → macet di optimum lokal awal.
- **Drift katastrofik.** Kedua algoritma bisa konvergen lalu divergen jika lanskap fitness bergeser (distribusi data baru). Pantau stabilitas fitness-terbaik.

```figure
swarm-stigmergy
```

## Bangun Sendiri

`code/main.py` mengimplementasikan:

- `LMPSO` — PSO di atas parameter prompt numerik (temperature, bobot top_k). "Generasi LLM" setiap partikel disimulasikan sebagai sebuah fungsi fitness ber-skrip. Menjalankan algoritma selama 30 iterasi dan menunjukkan konvergensi g_best.
- `AMRO_S` — routing bergaya ACO. 3 agent, 4 tipe tugas, matriks feromon, 100 task yang di-routing. Mencetak distribusi (tipe_tugas → pilihan agent) seiring waktu untuk menunjukkan pembentukan jejak.
- Perbandingan: routing acak vs routing ACO pada stream tugas yang sama. Mengukur kualitas dan latensi.

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan:
- LMPSO: fitness g_best membaik dari acak ke mendekati-optimal selama 30 iterasi.
- AMRO-S: tabel feromon menstabil pada agent yang tepat per tipe-tugas; routing ACO mengalahkan acak ~30-40% dalam kualitas dan juga mengurangi latensi (lebih sedikit percobaan ulang).

## Gunakan

`outputs/skill-swarm-optimizer.md` membantu memilih di antara PSO, ACO, algoritma genetik, dan optimizer berbasis gradien untuk masalah optimasi LLM / agent.

## Rilis

- **Mulai kecil.** 10-20 partikel, 20-50 iterasi. Naikkan skala hanya jika kurva konvergensi menunjukkan gain yang jelas.
- **Catat feromon atau g_best per iterasi.** Mendebug optimizer swarm tanpa jejak itu menyakitkan.
- **Pembaruan bergerbang-kualitas.** Terutama untuk routing ACO: agent yang cepat-dan-salah tidak boleh menumpuk feromon.
- **Setel ulang reda pada pergeseran distribusi.** Ketika distribusi eval-mu berubah, feromon yang tua sudah basi; setel ulang atau gandakan laju reda secara sementara.
- **Batas biaya per iterasi.** Keluarkan sebuah metrik biaya-per-iterasi. PSO yang berbiaya $500 / iterasi dengan gain 0.5% tidak layak dikirim.

## Latihan

1. Jalankan `code/main.py`. Amati konvergensi LMPSO. Ubah-ubah ukuran populasi 5, 10, 20, 50. Pada ukuran berapa waktu-konvergen jenuh?
2. Implementasikan sebuah eksperimen "drift katastrofik": setelah iterasi 30, ganti fungsi fitness-nya. Seberapa cepat PSO beradaptasi? Apakah menyetel ulang `p_best` menolong?
3. Tambahkan sebuah gerbang kualitas ke AMRO-S: setoran feromon hanya pada run dengan skor eval > 0.7. Bagaimana ini mengubah konvergensi dibanding versi tanpa gerbang?
4. Baca LMPSO (arXiv:2504.09247). Petakan "kecepatan sebagai prompt" dari makalah itu kembali ke kecepatan numerikmu. Apa yang hilang dalam simulasi dan apa yang dipertahankan?
5. Baca AMRO-S (arXiv:2603.12933). Implementasikan "fast-path inferensi" yang dipisahkan itu dengan pembaruan feromon asinkron. Bagaimana ini mengubah latensi sistem di bawah beban berkelanjutan?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| PSO | "Particle Swarm Optimization" | Kennedy-Eberhart 1995. Population-based gradient-free optimizer. |
| ACO | "Ant Colony Optimization" | Dorigo 1992. Path/route optimization via pheromone trails. |
| LMPSO | "PSO with LLM generation" | arXiv:2504.09247. Velocity is a prompt; LLM produces candidates. |
| Model Swarms | "PSO on expert weights" | arXiv:2410.11163. Gradient-free update on model parameter subspace. |
| AMRO-S | "ACO for agent routing" | arXiv:2603.12933. Pheromone matrix over task-type × agent. |
| p_best / g_best | "Personal / global best" | Per-particle and swarm-wide best solutions found so far. |
| Pheromone | "Routing memory" | Strength on an edge; decays over time; deposits on quality. |
| Quality-gated update | "Only learn from good runs" | Pheromone deposit conditioned on quality check. |
| Catastrophic drift | "Distribution shift" | Fitness landscape changes; old p_best and pheromones become stale. |

## Bacaan Lanjutan

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — the 1995 PSO paper
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) — 1992 ACO foundations
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — PSO for structured LLM outputs
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — PSO on model-weight subspace
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) — pheromone-driven routing with quality gate
