# Sim-to-Real Transfer

> Policy yang dilatih di simulator dan gagal di hardware adalah policy yang menghafal simulator. Domain randomization, domain adaptation, dan system identification adalah tiga alat untuk membuat controller yang dipelajari melewati reality gap.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Time:** ~45 minutes

## Masalahnya

Melatih robot sungguhan lambat, berbahaya, dan mahal. Biped membutuhkan jutaan episode training untuk belajar berjalan; biped sungguhan yang jatuh bahkan sekali merusak hardware. Simulasi memberimu reset tanpa batas, reproduksibilitas deterministik, environment paralel, dan tanpa kerusakan fisik.

Tetapi simulator itu salah. Bearing punya lebih banyak gesekan daripada model MuJoCo. Kamera punya distorsi lensa yang tidak disertakan simulator. Motor punya delay, backlash, dan saturasi yang dilewati 99% model sim. Angin, debu, dan pencahayaan bervariasi menyabotase policy yang dilatih pada rendering steril. **Reality gap** — perbedaan sistematis antara distribusi sim dan distribusi nyata — adalah masalah pusat RL yang dideploy untuk robotika.

Kamu membutuhkan policy yang *tahan terhadap pergeseran distribusi sim-to-real*. Tiga pendekatan historis: randomize simulator (domain randomization), adaptasi policy dengan sedikit data nyata (domain adaptation / fine-tuning), atau identifikasi parameter sistem nyata dan mencocokkannya (system identification). Di 2026 resep dominan menggabungkan ketiganya dengan simulasi paralel masif (Isaac Sim, Isaac Lab, Mujoco MJX on GPU).

## Konsepnya

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**Domain Randomization (DR).** Tobin et al. 2017, Peng et al. 2018. Selama training, randomize setiap parameter sim yang mungkin berbeda pada robot nyata: massa, koefisien gesekan, motor PD gains, noise sensor, posisi kamera, pencahayaan, tekstur, model kontak. Policy mempelajari distribusi kondisional atas "sim mana yang dipakai hari ini" dan mengeneralisasi ke seluruh rentang. Jika robot nyata berada dalam envelope training, policy bekerja.

- **Kelebihan:** tidak butuh data nyata. Satu resep, banyak robot.
- **Kekurangan:** training yang over-randomized menghasilkan policy "universal" tetapi terlalu hati-hati. Terlalu banyak noise ≈ terlalu banyak regularisasi.

**System Identification (SI).** Fit parameter simulator ke data dunia-nyata sebelum training. Jika kamu bisa mengukur gesekan sendi lengan pada robot nyata, colokkan itu ke sim. Lalu latih policy yang mengharapkan nilai-nilai itu. Membutuhkan akses ke sistem nyata tetapi mengurangi reality gap secara langsung.

- **Kelebihan:** target training presisi, rendah-noise.
- **Kekurangan:** error model residu tidak terlihat oleh policy; efek kecil yang tidak ter-identifikasi (mis., motor deadband) tetap merusak deployment.

**Domain Adaptation.** Latih di sim, fine-tune dengan sedikit data nyata. Dua ragam:

- **Real2Sim2Real:** pelajari simulator residu `f(s, a, z) - f_sim(s, a)` menggunakan rollouts nyata, latih di sim yang sudah dikoreksi. Menutup gap tanpa banyak data nyata.
- **Observation adaptation:** latih policy yang memetakan obs nyata → obs mirip-sim melalui feature extractor yang dipelajari (mis., GAN pixel-to-pixel). Controller tetap di sim.

**Privileged learning / teacher-student.** Miki et al. 2022 (ANYmal quadruped). Latih *teacher* di simulasi yang memiliki akses ke informasi privilese (friction ground-truth, ketinggian terrain, IMU drift). Distilasi *student* yang hanya melihat observasi sensor nyata. Student belajar menyimpulkan fitur privilese dari history, tahan lintas parameter fisik.

**Simulasi paralel masif.** 2024–2026. Isaac Lab, Mujoco MJX, Brax semuanya menjalankan ribuan robot paralel pada satu GPU. PPO dengan 4.096 humanoid paralel mengumpulkan bertahun-tahun pengalaman dalam hitungan jam. "Reality gap" menyusut seiring distribusi training melebar; DR menjadi hampir gratis ketika masing-masing dari 4.096 env itu memiliki parameter randomized yang berbeda.

**Resep dunia-nyata 2026 (contoh berjalan quadruped):**

1. Sim paralel masif dengan gravity, friction, motor gains, payload yang domain-randomized.
2. Policy teacher dilatih dengan info privilese (peta terrain, body velocity ground truth).
3. Policy student didistilasi dari teacher hanya menggunakan proprioception (leg joint encoders).
4. Opsional observation adaptation via autoencoder pada IMU nyata.
5. Deploy. Zero-shot di 10+ environments. Jika gagal, lakukan beberapa menit fine-tuning dunia-nyata dengan PPO yang constraints-keselamatannya.

```figure
f3-reality-gap
```

## Bangun Sendiri

Kode pelajaran ini adalah demonstrasi kecil domain randomization pada GridWorld dengan transisi *berisik*. Kami melatih policy yang mengalami slip probabilities yang di-randomize di "sim" dan mengevaluasi di "real" dengan tingkat slip yang tidak pernah dilihatnya selama training. Bentuknya terpetakan langsung ke transfer MuJoCo-to-hardware.

### Langkah 1: sim terparameterisasi

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` adalah parameter yang diekspos simulator. Di robotika nyata bisa berupa gesekan, massa, motor gain — apa pun yang bergeser antara sim dan nyata.

### Langkah 2: latih dengan DR

Di awal setiap episode, sample `slip ~ Uniform[0.0, 0.4]`. Latih PPO / Q-learning / apa saja. Lakukan ini untuk banyak episodes.

### Langkah 3: evaluasi zero-shot pada "slip" nyata

Evaluasi pada `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. Empat pertama berada dalam dukungan training; `0.5` dan `0.7` di luar. Policy yang dilatih DR harus tetap mendekati-optimal di dalam dukungan dan terdegradasi dengan anggun di luar. Policy yang dilatih slip-tetap akan rapuh di luar slip training-nya.

### Langkah 4: bandingkan dengan training sempit

Latih policy kedua dengan `slip = 0.0` saja. Evaluasi pada sweep `slip` yang sama. Kamu akan melihat penurunan katastrofik segera setelah slip nyata > 0.

## Jebakan

- **Terlalu banyak randomization.** Latih pada `slip ∈ [0, 0.9]` dan policy-mu begitu risk-averse sehingga tidak pernah mencoba jalur optimal. Cocokkan dengan distribusi dunia-nyata yang *diharapkan*, bukan "apa pun bisa terjadi."
- **Terlalu sedikit randomization.** Latih pada iris tipis dan policy tidak bisa mengeneralisasi sama sekali. Gunakan adaptive curriculum (Automatic Domain Randomization) yang melebarkan distribusi seiring policy membaik.
- **Ruang parameter yang salah-identifikasi.** Randomize hal yang salah (hue kamera ketika gap sebenarnya adalah motor delay) dan DR tidak membantu. Profil robot nyata terlebih dahulu.
- **Kebocoran info privilese.** Teacher yang menggunakan state global untuk aksi, bukan hanya observasi, dapat menghasilkan student yang tidak bisa menyusul. Pastikan policy teacher dapat direalisasikan student mengingat history observasi.
- **Kegagalan transfer sim-to-sim.** Jika policy-mu tidak tahan terhadap varian sim yang lebih sulit, ia juga tidak akan tahan terhadap dunia nyata. Selalu uji pada varian sim held-out sebelum deploy.
- **Tanpa safety envelope dunia-nyata.** Policy yang bekerja di sim dan "bekerja di nyata" tanpa perisai keselamatan tingkat-rendah tetap bisa merusak hardware. Tambahkan rate limits, torque limits, joint limits di controller yang tidak dipelajari.

## Gunakan

Stack sim-to-real 2026:

| Domain | Stack |
|--------|-------|
| Legged locomotion (ANYmal, Spot, humanoid) | Isaac Lab + DR + privileged teacher / student |
| Manipulation (dexterous hands, pick-and-place) | Isaac Lab + DR + DR-GAN for vision |
| Autonomous driving | CARLA / NVIDIA DRIVE Sim + DR + real fine-tune |
| Drone racing | RotorS / Flightmare + DR + online adaptation |
| Finger/in-hand manipulation | OpenAI Dactyl (DR at unprecedented scale) |
| Industrial arms | MuJoCo-Warp + SI + small real fine-tune |

Untuk kontrol di semua skala, workflow-nya konsisten: fit sim sebaik mungkin, randomize apa yang tidak bisa kamu fit, latih policy yang raksasa, distilasi, deploy dengan perisai keselamatan.

## Rilis

Simpan sebagai `outputs/skill-sim2real-planner.md`:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Latihan

1. **Mudah.** Latih agen Q-learning pada GridWorld slip-tetap (slip=0.0). Evaluasi pada slip ∈ {0.0, 0.1, 0.3, 0.5}. Plot return vs slip.
2. **Sedang.** Latih agen DR Q-learning yang sample `slip ~ Uniform[0, 0.3]`. Evaluasi sweep yang sama. Berapa banyak yang dibeli DR pada slip=0.5 (out-of-distribution)?
3. **Sulit.** Implementasikan curriculum: mulai dengan slip=0.0, lebarkan rentang DR setiap kali policy mencapai 90% dari optimal. Ukur total langkah environment untuk mencapai slip=0.3 zero-shot vs baseline DR tetap.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Reality gap | "Sim-to-real difference" | Distribution shift between training and deployment physics/sensing. |
| Domain randomization (DR) | "Train across random sims" | Randomize sim parameters during training so policy generalizes. |
| System identification (SI) | "Measure real and fit sim" | Estimate real physical parameters; set sim to match. |
| Domain adaptation | "Fine-tune on real data" | Small real-world fine-tune after sim training; may adapt obs or dynamics. |
| Privileged info | "Ground truth for teacher" | Information only the sim has; student must infer it from obs history. |
| Teacher/student | "Distill privileged -> observable" | Teacher trained with shortcuts; student learns to mimic without them. |
| ADR | "Automatic Domain Randomization" | Curriculum that widens DR ranges as the policy improves. |
| Real2Sim | "Close the gap with real data" | Learn a residual to make the sim mimic real rollouts. |

## Bacaan Lanjutan

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — the original DR paper (vision for robotics).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — DR for dynamics, quadruped locomotion.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl, ADR at scale.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — teacher-student for ANYmal.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — the massively parallel sim that drives 2025–2026 deployments.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR curriculum method.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — the Dyna framing (use a model for planning + rollouts) that underpins modern sim-to-real pipelines.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — taxonomy of sim-to-real methods with benchmark results.
