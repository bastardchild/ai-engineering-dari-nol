# MARL — MADDPG, QMIX, MAPPO

> Warisan reinforcement learning dari koordinasi multi-agent yang masih membentuk sistem LLM-agent pada 2026. **MADDPG** (Lowe et al., NeurIPS 2017, arXiv:1706.02275) memperkenalkan Centralized Training, Decentralized Execution (CTDE): setiap kritik melihat keadaan dan aksi semua agen selama training; saat pengujian hanya aktor lokal yang berjalan. Berlaku untuk setting kooperatif, kompetitif, dan campuran. **QMIX** (Rashid et al., ICML 2018, arXiv:1803.11485) adalah value-decomposition dengan mixing network monoton; Q per-agen digabung menjadi Q gabungan sehingga `argmax` terdistribusi dengan bersih — dominan di StarCraft Multi-Agent Challenge (SMAC). **MAPPO** (Yu et al., NeurIPS 2022, arXiv:2103.01955) adalah PPO dengan value function terpusat; "surprisingly effective" pada particle-world, SMAC, Google Research Football, Hanabi dengan tuning minimal. Ketiganya menopang training policy untuk tim agen yang harus bertindak secara desentral. MAPPO adalah **baseline kooperatif-MARL default 2026**. Pelajaran ini membangun ketiganya dari mainan grid-world kecil dan mengukir tiga ide itu ke dalam ingatan otot sebelum menyentuh training LLM-agent.

**Type:** Learn
**Languages:** Python (stdlib, small NumPy-free implementations)
**Prerequisites:** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~90 minutes

## Masalah

Sistem LLM-agent semakin sering melatih policy untuk koordinasi antar-agen: kapan harus mengalah, kapan harus bertindak, peer mana yang harus dipanggil. Literatur yang mengajarkan cara melatih policy semacam itu adalah Multi-Agent Reinforcement Learning (MARL), yang mendahului gelombang LLM dan punya sekumpulan algoritma dominan.

Membaca paper MARL tanpa kosakata polanya menyakitkan. Centralized training dengan decentralized execution (CTDE), value decomposition, dan centralized critic bukan buzzword — itu jawaban spesifik untuk masalah spesifik:

- Independent RL (tiap agen belajar sendiri) bersifat non-stationary dari sudut pandang tiap agen. Buruk.
- Centralized RL (satu agen mengendalikan semua) tidak skalabel dan melanggar batasan eksekusi.
- CTDE mengambil yang terbaik dari keduanya: training dengan informasi global, deployment dengan policy lokal.

## Konsep

### Tiga environment yang dipakai para paper

- **Particle World (multi-agent particle env).** Fisika 2D sederhana dengan tugas kooperatif/kompetitif. Testbed asli MADDPG.
- **StarCraft Multi-Agent Challenge (SMAC).** Micro-management kooperatif, observasi parsial. Testbed QMIX. Aksi diskret, keadaan kontinu.
- **Google Research Football, Hanabi, MPE.** Baseline MAPPO.

Environment berbeda punya jenis aksi/observasi berbeda. Algoritmanya menyesuaikan.

### MADDPG (2017) — pola CTDE

Setiap agen `i` punya actor `mu_i(o_i)` yang memetakan observasinya sendiri ke aksi. Tiap agen juga punya kritik `Q_i(x, a_1, ..., a_n)` yang melihat semua observasi dan semua aksi selama training. Actor diperbarui dengan policy gradient terhadap penilaian kritik.

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

Kenapa CTDE: saat training, kita tahu aksi semua orang; itu dipakai untuk menekan varians tiap kritik. Saat deployment, tiap agen hanya melihat `o_i` dan memanggil `mu_i(o_i)`.

Mode gagal: kritik tumbuh seiring jumlah agen N (input memuat semua aksi). Tidak skalabel melewati ~10 agen tanpa aproksimasi.

### QMIX (2018) — value decomposition

Hanya kooperatif. Reward global adalah jumlah dari fungsi monoton atas Q-value per-agen:

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

Sifat monoton menjamin `argmax_a Q_tot` dapat dihitung dengan tiap agen memilih `argmax_{a_i} Q_i` secara independen. Itu **persis properti decentralized execution** yang kamu butuhkan. Saat training, mixing network menghasilkan `Q_tot` dari Q per-agen.

Kenapa QMIX menang di SMAC: micro-management StarCraft yang kooperatif punya agen homogen, obs lokal, global reward — pas sekali untuk value decomposition.

Mode gagal: batasan monoton terlalu ketat; sebagian tugas punya struktur reward yang tidak dapat didekomposisi secara monoton (satu agen mengorbankan diri untuk tim). Ekstensi (QTRAN, QPLEX) melonggarkannya.

### MAPPO (2022) — default yang terlewatkan

Multi-Agent PPO: PPO dengan value function terpusat. Tiap agen punya policy sendiri; semua agen berbagi (atau punya per-agen) value function yang melihat keadaan penuh. Yu et al. 2022 membandingkan MAPPO terhadap MADDPG, QMIX, dan ekstensinya pada lima benchmark lalu menemukan:

- MAPPO menyamai atau mengalahkan metode MARL off-policy pada particle-world, SMAC, Google Research Football, Hanabi, MPE.
- Tuning hyperparameter minimal.
- Training stabil; dapat direproduksi lintas seed.

Komunitas meremehkan MARL on-policy sampai paper ini. Pada 2026, MAPPO adalah baseline default untuk MARL kooperatif; metode baru apa pun harus mengalahkannya.

### Kenapa engineer LLM-agent peduli

Tiga penggunaan langsung:

1. **Router training.** Sebuah meta-agent memilih sub-agent mana yang menangani tugas. Ini masalah MARL dengan N sub-agent desentral dan satu router terpusat. MAPPO cocok.
2. **Role emergence.** Dalam simulasi agen generatif, melatih agen untuk mengadopsi peran komplementer seiring waktu adalah masalah MARL tersamar. Value decomposition gaya QMIX memaksa komplemen sejak awal.
3. **Multi-agent tool use.** Saat agen berbagi alat dan bersaing untuk anggaran, melatih mereka lewat CTDE menghasilkan policy lokal yang siap dideploy dan menghormati batasan sumber daya.

Catatan praktis: pada 2026, sebagian besar sistem LLM-agent produksi men-prompt policy mereka alih-alih melatihnya. MARL masuk ketika kamu punya (a) banyak data interaksi, (b) sinyal reward yang jelas, dan (c) kesediaan berinvestasi pada infrastruktur training.

### CTDE sebagai pola desain di luar RL

Tanpa training pun, CTDE adalah pola arsitektur yang berguna:

- Saat *desain*, asumsikan visibilitas tim penuh.
- Saat *runtime*, terapkan eksekusi desentral: tiap agen hanya melihat `o_i`.

Pola ini memaksamu menjaga state per-agen tetap eksplisit dan memikirkan observasi parsial sejak awal. Banyak sistem produksi multi-agent diam-diam mengasumsikan state bersama di mana-mana — disiplin CTDE mencegah hal itu.

### Masalah non-stationarity

Saat beberapa agen belajar serentak, environment tiap agen (yang mencakup policy agen lain) bersifat non-stationary. Bukti RL klasik satu-agen runtuh. Algoritma MARL di pelajaran ini semuanya menanganinya:

- MADDPG: kritik global melihat semua aksi, jadi estimasi nilainya stationer.
- QMIX: value decomposition memindahkan pembelajaran ke ruang Q gabungan tempat optimalitas terdefinisi baik.
- MAPPO: value function terpusat meredam varians dari perubahan policy agen lain.

Dalam sistem LLM-agent, non-stationarity bermanifestasi sebagai "agenku bekerja bulan lalu, sekarang agen lain di hulu berubah, milikku jadi bermasalah." Melatih MARL dengan CTDE adalah perbaikan yang bernalar; perbaikan di level prompt lebih cepat tapi kurang tahan lama.

### Yang TIDAK dibahas pelajaran ini

Melatih jaringan sungguhan adalah topik Phase 09. Pelajaran ini membangun versi policy ber-skrip yang mendemonstrasikan pola CTDE, value-decomposition, dan centralized-value tanpa update gradien. Tujuannya menginternalisasi pola-pola itu sebelum kamu mengangkat pustaka MARL penuh (PyMARL, MARLlib, RLlib multi-agent).

```figure
sw-ctde
```

## Bangun Sendiri

`code/main.py` mengimplementasikan tiga demonstrasi pola, semuanya pada grid-world kooperatif 2-agen mungil:

- Environment: 2 agen pada grid 4x4, satu pellet reward. Reward = 1 jika ada agen mencapai pellet; tugas selesai.
- `IndependentAgents` — tiap agen memperlakukan agen lain sebagai environment. Baseline.
- `MADDPGStyle` — kritik terpusat menghitung nilai gabungan; policy actor diperbarui darinya. Peningkatan policy ber-skrip.
- `QMIXStyle` — value decomposition dengan mixer monoton.
- `MAPPOStyle` — value function terpusat; policy diperbarui terhadap baseline bersama.

Keempatnya menjalankan episode yang sama dan melaporkan rata-rata langkah-ke-tujuan. Varian CTDE konvergen ke jalur lebih pendek dibanding baseline independen.

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan: agen independen menempuh ~6 langkah rata-rata; varian CTDE konvergen menuju ~3,5 langkah (optimal untuk grid 4x4 adalah 3). Perbedaan polanya tetap terlihat meski policy ber-skrip.

## Gunakan

`outputs/skill-marl-picker.md` adalah skill yang memilih algoritma MARL untuk tugas multi-agent tertentu: kooperatif vs kompetitif, homogen vs heterogen, jenis action-space, skala, sinyal reward.

## Rilis

MARL di produksi itu jarang. Ketika kamu memakainya:

- **Mulai dari MAPPO.** Paper 2022 menetapkannya sebagai baseline; mereproduksinya lebih dulu menghemat berminggu-minggu mengejar metode yang lebih rumit.
- **Log stream observasi dan aksi setiap agen.** Mendebug MARL tanpa jejak per-agen itu harapan kosong.
- **Pisahkan kode training dari kode eksekusi.** CTDE itu disiplin; jalankan eksekusi agar benar-benar hanya melihat `o_i`.
- **Peringatan reward shaping.** MARL sangat sensitif terhadap desain reward. Satu bug koordinasi di shaping membuat agen belajar mengeksploitasinya. Jalankan tes advesarial.
- **Untuk agen LLM**, pertimbangkan policy level-prompt lebih dulu. Baru investasi pada training MARL ketika data interaksi + sinyal reward + infrastruktur semuanya sudah ada.

## Latihan

1. Jalankan `code/main.py`. Ukur selisih langkah-ke-tujuan antara agen independen dan agen bergaya MAPPO. Apakah selisihnya membesar atau mengecil pada grid 6x6?
2. Implementasikan varian kompetitif: dua agen, satu pellet, hanya yang pertama sampai yang mendapat reward. Pola mana yang menangani kompetisi dengan bersih? MADDPG secara historis.
3. Baca MADDPG (arXiv:1706.02275) Bagian 3. Implementasikan aturan update kritik yang persis itu secara simbolis dalam pseudocode dengan kata-katamu sendiri.
4. Baca MAPPO (arXiv:2103.01955). Kenapa para penulis berargumen value terpusat + PPO mengalahkan MARL off-policy pada benchmark mereka? Sebutkan tiga klaim terkuat.
5. Terapkan CTDE sebagai pola desain pada sistem LLM-agent hipotetis (misalnya research agent + summarizer + coder). Informasi gabungan apa yang tersedia saat desain tapi tidak saat runtime?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MARL | "Multi-Agent RL" | Reinforcement learning for multi-agent systems. |
| CTDE | "Centralized Training, Decentralized Execution" | Train with global info; deploy with local policies. |
| MADDPG | "Multi-Agent DDPG" | CTDE with per-agent critic seeing all observations + actions. |
| QMIX | "Value decomposition" | Monotonic mixing of per-agent Qs. Cooperative. |
| MAPPO | "Multi-Agent PPO" | PPO with centralized value function. 2026 default baseline. |
| Value decomposition | "Sum of individual Qs" | Joint Q represented as a monotone function of per-agent Qs. |
| Non-stationarity | "Moving targets" | Each agent's env changes as others learn. The core MARL problem. |
| On-policy / off-policy | "Learn from current / replay" | PPO is on-policy (MAPPO); DDPG and Q-learning are off-policy. |
| SMAC | "StarCraft Multi-Agent Challenge" | Cooperative micromanagement benchmark; QMIX's homegrown ground. |

## Bacaan Lanjutan

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG; NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX; ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO; NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — readable framing of the MAPPO result
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge
