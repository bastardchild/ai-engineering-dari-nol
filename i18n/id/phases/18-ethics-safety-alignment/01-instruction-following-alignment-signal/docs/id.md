# Instruction-Following sebagai Sinyal Alignment

> Setiap kritik RLHF di kemudian hari berargumen melawan pipeline ini. Sebelum kamu mempelajari bagaimana tekanan optimasi mendistorsi proksi, kamu harus melihat proksinya. InstructGPT (Ouyang et al., 2022) mendefinisikan arsitektur referensi: supervised fine-tuning pada pasangan instruksi-respons, reward model yang dilatih pada ranking preferensi berpasangan, dan PPO terhadap reward model dengan penalti KL ke kebijakan SFT. InstructGPT 1.3B lebih disukai daripada GPT-3 175B. Hasil tunggal itu alasan setiap lab frontier pada 2026 masih merilis pipeline pasca-training berbentuk-RLHF.

**Type:** Learn
**Languages:** Python (stdlib, toy three-stage pipeline)
**Prerequisites:** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**Time:** ~45 minutes

## Tujuan Pembelajaran

- Sebutkan tiga tahap pipeline InstructGPT dan loss yang dipakai di masing-masing.
- Jelaskan mengapa model 1.3B yang di-tune-instruksi mengalahkan GPT-3 mentah 175B pada evaluasi preferensi manusia.
- Nyatakan apa yang dilindungi penalti KL pada tahap 3 dan mengapa menghilangkannya runtuh menjadi perilaku mode-seeking.
- Jelaskan alignment tax dan mitigasi PPO-ptx yang dipakai Ouyang et al. melawannya.

## Masalahnya

Model bahasa pra-terlatih mengomplesi teks. Mereka tidak menjawab pertanyaan. Tanyakan ke GPT-3 "write a Python function that reverses a list" dan kamu sering mendapat kembali prompt lain, karena sebagian besar distribusi training adalah teks web yang berlanjut dengan teks web lagi. Model menjalankan tugasnya — tugasnya yang salah.

Proksi yang dipakai setiap lab serius untuk memperbaikinya adalah preferensi manusia. Dua komplesi diberikan ke rater; rater memilih yang lebih baik; reward model mempelajari rater. Lalu loop RL menggeser kebijakan ke output yang diskor tinggi oleh reward model. Itu seluruh tesis InstructGPT dalam tiga kalimat. Sisanya paper adalah engineering.

## Konsepnya

### Langkah 1: supervised fine-tuning (SFT)

Kumpulkan pasangan prompt-respons di mana respons adalah apa yang akan ditulis manusia berniat-baik. Ouyang et al. memakai 13k prompt dari pelabel dan API OpenAI. Fine-tune model dasar pada data ini dengan loss cross-entropy standar.

Apa yang diberikan SFT: model kini menjawab pertanyaan alih-alih melanjutkannya. Apa yang tidak diberikannya: sinyal apa pun tentang jawaban mana yang disukai rater saat banyak yang plausibel.

### Langkah 2: reward model (RM)

Untuk tiap prompt, sampel K komplesi dari model SFT. Pelabel merankingnya. Latih reward model yang menskor pasangan prompt-respons apa pun sehingga, untuk pasangan di mana `y_w` lebih disukai daripada `y_l`:

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

Ini loss preferensi berpasangan Bradley-Terry. RM biasanya diinisialisasi dari model SFT dengan LM head diganti scalar head.

Reward model berukuran kecil: 6B cukup untuk InstructGPT 175B. Mereka juga rapuh — seksi 5 paper kebanyakan tentang perilaku reward-hacking yang muncul pada skala kecil.

### Langkah 3: PPO dengan penalti KL

Definisikan objektif:

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

Maksimalkan dengan PPO. Term KL menjaga `pi` agar tidak melayang jauh dari kebijakan SFT. Tanpanya, optimizer menemukan contoh adversarial — string yang berskor tinggi pada RM karena RM tidak pernah melihatnya, bukan karena manusia benar-benar menyukainya.

Koefisien KL `beta` adalah hiperparameter RLHF tunggal terpenting. Terlalu rendah: reward hacking. Terlalu tinggi: tanpa perbaikan atas SFT.

### Alignment tax

Setelah RLHF, model disukai manusia tetapi meregresi pada benchmark standar (SQuAD, HellaSwag, DROP). Ouyang et al. menyebut ini alignment tax dan memperbaikinya dengan PPO-ptx: campurkan gradien pra-training ke objektif RL agar model tidak lupa mengerjakan task hilir yang tidak pernah di-reward.

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx menjadi standar. Anthropic, DeepMind, dan Meta semuanya memakai suatu varian.

### Hasilnya

InstructGPT 1.3B (SFT + RM + PPO-ptx) disukai pelabel daripada GPT-3 dasar 175B sekitar 70% dari waktu. Selisih melebar pada prompt hidden-test dari trafik produksi. Dua hal yang bisa dibaca dari angka ini:

1. Alignment adalah sumbu berbeda dari kapabilitas. Model 175B punya lebih banyak kapabilitas; model 1.3B punya lebih banyak alignment; pelabel menyukai yang aligned.
2. Lantai kapabilitas ditetapkan model dasar. Kamu tidak bisa me-RLHF model dasar hingga mengetahui fakta yang tidak pernah dilihatnya.

### Mengapa ini titik referensi untuk Fase 18

Setiap kritik pada pelajaran kemudian — reward hacking (Pelajaran 2), DPO (Pelajaran 3), sycophancy (Pelajaran 4), CAI (Pelajaran 5), sleeper agent (Pelajaran 7), alignment faking (Pelajaran 9) — berargumen melawan sebagian pipeline ini. Reward hacking menyerang tahap 2. DPO meruntuhkan tahap 2 dan 3. CAI mengganti pelabel manusia. Sycophancy menunjukkan pelabel adalah sinyal bias. Alignment faking menunjukkan kebijakan bisa memutari tahap 3 sepenuhnya. Kamu tidak bisa mengikuti satu pun kritik ini tanpa pipeline di kepalamu dulu.

```figure
al-instruct-pipeline
```

## Gunakan

`code/main.py` mensimulasikan tiga tahap pada data preferensi mainan. "Kebijakan" dasar adalah koin bias atas aksi {A, B, C}. Tahap 1 SFT meniru aksi pelabel pada 200 prompt. Tahap 2 mem-fit reward model Bradley-Terry dari 500 ranking berpasangan. Tahap 3 menjalankan update PPO sederhana dengan penalti KL ke kebijakan SFT. Kamu bisa menyaksikan reward naik, divergensi KL tumbuh, dan kebijakan melayang — dan kamu bisa mematikan term KL untuk melihat reward hacking muncul dalam 50 langkah update.

Apa yang perlu dilihat:

- Trajektori reward dengan `beta = 0.1` vs `beta = 0.0`.
- KL(pi || pi_SFT) selama langkah training.
- Distribusi aksi final dibanding preferensi pelabel.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-instructgpt-explainer.md`. Dengan deskripsi pipeline RLHF atau abstrak paper, ia mengidentifikasi tahap mana dari ketiganya yang dimodifikasi, loss apa yang dipakai pada tiap tahap, dan apakah penalti KL atau regularizer setara hadir.

## Latihan

1. Jalankan `code/main.py`. Tetapkan `beta = 0.0` dan laporkan distribusi aksi setelah 200 langkah PPO. Jelaskan perilaku mode-seeking dalam satu paragraf.

2. Ubah reward model agar punya bias +0.5 untuk aksi B (bug reward tersimulasi). Jalankan PPO dengan `beta = 0.1`. Apakah penalti KL mencegah kebijakan mengeksploitasi bias? Pada `beta` berapa eksploitasi menjadi terlihat?

3. Baca Ouyang et al. (arXiv:2203.02155) Gambar 1. Reproduksi kurva preferensi-pelabel dengan menjalankan PPO selama 1, 5, 20, 100 langkah dan mengukur preferensi terhadap model SFT.

4. Paper Seksi 4.3 melaporkan InstructGPT 1.3B mengalahkan GPT-3 175B sekitar 70% dari waktu. Mengapa rasionya lebih tinggi pada prompt produksi tersembunyi daripada pada prompt milik pelabel sendiri?

5. Ganti loss PPO dengan DPO (Fase 10 · 08) pada data preferensi sama. Bandingkan drift kebijakan final (KL ke SFT) dan reward final. Metode mana melayang lebih jauh pada reward yang disamakan?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| SFT | "instruction tuning" | Stage 1: cross-entropy fine-tune on prompt-response pairs |
| Reward model | "the RM" | Scalar regressor over (prompt, response) trained with Bradley-Terry on pairwise labels |
| Bradley-Terry | "pairwise preference loss" | -log sigmoid(r_w - r_l); reduces pairwise ranking to binary classification |
| KL penalty | "the regularizer" | `beta * KL(pi \|\| pi_SFT)` — keeps the RL policy near the SFT anchor |
| PPO-ptx | "PPO with pretraining mix" | Adds a fraction of pre-training log-likelihood to the PPO objective to offset the alignment tax |
| Alignment tax | "the RLHF regression" | Post-RLHF drop on standard benchmarks that RLHF did not target |
| Labeler preference | "the ground truth" | Sample of human rankings; the RM is a statistical proxy for this, not for "human values" |

## Further Reading

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — the InstructGPT paper, foundation for every RLHF pipeline that followed
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — the RLHF-for-summarization predecessor
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — the original preference-based RL formulation
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — Anthropic's HH extension of the InstructGPT pipeline
