# Differential Privacy untuk LLM

> DP-SGD tetap menjadi standar — update gradien yang disuntik noise memberikan jaminan formal (epsilon, delta). Overhead dalam komputasi, memori, dan utilitas substansial; DP fine-tuning parameter-efficient (LoRA + DP-SGD) adalah konfigurasi 2025 yang umum (ACM 2025). Dua badan bukti yang bertegangan: membership inference berbasis canary (Duan et al., 2024) melaporkan keberhasilan terbatas terhadap model bahasa; ekstraksi data training (Carlini et al., 2021; Nasr et al., 2025) memulihkan memorisasi verbatim yang substansial. Resolusi (arXiv:2503.06808, Maret 2025): kesenjangannya ada pada apa yang diukur — canary yang disisipkan vs data "paling dapat diekstraksi". Desain canary baru memungkinkan MIA berbasis loss tanpa shadow model dan menghasilkan audit DP pertama yang nontrivial pada LLM yang dilatih pada data nyata dengan jaminan DP yang realistis. Alternatif: PMixED (arXiv:2403.15638) — prediksi privat saat inference melalui mixture of experts pada distribusi token berikutnya; DP synthetic data generation (Google Research 2024). Serangan yang muncul: Differential Privacy Reversal via LLM Feedback — kebocoran confidence-score.

**Type:** Build
**Languages:** Python (stdlib, DP-SGD noise-injection and ε-δ accountant demonstration)
**Prerequisites:** Phase 01 · 09 (information theory), Phase 10 · 01 (large-model training)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Definisikan (epsilon, delta)-differential privacy dan nyatakan resep DP-SGD.
- Jelaskan tegangan 2024-2025: canary MIA vs ekstraksi data training memberikan gambaran yang berbeda.
- Jelaskan PMixED dan mengapa prediksi privat saat inference adalah alternatif untuk DP training.
- Jelaskan serangan Differential Privacy Reversal via LLM Feedback.

## Masalahnya

LLM menghafal. Carlini et al. 2021 menunjukkan model bahasa produksi mereproduksi teks training secara verbatim saat diminta. DP adalah pertahanan formal: latih sehingga output terbukti tidak peka terhadap satu contoh training mana pun. Bukti 2024-2025 menunjukkan DP-SGD diperlukan tapi nilai ε yang dideploy mungkin tidak cocok dengan threat model.

## Konsepnya

### (ε, δ)-differential privacy

Sebuah algoritme acak M adalah (ε, δ)-DP jika untuk dua dataset mana pun yang berbeda pada satu contoh dan event S mana pun:
P(M(D) in S) <= e^ε * P(M(D') in S) + δ.

Interpretasi: distribusi output cukup dekat (diparameterisasi oleh ε) sehingga kontribusi setiap individu tunggal tidak dapat disimpulkan secara andal, kecuali dengan probabilitas δ.

### DP-SGD

Abadi et al. 2016. Resep standar:
1. Sampel sebuah mini-batch.
2. Hitung gradien per-contoh.
3. Setiap gradien per-contoh di-clip ke ambang C.
4. Jumlahkan gradien yang sudah di-clip dan tambahkan noise Gaussian dengan std σ * C.
5. Gunakan jumlah yang ber-noise untuk memperbarui parameter.

Biaya privasi dilacak oleh sebuah accountant (Moments Accountant, Rényi DP accountant). Nilai ε yang dilaporkan dalam literatur LLM bervariasi luas menurut threat model, sensitivitas data, dan target utilitas; tidak ada default ε yang "aman" secara universal. Contoh yang dipublikasikan membentang sekitar ε ≈ 1–10 pada beberapa pengaturan training LLM, tapi ini bersifat ilustratif — bukan default yang direkomendasikan. ε lebih rendah umumnya memerlukan lebih banyak noise dan dapat meningkatkan kerugian utilitas.

### LoRA + DP-SGD

DP-SGD penuh pada model frontier tidak layak. LoRA (Hu et al. 2022) membatasi update gradien ke adapter kecil, mengurangi penyimpanan gradien per-contoh. LoRA + DP-SGD adalah konfigurasi 2025 yang umum. Jaminan DP berlaku untuk adapter; model dasar dipegang tetap.

### Tegangan 2024-2025

Dua garis bukti:

- **Canary MIA (Duan et al. 2024).** Sisipkan canary unik ke dalam data training, ukur apakah penyerang membership-inference dapat mengidentifikasikannya. Melaporkan keberhasilan terbatas pada model bahasa. Menyarankan MIA sulit.
- **Training-data extraction (Carlini 2021, Nasr et al. 2025).** Beri prompt pada model dengan sebuah prefix; ukur apakah ia memulihkan teks verbatim dari training. Melaporkan memorisasi yang substansial. Menyarankan MIA mudah dalam arti yang relevan.

Resolusi Maret 2025 (arXiv:2503.06808): keduanya mengukur hal yang berbeda. MIA bertanya "apakah contoh e ada di D?" pada canary yang disisipkan. Ekstraksi bertanya "apa yang bisa saya pulihkan dari D?" Contoh yang "paling dapat diekstraksi" adalah yang penting untuk privasi; canary melaporkan hal ini secara under-report karena tidak dioptimalkan untuk dapat diekstraksi.

Desain canary baru. MIA berbasis loss tanpa shadow model. Audit DP pertama yang nontrivial pada LLM dengan data nyata dan jaminan DP yang realistis.

### Alternatif untuk DP training

- **PMixED (arXiv:2403.15638).** Prediksi privat saat inference. Mixture of experts pada distribusi token berikutnya; setiap expert melihat sebuah shard data training; agregasi menambahkan noise untuk DP. Menghindari DP training sama sekali.
- **DP synthetic data generation (Google Research 2024).** Fine-tune LoRA dengan DP-SGD, sampel data sintetis, latih sebuah classifier hulu pada data sintetis.

Keduanya menghindari biaya utilitas dari DP training penuh dengan harga threat model yang berbeda.

### Differential Privacy Reversal via LLM Feedback

Serangan 2025 yang muncul. Gunakan confidence scores dari model yang dilatih-DP sebagai oracle untuk mengidentifikasi ulang individu. Bahkan ketika output tidak membocorkan, distribusi confidence bisa.

Pertahanannya: jangan mengekspos confidence, atau truncate/kuantisasi mereka sebelum eksposur. Ini adalah persyaratan tambahan di luar training (ε, δ)-DP.

### Di mana ini cocok di Fase 18

Pelajaran 20-21 adalah bias/keadilan. Pelajaran 22 adalah privasi. Pelajaran 23 adalah provenance melalui watermarking. Pelajaran 27 membahas lapisan data-provenance regulasi.

```figure
an-dp-clip-noise
```

## Gunakan

`code/main.py` mensimulasikan DP-SGD pada sebuah dataset klasifikasi biner mainan. Kamu bisa menyapu noise multiplier σ dan clipping norm C dan melacak anggaran (ε, δ) serta biaya akurasinya. Sebuah "canary attack" menyisipkan sebuah contoh training unik dan mengukur apakah tes log-loss dapat mendeteksinya sebelum dan sesudah DP.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-dp-audit.md`. Diberi sebuah klaim DP pada deployment model bahasa, ia mengaudit: nilai (ε, δ), accountant yang digunakan, protokol evaluasi MIA, dan apakah vektor eksposur-confidence telah dinilai.

## Latihan

1. Jalankan `code/main.py`. Sapu σ dalam {0.5, 1.0, 2.0} dan laporkan trade-off (ε, δ)-akurasi. Identifikasi titik di mana utilitas runtuh.

2. Implementasikan penyisipan canary dan sebuah tes log-loss. Ukur laju deteksi sebelum dan sesudah DP-SGD pada σ = 1.0.

3. Baca Nasr et al. 2025 tentang ekstraksi data training. Mengapa keberhasilan ekstraksi tidak runtuh di bawah ε yang moderat? Apa implikasinya bagi MIA-sebagai-evaluasi?

4. Rancang sebuah deployment yang menggunakan PMixED (arXiv:2403.15638) yang beroperasi sepenuhnya saat inference. Threat model apa yang ditangani PMixED yang tidak ditangani DP-SGD?

5. Sketsa serangan DP Reversal via LLM Feedback. Rancang sebuah countermeasure yang membatasi kebocoran confidence-score dan perkirakan biaya deploynya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| DP | "(ε, δ)-differential privacy" | Formal privacy: output distribution close under neighbouring-dataset change |
| DP-SGD | "noise-injected SGD" | Gradient clipping + Gaussian noise addition; standard DP training |
| LoRA + DP-SGD | "efficient private fine-tune" | DP-SGD on low-rank adapters; standard 2025 configuration |
| MIA | "membership inference" | Attack that determines whether an example was in training data |
| Canary | "inserted watermark example" | Unique training example used to measure DP leakage |
| PMixED | "private inference mixture" | Inference-time DP via mixture-of-experts on next-token distributions |
| DP Reversal | "confidence leakage attack" | Attack that uses a model's confidence as an oracle for re-identification |

## Bacaan Lanjutan

- [Abadi et al. — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) — the standard DP training algorithm
- [Carlini et al. — Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) — the canonical extraction paper
- [Duan et al. — Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) — limited-success MIA
- [Kowalczyk et al. — Auditing DP for LLMs (arXiv:2503.06808, March 2025)](https://arxiv.org/abs/2503.06808) — resolution of the tension
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638) — inference-time private prediction
