# Evaluasi — FID, CLIP Score, Preferensi Manusia

> Setiap leaderboard model generatif mengutip FID, CLIP score, dan win rate dari arena preferensi-manusia. Setiap angka punya mode kegagalan yang bisa di-game peneliti yang bertekad. Kalau kamu tidak tahu mode kegagalannya, kamu tidak bisa membedakan perbaikan nyata dari run gaming.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 01 (Taxonomy), Phase 2 · 04 (Evaluation Metrics)
**Time:** ~45 minutes

## Masalahnya

Model generatif dinilai pada *kualitas sampel* dan *kepatuhan conditioning*. Keduanya tidak punya ukuran closed-form. Modelmu harus merender 10.000 gambar; sesuatu harus memberi mereka angka; kamu harus memercayai angka melintasi keluarga model, melintasi resolusi, melintasi arsitektur. Tiga metrik selamat dari ujian 2014-2026:

- **FID (Fréchet Inception Distance).** Jarak antara dua distribusi — nyata dan terbangkitkan — di ruang fitur network Inception. Lebih rendah lebih baik.
- **Skor CLIP.** Kemiripan cosine antara embedding CLIP-gambar dari gambar terbangkitkan dan embedding CLIP-teks prompt. Lebih tinggi lebih baik. Mengukur kepatuhan prompt.
- **Preferensi manusia.** Adu dua model head-to-head pada prompt yang sama, minta manusia (atau model kelas-GPT-4) memilih yang lebih baik, agregat menjadi skor Elo.

Kamu juga akan melihat: IS (inception score, sebagian besar pensiun), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. Masing-masing mengoreksi satu kegagalan sebelumnya.

## Konsepnya

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID — kualitas sampel

Heusel et al. (2017). Langkah:

1. Ekstrak fitur Inception-v3 (2048-D) untuk N gambar nyata dan N terbangkitkan.
2. Fit Gaussian ke setiap pool: hitung mean `μ_r, μ_g` dan kovariansi `Σ_r, Σ_g`.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

Interpretasi: jarak Fréchet antara dua Gaussian multivariat di ruang fitur. Lebih rendah = distribusi lebih mirip.

Mode kegagalan:
- **Bias pada N kecil.** FID adalah mean-squared atas distribusi fitur — N kecil mengunder-estimasi kovariansi, memberi FID palsu-rendah. Selalu pakai N ≥ 10.000.
- **Bergantung-Inception.** Inception-v3 dilatih pada ImageNet. Domain jauh dari ImageNet (wajah, seni, gambar teks) menghasilkan FID tak bermakna. Pakai ekstraktor fitur khusus-domain.
- **Gaming.** Overfitting ke prior Inception memberi FID rendah tanpa perbaikan kualitas visual. Kalahkan dengan CMMD (di bawah).

### Skor CLIP — kepatuhan prompt

Radford et al. (2021). Untuk gambar terbangkitkan + prompt:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

Rata-ratakan pada 30k gambar terbangkitkan → skalar yang bisa dibandingkan antar model.

Mode kegagalan:
- **Titik-buta CLIP sendiri.** CLIP punya penalaran komposisional lemah ("a red cube on a blue sphere" sering gagal). Model bisa peringkat baik pada skor CLIP tanpa benar-benar mengikuti prompt kompleks.
- **Bias prompt-pendek.** Prompt pendek punya lebih banyak kecocokan CLIP-gambar di alam liar. Prompt lebih panjang punya skor CLIP lebih rendah secara mekanis.
- **Gaming prompt.** Menyertakan "high quality, 4k, masterpiece" dalam prompt menggelembungkan skor CLIP tanpa memperbaiki binding gambar-teks.

CMMD (Jayasumana et al., 2024) memperbaiki sebagiannya: memakai fitur CLIP alih-alih Inception, maximum-mean discrepancy alih-alih Fréchet. Lebih baik mendeteksi perbedaan kualitas halus.

### Preferensi manusia — ground truth

Pilih pool prompt. Bangkitkan dengan model A dan model B. Tunjukkan pasangan ke manusia (atau juri LLM kuat). Agregat kemenangan menjadi skor Elo atau Bradley-Terry. Benchmark:

- **PartiPrompts (Google)**: 1.600 prompt beragam, 12 kategori.
- **HPSv2**: 107k anotasi manusia, dipakai luas sebagai proksi otomatis.
- **ImageReward**: 137k pasang preferensi prompt-gambar, berlisensi-MIT.
- **PickScore**: dilatih pada 2.6M preferensi Pick-a-Pic.
- **Arena gambar ala-Chatbot-Arena**: https://imagearena.ai/ dan lainnya.

Mode kegagalan:
- **Variansi juri.** Non-ahli punya preferensi berbeda dari ahli. Pakai keduanya.
- **Distribusi prompt.** Prompt cherry-picked menguntungkan satu keluarga. Selalu dokumentasikan.
- **Reward hacking juri-LLM.** Juri-GPT-4 terkecoh oleh output cantik-tapi-salah. Triangulasi dengan manusia.

## Pakai bersama

Laporan eval produksi seharusnya mencakup:

1. FID pada 10-30k sampel terhadap distribusi nyata held-out (kualitas sampel).
2. Skor CLIP / CMMD pada sampel yang sama vs prompt mereka (kepatuhan).
3. Win rate dalam arena ter-blind vs model sebelumnya (preferensi keseluruhan).
4. Analisis mode kegagalan: 50 output sampel acak, ditandai untuk masalah yang diketahui (anatomi tangan, rendering teks, jumlah objek konsisten).

Satu metrik tunggal adalah kebohongan. Tiga metrik yang saling menguatkan + reviu kualitatif adalah klaim.

```figure
gx-fid-distributions
```

## Bangun Sendiri

`code/main.py` mengimplementasikan FID, ala-CLIP-score, dan agregasi Elo pada "vektor fitur" sintetis (kita memakai vektor 4-D sebagai pengganti fitur Inception). Kamu melihat:

- Komputasi FID pada N kecil dan N besar — biasnya.
- "Skor CLIP" sebagai kemiripan cosine antar pool fitur.
- Aturan update Elo dari stream preferensi sintetis.

### Langkah 1: FID dalam empat baris

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Langkah 2: cosine-similarity ala-CLIP

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Langkah 3: agregasi Elo

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## Jebakan

- **FID pada N=1000.** Heuristik tak andal di bawah N=10k. Makalah yang melaporkan FID N-rendah sedang gaming.
- **Membandingkan FID lintas resolusi.** Resize 299×299 milik Inception mengubah distribusi fitur. Bandingkan hanya pada resolusi sebanding.
- **Melaporkan satu seed.** Jalankan minimal 3 seed. Laporkan std.
- **Inflasi skor CLIP via negative prompt.** Sebagian pipeline mendongkrak CLIP dengan over-fit ke prompt. Cek saturasi visual.
- **Bias Elo dari overlap prompt.** Kalau kedua model melihat prompt benchmark saat training, Elo tak bermakna. Pakai set prompt held-out.
- **Skew crowd berbayar eval manusia.** Anotator Prolific, MTurk condong lebih-muda / ramah-teknologi. Campur dengan ahli seni/desain yang direkrut.

## Gunakan

Protokol eval produksi pada 2026:

| Pillar | Minimum | Recommended |
|--------|---------|-------------|
| Sample quality | FID on 10k vs held-out real | + CMMD on 5k + FID on subset per category |
| Prompt adherence | CLIP score on 30k | + HPSv2 + ImageReward + VQA-style question answering |
| Preference | 200 blinded pairs vs baseline | + 2000 paired human + LLM-judge + Chatbot Arena |
| Failure analysis | 50 hand-flagged | 500 hand-flagged + automated safety classifier |

Keempat pilar dalam satu laporan = klaim. Satu saja = marketing.

## Rilis

Simpan `outputs/skill-eval-report.md`. Skill menerima checkpoint model baru + baseline dan mengeluarkan rencana eval penuh: ukuran sampel, metrik, probe mode-kegagalan, kriteria sign-off.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Bandingkan FID pada N=100 vs N=1000 pada distribusi sintetis yang sama. Laporkan magnitudo bias.
2. **Sedang.** Implementasikan CMMD dari fitur ala-CLIP sintetis (lihat Jayasumana et al., 2024 untuk rumusnya). Bandingkan sensitivitas terhadap perbedaan kualitas vs FID.
3. **Sulit.** Replikasi setup HPSv2: ambil 1000 pasang gambar-prompt dari subset Pick-a-Pic, fine-tune scorer kecil berbasis-CLIP pada preferensi, dan ukur kesepakatannya dengan set held-out.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | Fréchet distance of Gaussian fits to real vs gen Inception features. |
| CLIP score | "Text-image similarity" | Cosine similarity between CLIP image and text embeddings. |
| CMMD | "FID's replacement" | CLIP-feature MMD; less biased, no Gaussian assumption. |
| IS | "Inception score" | Exp KL(p(y|x) \|\| p(y)); correlates poorly on modern models, retired. |
| HPSv2 / ImageReward / PickScore | "Learned preference proxies" | Small models trained on human preferences; used as automatic judges. |
| Elo | "Chess rating" | Bradley-Terry aggregation of pairwise wins. |
| PartiPrompts | "The benchmark prompt set" | 1,600 Google-curated prompts across 12 categories. |
| FD-DINO | "Self-sup replacement" | FD using DINOv2 features; better for out-of-ImageNet domains. |

## Catatan produksi: evaluasi juga workload inference

Menjalankan FID pada 10k sampel berarti membangkitkan 10k gambar. Untuk SDXL base 50 langkah pada 1024² di satu L4, itu ~11 jam inference single-request. Budget evaluasi itu nyata, dan pembingkaian persis skenario offline-inference (maksimalkan throughput, abaikan TTFT):

- **Batch keras, lupakan latensi.** Eval offline = batching statis pada ukuran terbesar yang muat di memori. `pipe(...).images` dengan `num_images_per_prompt=8` pada H100 80GB berjalan 4-6× lebih cepat wall-clock daripada single-request.
- **Cache fitur nyata.** Ekstraksi fitur Inception (FID) atau CLIP (skor-CLIP, CMMD) atas set referensi nyata dijalankan *sekali*, disimpan sebagai `.npz`. Jangan hitung ulang per eval.

Untuk gate CI / regresi: jalankan FID + skor CLIP pada subset 500 sampel per PR (~30 mnt); jalankan FID 10k penuh + HPSv2 + Elo tiap malam.

## Bacaan Lanjutan

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — makalah FID.
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD.
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP.
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2.
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward.
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts.
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — survei mode-kegagalan.
