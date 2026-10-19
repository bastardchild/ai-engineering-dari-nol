# Watermarking — SynthID, Stable Signature, C2PA

> Tiga teknologi membingkai provenance konten buatan-AI tahun 2026. SynthID (Google DeepMind) — watermarking gambar diluncurkan Agustus 2023, teks+video Mei 2024 (Gemini + Veo), teks di-open-source Oktober 2024 via Responsible GenAI Toolkit, detektor multi-media terpadu November 2025 bersama Gemini 3 Pro. Watermarking teks menyesuaikan probabilitas sampling token berikutnya secara tak-terlihat; watermark gambar/video bertahan terhadap kompresi, cropping, filter, dan perubahan frame-rate. Stable Signature (Fernandez et al., ICCV 2023, arXiv:2303.15435) — fine-tune decoder latent diffusion sehingga setiap output mengandung pesan tetap; gambar yang dihasilkan dengan konten ter-crop (10%) terdeteksi >90% pada FPR<1e-6. Lanjutan "Stable Signature is Unstable" (arXiv:2405.07145, Mei 2024) — fine-tuning menghapus watermark sambil mempertahankan kualitas. C2PA — standar metadata yang ditandatangani kriptografis dan tamper-evident (C2PA 2.2 Explainer 2025). Watermarking dan C2PA saling melengkapi: metadata bisa di-strip tapi membawa provenance yang lebih kaya; watermark bertahan melalui transcoding tapi membawa lebih sedikit informasi.

**Type:** Build
**Languages:** Python (stdlib, token-watermark embed + detect)
**Prerequisites:** Phase 10 · 04 (sampling), Phase 01 · 09 (information theory)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan watermarking level-token (gaya SynthID-text) dan mekanisme yang membuatnya dapat dideteksi.
- Jelaskan Stable Signature dan serangan penghapusan 2024 yang memecahkannya.
- Nyatakan peran C2PA dan mengapa ia melengkapi watermarking.
- Jelaskan keterbatasan utama: sinyal khusus-model, robustness di bawah paraphrase, dan serangan yang mempertahankan makna (arXiv:2508.20228).

## Masalahnya

Tahun 2023-2024 menyaksikan deepfake dan konten buatan-AI memasuki konteks politik dan konsumen secara besar-besaran. Watermarking adalah sinyal provenance teknis yang diusulkan: tandai generasi pada saat pembuatan, deteksi nanti. Bukti 2025: tidak ada watermark yang robust tanpa syarat, tapi ditumpuk dengan metadata C2PA, kombinasi tersebut memberikan cerita provenance yang dapat digunakan.

## Konsepnya

### Watermarking teks (gaya SynthID-text)

Mekanisme Kirchenbauer et al. 2023, diproduksi-kan oleh Google:

1. Pada setiap langkah decoding, hash K token sebelumnya untuk menghasilkan sebuah partisi pseudo-acak dari vocabulary menjadi set "green" dan "red".
2. Bias sampling ke arah set green dengan menambahkan δ ke logits green.
3. Generasi berisi lebih banyak token green daripada yang dihasilkan secara kebetulan.

Deteksi: hash ulang setiap prefix, hitung token green dalam generasi, hitung sebuah z-score. Z-score >0 untuk teks ber-watermark, ~0 untuk teks manusia.

Sifat-sifat:
- Tak-terlihat oleh pembaca (δ cukup kecil sehingga kerugian kualitas minor).
- Dapat dideteksi dengan akses ke fungsi partisi vocabulary.
- Tidak robust terhadap paraphrase — menulis ulang teks menghancurkan sinyalnya.

SynthID-text di-open-source pada Oktober 2024 via Responsible GenAI Toolkit Google.

### Stable Signature (gambar)

Fernandez et al. ICCV 2023. Fine-tune decoder latent diffusion sehingga setiap gambar yang dihasilkan mengandung sebuah pesan biner tetap yang tertanam dalam representasi latent. Deteksi di-decode dari latent dengan sebuah neural decoder. Gambar yang di-crop (hingga 10% konten) terdeteksi >90% pada FPR<1e-6.

Mei 2024 "Stable Signature is Unstable" (arXiv:2405.07145): fine-tuning decoder menghapus watermark sambil mempertahankan kualitas gambar. Fine-tuning pasca-generasi yang adversarial murah; robustness adversarial watermark terbatas.

### Detektor terpadu SynthID (November 2025)

Bersama Gemini 3 Pro: sebuah detektor multi-media yang membaca sinyal SynthID dari teks, gambar, audio, dan video dalam satu API. Menyatukan stack provenance Google.

### C2PA

Coalition for Content Provenance and Authenticity. Standar metadata tamper-evident yang ditandatangani kriptografis. C2PA 2.2 Explainer (2025). Sebuah manifes C2PA merekam klaim provenance (siapa yang membuat, kapan, transformasi apa saja) yang ditandatangani dengan kunci pembuatnya.

Melengkapi watermarking:
- Metadata bisa di-strip; watermark tidak (dengan mudah).
- Metadata kaya (rantai provenance penuh); watermark membawa bit.
- C2PA bergantung pada adopsi platform; watermark tertanam secara otomatis.

Google mengintegrasikan keduanya di Search, Ads, dan "About this image."

### Keterbatasan

- **Model-specific.** Watermark SynthID menandai generasi dari model yang mendukung SynthID. Generasi dari model tanpa SynthID tidak di-watermark, jadi "tidak ada sinyal SynthID" bukan bukti keaslian.
- **Paraphrase.** Watermark teks tidak bertahan terhadap paraphrase yang mempertahankan makna.
- **Serangan transformasi.** arXiv:2508.20228 (2025) menunjukkan serangan yang mempertahankan makna dan menghancurkan baik watermark teks maupun banyak watermark gambar.
- **Penghapusan via fine-tuning.** Menurut "Stable Signature is Unstable," fine-tuning pasca-generasi menghapus watermark yang tertanam.

### EU AI Act Article 50

Transparency Code untuk pelabelan konten buatan-AI (draf pertama Desember 2025, draf kedua Maret 2026, final yang diharapkan Juni 2026 menurut [halaman status European Commission](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)). Code masih berupa draf per April 2026 dan timeline dapat berubah. Lapisan regulasi yang memerlukan lapisan teknis. Deepfake harus diberi label.

### Di mana ini cocok di Fase 18

Pelajaran 22-23 tentang apa yang dipancarkan model (data privat, sinyal provenance). Pelajaran 27 membahas tata kelola data training. Pelajaran 24 adalah kerangka regulasi yang memerlukan langkah-langkah teknis ini.

```figure
an-watermark-greenlist
```

## Gunakan

`code/main.py` membangun sebuah watermark teks mainan. Token adalah bilangan bulat 0..N-1; sampling yang di-watermark bias ke arah set green yang didefinisikan oleh hash. Sebuah detektor menghitung z-score token-green. Kamu dapat mengamati deteksi pada generasi 1000-token, menyaksikan paraphrase menghancurkan sinyal, dan mengukur laju false-positive pada teks manusia.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-provenance-audit.md`. Diberi sebuah deployment konten dengan klaim provenance, ia mengaudit: mekanisme watermark (jika ada), rantai penandatanganan C2PA (jika ada), robustness adversarial masing-masing, dan cakupan per-modalitas.

## Latihan

1. Jalankan `code/main.py`. Laporkan z-score untuk generasi 1000-token ber-watermark vs teks yang ditulis oleh manusia. Identifikasi laju false-positive pada ambang kepercayaan 95%.

2. Implementasikan serangan paraphrase yang mengganti 30% token dengan sinonim. Ukur ulang z-score-nya.

3. Baca Kirchenbauer et al. 2023 Bagian 6 tentang robustness. Mengapa watermark teks gagal di bawah paraphrase tapi watermark gambar bertahan terhadap cropping?

4. Rancang sebuah deployment yang menggunakan SynthID-text + metadata C2PA. Jelaskan rantai provenance yang dilihat seorang konsumen. Identifikasi satu mode gagal dari setiap komponen.

5. Hasil 2024 "Stable Signature is Unstable" menunjukkan fine-tuning menghapus watermark gambar. Rancang kontrol deployment yang membatasi serangan ini — misalnya, wajibkan rilis bertanda-tangan dari checkpoint yang telah di-fine-tune.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| SynthID | "Google's watermark" | Cross-modal provenance signal; text, image, audio, video |
| Token watermark | "Kirchenbauer-style" | Biased-sampling text watermark detectable via green-token z-score |
| Stable Signature | "image watermark" | Fine-tuned-decoder watermark; ICCV 2023 |
| C2PA | "the metadata standard" | Cryptographically signed tamper-evident provenance metadata |
| Paraphrase robustness | "does rewording break it" | Text watermark property; currently limited |
| Fine-tune removal | "adversarial unwatermark" | Attack that removes image watermark via decoder fine-tuning |
| Cross-modal detector | "unified SynthID" | November 2025 unified API across modalities |

## Bacaan Lanjutan

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — the token-watermark mechanism
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — image watermark paper
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — the removal attack
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — the cross-modal watermark
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — metadata standard
