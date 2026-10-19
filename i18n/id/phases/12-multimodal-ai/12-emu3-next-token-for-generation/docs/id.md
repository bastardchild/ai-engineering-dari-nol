# Emu3: Next-Token Prediction untuk Generasi Gambar dan Video

> Emu3 BAAI (Wang et al., September 2024) adalah hasil 2024 yang seharusnya mengakhiri debat diffusion-versus-autoregresif. Satu transformer decoder-only bergaya Llama, dilatih hanya pada objective next-token-prediction, melintasi vocabulary terpadu teks + token gambar VQ + token video VQ 3D, mengalahkan SDXL pada generasi gambar dan LLaVA-1.6 pada persepsi. Tanpa loss CLIP. Tanpa jadwal diffusion. Classifier-free guidance digunakan saat inference untuk kualitas, tetapi objective training inti adalah next-token prediction dengan teacher forcing. Diterbitkan di Nature. Pelajaran ini membaca tesis Emu3 — mengapa tokenizer yang lebih baik plus skala adalah semua yang kamu butuhkan — dan membandingkan dengan pendekatan diffusion.

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 minutes

## Tujuan Pembelajaran

- Menjelaskan mengapa objective next-token loss-tunggal Emu3 bekerja meskipun asumsi lama bahwa diffusion diperlukan untuk kualitas gambar.
- Mendeskripsikan tokenizer video 3D: seperti apa codebook VQ spatiotemporal, mengapa patch membentang waktu.
- Membandingkan Emu3 vs Stable Diffusion XL pada (komputasi training, biaya inference, plafon kualitas).
- Menyebutkan tiga peran yang dimainkan model Emu3 yang sama: Emu3-Gen (generasi gambar), Emu3-Chat (persepsi), Emu3-Stage2 (generasi video).

## Masalahnya

Wisdom konvensional sampai 2024: generasi gambar membutuhkan diffusion. Argumennya: token gambar diskrit kehilangan terlalu banyak informasi untuk merekonstruksi detail, dan sampling autoregresif mengakumulasikan error di ribuan token. Stable Diffusion, DALL-E 3, Imagen, Midjourney semuanya menggunakan beberapa bentuk diffusion. Chameleon (Pelajaran 12.11) sebagian membuktikan ini salah pada skala kecil tetapi tidak menyamai SDXL pada kualitas.

Emu3 menyerang argumen itu secara langsung. Klaimnya: visual tokenizer yang lebih baik + skala yang cukup + loss next-token = generasi gambar yang mengalahkan diffusion dalam model yang sama yang juga melakukan persepsi.

Taruhan itu kontroversial saat diterbitkan. Dua tahun kemudian, keluarga generasi-terpadu open-source (Emu3, Show-o, Janus-Pro, Transfusion) adalah jalur default untuk riset; model frontier produksi tampaknya menggunakan variasi tertentu.

## Konsepnya

### Tokenizer Emu3

Bahan kuncinya adalah visual tokenizer. Emu3 melatih tokenizer kelas IBQ custom (Inverse Bottleneck Quantizer, keluarga SBER-MoVQGAN) pada reduksi resolusi 8x8 per token. Gambar 512x512 menjadi 64x64 = 4096 token pada ukuran codebook 32768.

Ini lebih besar daripada 1024 token Chameleon per 512x512 pada K=8192 tetapi lebih murah per token (lookup codebook lebih kecil, codec lebih sederhana). Metrik kunci: rekonstruksi PSNR pada 30,5 dB, kompetitif dengan ruang latent kontinu Stable Diffusion pada 32 dB.

Untuk video: tokenizer VQ 3D meng-encode patch spatiotemporal (4x4x4 piksel) menjadi satu integer. Klip 4s pada 8 FPS memiliki 32 frame; pada 256x256 dengan reduksi spatial 4x dan temporal 4x, jumlah token adalah (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32.768 token.

Kualitas tokenizer adalah plafonnya. Kontribusi Emu3 sebagian adalah "kami melatih tokenizer yang sangat bagus."

### Training loss-tunggal

Emu3 menggunakan satu objective: next-token prediction pada vocabulary bersama melintasi token teks, token gambar 2D, dan token video 3D. Bobot dikalikan dengan faktor spesifik-modality selama training untuk menyeimbangkan kontribusi, tetapi fungsi loss-nya identik.

Latih pada campuran:
- Generasi gambar: `<text caption> <image> image_tokens </image>`
- Persepsi gambar: `<image> image_tokens </image> <question> text_tokens`
- Generasi video: `<text caption> <video> video_tokens </video>`
- Persepsi video: analog.
- Teks saja: NTP standar.

Model belajar kapan harus memancarkan token gambar vs token teks dari distribusi data. Generasi muncul dari model yang memprediksi token gambar setelah tag `<image>`.

### Classifier-free guidance dan suhu

Generasi gambar autoregresif menjadi jauh lebih baik dengan classifier-free guidance (CFG) saat inference. Emu3 menggunakannya: generate dua kali, sekali dengan caption penuh, sekali dengan caption kosong, campur logits dengan bobot panduan (tipikal 3,0-7,0). Ini trik CFG yang sama yang digunakan diffusion, dipinjam ke pengaturan autoregresif.

Suhu itu penting: terlalu tinggi, artefak; terlalu rendah, mode collapse. Suhu yang direkomendasikan Emu3 adalah 1,0 untuk persepsi, 0,8 untuk generasi gambar.

### Tiga peran, satu model

Emu3 dikirim sebagai tiga API yang berbeda secara fungsional tetapi satu set bobot yang mendasari:

- Emu3-Gen. Generasi gambar. Input teks, output token gambar.
- Emu3-Chat. VQA dan captioning. Input gambar (token), output teks.
- Emu3-Stage2. Generasi video dan video VQA. Input teks atau video, output teks atau video.

Tanpa head spesifik-tugas. Hanya template prompt berbeda. Checkpoint yang sama.

### Benchmark

Dari paper Emu3 (September 2024):

- Generasi gambar: mengalahkan SDXL pada FID MJHQ-30K (5,4 vs 5,6), GenEval keseluruhan (0,54 vs 0,55 — seri statistik), dan komposit Deep-Eval setara.
- Persepsi gambar: mengalahkan LLaVA-1.6 pada VQAv2 (75,1 vs 72,4) dan kira-kira menyamai pada MMMU.
- Generasi video: kualitas klip-4-detik pada FVD kompetitif dengan model yang di-benchmark publik era Sora.

Angkanya tidak selalu menang — Emu3 menukar satu poin di sini untuk poin di sana — tetapi klaim "next-token prediction adalah semua yang kamu butuhkan" dapat dipertahankan lintas modalitas.

### Biaya komputasi

Emu3 dilatih pada ~300 miliar token multimodal dengan model 7B param. GPU-hour kira-kira sebanding dengan pretraining Llama-2-7B (2k-4k GPU-year pada silikon kelas A100). Model diffusion seperti Stable Diffusion 3 melatih dalam budget serupa tetapi membutuhkan teks encoder terpisah dan pipeline yang lebih kompleks.

Saat inference, Emu3 lebih lambat daripada SDXL per gambar: 4096 token gambar pada 30 tok/s adalah ~2 menit per gambar 512x512, vs 2-5 detik untuk SDXL. Speculative decoding dan optimasi KV-cache menyempitkan kesenjangan tetapi tidak menutupnya. Generasi gambar autoregresif berat-komputasi; ini adalah trade-off yang berlaku.

### Mengapa ini penting

Kontribusi mendalam Emu3 bersifat konseptual. Jika next-token prediction menskala untuk menyamai diffusion pada generasi gambar, jalur model-terpadu (satu loss, satu backbone, modalitas apa pun) layak. Model masa depan tidak membutuhkan teks encoder terpisah, jadwal diffusion terpisah, VAE terpisah. Satu transformer, satu tokenizer per modality, skala.

Show-o, Janus-Pro, dan InternVL-U semuanya membangun di atas atau menantang tesis ini. Lab Tiongkok (BAAI, DeepSeek) menerbitkan lebih agresif ke arah ini daripada lab AS sampai 2025.

```figure
l5-emu3-next-token
```

## Gunakan

`code/main.py` membangun dua bagian main-main:

- Kalkulator jumlah tokenizer VQ 2D vs 3D: diberi (resolusi, patch, panjang-klip, FPS), hitung jumlah token untuk gambar vs video.
- Sampler token-gambar autoregresif dengan classifier-free guidance pada suhu.

Implementasi CFG-nya cocok dengan resep Emu3 — campur logits kondisional dan tak-kondisional dengan bobot panduan.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-token-gen-cost-analyzer.md`. Diberi spesifikasi produk generasi (gambar atau video, resolusi target, tier kualitas, budget latensi), ia menghitung jumlah token, biaya inference, dan memilih keluarga Emu3 vs diffusion.

## Latihan

1. Emu3 menghasilkan 4096 token per gambar 512x512 pada reduksi 8x8. Hitung ekuivalennya untuk 1024x1024 dan 2048x2048. Apa yang terjadi pada latensi inference?

2. Baca Emu3 Bagian 3.3 tentang tokenizer video. Deskripsikan bentuk patch VQ 3D dan mengapa itu 4x4x4 bukan 8x8x1.

3. Bobot classifier-free guidance 5,0 vs 3,0: efek visual apa? Telusuri matematikanya di `code/main.py`.

4. Hitung FLOPs training untuk Emu3-7B pada 300B token dan bandingkan dengan Stable Diffusion 3. Mana yang lebih mahal untuk dilatih?

5. Emu3 mengalahkan SDXL pada FID tetapi tidak pada VQAv2 vs VLM spesialis. Jelaskan mengapa pendekatan loss-terpadu menunjukkan kekuatan berbeda vs spesialis pada benchmark berbeda.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Next-token prediction | "NTP" | Standard autoregressive loss: predict token[i+1] given token[0..i]; works for every modality when tokenized |
| IBQ tokenizer | "Inverse bottleneck quantizer" | A class of VQ-VAE with larger codebooks (32768+) and better reconstruction than Chameleon's |
| 3D VQ | "Spatiotemporal quantizer" | Codebook indexed by (time, row, col); one token covers a 4x4x4 pixel cube |
| Classifier-free guidance | "CFG" | Mix conditional and unconditional logits with weight gamma; boosts image quality at inference |
| Unified vocabulary | "Shared tokens" | Text + image + video all draw from the same integer space; model predicts whichever modality comes next |
| MJHQ-30K | "Image gen benchmark" | Midjourney-quality benchmark with 30k prompts; Emu3 reports FID here |

## Bacaan Lanjutan

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
