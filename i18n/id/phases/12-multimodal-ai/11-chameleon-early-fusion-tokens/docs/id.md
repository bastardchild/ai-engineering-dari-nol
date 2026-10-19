# Chameleon dan Model Multimodal Early-Fusion Token-Only

> Setiap VLM yang kita lihat sejauh ini menjaga gambar dan teks tetap terpisah. Token visual datang dari vision encoder, mengalir ke projector, lalu bertemu teks di dalam LLM. Vocabulary visi dan teks tidak pernah bertumpang tindih. Chameleon (Meta, Mei 2024) bertanya: bagaimana jika ya? Latih sebuah VQ-VAE yang mengubah gambar menjadi sekuens token diskrit dari vocabulary bersama. Setiap dokumen multimodal kini satu sekuens — token teks dan token gambar berselang-seling, satu loss autoregresif. Efek sampingnya: model bisa menghasilkan output multi-modality — berselang-seling token teks dan token gambar dalam satu panggilan inference. Pelajaran ini membaca tesis early-fusion dan membangun versi main-main end to end.

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Menjelaskan mengapa vocabulary bersama + loss tunggal mengubah apa yang bisa dilakukan model.
- Mendeskripsikan bagaimana VQ-VAE men-tokenisasi gambar menjadi sekuens diskrit yang kompatibel dengan objective next-token transformer.
- Menyebutkan trik stabilitas-training Chameleon: QK-Norm, penempatan dropout, urutan LayerNorm.
- Membandingkan Chameleon vs pendekatan Q-Former BLIP-2 dan mendeskripsikan kapan masing-masing adalah pilihan yang tepat.

## Masalahnya

VLM berbasis adapter (LLaVA, BLIP-2, Qwen-VL) memperlakukan teks dan gambar sebagai dua hal berbeda. Sebuah token teks melewati `embed(text_token)`; sebuah gambar melewati `visual_encoder(image) → projector → ... pseudo_tokens`. Model memiliki dua jalur input yang bergabung di tengah jalan.

Tiga konsekuensi:

1. Hanya bisa mengonsumsi gambar, tidak memancarkannya. Output hanya teks.
2. Dokumen multi-modality (paragraf dan gambar berselang-seling, seperti dalam artikel) canggung — kamu harus parse input multimodal di luar model atau merantai generasi.
3. Ketidakcocokan distribusional. Token visual dan token teks hidup di region berbeda dari ruang hidden, menciptakan masalah alignment halus.

Chameleon menolak premisnya: gambar hanyalah sekuens token diskrit dari vocabulary bersama. Latih model pada dokumen interleaved, satu loss, satu decoder autoregresif, dan kamu membuka generasi multi-modality secara gratis.

## Konsepnya

### VQ-VAE sebagai image tokenizer

Tokenizer adalah autoencoder variasi ter-kuantisasi vektor. Arsitekturnya:

- Encoder: CNN + ViT yang memetakan gambar ke spatial feature map, katakan fitur 32x32 berdimensi 256.
- Codebook: vocabulary yang dipelajari berisi K vektor (Chameleon menggunakan 8192), juga dim 256.
- Kuantisasi: untuk setiap fitur spatial, cari entri codebook terdekat berdasarkan jarak L2. Ganti fitur kontinu dengan indeks integer.
- Decoder: CNN yang membawa fitur ter-kuantisasi kembali ke piksel.

Training: loss rekonstruksi VAE + commitment loss + codebook loss. Indeks codebook membentuk alfabet diskrit untuk gambar.

Untuk Chameleon: satu gambar menjadi 32*32 = 1024 token yang diambil dari vocabulary 8192. Gabungkan dengan token teks (dari vocabulary BPE LLM, katakan 32000). Vocabulary akhir: 40192. Transformer melihat satu sekuens, satu loss.

### Vocabulary bersama

Vocabulary Chameleon menggabungkan token teks, token gambar, dan pemisah modality. Setiap token memiliki satu ID. Lapisan input embedding memetakan setiap ID ke vektor hidden berdimensi D. Proyeksi output memetakan hidden kembali ke logits vocabulary. Softmax memilih token berikutnya, apa pun modality-nya.

Pemisah itu penting: tag `<image>` dan `</image>` membingkai sekuens token-gambar. Saat generasi, jika model memancarkan `<image>`, perangkat lunak hilir mengetahui 1024 token berikutnya adalah indeks VQ untuk dikirim ke decoder untuk rendering piksel.

### Generasi multi-modality

Inference adalah prediksi next-token dalam vocabulary bersama. Prompt contoh: "Draw a cat and describe it." Chameleon memancarkan:

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

Model memilih urutannya secara otonom — ia mungkin menghasilkan gambar lalu teks, teks lalu gambar, atau berselang-seling. Decoder yang sama, loss yang sama.

Bandingkan dengan VLM adapter tempat generasi hanya-teks. Chameleon membuka kembali pertanyaan modalitas output model.

### Stabilitas training — QK-Norm, dropout, urutan LayerNorm

Training early-fusion tidak stabil pada skala besar. Paper Chameleon mendokumentasikan tiga trik:

- QK-Norm. Terapkan LayerNorm pada proyeksi query dan key di dalam attention, sebelum dot product. Mencegah ledakan magnitudo logit pada kedalaman. Digunakan oleh beberapa model besar pasca-2024.
- Penempatan dropout. Dropout setelah setiap residual-add, bukan hanya setelah attention dan MLP. Regularisasi lebih diperlukan saat gradien dari token visual bisa mendominasi.
- Urutan LayerNorm. Pre-LN pada cabang residual (standar), plus LN ekstra pada skip connection blok terakhir. Menstabilkan aliran gradien layer akhir.

Tanpa trik ini, training Chameleon 34B param menyimpang di beberapa checkpoint. Dengan mereka, ia konvergen. Resep training sama kontribusinya dengan arsitektur.

### Plafon rekonstruksi tokenizer

VQ-VAE bersifat lossy. Pada 8192 entri codebook dan 1024 token per gambar 512x512, rekonstruksi PSNR mentok sekitar 26-28 dB. Ini cukup untuk generasi gambar yang dapat dikenali tetapi secara nyata lebih buruk daripada diffusion ruang-kontinu (Stable Diffusion 3 mencapai 32+ dB).

Tokenizer adalah bottleneck-nya. Tokenizer yang lebih baik (MAGVIT-v2, IBQ, SBER-MoVQGAN) mengangkat plafonnya. Emu3 (Pelajaran 12.12) mencapai generasi kualitas-SDXL hanya melalui tokenizer yang lebih baik.

### Chameleon vs BLIP-2 / LLaVA

Chameleon (early fusion, vocabulary bersama):
- Satu loss, satu decoder.
- Menghasilkan output multi-modality.
- Tokenizer adalah plafon kualitasnya.
- Mahal: decoder VQ-VAE per gambar yang dihasilkan di jalur inference.

BLIP-2 / LLaVA (late fusion, menara terpisah):
- Visi masuk, teks keluar saja.
- Menggunakan ulang LLM yang sudah di-pretrain.
- Tanpa bottleneck tokenizer untuk pemahaman.
- Murah: satu kali forward pass.

Pilih berdasarkan tugas. Jika kamu butuh generasi gambar, keluarga Chameleon. Jika kamu hanya butuh pemahaman, VLM-adapter lebih sederhana dan menggunakan ulang lebih banyak komputasi yang sudah di-pretrain.

### Fuyu dan AnyGPT

Fuyu (Adept, 2023) adalah pendekatan terkait: lewati vision encoder terpisah sepenuhnya, berikan patch gambar mentah melalui proyeksi input LLM seolah-olah itu token, tanpa tokenizer. Lebih sederhana daripada Chameleon, kehilangan generasi output vocabulary-bersama.

AnyGPT (Zhan et al., 2024) memperluas Chameleon ke empat modalitas: teks, gambar, ucapan, musik. Trik VQ-VAE yang sama untuk masing-masing, transformer bersama. Generasi any-to-any. Dibahas lebih lanjut di Pelajaran 12.16.

```figure
vq-codebook
```

## Gunakan

`code/main.py` membangun model early-fusion end-to-end main-main:

- Kuantizer bergaya VQ-VAE kecil yang memetakan patch 8x8 ke indeks codebook (K=16).
- Vocabulary bersama dari (id teks 0..31) + (id gambar 32..47) + (pemisah 48, 49).
- Decoder autoregresif main-main (tabel bigram) yang dilatih pada caption sintetis + sekuens token-gambar.
- Loop sampling yang memancarkan token teks + gambar berselang-seling yang diberi sebuah prompt.

Kodenya secara sengaja menjaga transformer tetap kecil (bigram) supaya kamu bisa menelusuri alur sinyal end to end.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-tokenizer-vs-adapter-picker.md`. Diberi spesifikasi produk (hanya-paham vs paham + hasilkan, kualitas gambar yang dibutuhkan, budget biaya), ia memilih di antara keluarga Chameleon (early fusion) dan keluarga LLaVA (late fusion) dan membenarkannya dengan aturan-of-thumb kuantitatif.

## Latihan

1. Chameleon menggunakan K=8192 entri codebook dan 1024 token per gambar 512x512. Perkirakan rasio kompresi vs gambar RGB 24-bit. Apakah itu lossy? Seberapa lossy?

2. Gambar 4K (3840x2160) pada densitas VQ-VAE yang sama menghasilkan berapa token gambar? Dapatkah model bergaya Chameleon menghasilkan gambar 4K dalam satu panggilan inference? Apa yang rusak lebih dulu — konteks, kualitas tokenizer, atau KV cache?

3. Implementasikan QK-Norm dalam Python murni. Diberi query dan key 64-dim, tampilkan dot product sebelum dan setelah LayerNorm. Mengapa kontrol magnitudo penting pada kedalaman?

4. Baca Chameleon Bagian 2.3 tentang stabilitas training. Deskripsikan mode kegagalan persis yang diamati paper pada 34B tanpa QK-Norm. Apa tanda-tanda "norm explosion" itu?

5. Perluas decoder main-main untuk memancarkan respons multi-modality yang diberi prompt teks-murni. Ukur seberapa sering model memilih gambar-dulu vs teks-dulu yang diberi distribusi data-training 60% teks-dulu / 40% gambar-dulu.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Early fusion | "Unified tokens" | Images converted to discrete tokens sharing the transformer's vocabulary from step one |
| VQ-VAE | "Image tokenizer" | CNN + ViT + codebook that maps images to integer indices the transformer can predict |
| Shared vocabulary | "One dictionary" | A single token ID space covering text + image + modality separators |
| QK-Norm | "Attention stabilizer" | LayerNorm applied to query and key before their dot product, prevents norm blowup |
| Mixed-modality generation | "Text + image output" | Inference that autonomously produces interleaved text and image tokens in one pass |
| Codebook size | "K entries" | Number of discrete vectors the VQ-VAE can quantize to; trades compression for fidelity |
| Tokenizer ceiling | "Reconstruction limit" | Best PSNR achievable by decoding VQ tokens; bounds the model's image quality |

## Bacaan Lanjutan

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
