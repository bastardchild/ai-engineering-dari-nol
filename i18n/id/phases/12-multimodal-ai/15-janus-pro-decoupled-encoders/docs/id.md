# Janus-Pro: Encoder Ter-Decouple untuk Model Multimodal Terpadu

> Model multimodal terpadu punya ketegangan yang tak terhindarkan. Pemahaman menginginkan fitur semantik — output vektor SigLIP atau DINOv2 yang kaya informasi level-konsep. Generasi menginginkan kode yang ramah-rekonstruksi — token VQ yang menyusun kembali menjadi piksel tajam. Kedua tujuan tidak kompatibel dalam satu encoder. Janus (DeepSeek, Oktober 2024) dan Janus-Pro (DeepSeek, Januari 2025) berargumen perbaikannya adalah berhenti mencoba: decouple kedua encoder. Berbagi badan transformer antar tugas, tapi arahkan pemahaman melalui SigLIP dan generasi melalui tokenizer VQ. Pada 7B, Janus-Pro mengalahkan DALL-E 3 di GenEval sambil menyamai LLaVA di MMMU. Pelajaran ini membaca mengapa dua encoder bekerja di titik di mana satu encoder gagal.

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 minutes

## Tujuan Pembelajaran

- Menjelaskan mengapa satu encoder bersama yang dipaksa berbagi berkompromi pada kualitas pemahaman atau generasi.
- Mendeskripsikan routing Janus-Pro: fitur SigLIP di sisi input untuk pemahaman, token VQ di input dan output untuk generasi.
- Menelusuri data-mix scaling yang membuat Janus-Pro berhasil di titik di mana Janus tidak.
- Membandingkan arsitektur ter-decouple (Janus-Pro), coupled-continuous (Transfusion), dan coupled-diskrit (Show-o).

## Masalahnya

Model terpadu berbagi satu badan transformer antara pemahaman dan generasi. Upaya sebelumnya (Chameleon, Show-o, Transfusion) semuanya memakai satu tokenizer visual untuk kedua arah. Tokenizer adalah kompromi:

- Dioptimalkan untuk rekonstruksi (generasi): VQ-VAE menangkap detail piksel halus tapi menghasilkan token dengan kekohärenan semantik lemah.
- Dioptimalkan untuk semantik (pemahaman): embedding SigLIP mengelompokkan gambar "kucing" dekat token "kucing" tapi tidak mengizinkan rekonstruksi yang baik.

Show-o dan Transfusion membayar ini dengan pajak kualitas yang terlihat pada satu arah. Janus-Pro bertanya: mengapa memerlukan satu tokenizer ketika tugasnya punya kebutuhan berbeda?

## Konsepnya

### Encoding visual ter-decouple

Arsitektur Janus-Pro memisahkan kedua encoder:

- Jalur pemahaman. Gambar input → SigLIP-SO400m → MLP 2-layer → badan transformer.
- Jalur generasi. Gambar input (jika meng-kondisikan pada gambar yang ada) → tokenizer VQ → ID token → badan transformer.
- Output generasi. Token gambar yang diprediksi transformer → decoder VQ → piksel.

Badan transformer dibagi bersama. Segala sesuatu di hulu dan hilir badan bersifat spesifik-tugas.

Input didiskriminasikan oleh format prompt: tag `<understand>` mengarahkan melalui SigLIP; `<generate>` mengarahkan melalui VQ. Atau routing-nya implisit dari tugas.

### Mengapa ini bekerja

Loss pemahaman mendapat fitur SigLIP, yang pra-training gaya CLIP sudah tune untuk kemiripan semantik. Benchmark persepsi model membaik dibanding Show-o / Transfusion karena fitur input lebih baik untuk tugasnya.

Loss generasi mendapat token VQ, yang telah sebuah tokenizer tune untuk rekonstruksi. Kualitas gambar membaik dibanding Show-o karena kode VQ menyusun kembali ke piksel dengan bersih.

Badan transformer bersama melihat dua distribusi input (SigLIP dan VQ) dan belajar bekerja dengan keduanya. Klaimnya: data yang cukup + parameter yang cukup, badan menyerap pergantian itu.

### Skala data — Janus vs Janus-Pro

Janus (asli, arXiv 2410.13848) memperkenalkan decoupling tapi pada skala kecil (1.3B param, data terbatas). Janus-Pro (arXiv 2501.17811) menskalakan:

- 7B param (vs 1.3B).
- 90M pasangan teks-gambar untuk tahap 1 (alignment) naik dari 72M.
- 72M untuk tahap 2 (terpadu) naik dari 26M.
- Menambahkan 200k sampel instruksi image-gen untuk tahap 3.

Intinya: Janus-Pro-7B menyamai LLaVA di MMMU (60.3 vs ~58) dan mengalahkan DALL-E 3 di GenEval (0.80 vs 0.67). Satu model open, kompetitif di kedua sisi spektrum terpadu.

### JanusFlow — varian rectified flow

JanusFlow (arXiv 2411.07975) menukar jalur generasi VQ dengan jalur generasi rectified-flow (continuous). Pecahannya menjadi SigLIP-untuk-pemahaman + rectified-flow-untuk-generasi. Plafon kualitas terangkat lebih jauh. Arsitektur tetap encoder-ter-decouple-badan-bersama.

### Tugas badan bersama

Badan transformer memproses sekuens terpadu tapi dengan dua distribusi input. Tugasnya:

- Untuk pemahaman: mengonsumsi fitur SigLIP + token teks → memancarkan teks secara autoregressive.
- Untuk generasi: mengonsumsi token teks + (opsional token VQ gambar) → memancarkan token VQ gambar secara autoregressive.

Badan tidak memiliki bobot spesifik-modalitas per blok. Ia adalah transformer bergaya-teks yang kamu harapkan ditemukan di dalam Qwen atau Llama, plus dua adapter input.

Menariknya, ini berarti badan Janus-Pro dapat diinisialisasi dari LLM yang sudah pretrain. Janus-Pro memang menginisialisasi dari DeepSeek-MoE-7B. Pilihan itu penting: LLM menyumbang kemampuan penalaran yang sulit dicapai model terpadu pure-from-scratch.

### Dibandingkan dengan InternVL-U

InternVL-U (Pelajaran 12.10) adalah lanjutan 2026. Ia menggabungkan:

- Pra-training multimodal native (backbone InternVL3).
- Routing encoder-ter-decouple (SigLIP masuk, kepala VQ + diffusion keluar).
- Pemahaman + generasi + editing terpadu.

InternVL-U menyerap pilihan arsitektural Janus-Pro ke dalam kerangka yang lebih besar. Ide encoder-ter-decouple kini menjadi default untuk model terpadu berskala besar.

### Keterbatasan

Encoder ter-decouple menambah kompleksitas arsitektur. Dua tokenizer untuk dilatih, dua jalur input untuk dipelihara, dua set mode kegagalan. Untuk produk yang tidak membutuhkan generasi, Janus-Pro over-engineered — pilih model pemahaman keluarga LLaVA.

Untuk produk yang tidak membutuhkan pemahaman, Janus-Pro overqualified — pilih model Stable Diffusion 3 / Flux.

Untuk produk yang membutuhkan keduanya, Janus-Pro kini adalah arsitektur open referensi.

```figure
l5-janus-decouple
```

## Gunakan

`code/main.py` mensimulasikan routing Janus-Pro:

- Dua encoder tiruan: mirip-SigLIP (menghasilkan vektor semantik dim-256) dan mirip-VQ (menghasilkan kode bilangan bulat).
- Router prompt yang memilih encoder berdasarkan tag tugas.
- Badan bersama (pengganti) yang memproses sekuens token terlepas dari encoder mana yang menghasilkannya.
- Peralihan dari tahap 1 (alignment) ke jadwal bertahap sampel berbobot tahap 3 (instruction tune).

Cetak jalur yang di-route untuk 3 contoh: QA gambar, T2I, editing gambar.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-decoupled-encoder-picker.md`. Diberi produk yang menginginkan generasi terpadu + pemahaman pada kualitas kira-kira frontier, ia memilih Janus-Pro, JanusFlow, atau InternVL-U dengan rekomendasi skala-data konkret.

## Latihan

1. Janus-Pro-7B mengalahkan DALL-E 3 di GenEval. Jelaskan mengapa model open 7B dapat menyamai model proprietary frontier pada generasi tapi tidak pada pemahaman.

2. Implementasikan fungsi router: diberi teks prompt, klasifikasikan sebagai `understand` atau `generate`. Bagaimana kamu menangani prompt ambigu seperti "deskripsikan lalu gambar-sketsa"?

3. JanusFlow menggantikan jalur VQ dengan rectified flow. Apa yang kini dihasilkan badan transformer, dan apa yang berubah pada loss-nya?

4. Usulkan tugas keempat yang dapat ditangani arsitektur Janus-Pro dengan satu encoder ter-decouple tambahan. Contoh: segmentasi gambar (gaya DINO), kedalaman (gaya MiDaS).

5. Baca Bagian 4.2 Janus-Pro tentang skala data. Tahap data mana yang menyumbang paling besar pada peningkatan kualitas T2I dibanding Janus?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Decoupled encoding | "Two visual encoders" | Separate tokenizer or encoder per direction: semantic for understanding, reconstruction for generation |
| Shared body | "One transformer" | Single transformer processes either encoder's output; no modality-specific weights |
| SigLIP for understanding | "Semantic features" | CLIP-family vision tower providing rich conceptual features but poor reconstruction |
| VQ for generation | "Reconstruction codes" | Vector-quantized tokens that decode cleanly back to pixels |
| JanusFlow | "Rectified-flow variant" | Janus-Pro with a continuous flow-matching generation head instead of VQ |
| Routing tag | "Task tag" | Prompt marker (`<understand>` / `<generate>`) that picks the input encoder |

## Bacaan Lanjutan

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
