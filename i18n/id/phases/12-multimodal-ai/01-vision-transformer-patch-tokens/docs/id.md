# Vision Transformers dan Patch-Token Primitif

> Sebelum melakukan sesuatu yang multimodal, sebuah gambar harus menjadi rangkaian token sebuah Transformer bisa makan. Tahun 2020 ViT kertas menjawab ini dengan 16x16 piksel patch, proyeksi linier, dan posisi embedding. Lima tahun kemudian setiap tahun 2026 model frontier(Claude Opus 4.7 dengan resolusi asli 2576 piksel, Gemini 3.1 Pro, Qwen3.5-Omni) masih dimulai dengan cara ini —encoder berubah dari ViT ke DINOv2 ke SigLIP 2, token register ditambahkan, skema posisi menjadi 2D-RoPE, tetapi primitif tetap dipertahankan. Pelajaran ini membaca patch-token pipeline ujung ke ujung dan membangunnya di stdlib Python sehingga sisa Fase 12 memiliki model mental konkret untuk "token visual".

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time:** ~120 minutes

## Tujuan Pembelajaran

- Ubah gambar berukuran HxWx3 menjadi rangkaian patch token dengan pengkodean posisi yang benar.
- Hitung panjang urutan, jumlah parameter, dan FLOP untuk sebuah ViT dari suatu hal (patch ukuran, resolusi, dimensi hidden, kedalaman).
- Sebutkan tiga peningkatan yang dilakukan ViT dari penelitian tahun 2020 hingga produksi tahun 2026: self-supervised pelatihan awal (DINO/MAE), token register, dan pengepakan resolusi asli.
- Pilih di antara CLS pooling, mean pooling, Dan token register untuk tugas hilir.

## Masalahnya

Transformer beroperasi pada urutan vektor. Teks sudah berupa urutan (byte atau token). Gambar adalah kisi piksel 2D dengan tiga saluran warna — bukan urutan. Jika kamu meratakan setiap piksel, gambar RGB 224x224 menjadi 150.528 token, dan self-attention pada panjang itu adalah non-starter (panjang barisan kuadrat).

Pendekatan pra-2020 memasang ekstraktor fitur CNN ke bagian depan: ResNet menghasilkan peta fitur 7x7 dengan vektor 2048 dim, masukkan 49 token tersebut ke sebuah Transformer. Ini berhasil tetapi mewarisi bias CNN (ekivariansi terjemahan, lokal receptive field) dan kehilangan Transformer selera untuk skala.

Dosovitskiy dkk. (2020) menanyakan pertanyaan blak-blakan: bagaimana jika kita melewatkan CNN? Pisahkan gambar menjadi ukuran tetap patch(katakanlah 16x16 piksel), proyeksikan masing-masing piksel secara linier patch menjadi vektor, tambahkan posisi embedding, dan masukkan urutannya ke vanilla Transformer. Pada saat itu, hal ini merupakan ajaran sesat – pandangan yang tidak berbelit-belit. Dengan data yang cukup (JFT-300M, kalau begitu LAION) itu mengalahkan ResNet pada ImageNet dan terus membaik.

Pada tahun 2026 ViT primitif adalah fondasi yang tidak perlu dipertanyakan lagi. Setiap VLM bobot terbuka vision tower adalah beberapa keturunan (DINOv2, SigLIP 2, CLIP, EVA, InternViT). Pertanyaannya bukan lagi “haruskah kita menggunakan patch?" tapi "apa patch ukuran, jadwal resolusi apa, tujuan prapelatihan apa, pengkodean posisi apa."

## Konsepnya

### Patch sebagai token

Diberikan sebuah gambar `x` bentuk `(H, W, 3)` dan sebuah patch ukuran `P`, kamu mengukir gambar ke dalam kotak `(H/P) x (W/P)` tidak tumpang tindih patch. Setiap patch adalah sebuah `P x P x 3` kubus piksel. Ratakan setiap kubus menjadi sebuah `3 P^2` vektor. Terapkan proyeksi linier bersama `W_E` bentuk `(3 P^2, D)` untuk memetakan masing-masing patch ke dalam dimensi tersembunyi model `D`.

Untuk ViT-B/16 konfigurasi kanonik:
- Resolusi 224, patch ukuran 16 → kisi 14x14 → 196 patch token.
- Setiap patch adalah `16 x 16 x 3 = 768` nilai piksel, diproyeksikan ke `D = 768`.
- Tambahkan yang bisa dipelajari `[CLS]` token → panjang urutan 197.

Itu patch proyeksi secara matematis identik dengan konvolusi 2D dengan ukuran kernel `P`, melangkah `P`, Dan `D` saluran keluaran. Begitulah cara kode produksi mengimplementasikannya -`nn.Conv2d(3, D, kernel_size=P, stride=P)`. Pembingkaian "proyeksi linier" bersifat konseptual; pembingkaian kernel efisien.

### posisional embedding

Patch tidak memiliki tatanan yang melekat — Transformer melihatnya sebagai tas. ViT awal menambahkan posisi 1D yang dapat dipelajari embedding(satu vektor 768 dim per posisi, 197 di antaranya). Berfungsi, tetapi mengaitkan model dengan resolusi pelatihan: pada inferensi kamu harus menginterpolasi tabel posisi jika kamu mengubah kisi.

Visi masa kini backbone gunakan 2D-RoPE (M-RoPE Qwen2-VL, SigLIP default 2) atau posisi 2D yang difaktorkan. 2D-RoPE memutar kueri dan vektor kunci berdasarkan patch indeks (baris, kolom), sehingga model menyimpulkan posisi 2D relatif dari sudut rotasi. Tidak ada tabel posisi. Model ini menangani ukuran grid sewenang-wenang pada inferensi.

### Token CLS, keluaran gabungan, dan token register

Apa yang dimaksud dengan representasi tingkat gambar? Tiga pilihan hidup berdampingan:

1. `[CLS]` token. Tambahkan vektor yang dapat dipelajari ke patch urutan. Bagaimanapun Transformer blok, status tersembunyi token CLS adalah representasi gambar. Diwarisi dari BERT. Digunakan oleh yang asli ViT, CLIP.
2. mean pooling. Rata-rata patch status tersembunyi keluaran token. Digunakan oleh SigLIP, DINOv2, sebagian besar VLM modern.
3. token register. Darcet dkk. (2023) mengamati bahwa ViT yang dilatih tanpa sink token yang eksplisit mengembangkan "artefak" bernorma tinggi patch pembajakan itu self-attention. Menambahkan 4–16 yang bisa dipelajari token register menyerap beban ini dan meningkatkan kualitas prediksi padat (segmentasi, kedalaman).DINOv2 Dan SigLIP 2 keduanya dikirimkan dengan register.

Pilihannya penting untuk tugas-tugas hilir. CLS baik untuk klasifikasi. Untuk VLM yang memberi makan patch token ke dalam LLM, kamu melewatkan pengumpulan seluruhnya — semuanya patch menjadi token masukan LLM. Register dibuang sebelum penyerahan (itu adalah perancah, bukan konten).

### Pra-pelatihan: diawasi, kontrastif, disamarkan, disuling sendiri

Tahun 2020 ViT telah dilatih sebelumnya dengan klasifikasi yang diawasi JFT-300 juta. Dengan cepat digantikan oleh:

- CLIP(2021): teks gambar kontrastif pada 400 juta pasang. Pelajaran 12.02.
- MAE(2021, He dkk.): masker 75% dari patch, rekonstruksi piksel. Diawasi sendiri, bekerja pada gambar murni.
- DINO(2021) /DINOv2(2023): distilasi mandiri dengan siswa-guru, tanpa label, tanpa keterangan. Tahun 2023 DINOv2 ViT-g/14 adalah visual murni terkuat backbone dan default untuk kasus penggunaan "fitur padat".
- SigLIP/SigLIP 2 (2023, 2025): CLIP dengan sigmoid loss Dan NaFlex untuk rasio aspek asli. Yang dominan vision tower pada tahun 2026 VLM terbuka(Qwen, Idefics2, LLaVA-OneVision).

Pilihan pra-pelatihan kamu menentukan apa yang akan dilakukan backbone baik untuk: CLIP/SigLIP untuk pencocokan semantik dengan teks, DINOv2 untuk fitur visual yang padat, MAE sebagai titik awal untuk penyempurnaan hilir.

### Hukum penskalaan

ViT scaling (Zhai et al. 2022) menetapkan bahwa sebuah ViT kualitasnya mematuhi hukum yang dapat diprediksi dalam ukuran model, ukuran data, dan komputasi. Pada komputasi tetap:
- Model lebih besar + lebih banyak data → kualitas lebih baik.
- Patch size adalah pengungkit pada panjang urutan vs kesetiaan.Patch 14 (khas untuk DINOv2/SigLIP SO400m) memberikan lebih banyak token per gambar dibandingkan patch 16; lebih baik untuk OCR dan tugas-tugas padat, lebih buruk untuk kecepatan.
- Resolusi adalah faktor besar lainnya. Beralih dari 224 ke 384 ke 512 hampir selalu membantu, dengan biaya kuadrat di FLOP.

ViT-g/14 (param 1B, patch 14, resolusi 224 → 256 token) dan SigLIP SO400m/14 (param 400M, patch 14) adalah dua pekerja keras encoder untuk tahun 2026 VLM terbuka.

### Jumlah parameter untuk sebuah ViT

Perhitungan lengkapnya ada di dalamnya `code/main.py`. Untuk ViT-B/16 di 224:

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

Setiap tempat parkir bola ViT dengan cara ini sebelum kamu memuat pos pemeriksaan. Itu backbone size menetapkan lantai VRAM kamu di VLM hilir mana pun.

### Konfigurasi produksi 2026

Itu encoder paling VLM terbuka dikirimkan pada tahun 2026 adalah SigLIP 2 SO400m/14 pada resolusi asli (NaFlex). Ini memiliki:
- Parameter 400 juta.
- Patch ukuran 14, resolusi default 384 → 729 patch token per gambar.
- mean pooling untuk tugas tingkat gambar; semua 729 patch mengalir ke LLM untuk VQA.
- 4 token register, dibuang sebelum penyerahan LLM.
- 2D-RoPE dengan penskalaan tingkat gambar untuk rasio aspek asli.

Setiap keputusan dalam konfigurasi itu ditelusuri kembali ke makalah yang dapat kamu baca.

```figure
image-patch-tokens
```

## Gunakan

`code/main.py` adalah sebuah patch tokenizer dan kalkulator geometri. Dibutuhkan (gambar H, W, patch P, tersembunyi D, kedalaman L) dan laporan:

- Bentuk grid dan panjang urutan setelah ditambal.
- Urutan token untuk gambar mainan sintetis 8x8 piksel (berjalan melalui jalur proyek + rata).
- Jumlah parameter dipecah berdasarkan patch penyematan, penyematan posisi, Transformer blok, dan kepala.
- FLOP per forward pass pada resolusi target.
- Tabel perbandingan di seluruh ViT-B/16@224, ViT-L/14@336, DINOv2 ViT-g/14@224, SigLIP SO400m/14 @ 384.

Jalankan. Cocokkan jumlah parameter dengan angka yang dipublikasikan. Bermain dengan patch ukuran dan resolusi untuk merasakan biaya penghitungan token.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-patch-geometry-reader.md`. Diberikan sebuah ViT konfigurasi (patch ukuran, resolusi, dimensi hidden, kedalaman), ini menghasilkan jumlah token, jumlah parameter, dan perkiraan VRAM dengan pembenaran. Gunakan skill ini setiap kali kamu memilih visi backbone untuk VLM - ini mencegah kejutan "token meledak dan konteks LLM saya terisi".

## Latihan

1. Hitung patch-panjang urutan token untuk Qwen2.5-VL pada input asli 1280x720 dengan patch ukuran 14. Bagaimana jika dibandingkan dengan representasi khusus CLS?

2. Bingkai 1080p (1920x1080) pada patch 14 menghasilkan berapa token? Pada 30 FPS dalam video berdurasi 5 menit, berapa total token visual? Biaya mana yang paling menghemat biaya kamu: pengumpulan, pengambilan sampel bingkai, atau penggabungan token?

3. Melaksanakan mean pooling lebih patch token dengan Python murni. Verifikasi itu mean pooling lebih dari 196 token sebuah DINOv2 keluaran sesuai dengan modelnya `forward` kembali ketika kamu meminta pooled embedding.

4. Baca Bagian 3 dari "Transformator Visi Perlu Register" (arXiv:2309.16588). Jelaskan dalam dua kalimat artefak apa yang diserap register dan mengapa hal itu penting untuk prediksi padat hilir.

5. Memodifikasi `code/main.py` untuk mendukung patch-n'-pack: diberikan daftar gambar dengan resolusi berbeda, menghasilkan satu rangkaian paket dan masker perhatian blok-diagonal. Verifikasi dengan Pelajaran 12.06 ketika kamu mencapainya.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Patch | "16x16 pixel square" | A fixed-size non-overlapping region of the input image; becomes one token |
| Patch embedding | "Linear projection" | A shared learned matrix (or Conv2d with stride=P) mapping flattened patch pixels to D-dim vectors |
| CLS token | "Class token" | Prepended learnable vector whose final hidden state represents the whole image; optional in 2026 |
| Register token | "Sink token" | Extra learnable tokens that absorb the high-norm attention artifacts ViTs develop during pretraining |
| Position embedding | "Positional info" | Per-position vector or rotation making the sequence-order-aware; 2D-RoPE is the modern default |
| Grid | "Patch grid" | The (H/P) x (W/P) 2D array of patches for a given resolution and patch size |
| NaFlex | "Native flexible resolution" | SigLIP 2 feature: single model serves multiple aspect ratios and resolutions without retraining |
| Backbone | "Vision tower" | The pretrained image encoder whose patch-token outputs feed the LLM in a VLM |
| Pooling | "Image-level summary" | Strategy to turn patch tokens into one vector: CLS, mean, attention pool, or register-based |
| Patch 14 vs 16 | "Finer vs coarser grid" | Patch 14 produces more tokens per image, better fidelity for OCR, slower; patch 16 is the classic default |

## Bacaan Lebih Lanjut

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929)— asli ViT.
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377)—MAE, self-supervised pelatihan awal.
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193)—distilasi mandiri dalam skala besar, tanpa label.
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588)—token register dan analisis artefak.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)— standar tahun 2026 vision tower.
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560)— hukum penskalaan empiris.
