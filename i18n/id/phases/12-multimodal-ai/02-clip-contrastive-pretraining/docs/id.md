# CLIP dan Pra-Pelatihan Bahasa Penglihatan Kontrastif

> OpenAI CLIP(2021) membuktikan satu ide yang cukup besar untuk mendukung lima tahun ke depan: menyelaraskan citra encoder dan sebuah teks encoder dalam ruang vektor yang sama hanya menggunakan pasangan teks gambar web yang berisik dan kontras loss. Tidak ada label yang diawasi. 400 juta pasang. Hasilnya embedding ruang tidak zero-shot klasifikasi, pengambilan gambar-teks, dan dicolokkan ke setiap VLM 2026 sebagai miliknya vision tower.SigLIP 2 (2025) mengganti softmax dengan sigmoid dan berskala masa lalu CLIP dengan biaya lebih rendah. Pelajaran ini memandu matematika dari InfoNCE ke sigmoid berpasangan loss dan membangun langkah pelatihan di stdlib Python.

**Type:** Build
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Dapatkan InfoNCE loss dari informasi timbal balik dan menerapkan versi vektor yang stabil secara numerik.
- Jelaskan mengapa sigmoid berpasangan loss(SigLIP) menskalakan ke batch 32768+ tanpa seluruh tuntutan overhead softmax.
- Berlari zero-shot ImageNet klasifikasi dengan membuat templat teks (`a photo of a {class}`) dan mengambil argmax atas kesamaan kosinus.
- Sebutkan keempat tuas tersebut CLIP/SigLIP prapelatihan memberi kamu: ukuran batch, temperature, templat cepat, kualitas data.

## Masalahnya

Pra-CLIP penglihatan diawasi. Kumpulkan dataset berlabel (ImageNet: 1,2 juta gambar, 1000 kelas), latih CNN, kirimkan. Label itu mahal, label bias terhadap apa yang bisa disetujui oleh pemberi label, dan label tidak akan berpindah ke tugas baru tanpa penyesuaian.

Web keterangan gambar memiliki lebih dari satu miliar pasangan berlabel longgar secara gratis. Gambar seekor anjing jenis Golden Retriever dengan teks alternatif "anjing saya Max di taman" membawa sinyal pengawasan — teks tersebut menjelaskan gambar tersebut. Pertanyaannya: bisakah kamu mengubahnya menjadi pelatihan yang bermanfaat?

CLIP Jawabannya: perlakukan pasangan keterangan gambar sebagai tugas yang cocok. Diberikan sekumpulan N gambar dan N keterangan, belajar mencocokkan setiap gambar dengan keterangannya sendiri terhadap N-1 pengecoh. Pengawasannya adalah "dua hal ini saling berkaitan; N-1 ini tidak." Tidak ada label kelas. Tidak ada anotasi manusia. Hanya kontras loss.

Hasilnya embedding ruang melakukan lebih dari itu CLIP dilatih untuk.ImageNet zero-shot berfungsi karena "foto kucing" disematkan di dekat gambar kucing yang tidak pernah diberi label kucing secara eksplisit. Ini adalah taruhan yang muncul setiap VLM 2026.

## Konsepnya

### Yang ganda encoder

CLIP memiliki dua menara:

- Gambar encoder `f`: ViT atau ResNet, menampilkan vektor D-dim per gambar.
- Teks encoder `g`: kecil Transformer, menampilkan vektor D-dim per keterangan.

Kedua menara menormalkan keluarannya ke satuan panjang. Kemiripannya adalah `cos(f(x), g(y)) = f(x)^T g(y)` karena keduanya merupakan unit-norma.

Untuk kumpulan N pasangan (gambar, keterangan), buat matriks kesamaan `S` bentuk `(N, N)`:

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

Di mana `tau` adalah seorang yang terpelajar temperature(CLIP diinisialisasi ke 0,07; dipelajari di ruang log).

### InfoNCE loss

CLIP menggunakan entropi silang simetris pada baris dan kolom:

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

Ini adalah InfoNCE. Softmax di CE memaksa setiap gambar untuk mencocokkan keterangannya lebih dari setiap keterangan lainnya dalam kumpulan. Yang "negatif" adalah semua item batch lainnya. Batch yang lebih besar = lebih banyak negatif = sinyal lebih kuat.CLIP dilatih pada angkatan 32k; skala penting.

### Temperature

`tau` mengontrol ketajaman softmax. Tau rendah → distribusi tajam, efek penambangan negatif keras. Tau tinggi → lembut, semua sampel berkontribusi.CLIP mempelajari log(1/tau), terpotong untuk mencegah keruntuhan.SigLIP 2 memperbaiki tau awal dan sebagai gantinya menggunakan bias yang dipelajari.

### Mengapa skala sigmoid lebih baik (SigLIP)

Softmax membutuhkan seluruh matriks kesamaan yang sinkron. Dalam pelatihan terdistribusi, kamu harus mengumpulkan semuanya embedding ke setiap replika, lalu lakukan softmax. Ini adalah kuadrat ukuran dunia untuk komunikasi.

SigLIP menggantikan softmax dengan sigmoid berdasarkan elemen: untuk setiap pasangan `(i, j)`, itu loss adalah klasifikasi biner dari "apakah ini pasangan yang cocok?" label kelas positif adalah diagonalnya, yang lainnya negatif. Itu loss adalah:

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` jika `i == j`, yang lain 0. Setiap pasangan loss mandiri. Tidak perlu berkumpul semua. Setiap GPU menghitung blok dan jumlah lokalnya.SigLIP 2 timbangan untuk batch 32k-512k murah dimana CLIP akan membutuhkan lebih banyak komunikasi secara proporsional.

### Klasifikasi zero-shot

Diberikan N nama kelas, untuk setiap kelas buatlah templat teks:

```
"a photo of a {class}"
```

Sematkan setiap templat dengan teks encoder. Sematkan gambar kamu dengan gambar tersebut encoder. Kesamaan kosinus Argmax = kelas prediksi. Tidak ada pelatihan pada kelas sasaran.

Templat yang cepat penting.CLIP Makalah asli menggunakan 80 templat per kelas (polos, artistik, foto, lukisan, dll.) dan dirata-ratakan embedding. +3 ImageNet poin. Penggunaan modern biasanya memilih satu atau dua templat.

### Probe linier dan penyempurnaan

Zero-shot adalah garis dasar. Probe linier (latih satu lapisan linier di atas lapisan beku CLIP fitur untuk kelas target kamu) ketukan zero-shot pada tugas dalam domain. Penyempurnaan penuh mengalahkan pemeriksaan linier pada dalam domain tetapi dapat merugikan zero-shot transfer. Tiga rezim dengan tiga trade-off.

### SigLIP 2: NaFlex dan fitur padat

SigLIP 2 (2025) menambahkan:
- NaFlex: model tunggal menangani rasio aspek dan resolusi variabel.
- Fitur padat yang lebih baik untuk segmentasi dan estimasi kedalaman, menargetkan penggunaan sebagai beku backbone di VLM.
- Multibahasa: dilatih dalam 100+ bahasa di mana CLIP hanya dalam bahasa Inggris.
- skala param 1B dimana CLIP mencapai puncaknya pada 400M.

Pada tahun 2026 VLM terbuka, SigLIP 2 SO400m/14 adalah standarnya vision tower.CLIP tetap menjadi default untuk pengambilan gambar-teks murni jika spesifik LAION Distribusi pelatihan -2B cocok dengan pola kueri kamu.

### SELURUH, DASAR, OpenCLIP, EVA-CLIP

ALIGN (Google, 2021): ide yang sama dengan CLIP, skala pasangan 1,8B, 90% berisik. Skala data yang terbukti berisik. OpenCLIP (LAION): reproduksi terbuka CLIP pada LAION-400M / 2B, berbagai skala, pos pemeriksaan terbuka.EVA-CLIP: menginisialisasi dari pemodelan gambar bertopeng; kuat backbone untuk VLM. DASAR: Google CLIP+ ALIGN hibrida. Semua keluarga sama, data dan tuning berbeda.

### Itu zero-shot langit-langit

CLIP-model kelas membatasi sekitar 76%ImageNet zero-shot(CLIP-G, OpenCLIP-G). Beyond memerlukan data yang jauh lebih besar (SigLIP 2 mendapat 80%+) atau perubahan arsitektur (head yang diawasi, lebih banyak parameter). Tolok ukurnya sudah jenuh; nilai sebenarnya adalah embedding ruang yang dikonsumsi VLM hilir.

```figure
multimodal-fusion
```

## Gunakan

`code/main.py` mengimplementasikan:

1. Sebuah mainan ganda encoder(fitur gambar berbasis hash, fitur text char) sehingga kamu dapat melihat bentuk InfoNCE tanpa numpy.
2. InfoNCE loss dalam Python murni (stabilitas numerik melalui log-sum-exp).
3. Sigmoid berpasangan loss untuk perbandingan.
4. A zero-shot klasifikasi rutin: menghitung kesamaan kosinus terhadap sekumpulan perintah teks, argmax untuk prediksi.

Jalankan dan tonton loss melengkung. Angka absolutnya hanyalah mainan; bentuknya sesuai dengan aslinya CLIP pelatih memancarkan.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-clip-zero-shot.md`. Dengan adanya sekumpulan gambar (melalui jalur) dan daftar kelas target, ia membuat perintah teks dengan CLIP templat, menyematkan kedua sisi dengan pos pemeriksaan yang dinyatakan (misalnya, `openai/clip-vit-large-patch14`), dan menampilkan prediksi 1 teratas / 5 teratas dengan skor kesamaan. Skill menolak untuk membuat klaim tentang kelas yang tidak ada dalam daftar cepat.

## Latihan

1. Terapkan InfoNCE untuk batch 4 pasang dengan tangan. Buat matriks kesamaan 4x4, jalankan softmax, pilih diagonal, hitung entropi silang. Verifikasi implementasi Python kamu terhadap perhitungan tangan ini.

2. SigLIP menggunakan parameter bias `b` sebagai tambahan temperature: `S'[i,j] = S[i,j]/tau + b`. Peran apa yang dilakukannya `b` mainkan ketika batch memiliki ketidakseimbangan kelas yang besar (lebih banyak negatif daripada positif per baris)? Membaca SigLIP Bagian 3 (arXiv:2303.15343).

3. Membangun sebuah zero-shot pengklasifikasi untuk kucing vs anjing. Coba dua templat cepat: `a photo of a {class}` Dan `a picture of a {class}`. Ukur akurasi pada 100 gambar uji. Apakah ansambel templat mengalahkan yang tunggal?

4. Hitung biaya komunikasi softmax InfoNCE vs sigmoid berpasangan untuk 512-GPU yang dijalankan pada batch 32k. Skala mana yang O(N), mana yang O(N^2)? Mengutip SigLIP Bagian 4.

5. Baca makalah hukum penskalaan OpenCLIP (arXiv:2212.07143, Cherti dkk.). Reproduksi kesimpulan mereka untuk penskalaan data dari gambar: pada ukuran model tetap, apa hubungan log-linear antara ImageNet zero-shot akurasi dan ukuran data pelatihan?

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| InfoNCE | "Contrastive loss" | Cross-entropy over a batch's similarity matrix; each item's positive is its paired item, negatives are everything else |
| Sigmoid loss | "SigLIP loss" | Per-pair binary cross-entropy; no softmax, no all-gather, scales cheaply in distributed training |
| Temperature | "tau" | Scalar that scales logits before softmax/sigmoid; controls sharpness of the distribution |
| Zero-shot | "no-finetune classification" | Use text prompts to construct class embeddings and classify by cosine similarity; no training on target classes |
| Prompt template | "a photo of a ..." | Text scaffold around a class name; affects zero-shot accuracy by 1-5 points |
| Dual encoder | "Two-tower" | One image encoder + one text encoder, outputs in shared D-dim space |
| Hard negative | "Tough distractor" | A negative similar enough to the positive that the model has to work to separate them |
| Linear probe | "Frozen + one layer" | Train only a linear classifier on top of frozen features; measures feature quality |
| NaFlex | "Native flexible resolution" | SigLIP 2 capability to ingest images at any aspect ratio and resolution without resizing |
| Temperature scaling | "log-parametrized tau" | CLIP parametrizes `log(1/tau)` so gradients behave; clips to prevent collapse to near-zero tau |

## Bacaan Lebih Lanjut

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020)— itu CLIP kertas.
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343)—SigLIP.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)— multibahasa +NaFlex.
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918)— menskalakan dengan data web yang berisik.
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143)— Hukum penskalaan OpenCLIP.
