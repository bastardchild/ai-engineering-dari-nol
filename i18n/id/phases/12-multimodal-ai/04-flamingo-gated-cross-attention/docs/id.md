# Flamingo dan Gated Cross-Attention untuk VLM Sedikit Pemotretan

> Flamingo DeepMind (2022) melakukan dua hal sebelum orang lain. Ini menunjukkan bahwa satu model dapat memproses rangkaian gambar, video, dan teks yang disisipkan secara sewenang-wenang. Dan hal ini menunjukkan bahwa VLM dapat belajar dalam konteks — berikan sebuah few-shot prompt dengan tiga pasangan contoh (gambar, keterangan) dan model memberi keterangan pada gambar baru tanpa langkah gradien apa pun. Mekanismenya: gated cross-attention lapisan, disisipkan di antara lapisan LLM yang ada yang dibekukan, dengan gerbang tanh terpelajar yang dimulai dari nol sehingga kemampuan teks LLM dipertahankan pada inisialisasi. Pelajaran ini memandu resampler Perceiver Flamingo dan gated cross-attention arsitektur - nenek moyang dari masukan dan sisipan Gemini Idefics2 token visual.

**Type:** Learn
**Languages:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites:** Phase 12 · 03 (BLIP-2 Q-Former)
**Time:** ~120 minutes

## Tujuan Pembelajaran

- Jelaskan caranya gated cross-attention mempertahankan kemampuan teks LLM yang dibekukan pada inisialisasi melalui tanh(gate) = 0.
- Telusuri resampler Perceiver: N gambar patch→ K memperbaiki kueri "laten" melalui cross-attention.
- Jelaskan bagaimana Flamingo menangani rangkaian gambar-teks yang disisipkan dengan penyembunyian kausal yang menghormati penempatan gambar.
- Reproduksi sebuah few-shot struktur prompt multimodal (3 contoh keterangan gambar kemudian gambar kueri).

## Masalahnya

BLIP-2 memasukkan 32 token visual ke dalam lapisan input LLM yang dibekukan. Berfungsi untuk satu gambar per prompt. Namun bagaimana jika kamu ingin memasukkan *banyak* gambar yang disisipkan dengan teks, seperti pada "ini gambar A, beri keterangan; ini gambar B, beri keterangan; sekarang ini gambar C, beri keterangan"? LLM self-attention perlu menangani token gambar dan token teks dalam satu aliran, dan pertanyaan tentang posisi mana yang dapat menangani gambar mana menjadi rumit.

Jawaban Flamingo: jangan mengubah aliran input LLM sama sekali. Masukkan ekstra cross-attention lapisan antara blok LLM yang ada. Token teks masih mengalir melalui kausal LLM self-attention seperti biasanya. Di antara setiap beberapa blok LLM, token teks juga memperhatikan fitur gambar melalui lapisan baru yang terjaga keamanannya. Gerbang (diinisialisasi ke nol) berarti pada langkah nol lapisan baru tidak dapat dioperasikan — model berperilaku persis seperti LLM yang telah dilatih sebelumnya. Saat pelatihan berlangsung, gerbang terbuka dan informasi visual mulai mengalir.

Pertanyaan kedua yang dijawab Flamingo: bagaimana kamu menangani sejumlah gambar yang bervariasi (0, 1, atau banyak) per prompt? Sebuah resampler Perceiver - kecil cross-attention modul yang mengambil jumlah berapa pun patch kamu memiliki dan menghasilkan sejumlah token laten visual. LLM cross-attention lapisan melihat bentuk yang sama terlepas dari berapa banyak gambar yang ditampilkan.

## Konsepnya

### LLM yang dibekukan

Flamingo dimulai dengan Chinchilla 70B LLM beku. Semua beban 70B tidak tersentuh. Teks yang ada self-attention dan FFN beroperasi normal.

### Pengambil sampel ulang penerima

Untuk setiap gambar di prompt, ViT menghasilkan N patch token. Resampler Perceiver memiliki K laten tetap yang dapat dipelajari (Flamingo menggunakan K=64). Setiap blok resampler terdiri dari dua sub-langkah:

1. Perhatian silang: K yang laten hadir di atas N patch token (Q dari laten, K/V dari patch).
2. Perhatian diri + FFN dalam laten.

Setelah 6 blok resampler, outputnya adalah K=64 token visual dim 1024, berapa pun jumlahnya patch itu ViT diproduksi. Gambar 224x224 (196 patch) dan gambar 480x480 (900 patch) keduanya keluar sebagai 64 token resampler.

Untuk video, resampler diterapkan secara temporal: setiap frame patch menghasilkan 64 laten, dan pengkodean posisi temporal memungkinkan model membedakan t=0 dari t=N. Video lengkapnya menjadi token visual T*64.

### berpagar cross-attention

Di antara setiap M lapisan LLM beku (Flamingo menggunakan M=4), masukkan yang baru gated cross-attention memblokir:

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` adalah skalar yang dapat dipelajari yang diinisialisasi ke nol.
- `tanh(0) = 0`, jadi pada init cabang yang terjaga keamanannya memberikan kontribusi nol.
- Sebagai `alpha` menjauh dari nol, yaitu cross-attention kontribusi tumbuh dengan lancar.
- Sambungan sisa berarti bahkan gerbang yang terbuka penuh tidak menimpa representasi teks LLM; itu hanya menambahkan informasi visual di atasnya.

Ini adalah satu-satunya pilihan desain paling penting di Flamingo: pengkondisian visual bersifat aditif, terjaga keamanannya, dan nol pada inisialisasi. Flamingo pada langkah 0 adalah Chinchilla 70B yang sempurna dengan input teks saja.

### Bertopeng cross-attention untuk input yang disisipkan

Dalam prompt seperti "<image A> caption A <image B> caption B <image C> ?", setiap token teks seharusnya hanya melihat gambar yang muncul sebelumnya secara berurutan. Itu cross-attention mask memberlakukan: token teks pada posisi `t` hanya memperhatikan token resampler gambar yang indeks gambarnya `i < i_t` Di mana `i_t` adalah gambar terbaru sebelum posisi `t`. "Hanya melihat gambar terakhir sebelumnya" atau "melihat semua gambar sebelumnya" keduanya merupakan pilihan yang valid; Flamingo memilih yang pertama.

### Dalam konteks few-shot sedang belajar

Prompt Flamingo terlihat seperti:

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

Model melihat pola penyelesaian dan mengeluarkan "burung" (atau apa pun yang ditampilkan gambar3). Tidak ada langkah gradien. Kemampuan pembelajaran dalam konteks LLM yang dibekukan dilakukan melalui gated cross-attention— inilah inti lucunya dari makalah ini dan mengapa hal itu penting.

### Data pelatihan

Flamingo dilatih pada tiga dataset:

1. MultiModal MassiveWeb (M3W): 43 juta halaman web dengan gambar dan teks yang disisipkan, merekonstruksi urutan bacaan.
2. Pasangan Gambar-Teks (ALIGN + LTIP): 4,4 miliar pasangan.
3. Pasangan Video-Teks (VTP): 27 juta klip video pendek.

OBELICS (2023) adalah reproduksi terbuka dari korpus web yang disisipkan, yang Idefics, Idefics2, dan sebagian besar model terbuka "mirip Flamingo" terus berlatih.

### OpenFlamingo dan Berang-berang

OpenFlamingo (2023) adalah reproduksi terbuka. Arsitektur identik (Perceiver resampler +gated cross-attention pada LLaMA atau MPT beku). Pos pemeriksaan di 3B, 4B, 9B. Kualitas tertinggal dari Flamingo karena basis LLM yang lebih kecil dan data yang lebih sedikit.

Otter (2023) dibangun di atas OpenFlamingo dengan instruction tuning pada MIMIC-IT (dataset instruksi multimodal), ditampilkan gated cross-attention berfungsi untuk mengikuti instruksi juga.

### Keturunan

- Idefik /Idefics2/ Idefics3: Memeluk Wajah gated cross-attention garis keturunan, semakin sederhana (Idefics2 menjatuhkan resampler dan memilih langsung patch token dengan pengumpulan adaptif).
- Transisi Flamingo ke Bunglon: pada tahun 2024 banyak tim beralih ke fusi awal (Pelajaran 12.11); Gaya Flamingo gated cross-attention tetap dalam produksi di mana backbone pembekuan diperlukan.
- Masukan sisipan Gemini: secara konseptual mewarisi fleksibilitas format sisipan Flamingo, meskipun mekanisme pastinya bersifat eksklusif.

### Perbandingan dengan BLIP-2

| | BLIP-2 | Flamingo |
|---|---|---|
| Visual bridge | Q-Former once at input | Gated cross-attention at every M layers |
| Visual tokens | 32 per image | 64 per image per cross-attn layer |
| Frozen LLM | Yes | Yes |
| Few-shot in-context | Weak | Strong — the paper's centerpiece |
| Interleaved inputs | No native support | Yes, the design target |
| Training data | 130M pairs | 1.3B pairs + 43M interleaved pages |
| Parameter count | 188M trained | ~10B trained (cross-attn layers) |
| Compute | Days on 8 A100s | Weeks on thousands of TPUv4 |

Pilih BLIP-2 untuk VQA gambar tunggal dengan anggaran terbatas. Pilih Flamingo/Idefics2 untuk disisipkan, few-shot, atau penalaran multi-gambar.

```figure
cross-attention-fusion
```

## Gunakan

`code/main.py` menunjukkan:

1. Sebuah resampler Perceiver pada 36 palsu patch token dengan 8 laten yang dapat dipelajari (Python murni cross-attention).
2. A gated cross-attention langkah dengan `alpha = 0`→ keluaran sama dengan masukan (LLM tidak berubah), lalu `alpha = 2.0`→ kontribusi visual tercampur.
3. Pembuat topeng disisipkan yang menghasilkan topeng perhatian 2D untuk rangkaian "(gambar 1) (teks 1) (gambar 2) (teks 2)".

## Rilis

Pelajaran ini menghasilkan `outputs/skill-gated-bridge-diagnostic.md`. Diberikan sebuah VLM terbuka Konfigurasinya (resampler Y/N, frekuensi lintas attn, skema gerbang), ini mengidentifikasi elemen garis keturunan Flamingo dan menjelaskan strategi pembekuan. Berguna untuk men-debug mengapa kinerja teks menurun (jawaban: gerbang menjadi terlalu lebar dan terlalu cepat).

## Latihan

1. Hitung jumlah parameter visual Flamingo-9B: 9B LLM + 1,4B gated cross-attention lapisan + 64 juta sampel ulang. Berapa bagian dari total parameter yang dilatih?

2. Terapkan sisa yang terjaga keamanannya `y = tanh(alpha) * cross + x` di PyTorch. Tunjukkan secara eksperimental hal itu dengan `alpha=0`, `y==x` tepatnya di init.

3. Baca OpenFlamingo Bagian 3.2 (arXiv:2308.01390) tentang cara mereka menangani banyak gambar dalam satu batch ketika setiap prompt memiliki jumlah gambar yang berbeda. Jelaskan strategi padding.

4. Mengapa Flamingo cross-attention mask biarkan token teks memperhatikan *hanya gambar terbaru* sebelumnya daripada semua gambar sebelumnya? Baca makalah Flamingo Bagian 2.4 dan jelaskan konsekuensinya.

5. Dalam konteks few-shot: membuat prompt dengan 4 contoh "gambar → warna objek utama" untuk varian Flamingo baru. Jelaskan pola akurasi yang diharapkan saat kamu memvariasikan jumlah contoh dari 0 hingga 8.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Perceiver resampler | "Fixed-latent cross-attention" | Module that produces K fixed tokens from a variable number of input patches |
| Gated cross-attention | "Tanh-gated bridge" | Residual layer `y = tanh(alpha)*cross + x`, learnable alpha, init 0 |
| Interleaved input | "Mixed sequence" | Prompt format with images and text mixed freely in reading order |
| Frozen LLM | "No LLM gradients" | The text LLM's weights do not update; only resampler + cross-attn layers train |
| Few-shot | "In-context examples" | Give a few (image, answer) pairs in the prompt; model generalizes without finetuning |
| OBELICS | "Interleaved web corpus" | Open dataset of 141M web pages with images and text in reading order |
| Chinchilla | "70B frozen base" | Flamingo's frozen text LLM, from DeepMind's Chinchilla paper |
| Gate schedule | "How alpha moves" | The rate at which the cross-attention gate opens during training |
| Cross-attn frequency | "Every M layers" | How often a gated cross-attention block is inserted; Flamingo uses M=4 |
| OpenFlamingo | "Open reproduction" | MosaicML/LAION open checkpoint at 3-9B; architecture-identical to Flamingo |

## Bacaan Lebih Lanjut

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198)— kertas asli.
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390)— reproduksi terbuka.
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527)— korpus web yang disisipkan.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795)— arsitektur Perceiver umum.
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726)— Keturunan Flamingo yang disesuaikan dengan instruksi.
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)— penyederhanaan modern dari pendekatan Flamingo.
