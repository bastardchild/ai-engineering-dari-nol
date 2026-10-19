# Dari CLIP ke BLIP-2 — Q-Mantan sebagai Jembatan Modalitas

> CLIP menyelaraskan gambar dan teks tetapi tidak dapat menghasilkan keterangan, menjawab pertanyaan, atau mengadakan percakapan. BLIP-2 (Salesforce, 2023) menyelesaikannya dengan jembatan kecil yang dapat dilatih: 32 vektor kueri yang dapat dipelajari menangani proses yang dibekukan ViT fitur melalui cross-attention, lalu masukkan langsung ke aliran masukan LLM yang dibekukan. Parameter jembatan 188M menghubungkan LLM 11B ke sebuah ViT-g/14. Setiap VLM berbasis adaptor hingga tahun 2026 — MiniGPT-4, InstructBLIP, LLaVA sepupu — adalah keturunan. Pelajaran ini membaca arsitektur Q-Former, menjelaskan pelatihan dua tahapnya, dan membuat versi mainan yang memasukkan token visual ke dalam dekoder teks beku.

**Type:** Build
**Languages:** Python (stdlib, cross-attention + learnable-query demo)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 7 (Transformers)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Jelaskan mengapa hambatan yang bisa dilatih antara penglihatan yang membeku encoder dan LLM beku mengalahkan penyesuaian menyeluruh dalam hal biaya dan stabilitas.
- Menerapkan sebuah cross-attention blok tempat sekumpulan kueri tetap yang dapat dipelajari menangani fitur gambar eksternal.
- Ikuti dua tahap prapelatihan BLIP-2: representasi (ITC + ITM + ITG) kemudian generatif (LM loss dengan decoder beku).
- Bandingkan Q-Former dengan proyektor MLP sederhana yang digunakan LLaVA dan berdebat ketika masing-masing pilihan menang.

## Masalahnya

kamu beku ViT yang menghasilkan 256 patch token dim 1408 per gambar. kamu memiliki LLM 7 miliar beku yang mengharapkan token embedding dari dim 4096. Jembatan yang jelas — lapisan linier dari 1408 hingga 4096 — berfungsi, tetapi memberi makan semua 256 patch token ke dalam konteks LLM dikenakan biaya 256 token tambahan per gambar. Lebih dari 32 gambar yang merupakan 8192 token dikonsumsi oleh modalitas visual saja.

Pertanyaan BLIP-2: dapatkah kamu mengompresi representasi gambar 256 token menjadi token yang jauh lebih sedikit (katakanlah 32) sambil menyimpan informasi yang cukup untuk LLM untuk memberi keterangan, menjawab pertanyaan, dan memberi alasan tentang gambar tersebut? Dan bisakah kamu melatih jembatan ini tanpa menyentuh beku backbone, menjaga biaya pelatihan hanya pada parameter jembatan?

Jawabannya: seorang Q-Mantan. 32 vektor "kueri" yang dapat dipelajari yang saling memperhatikan ViT'S patch token, menghasilkan ringkasan visual 32 token yang digunakan LLM. Total parameter 188 juta. Dilatih dengan tujuan kontrastif, serasi, dan generatif sebelum menyentuh LLM.

## Konsepnya

### Kueri yang bisa dipelajari

Trik inti Q-Former: alih-alih membiarkan token teks LLM memperhatikan gambar patch, perkenalkan kumpulan baru 32 vektor kueri yang dapat dipelajari `Q` dan biarkan *mereka* memperhatikan gambar patch. Kueri adalah parameter model — kueri tersebut dipelajari selama pelatihan dan 32 kueri yang sama digunakan untuk setiap gambar.

Setelah cross-attention, setiap kueri menyimpan ringkasan terkompresi dari gambar — "deskripsikan objek utama", "deskripsikan latar belakang", "hitung objek", dll. Kueri tidak secara harfiah berspesialisasi pada label semantik; mereka mempelajari pengkodean apa pun yang dilakukan di hilir loss menjatuhkan.

### Arsitektur

Q-Former berukuran kecil Transformer(12 lapisan, ~100 juta parameter) dengan dua jalur:

1. Jalur kueri: 32 vektor kueri mengalir self-attention(di antara mereka sendiri), lalu cross-attention atas beku ViT'S patch token, lalu FFN.
2. Jalur teks: sebuah BERT-seperti teks encoder berbagi self-attention dan bobot FFN dengan jalur kueri. Perhatian silang dinonaktifkan untuk jalur teks.

Pada waktu pelatihan kedua jalur berjalan. Kueri dan teks berinteraksi melalui berbagi self-attention, yang berarti kueri dapat mengkondisikan teks untuk tugas yang memerlukannya (ITM, ITG). Pada waktu inferensi untuk handoff VLM, hanya kueri yang mengalir, menghasilkan 32 token visual.

### Pelatihan dua tahap

Latihan awal BLIP-2 dalam dua tahap:

Tahap 1: pembelajaran representasi (tanpa LLM). Tiga loss:
- ITC (kontrastif gambar-teks): CLIP-gaya kontras antara token kueri gabungan dan token CLS teks.
- ITM (pencocokan gambar-teks): pengklasifikasi biner - apakah pasangan gambar-teks ini cocok? Penambangan negatif yang keras.
- ITG (pembuatan teks berbasis gambar): LM kausal langsung pada teks, dikondisikan pada kueri. Memaksa kueri untuk menyandikan konten yang dapat dihasilkan teks.

Hanya kereta Q-Mantan. Itu ViT dibekukan. Tidak ada LLM yang terlibat.

Tahap 2: pembelajaran generatif. Lampirkan LLM beku (OPT-2.7B atau Flan-T5-XL, dll.). Proyeksikan 32 keluaran kueri ke LLM embedding dim melalui lapisan linier kecil. Tambahkan mereka ke prompt teks. Latih hanya proyeksi linier dan Q-Former pada LM loss pada rangkaian prompt + gambar + keterangan yang digabungkan.

Setelah tahap 2, proyeksi Q-Former + menjadi adaptor visual lengkap. Kesimpulannya: gambar →ViT→ Q-Mantan → proyek linier → ditambahkan ke teks → LLM beku mengeluarkan keluaran.

### Parameter ekonomi

BLIP-2 dengan ViT-g/14 (1.1B, beku) + OPT-6.7B (6.7B, beku) + Q-Former (188M, dilatih) = total 8B, 188M dilatih. Q-Former sendiri adalah ~2,4% dari parameter stack penuh. Biaya pelatihan mencerminkan hal ini: jumlah hari untuk beberapa A100 vs minggu untuk end-to-end.

Kualitas: BLIP-2 cocok atau mengalahkan Flamingo-80B zero-shot VQA sekaligus 50x lebih kecil. Jembatan itu berfungsi.

### InstructBLIP dan Q-Former yang sadar akan instruksi

InstructBLIP (2023) memperluas Q-Former dengan input tambahan: teks instruksi itu sendiri. Pada cross-attention kali ini, kueri sekarang memiliki akses ke kedua gambar tersebut patch dan instruksinya. Kueri dapat mengkhususkan per-instruksi ("menghitung mobil", "menggambarkan suasana hati") daripada mempelajari satu ringkasan tetap. Tolok ukur perolehan tugas-tugas yang tertunda.

### MiniGPT-4 dan pendekatan khusus proyektor

MiniGPT-4 mempertahankan Q-Former tetapi hanya melatih proyeksi linier keluaran sambil membekukan yang lainnya. Murah, namun biaya adalah kualitas — pertanyaannya adalah milik BLIP-2, bukan milik kamu. Bagus untuk iterasi cepat, bukan arsitektur terbaik.

### Mengapa LLaVA menjadi lebih sederhana

LLaVA(2023, Pelajaran 12.05) menggantikan Q-Former dengan MLP 2 lapis biasa yang memproyeksikan setiap ViT patch token ke ruang LLM — 576 token per gambar untuk grid 24x24, semuanya dimasukkan ke LLM. Kompresi lebih buruk tetapi membiarkan LLM menanganinya secara mentah patch. Pada saat itu hal ini masih kontroversial; pada akhir tahun 2023 menjadi dominan karena data instruksi visual (LLaVA-Instruct-150k) membuktikan bahwa MLP dapat dilatih untuk mempertahankan sinyal yang cukup. Pengorbanannya: LLaVA Konteksnya terisi lebih cepat, namun berskala secara alami ke multi-gambar dan video.

Pada tahun 2026, pembagian bidang: Q-Former bertahan dalam hal anggaran token penting (video panjang, banyak gambar); Proyektor MLP mendominasi di mana kualitas mentah per token adalah prioritasnya.

### berpagar cross-attention: Flamingo, nenek moyang

Flamingo (Pelajaran 12.04) sudah ada sebelum BLIP-2 dan menggunakan hal yang sama cross-attention ide tetapi pada setiap lapisan LLM yang dibekukan, bukan sebagai jembatan tunggal. BLIP-2 menunjukkan kamu dapat mengompres ke lapisan masukan saja dan masih berfungsi. Gemini dan Idefics menggabungkan keduanya: token masukan yang disisipkan ditambah opsional gated cross-attention untuk dalam konteks few-shot.

### Keturunan tahun 2026

- Q-Mantan: BLIP-2, InstructBLIP, MiniGPT-4, dan sebagian besar model bahasa video karena alasan anggaran token.
- Pengambil sampel ulang penerima: varian Flamingo (Pelajaran 12.04); Keluarga Idefis, Elang, OmniMAE.
- Proyektor MLP: LLaVA, LLaVA-Berikutnya, LLaVA-OneVision, Kambrium-1.
- Kelompok perhatian: VILA, PaliGemma.

Keempatnya valid. Pertanyaan penentunya adalah apakah kamu dibatasi pada anggaran token atau kualitas per token.

```figure
modality-projection
```

## Gunakan

`code/main.py` membangun gaya stdlib Q-Mantan cross-attention:

1. Simulasikan 256 gambar patch token (dim 128).
2. Buat instance 32 kueri yang dapat dipelajari (redupkan 128).
3. Jalankan produk titik-skala cross-attention(Q dari pertanyaan, K/V dari patch).
4. Proyeksikan ke LLM-dim (512) melalui lapisan linier.
5. Keluarkan 32 token visual siap LLM.

Semua matematika dengan Python murni (loop bersarang di atas vektor). Mainan tapi bentuknya benar. Matriks perhatian-bobot dicetak sehingga kamu dapat melihat yang mana patch setiap kueri diambil.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-modality-bridge-picker.md`. Diberikan konfigurasi target VLM (vision encoder jumlah token, anggaran konteks LLM, batasan penerapan, target kualitas), ia merekomendasikan resampler Q-Former vs MLP vs Perceiver dengan justifikasi singkat dan perkiraan jumlah parameter untuk setiap jembatan.

## Latihan

1. Menerapkan cross-attention blok di PyTorch. Verifikasi bahwa dengan 32 kueri dan 256 kunci/nilai, matriks bobot perhatian adalah 32 x 256 dan setiap baris berjumlah 1 setelah softmax.

2. Dalam BLIP-2 tahap 1 Q-Mantan menjalankan tiga loss secara bersamaan: ITC, ITM, ITG. Tuliskan tanda tangan penerusan untuk masing-masing dalam kode semu. Yang mana yang membutuhkan teks encoder jalur untuk aktif?

3. Bandingkan jumlah parameter: Q-Former (12 lapisan, 768 tersembunyi) vs proyektor MLP 2 lapisan (1408 → 4096, dua lapisan). Pada skala LLM berapa biaya 188 juta Q-Former menghasilkan efisiensi pelatihan?

4. Baca Bagian 3.2 makalah BLIP-2 (arXiv:2301.12597) tentang bagaimana Q-Former diinisialisasi. Jelaskan mengapa inisialisasi dari BERT-base (bukan acak) mempercepat konvergensi.

5. Untuk video berdurasi 10 menit pada 1 FPS yang diambil sampelnya menjadi 60 bingkai, hitung biaya token per bingkai pada (Q-Former → 32 token/frame) vs (proyektor MLP → 576 token/frame). Mana yang cocok dengan jendela konteks LLM 128 ribu token?

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Q-Former | "Querying transformer" | Small transformer with 32 learnable query vectors that cross-attend to frozen ViT features |
| Learnable queries | "Soft prompt for vision" | A fixed set of parameters that serve as the query side of cross-attention; learned per model, shared across all inputs |
| Cross-attention | "Q from here, K/V from there" | Attention where query, key, and value come from different sources; how the queries pull from ViT patches |
| ITC | "Image-text contrastive" | CLIP-style loss applied to Q-Former pooled queries vs text CLS |
| ITM | "Image-text matching" | Binary classifier on hard-negative-mined pairs; forces the queries to discriminate fine-grained mismatches |
| ITG | "Image-grounded text generation" | Causal LM loss where text is generated conditioned on queries; forces queries to encode text-decodable content |
| Two-stage pretraining | "Representation then generative" | Stage 1 trains Q-Former alone (ITC/ITM/ITG); Stage 2 attaches frozen LLM and trains only the projection + Q-Former |
| Frozen backbone | "Do not finetune" | The vision encoder and LLM weights are fixed; only the bridge trains |
| Projection head | "Linear to LLM dim" | Final linear layer mapping Q-Former output to the LLM's embedding dimension |
| Perceiver resampler | "Flamingo's version" | Similar learnable-query cross-attention, used by Flamingo at every layer rather than as a single bridge |

## Bacaan Lebih Lanjut

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597)— kertas inti.
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086)— pendahulu dengan trio ITC/ITM/ITG.
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651)— "selaraskan sebelum sekering" — nenek moyang konseptual pelatihan tahap 1.
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500)— Q-Former yang sadar akan instruksi.
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592)— pendekatan khusus proyektor.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795)— arsitektur umum untuk kueri yang dapat dipelajari cross-attention.
