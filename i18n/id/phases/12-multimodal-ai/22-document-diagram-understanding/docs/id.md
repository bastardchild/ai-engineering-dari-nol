# Pemahaman Dokumen dan Diagram

> Dokumen bukan foto. PDF, paper ilmiah, faktur, atau formulir tulisan-tangan punya layout, tabel, diagram, catatan kaki, header, dan struktur semantik yang tidak bisa ditangkap oleh pemahaman gambar biasa. Tumpukan pra-VLM adalah pipeline: OCR Tesseract + LayoutLMv3 + heuristik ekstraksi tabel. Gelombang VLM menggantinya dengan model tanpa-OCR — Donut (2022), Nougat (2023), DocLLM (2023) — yang mengeluarkan markup terstruktur secara langsung. Menjelang 2026, frontier hanyalah "berikan gambar halaman ke Claude Opus 4.7 pada 2576px native," dan output markup terstruktur datang dengan gratis. Pelajaran ini membaca busur tiga-era dari document AI.

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Menjelaskan tiga era document AI: pipeline OCR, tanpa-OCR, VLM-native.
- Mendeskripsikan tiga aliran input LayoutLMv3: teks, layout (bbox), patch gambar, dengan masking terpadu.
- Membandingkan Donut (tanpa-OCR, gambar → markup), Nougat (paper ilmiah → LaTeX), DocLLM (generatif sadar-layout), PaliGemma 2 (VLM-native).
- Memilih model dokumen untuk tugas baru (faktur, paper ilmiah, formulir tulisan-tangan, struk Tiongkok).

## Masalahnya

"Pahami PDF ini" terdengar sederhana namun sulit. Informasinya berada di:

- Konten teks (90% dari sinyal).
- Layout (header, catatan kaki, sidebar, format dua-kolom).
- Tabel (baris, kolom, sel yang digabung).
- Gambar dan diagram.
- Anotasi tulisan-tangan.
- Font dan tipografi (judul vs tubuh teks).

OCR mentah membuang teks dan kehilangan sisanya. Sistem yang peduli pada faktur perlu tahu bahwa "Total: $1,245" berasal dari kanan-bawah, bukan dari catatan kaki.

## Konsepnya

### Era 1 — pipeline OCR (pra-2021)

Tumpukan klasik:

1. PDF → gambar per halaman.
2. Tesseract (atau OCR komersial) mengekstrak teks dengan bounding box per kata.
3. Analyzer layout mengidentifikasi blok (header, tabel, paragraf).
4. Pengenal struktur tabel mengurai tabel.
5. Aturan domain + regex mengekstrak field.

Bekerja untuk teks cetak bersih. Gagal pada tulisan-tangan, pindaian miring, tabel kompleks, skrip non-Inggris. Setiap mode kegagalan membutuhkan jalur pengecualian khusus.

### TrOCR (2021)

TrOCR (Li et al., arXiv:2109.10282) mengganti CNN-CTC klasik Tesseract dengan transformer encoder-decoder yang dilatih pada gambar teks sintetis + nyata. Kemenangan bersih pada teks tulisan-tangan dan multilingual. Masih sebuah pipeline (detektor lalu TrOCR lalu layout), tetapi langkah OCR membaik secara dramatis.

### Era 2 — tanpa-OCR (2022-2023)

Model tanpa-OCR pertama berkata: lewati deteksi sama sekali, petakan piksel gambar ke output terstruktur secara langsung.

Donut (Kim et al., arXiv:2111.15664):
- Transformer encoder-decoder, encodernya Swin-B.
- Outputnya JSON untuk pemahaman formulir, markdown untuk ringkasan, atau skema spesifik-tugas apa pun.
- Tanpa OCR, tanpa layout, tanpa deteksi.

Nougat (Blecher et al., arXiv:2308.13418):
- Dilatih secara khusus pada paper ilmiah.
- Outputnya LaTeX / markdown.
- Menangani persamaan matematika, layout multi-kolom, gambar.
- Model yang dipanggil setiap parser arXiv.

Ini adalah spesialis, bukan generalis. Donut pada paper ilmiah gagal; Nougat pada faktur gagal.

### LayoutLMv3 (2022)

Jalur yang berbeda. LayoutLMv3 (Huang et al., arXiv:2204.08387) mempertahankan OCR tetapi menambahkan pemahaman layout:

- Tiga aliran input: token teks OCR, bounding box 2D per token, patch gambar.
- Objektif training masked di ketiga modalitas (teks masked, patch masked, layout masked).
- Downstream: klasifikasi, ekstraksi entitas, QA tabel.

LayoutLMv3 adalah puncak dari pemahaman dokumen berbasis OCR. Kuat pada formulir dan faktur. Membutuhkan OCR di hulu. Akurasi pra-VLM terbaik pada benchmark dokumen terstandardisasi.

### DocLLM (2023)

DocLLM (Wang et al., arXiv:2401.00908) adalah saudara generatif dari LayoutLM. Menghasilkan jawaban bebas-format yang dibingkai oleh token layout. Lebih baik untuk QA pada dokumen; masih bergantung pada input OCR.

### Era 3 — VLM-native (2024+)

2024 VLM menjadi cukup baik untuk mengganti pipeline sepenuhnya. Berikan gambar halaman penuh pada resolusi tinggi ke VLM, ajukan pertanyaannya, dapatkan jawabannya.

- LLaVA-NeXT 336-tile AnyRes bekerja untuk dokumen kecil.
- Qwen2.5-VL resolusi-dinamis menangani 2048+ piksel secara native.
- Claude Opus 4.7 mendukung dokumen 2576px.
- PaliGemma 2 (April 2025) dilatih secara khusus untuk dokumen + tulisan-tangan.

Gap antara VLM-native dan pipeline OCR menutup dengan cepat. Menjelang 2026, VLM-native menang pada:

- Teks adegan (tulisan-tangan + cetak, skrip campuran).
- Tabel kompleks dengan sel yang digabung.
- Persamaan matematika yang tertanam dalam teks.
- Gambar dengan anotasi teks.

Pipeline OCR masih menang pada:

- Beban kerja pindaian-murni dalam skala besar di mana latensi per-halaman berarti.
- Keandalan pipeline (kegagalan deterministik vs halusinasi VLM).
- Lingkungan teregulasi yang menuntut output OCR yang dapat diaudit.

### Frontier Claude 4.7 / GPT-5

Pada input native 2576 piksel, VLM frontier melakukan pemahaman dokumen pada akurasi yang mendekati manusia. Angka benchmark dari awal 2026:

- DocVQA: Claude 4.7 ~95,1, PaliGemma 2 ~88,4, Nougat ~77,3, LayoutLMv3 dengan pipeline ~83.
- ChartQA: Claude 4.7 ~92,2, GPT-4V ~78.
- VisualMRC: Claude 4.7 ~94.

Gap model tertutup sebagian besar terletak pada resolusi dan skala basis-LLM. Model open berukuran 7B tertinggal beberapa poin tetapi sedang menyusul.

### Persamaan matematika dan output LaTeX

Paper ilmiah membutuhkan output LaTeX yang tepat untuk persamaan. Nougat dilatih untuk ini. VLM yang dilatih dengan target LaTeX (Qwen2.5-VL-Math, turunan Nougat) menghasilkan LaTeX yang usable. Tanpa pelatihan LaTeX eksplisit, VLM menghasilkan transkripsi yang terbaca tetapi tidak presisi.

Untuk pipeline paper ilmiah di 2026: rangkai Nougat pada PDF-nya, lalu sebuah VLM pada halaman-halaman yang sulit.

### Tulisan-tangan

Masih subtugas tersulit. Campuran cetak + tulisan-tangan (catatan dokter, formulir terisi) adalah wilayah di mana pipeline OCR masih mengalahkan VLM dari segi biaya. VLM tulisan-tangan-murni sedang membaik (Claude 4.7, PaliGemma 2).

### Resep 2026

Untuk proyek document-AI baru:

- Faktur cetak-murni berskala besar: LayoutLMv3 + aturan, hemat-biaya.
- Dokumen campuran (ilmiah + tulisan-tangan + formulir): VLM-native (PaliGemma 2 atau Qwen2.5-VL).
- Ingestion arXiv penuh: Nougat untuk matematika, VLM untuk gambar.
- Regulasi: pipeline OCR + validator VLM untuk cross-check.

```figure
mm-doc-layout
```

## Gunakan

`code/main.py`:

- Tokenizer sadar-layout mainan: diberi pasangan (teks, bbox), menghasilkan input gaya LayoutLMv3.
- Generator skema tugas gaya Donut: template JSON untuk formulir.
- Perbandingan anggaran token per halaman di antara pipeline OCR, Donut, Nougat, dan VLM-native.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-document-ai-stack-picker.md`. Diberi proyek document-AI (domain, skala, kualitas, regulasi), memilih di antara pipeline OCR, spesialis tanpa-OCR, dan VLM-native.

## Latihan

1. Proyekmu memproses 10M faktur per hari. Tumpukan mana yang meminimalkan biaya-per-halaman tanpa kehilangan akurasi?

2. Mengapa LayoutLMv3 mengungguli VLM CLIP-murni pada QA formulir tetapi tertinggal pada teks adegan? Apa yang dikorbankan oleh aliran bbox?

3. Nougat menghasilkan LaTeX. Usulkan sebuah kasus uji di mana output VLM-native mengalahkan Nougat pada kesetiaan LaTeX, dan sebuah kasus di mana Nougat menang.

4. Baca paper PaliGemma 2 (Google, 2024). Penambahan data training kunci apa yang mengangkat akurasi dokumen dibanding PaliGemma 1?

5. Rancang hibrida yang aman-regulasi: pipeline OCR sebagai primer, VLM sebagai cross-check sekunder. Bagaimana kamu menyelesaikan perbedaan pendapat?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OCR pipeline | "Tesseract-style" | Stage-wise stack: detect -> OCR -> layout -> rules; deterministic, fragile |
| OCR-free | "Donut-style" | Image-to-output transformer that skips explicit OCR; single model |
| Layout-aware | "LayoutLM" | Input includes per-token bbox coordinates; unified masking across modalities |
| VLM-native | "Frontier VLM" | Feed page image directly to Claude/GPT/Qwen VLM at high resolution; no pipeline |
| DocVQA | "Doc benchmark" | Document VQA standard; most-cited score |
| Markup output | "LaTeX / MD" | Structured output format instead of free-form text; enables downstream automation |

## Bacaan Lanjutan

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
