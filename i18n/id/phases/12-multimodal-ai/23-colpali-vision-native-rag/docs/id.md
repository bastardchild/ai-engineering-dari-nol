# ColPali dan Document RAG Vision-Native

> RAG tradisional mem-parse PDF menjadi teks, memotongnya menjadi chunk, meng-embed chunk, menyimpan vektor. Setiap langkah kehilangan sinyal: OCR menjatuhkan data grafik, chunking memutus baris tabel, embedding teks mengabaikan gambar. ColPali (Faysse et al., Juli 2024) mengajukan pertanyaan yang lebih sederhana: mengapa harus mengekstrak teks sama sekali? Embed gambar halaman secara langsung via PaliGemma, gunakan late interaction gaya ColBERT untuk retrieval, dan pertahankan semua sinyal layout, gambar, font, dan formatasi yang dibawa dokumen. Benchmark yang dipublikasikan: akurasi end-to-end 20-40% lebih baik daripada text-RAG pada dokumen yang kaya-visual. ColQwen2, ColSmol, dan VisRAG memperluas pola ini. Pelajaran ini membaca tesis RAG vision-native dan membangun sebuah indexer kecil mirip ColPali.

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Menjelaskan perbedaan antara retrieval bi-encoder (satu vektor per dokumen) dan retrieval late-interaction (banyak vektor per dokumen).
- Mendeskripsikan operasi MaxSim milik ColBERT dan bagaimana ColPali menggeneralisasikannya dari token teks ke patch gambar.
- Membangun indexer kecil mirip ColPali: halaman → embedding patch → MaxSim di atas embedding istilah-query → halaman top-k.
- Membandingkan ColPali + generator Qwen2.5-VL vs text-RAG + GPT-4 pada use case faktur / laporan keuangan.

## Masalahnya

Text-RAG pada PDF membuang sebagian besar dokumen. Pertumbuhan pendapatan Q3 sebuah laporan keuangan biasanya ada di grafik; temuan sebuah laporan medis ada di gambar beranotasi; blok tanda tangan sebuah kontrak hukum adalah fakta layout, bukan fakta teks.

Pipeline text-RAG:

1. PDF → teks via OCR / pdftotext.
2. Teks → chunk 300-500 token.
3. Chunk → embedding bi-encoder (satu vektor).
4. Query pengguna → embedding → cosine similarity → chunk top-k.
5. Chunk + query → LLM.

Lima langkah yang lossy. Grafik tidak tertangkap. Tabel terputus di antara chunk. Layout multi-kolom diratakan. Anotasi gambar hilang.

Perbaikan ColPali: lewati OCR, embed gambar halaman secara langsung. Gunakan late interaction gaya ColBERT untuk retrieval agar model bisa attend ke patch halus pada saat query.

## Konsepnya

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) adalah metode retrieval teks. Alih-alih satu vektor per dokumen, ia menghasilkan satu vektor per token. Pada saat query:

- Token query mendapat embedding-nya sendiri (N_q vektor).
- Token dokumen mendapat embedding (N_d vektor, biasanya di-cache).
- Skor = jumlah di atas token query dari maksimum di atas token dokumen dari cosine similarity: Σ_i max_j cos(q_i, d_j).

Ini adalah operasi MaxSim. Setiap token query "memilih" token dokumen yang paling cocok. Skor akhir adalah jumlahnya.

Pro: recall kuat, menangani semantik level-istilah. Kontra: N_d vektor per dokumen, penyimpanan mahal.

### ColPali

ColPali (Faysse et al., arXiv:2407.01449) menerapkan pola ColBERT ke gambar.

- Setiap halaman di-encode oleh PaliGemma (ViT + bahasa) menjadi embedding patch: N_p vektor per halaman.
- Setiap query pengguna (teks) di-encode menjadi embedding token-query: N_q vektor.
- Skor = Σ_i max_j cos(q_i, p_j), yaitu MaxSim di atas token-teks-query dan patch-gambar-halaman.
- Retrieve halaman top-k berdasarkan skor total.

Pada saat ingestion dokumen: embed setiap halaman dengan PaliGemma, simpan semua embedding patch. Pada saat query: embed token query, hitung MaxSim terhadap semua embedding halaman yang tersimpan, kembalikan halaman top-k.

Pro: end-to-end mengalahkan text-RAG sebesar 20-40% pada dokumen yang kaya-visual. Setiap vektor-patch menangkap layout dan konten lokal.

Kontra: N_p patch × float 4-byte × vektor dimensi-D per halaman = penyimpanan tumbuh cepat. Dimitigasi oleh kuantisasi PQ / OPQ.

### ColQwen2 dan ColSmol

ColQwen2 (illuin-tech, 2024-2025) menukar PaliGemma dengan Qwen2-VL. Encoder dasar lebih baik, retrieval lebih baik.

ColSmol adalah varian skala-lebih-kecil untuk penggunaan lokal / edge. Retriever ColSmol pada ~1B parameter berjalan pada GPU konsumen.

### VisRAG

VisRAG (Yu et al., arXiv:2410.10594) adalah varian berbeda: alih-alih MaxSim pada patch, melakukan pooling setiap halaman menjadi satu vektor dengan VLM lalu retrieve dengan bi-encoder. Indexing lebih cepat + penyimpanan lebih kecil, recall lebih lemah.

Trade-off kualitas-vs-biaya: ColPali untuk kualitas, VisRAG untuk skala.

### M3DocRAG

M3DocRAG (Cho et al., arXiv:2411.04952) memperluas retrieval multimodal ke penalaran multi-halaman multi-dokumen. Me-retrieve halaman di lintas dokumen, menyusun konteks multi-halaman untuk VLM.

### ViDoRe — benchmarknya

Benchmark pendamping ColPali. Visual Document Retrieval Evaluation. Tugasnya mencakup laporan keuangan, paper ilmiah, dokumen administratif, rekam medis, manual. Metrik: nDCG@5.

ColPali-v1 mencetak ~80% nDCG@5 di ViDoRe; text-RAG pada dokumen yang sama mencetak ~50-60%.

### Pipeline RAG end-to-end

Untuk RAG vision-native:

1. Ingest: PDF → gambar halaman → encoding PaliGemma → simpan semua embedding patch.
2. Query: teks pengguna → embedding token-query → MaxSim terhadap semua halaman terindeks → halaman top-k.
3. Generate: gambar halaman top-k + query → VLM (Qwen2.5-VL atau Claude) → jawaban.

Tidak ada OCR di mana pun. Gambar, grafik, font, layout semuanya mengalir ke jawaban.

### Matematika penyimpanan

Laporan keuangan 50 halaman dengan 729 patch per halaman dan embedding dimensi-128:

- ColPali: 50 * 729 * 128 * 4 byte = ~18 MB mentah, ~4 MB setelah PQ.
- Text-RAG: 50 chunk * dimensi-768 * 4 byte = ~150 kB.

ColPali butuh ~30x penyimpanan lebih per dokumen. Pada skala besar, OPQ / PQ menurunkannya ke ~5-10x, biasanya masih dapat ditoleransi.

### Saat text-RAG masih menang

- Dokumen teks-murni tanpa sinyal layout (artikel wiki, log chat). Text-RAG lebih sederhana dan penyimpanannya lebih murah.
- Arsip multi-juta-halaman di mana penyimpanan mendominasi biaya.
- Persyaratan regulasi ketat yang menuntut teks OCR yang dapat diekstrak di samping retrieval.

Untuk sisanya di 2026 — laporan keuangan, paper ilmiah, kontrak hukum, rekam medis, dokumentasi UX — RAG vision-native menang.

```figure
mm-maxsim
```

## Gunakan

`code/main.py`:

- Encoder patch mainan: memetakan sebuah "halaman" (grid kecil dari vektor fitur) ke array embedding patch.
- Skorer MaxSim: menghitung skor gaya ColBERT di antara himpunan embedding token-query dan himpunan patch halaman.
- Mengindeks 5 halaman mainan, menjalankan 3 query, mengembalikan top-k dengan skor.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-vision-rag-designer.md`. Diberi proyek document-RAG, memilih ColPali / ColQwen2 / VisRAG / text-RAG dan mengukur penyimpanannya.

## Latihan

1. Laporan tahunan 200 halaman pada 729 patch per halaman, emb dimensi-128, float 4-byte. Hitung penyimpanan mentah dan penyimpanan terkompresi PQ (8x).

2. MaxSim adalah Σ_i max_j cos(q_i, p_j). Apa yang ditangkap oleh jumlah ini yang tidak ditangkap oleh kemiripan rerata sederhana?

3. ColPali mengindeks halaman sebagai himpunan patch. Apa yang berubah jika kita malah mengindeks pada level kata (seperti yang dilakukan ColBERT)? Trade-off-nya?

4. Rancang pipeline end-to-end untuk korpus 1 juta halaman dengan anggaran latensi 500ms per query. Pilih ColQwen2 / VisRAG dan berikan justifikasi.

5. Baca M3DocRAG (arXiv:2411.04952). Deskripsikan pola attention multi-halaman dan bagaimana ia berbeda dari retrieval ColPali satu-halaman.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Late interaction | "ColBERT-style" | Retrieval using per-token or per-patch embeddings + MaxSim, not a single doc vector |
| MaxSim | "Max-over-patches" | For each query token, pick the highest-similarity document token; sum across query |
| Bi-encoder | "Single-vector" | One vector per document; faster but loses granularity |
| Multi-vector | "Many-vectors-per-doc" | Store N_p vectors per document / page; storage cost grows but recall improves |
| Patch embedding | "Page feature" | One vector per image patch from a VLM encoder, cached per page |
| ViDoRe | "Vision doc bench" | ColPali's benchmark suite for visual document retrieval |
| PQ quantization | "Product quantization" | Compression that maintains vector similarity while shrinking storage ~8x |

## Bacaan Lanjutan

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
