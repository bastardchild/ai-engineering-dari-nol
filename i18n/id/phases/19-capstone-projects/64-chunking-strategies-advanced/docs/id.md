# Strategi Chunking, Dibandingkan

> Chunking menentukan apa yang bisa muncul dari retrievermu. Jika batasnya salah, tak ada embedding model, tak ada reranker, tak ada LLM yang bisa memperbaiki kerusakan di hilir.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 lessons 04 (embeddings), 06 (RAG), 07 (advanced RAG); Phase 19 Track B foundations (lessons 20-29)
**Time:** ~90 minutes

## Tujuan Pembelajaran
- Implementasikan lima strategi chunking dari nol: fixed-window, sentence, recursive-split, semantic clustering, dan structural markdown headers.
- Ukur recall@k pada korpus fixture dengan answer span berlabel gold dan jelaskan mengapa satu strategi menang pada prosa sementara strategi berbeda menang pada dokumen teknis.
- Baca distribusi panjang chunk dan kenali mode kegagalan yang disuntikkan tiap strategi: orphan sentences, mid-symbol cuts, header-only chunks, semantic drift.
- Pilih default untuk korpus baru tanpa menjalankan benchmark dengan memeriksa tiga properti: tipe dokumen, panjang paragraf rata-rata, dan apakah formatnya membawa struktur eksplisit.

## Masalahnya

Setiap pipeline RAG dimulai dengan memotong dokumen sumber menjadi potongan yang cukup kecil agar muat di embedding model, dan cukup besar agar tiap potongan membawa ide yang berdiri sendiri. Pilihan di mana memotong bukan hyperparameter. Ia adalah batas atas apa yang bisa dikembalikan retriever.

Query yang menanyakan "what does the budget abort threshold look like" hanya bisa berhasil jika chunk yang memuat abort threshold dapat dijangkau. Jika fixed-window splitter memotong nilai threshold dari konteks sekitarnya, embedding berpindah ke cluster yang berbeda, skor BM25 turun, para reranker melihat noise, dan jawaban yang dihasilkan LLM menjadi salah. Paper 2024 "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs" mengukur ayunan absolut 35 persen pada recall retrieval semata-mata dari pilihan chunking. Lanjutannya pada 2025 tentang contextual chunk headers menyempitkan gap itu tetapi tidak menutupnya.

Pelajaran ini membangun lima strategi berdampingan, menjalankannya terhadap korpus fixture dengan answer span berlabel gold, dan membiarkanmu membaca angka recall-nya sendiri.

## Konsepnya

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

### Fixed-window

Baseline brute-force. Potong setiap N karakter. Secara opsional beri overlap sehingga kalimat yang terpotong di posisi N muncul utuh di dalam chunk yang dimulai di posisi N - overlap. Cepat, deterministik, buruk di batas. Gunakan sebagai kontrol, bukan default.

### Sentence

Pisahkan di batas kalimat memakai regex atau state machine sederhana. Masukkan satu atau lebih kalimat ke dalam satu chunk hingga target anggaran karakter tercapai. Berhenti memotong di tengah kata. Masih memotong di tengah paragraf dan tengah bagian. Default di banyak pipeline RAG awal dan pilihan yang wajar untuk prosa tanpa struktur lain.

### Recursive split

Strategi hierarki yang dipopulerkan library era 2023. Coba pisahkan di separator terkuat dulu (double newline, paragraf), mundur ke berikutnya (single newline), lalu ke kalimat, lalu ke karakter. Rekursi berhenti ketika chunk muat di anggaran. Kuat pada dokumen yang strukturnya tidak konsisten karena menyesuaikan per wilayah.

### Semantic clustering

Embed setiap kalimat. Cluster kalimat berurutan yang berbagi satu topik centroid. Potong setiap kali kemiripan berjalan terhadap centroid jatuh di bawah ambang. Batas-batasnya mencerminkan makna, bukan karakter. Lebih lambat dibangun dan bergantung pada embedding model, tetapi tahan terhadap dokumen yang mengganti topik di dalam paragraf.

### Structural markdown headers

Untuk dokumen yang membawa struktur eksplisit (markdown, reStructuredText, RFC-style numbered sections), potong di batas heading. Setiap chunk menjadi heading plus segala sesuatu di bawahnya hingga heading berikutnya pada level yang sama atau lebih tinggi. Chunk paling kecil per topik, tetapi hanya tersedia ketika korpusnya well-formed.

### Bagaimana recall@k mengukur pilihan batas

Query berlabel gold membawa offset karakter yang persis dari answer span di dalam dokumen sumber. Setelah chunking, kamu bertanya: apakah ada di antara top-k chunk yang dikembalikan retriever yang tumpang-tindih dengan gold span? Jika ya, recall@k untuk query itu 1. Jika tidak, nilainya 0. Rata-ratakan ke seluruh kueri. Jalankan evaluasi yang sama untuk tiap strategi, dan sebarannya menunjukkan kebijakan batas mana yang bertahan pada korpus yang kamu miliki.

```figure
ci-chunk-boundaries
```

## Bangun Sendiri

`code/main.py` mengimplementasikan:

- `fixed_window(text, size, overlap)` - baseline-nya.
- `sentence_chunks(text, target)` - packer kalimat sederhana.
- `recursive_split(text, separators, target)` - rekursi hierarki.
- `semantic_chunks(text, similarity_threshold)` - clustering berbasis centroid di atas mock embedding deterministik.
- `structural_markdown(text)` - splitter yang sadar header.
- `mock_embed(text, dim)` - embedding berbasis hash agar loop berjalan offline.
- `DenseIndex` - bentuk yang sama dipakai di pelajaran hybrid retrieval Track B Phase 19.
- `eval_recall(strategy, corpus, queries, k)` - loop perbandingannya.
- Sebuah `main()` yang menjalankan setiap strategi pada korpus fixture dan mencetak tabel recall@k.

Jalankan:

```bash
python3 code/main.py
```

Outputnya adalah tabel kecil dengan satu baris per strategi dan satu kolom per k. Sentence kalah pada fixture terstruktur. Structural-markdown menang pada fixture markdown. Recursive bertahan pada fixture campuran karena rekursinya menyesuaikan diri. Semantic clustering menang pada fixture prosa yang tidak punya petunjuk struktur berguna.

## Mode kegagalan yang tidak akan ditutupi tabel

**Orphan sentences.** Packing kalimat menghasilkan chunk yang melewatkan kalimat topik. Embeddingnya lalu menunjuk ke cluster yang salah.

**Mid-symbol cuts.** Fixed-window di dalam kode atau YAML akan memecah identifier menjadi dua. Kedua belah setengahnya ter-embed menjadi noise.

**Header-only chunks.** Structural markdown mengeluarkan chunk yang tidak berisi apa pun selain `## Title`. Saring yang demikian atau lampirkan paragraf pertama dari chunk berikutnya.

**Semantic drift.** Semantic clustering memotong terlalu sedikit ketika korpusnya seragam topiknya. Chunk 5000 karakter memadatkan banyak jawaban spesifik ke dalam satu embedding yang menyebar. Gabungkan semantic dengan batas karakter yang keras.

**Stale embeddings.** Semantic clustering memakai sebuah embedding model. Jika kamu mengganti model, kamu juga mengganti chunk-nya. Pin model chunk terpisah dari model retrieval, atau bangun ulang index-nya bersamaan.

## Memilih default tanpa menjalankan benchmark

Tiga properti menentukan chunker default untuk korpus baru.

| Property | Value | Default |
|----------|-------|---------|
| Document type | Prose with no structure | Recursive split, target 800 |
| Document type | Markdown / RFC / API docs | Structural markdown |
| Document type | Code | AST-aware (out of scope; see Phase 19 lesson 02) |
| Paragraph length | Long, single topic | Sentence, target 500 |
| Paragraph length | Short, mixed topics | Semantic, threshold 0.6 |

Ketika ragu, pilih recursive split. Ia adalah baseline strategi-tunggal terkuat.

## Gunakan

Pola produksi:

- Jalankan eval sebelum kamu merilis pipeline baru; jangan percaya strategi yang jadi default library-mu.
- Jalankan ulang eval setiap kali kamu mengganti embedding model atau campuran korpus; pemenangnya tergantung korpus.
- Simpan nama strategi di metadata tiap chunk agar kamu bisa menelusuri regresi kemudian.

## Rilis

Sistem RAG end-to-end Track F di pelajaran 69 memakai chunker yang dipilih di sini sebagai tahap pertamanya. Eval harness di pelajaran 68 membaca recall@k dari bentuk yang sama dengan yang dikembalikan `eval_recall` di pelajaran ini. Pilih strategi yang menang di korpusmu dan teruskan ke depan.

## Latihan

1. Tambahkan strategi keenam: token-window memakai `tiktoken` alih-alih hitungan karakter. Bandingkan dengan fixed-window pada fixture yang sama.
2. Suntikkan fraksi 30 persen blok kode ke dalam fixture prosa. Jalankan ulang tabel itu. Jelaskan mengapa setiap strategi selain structural markdown kehilangan recall.
3. Ganti embedding deterministik dengan milik provider asli proyekmu. Ukur delta recall semantic-clustering. Laporkan apakah sebaran antar strategi melebar atau menyempit.
4. Tambahkan field `summary` per chunk: deskripsi centroid satu kalimat. Jalankan ulang eval dengan summary ditambahkan ke badan chunk. Ukur kenaikan recall-nya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Recall@k | "Did we get the right chunk?" | Fraction of queries where any of the top-k chunks overlaps the gold answer span |
| Chunk overlap | "Sliding window" | Re-include the last N characters of the previous chunk in the next chunk |
| Structural splitter | "Header-aware chunks" | Cut at H1/H2/H3 boundaries; the heading text is part of the chunk |
| Semantic chunker | "Topic-aware chunks" | Embed sentences, cluster by centroid similarity, cut on drift |
| Centroid drift | "Topic shift" | Cosine similarity between the running mean and the next sentence drops past a threshold |

## Bacaan Lanjutan

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (arXiv 2406.15319)](https://arxiv.org/abs/2406.15319)
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- Phase 11 pelajaran 06 - dasar-dasar RAG
- Phase 11 pelajaran 07 - RAG lanjutan
- Phase 19 pelajaran 65 - hybrid retrieval yang mengurutkan chunk yang dihasilkan di sini
- Phase 19 pelajaran 68 - eval harness yang menilai pilihan strategi di produksi
