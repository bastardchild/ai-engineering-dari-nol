# Model Embedding — Deep Dive 2026

> Word2Vec memberimu vektor per kata. Model embedding modern memberimu vektor per passage, lintas-lingual, dengan pandangan sparse, dense, dan multi-vektor, berukuran pas dengan indeksmu. Salah pilih dan RAG-mu mengambil hal yang salah.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 minutes

## Masalahnya

Sistem RAG-mu mengambil passage salah 40% waktu. Biang keroknya jarang vector database atau prompt. Melainkan model embedding.

Memilih embedding di 2026 berarti memilih lintas lima sumbu:

1. **Dense vs sparse vs multi-vektor.** Satu vektor per passage, atau satu per token, atau bag-of-words berbobot sparse.
2. **Cakupan bahasa.** Model monolingual Inggris masih menang pada tugas khusus-Inggris. Model multilingual menang ketika korpus campuran.
3. **Panjang konteks.** 512 token vs 8.192 vs 32.768 — dan kapasitas efektif riil sering 60-70% dari maks yang diiklankan.
4. **Budget dimensi.** 3.072 float pada presisi penuh = 12 KB per vektor. Pada 100M vektor, penyimpanannya $1,300/bulan. Trunkasi Matryoshka memangkasnya 4×.
5. **Terbuka vs hosted.** Open-weight berarti kamu mengendalikan stack dan data. Hosted berarti kamu menukar kendali dengan selalu-terbaru.

Pelajaran ini menyebutkan tradeoff supaya kamu memilih berdasarkan bukti, bukan berdasarkan apa yang populer kuartal lalu.

## Konsepnya

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**Embedding dense.** Satu vektor per passage (biasanya 384-3.072 dimensi). Kemiripan kosinus memeringkat passage berdasarkan kedekatan semantik. `text-embedding-3-large` OpenAI, mode dense BGE-M3, Voyage-3. Pilihan default.

**Embedding sparse.** Ala SPLADE. Transformer memprediksi bobot untuk setiap token vocab, lalu menolkan kebanyakannya. Hasilnya adalah vektor sparse berukuran |vocab|. Menangkap pencocokan leksikal (seperti BM25) tetapi dengan bobot term terpelajari. Kuat pada query berat-keyword.

**Multi-vektor (late interaction).** ColBERTv2, Jina-ColBERT. Satu vektor per token. Skoring dengan MaxSim: untuk tiap token query, temukan token dokumen paling mirip, jumlahkan skornya. Lebih mahal disimpan dan diskorkan, tetapi menang pada query panjang dan korpus spesifik-domain.

**BGE-M3: ketiganya sekaligus.** Satu model mengeluarkan representasi dense, sparse, dan multi-vektor secara simultan. Masing-masing bisa di-query independen; skor difusi via jumlah berbobot. Default 2026 ketika kamu ingin fleksibilitas dari satu checkpoint.

**Matryoshka Representation Learning.** Dilatih sehingga N dimensi pertama vektor membentuk embedding mandiri yang berguna. Pangkas vektor 1.536-dim ke 256 dim dan bayar ~1% akurasi untuk penghematan penyimpanan 6×. Didukung text-3 OpenAI, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+.

### Leaderboard MTEB bercerita sebagian

Massive Text Embedding Benchmark — 56 tugas lintas 8 tipe tugas saat peluncuran (2022), diperluas ke 100+ tugas di MTEB v2. Di awal 2026, Gemini Embedding 2 memuncaki retrieval (67.71 MTEB-R). Cohere embed-v4 memimpin umum (65.2 MTEB). BGE-M3 memimpin multilingual open-weight (63.0). Leaderboard perlu tetapi tidak cukup — selalu benchmark pada domainmu.

### Pola tiga tingkat

| Use case | Pattern |
|----------|---------|
| Fast first-pass | Dense bi-encoder (BGE-M3, text-3-small) |
| Recall boost | Sparse (SPLADE, BGE-M3 sparse) + RRF fuse |
| Precision on top-50 | Multi-vector (ColBERTv2) or cross-encoder reranker |

Kebanyakan stack produksi memakai ketiganya.

```figure
gx-matryoshka
```

## Bangun Sendiri

### Langkah 1: baseline — embedding dense dengan Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` membuat dot product sama dengan kemiripan kosinus. Selalu setel.

### Langkah 2: trunkasi Matryoshka

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

Normalisasi ulang setelah trunkasi. Nomic v1.5, text-3 OpenAI, dan Voyage-4 dilatih sehingga ini lossless untuk beberapa level pertama. Model non-Matryoshka (Sentence-BERT orisinal) terdegradasi tajam ketika dipangkas.

### Langkah 3: multi-fungsi BGE-M3

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

Tiga indeks, satu panggilan inference. Fusi skor:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

Tune bobotnya pada domainmu.

### Langkah 4: eval MTEB pada tugas kustom

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

Jalankan model kandidatmu pada subset *representatif*. Jangan percaya peringkat leaderboard saja — domainmu penting.

### Langkah 5: kosinus buatan tangan dari nol

Lihat `code/main.py`. Embedding Averaged Hashing Trick (stdlib-only). Tidak kompetitif dengan embedding transformer, tetapi menunjukkan bentuknya: tokenisasi → vektor → normalisasi → dot product.

## Jebakan

- **Model sama untuk query dan dok.** Sebagian model (Voyage, Jina-ColBERT) memakai encoding asimetris — query dan dokumen lewat jalur berbeda. Selalu cek model card.
- **Prefiks hilang.** Model `bge-*` butuh `"Represent this sentence for searching relevant passages: "` di depan query. Gap recall 3-5 poin jika kamu lupa.
- **Trunkasi Matryoshka berlebih.** 1.536 → 256 biasanya aman. 1.536 → 64 tidak. Validasi pada eval set-mu.
- **Trunkasi konteks.** Kebanyakan model diam-diam mem-truncate input melebihi panjang maksnya. Dok panjang butuh chunking (lihat pelajaran 23).
- **Mengabaikan ekor latensi.** Skor MTEB menyembunyikan latensi p99. Model 600M mungkin mengalahkan model 335M sebesar 2 poin tetapi biayanya 3× per query.

## Gunakan

Stack 2026:

| Situation | Pick |
|-----------|------|
| English-only, fast, API | `text-embedding-3-large` or `voyage-3-large` |
| Open-weight, English | `BAAI/bge-large-en-v1.5` |
| Open-weight, multilingual | `BAAI/bge-m3` or `Qwen3-Embedding-8B` |
| Long context (32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| CPU-only deployment | Nomic Embed v2 (137M params, MoE) |
| Storage-constrained | Matryoshka-truncated + int8 quantization |
| Keyword-heavy queries | Add SPLADE sparse, RRF-fuse with dense |

Pola 2026: mulai dengan BGE-M3 atau text-3-large, evaluasi pada domainmu dengan MTEB, tukar jika model spesifik-domain menang lebih dari 3 poin.

## Rilis

Simpan sebagai `outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## Latihan

1. **Mudah.** Encode 100 kalimat dengan `bge-small-en-v1.5` pada dim penuh (384), lalu pada Matryoshka 128. Ukur penurunan MRR pada 10 query.
2. **Sedang.** Bandingkan dense, sparse, dan colbert BGE-M3 pada 500 passage dari domainmu. Mana yang menang pada recall@10? Apakah fusi RRF mengalahkan mode tunggal terbaik?
3. **Sulit.** Jalankan MTEB pada tiga model kandidat lintas 2 tugas domain teratasmu. Laporkan skor MTEB, latensi p99 pada batch 100-query, dan $/1M query. Pilih yang Pareto-optimal.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Dense embedding | The vector | One fixed-size vector per text. Cosine similarity for ranking. |
| Sparse embedding | Learned BM25 | One weight per vocab token; mostly zeros; trained end-to-end. |
| Multi-vector | ColBERT-style | One vector per token; MaxSim scoring; bigger index, better recall. |
| Matryoshka | Russian doll trick | First N dims are a valid smaller embedding on their own. |
| MTEB | The benchmark | Massive Text Embedding Benchmark — 56 tasks at launch, 100+ in v2. |
| BEIR | The retrieval benchmark | 18 zero-shot retrieval tasks; often cited for cross-domain robustness. |
| Asymmetric encoding | Query ≠ doc path | Model uses different projections for queries and documents. |

## Bacaan Lanjutan

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — the bi-encoder paper.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — the leaderboard paper.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — the unified three-mode model.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — the dimension-ladder training objective.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — late interaction in production.
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — live rankings.
