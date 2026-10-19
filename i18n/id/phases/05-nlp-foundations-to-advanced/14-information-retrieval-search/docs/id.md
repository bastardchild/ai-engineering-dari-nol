# Information Retrieval dan Pencarian

> BM25 presisi tetapi rapuh. Dense menebar jaring lebar tetapi melewatkan keyword. Hibrida adalah default 2026. Selebihnya tuning.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## Masalahnya

Pengguna mengetik "what happens if someone lies to get money" dan mengharapkan menemukan statuta yang benar-benar mengaturnya: "Section 420 IPC." Pencarian keyword melewatkannya sepenuhnya (tanpa vocabulary bersama). Pencarian semantik melewatkannya jika embedding tidak dilatih pada teks legal. Pencarian nyata harus menangani keduanya.

IR adalah pipeline di bawah setiap sistem RAG, setiap search bar, setiap lookup fuzzy situs docs. Arsitektur 2026 yang berhasil di produksi bukanlah satu metode. Ia adalah rantai metode komplementer, masing-masing menangkap kegagalan metode sebelumnya.

Pelajaran ini membangun tiap bagiannya dan menyebutkan kegagalan mana yang ditangkap masing-masing.

## Konsepnya

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

Empat lapisan. Pilih yang kamu butuhkan.

1. **Retrieval sparse (BM25).** Cepat, presisi pada kecocokan eksak, buruk pada semantik. Berjalan atas inverted index. Sub-10ms per query pada jutaan dokumen. Membuatmu benar pada referensi statuta, kode produk, pesan error, named entity.
2. **Retrieval dense.** Encode query dan dokumen menjadi vektor. Pencarian nearest neighbor. Menangkap parafrasa dan kemiripan semantik. Melewatkan kecocokan keyword eksak yang berbeda satu karakter. 50-200ms per query dengan FAISS atau vector DB.
3. **Fusi.** Gabungkan daftar peringkat dari sparse dan dense. Reciprocal Rank Fusion (RRF) adalah default mudah karena mengabaikan skor mentah (yang hidup dalam skala berbeda) dan hanya memakai posisi peringkat. Fusi berbobot adalah opsi ketika kamu tahu satu sinyal mendominasi untuk domainmu.
4. **Rerank cross-encoder.** Ambil 30 teratas dari fusi. Jalankan cross-encoder (query + dokumen bersama, skorkan tiap pasangan). Simpan 5 teratas. Cross-encoder lebih lambat per pasangan daripada bi-encoder tetapi jauh lebih akurat. Kamu mengamortisasinya dengan hanya menjalankannya pada 30 teratas.

Retrieval tiga-arah (BM25 + dense + learned-sparse seperti SPLADE) mengungguli dua-arah dalam benchmark 2026 tetapi butuh infrastruktur untuk indeks learned-sparse. Bagi kebanyakan tim, dua-arah plus rerank cross-encoder adalah sweet spot.

```figure
gx-hybrid-retrieval
```

## Bangun Sendiri

### Langkah 1: BM25 dari nol

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

Dua parameter yang patut diketahui. `k1=1.5` mengontrol saturasi term-frequency; lebih tinggi berarti bobot lebih pada pengulangan term. `b=0.75` mengontrol normalisasi panjang; 0 mengabaikan panjang dokumen, 1 menormalisasi penuh. Default-nya adalah rekomendasi Robertson dari paper aslinya dan jarang perlu di-tune.

### Langkah 2: retrieval dense dengan bi-encoder

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

Normalisasi-L2 embedding supaya dot product sama dengan kosinus. `all-MiniLM-L6-v2` itu 384-dim, cepat, dan cukup kuat untuk kebanyakan retrieval Inggris. Untuk kerja multilingual, pakai `paraphrase-multilingual-MiniLM-L12-v2`. Untuk akurasi teratas, `bge-large-en-v1.5` atau `e5-large-v2`.

### Langkah 3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

Konstanta `k=60` berasal dari paper RRF aslinya. `k` lebih tinggi meratakan kontribusi perbedaan peringkat; `k` lebih rendah membuat peringkat teratas mendominasi. 60 adalah default yang dipublikasikan dan jarang perlu di-tune.

### Langkah 4: pencarian hibrida + rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

Tiga tahap tersusun. BM25 menemukan kecocokan leksikal. Dense menemukan kecocokan semantik. RRF menggabungkan kedua peringkat tanpa perlu kalibrasi skor. Cross-encoder memberi skor ulang 30 teratas memakai pasangan query-dokumen bersama, yang menangkap relevansi rinci yang terlewat bi-encoder. Simpan 5 teratas.

### Langkah 5: evaluasi

| Metric | Meaning |
|--------|---------|
| Recall@k | Of queries where the correct document exists, how often is it in the top-k? |
| MRR (Mean Reciprocal Rank) | Average of 1/rank of first relevant document. |
| nDCG@k | Accounts for relevance gradations, not just binary relevant/not. |

Untuk RAG secara spesifik, **Recall@k** retriever adalah angka terpenting. Readermu tidak bisa menjawab jika passage yang benar tidak ada dalam himpunan yang diambil.

Tips debugging: untuk query yang gagal, bedakan peringkat sparse dan dense. Jika satu menemukan dokumen benar dan satunya tidak, kamu punya vocabulary mismatch (perbaiki: tambahkan separuh yang hilang) atau ambiguitas semantik (perbaiki: embedding lebih baik atau reranker).

## Gunakan

Stack 2026:

| Scale | Stack |
|-------|-------|
| 1k-100k docs | In-memory BM25 + `all-MiniLM-L6-v2` embeddings + RRF. No separate DB. |
| 100k-10M docs | FAISS or pgvector for dense + Elasticsearch / OpenSearch for BM25. Run in parallel. |
| 10M+ docs | Qdrant / Weaviate / Vespa / Milvus with hybrid support. Cross-encoder rerank on top-30. |
| Best-quality frontier | Three-way (BM25 + dense + SPLADE) + ColBERT late-interaction reranking |

Apa pun yang kamu pilih, anggarkan untuk evaluasi. Benchmark recall retrieval sebelum benchmarking akurasi RAG end-to-end. Reader tidak bisa memperbaiki apa yang terlewat retriever.

### Pelajaran hasil kerja keras dari RAG produksi 2026

- **80% kegagalan RAG berakar pada ingestion dan chunking, bukan model.** Tim menghabiskan berminggu-minggu menukar LLM dan men-tune prompt sementara retrieval diam-diam mengembalikan konteks salah setiap tiga query. Perbaiki chunking dulu.
- **Strategi chunking lebih penting daripada ukuran chunk.** Pecahan ukuran-tetap merusak tabel, kode, dan header bersarang. Sadar-kalimat adalah default; chunking semantik atau berbasis LLM terbayar untuk docs teknis dan manual produk.
- **Pola parent-doc.** Ambil "child" chunk kecil untuk presisi. Ketika banyak child dari induk yang sama muncul, tukar ke blok induk untuk menjaga konteks. Ini konsisten mengangkat kualitas jawaban tanpa retraining.
- **k_rerank=3 biasanya optimal.** Setiap chunk tambahan setelah itu menambah biaya token dan latensi generasi tanpa mengangkat kualitas jawaban. Jika k=8 masih lebih baik daripada k=3 bagimu, reranker-nya underperform.
- **HyDE / ekspansi query.** Hasilkan jawaban hipotetis dari query, embed itu, ambil. Menjembatani gap frasa antara pertanyaan pendek dan dokumen panjang. Kenaikan presisi gratis tanpa training.
- **Budget konteks di bawah 8K token.** Hit konsisten pada batas itu berarti ambang reranker terlalu longgar.
- **Versikan semuanya.** Prompt, aturan chunking, model embedding, reranker. Drift apa pun diam-diam merusak kualitas jawaban. Gate CI pada faithfulness, presisi konteks, dan tingkat pertanyaan-tak-terjawab memblokir regresi sebelum pengguna melihatnya.
- **Retrieval tiga-arah (BM25 + dense + learned-sparse seperti SPLADE) mengungguli dua-arah** pada benchmark 2026, terutama untuk query yang mencampur proper noun dengan semantik. Kirim ketika infrastruktur mendukung indeks SPLADE.

Desain retrieval yang benar mengurangi halusinasi 70-90% menurut pengukuran industri 2026. Kebanyakan gain performa RAG berasal dari retrieval lebih baik, bukan fine-tuning model.

## Rilis

Simpan sebagai `outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## Latihan

1. **Mudah.** Implementasikan `hybrid_search` di atas pada korpus 500-dokumen. Uji 20 query. Bandingkan recall pada 5 antara BM25-saja, dense-saja, dan hibrida.
2. **Sedang.** Tambahkan perhitungan MRR. Untuk tiap query uji dengan dokumen benar yang diketahui, temukan peringkat dok benar dalam peringkat BM25, dense, dan hibrida. Laporkan MRR masing-masing.
3. **Sulit.** Fine-tune encoder dense pada domainmu memakai MultipleNegativesRankingLoss (Sentence Transformers). Bangun set training dari 500 pasangan query-dokumen. Bandingkan recall sebelum dan sesudah fine-tune.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BM25 | Keyword search | Okapi BM25. Scores documents by term frequency, IDF, and length. |
| Dense retrieval | Vector search | Encode query + doc into vectors, find nearest neighbors. |
| Bi-encoder | Embedding model | Encodes query and doc independently. Fast at query time. |
| Cross-encoder | Reranker model | Encodes query + doc together. Slow but accurate. |
| RRF | Rank fusion | Combine two rankings by summing `1/(k + rank)`. |
| Recall@k | Retrieval metric | Fraction of queries where a relevant doc is in the top-k. |

## Bacaan Lanjutan

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — the definitive BM25 treatment.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, the canonical bi-encoder.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — the learned-sparse retriever that closes the gap with dense.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF paper.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — late-interaction retrieval.
