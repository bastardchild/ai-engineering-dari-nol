# Strategi Chunking untuk RAG

> Konfigurasi chunking memengaruhi kualitas retrieval sama besarnya dengan pilihan model embedding (Vectara NAACL 2025). Salah chunking dan reranking sebanyak apa pun tak menyelamatkanmu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## Masalahnya

Kamu memasukkan kontrak 50-halaman ke sistem RAG. Pengguna bertanya: "What is the termination clause?" Retriever mengembalikan halaman sampul. Mengapa? Karena model dilatih pada chunk 512-token dan klausul terminasi duduk 20 halaman di dalam, terbelah di jeda halaman, tanpa keyword lokal yang mengikatnya ke query.

Perbaikannya bukan "beli model embedding lebih baik." Perbaikannya adalah chunking. Seberapa besar? Overlap? Pecah di mana? Dengan konteks sekitar?

Benchmark Feb 2026 menunjukkan hasil mengejutkan:

- Studi Vectara 2026: chunking rekursif 512-token mengalahkan chunking semantik 69% → 54% akurasi.
- SPLADE + Mistral-8B pada Natural Questions: overlap memberi nol manfaat terukur.
- Tebing konteks: kualitas respons turun tajam sekitar 2.500 token konteks.

Jawaban "jelas" (chunking semantik, overlap 20%, 1000 token) sering salah. Pelajaran ini membangun intuisi untuk enam strategi dan memberi tahu kapan meraih mana.

## Konsepnya

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Chunking tetap.** Pecah setiap N karakter atau token. Baseline paling sederhana. Patah di tengah kalimat. Kompresi bagus, koherensi buruk.

**Rekursif.** `RecursiveCharacterTextSplitter` LangChain. Coba pecah pada `\n\n` dulu, lalu `\n`, lalu `.`, lalu spasi. Jatuh mundur dengan bersih. Default 2026.

**Semantik.** Embed tiap kalimat. Hitung kemiripan kosinus antar kalimat bersebelahan. Pecah di mana kemiripan turun di bawah ambang. Menjaga koherensi topik. Lebih lambat; kadang menghasilkan fragmen mungil 40-token yang merugikan retrieval.

**Kalimat.** Pecah pada batas kalimat. Satu kalimat per chunk atau window N kalimat. Menyamai chunking semantik sampai ~5k token dengan fraksi biayanya.

**Parent-document.** Simpan child chunk kecil untuk retrieval *dan* chunk induk lebih besar untuk konteks. Ambil via child; kembalikan induk. Terdegradasi anggun: child chunk buruk tetap mengembalikan induk yang wajar.

**Late chunking (2024).** Embed seluruh dokumen pada level token dulu, lalu pool embedding token menjadi embedding chunk. Menjaga konteks lintas-chunk. Berhasil dengan embedder konteks-panjang (BGE-M3, Jina v3). Komputasi lebih tinggi.

**Retrieval kontekstual (Anthropic, 2024).** Awali tiap chunk dengan ringkasan hasil-LLM tentang posisinya dalam dokumen ("This chunk is section 3.2 of the termination clauses..."). Peningkatan retrieval 35-50% dalam benchmark Anthropic sendiri. Mahal diindeks.

### Aturan yang mengalahkan setiap default

Cocokkan ukuran chunk dengan tipe query:

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

Benchmark NVIDIA 2026. Chunk harus cukup besar memuat jawaban plus konteks lokal, cukup kecil sehingga top-K retriever fokus pada jawaban alih-alih noise konteks.

```figure
n5-chunk-cuts
```

## Bangun Sendiri

### Langkah 1: chunking tetap dan rekursif

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Langkah 2: chunking semantik

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

Tune `threshold` pada domainmu. Terlalu tinggi → fragmen. Terlalu rendah → satu chunk raksasa.

### Langkah 3: parent-document

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

Wawasan kunci: dedupe induk. Banyak child bisa dipetakan ke induk yang sama; mengembalikan semuanya membuang konteks.

### Langkah 4: retrieval kontekstual (pola Anthropic)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

Indeks chunk yang telah dikontekstualisasi. Saat query, retrieval diuntungkan sinyal sekitar tambahan.

### Langkah 5: evaluasi

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

Selalu benchmark. Strategi "terbaik" untuk korpusmu mungkin tidak cocok dengan blog post mana pun.

## Jebakan

- **Chunking dievaluasi hanya pada query faktoid.** Query multi-hop mengungkap pemenang yang sangat berbeda. Pakai eval set terstratifikasi tipe-query.
- **Chunking semantik tanpa ukuran minimum.** Menghasilkan fragmen 40-token yang merugikan retrieval. Selalu tegakkan `min_tokens`.
- **Overlap sebagai cargo cult.** Studi 2026 menemukan overlap sering memberi nol manfaat dan menggandakan biaya indeks. Ukur, jangan asumsikan.
- **Tanpa penegakan min/max.** Chunk 5 token atau 5000 token sama-sama merusak retrieval. Jepit.
- **Chunking lintas-dok.** Jangan biarkan chunk merentang dua dokumen. Selalu chunk per-dok, lalu gabung.

## Gunakan

Stack 2026:

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| Heavy cross-reference (contracts, papers) | Late chunking or contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances (tweets, reviews) | One document = one chunk |

Mulai dengan rekursif 512. Ukur recall@5 pada eval set 50-query. Tune dari sana.

## Rilis

Simpan sebagai `outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Latihan

1. **Mudah.** Chunk satu dokumen 20-halaman dengan fixed(512, 0), recursive(512, 0), dan recursive(512, 100). Bandingkan jumlah chunk dan kualitas batas.
2. **Sedang.** Bangun eval set 30-query atas 5 dokumen. Ukur recall@5 untuk rekursif, semantik, dan parent-document. Mana yang menang? Apakah cocok dengan blog post?
3. **Sulit.** Implementasikan retrieval kontekstual. Ukur peningkatan MRR atas baseline rekursif. Laporkan biaya indeks (panggilan LLM) vs gain akurasi.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Chunk | A piece of a doc | Sub-document unit that gets embedded, indexed, and retrieved. |
| Overlap | Safety margin | N tokens shared between adjacent chunks; often useless in 2026 benchmarks. |
| Semantic chunking | Smart chunking | Split where adjacent-sentence embedding similarity drops. |
| Parent-document | Two-level retrieval | Retrieve small children, return larger parents. |
| Late chunking | Chunk after embedding | Embed full doc at token level, pool into chunk vectors. |
| Contextual retrieval | Anthropic's trick | LLM-generated summary prepended to each chunk before indexing. |
| Context cliff | 2500-token wall | Quality drop observed around 2.5k context tokens in RAG (Jan 2026). |

## Bacaan Lanjutan

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — the default in production.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — chunking matters as much as embedding choice.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — the late chunking paper.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 35-50% retrieval improvement with LLM-generated context prefixes.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — chunk size by query type.
