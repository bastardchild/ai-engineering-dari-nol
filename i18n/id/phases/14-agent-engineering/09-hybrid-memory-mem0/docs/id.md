# Hibrida Memori: Vector + Graf + KV

> Hibrida memori berjalan tiga stores di paralel — vector untuk semantic similarity, KV untuk cepat fact lookup, graf untuk entity-relationship penalaran — dengan sebuah scoring layer itu fuses mereka pada retrieval. Ini adalah sebuah widely used produksi pola untuk eksternal memori; Mem0 (Chhikara et al., 2025) adalah satu implementasi referensi.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan mengapa sebuah tunggal store (vector hanya, graf hanya, KV hanya) adalah insufficient untuk agen memori.
- Name Mem0's tiga paralel stores dan apa setiap satu optimizes untuk.
- Deskripsikan Mem0's fusion scoring — relevance, importance, recency — dan mengapa itu adalah sebuah weighted sum, tidak sebuah hierarchy.
- Implementasikan sebuah toy tiga-store memori di stdlib dengan sebuah `add()` itu menulis ke semua tiga dan sebuah `search()` itu fuses hasil.

## Masalahnya

Satu store adalah salah untuk satu dari tiga query classes:

- **Semantic similarity** — "apa melakukan kita discuss tentang agen drift terakhir week?" Vector wins; KV dan graf miss.
- **Fact lookup** — "apa adalah user's phone number?" KV wins; vector adalah wasteful, graf adalah overkill.
- **Relationship reasoning** — "yang customers share sama billing entity?" Graf wins; vector dan KV cannot jawaban.

Produksi agen issue semua tiga di satu sesi. SEBUAH tunggal-store memori adalah selalu salah untuk dua dari mereka. Mem0's contribution adalah wiring semua tiga di balik sebuah tunggal `add`/`search` surface dengan sebuah scoring function itu fuses mereka.

## Konsepnya

### Tiga stores di paralel

Mem0 (arXiv:2504.19413, April 2025) pada `add(text, user_id, metadata)`:

1. Extract candidate facts dari text (sebuah LLM-driven langkah).
2. Tulis setiap fact ke vector store (embedding) untuk semantic pencarian.
3. Tulis setiap fact ke KV store keyed pada (user_id, fact_type, entity) untuk O(1) lookup.
4. Tulis setiap fact ke graf store (Mem0g) sebagai typed edges untuk relationship queries.

Pada `search(query, user_id)`:

1. Vector store mengembalikan top-k oleh embedding cosine.
2. KV store mengembalikan direct hits keyed pada query-derived (user_id, type, entity).
3. Graf store mengembalikan subgraph reachable dari query entities.
4. SEBUAH scoring layer fuses tiga.

### Fusion scoring

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance** — vector cosine, KV exact match, graf path weight.
- **Importance** — tagged pada tulis time atau learned (beberapa facts matter more: names, IDs, policies).
- **Recency** — exponential decay atas time since terakhir tulis atau baca.

Weights adalah tuned per product. Higher `w_recency` untuk chat agen; higher `w_importance` untuk compliance agen; higher `w_relevance` untuk retrieval agen.

### Mem0g dan temporal penalaran

Mem0g adds sebuah conflict detector. Ketika sebuah baru fact contradicts sebuah existing edge, existing edge adalah marked invalid but tidak deleted. Temporal queries ("apa adalah user's city di March?") traverse valid-pada-time subgraph.

Ini adalah compliance-grade perilaku Letta's invalidation pola generalizes.

### Benchmark numbers

 Mem0 paper reports (2025):

- **LoCoMo** (long-form conversation memori): 91.6
- **LongMemEval** (long-horizon episodic memori): 93.4
- **BEAM 1M** (1M-token memori benchmark): 64.1

Comparison baselines (penuh-context 128k LLM, flat vector store, flat KV) semua lose oleh 10+ points. Benchmark alone don't justify choice — bentuk operasional melakukan — but numbers tunjukkan fusion design adalah tidak sebuah rounding error.

### Scope taxonomy

Mem0 splits memori oleh scope:

- **User memory** — persists lintas sesi, keyed pada `user_id`.
- **Session memory** — persists di dalam satu thread.
- **Agent memory** — per-agen instance state.

Setiap tulis picks satu scope. Retrieval bisa query lintas scopes dengan per-scope weights. Mixing scopes tanpa thought adalah bagaimana kamu dapatkan " assistant told Alice tentang Bob's project" incidents.

### Di mana ini pola goes salah

- **Embedding drift.** Vector hasil itu look right pada pertama hundred queries degrade sebagai corpus yang terus bertambah. Add periodic re-embedding dari top-N-used records.
- **KV schema creep.** `(user_id, type, entity)` looks sederhana sampai setiap tim adds mereka own `type`. Audit type set quarterly.
- **Graph explosion.** Satu noisy extractor adds 50 edges per message. Cap graf menulis per `add` panggil; drop low-confidence edges.

```figure
ae-memory-fusion
```

## Bangun Sendiri

`code/main.py` implements tiga-store pola di stdlib:

- `VectorStore` — naive token-overlap similarity sebagai sebuah embedding stand-di.
- `KVStore` — dict keyed pada `(user_id, fact_type, entity)`.
- `GraphStore` — typed edges (subject, relation, object, valid).
- `Mem0` — top-level facade dengan `add()`, `search()`, fusion scoring, dan scope-aware retrieval.
- SEBUAH worked trace pada sebuah multi-user, multi-sesi conversation.

Jalankan:

```
python3 code/main.py
```

 output menunjukkan tiga terpisah recall paths plus fused top-k. Flip scoring weights pada top dari `main()` dan watch ranking change.

## Gunakan

- **Mem0 (Apache 2.0)** — produksi-ready. Mandiri-host dengan Postgres + Qdrant + Neo4j, atau pakai managed cloud.
- **Letta** — tiga-tier core/recall/archival; bring kamu own vector dan graf backends.
- **Zep** — commercial alternative dengan temporal KG dan fact extraction.
- **Custom builds** — ketika kamu butuhkan exact kontrol atas extractor (compliance) atau fusion weights (suara agen di mana recency dominates).

## Rilis

`outputs/skill-hybrid-memory.md` menghasilkan sebuah tiga-store memori scaffold dengan sebuah fusion scorer, scope taxonomy, dan temporal invalidation wired di.

## Latihan

1. Ganti toy vector similarity dengan sebuah nyata embedding model (sentence-transformers, Ollama, OpenAI embeddings). Measure recall@10 pada sebuah synthetic long conversation. Melakukan ranking drift atas 1000 menulis?
2. Tambahkan sebuah temporal query: `search(query, as_of=timestamp)`. Kembalikan hanya records valid pada atau sebelum itu time. Yang store membutuhkan sebagian besar kerja?
3. Implementasikan sebuah conflict detector: jika sebuah incoming fact contradicts sebuah graf edge, invalidate lama edge dan log keduanya. Tes pada "user lives di Berlin" -> "user lives di Lisbon."
4. Port fusion scorer ke sertakan sebuah `user_feedback` dimension (thumbs-up pada retrieved records). Bagaimana lakukan kamu prevent gaming (agen itu hanya mengembalikan records itu sudah liked)?
5. Baca Mem0 docs (`docs.mem0.ai`). Port toy ke `mem0` klien memanggil. Bandingkan retrieval quality pada sama 20 tes queries.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Hybrid memory | "Vector plus graph plus KV" | Three stores written in parallel, fused on retrieval |
| Fact extraction | "Memory ingestion" | LLM step that breaks text into (entity, relation, fact) tuples |
| Fusion scoring | "Relevance ranking" | Weighted sum of relevance, importance, recency |
| Scope | "Memory namespace" | user / session / agent — determines who sees what |
| Mem0g | "Memory graph" | Typed edges with temporal validity for relationship queries |
| Temporal invalidation | "Soft delete" | Mark contradicted edges invalid; never delete |
| Embedding drift | "Retrieval rot" | Vector quality degrades as corpus grows; re-embed periodically |

## Bacaan Lanjutan

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — original paper
- [Mem0 docs](https://docs.mem0.ai/platform/overview) — produksi API, SDKs, managed cloud
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — virtual-context predecessor
- [Letta, Memori Blocks blog](https://www.letta.com/blog/memory-blocks) — tiga-tier sibling design
