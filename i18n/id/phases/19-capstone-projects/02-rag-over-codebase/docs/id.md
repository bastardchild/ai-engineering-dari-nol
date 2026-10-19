# Capstone 02 — RAG di Atas Codebase (Pencarian Semantik Lintas-Repo)

> Setiap organisasi engineering serius pada 2026 menjalankan code search internal yang memahami makna, bukan sekadar string. Sourcegraph Amp, codebase answers milik Cursor, graph enterprise Augment, repomap Aider, MCP internal Pinterest — bentuknya sama. Menyerap banyak repo, parse dengan tree-sitter, embed chunk setingkat fungsi dan kelas, hybrid-search, re-rank, jawab dengan sitasi. Capstone ini meminta kamu membangun satu yang menangani 2 juta baris kode di 10 repo dan bertahan pada re-indexing inkremental setiap git push.

**Type:** Capstone
**Languages:** Python (ingestion), TypeScript (API + UI)
**Prerequisites:** Phase 5 (NLP foundations), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P17
**Time:** 30 hours

## Masalahnya

Pada 2026 setiap coding agent frontier dibekali lapisan retrieval codebase karena context window saja tidak menyelesaikan pertanyaan lintas-repo. Context 1M-token milik Claude membantu; itu tidak menghilangkan kebutuhan akan retrieval berperingkat. Pencarian cosine sederhana atas chunk mentah meracuni hasil pada kode yang di-generate, pada duplikasi monorepo, dan pada ekor panjang simbol yang jarang di-import. Jawaban produksinya adalah pencarian hibrida (dense + BM25) atas chunk yang sadar-AST dengan sebuah re-ranker, ditopang graph referensi simbol.

Kamu mempelajarinya dengan mengindeks armada yang nyata — bukan satu repo tutorial — lalu mengukur MRR@10, faithfulness sitasi, dan kesegaran inkremental. Mode kegagalannya bersifat infrastruktur: monorepo berisi 100k file, sebuah push yang menyentuh separuh file, sebuah query yang perlu menyeberangi empat repo untuk menjawab dengan benar.

## Konsepnya

Pipeline ingestion yang sadar-AST mem-parse setiap file dengan tree-sitter, mengekstrak node fungsi dan kelas, lalu memecah di batas node alih-alih jendela token tetap. Setiap chunk mendapat tiga representasi: embedding dense (Voyage-code-3 atau nomic-embed-code), istilah BM25 yang sparse, dan ringkasan bahasa alami pendek. Ringkasan menambahkan modalitas ketiga yang bisa diretrieve — pengguna bertanya "bagaimana X di-authorize" dan ringkasan menyebut "authz", meski kode hanya punya `check_permission`.

Retrieval-nya hibrida. Sebuah query menembakkan pencarian dense dan BM25 sekaligus, menggabungkan top-k, lalu menyerahkan union-nya ke re-ranker cross-encoder (Cohere rerank-3 atau bge-reranker-v2-gemma-2b). Daftar hasil yang sudah di-rank dikirim ke synthesizer context panjang (Claude Sonnet 4.7 dengan prompt caching, atau Llama 3.3 70B self-hosted) dengan instruksi mengutip setiap klaim beserta file dan rentang baris. Jawaban tanpa sitasi ditolak oleh post-filter.

Kesegaran inkremental adalah masalah infrastruktur. Git push memicu sebuah diff: file mana yang berubah, simbol mana yang berubah. Hanya chunk yang terpengaruh yang di-embed ulang. Edge simbol lintas-file yang terpengaruh (import, pemanggilan method) dihitung ulang. Indeks tetap konsisten tanpa memproses ulang 2 juta baris setiap commit.

## Arsitektur

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## Stack

- Parsing: tree-sitter dengan 17 tata bahasa (Python, TS, Rust, Go, Java, C++, dll.)
- Embedding dense: Voyage-code-3 (hosted) atau nomic-embed-code-v1.5 (self-host), fallback bge-code-v1
- Indeks sparse: Tantivy (Rust) dengan BM25F, bobot field untuk nama simbol vs body
- Vector DB: Qdrant 1.12 dengan hybrid search, atau pgvector + pgvectorscale untuk tim di bawah 50M vektor
- Model ringkasan chunk: Claude Haiku 4.5 atau Gemini 2.5 Flash, dengan prompt caching
- Re-ranker: Cohere rerank-3 atau bge-reranker-v2-gemma-2b self-hosted
- Orkestrasi: LlamaIndex Workflows untuk ingestion, LangGraph untuk agen query
- Synthesizer: Claude Sonnet 4.7 (context 1M) dengan prompt caching
- Graph simbol: Neo4j (managed) atau kuzu (embedded) untuk edge import dan panggilan
- Observability: span Langfuse per langkah retrieval + sintesis

```figure
ce-hybrid-retrieval
```

## Bangun Sendiri

1. **Walker ingestion.** Iterasi git history pada setiap hook push. Kumpulkan file yang berubah. Untuk setiap file, parse dengan tree-sitter, ekstrak node fungsi dan kelas beserta source span lengkapnya. Keluarkan record chunk `{repo, path, start_line, end_line, symbol, body}`.

2. **Peringkas chunk.** Kumpulkan chunk menjadi batch panggilan Haiku 4.5 dengan prompt caching pada preamble system. Prompt: "Summarize this function in one sentence, naming its public contract and side effects." Simpan ringkasan bersama chunk-nya.

3. **Pool embedding.** Dua antrean paralel: dense (Voyage-code-3 batch 128) dan ringkasan (model sama, tapi pada string ringkasan). Tulis vektor ke Qdrant dengan payload `{repo, path, start_line, end_line, symbol, kind}`.

4. **Indeks BM25.** Indeks Tantivy berbobot field: nama simbol bobot 4, body simbol bobot 1, ringkasan bobot 2. Memungkinkan query "find the function named X" berdampingan dengan "find the function that does X".

5. **Graph simbol.** Untuk setiap chunk, catat edge: import (file ini memakai simbol Y dari repo Z), panggilan (fungsi ini memanggil method M pada kelas C), inheritance. Simpan di kuzu. Dipakai saat query untuk memperluas retrieval melintasi batas repo.

6. **Agen query.** LangGraph dengan tiga node. `retrieve` menembakkan dense + BM25 secara paralel, mendeduplikasi berdasarkan (repo, path, symbol). `rerank` menjalankan cross-encoder pada top-50 dan menyimpan top-10. `synth` memanggil Claude Sonnet 4.7 dengan chunk hasil rerank dalam context, meng-cache system prompt, dan mewajibkan sitasi file:line.

7. **Penegakan sitasi.** Parse output model; klaim apa pun tanpa jangkar `(repo/path:start-end)` ditandai untuk ditanya ulang atau dibuang. Kembalikan hanya jawaban yang sudah bersitasi kepada pengguna.

8. **Re-index inkremental.** Setiap webhook, hitung diff setingkat simbol. Hanya embed ulang chunk yang teksnya berubah. Hitung ulang edge simbol untuk chunk yang import-nya berubah. Ukur: push 50 file selesai re-index dalam waktu di bawah 60 detik untuk armada 2 juta LOC.

9. **Eval.** Beri label 100 pertanyaan lintas-repo dengan jawaban gold file:line. Ukur MRR@10, nDCG@10, faithfulness sitasi (fraksi klaim dengan jangkar yang bisa diverifikasi), dan latensi p50/p99.

## Gunakan

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## Rilis

Skill deliverable `outputs/skill-codebase-rag.md`. Diberi korpus berisi repo, skill ini mendirikan pipeline ingestion, indeks hibrida, dan agen query, lalu mengembalikan jawaban bersitasi untuk pertanyaan lintas-repo apa pun. Rubrik:

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Retrieval quality | MRR@10 and nDCG@10 on a 100-question held-out set |
| 20 | Citation faithfulness | Fraction of answer claims with verifiable file:line anchors |
| 20 | Latency and scale | p95 query latency at 10k QPS on the indexed corpus size |
| 20 | Incremental indexing correctness | Time from git push to searchable on a 50-file commit |
| 15 | UX and answer formatting | Citation clickability, snippet previews, follow-up affordance |
| **100** | | |

## Latihan

1. Tukar Voyage-code-3 dengan nomic-embed-code self-hosted. Ukur selisih MRR@10-nya. Laporkan apakah jaraknya tertutup saat re-ranking diaktifkan.

2. Suntikkan 20% kode hasil generate (boilerplate buatan LLM) ke dalam korpus lalu evaluasi ulang. Amati keracunan retrieval. Tambahkan flag "generated" ke payload dan turunkan bobot hit tersebut.

3. Benchmark hybrid search Qdrant versus pgvector + pgvectorscale pada ukuran korpusmu. Laporkan p99 pada batch size 1.

4. Tambahkan pemeriksaan drift berbasis sampling: setiap minggu, jalankan ulang eval 100-pertanyaan. Beri peringatan saat MRR@10 turun > 5%.

5. Perluas ke resolusi simbol lintas-bahasa: sebuah fungsi Python yang memanggil service Go lewat gRPC. Pakai graph simbol untuk menautkannya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| AST-aware chunking | "Function-level splits" | Cutting code at tree-sitter node boundaries instead of fixed token windows |
| Hybrid search | "Dense + sparse" | Run BM25 and vector search in parallel, merge top-k, rerank |
| Cross-encoder rerank | "Second-stage rank" | Model that scores each (query, candidate) pair together, more accurate than cosine |
| Prompt caching | "Cached system prompt" | 2026 Claude / OpenAI feature that discounts repeat prefix tokens up to 90% |
| Symbol graph | "Code graph" | Edges for imports, calls, inheritance across files and repos |
| Citation faithfulness | "Grounded answer rate" | Fraction of claims a user can verify by clicking the anchor and reading the referenced span |
| Incremental re-index | "Push-to-search time" | Wall-clock from git push to the changed symbols being queryable |

## Bacaan Lanjutan

- [Sourcegraph Amp](https://ampcode.com) — code intelligence lintas-repo produksi
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — deep-dive referensi untuk capstone ini
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tampilan repo berperingkat berbasis tree-sitter
- [Augment Code enterprise graph](https://www.augmentcode.com) — RAG graph simbol komersial
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — implementasi referensi
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — detail Voyage-code-3
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — referensi cross-encoder
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — referensi platform internal
