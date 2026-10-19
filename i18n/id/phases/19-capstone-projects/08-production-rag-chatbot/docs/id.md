# Capstone 08 — Chatbot RAG Produksi untuk Vertikal Terregulasi

> Harvey, Glean, Mendable, dan LlamaCloud semuanya menjalankan bentuk produksi yang sama pada 2026. Serap dengan docling atau Unstructured serta ColPali untuk visual. Hybrid search. Re-rank dengan bge-reranker-v2-gemma. Sintesis dengan Claude Sonnet 4.7 memakai prompt caching pada hit rate 60-80%. Jaga dengan Llama Guard 4 dan NeMo Guardrails. Pantau dengan Langfuse dan Phoenix. Nilai dengan RAGAS pada golden set 200 pertanyaan. Bangun satu di domain terregulasi (hukum, klinis, asuransi), dan capstone-nya adalah lulus golden set, red team, dan dashboard drift.

**Type:** Capstone
**Languages:** Python (pipeline + API), TypeScript (chat UI)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P7 · P11 · P12 · P17 · P18
**Time:** 30 hours

## Masalahnya

RAG di domain terregulasi (kontrak hukum, protokol uji klinis, polis asuransi) adalah bentuk produksi yang paling banyak dirilis pada 2026 karena ROI-nya jelas dan taruhannya konkret. Harvey (Allen & Overy) membangunnya untuk hukum. Mendable merilis versi developer-docs. Glean mencakup enterprise search. Polanya: serap dengan fidelitas tinggi, retrieve secara hibrida dengan rerank, sintesis dengan penegakan sitasi dan prompt caching, jaga dengan beberapa lapisan keselamatan, dan pantau drift secara berkelanjutan.

Bagian sulitnya bukan modelnya. Melainkan kepatuhan yang sadar-yurisdiksi (HIPAA, GDPR, SOC2), auditabilitas setingkat sitasi, kontrol biaya (prompt caching memberi diskon 60-90% saat hit rate tinggi), deteksi halusinasi lewat faithfulness RAGAS, dan deteksi drift saat dokumen sumber diperbarui tanpa indeks ikut menyusul. Capstone ini meminta kamu merilis semuanya pada golden set 200 pertanyaan dengan suite red team di sampingnya.

## Konsepnya

Pipeline punya dua sisi. **Ingestion**: docling atau Unstructured mem-parse dokumen terstruktur; ColPali menangani yang kaya-visual; chunk mendapat ringkasan, tag, dan label akses berbasis peran. Vektor masuk ke pgvector + pgvectorscale (di bawah 50M vektor) atau Qdrant Cloud; BM25 sparse berjalan bersandingan. **Percakapan**: LangGraph menangani memory dan multi-turn; setiap query menjalankan retrieval hibrida, di-rank ulang dengan bge-reranker-v2-gemma-2b, disintesis dengan Claude Sonnet 4.7 (prompt-cached), output-nya melewati Llama Guard 4 dan NeMo Guardrails, lalu mengeluarkan respons yang berjangkar sitasi.

Stack eval-nya punya empat lapis. **Golden set** (200 Q/A berlabel dengan sitasi) untuk kebenaran. **Red team** (jailbreak, percobaan ekstraksi PII, pertanyaan di luar domain) untuk keselamatan. **RAGAS** untuk faithfulness / answer relevance / context precision secara otomatis per-turn. **Dashboard drift** (Arize Phoenix) yang memantau kualitas retrieval dan skor halusinasi setiap minggu.

Prompt caching adalah tuas biayanya. Claude 4.5+ dan GPT-5+ mendukung caching system prompt + konteks hasil retrieval. Pada hit rate 60-80%, biaya per-query turun 3-5x. Pipeline harus dirancang untuk prefix yang stabil (system prompt + konteks hasil rerank di depan) guna mencapai cache hit rate yang tinggi.

## Arsitektur

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## Stack

- Ingestion: Unstructured.io atau docling untuk dokumen terstruktur; ColPali untuk PDF kaya-visual
- Vector DB: pgvector + pgvectorscale di bawah 50M vektor; Qdrant Cloud selain itu
- Sparse: Tantivy BM25 dengan bobot field
- Orkestrasi: LlamaIndex Workflows (ingestion) + LangGraph (percakapan)
- Re-ranker: bge-reranker-v2-gemma-2b self-hosted atau Voyage rerank-2 hosted
- LLM: Claude Sonnet 4.7 dengan prompt caching; fallback Llama 3.3 70B self-hosted
- Eval: RAGAS 0.2 online, DeepEval untuk suite halusinasi dan jailbreak
- Observability: Langfuse self-hosted dengan antrian anotasi; Arize Phoenix untuk drift
- Guardrails: classifier input/output Llama Guard 4, policy NeMo Guardrails v0.12, scrub PII Presidio
- Kepatuhan: label akses berbasis peran pada chunk; tag yurisdiksi untuk GDPR/HIPAA

```figure
canary-rollout
```

## Bangun Sendiri

1. **Ingestion.** Parse korpusmu (1000-10000 dokumen untuk build yang serius) dengan Unstructured atau docling. Untuk halaman hasil scan / kaya-visual, arahkan lewat ColPali. Hasilkan chunk dengan ringkasan, label peran, tag yurisdiksi.

2. **Indeks.** Embedding dense (Voyage-3 atau Nomic-embed-v2) ke pgvector + pgvectorscale. Side-index BM25 via Tantivy. Filter peran dan yurisdiksi sebagai payload.

3. **Retrieve hibrida.** Filter berdasarkan peran+yurisdiksi lebih dulu; lalu dense + BM25 paralel; gabungkan dengan reciprocal rank fusion; top-20 ke reranker; top-5 ke synth.

4. **Sintesis dengan prompt caching.** System prompt + policy statis di cache header; konteks hasil rerank sebagai ekstensi cache; pertanyaan pengguna sebagai suffix yang tidak di-cache. Targetkan cache hit rate 60-80% pada kondisi stabil.

5. **Guardrails.** Llama Guard 4 pada input; rails NeMo Guardrails memblokir pertanyaan di luar domain atau topik yang dilarang policy; Presidio meny scrub PII yang tidak sengaja ada di keluaran; post-filter penegakan sitasi.

6. **Golden set.** 200 pasangan Q/A yang diberi label oleh ahli domain beserta (jawaban, sitasi). Nilai agen pada kecocokan sitasi persis, kebenaran jawaban, faithfulness (RAGAS).

7. **Red team.** 50 prompt adversarial: jailbreak (PAIR, TAP), percobaan ekstraksi PII, di luar domain, kebocoran lintas-yurisdiksi. Nilai dengan pass/fail dan tingkat keparahan.

8. **Dashboard drift.** Arize Phoenix melacak kualitas retrieval (nDCG, faithfulness sitasi) setiap minggu. Beri peringatan pada penurunan 5%.

9. **Laporan biaya.** Langfuse: hit rate prompt-caching, token per query, rincian $/query per tahap.

## Gunakan

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## Rilis

`outputs/skill-production-rag.md` menjelaskan deliverablenya. Sebuah chatbot domain terregulasi yang di-deploy dengan label kepatihan, lolos rubrik, dan dipantau dengan monitoring drift live.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | RAGAS faithfulness + answer relevance | Online scores on the golden set (200 Q/A) |
| 20 | Citation correctness | Fraction of answers with verifiable source anchors |
| 20 | Guardrail coverage | Llama Guard 4 pass rate + jailbreak suite results |
| 20 | Cost / latency engineering | Prompt-cache hit rate, p95 latency, $/query |
| 15 | Drift monitoring dashboard | Phoenix live dashboard with weekly retrieval-quality trend |
| **100** | | |

## Latihan

1. Bangun slice korpus kedua di bawah yurisdiksi berbeda (mis. HIPAA berdampingan dengan GDPR). Demonstrasikan filter peran+yurisdiksi mencegah cross-leak pada probe lintas-yurisdiksi 20-pertanyaan.

2. Ukur hit rate prompt-cache selama satu minggu lalu lintas produksi. Identifikasi query mana yang memecah prefix cache. Restrukturisasi.

3. Tambahkan memory multi-turn dengan buffer ringkasan 10k token. Ukur apakah faithfulness turun seiring percakapan bertambah.

4. Tukar Claude Sonnet 4.7 dengan Llama 3.3 70B self-hosted. Ukur $/query dan selisih faithfulness-nya.

5. Tambahkan mode "unsure": bila skor rerank teratas di bawah sebuah ambang, agen berkata "I do not have confident citations" alih-alih menjawab. Ukur penurunan false-confidence.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Prompt caching | "Cached system + context" | Claude/OpenAI feature: cached prefix tokens discounted 60-90% on hit |
| RAGAS | "RAG evaluator" | Automated scoring of faithfulness, answer relevance, context precision |
| Golden set | "Labeled eval" | 200+ expert-labeled Q/A with citations; the ground truth |
| Jurisdiction tag | "Compliance label" | GDPR/HIPAA/SOC2 scope attached to chunks; enforced by retrieval filter |
| Citation faithfulness | "Grounded answer rate" | Fraction of claims backed by retrievable source spans |
| Drift | "Retrieval quality decay" | Weekly change in nDCG or citation score; alert threshold 5% |
| Red team | "Adversarial eval" | Pre-release jailbreak, PII extraction, off-domain probes |

## Bacaan Lanjutan

- [Harvey AI](https://www.harvey.ai) — stack produksi hukum referensi
- [Glean enterprise search](https://www.glean.com) — referensi RAG berskala enterprise
- [Mendable documentation](https://mendable.ai) — referensi RAG developer-docs
- [LlamaCloud Parse + Index](https://docs.cloud.llamaindex.ai/llamaparse/getting_started) — ingestion managed
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — referensi tuas biaya
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — framework eval RAG kanonikal
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — observability drift referensi
- [Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — classifier keselamatan 2026
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — framework rail policy
