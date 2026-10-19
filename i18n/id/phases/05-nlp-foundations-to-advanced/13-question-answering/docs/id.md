# Sistem Penjawab Pertanyaan

> Tiga sistem membentuk QA modern. Ekstraktif menemukan span. Retrieval-augmented meng-grounding-nya pada dokumen. Generatif menghasilkan jawaban. Setiap asisten AI modern adalah campuran ketiganya.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75 minutes

## Masalahnya

Seorang pengguna mengetik "When did the first iPhone launch?" dan mengharapkan "June 29, 2007." Bukan "Apple's history is long and varied." Bukan "2007" yang duduk terisolasi tanpa kalimat. Jawaban langsung, ter-grounding, benar.

Tiga arsitektur telah mendominasi QA selama dekade terakhir.

- **QA ekstraktif.** Diberi pertanyaan dan passage yang diketahui memuat jawaban, temukan indeks awal dan akhir span jawaban di passage. SQuAD adalah benchmark kanonis.
- **QA open-domain.** Passage tidak diberikan. Ambil passage relevan dulu, lalu ekstrak atau hasilkan jawaban. Ini fondasi setiap pipeline RAG kini.
- **QA generatif / closed-book.** Model bahasa besar menjawab dari memori parametriknya. Tanpa retrieval. Tercepat saat inference, paling tak andal pada fakta.

Tren di 2026 adalah hibrida: ambil beberapa passage terbaik, lalu prompt model generatif untuk menjawab dengan grounding pada passage itu. Itulah RAG, dan pelajaran 14 membahas separuh retrieval secara mendalam. Pelajaran ini membangun separuh QA.

## Konsepnya

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**Ekstraktif.** Encode pertanyaan dan passage bersama dengan transformer (keluarga BERT). Latih dua head yang memprediksi indeks token awal dan akhir jawaban. Loss adalah cross-entropy atas posisi valid. Output adalah span dari passage. Tak pernah berhalusinasi (by construction), tak pernah menangani pertanyaan yang tak bisa dijawab passage (by construction).

**Retrieval-augmented (RAG).** Dua tahap. Pertama, retriever menemukan `k` passage teratas dari korpus. Kedua, reader (ekstraktif atau generatif) menghasilkan jawaban memakai passage itu. Pemisahan retriever-reader memungkinkan masing-masing dilatih dan dievaluasi independen. RAG modern sering menambahkan reranker di antaranya.

**Generatif.** LLM decoder-only (GPT, Claude, Llama) menjawab dari bobot terpelajari. Tanpa langkah retrieval. Unggul pada pengetahuan umum, katastropik pada fakta jarang atau baru. Tingkat halusinasi berkorelasi terbalik dengan frekuensi fakta di data pretraining.

```figure
qa-span
```

## Bangun Sendiri

### Langkah 1: QA ekstraktif dengan model pra-terlatih

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` dilatih pada SQuAD 2.0, yang mencakup pertanyaan tak terjawab. Secara default, pipeline `question-answering` mengembalikan span berskor tertinggi bahkan ketika skor null model menang — ia *tidak* otomatis mengembalikan jawaban kosong. Untuk mendapat perilaku "tanpa jawaban" eksplisit, teruskan `handle_impossible_answer=True` ke panggilan pipeline: pipeline lalu mengembalikan jawaban kosong hanya ketika skor null melebihi setiap skor span. Selalu cek field `score` bagaimana pun caranya.

### Langkah 2: pipeline retrieval-augmented (sketsa)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

Pipeline dua tahap. Retriever dense (Sentence-BERT) menemukan passage relevan via kemiripan semantik. Reader ekstraktif (RoBERTa-SQuAD) menarik span jawaban dari gabungan passage teratas. Berhasil pada korpus kecil. Untuk korpus sejuta-dokumen, pakai FAISS atau vector database.

### Langkah 3: generatif dengan RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Pola prompt penting. Memberi tahu model secara eksplisit untuk grounding pada konteks dan mengembalikan "I don't know" ketika konteks tak cukup memangkas tingkat halusinasi 40-60% dibanding prompting naif. Pola yang lebih elaborat menambahkan sitasi, skor keyakinan, dan ekstraksi terstruktur.

### Langkah 4: evaluasi yang mencerminkan dunia nyata

SQuAD memakai **Exact Match (EM)** dan **F1 level-token**. EM adalah kecocokan ketat setelah normalisasi (lowercase, buang tanda baca, buang artikel) — prediksi cocok tepat atau skornya 0. F1 dihitung atas overlap token antara prediksi dan referensi dan memberi kredit parsial. Keduanya kurang menghargai parafrasa: "June 29, 2007" vs "June 29th, 2007" biasanya mendapat 0 EM (ordinal merusak normalisasi) tetapi tetap mendapat F1 substansial dari token yang overlap.

Untuk QA produksi:

- **Akurasi jawaban** (dinilai LLM atau manusia, karena metrik tidak menangkap kesetaraan semantik).
- **Akurasi sitasi.** Apakah passage yang disitasi benar-benar mendukung jawaban? Mudah dicek otomatis dengan pencocokan string antara sitasi yang dihasilkan dan passage yang diambil.
- **Kalibrasi refusal.** Ketika jawaban tidak ada di passage yang diambil, apakah sistem benar berkata "I don't know"? Ukur tingkat keyakinan palsu.
- **Recall retrieval.** Sebelum mengevaluasi reader, ukur apakah retriever memasukkan passage benar ke `k` teratas. Reader tidak bisa memperbaiki passage yang hilang.

### RAGAS: framework eval produksi 2026

`RAGAS` dibuat khusus untuk sistem RAG dan menjadi default pengiriman di 2026. Ia menilai empat dimensi tanpa membutuhkan referensi emas:

- **Faithfulness.** Apakah setiap klaim dalam jawaban berasal dari konteks yang diambil? Diukur via entailment berbasis NLI. Metrik halusinasimu yang utama.
- **Relevansi jawaban.** Apakah jawaban menjawab pertanyaan? Diukur dengan menghasilkan pertanyaan hipotetis dari jawaban dan membandingkannya dengan pertanyaan asli.
- **Presisi konteks.** Dari chunk yang diambil, berapa fraksi yang benar-benar relevan? Presisi rendah = noise dalam prompt.
- **Recall konteks.** Apakah himpunan yang diambil memuat semua informasi yang dibutuhkan? Recall rendah = reader tak bisa berhasil.

Skor bebas-referensi memungkinkanmu mengevaluasi pada traffic produksi live tanpa jawaban emas kurasi. Lapiskan LLM-as-judge di atasnya untuk pertanyaan terbuka di mana metrik exact-match tak berguna.

`pip install ragas`. Pasang retriever + reader-mu. Dapatkan empat skalar per query. Alert pada regresi.

## Gunakan

Stack 2026.

| Use case | Recommended |
|---------|-------------|
| Given passage, find answer span | `deepset/roberta-base-squad2` |
| Over a fixed corpus, closed-book not acceptable | RAG: dense retriever + LLM reader |
| Real-time over a document store | RAG with hybrid (BM25 + dense) retriever + reranker (lesson 14) |
| Conversational QA (follow-up questions) | LLM with conversation history + RAG on each turn |
| Highly factual, regulated domains | Extractive over an authoritative corpus; never generative alone |

QA ekstraktif tidak modis di 2026 karena RAG dengan LLM menangani lebih banyak kasus. Ia masih dikirim dalam konteks di mana kutipan harfiah diwajibkan: riset legal, compliance regulasi, alat audit.

## Rilis

Simpan sebagai `outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## Latihan

1. **Mudah.** Siapkan pipeline ekstraktif SQuAD di atas pada 10 passage Wikipedia. Buat 10 pertanyaan dengan tangan. Ukur seberapa sering jawabannya benar. Kamu akan melihat 7-9 benar jika passage dan pertanyaan bersih.
2. **Sedang.** Tambahkan pengklasifikasi refusal. Ketika skor retrieval teratas di bawah ambang (mis. kosinus 0.3), kembalikan "I don't know" alih-alih memanggil reader. Tune ambang pada held-out set.
3. **Sulit.** Bangun pipeline RAG atas korpus 10.000-dokumen pilihanmu. Implementasikan retrieval hibrida (BM25 + dense) dengan fusi RRF (lihat pelajaran 14). Ukur akurasi jawaban dengan dan tanpa langkah hibrida. Dokumentasikan tipe pertanyaan mana yang paling diuntungkan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Extractive QA | Find the answer span | Predict start and end indices of the answer within a given passage. |
| Open-domain QA | QA over a corpus | No given passage; must retrieve then answer. |
| RAG | Retrieve then generate | Retrieval-augmented generation. Retriever + reader pipeline. |
| SQuAD | Canonical benchmark | Stanford Question Answering Dataset. EM + F1 metrics. |
| Hallucination | Made-up answer | Reader output not supported by retrieved context. |
| Refusal calibration | Know when to shut up | System correctly says "I don't know" when unable to answer. |

## Bacaan Lanjutan

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — the benchmark paper.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, the canonical dense retriever for QA.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — the paper that named RAG.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — comprehensive RAG survey.
