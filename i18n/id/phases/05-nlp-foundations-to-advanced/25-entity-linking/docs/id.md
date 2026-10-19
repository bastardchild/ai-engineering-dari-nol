# Entity Linking & Disambiguasi

> NER menemukan "Paris." Entity linking memutuskan: Paris, Prancis? Paris Hilton? Paris, Texas? Paris (pangeran Troya)? Tanpa linking, knowledge graph-mu tetap ambigu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 minutes

## Masalahnya

Sebuah kalimat berbunyi: "Jordan beat the press." NER-mu menandai "Jordan" sebagai PERSON. Bagus. Tetapi Jordan *yang mana*?

- Michael Jordan (basket)?
- Michael B. Jordan (aktor)?
- Michael I. Jordan (profesor ML Berkeley — ya, kebingungan ini nyata dalam paper ML)?
- Jordan (negara)?
- Jordan (nama depan Ibrani)?

Entity linking (EL) me-resolve tiap mention ke entri unik dalam knowledge base: Wikidata, Wikipedia, DBpedia, atau KB domainmu. Dua subtugas:

1. **Pembuatan kandidat.** Diberi "Jordan," entri KB mana yang masuk akal?
2. **Disambiguasi.** Diberi konteksnya, kandidat mana yang benar?

Kedua langkah bisa dipelajari. Keduanya di-benchmark. Pipeline gabungannya telah stabil selama satu dekade — yang berubah adalah kualitas disambiguator.

## Konsepnya

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**Pembuatan kandidat.** Diberi bentuk permukaan mention ("Jordan"), cari kandidat dalam indeks alias. Kamus alias Wikipedia mencakup kebanyakan named entity: "JFK" → John F. Kennedy, Jacqueline Kennedy, bandara JFK, JFK (film). Indeks tipikal mengembalikan 10-30 kandidat per mention.

**Disambiguasi: tiga pendekatan.**

1. **Prior + konteks (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. Berhasil baik, cepat, tanpa training.
2. **Berbasis embedding (ESS / REL / Blink).** Encode mention + konteks. Encode deskripsi tiap kandidat. Pilih kosinus maks. Default 2020-2024.
3. **Generatif (GENRE, 2021; berbasis LLM, 2023+).** Decode nama kanonis entitas token-per-token. Dikendalikan ke trie nama entitas valid sehingga output dijamin ID KB valid.

**End-to-end vs pipeline.** Model modern (ELQ, BLINK, ExtEnD, GENRE) menjalankan NER + pembuatan kandidat + disambiguasi dalam satu pass. Sistem pipeline masih mendominasi produksi karena kamu bisa menukar komponen.

### Dua pengukuran

- **Recall mention (pembuatan kandidat).** Fraksi mention emas di mana entri KB benar muncul dalam daftar kandidat. Lantai seluruh pipeline.
- **Akurasi / F1 disambiguasi.** Diberi kandidat benar, seberapa sering top-1 benar.

Selalu laporkan keduanya. Sistem dengan disambiguasi 99% pada recall kandidat 80% adalah pipeline 80%.

```figure
gx-entity-linking
```

## Bangun Sendiri

### Langkah 1: bangun indeks alias dari redirect Wikipedia

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Data alias Wikipedia: ~18M pasangan (alias, entitas). Unduh dari dump Wikidata. Simpan sebagai inverted index.

### Langkah 2: disambiguasi berbasis konteks

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Overlap Jaccard adalah mainan. Ganti dengan kemiripan kosinus pada embedding (lihat langkah-2 `code/main.py` untuk versi transformer).

### Langkah 3: berbasis embedding (ala BLINK)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

Saat indeks, embed setiap entitas KB sekali. Saat query, embed mention + konteks sekali, dot-product terhadap pool kandidat, pilih maks.

### Langkah 4: entity linking generatif (konsep)

GENRE men-decode judul Wikipedia entitas karakter-per-karakter. Decoding terkendali (lihat pelajaran 20) memastikan hanya judul valid yang bisa dikeluarkan. Integrasi erat dengan trie dukungan-KB. Keturunan modernnya adalah REL-GEN dan EL prompt-LLM dengan output terstruktur.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

Dikombinasikan dengan whitelist (Outlines `choice`), ini pipeline EL paling sederhana untuk dikirim di 2026.

### Langkah 5: evaluasi pada AIDA-CoNLL

AIDA-CoNLL adalah benchmark EL standar: 1.393 artikel Reuters, 34k mention, entitas Wikipedia. Laporkan akurasi in-KB (`P@1`) dan tingkat deteksi-NIL out-of-KB.

## Jebakan

- **Penanganan NIL.** Sebagian mention tidak ada di KB (entitas baru, orang tak dikenal). Sistem harus memprediksi NIL alih-alih menebak entitas salah. Diukur terpisah.
- **Error batas mention.** NER hulu melewatkan span parsial ("Bank of America" ditandai hanya "Bank"). Recall EL turun.
- **Bias popularitas.** Sistem terlatih over-prediksi entitas sering. Mention "Michael I. Jordan" pada paper ML sering tertaut ke Jordan si pemain basket.
- **EL lintas-lingual.** Memetakan mention dalam teks Mandarin ke entitas Wikipedia Inggris. Butuh encoder multilingual atau langkah penerjemahan.
- **KB basi.** Perusahaan, kejadian, orang baru tidak ada di dump Wikipedia tahun lalu. Pipeline produksi butuh loop refresh.

## Gunakan

Stack 2026:

| Situation | Pick |
|-----------|------|
| General-purpose English + Wikipedia | BLINK or REL |
| Cross-lingual, KB = Wikipedia | mGENRE |
| LLM-friendly, few mentions/day | Prompt Claude/GPT-4 with candidate list + constrained JSON |
| Domain-specific KB (medical, legal) | Custom BERT with KB-aware retrieval + fine-tune on domain AIDA-style set |
| Extremely low-latency | Exact-match prior only (Milne-Witten baseline) |
| Research SOTA | GENRE / ExtEnD / generative LLM-EL |

Pola produksi yang dikirim di 2026: NER → coref → EL pada tiap mention → runtuhkan klaster ke satu entitas kanonis per klaster. Output: satu ID KB per entitas dalam dokumen, bukan satu per mention.

## Rilis

Simpan sebagai `outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## Latihan

1. **Mudah.** Implementasikan disambiguator prior+konteks di `code/main.py` pada 10 mention ambigu (Paris, Jordan, Apple). Labeli entitas benar dengan tangan. Ukur akurasi.
2. **Sedang.** Encode 50 mention ambigu dengan sentence transformer. Embed deskripsi tiap kandidat. Bandingkan disambiguasi berbasis embedding dengan overlap konteks Jaccard.
3. **Sulit.** Bangun KB domain 1k-entitas (mis. karyawan + produk di perusahaanmu). Implementasikan NER + EL end-to-end. Ukur presisi dan recall pada 100 kalimat held-out.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Entity linking (EL) | Link to Wikipedia | Map a mention to a unique KB entry. |
| Candidate generation | Who could it be? | Return a shortlist of plausible KB entries for a mention. |
| Disambiguation | Pick the right one | Score candidates using context, pick the winner. |
| Alias index | The lookup table | Map from surface form → candidate entities. |
| NIL | Not in KB | Explicit prediction that no KB entry matches. |
| KB | Knowledge base | Wikidata, Wikipedia, DBpedia, or your domain KB. |
| AIDA-CoNLL | The benchmark | 1,393 Reuters articles with gold entity links. |

## Bacaan Lanjutan

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://researchcommons.waikato.ac.nz/entities/publication/b9a0b520-abc5-47c5-a86a-da6c579893ab) — the foundational prior+context approach.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — the embedding-based workhorse.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — generative EL with constrained decoding.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — the benchmark paper.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — the open production stack.
