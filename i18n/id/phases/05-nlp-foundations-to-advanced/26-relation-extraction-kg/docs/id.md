# Ekstraksi Relasi & Konstruksi Knowledge Graph

> NER menemukan entitasnya. Entity linking menjangkarkannya. Ekstraksi relasi menemukan edge di antaranya. Knowledge graph adalah jumlah node, edge, dan provenance-nya.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 minutes

## Masalahnya

Seorang analis membaca: "Tim Cook became CEO of Apple in 2011." Empat fakta:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Ekstraksi Relasi (RE) mengubah teks bebas menjadi triple terstruktur `(subject, relation, object)`. Agregat lintas korpus dan kamu punya knowledge graph. Agregat dan query dan kamu punya substrat penalaran untuk RAG, analitik, atau audit compliance.

Masalah 2026: LLM mengekstrak relasi dengan antusias. Terlalu antusias. Mereka menghalusinasikan triple yang tidak didukung teks sumber. Tanpa provenance, kamu tak bisa membedakan triple asli dari fiksi yang masuk akal. Jawaban 2026 adalah pipeline anchor-and-verify ala AEVS.

## Konsepnya

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**Bentuk triple.** `(subject_entity, relation_type, object_entity)`. Relasi berasal dari ontologi tertutup (properti Wikidata, FIBO, UMLS) atau himpunan terbuka (ala OpenIE, apa pun boleh).

**Tiga pendekatan ekstraksi.**

1. **Berbasis aturan / pola.** Pola Hearst: "X such as Y" → `(Y, isA, X)`. Plus regex buatan tangan. Rapuh, presisi, dapat dijelaskan.
2. **Pengklasifikasi tersupervisi.** Diberi dua mention entitas dalam kalimat, prediksi relasi dari himpunan tetap. Dilatih pada TACRED, ACE, KBP. Standar 2015–2022.
3. **LLM generatif.** Prompt model untuk mengeluarkan triple. Berhasil langsung. Butuh provenance, atau menghalusinasikan sampah yang tampak masuk akal.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Framework mitigasi halusinasi kini:

- **Anchor.** Identifikasi setiap span entitas dan span frasa-relasi dengan posisi eksak.
- **Extract.** Hasilkan triple yang tertaut ke span anchor.
- **Verify.** Cocokkan tiap elemen triple kembali ke teks sumber; tolak apa pun yang tak didukung.
- **Supplement.** Pass cakupan memastikan tak ada span ber-anchor yang terbuang.

Halusinasi turun tajam. Butuh komputasi lebih tetapi bisa diaudit.

**Tradeoff terbuka-vs-tertutup.**

- **Ontologi tertutup.** Daftar properti tetap (mis. 11.000+ properti Wikidata). Terprediksi. Bisa di-query. Sulit mengarang.
- **Open IE.** Frasa verbal apa pun menjadi relasi. Recall tinggi. Presisi rendah. Berantakan di-query.

KG produksi biasanya mencampur: open IE untuk discovery, lalu kanonikalisasi relasi ke ontologi tertutup sebelum digabung ke graf utama.

```figure
relation-triples
```

## Bangun Sendiri

### Langkah 1: ekstraksi berbasis pola

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

Lihat `code/main.py` untuk ekstraktor mainan penuh. Pola Hearst masih dikirim dalam pipeline spesifik-domain karena bisa di-debug.

### Langkah 2: klasifikasi relasi tersupervisi

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL adalah ekstraktor relasi seq2seq: teks masuk, triple keluar, sudah dalam ID properti Wikidata. Di-fine-tune pada data distant-supervision. Baseline open-weights standar.

### Langkah 3: ekstraksi prompt-LLM dengan anchoring

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

Verifikasi setiap span yang dikembalikan terhadap sumber. Tolak apa pun di mana `text[start:end] != triple_entity`. Ini langkah "verify" AEVS dalam bentuk minimalnya.

### Langkah 4: kanonikalisasi ke ontologi tertutup

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Kanonikalisasi sering 60-80% dari kerja engineering. Anggarkan untuk itu.

### Langkah 5: bangun graf kecil dan query

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

Ini atom dari setiap sistem RAG-over-KG. Skalakan dengan triple store RDF (Blazegraph, Virtuoso), property graph (Neo4j), atau graph store ter-augmentasi-vektor.

## Jebakan

- **Koreferensi sebelum RE.** "He founded Apple" — RE perlu tahu siapa "he". Jalankan coref dulu (pelajaran 24).
- **Kanonikalisasi entitas.** "Apple Inc" dan "Apple" harus resolve ke node yang sama. Entity linking dulu (pelajaran 25).
- **Triple terhalusinasi.** LLM mengeluarkan triple yang tidak didukung teks. Tegakkan verifikasi span.
- **Drift kanonikalisasi relasi.** Relasi Open IE inkonsisten ("was born in," "came from," "is a native of"). Runtuhkan ke ID kanonis atau graf tak bisa di-query.
- **Error temporal.** "Tim Cook is CEO of Apple" — benar kini, salah di 2005. Banyak relasi berbatas waktu. Pakai qualifier (`P580` waktu mulai, `P582` waktu akhir di Wikidata).
- **Ketidakcocokan domain.** REBEL dilatih pada Wikipedia. Teks legal, medis, dan saintifik sering butuh model RE fine-tune-domain.

## Gunakan

Stack 2026:

| Situation | Pick |
|-----------|------|
| Fast production, general domain | REBEL or LlamaPred with Wikidata canonicalization |
| Domain-specific (biomed, legal) | SciREX-style domain fine-tune + custom ontology |
| LLM-prompted, audited output | AEVS pipeline: anchor → extract → verify → supplement |
| High-volume news IE | Pattern-based + supervised hybrid |
| Building a KG from scratch | Open IE + manual canonicalization pass |
| Temporal KG | Extract with qualifiers (start/end time, point in time) |

Pola integrasinya: NER → coref → entity linking → ekstraksi relasi → pemetaan ontologi → pemuatan graf. Setiap tahap adalah gerbang kualitas potensial.

## Rilis

Simpan sebagai `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## Latihan

1. **Mudah.** Jalankan ekstraktor pola di `code/main.py` pada 5 kalimat artikel berita. Cek presisi dengan tangan.
2. **Sedang.** Pakai REBEL (atau LLM kecil) pada kalimat yang sama. Bandingkan triple. Ekstraktor mana yang presisinya lebih tinggi? Recall-nya lebih tinggi?
3. **Sulit.** Bangun pipeline AEVS: ekstrak dengan LLM + verifikasi span terhadap sumber. Ukur tingkat halusinasi sebelum vs sesudah langkah verify pada 50 kalimat ala Wikipedia.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Triple | Subject-relation-object | `(s, r, o)` tuple that is the atomic unit of a KG. |
| Open IE | Extract anything | Open-vocabulary relation phrases; high recall, low precision. |
| Closed ontology | Fixed schema | Bounded set of relation types (Wikidata, UMLS, FIBO). |
| Canonicalization | Normalize everything | Map surface names / relations to canonical ids. |
| AEVS | Grounded extraction | Anchor-Extraction-Verification-Supplement pipeline (2026). |
| Provenance | Source-of-truth link | Every triple carries a doc id + char-span to its source. |
| Distant supervision | Cheap labels | Align text with an existing KG to create training data. |

## Bacaan Lanjutan

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — the distant-supervision paper.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE workhorse.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — joint IE.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 hallucination-mitigation design.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — canonical graph queries.
