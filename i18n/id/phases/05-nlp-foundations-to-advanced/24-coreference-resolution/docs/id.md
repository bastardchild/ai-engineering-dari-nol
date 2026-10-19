# Resolusi Koreferensi

> "She called him. He did not answer. The doctor was at lunch." Tiga referensi ke dua orang dan tak ada yang disebut namanya. Resolusi koreferensi mencari tahu siapa itu siapa.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 minutes

## Masalahnya

Ekstrak setiap mention Apple Inc. dari artikel 300-kata. Mudah ketika artikel bilang "Apple." Sulit ketika ia bilang "the company," "they," "Cupertino's technology giant," atau "Jobs's firm." Tanpa me-resolve mention ini ke entitas yang sama, pipeline NER-mu melewatkan 60-80% mention.

Resolusi koreferensi menautkan setiap ekspresi yang merujuk entitas dunia-nyata yang sama ke dalam satu klaster. Ia adalah lem antara NLP level-permukaan (NER, parsing) dan semantik downstream (IE, QA, peringkasan, KG).

Mengapa penting di 2026:

- Peringkasan: "The CEO announced..." vs "Tim Cook announced..." — ringkasannya harus menyebut nama CEO.
- Penjawab pertanyaan: "Who did she call?" menuntut me-resolve "she."
- Ekstraksi informasi: knowledge graph dengan "PER1 founded Apple" dan "Jobs founded Apple" sebagai entri terpisah itu salah.
- IE multi-dokumen: menggabungkan mention lintas artikel tentang kejadian yang sama adalah koreferensi lintas-dokumen.

## Konsepnya

![Coreference clustering: mentions → entities](../assets/coref.svg)

**Tugasnya.** Input: dokumen. Output: clustering mention (span) di mana tiap klaster merujuk satu entitas.

**Tipe mention.**

- **Named entity.** "Tim Cook"
- **Nominal.** "the CEO", "the company"
- **Pronominal.** "he", "she", "they", "it"
- **Apositif.** "Tim Cook, Apple's CEO,"

**Arsitektur.**

1. **Berbasis aturan (Hobbs, 1978).** Resolusi pronomina berbasis pohon sintaksis memakai aturan gramatika. Baseline bagus. Mengejutkan sulit dikalahkan pada pronomina.
2. **Pengklasifikasi pasangan-mention.** Untuk setiap pasangan mention (m_i, m_j), prediksi apakah keduanya korefer. Kelompokkan via transitive closure. Standar pra-2016.
3. **Pemeringkat-mention.** Untuk tiap mention, rangking kandidat anteseden (termasuk "tanpa anteseden"). Pilih teratas.
4. **Span-based end-to-end (Lee et al., 2017).** Encoder transformer. Enumerasi semua span kandidat sampai batas panjang. Prediksi skor mention. Prediksi probabilitas-anteseden untuk tiap span. Kelompokkan greedy. Default modern.
5. **Generatif (2024+).** Prompt LLM: "List every pronoun in this text and its antecedent." Bagus pada kasus mudah, kesulitan pada dokumen panjang dan referen jarang.

**Metrik evaluasi.** Lima metrik standar (MUC, B³, CEAF, BLANC, LEA) karena tak ada satu metrik yang menangkap kualitas clustering. Laporkan rata-rata tiga pertama sebagai CoNLL F1. State-of-the-art di 2026 pada CoNLL-2012: ~83 F1.

**Kasus sulit yang dikenal.**

- Deskripsi definit yang merujuk entitas yang diperkenalkan halaman-halaman sebelumnya.
- Anafora bridging ("the wheels" → mobil yang disebut sebelumnya).
- Anafora nol dalam bahasa seperti Mandarin dan Jepang.
- Katafora (pronomina sebelum referen): "When **she** walked in, Mary smiled."

```figure
coref-links
```

## Bangun Sendiri

### Langkah 1: koreferensi neural pra-terlatih (AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

Pada dokumen lebih panjang, kamu mendapat sesuatu seperti:
- Klaster 1: [Apple, The company, they]
- Klaster 2: [new products]

### Langkah 2: resolver pronomina berbasis aturan (pengajaran)

Lihat `code/main.py` untuk implementasi stdlib-only:

1. Ekstrak mention: named entity (span kapital), pronomina (lookup dict), deskripsi definit ("the X").
2. Untuk tiap pronomina, lihat K mention sebelumnya dan skor berdasarkan:
   - kesesuaian gender/jumlah (heuristik)
   - kebaruan (yang lebih dekat menang)
   - peran sintaksis (subjek diprioritaskan)
3. Tautkan anteseden berskor tertinggi.

Tidak kompetitif dengan model neural. Tetapi menunjukkan ruang pencarian dan keputusan yang harus dibuat model end-to-end.

### Langkah 3: memakai LLM untuk koreferensi

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

Dua mode kegagalan yang perlu diwaspadai. Pertama, LLM over-merge ("him" dan "her" yang merujuk dua orang berbeda). Kedua, LLM diam-diam membuang mention dalam dokumen panjang. Selalu verifikasi dengan cek span-offset.

### Langkah 4: evaluasi

Skrip standar conll-2012 menghitung MUC, B³, CEAF-φ4 dan melaporkan rata-ratanya. Untuk eval in-house, mulai dengan presisi dan recall level-span pada test set beranotasi, lalu tambah F1 mention-linking.

## Jebakan

- **Ledakan singleton.** Sebagian sistem melaporkan setiap mention sebagai klasternya sendiri. B³ toleran. MUC menghukum ini. Selalu cek ketiga metrik.
- **Pronomina dalam konteks panjang.** Performa turun ~15 F1 pada dokumen di atas 2.000 token. Chunk dengan hati-hati.
- **Asumsi gender.** Aturan gender hard-code patah pada referen non-biner, organisasi, hewan. Pakai model terpelajari atau skoring netral.
- **Drift LLM pada dok panjang.** Satu panggilan API tidak bisa andal mengelompokkan mention lintas 50+ paragraf. Pakai sliding-window + merge.

## Gunakan

Stack 2026:

| Situation | Pick |
|-----------|------|
| English, single document | `en_coreference_web_trf` (spaCy-experimental) or AllenNLP neural coref |
| Multilingual | SpanBERT / XLM-R trained on OntoNotes or Multilingual CoNLL |
| Cross-document event coref | Specialized end-to-end models (2025–26 SOTA) |
| Quick LLM baseline | GPT-4o / Claude with structured-output coref prompt |
| Production dialog systems | Rule-based fallback + neural primary + manual review for critical slots |

Pola integrasi yang dikirim di 2026: jalankan NER dulu, jalankan coref, gabungkan klaster coref ke entitas NER. Tugas downstream melihat satu entitas per klaster, bukan satu entitas per mention.

## Rilis

Simpan sebagai `outputs/skill-coref-picker.md`:

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## Latihan

1. **Mudah.** Jalankan resolver berbasis aturan di `code/main.py` pada 5 paragraf buatan tangan. Ukur akurasi mention-link terhadap ground truth.
2. **Sedang.** Pakai model coref neural pra-terlatih pada artikel berita. Bandingkan klaster dengan anotasi manualmu sendiri. Di mana ia gagal?
3. **Sulit.** Bangun pipeline NER yang diperkaya-coref: NER dulu, lalu gabung via klaster coref. Ukur peningkatan cakupan-entitas vs NER-saja pada 100 artikel.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Mention | A reference | A span of text that refers to an entity (name, pronoun, noun phrase). |
| Antecedent | What "it" refers to | The earlier mention a later one corefers with. |
| Cluster | The entity's mentions | Set of mentions that all refer to the same real-world entity. |
| Anaphora | Backward reference | Later mention refers to earlier ("he" → "John"). |
| Cataphora | Forward reference | Earlier mention refers to later ("When he arrived, John..."). |
| Bridging | Implicit reference | "I bought a car. The wheels were bad." (wheels of THAT car.) |
| CoNLL F1 | The number on leaderboards | Average of MUC, B³, CEAF-φ4 F1 scores. |

## Bacaan Lanjutan

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — canonical textbook chapter.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — span-based end-to-end.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — pretraining that improves coref.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — the benchmark.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — the rule-based classic.
