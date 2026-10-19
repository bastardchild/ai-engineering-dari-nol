# POS Tagging dan Parsing Sintaksis

> Gramatika sempat tidak modis untuk sementara. Lalu setiap pipeline LLM perlu memvalidasi ekstraksi terstruktur, dan ia kembali.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## Masalahnya

Pelajaran 01 menjanjikan bahwa lemmatisasi butuh part-of-speech tag. Tanpa tahu `running` adalah verba, lemmatizer tidak bisa mereduksinya menjadi `run`. Tanpa tahu `better` adalah adjektiva, ia tidak bisa mereduksi menjadi `good`.

Janji itu menyembunyikan seluruh subbidang. Part-of-speech tagging menetapkan kategori gramatikal. Parsing sintaksis memulihkan struktur pohon kalimat: kata mana memodifikasi mana, verba mana mengatur argumen mana. NLP klasik menghabiskan dua puluh tahun menyempurnakan keduanya. Lalu deep learning meruntuhkannya menjadi tugas klasifikasi-token di atas transformer pra-terlatih, dan komunitas riset move on.

Bukan komunitas terapan. Setiap pipeline ekstraksi terstruktur masih memakai POS dan dependency tree di balik layar. JSON yang dihasilkan LLM divalidasi terhadap konstrain gramatikal. Sistem penjawab pertanyaan mengurai query memakai dependency parse. Evaluator kualitas terjemahan mesin memeriksa kesejajaran parse tree.

Layak diketahui. Pelajaran ini mengenalkan tagset, baseline, dan titik di mana kamu berhenti mengimplementasikan dari nol dan memanggil spaCy.

## Konsepnya

**POS tagging** melabeli tiap token dengan kategori gramatikal. Tagset **Penn Treebank (PTB)** adalah default Inggris. 36 tag dengan pembedaan yang menurut pembaca awam terlalu rinci: `NN` nomina tunggal, `NNS` nomina jamak, `NNP` proper noun tunggal, `VBD` verba past tense, `VBZ` verba present orang ketiga tunggal, dan seterusnya. Tagset **Universal Dependencies (UD)** lebih kasar (17 tag) dan agnostik bahasa; ia menjadi default untuk kerja lintas-lingual.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Parsing sintaksis** menghasilkan pohon. Dua gaya utama:

- **Constituency parsing.** Frasa nomina, frasa verba, frasa preposisional saling bersarang. Output adalah pohon kategori non-terminal (NP, VP, PP) dengan kata sebagai daun.
- **Dependency parsing.** Setiap kata punya satu kata head yang menjadi sandarannya, dilabeli relasi gramatikal. Output adalah pohon di mana setiap edge adalah triple (head, dependent, relation).

Dependency parsing menang di tahun 2010-an karena menggeneralisasi bersih lintas bahasa, terutama yang urutan katanya bebas.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## Bangun Sendiri

### Langkah 1: baseline tag tersering

POS tagger paling bodoh yang berhasil. Untuk setiap kata, prediksi tag yang paling sering ia sandang saat training.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Pada korpus Brown, baseline ini mencapai akurasi ~85%. Tidak bagus, tetapi lantai yang tak boleh ditembus model serius mana pun.

### Langkah 2: tagger HMM bigram

Modelkan probabilitas gabungan sekuens:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

Dua tabel: probabilitas transisi (tag diberi tag sebelumnya), probabilitas emisi (kata diberi tag). Estimasi keduanya dari hitungan dengan smoothing Laplace. Decode dengan Viterbi (pemrograman dinamis atas lattice tag).

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

HMM bigram pada Brown mencapai akurasi ~93%. Lompatan dari 85% ke 93% sebagian besar berasal dari probabilitas transisi — model mempelajari `DET NOUN` itu umum dan `NOUN DET` itu jarang.

### Langkah 3: mengapa tagger modern mengalahkannya

Probabilitas transisi + emisi itu lokal. Keduanya tidak bisa menangkap bahwa `saw` adalah nomina dalam "I bought a saw" tetapi verba dalam "I saw the movie." CRF dengan fitur sembarang (sufiks, bentuk kata, kata sebelum dan sesudah, kata itu sendiri) mencapai ~97%. BiLSTM-CRF atau transformer mencapai ~98%+.

Langit-langit tugas ini ditentukan oleh ketidaksepakatan anotator. Anotator manusia sepakat sekitar 97% pada Penn Treebank. Model di atas 98% mungkin overfit pada test set.

### Langkah 4: sketsa dependency parsing

Dependency parsing penuh dari nol di luar cakupan; perlakuan buku teks kanonis ada di Jurafsky and Martin. Dua keluarga klasik yang perlu diketahui:

- Parser **berbasis transisi** (arc-eager, arc-standard) bertindak seperti parser shift-reduce: membaca token, menggesernya ke stack, dan menerapkan aksi reduce yang menciptakan arc. Decoding greedy itu cepat. Implementasi klasiknya MaltParser. Versi neural modern: parser berbasis transisi Chen and Manning.
- Parser **berbasis graf** (algoritme Eisner, biaffine Dozat-Manning) memberi skor setiap edge head-dependent yang mungkin dan memilih maximum spanning tree. Lebih lambat tetapi lebih akurat.

Untuk kebanyakan kerja terapan, panggil spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

Baca kolom `dep` dari bawah ke atas dan struktur gramatikal kalimat pun terungkap.

## Gunakan

Setiap pustaka NLP produksi menyediakan parser POS dan dependensi sebagai bagian pipeline standar.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). Cepat, akurat, terintegrasi dengan tokenisasi + NER + lemmatisasi. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (relasi dependensi).
- **Stanford NLP (stanza)**. Penerus CoreNLP dari Stanford. State-of-the-art pada 60+ bahasa.
- **trankit**. Berbasis transformer, akurasi UD bagus.
- **NLTK**. `pos_tag`. Bisa dipakai, lambat, lawas. Baik untuk pengajaran.

### Di mana ini masih penting di 2026

- **Lemmatisasi.** Pelajaran 01 butuh POS untuk melemmatisasi dengan benar. Selalu.
- **Ekstraksi terstruktur dari output LLM.** Validasi bahwa kalimat yang dihasilkan mematuhi konstrain gramatikal (mis. kesesuaian subjek-verba, modifier wajib).
- **Sentimen berbasis aspek.** Dependency parse memberi tahu adjektiva mana memodifikasi nomina mana.
- **Pemahaman query.** "movies directed by Wes Anderson starring Bill Murray" terurai menjadi konstrain terstruktur via parse.
- **Transfer lintas-lingual.** Tag UD dan relasi dependensi agnostik bahasa, memungkinkan analisis terstruktur zero-shot atas bahasa baru.
- **Pipeline komputasi rendah.** Jika kamu tidak bisa mengirim transformer, POS + dependency parse + gazetteer membawamu jauh secara mengejutkan.

## Rilis

Simpan sebagai `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## Latihan

1. **Mudah.** Dengan baseline tag-tersering pada korpus kecil bertag (mis. subset Brown NLTK), ukur akurasi pada kalimat held-out. Verifikasi hasil ~85%.
2. **Sedang.** Latih HMM bigram di atas dan laporkan presisi/recall per tag. Tag mana yang paling sering dikacaukan HMM?
3. **Sulit.** Pakai dependency parse spaCy untuk mengekstrak triple subjek-verba-objek dari sampel 1000 kalimat. Evaluasi pada 50 triple berlabel manual. Dokumentasikan di mana ekstraksi gagal (seringnya pasif, koordinasi, dan subjek elided).

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| POS tag | Word's type | Grammatical category. PTB has 36; UD has 17. |
| Penn Treebank | Standard tagset | English-specific. Fine-grained verb tenses and noun number. |
| Universal Dependencies | Multilingual tagset | Coarser than PTB; language-neutral; defaults for cross-lingual work. |
| Dependency parse | Sentence tree | Each word has one head, each edge has a grammatical relation. |
| Viterbi | Dynamic programming | Finds the highest-probability tag sequence given emissions and transitions. |

## Bacaan Lanjutan

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — the canonical textbook treatment of POS and parsing.
- [Universal Dependencies project](https://universaldependencies.org/) — the cross-lingual tagset and treebank collection used by every multilingual parser.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — practical reference for every attribute exposed on `Token`.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — the paper that brought neural parsers into the mainstream.
