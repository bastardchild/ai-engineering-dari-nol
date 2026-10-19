# Pengenalan Entitas Bernama

> Tarik keluar nama-namanya. Terdengar mudah sampai kamu berhadapan dengan batas ambigu, entitas bersarang, dan jargon domain.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 minutes

## Masalahnya

"Apple sued Google over its iPhone search deal in the US." Lima entitas: Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (mungkin), US (GPE). Sistem NER yang bagus mengekstrak semuanya dengan tipe benar. Yang buruk melewatkan iPhone, mengacaukan Apple si buah dengan Apple si perusahaan, dan melabeli "US" sebagai PERSON.

NER adalah kuda pekerja di balik setiap pipeline ekstraksi terstruktur. Parsing resume, pemindaian log compliance, anonimisasi rekam medis, pemahaman query pencarian, grounding untuk respons chatbot, ekstraksi kontrak legal. Kamu tak pernah benar-benar melihatnya; kamu selalu bergantung padanya.

Pelajaran ini menempuh jalur klasik (berbasis aturan, HMM, CRF) menuju yang modern (BiLSTM-CRF, lalu transformer). Setiap langkah memecahkan keterbatasan spesifik langkah sebelumnya. Polanya adalah pelajarannya.

## Konsepnya

**BIO tagging** (atau BILOU) mengubah ekstraksi entitas menjadi masalah pelabelan sekuens. Labeli tiap token dengan `B-TYPE` (awal entitas), `I-TYPE` (dalam entitas), atau `O` (di luar entitas mana pun).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Entitas multi-token dirangkai: `New B-GPE`, `York I-GPE`, `City I-GPE`. Model yang memahami BIO bisa mengekstrak span sembarang.

Progresi arsitekturnya:

- **Berbasis aturan.** Regex + lookup gazetteer. Presisi tinggi pada entitas dikenal, nol cakupan pada yang baru.
- **HMM.** Hidden Markov Model. Probabilitas emisi token diberi tag, probabilitas transisi tag-ke-tag. Decode Viterbi. Dilatih pada data berlabel.
- **CRF.** Conditional Random Field. Seperti HMM tetapi diskriminatif, sehingga kamu bisa mencampur fitur sembarang (bentuk kata, kapitalisasi, kata tetangga). Masih kuda pekerja produksi klasik di 2026 untuk deployment sumber daya rendah.
- **BiLSTM-CRF.** Fitur neural alih-alih buatan tangan. LSTM membaca kalimat dua arah, layer CRF di atasnya menegakkan sekuens tag yang konsisten.
- **Berbasis transformer.** Fine-tune BERT dengan head klasifikasi-token. Akurasi terbaik. Komputasi terbesar.

```figure
ner-bio-tagging
```

## Bangun Sendiri

### Langkah 1: helper BIO tagging

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Langkah 2: fitur buatan tangan

Untuk NER klasik (non-neural), fitur adalah permainannya. Yang berguna:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` mengembalikan `xXxxxx`. `word_shape("USA-2024")` mengembalikan `XXX-dddd`. Pola kapitalisasi adalah sinyal kuat untuk proper noun.

### Langkah 3: baseline aturan + kamus sederhana

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Gazetteer produksi punya jutaan entri yang di-scrape dari Wikipedia dan DBpedia. Cakupannya bagus. Disambiguasi (`Apple` si perusahaan vs si buah) buruk. Itulah alasan model statistik menang.

### Langkah 4: langkah CRF (sketsa, bukan impl penuh)

CRF penuh dari nol dalam 50 baris tidak mencerahkan tanpa fondasi teori probabilitas. Pakai `sklearn-crfsuite` saja:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` dan `c2` adalah regularisasi L1 dan L2. `all_possible_transitions=True` membiarkan model mempelajari sekuens ilegal (mis. `I-ORG` setelah `O`) itu improbable, begitulah CRF menegakkan konsistensi BIO tanpa kamu menulis konstrainnya.

### Langkah 5: apa yang ditambahkan BiLSTM-CRF

Fitur menjadi terpelajari. Input: embedding token (GloVe atau fastText). LSTM membaca kiri-ke-kanan dan kanan-ke-kiri. Hidden state gabungan masuk ke layer output CRF. CRF tetap menegakkan konsistensi sekuens-tag; LSTM menggantikan fitur buatan tangan dengan yang terpelajari.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

Untuk layer CRF, pakai `torchcrf.CRF` (pip install pytorch-crf). Gain atas CRF buatan tangan terukur tetapi lebih kecil dari dugaan kecuali kamu punya puluhan ribu kalimat berlabel.

## Gunakan

spaCy menyediakan NER setingkat produksi secara default.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

Perhatikan `iPhone` dilabeli `ORG` alih-alih `PRODUCT` — model kecil spaCy punya cakupan entitas-produk yang lemah. Model besar (`en_core_web_lg`) lebih baik. Model transformer (`en_core_web_trf`) lebih baik lagi.

Hugging Face untuk NER berbasis BERT:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` menggabungkan token B-X, I-X yang berurutan menjadi span. Tanpanya, kamu mendapat label level-token dan harus menggabungkannya sendiri.

### NER berbasis LLM (opsi 2026)

NER zero-shot dan few-shot LLM kini kompetitif dengan model fine-tune di banyak domain, dan jauh lebih baik ketika data berlabel langka.

- **Prompting zero-shot.** Beri LLM daftar tipe entitas dan skema contoh. Minta output JSON. Berhasil langsung; akurasinya sedang pada domain baru.
- **Prompting ala ZeroTuneBio.** Uraikan tugas menjadi ekstraksi kandidat → penjelasan makna → penilaian → pengecekan ulang. Prompt multi-tahap (bukan one-shot) mengangkat akurasi secara substansial pada NER biomedis. Pola yang sama berlaku untuk domain legal, finansial, dan saintifik.
- **Prompting dinamis dengan RAG.** Ambil contoh berlabel paling mirip dari seed set kecil beranotasi untuk setiap panggilan inference; bangun prompt few-shot secara on the fly. Dalam benchmark 2026, ini mengangkat F1 NER biomedis GPT-4 sebesar 11-12% atas prompting statis.
- **Dekomposisi per tipe entitas.** Untuk dokumen panjang, satu panggilan yang mengekstrak semua tipe entitas sekaligus kehilangan recall seiring bertambahnya panjang. Jalankan satu pass ekstraksi per tipe entitas. Biaya inference lebih tinggi, akurasi jauh lebih tinggi. Ini pola standar untuk catatan klinis dan kontrak legal.

Rekomendasi produksi per 2026: mulai dengan baseline zero-shot LLM sebelum kamu mengumpulkan data training. Sering kali F1-nya cukup baik sehingga kamu tak perlu fine-tune.

### Di mana NER klasik masih menang

Bahkan dengan LLM tersedia, NER klasik menang ketika:

- Budget latensi di bawah 50ms.
- Kamu punya ribuan contoh berlabel dan butuh F1 98%+.
- Domain punya ontologi stabil di mana CRF atau BiLSTM pra-terlatih mentransfer dengan baik.
- Konstrain regulasi menuntut model on-prem yang non-generatif.

### Di mana ia runtuh

- **Pergeseran domain.** NER terlatih CoNLL pada kontrak legal performanya lebih buruk daripada gazetteer. Fine-tune pada domainmu.
- **Entitas bersarang.** "Bank of America Tower" sekaligus ORG dan FACILITY. BIO standar tidak bisa merepresentasikan span tumpang tindih. Kamu butuh NER bersarang (multi-pass atau model berbasis span).
- **Entitas panjang.** "United States Federal Deposit Insurance Corporation." Model level-token kadang memecahnya. Pakai `aggregation_strategy` atau pasca-proses.
- **Tipe jarang.** Label NER medis seperti DRUG_BRAND, ADVERSE_EVENT, DOSE. Model tujuan umum tidak tahu. Scispacy dan BioBERT adalah titik awalnya.

## Rilis

Simpan sebagai `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## Latihan

1. **Mudah.** Implementasikan `bio_to_spans` (kebalikan dari `spans_to_bio`) dan verifikasi konsistensi round-trip pada 10 kalimat.
2. **Sedang.** Latih CRF sklearn-crfsuite di atas pada dataset NER Inggris CoNLL-2003. Laporkan F1 per entitas memakai `seqeval`. Hasil tipikal: ~84 F1.
3. **Sulit.** Fine-tune `distilbert-base-cased` pada dataset NER spesifik-domain (medis, legal, atau finansial). Bandingkan dengan model kecil spaCy. Dokumentasikan cek kebocoran data dan tulis apa yang mengejutkanmu.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| NER | Extract names | Label token spans with types (PERSON, ORG, GPE, DATE, ...). |
| BIO | Tagging scheme | `B-X` begins, `I-X` continues, `O` outside. |
| BILOU | Better BIO | Adds `L-X` (last), `U-X` (unit) for cleaner boundaries. |
| CRF | Structured classifier | Models transitions between labels, not just emissions. Enforces valid sequences. |
| Nested NER | Overlapping entities | One span is a different entity than a sub-span of it. BIO cannot express this. |
| Entity-level F1 | Proper NER metric | Predicted span must match true span exactly. Token-level F1 overstates accuracy. |

## Bacaan Lanjutan

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — the BiLSTM-CRF paper. Canonical.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — introduces the token-classification pattern that became standard.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — practical reference for every attribute on `Doc.ents` and `Span`.
- [seqeval](https://github.com/chakki-works/seqeval) — the correct metric library. Use it always.
