# Pemrosesan Teks — Tokenisasi, Stemming, Lemmatisasi

> Bahasa itu kontinu. Model itu diskret. Pra-pemrosesan adalah jembatannya.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## Masalahnya

Sebuah model tidak bisa membaca "The cats were running." Ia membaca bilangan bulat.

Setiap sistem NLP dibuka dengan tiga pertanyaan yang sama. Di mana sebuah kata dimulai. Apa akar kata tersebut. Bagaimana memperlakukan "run", "running", "ran" sebagai hal yang sama ketika itu membantu, dan sebagai hal yang berbeda ketika tidak membantu.

Kalau tokenisasi salah, model belajar dari sampah. Jika tokenizermu memperlakukan `don't` sebagai satu token tetapi `do n't` sebagai dua token, distribusi training terbelah. Jika stemmermu menggabungkan `organization` dan `organ` menjadi stem yang sama, pemodelan topik mati. Jika lemmatizermu butuh konteks part-of-speech tetapi kamu tidak memberikannya, verba diperlakukan sebagai nomina.

Pelajaran ini membangun ketiga langkah pra-pemrosesan dari nol, lalu menunjukkan bagaimana NLTK dan spaCy melakukan pekerjaan yang sama supaya kamu bisa melihat tradeoff-nya.

## Konsepnya

Tiga operasi. Masing-masing punya tugas dan mode kegagalan.

**Tokenisasi** memecah string menjadi token. "Token" sengaja dibuat samar karena granularitas yang tepat tergantung tugasnya. Level kata untuk NLP klasik. Subword untuk transformer. Karakter untuk bahasa tanpa spasi.

**Stemming** memangkas sufiks dengan aturan. Cepat, agresif, bodoh. `running -> run`. `organization -> organ`. Yang kedua itu adalah mode kegagalannya.

**Lemmatisasi** mereduksi kata ke bentuk kamusnya memakai pengetahuan gramatika. Lebih lambat, akurat, butuh tabel lookup atau penganalisis morfologi. `ran -> run` (perlu tahu "ran" adalah bentuk lampau dari "run"). `better -> good` (perlu tahu bentuk komparatif).

Aturan praktis. Pakai stemming ketika kecepatan penting dan kamu bisa mentoleransi noise (pengindeksan pencarian, klasifikasi kasar). Pakai lemmatisasi ketika makna penting (penjawab pertanyaan, pencarian semantik, apa pun yang akan dibaca pengguna).

```figure
edit-distance
```

## Bangun Sendiri

### Langkah 1: tokenizer kata regex

Tokenizer paling sederhana yang berguna memecah berdasarkan karakter non-alfanumerik sambil mempertahankan tanda baca sebagai token tersendiri. Tidak sempurna, tidak final, tetapi berjalan dalam satu baris.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

Tiga pola sesuai urutan prioritas. Kata dengan apostrof dalam opsional (`don't`, `it's`). Angka murni. Setiap satu karakter non-spasi non-alfanumerik sebagai token mandiri (tanda baca).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

Mode kegagalan yang perlu diperhatikan. `3pm` terpecah menjadi `['3', 'pm']` karena kita bergantian antara rangkaian huruf dan rangkaian digit. Cukup baik untuk kebanyakan tugas. URL, email, hashtag semuanya rusak. Untuk produksi, tambahkan pola sebelum pola umum.

### Langkah 2: stemmer Porter (hanya langkah 1a)

Algoritme Porter penuh punya lima fase aturan. Langkah 1a saja sudah mencakup sufiks Inggris yang paling sering muncul dan mengajarkan polanya.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

Baca aturannya dari atas ke bawah. Aturan `ies -> i` adalah alasan `ponies -> poni`, bukan `pony`. Porter asli punya langkah 1b yang akan memperbaikinya. Aturan-aturan saling bersaing. Aturan yang lebih awal menang. Urutannya lebih penting daripada aturan tunggal mana pun.

### Langkah 3: lemmatizer berbasis lookup

Lemmatisasi yang benar butuh morfologi. Versi pengajaran yang terkendali memakai tabel lemma kecil dan fallback.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

Kasus terakhir adalah momen pengajaran kuncinya. `watched` tidak ada di tabel kita dan fallback kita hanya menangani `ing`. Lemmatisasi asli mencakup `ed`, verba tak beraturan, adjektiva komparatif, jamak dengan perubahan bunyi (`children -> child`). Inilah alasan sistem produksi memakai WordNet, morphologizer spaCy, atau penganalisis morfologi penuh.

### Langkah 4: rangkai semuanya

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

Bagian yang hilang adalah POS tagger. Fase 5 · 07 (POS Tagging) membangunnya. Untuk sekarang, default-kan semuanya ke `NOUN` dan akui keterbatasannya.

## Gunakan

NLTK dan spaCy menyediakan versi produksinya. Masing-masing hanya beberapa baris.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` menangani kontraksi, Unicode, dan edge case yang terlewat regex-mu. `PorterStemmer` menjalankan kelima fase. `WordNetLemmatizer` butuh POS tag diterjemahkan dari skema Penn Treebank NLTK ke himpunan singkatan WordNet. Pengkabelan penerjemah di atas adalah bagian yang paling sering dilewatkan tutorial.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy menyembunyikan seluruh pipeline di balik `nlp(text)`. Tokenisasi, POS tagging, dan lemmatisasi semuanya berjalan. Lebih cepat daripada NLTK dalam skala besar. Lebih akurat secara default. Tradeoff-nya adalah kamu tidak bisa dengan mudah menukar komponen individual.

### Kapan memilih yang mana

| Situation | Pick |
|-----------|------|
| Teaching, research, swapping components | NLTK |
| Production, multi-language, speed matters | spaCy |
| Transformer pipeline (you'll tokenize with the model's tokenizer anyway) | Use `tokenizers` / `transformers` and skip classical preprocessing |

### Dua mode kegagalan yang tak pernah diperingatkan siapa pun

Kebanyakan tutorial mengajarkan algoritmenya lalu berhenti. Dua hal akan menggigit pipeline pra-pemrosesan nyata, dan keduanya hampir tak pernah dibahas.

**Drift reproduksibilitas.** NLTK dan spaCy mengubah perilaku tokenisasi dan lemmatizer antar versi. Apa yang menghasilkan `['do', "n't"]` di spaCy 2.x bisa menghasilkan `["don't"]` di 3.x. Modelmu dilatih pada satu distribusi. Inference kini berjalan pada distribusi berbeda. Akurasi diam-diam menurun dan tak ada yang tahu sebabnya. Pin versi pustaka di `requirements.txt`. Tulis uji regresi pra-pemrosesan yang membekukan tokenisasi ekspektasi dari 20 kalimat contoh. Jalankan pada setiap upgrade.

**Ketidakcocokan training / inference.** Latih dengan pra-pemrosesan agresif (lowercase, penghapusan stopword, stemming), deploy pada input mentah pengguna, saksikan performa anjlok. Ini adalah kegagalan NLP produksi paling umum. Jika kamu melakukan pra-pemrosesan saat training, kamu harus menjalankan fungsi yang identik saat inference. Kirim pra-pemrosesan sebagai fungsi di dalam paket model, bukan sebagai sel notebook yang ditulis ulang tim serving.

## Rilis

Sebuah prompt reusable yang membantu engineer memilih strategi pra-pemrosesan tanpa membaca tiga buku teks.

Simpan sebagai `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## Latihan

1. **Mudah.** Perluas `tokenize` agar mempertahankan URL sebagai token tunggal. Uji: `tokenize("Visit https://example.com today.")` harus menghasilkan satu token URL.
2. **Sedang.** Implementasikan langkah 1b Porter. Jika sebuah kata mengandung vokal dan berakhiran `ed` atau `ing`, hapus akhirannya. Tangani aturan konsonan ganda (`hopping -> hop`, bukan `hopp`).
3. **Sulit.** Bangun lemmatizer yang memakai WordNet sebagai tabel lookup tetapi fallback ke stemmer Porter-mu ketika WordNet tidak punya entrinya. Ukur akurasi pada korpus bertag melawan WordNet polos dan Porter polos.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Token | A word | Whatever unit the model consumes. Can be word, subword, character, or byte. |
| Stem | Root of a word | Result of rule-based suffix stripping. Not always a real word. |
| Lemma | Dictionary form | The form you'd look up. Requires grammatical context to compute correctly. |
| POS tag | Part of speech | Category like NOUN, VERB, ADJ. Needed to lemmatize accurately. |
| Morphology | Word shape rules | How a word changes form based on tense, number, case. Lemmatization depends on it. |

## Bacaan Lanjutan

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — the original paper, five pages, still the clearest explanation.
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — how a real pipeline is wired.
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — tokenization edge cases you haven't thought of yet.
