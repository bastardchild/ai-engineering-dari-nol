# Terjemahan Mesin

> Penerjemahan adalah tugas yang membiayai riset NLP selama tiga puluh tahun dan masih membiayainya sekarang.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## Masalahnya

Sebuah model membaca kalimat dalam satu bahasa dan menghasilkan kalimat dalam bahasa lain. Panjang bervariasi. Urutan kata bervariasi. Sebagian kata sumber dipetakan ke banyak kata target dan sebaliknya. Idiom menolak pemetaan satu-ke-satu. "I miss you" dalam bahasa Prancis adalah "tu me manques" — harfiahnya "kamu kurang bagiku." Tak ada alignment level-kata yang selamat dari itu.

Terjemahan mesin adalah tugas yang memaksa NLP menciptakan encoder-decoder, attention, transformer, dan akhirnya seluruh paradigma LLM. Setiap langkah maju tiba karena kualitas terjemahan terukur dan gap antara manusia dan mesin membandel.

Pelajaran ini melewatkan pelajaran sejarah dan mengajarkan pipeline kerja 2026: encoder-decoder multilingual pra-terlatih (NLLB-200 atau mBART), tokenisasi subword, beam search, evaluasi BLEU dan chrF, dan segenggam mode kegagalan yang masih lolos ke produksi tanpa terdeteksi.

## Konsepnya

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

MT modern adalah encoder-decoder transformer yang dilatih pada teks paralel. Encoder membaca sumber dalam tokenisasi bahasanya. Decoder menghasilkan target, satu subword per satu, memakai output encoder via cross-attention (pelajaran 10). Decoding memakai beam search untuk menghindari jebakan decoding greedy. Output di-detokenisasi, di-detruecase, dan diskorkan terhadap referensi.

Tiga pilihan operasional mendorong kualitas MT dunia nyata.

- **Tokenizer.** BPE SentencePiece yang dilatih pada korpus campuran bahasa. Vocabulary bersama lintas bahasa inilah yang memungkinkan pasangan zero-shot di NLLB.
- **Ukuran model.** NLLB-200 distilled 600M muat di laptop. NLLB-200 3.3B adalah default produksi yang dipublikasikan. 54.5B adalah langit-langit riset.
- **Decoding.** Lebar beam 4-5 untuk konten umum. Penalti panjang untuk menghindari output terlalu pendek. Decoding terkendali ketika kamu butuh konsistensi terminologi.

```figure
seq2seq-alignment
```

## Bangun Sendiri

### Langkah 1: panggilan MT pra-terlatih

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

Tiga hal penting di sini. `src_lang` memberi tahu tokenizer skrip dan segmentasi mana yang dipakai. `forced_bos_token_id` memberi tahu decoder bahasa apa yang dihasilkan. Keduanya trik spesifik NLLB; mBART dan M2M-100 memakai konvensinya sendiri dan tidak saling tertukar.

### Langkah 2: BLEU dan chrF

BLEU mengukur overlap n-gram antara output dan referensi. Empat ukuran n-gram referensi (1-4), rata-rata geometris presisi, penalti brevity untuk output terlalu pendek. Skornya dalam [0, 100]. Umum dipakai. Menyebalkan diinterpretasikan: 30 BLEU itu "bisa dipakai"; 40 itu "bagus"; 50 itu "luar biasa"; perbedaan di bawah 1 BLEU adalah noise.

chrF mengukur F-score level karakter. Lebih sensitif untuk bahasa yang kaya morfologi di mana BLEU kurang menghitung kecocokan. Sering dilaporkan berdampingan BLEU.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

Selalu pakai `sacrebleu`. Ia menormalisasi tokenisasi sehingga skor sebanding lintas paper. Menghitung BLEU sendiri adalah cara menghasilkan benchmark menyesatkan.

### Hierarki evaluasi tiga tingkat (2026)

Evaluasi MT modern memakai tiga keluarga metrik komplementer. Rilis dengan setidaknya dua.

- **Heuristik** (BLEU, chrF). Cepat, berbasis referensi, dapat diinterpretasi, tidak sensitif terhadap parafrasa. Pakai untuk perbandingan legacy dan deteksi regresi.
- **Terpelajari** (COMET, BLEURT, BERTScore). Model neural yang dilatih pada penilaian manusia; membandingkan kemiripan semantik terjemahan dengan sumber dan referensi. COMET punya asosiasi tertinggi dengan riset MT sejak 2023 dan menjadi default produksi 2026 di mana kualitas penting.
- **LLM-as-judge** (bebas referensi). Prompt model besar untuk menilai terjemahan pada kefasihan, kecukupan, nada, kesesuaian budaya. GPT-4-as-judge cocok dengan kesepakatan manusia ~80% ketika rubrik dirancang baik. Pakai untuk konten terbuka di mana referensi tidak ada.

Stack praktis 2026: `sacrebleu` untuk BLEU dan chrF, `unbabel-comet` untuk COMET, dan LLM yang di-prompt untuk sinyal akhir yang menghadap manusia. Kalibrasi setiap metrik terhadap 50-100 contoh berlabel manusia sebelum memercayainya pada data produksi.

Metrik bebas referensi (COMET-QE, BLEURT-QE, LLM-as-judge) memungkinkanmu mengevaluasi terjemahan tanpa referensi, yang penting untuk pasangan bahasa long-tail di mana terjemahan referensi tidak ada.

### Langkah 3: apa yang rusak di produksi

Pipeline kerja di atas akan menerjemahkan dengan fasih 80% waktu dan gagal diam-diam 20% sisanya. Mode kegagalan bernama:

- **Halusinasi.** Model mengarang konten yang tidak ada di sumber. Umum pada kosakata domain asing. Gejala: output fasih tetapi mengklaim fakta yang tak dinyatakan sumber. Mitigasi: decoding terkendali pada term domain, review manusia pada konten teregulasi, monitoring untuk output yang jauh lebih panjang daripada input.
- **Generasi off-target.** Model menerjemahkan ke bahasa yang salah. NLLB mengejutkan rentannya pada pasangan bahasa jarang. Mitigasi: verifikasi `forced_bos_token_id` dan selalu decode dengan cek model language-ID pada output.
- **Drift terminologi.** "Sign up" menjadi "s'inscrire" di dok 1 dan "créer un compte" di dok 2. Untuk teks UI dan string yang menghadap pengguna, konsistensi lebih penting daripada kualitas mentah. Mitigasi: decoding terkendali-glosarium atau kamus pasca-edit.
- **Ketidakcocokan formalitas.** "tu" vs "vous" Prancis, tingkat kesopanan Jepang. Model memilih bentuk mana pun yang lebih umum saat training. Untuk konten pelanggan ini biasanya salah. Mitigasi: awalan prompt dengan token formalitas jika model mendukungnya, atau fine-tune model kecil pada korpus khusus-formal.
- **Ledakan panjang pada input pendek.** Kalimat input sangat pendek sering menghasilkan terjemahan kepanjangan karena penalti panjang jatuh dari tebing di bawah ~5 token sumber. Mitigasi: batas panjang-maks keras proporsional terhadap panjang sumber.

### Langkah 4: fine-tuning untuk domain

Model pra-terlatih adalah generalis. Penerjemahan legal, medis, atau dialog game mendapat manfaat terukur dari fine-tuning pada data paralel domain. Resepnya tidak eksotis:

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

Beberapa ribu contoh paralel berkualitas tinggi mengalahkan beberapa ratus ribu contoh noisy hasil scrape web. Kualitas data training adalah tuas produksi tunggal terbesar.

## Gunakan

Stack produksi 2026 untuk MT:

| Use case | Recommended starting point |
|---------|---------------------------|
| Any-to-any, 200 languages | `facebook/nllb-200-distilled-600M` (laptop) or `nllb-200-3.3B` (production) |
| English-centric, high quality, 50 languages | `facebook/mbart-large-50-many-to-many-mmt` |
| Short runs, cheap inference, English-French/German/Spanish | Helsinki-NLP / Marian models |
| Latency-critical browser-side | ONNX-quantized Marian (~50 MB) |
| Maximum quality, willing to pay | GPT-4 / Claude / Gemini with translation prompts |

LLM kini mengungguli model MT khusus pada beberapa pasangan bahasa per 2026, terutama pada konten idiomatis dan konteks panjang. Tradeoff-nya adalah biaya per token dan latensi. Pilih LLM ketika panjang konteks, konsistensi gaya, atau adaptasi domain via prompting lebih penting daripada throughput.

## Rilis

Simpan sebagai `outputs/skill-mt-evaluator.md`:

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## Latihan

1. **Mudah.** Terjemahkan paragraf Inggris 5-kalimat ke Prancis lalu kembali ke Inggris memakai `nllb-200-distilled-600M`. Ukur seberapa dekat round-trip dengan aslinya. Kamu akan melihat preservasi semantik dengan drift pilihan kata.
2. **Sedang.** Implementasikan cek language-ID pada output terjemahan memakai `fasttext lid.176` atau `langdetect`. Integrasikan ke panggilan MT supaya generasi off-target tertangkap sebelum dikembalikan.
3. **Sulit.** Fine-tune `nllb-200-distilled-600M` pada korpus domain 5.000-pasangan pilihanmu. Ukur BLEU pada held-out set sebelum dan sesudah fine-tuning. Laporkan jenis kalimat mana yang membaik dan mana yang regresi.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BLEU | Translation score | N-gram precision with brevity penalty. [0, 100]. |
| chrF | Character F-score | Character-level F-score. More sensitive for morphologically rich languages. |
| NMT | Neural MT | Transformer encoder-decoder trained on parallel text. The 2017+ default. |
| NLLB | No Language Left Behind | Meta's 200-language MT model family. |
| Constrained decoding | Controlled output | Force specific tokens or n-grams to appear / not appear in the output. |
| Hallucination | Invented content | Model output that is not supported by the source. |

## Bacaan Lanjutan

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — the NLLB paper.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — why `sacrebleu` is the only correct way to report BLEU.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — the chrF paper.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — practical fine-tuning walkthrough.
