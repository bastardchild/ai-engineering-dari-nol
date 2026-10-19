# NLP Multilingual

> Satu model, 100+ bahasa, nol data training untuk kebanyakannya. Transfer lintas-lingual adalah keajaiban praktis tahun 2020-an.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45 minutes

## Masalahnya

Bahasa Inggris punya miliaran contoh berlabel. Urdu punya ribuan. Maithili hampir tidak ada. Sistem NLP praktis apa pun yang melayani audiens global harus bekerja pada ekor panjang bahasa di mana data training spesifik-tugas tidak ada.

Model multilingual memecahkannya dengan melatih satu model pada banyak bahasa sekaligus. Representasi bersama memungkinkan model mentransfer keterampilan yang dipelajari di bahasa kaya-sumber-daya ke yang miskin-sumber-daya. Fine-tune model pada analisis sentimen Inggris, dan ia menghasilkan prediksi sentimen Urdu yang mengejutkan baiknya secara out of the box. Itulah transfer lintas-lingual zero-shot, dan ia telah membentuk ulang cara NLP dikirim ke dunia.

Pelajaran ini menyebutkan tradeoff, model kanonis, dan satu keputusan yang menjegal tim yang baru dalam kerja multilingual: memilih bahasa sumber untuk transfer.

## Konsepnya

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**Vocabulary bersama.** Model multilingual memakai tokenizer SentencePiece atau WordPiece yang dilatih pada teks dari semua bahasa target. Vocabulary-nya bersama: unit subword yang sama merepresentasikan morfem yang sama lintas bahasa terkait. `anti-` dalam bahasa Inggris dan Italia mendapat token yang sama.

**Representasi bersama.** Transformer yang di-pretrain pada masked language modeling lintas banyak bahasa mempelajari bahwa kalimat mirip-semantik dalam bahasa berbeda menghasilkan hidden state mirip. mBERT, XLM-R, dan NLLB semuanya menunjukkan ini. Embedding untuk "cat" dalam bahasa Inggris mengelompok dekat "chat" dalam bahasa Prancis dan "gato" dalam bahasa Spanyol, begitu pula embedding kalimat penuh.

**Transfer zero-shot.** Fine-tune model pada data berlabel dalam satu bahasa (biasanya Inggris). Saat inference, jalankan pada bahasa lain yang didukung model. Tanpa label bahasa-target. Hasilnya kuat untuk bahasa yang tipologisnya terkait dan lebih lemah untuk yang jauh.

**Fine-tuning few-shot.** Tambahkan 100-500 contoh berlabel dalam bahasa target. Akurasi melonjak ke 95-98% dari baseline Inggris pada tugas klasifikasi. Ini tuas paling hemat-biaya dalam NLP multilingual.

## Model-modelnya

| Model | Year | Coverage | Notes |
|-------|------|----------|-------|
| mBERT | 2018 | 104 languages | Trained on Wikipedia. First practical multilingual LM. Weak on low-resource. |
| XLM-R | 2019 | 100 languages | Trained on CommonCrawl (much larger than Wikipedia). Sets the cross-lingual baseline. Base 270M, Large 550M. |
| XLM-V | 2023 | 100 languages | XLM-R with 1M-token vocabulary (vs 250k). Better on low-resource. |
| mT5 | 2020 | 101 languages | T5 architecture for multilingual generation. |
| NLLB-200 | 2022 | 200 languages | Meta's translation model; includes 55 low-resource languages. |
| BLOOM | 2022 | 46 languages + 13 programming | Open 176B LLM trained multilingually. |
| Aya-23 | 2024 | 23 languages | Cohere's multilingual LLM. Strong on Arabic, Hindi, Swahili. |

Pilih berdasarkan use case. Klasifikasi bekerja baik dengan XLM-R-base sebagai default yang waras. Tugas generasi memanggil mT5 atau NLLB tergantung penerjemahan vs generasi terbuka. Kerja ala LLM dipasangkan dengan Aya-23 atau Claude memakai prompting multilingual eksplisit.

## Keputusan bahasa-sumber (riset 2026)

Kebanyakan tim default ke Inggris sebagai sumber fine-tuning. Riset terbaru (2026) menunjukkan ini sering salah.

Kemiripan bahasa memprediksi kualitas transfer lebih baik daripada ukuran korpus mentah. Untuk target Slavia, Jerman atau Rusia sering mengalahkan Inggris. Untuk target Indik, Hindi sering mengalahkan Inggris. Metrik kemiripan **qWALS** (2026, berdasarkan fitur World Atlas of Language Structures) mengkuantifikasinya. **LANGRANK** (Lin et al., ACL 2019) adalah metode terpisah yang lebih awal yang memeringkat kandidat bahasa sumber dari kombinasi kemiripan linguistik, ukuran korpus, dan kekerabatan genetik.

Aturan praktis: jika bahasa targetmu punya kerabat kaya-sumber-daya yang tipologisnya dekat, coba fine-tuning pada bahasa itu dulu, lalu bandingkan dengan fine-tune Inggris.

```figure
n5-crosslingual-bridge
```

## Bangun Sendiri

### Langkah 1: klasifikasi lintas-lingual zero-shot

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

Satu model, tiga bahasa, API sama. XLM-R yang dilatih pada data NLI mentransfer baik ke klasifikasi via trik entailment.

### Langkah 2: ruang embedding multilingual

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Terjemahan mendarat berdekatan dalam ruang embedding. Kalimat Inggris berbeda mendarat lebih jauh. Inilah yang membuat retrieval, clustering, dan kemiripan lintas-lingual bekerja.

### Langkah 3: strategi fine-tuning few-shot

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

Untuk 100-500 contoh bahasa-target, `num_train_epochs=5` dan `learning_rate=2e-5` adalah default aman. Learning rate lebih tinggi menyebabkan alignment multilingual runtuh dan kamu mendapat model khusus-Inggris.

## Evaluasi yang benar-benar bekerja

- **Akurasi per bahasa pada held-out set.** Bukan agregat. Agregat menyembunyikan ekor panjang.
- **Benchmark melawan baseline monolingual.** Untuk bahasa dengan data cukup, model monolingual yang dilatih dari nol kadang mengalahkan yang multilingual. Uji.
- **Uji level-entitas.** Named entity dalam bahasa target. Model multilingual sering punya tokenisasi lemah untuk skrip yang jauh dari Latin.
- **Konsistensi lintas-lingual.** Makna sama dalam dua bahasa harus menghasilkan prediksi sama. Ukur gap-nya.

## Gunakan

Stack 2026:

| Task | Recommended |
|-----|-------------|
| Classification, 100 languages | XLM-R-base (~270M) fine-tuned |
| Zero-shot text classification | `joeddav/xlm-roberta-large-xnli` |
| Multilingual sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Translation, 200 languages | `facebook/nllb-200-distilled-600M` (see lesson 11) |
| Generative multilingual | Claude, GPT-4, Aya-23, mT5-XXL |
| Low-resource language NLP | XLM-V or a domain-specific fine-tune on related high-resource language |

Selalu anggarkan fine-tuning dalam bahasa target jika performa penting. Zero-shot adalah titik awal, bukan jawaban final.

### Pajak tokenisasi (apa yang salah untuk bahasa sumber-daya-rendah)

Model multilingual berbagi satu tokenizer lintas semua bahasanya. Vocabulary itu dilatih pada korpus yang didominasi Inggris, Prancis, Spanyol, Mandarin, Jerman. Untuk bahasa apa pun di luar himpunan dominan, tiga pajak bertumpuk diam-diam:

- **Pajak fertilitas.** Teks bahasa sumber-daya-rendah ter-tokenisasi menjadi jauh lebih banyak token per kata daripada Inggris. Kalimat Hindi bisa butuh 3-5x token dari kalimat Inggris setara. 3-5x itu memakan context window, efisiensi training, dan latensimu.
- **Pajak pemulihan varian.** Setiap typo, varian diakritik, ketidakcocokan normalisasi Unicode, atau variasi huruf besar-kecil menjadi sekuens cold-start yang tak terkait dalam ruang embedding. Model tidak bisa mempelajari korespondensi ortografis yang bagi penutur asli sudah jelas.
- **Pajak spillover kapasitas.** Pajak 1 dan 2 mengonsumsi posisi konteks, kedalaman layer, dan dimensi embedding. Yang tersisa untuk penalaran aktual secara sistematis lebih kecil daripada yang didapat bahasa kaya-sumber-daya dari model yang sama.

Gejala praktisnya: modelmu berlatih normal pada Hindi, kurva loss tampak benar, perplexity eval tampak wajar, dan output produksi halus salahnya. Morfologi runtuh di tengah kalimat. Infleksi jarang tetap tak terpulihkan. **Kamu tidak bisa men-data-scale keluar dari tokenizer rusak.**

Mitigasi: pilih tokenizer dengan cakupan bagus untuk bahasa targetmu (vocabulary 1M-token XLM-V adalah perbaikan langsung); verifikasi fertilitas tokenisasi pada teks target held-out sebelum training; pakai fallback level-byte (SentencePiece `byte_fallback=True`, BPE level-byte ala GPT-2) untuk skrip yang benar-benar long-tail supaya tak ada yang OOV.

## Rilis

Simpan sebagai `outputs/skill-multilingual-picker.md`:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## Latihan

1. **Mudah.** Jalankan pipeline klasifikasi zero-shot pada 10 kalimat per bahasa lintas Inggris, Prancis, Hindi, dan Arab. Laporkan akurasi masing-masing. Kamu akan melihat Prancis kuat, Hindi lumayan, Arab bervariasi.
2. **Sedang.** Pakai `paraphrase-multilingual-MiniLM-L12-v2` untuk membangun retriever lintas-lingual atas korpus kecil campuran bahasa. Query dalam bahasa Inggris, ambil dokumen dalam bahasa apa pun. Ukur recall@5.
3. **Sulit.** Bandingkan fine-tuning sumber-Inggris dan sumber-Hindi untuk tugas klasifikasi Hindi. Pakai 500 contoh bahasa-target untuk fine-tuning few-shot di kedua rezim. Laporkan sumber mana yang menghasilkan akurasi Hindi lebih baik dan berapa besarnya. Ini tesis LANGRANK dalam miniatur.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Multilingual model | One model, many languages | Shared vocabulary and parameters across languages. |
| Cross-lingual transfer | Train on one language, run on another | Fine-tune on source, evaluate on target without target-language labels. |
| Zero-shot | No target-language labels | Transfer without fine-tuning on the target language. |
| Few-shot | Small target labels | 100-500 target-language examples used for fine-tuning. |
| mBERT | First multilingual LM | 104-language BERT pretrained on Wikipedia. |
| XLM-R | Standard cross-lingual baseline | 100-language RoBERTa pretrained on CommonCrawl. |
| NLLB | Meta's 200-language MT | No Language Left Behind. Includes 55 low-resource languages. |

## Bacaan Lanjutan

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — the XLM-R paper.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — the analysis paper that started the cross-lingual transfer research line.
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 paper.
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya, Cohere's multilingual LLM.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — the qWALS / LANGRANK source-language paper.
