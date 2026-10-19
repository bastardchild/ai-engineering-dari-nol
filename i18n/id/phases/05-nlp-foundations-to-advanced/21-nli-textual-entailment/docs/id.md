# Inferensi Bahasa Alami — Entailment Tekstual

> "t meng-entail h" berarti manusia yang membaca t akan menyimpulkan h benar. NLI adalah tugas memprediksi entailment / contradiction / neutral. Membosankan di permukaan, menahan beban di produksi.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 minutes

## Masalahnya

Kamu membangun peringkas. Ia menghasilkan ringkasan. Bagaimana kamu tahu ringkasan itu tidak memuat halusinasi?

Kamu membangun chatbot. Ia menjawab "yes." Bagaimana kamu tahu jawabannya didukung passage yang diambil?

Kamu perlu mengklasifikasikan 10.000 artikel berita berdasarkan topik. Kamu tidak punya label training. Bisakah memakai ulang model?

Ketiga masalah mereduksi ke Natural Language Inference. NLI bertanya: diberi premis `t` dan hipotesis `h`, apakah `h` di-entail oleh `t`, dikontradiksi, atau netral (tak terkait)?

- **Cek halusinasi:** `t` = dokumen sumber, `h` = klaim ringkasan. Bukan entailment = halusinasi.
- **QA ter-grounding:** `t` = passage yang diambil, `h` = jawaban yang dihasilkan. Bukan entailment = fabrikasi.
- **Klasifikasi zero-shot:** `t` = dokumen, `h` = label terverbalisasi ("This is about sports"). Entailment = label prediksi.

Satu tugas, tiga kegunaan produksi. Inilah alasan setiap framework evaluasi RAG menyediakan model NLI di balik layar.

## Konsepnya

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**Ketiga label.**

- **Entailment.** `t` → `h`. "The cat is on the mat" meng-entail "There is a cat."
- **Kontradiksi.** `t` → ¬`h`. "The cat is on the mat" mengkontradiksi "There is no cat."
- **Netral.** Tak ada inferensi ke arah mana pun. "The cat is on the mat" netral terhadap "The cat is hungry."

**Bukan entailment logis.** NLI adalah inferensi bahasa *alami* — apa yang disimpulkan pembaca manusia tipikal, bukan logika ketat. "John walked his dog" meng-entail "John has a dog" dalam NLI, tetapi logika orde-pertama ketat hanya mengakuinya jika kamu mengaksiomakan kepemilikan.

**Dataset.**

- **SNLI** (2015). 570k pasangan beranotasi manusia, caption gambar sebagai premis. Domain sempit.
- **MultiNLI** (2017). 433k pasangan lintas 10 genre. Korpus training standar di 2026.
- **ANLI** (2019). NLI adversarial. Manusia menulis contoh yang dirancang khusus mematahkan model yang ada. Lebih sulit.
- **DocNLI, ConTRoL** (2020–21). Premis sepanjang dokumen. Menguji inferensi multi-hop dan jarak-jauh.

**Arsitekturnya.** Encoder transformer (BERT, RoBERTa, DeBERTa) membaca `[CLS] premise [SEP] hypothesis [SEP]`. Representasi `[CLS]` masuk ke softmax 3-arah. Latih pada MNLI, evaluasi pada benchmark held-out, dapatkan akurasi 90%+ pada pasangan in-distribution.

**Zero-shot via NLI.** Diberi dokumen dan label kandidat, ubah tiap label menjadi hipotesis ("This text is about sports"). Hitung probabilitas entailment masing-masing. Pilih yang maks. Inilah mekanisme di balik pipeline `zero-shot-classification` Hugging Face.

```figure
nli-router
```

## Bangun Sendiri

### Langkah 1: jalankan model NLI pra-terlatih

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

Untuk NLI produksi, `facebook/bart-large-mnli` dan `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` adalah default terbuka. DeBERTa-v3 memuncaki leaderboard.

### Langkah 2: klasifikasi zero-shot

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

Template-nya "This example is about {label}." secara default. Kustomisasi dengan `hypothesis_template`. Tanpa data training. Tanpa fine-tuning. Berhasil langsung.

### Langkah 3: cek faithfulness untuk RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

Ini inti faithfulness RAGAS. Pecah jawaban yang dihasilkan menjadi klaim atomik. Cek tiap klaim terhadap konteks yang diambil. Laporkan fraksi yang entail.

### Langkah 4: pengklasifikasi NLI buatan tangan (konseptual)

Lihat `code/main.py` untuk mainan stdlib-only: premis dan hipotesis dibandingkan via overlap leksikal + deteksi negasi. Tidak kompetitif dengan model transformer — tetapi menunjukkan bentuk tugasnya: dua teks masuk, label 3-arah keluar, loss = cross-entropy atas `{entail, contradict, neutral}`.

## Jebakan

- **Shortcut hipotesis-saja.** Model bisa memprediksi label dari hipotesis saja pada ~60% di SNLI karena "not", "nobody", "never" berkorelasi dengan kontradiksi. Baseline kuat untuk mendeteksi kebocoran label.
- **Heuristik overlap leksikal.** Heuristik subsekuens ("setiap subsekuens di-entail") lolos SNLI tetapi gagal HANS/ANLI. Pakai benchmark adversarial.
- **Degradasi panjang-dokumen.** Model NLI kalimat-tunggal turun 20+ F1 pada premis sepanjang dokumen. Pakai model terlatih DocNLI untuk konteks panjang.
- **Sensitivitas template zero-shot.** "This example is about {label}" vs "{label}" vs "The topic is {label}" bisa mengayunkan akurasi 10+ poin. Tune template-nya.
- **Ketidakcocokan domain.** MNLI berlatih pada Inggris umum. Teks legal, medis, dan saintifik butuh model NLI spesifik-domain (mis. SciNLI, MedNLI).

## Gunakan

Stack 2026:

| Use case | Model |
|---------|-------|
| General-purpose NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Fast / edge | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot classification (lightweight) | `facebook/bart-large-mnli` |
| Document-level NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Multilingual | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Hallucination detection in RAG | NLI layer inside RAGAS / DeepEval |

Meta-pola 2026: NLI adalah lakban pemahaman teks. Kapan pun kamu butuh "apakah A mendukung B?" atau "apakah A mengkontradiksi B?" — raih NLI sebelum meraih panggilan LLM lain.

## Rilis

Simpan sebagai `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## Latihan

1. **Mudah.** Jalankan `facebook/bart-large-mnli` pada 20 triple (premis, hipotesis, label) buatan tangan yang mencakup ketiga kelas. Ukur akurasi. Tambahkan jebakan "heuristik subsekuens" adversarial ("I did not eat the cake" vs "I ate the cake") dan lihat apakah ia patah.
2. **Sedang.** Bandingkan template zero-shot `"This text is about {label}"` melawan `"The topic is {label}"` dan `"{label}"` pada 100 headline AG News. Laporkan ayunan akurasi.
3. **Sulit.** Bangun pemeriksa faithfulness RAG: dekomposisi klaim-atomik + NLI per klaim. Evaluasi pada 50 jawaban hasil-RAG dengan konteks emas. Ukur tingkat false-positive dan false-negative vs label tangan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| NLI | Natural Language Inference | 3-way classification of premise-hypothesis relationship. |
| RTE | Recognizing Textual Entailment | Older name for NLI; same task. |
| Entailment | "t implies h" | A typical reader would conclude h is true given t. |
| Contradiction | "t rules out h" | A typical reader would conclude h is false given t. |
| Neutral | "undecided" | No inference from t to h either way. |
| Zero-shot classification | NLI as classifier | Verbalize labels as hypotheses, pick max entailment. |
| Faithfulness | Is the answer supported? | NLI over (retrieved context, generated answer). |

## Bacaan Lanjutan

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — the ANLI benchmark.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — the 2026 NLI workhorse.
