# Peringkasan Teks

> Sistem ekstraktif memberi tahu apa yang dikatakan dokumen. Sistem abstraktif memberi tahu apa yang dimaksud penulis. Tugas berbeda, jebakan berbeda.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Time:** ~75 minutes

## Masalahnya

Artikel berita 2.000 kata mendarat di feed-mu. Kamu butuh 120 kata yang menangkapnya. Kamu bisa memilih tiga kalimat terpenting dari artikel (ekstraktif) atau menulis ulang kontennya dengan kata-katamu sendiri (abstraktif). Keduanya disebut peringkasan. Keduanya adalah masalah yang sama sekali berbeda.

Peringkasan ekstraktif adalah masalah pemeringkatan. Skorkan setiap kalimat, kembalikan `k` teratas. Output selalu gramatikal karena diangkat verbatim. Risikonya adalah melewatkan konten yang tersebar di seluruh artikel.

Peringkasan abstraktif adalah masalah generasi. Transformer menghasilkan teks baru yang dikondisikan pada input. Output fasih dan kompresif tetapi bisa menghalusinasikan fakta yang tidak ada di sumber. Risikonya adalah fabrikasi yang percaya diri.

Pelajaran ini membangun keduanya, dengan mode kegagalan yang dimiliki masing-masing.

## Konsepnya

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**Ekstraktif.** Perlakukan artikel sebagai graf di mana node adalah kalimat dan edge adalah kemiripan. Jalankan PageRank (atau semacamnya) atas graf untuk memberi skor kalimat berdasarkan seberapa terhubung ia dengan yang lain. Kalimat berskor tertinggi adalah ringkasannya. Implementasi kanonisnya adalah **TextRank** (Mihalcea and Tarau, 2004).

**Abstraktif.** Fine-tune encoder-decoder transformer (BART, T5, Pegasus) pada pasangan dokumen-ringkas. Saat inference, model membaca dokumen dan menghasilkan ringkasan token-per-token via cross-attention. Pegasus khususnya memakai objektif pretraining gap-sentence yang membuatnya unggul dalam peringkasan tanpa banyak fine-tuning.

Evaluasi dengan **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation). ROUGE-1 dan ROUGE-2 memberi skor overlap unigram dan bigram. ROUGE-L memberi skor longest common subsequence. Lebih tinggi lebih baik tetapi 40 ROUGE-L itu "bagus" dan 50 itu "luar biasa." Setiap paper melaporkan ketiganya. Pakai paket `rouge-score`.

```figure
summarize-collapse
```

## Bangun Sendiri

### Langkah 1: TextRank (ekstraktif)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

Dua hal yang patut disebut. Fungsi kemiripan memakai overlap kata yang dinormalisasi-log, varian TextRank aslinya. Kosinus vektor TF-IDF juga bisa. Faktor damping 0.85 dan jumlah iterasi adalah default PageRank.

### Langkah 2: abstraktif dengan BART

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN di-fine-tune pada korpus CNN/DailyMail. Ia menghasilkan ringkasan gaya berita langsung. Untuk domain lain (paper saintifik, dialog, legal), pakai checkpoint Pegasus yang sesuai atau fine-tune pada data targetmu.

### Langkah 3: evaluasi ROUGE

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

Selalu pakai stemming. Tanpanya, "running" dan "run" dihitung sebagai kata berbeda dan ROUGE kurang menghitung.

### Melampaui ROUGE (eval peringkasan 2026)

ROUGE telah menjadi metrik peringkasan dominan selama dua puluh tahun dan tidak cukup sendirian di 2026. Meta-analisis skala besar atas paper NLG menunjukkan:

- **BERTScore** (kemiripan embedding kontekstual) menguat melalui 2023 dan kini dilaporkan berdampingan ROUGE di kebanyakan paper peringkasan.
- **BARTScore** memperlakukan evaluasi sebagai generasi: skorkan ringkasan berdasarkan seberapa besar BART pra-terlatih menilainya mungkin diberi sumber.
- **MoverScore** (Earth Mover's Distance atas embedding kontekstual) mencapai posisi teratas dalam benchmark peringkasan 2025 karena menangkap overlap semantik lebih baik daripada ROUGE.
- **FactCC** dan **faithfulness berbasis QA** umum di 2021-2023, kini sering digantikan **G-Eval** (rantai prompt GPT-4 yang menilai koherensi, konsistensi, kefasihan, relevansi dengan penalaran chain-of-thought).
- **G-Eval** dan pendekatan LLM-judge sejenis cocok dengan penilaian manusia ~80% ketika rubrik dirancang baik.

Rekomendasi produksi: laporkan ROUGE-L untuk perbandingan legacy, BERTScore untuk overlap semantik, G-Eval untuk koherensi dan faktualitas. Kalibrasi terhadap 50-100 ringkasan berlabel manusia.

### Langkah 4: masalah faktualitas

Ringkasan abstraktif rentan halusinasi. Ringkasan ekstraktif membawa risiko halusinasi jauh lebih rendah karena output diangkat verbatim dari sumber, meski masih bisa menyesatkan jika kalimat sumber didekontekstualisasi, kedaluwarsa, atau dikutip di luar urutan. Ini alasan tunggal terbesar sistem produksi masih memilih metode ekstraktif untuk konten yang dekat-compliance.

Tipe halusinasi yang perlu disebut:

- **Pertukaran entitas.** Sumber bilang "John Smith." Ringkasan bilang "John Brown."
- **Drift angka.** Sumber bilang "25,000." Ringkasan bilang "25 million."
- **Pembalikan polaritas.** Sumber bilang "rejected the offer." Ringkasan bilang "accepted the offer."
- **Fabrikasi fakta.** Sumber tidak menyebut CEO. Ringkasan bilang CEO menyetujui.

Pendekatan evaluasi yang berhasil:

- **FactCC.** Pengklasifikasi biner yang dilatih pada entailment antara kalimat sumber dan kalimat ringkasan. Memprediksi faktual/tidak-faktual.
- **Faktualitas berbasis QA.** Tanyakan pada model QA pertanyaan yang jawabannya ada di sumber. Jika ringkasan mendukung jawaban berbeda, tandai.
- **F1 level-entitas.** Bandingkan named entity di sumber vs ringkasan. Entitas yang hanya ada di ringkasan patut dicurigai.

Untuk apa pun yang menghadap pengguna di mana faktualitas penting (berita, medis, legal, finansial), ekstraktif adalah default yang lebih aman. Abstraktif butuh cek faktualitas dalam loop.

## Gunakan

Stack 2026:

| Use case | Recommended |
|---------|-------------|
| News, 3-5 sentence summary, English | `facebook/bart-large-cnn` |
| Scientific papers | `google/pegasus-pubmed` or a tuned T5 |
| Multi-document, long-form | Any LLM with 32k+ context, prompted |
| Dialog summarization | `philschmid/bart-large-cnn-samsum` |
| Extractive, low hallucination risk by construction | TextRank or `sumy`'s LSA / LexRank |

LLM dengan konteks panjang sering mengalahkan model khusus di 2026 ketika komputasi bukan konstrain. Tradeoff-nya adalah biaya dan reproduksibilitas; model khusus memberi output lebih konsisten.

## Rilis

Simpan sebagai `outputs/skill-summary-picker.md`:

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## Latihan

1. **Mudah.** Jalankan TextRank pada 5 artikel berita. Bandingkan 3 kalimat teratas dengan ringkasan referensi. Ukur ROUGE-L. Kamu akan melihat 30-45 ROUGE-L pada artikel gaya CNN/DailyMail.
2. **Sedang.** Implementasikan faktualitas level-entitas: ekstrak named entity dari sumber dan ringkasan (spaCy), hitung recall entitas sumber di ringkasan dan presisi entitas ringkasan terhadap sumber. Presisi tinggi dan recall rendah berarti aman tetapi ringkas; presisi rendah berarti entitas terhalusinasi.
3. **Sulit.** Bandingkan BART-large-CNN melawan LLM (Claude atau GPT-4) pada 50 artikel CNN/DailyMail. Laporkan ROUGE-L, faktualitas (via F1 entitas), dan biaya per ringkasan. Dokumentasikan di mana masing-masing menang.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Extractive | Pick sentences | Return sentences verbatim from the source. Never hallucinates. |
| Abstractive | Rewrite | Generate new text conditioned on source. Can hallucinate. |
| ROUGE | Summary metric | N-gram / LCS overlap between system output and reference. |
| TextRank | Graph-based extractive | PageRank over sentence similarity graph. |
| Factuality | Is it right | Whether summary claims are supported by the source. |
| Hallucination | Made-up content | Content in the summary that the source does not support. |

## Bacaan Lanjutan

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — the extractive canonical paper.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — the BART paper.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus and the gap-sentence objective.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE paper.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — the factuality landscape paper.
