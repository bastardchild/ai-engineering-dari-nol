# Evaluasi LLM — RAGAS, DeepEval, G-Eval

> Exact-match dan F1 melewatkan kesetaraan semantik. Review manusia tidak scale. LLM-as-judge adalah jawaban produksi — dengan kalibrasi cukup untuk memercayai angkanya.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## Masalahnya

Sistem RAG-mu menjawab: "June 29th, 2007."
Referensi emasnya: "June 29, 2007."
Exact Match memberi skor 0. F1 memberi skor ~75%. Manusia akan memberi skor 100%.

Kini kalikan dengan 10.000 kasus uji. Kalikan lagi dengan setiap perubahan pada retriever, chunking, prompt, atau model. Kamu butuh evaluator yang memahami makna, berjalan murah dalam skala, tidak berbohong tentang regresi, dan memunculkan mode kegagalan yang tepat.

2026 punya tiga framework yang menguasai masalah ini.

- **RAGAS.** Retrieval-Augmented Generation ASsessment. Empat metrik RAG (faithfulness, answer-relevance, context-precision, context-recall) dengan backend NLI + LLM-judge. Didukung riset, ringan.
- **DeepEval.** Pytest untuk LLM. Metrik G-Eval, task-completion, hallucination, bias. Native-CI/CD.
- **G-Eval.** Sebuah metode (dan metrik DeepEval): LLM-as-judge dengan chain-of-thought, kriteria kustom, skor 0-1.

Ketiganya bersandar pada LLM-as-judge. Pelajaran ini membangun intuisi untuk metodenya dan lapisan kepercayaan di sekitarnya.

## Konsepnya

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge.** Ganti metrik statis dengan LLM yang menilai output diberi rubrik. Diberi `(query, context, answer)`, prompt LLM judge: "Score 0-1 on faithfulness." Kembalikan skornya.

Mengapa berhasil: LLM mengaproksimasi penilaian manusia dengan fraksi kecil biayanya. GPT-4o-mini pada ~$0.003 per kasus yang dinilai memungkinkan run eval regresi 1000-sampel di bawah $5.

Mengapa gagal diam-diam:

1. **Bias judge.** Judge menyukai jawaban lebih panjang, jawaban dari keluarga modelnya sendiri, jawaban yang cocok gaya prompt.
2. **Kegagalan parsing JSON.** JSON buruk → skor NaN → diam-diam dikeluarkan dari agregat. Pengguna RAGAS tahu pedihnya. Gerbang dengan try/except + mode kegagalan eksplisit.
3. **Drift lintas versi model.** Meng-upgrade judge mengubah setiap metrik. Bekukan model + versi judge.

**Empat RAG.**

| Metric | Question | Backend |
|--------|----------|---------|
| Faithfulness | Does each claim in the answer come from the retrieved context? | NLI-based entailment |
| Answer relevance | Does the answer address the question? | Generate hypothetical questions from answer; compare to real question |
| Context precision | Of retrieved chunks, what fraction were relevant? | LLM-judge |
| Context recall | Did retrieval return everything needed? | LLM-judge against gold answer |

**G-Eval.** Definisikan kriteria kustom: "Did the answer cite the correct source?" Framework mengembang otomatis menjadi langkah evaluasi chain-of-thought, lalu menilai 0-1. Bagus untuk dimensi kualitas spesifik-domain yang tidak dicakup RAGAS.

**Kalibrasi.** Jangan pernah percaya skor judge mentah sampai kamu punya korelasi terhadap label manusia. Jalankan 100 contoh berlabel tangan. Plot judge vs manusia. Hitung Spearman rho. Jika rho < 0.7, rubrik judgemu perlu kerja.

```figure
n5-judge-gauge
```

## Bangun Sendiri

### Langkah 1: faithfulness dengan NLI (ala RAGAS)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

Uraikan jawaban menjadi klaim atomik. Cek-NLI tiap klaim terhadap konteks yang diambil. Faithfulness = fraksi yang didukung.

### Langkah 2: relevansi jawaban

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

Jika jawaban mengimplikasikan pertanyaan berbeda dari yang ditanyakan, relevansi turun.

### Langkah 3: metrik kustom G-Eval

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Langkah evaluasi adalah rubriknya. Langkah eksplisit lebih stabil daripada prompt "score 0-1" implisit.

### Langkah 4: gerbang CI

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

Kirim sebagai file pytest. Jalankan pada setiap PR. Blokir merge pada regresi.

### Langkah 5: eval mainan dari nol

Lihat `code/main.py`. Aproksimasi faithfulness stdlib-only (overlap klaim jawaban dengan konteks) dan relevansi (overlap token jawaban dengan token pertanyaan). Bukan produksi. Menunjukkan bentuknya.

## Jebakan

- **Tanpa kalibrasi.** Judge dengan korelasi 0.3 terhadap label manusia adalah noise. Wajibkan run kalibrasi sebelum rilis.
- **Evaluasi-diri.** Memakai LLM yang sama untuk menghasilkan dan menilai menggelembungkan skor 10-20%. Pakai keluarga model berbeda untuk judge.
- **Bias posisional dalam judging berpasangan.** Judge menyukai opsi pertama yang disajikan. Selalu acak urutan dan jalankan keduanya.
- **Agregat mentah menyembunyikan kegagalan.** Skor rata-rata 0.85 sering menyembunyikan 5% kegagalan katastropik. Selalu periksa kuantil terbawah.
- **Pembusukan dataset emas.** Eval set tak berversi yang melayang seiring waktu merusak perbandingan longitudinal. Tandai dataset pada setiap perubahan.
- **Biaya LLM.** Dalam skala, panggilan judge mendominasi biaya. Pakai model termurah yang memenuhi ambang kalibrasi. GPT-4o-mini, Claude Haiku, Mistral-small.

## Gunakan

Stack 2026:

| Use case | Framework |
|---------|-----------|
| RAG quality monitoring | RAGAS (4 metrics) |
| CI/CD regression gates | DeepEval + pytest |
| Custom domain criteria | G-Eval within DeepEval |
| Online live-traffic monitoring | RAGAS with reference-free mode |
| Human-in-the-loop spot checks | LangSmith or Phoenix with annotation UI |
| Red-teaming / safety eval | Promptfoo + DeepEval |

Stack tipikal: RAGAS untuk monitoring, DeepEval untuk CI, G-Eval untuk dimensi baru. Jalankan ketiganya; mereka berguna tidak sepakatnya.

## Rilis

Simpan sebagai `outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## Latihan

1. **Mudah.** Pakai RAGAS pada 10 contoh RAG dengan halusinasi yang diketahui. Verifikasi metrik faithfulness menangkap masing-masing.
2. **Sedang.** Labeli 50 jawaban QA dengan tangan 0-1 untuk kebenaran. Skorkan dengan G-Eval. Ukur Spearman rho antara judge dan manusia.
3. **Sulit.** Bangun gerbang CI pytest dengan DeepEval. Regresikan retriever dengan sengaja. Verifikasi gerbangnya gagal. Tambahkan alerting kuantil-terbawah via cek ambang pada 10% terendah.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| LLM-as-judge | Scoring with an LLM | Prompt a judge model to score outputs 0-1 given a rubric. |
| RAGAS | The RAG metric library | Open-source eval framework with 4 reference-free RAG metrics. |
| Faithfulness | Is the answer grounded? | Fraction of answer claims entailed by retrieved context. |
| Context precision | Were retrieved chunks relevant? | Fraction of top-K chunks that actually mattered. |
| Context recall | Did retrieval find everything? | Fraction of gold-answer claims supported by retrieved chunks. |
| G-Eval | Custom LLM judge | Rubric + chain-of-thought eval steps + 0-1 score. |
| Calibration | Trust but verify | Spearman correlation between judge score and human score. |

## Bacaan Lanjutan

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — the RAGAS paper.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — the G-Eval paper.
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — open production stack.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — biases, calibration, limits.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — unifying framework that integrates RAGAS, DeepEval, Phoenix.
