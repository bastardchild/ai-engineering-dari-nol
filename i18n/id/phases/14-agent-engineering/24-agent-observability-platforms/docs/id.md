# Agen Observabilitas: Langfuse, Phoenix, Opik

> Tiga buka-source agen observabilitas platforms dominate 2026. Langfuse (MIT) — 6M+ installs/month, tracing + prompt management + eval + sesi replay. Arize Phoenix (Elastic 2.0) — deep agen-specific eval, RAG relevancy, OpenInference auto-instrumentation. Comet Opik (Apache 2.0) — automated prompt optimization, guardrail, LLM-judge hallucination detection.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45 minutes

## Tujuan Pembelajaran

- Name tiga top buka-source agen observabilitas platforms dan mereka licenses.
- Distinguish apa setiap satu adalah strongest pada: Langfuse (prompt mgmt + sesi), Phoenix (RAG + auto-instrumentation), Opik (optimization + guardrail).
- Jelaskan mengapa 89% dari organizations report having agen observabilitas di place oleh 2026.
- Implementasikan sebuah stdlib trace-ke-dashboard pipeline dengan LLM-judge evaluasi.

## Masalahnya

OTel GenAI (Pelajaran 23) memberi kamu schema. Kamu tetap butuhkan platform itu ingests spans, berjalan evaluations, stores prompt versions, dan surfaces regressions. tiga contenders setiap emphasize berbeda bagian dari lifecycle.

## Konsepnya

### Langfuse (MIT)

- 6M+ SDK installs/month, 19k+ GitHub stars.
- Features: tracing, prompt management dengan versioning + playground, evaluations (LLM-sebagai-judge, user feedback, custom), sesi replays.
- June 2025: formerly commercial modules (LLM-sebagai-sebuah-judge, annotation queues, prompt experiments, Playground) buka-sourced di bawah MIT.
- Strongest untuk: ujung-ke-ujung observabilitas dengan tight prompt-management loop.

### Arize Phoenix (Elastic License 2.0)

- Deeper agen-specific evaluasi: trace clustering, anomaly detection, retrieval relevancy untuk RAG.
- Native OpenInference auto-instrumentation.
- Pairs dengan managed Arize AX untuk produksi.
- Tidak prompt versioning — positioned sebagai sebuah drift/behavioral-regression tool alongside broader platforms.
- Strongest untuk: RAG relevancy, behavioral drift, anomaly detection.

### Comet Opik (Apache 2.0)

- Automated prompt optimization melalui SEBUAH/B experiments.
- Guardrail (PII redaction, topical constraints).
- LLM-judge hallucination detection.
- Benchmark dari Comet's own measurement: Opik logs + eval di 23.44s vs Langfuse 327.15s (~14x gap) — ambil vendor benchmark sebagai directional.
- Strongest untuk: optimization loop, automated experimentation, guardrail enforcement.

### Industry data

Per Maxim (2026 field analysis): 89% dari organizations memiliki agen observabilitas di place; quality issues adalah top produksi barrier (32% dari respondents cite mereka).

### Picking satu

| Need | Pick |
|------|------|
| All-in-one with prompt management | Langfuse |
| Deep RAG evaluation + drift | Phoenix |
| Automated optimization + guardrails | Opik |
| Open licensing, no ELv2 | Langfuse (MIT) or Opik (Apache 2.0) |
| Datadog / New Relic integration | Any — they all export OTel |

### Di mana ini pola goes salah

- **No eval strategy.** Tracing tanpa evaluasi adalah just expensive logging.
- **Self-rolled LLM-judge without grounding.** CRITIC pola (Pelajaran 05) applies — judges butuhkan eksternal tool untuk factual verifikasi.
- **Prompt versions not tied to traces.** Ketika prod regresses, kamu cannot bisect ke prompt itu caused itu.

```figure
wb-trace-ingest
```

## Bangun Sendiri

`code/main.py` implements sebuah stdlib trace collector + LLM-judge evaluator:

- Ingest GenAI-shaped spans.
- Group oleh sesi, tag failed berjalan (pemicu guardrail, low-confidence eval).
- SEBUAH scripted LLM-judge itu scores agen responses pada sebuah rubric.
- SEBUAH dashboard-like summary: kegagalan rate, top kegagalan reasons, eval score distribution.

Jalankan:

```
python3 code/main.py
```

Output: per-sesi eval scores dan kegagalan categorization matching apa Langfuse/Phoenix/Opik akan tunjukkan.

## Gunakan

- **Langfuse** mandiri-hosted atau cloud; wire via OTel atau mereka SDK.
- **Arize Phoenix** mandiri-hosted; auto-instrument OpenInference.
- **Comet Opik** mandiri-hosted atau cloud; automated optimization loop.
- **Datadog LLM Observability** untuk mixed ops+ML teams itu sudah jalankan Datadog.

## Rilis

`outputs/skill-obs-platform-wiring.md` picks sebuah platform dan wires traces + eval + prompt versions menjadi sebuah existing agen.

## Latihan

1. Export sebuah week dari OTel traces ke Langfuse cloud (free tier). Yang sesi failed? Mengapa?
2. Tulis sebuah LLM-judge rubric untuk kamu domain (factual correctness, tone, scope adherence). Tes pada 50 traces.
3. Bandingkan Langfuse prompt versioning terhadap Phoenix's trace clustering. Yang tells kamu apa broke faster?
4. Baca Opik's guardrail docs. Wire sebuah PII redaction guardrail ke satu dari kamu agen berjalan.
5. Benchmark tiga pada kamu corpus. Ignore vendor-published numbers; measure kamu own.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tracing | "Spans collector" | Ingest OTel / SDK spans; index by session |
| Prompt management | "Prompt CMS" | Versioned prompts tied to traces |
| LLM-as-judge | "Automated eval" | Separate LLM scores agent output against a rubric |
| Session replay | "Trace playback" | Step through past runs for debugging |
| RAG relevancy | "Retrieval quality" | Does the retrieved context match the query |
| Trace clustering | "Behavioral grouping" | Cluster similar runs for drift detection |
| Guardrail enforcement | "Policy at log time" | PII/toxicity/scope checks on logged content |

## Bacaan Lanjutan

- [Langfuse docs](https://langfuse.com/) — tracing, eval, prompt mgmt
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — auto-instrumentation, drift
- [Comet Opik](https://www.comet.com/site/products/opik/) — optimization + guardrail
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — schema semua tiga consume
