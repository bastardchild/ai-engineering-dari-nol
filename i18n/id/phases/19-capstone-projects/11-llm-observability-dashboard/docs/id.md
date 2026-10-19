# Capstone 11 — Observability & Dashboard Eval LLM

> Langfuse menjadi open-core. Arize Phoenix mempublikasikan pemetaan semconv GenAI 2026. Helicone dan Braintrust sama-sama menambah porsi pada atribusi biaya per pengguna. OpenLLMetry milik Traceloop menjadi instrumen SDK de-facto. Bentuk produksinya adalah ClickHouse untuk trace, Postgres untuk metadata, Next.js untuk UI, dan pasukan kecil eval job (DeepEval, RAGAS, LLM-judge) yang berjalan di atas trace ber-sampling. Bangun satu yang self-hosted, ingesti dari setidaknya empat keluarga SDK, dan demonstrasikan penangkapan regresi yang disuntikkan dalam waktu kurang dari lima menit.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P17 · P18
**Time:** 25 hours

## Masalahnya

Setiap tim AI yang menjalankan traffic produksi pada 2026 mempertahankan plane observability di samping modelnya. Atribusi biaya. Deteksi halusinasi. Monitoring drift. Sinyal jailbreak. Dashboard SLO. Peringatan kebocoran PII. Referensi open-source — Langfuse, Phoenix, OpenLLMetry — konvergen pada semantic convention GenAI dari OpenTelemetry sebagai skema ingesti. Kini kamu bisa menginstrumentasi OpenAI, Anthropic, Google, LangChain, LlamaIndex, dan vLLM dengan satu SDK dan mengirim span yang kompatibel.

Kamu akan membangun dashboard self-hosted yang meng-ingest dari setidaknya empat keluarga SDK, menjalankan sekumpulan kecil eval job di atas trace ber-sampling, mendeteksi drift, dan memberi peringatan. Ukurannya: mengingat regresi yang disuntikkan secara sengaja (sebuah prompt yang mulai menghasilkan PII), dashboard menangkapnya dan memicu peringatan dalam waktu kurang dari lima menit.

## Konsepnya

Ingesti memakai OTLP HTTP. SDK menghasilkan span semconv GenAI: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`. Span mendarat di ClickHouse untuk analitik kolomnik; metadata (pengguna, sesi, aplikasi) mendarat di Postgres.

Eval berjalan sebagai job batch di atas trace ber-sampling. DeepEval menilai faithfulness, toxicity, dan answer relevance. RAGAS menilai metrik retrieval ketika trace membawa konteks retrieval. LLM-judge kustom menjalankan pemeriksaan spesifik-domain (kebocoran PII, respons di luar kebijakan). Run eval menulis kembali ke ClickHouse yang sama sebagai span eval yang tertaut ke trace induk.

Deteksi drift mengawasi distribusi ruang embedding dari waktu ke waktu (divergensi PSI atau KL pada embedding prompt) ditambah tren skor eval. Peringatan diberi makan ke Prometheus Alertmanager lalu Slack / PagerDuty. UI-nya adalah Next.js 15 dengan Recharts.

## Arsitektur

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## Stack

- Ingest: SDK OpenTelemetry + semantic convention GenAI; transport OTLP HTTP
- Collector: OpenTelemetry Collector dengan prosesor tail-sampling (untuk kontrol biaya)
- Penyimpanan: ClickHouse untuk span, Postgres untuk metadata, S3 untuk arsip event mentah
- Eval: DeepEval, RAGAS 0.2, paket evaluator Arize Phoenix, LLM-judge kustom
- Drift: PSI / KL pada embedding prompt gabungan (sentence-transformers) mingguan
- Alerting: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- SDK yang didukung out of the box: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

```figure
ce-otel-drift
```

## Bangun Sendiri

1. **Konfigurasi collector.** OpenTelemetry Collector dengan receiver OTLP HTTP, tail-sampler yang menyimpan 100% trace ber-error dan 10% trace sukses, serta exporter ke ClickHouse dan S3.

2. **Skema ClickHouse.** Tabel `spans` dengan kolom yang mencerminkan semconv GenAI: `gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id`, plus kantong JSON untuk payload panjang. Tambahkan indeks sekunder berdasarkan user_id dan app_id.

3. **Test cakupan SDK.** Tulis aplikasi klien kecil yang memakai setiap SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) dengan auto-instrumentasi OpenLLMetry. Verifikasi masing-masing menghasilkan span GenAI kanonis yang mendarat di ClickHouse.

4. **Job eval.** Job terjadwal membaca trace ber-sampling 15 menit terakhir dan menjalankan DeepEval untuk faithfulness, toxicity, dan answer relevance. Output-nya adalah span eval yang tertaut ke trace induk.

5. **LLM-judge kustom.** Seorang judge kebocoran PII: diberi sebuah respons, memanggil guard LLM untuk menilai kemungkinan kebocoran PII. Respons berskor tinggi masuk ke antrean triase.

6. **Deteksi drift.** Job mingguan menghitung PSI antara embedding prompt gabungan minggu ini dengan baseline 4 minggu terakhir. Jika PSI di atas ambang, beri peringatan.

7. **Dashboard.** Next.js 15 dengan halaman-halaman: ringkasan (span/detik, biaya/pengguna, latensi p95), trace (pencarian + waterfall), eval (tren faithfulness, toxicity), drift (PSI dari waktu ke waktu), peringatan.

8. **Rantai alerting.** Prometheus exporter membaca agregat skor eval dan persentil latensi; Alertmanager mengarahkan ke Slack untuk peringatan dan PagerDuty untuk pelanggaran kritis.

9. **Probe regresi.** Suntikkan bug: chatbot yang dievaluasi mulai membocorkan SSN palsu 1% dari waktu. Ukur MTTR: dari bug di-deploy sampai peringatan Slack.

## Gunakan

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Rilis

`outputs/skill-llm-observability.md` adalah deliverable-nya. Diberi sebuah aplikasi LLM, dashboard meng-ingest trace-nya, menjalankan eval, memperingatkan soal drift, dan menampilkan rincian biaya/pengguna di Next.js.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema coverage | Number of SDK families producing canonical GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | MTTR on injected regression (under 5 minutes target) |
| 20 | Cost / scale | Sustained ingest at 1k spans/sec without backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain exercised end to end |
| **100** | | |

## Latihan

1. Tambahkan instrumentasi kustom untuk framework Haystack. Verifikasi span kanonis mendarat di ClickHouse dengan atribut `gen_ai.*` yang setia.

2. Tukar DeepEval dengan evaluator Phoenix pada trace yang sama. Ukur drift skor antara dua mesin eval itu.

3. Tajamkan detektor drift: hitung PSI per app-id alih-alih global. Tampilkan jejak drift per aplikasi.

4. Tambahkan halaman "user impact": biaya-per-pengguna dan laju-kegagalan-per-pengguna dengan sparkline.

5. Bangun kebijakan tail-sampling yang menyimpan 100% trace dengan toxicity > 0.5 plus sampel berlapis 10% dari sisanya. Ukur bias sampling yang diperkenalkan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry spec for LLM span attributes (system, model, tokens) |
| Tail sampling | "Post-trace sample" | Collector decides to keep or drop a trace after it completes (can peek errors) |
| PSI | "Population stability index" | Drift metric comparing two distributions; > 0.2 typically signals meaningful drift |
| LLM-judge | "Eval as model" | An LLM scoring another LLM's output on a rubric (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Keep-rule" | Rule that decides which traces to persist vs drop; errored + sample-rate |
| Eval span | "Linked eval trace" | Child span carrying an eval score linked to the original LLM call span |
| Cost per user | "Unit economics" | Dollar cost attributed to a user_id over a window; key product metric |

## Bacaan Lanjutan

- [Langfuse](https://github.com/langfuse/langfuse) — the reference open-core observability platform
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — alternate reference with strong drift support
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — auto-instrumentation SDK family
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the ingest schema
- [Helicone](https://www.helicone.ai) — alternate hosted observability
- [Braintrust](https://www.braintrust.dev) — alternate eval-first platform
- [ClickHouse documentation](https://clickhouse.com/docs) — columnar span store
- [DeepEval](https://github.com/confident-ai/deepeval) — evaluator library
