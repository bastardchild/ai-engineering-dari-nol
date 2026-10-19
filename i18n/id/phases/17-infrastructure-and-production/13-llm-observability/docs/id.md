# Pemilihan Stack Observabilitas LLM

> Pasar observabilitas 2026 terbagi dua kategori. Platform pengembangan (LangSmith, Langfuse, Comet Opik) membundel monitoring dengan eval, manajemen prompt, replay sesi. Tool gateway/instrumentasi (Helicone, SigNoz, OpenLLMetry, Phoenix) fokus pada telemetri. Langfuse berlisensi MIT pada inti dengan keseimbangan OSS kuat (50K event/bulan gratis di cloud). Phoenix OpenTelemetry-native di bawah Elastic License 2.0 — bagus untuk visualisasi drift/RAG, bukan backend produksi persisten. Arize AX memakai integrasi Iceberg/Parquet tanpa-salinan dengan klaim 100x lebih murah daripada observabilitas monolitik. LangSmith memimpin untuk LangChain/LangGraph, $39/pengguna/bulan, self-host hanya di Enterprise. Helicone berbasis-proxy dengan setup 15-30 menit, 100K req/bulan gratis, tetapi kurang dalam pada trace agen. Pola produksi umum: Gateway (Helicone/Portkey) + platform eval (Phoenix/TruLens) direkatkan oleh OpenTelemetry.

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Bedakan platform pengembangan (terbundel: eval + prompt + sesi) dari tool gateway/telemetri (hanya trace + metrik).
- Petakan enam tool utama (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) ke lisensi, harga, dan kasus penggunaan sweet-spot-nya.
- Jelaskan pola lem-OpenTelemetry yang memungkinkanmu menggabung tool gateway dengan platform eval terpisah.
- Sebutkan pembeda biaya 2026 (pendekatan tanpa-salinan Arize AX vs ingest monolitik) dan nyatakan pengali kasar 100x.

## Masalahnya

Kamu merilis fitur LLM. Ia bekerja. Kamu tidak punya visibilitas ke kegagalan prompt, loop tool, regresi latensi, lonjakan biaya, atau hit rate prompt-cache. Kamu mencari "LLM observability" dan mendapat delapan tool yang semuanya mengklaim memecahkan masalah sama pada tiga titik harga berbeda.

Mereka tidak memecahkan masalah sama. LangSmith menjawab "mengapa run LangGraph ini gagal?" Phoenix menjawab "apakah pipeline RAG-ku drifting?" Helicone menjawab "aplikasi mana yang membakar token?" Langfuse menjawab "bisakah aku men-self-host semuanya?" Tool berbeda, audiens berbeda.

Memilih melibatkan empat sumbu: stack (LangChain? SDK mentah? multi-vendor?), toleransi lisensi (hanya MIT? Elastic OK? komersial boleh?), budget (tier gratis? $100/bulan? $1000/bulan?), dan self-host (wajib? nice-to-have? tidak pernah?).

## Konsepnya

### Dua kategori

**Platform pengembangan** membundel observabilitas dengan eval, manajemen prompt, versioning dataset, replay sesi. Kamu menjalankan eksperimen, melihat prompt mana yang bekerja, meregresi-dataset prompt baru terhadap pemenang lama. LangSmith, Langfuse, Comet Opik.

**Tool gateway/telemetri** menginstrumentasikan panggilan inferensi — prompt, respons, token, latensi, model, biaya. Helicone, SigNoz, OpenLLMetry, Phoenix. Minimalis. Bisa digabung dengan tool eval terpisah via OpenTelemetry.

### Langfuse — keseimbangan OSS

- Inti berlisensi Apache / MIT; self-host via Docker.
- Tier gratis cloud: 50K event/bulan. Berbayar: $29/bulan untuk tim.
- Eval, manajemen prompt, trace, dataset. Cakupan wajar atas keempat fitur dev-platform.
- Sweet spot: kamu menginginkan fitur kelas-LangSmith tetapi harus self-host atau bertahan pada lisensi OSS.

### Phoenix (Arize) — telemetri-dulu, OpenTelemetry-native

- Elastic License 2.0; self-host sepele.
- Bagus pada visualisasi RAG dan drift. Scatter plot ruang-embedding tersedia kelas-satu.
- Tidak dirancang sebagai backend produksi persisten — terutama observabilitas waktu-pengembangan.
- Sweet spot: pengembangan pipeline RAG, debugging drift, dipasangkan dengan gateway terpisah untuk produksi.

### Arize AX — permainan skala

- Komersial. Integrasi data lake tanpa-salinan via Iceberg/Parquet.
- Klaim ~100x lebih murah daripada observabilitas monolitik (kelas-Datadog) pada skala. Matematikanya: kamu menyimpan trace di Parquet milikmu pada S3; Arize membaca langsung.
- Sweet spot: >10M trace/hari, data lake sudah ada, menginginkan dasbor spesifik-LLM tanpa harga Datadog.

### LangSmith — LangChain/LangGraph dulu

- Komersial, $39/pengguna/bulan. Self-host hanya pada Enterprise.
- Terbaik di kelasnya untuk stack LangChain dan LangGraph. Kalau kamu tidak memakai keduanya, ia kurang menarik.
- Sweet spot: tim yang berkomitmen ke LangChain, bersedia membayar.

### Helicone — minimum viable berbasis-proxy

- Setup 15-30 menit dengan menukar `OPENAI_API_BASE`-mu ke proxy Helicone.
- Berlisensi MIT; 100K req/bulan gratis, berbayar $20/bulan+.
- Termasuk failover, caching, rate limit — bertindak sebagai gateway juga.
- Kurang dalam pada trace agen / multi-langkah.
- Sweet spot: mulai cepat, aplikasi stack-tunggal, butuh gateway + observabilitas dalam satu.

### Opik (Comet) — dev platform OSS

- Apache 2.0, sepenuhnya OSS.
- Himpunan fitur mirip Langfuse dengan heritage Comet.
- Sweet spot: tim ML yang sudah memakai Comet, menginginkan observabilitas LLM di panel sama.

### SigNoz — APM penuh OpenTelemetry-first

- Apache 2.0. Menangani APM umum plus LLM via OpenTelemetry.
- Sweet spot: observabilitas terpadu lintas layanan dan panggilan LLM.

### Lemnya: OpenTelemetry + konvensi semantik GenAI

OpenTelemetry mempublikasikan konvensi semantik GenAI pada akhir 2025 (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`). Tool yang mengonsumsi OTel bisa berinterop. Pola produksi yang muncul:

1. Emisikan OTel dengan konvensi GenAI dari setiap panggilan LLM.
2. Rutekan ke gateway (Helicone / Portkey) untuk harian.
3. Kirim-ganda ke platform eval (Phoenix / Langfuse) untuk regresi.
4. Arsipkan di data lake (Iceberg) untuk analisis jangka-panjang via Arize AX atau DuckDB.

### Jebakan: menginstrumentasikan pada lapisan yang salah

Menginstrumentasikan di dalam framework agenmu (mis. menambah trace LangSmith) menggandengmu ke framework itu. Menginstrumentasikan pada lapisan HTTP/OpenAI-SDK (via OpenLLMetry atau gateway-mu) bersifat portabel.

### Sampling — kamu tidak bisa menyimpan semuanya

Pada >1M request/hari, retensi trace-penuh lebih mahal daripada panggilan LLM-nya. Sampel menurut aturan: 100% error, 100% biaya-tinggi, 5% sukses. Pertahankan agregat selalu; pertahankan mentah untuk ekor panjang.

### Angka yang perlu kamu ingat

- Cloud gratis Langfuse: 50K event/bulan.
- LangSmith: $39/pengguna/bulan.
- Gratis Helicone: 100K req/bulan.
- Klaim Arize AX: ~100x lebih murah daripada monolitik pada skala.
- Konvensi GenAI OpenTelemetry: rilis 2025, diadopsi luas 2026.

```figure
i4-otel-glue
```

## Gunakan

`code/main.py` mensimulasikan hari 1M-trace lintas strategi retensi (ingest 100%, sampling, sampling + error). Melaporkan biaya storage dan apa yang hilang pada masing-masing.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-observability-stack.md`. Dengan stack, skala, budget, posture lisensi, ia memilih tool.

## Latihan

1. Timmu memakai LangChain dan menginginkan observabilitas OSS self-hosted. Pilih Langfuse atau Opik dan justifikasi.
2. Pada 5M trace/hari dengan kuotasi Datadog $150K/bulan, hitung titik-impas untuk Arize AX.
3. Rancang himpunan atribut GenAI OpenTelemetry yang harus diwajibkan guideline org-mu pada setiap panggilan LLM.
4. Argumenkan apakah Phoenix saja cukup untuk produksi. Kapan ia tidak cukup?
5. Helicone menambah overhead proxy 20 ms. Pada P99 TTFT 300 ms, apakah itu dapat diterima? Bagaimana kalau SLA 100 ms?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OpenLLMetry | "OTel for LLMs" | Open-source OpenTelemetry instrumentation for LLMs |
| GenAI conventions | "OTel attributes" | Standard OTel attribute names for LLM calls |
| LangSmith | "LangChain observability" | Commercial platform bundled with LangChain ecosystem |
| Langfuse | "OSS LangSmith" | MIT OSS with similar feature set |
| Phoenix | "Arize dev tool" | OpenTelemetry-native dev/eval platform |
| Arize AX | "scale observability" | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | "proxy observability" | HTTP proxy collecting LLM telemetry + gateway features |
| Opik | "Comet LLM" | Apache 2.0 OSS dev platform from Comet |
| Session replay | "trace rerun" | Replay a full agent session with tool calls |
| Eval | "offline test" | Running candidate model/prompt over labeled dataset |

## Further Reading

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
