# OpenTelemetry GenAI — Menelusuri Panggilan Tool Ujung-ke-Ujung

> Sebuah agent memanggil lima tool, tiga server MCP, dan dua sub-agent. Kamu membutuhkan satu trace untuk semuanya. Semantic convention OpenTelemetry GenAI (atribut stabil di v1.37 ke atas) adalah standar 2026, yang didukung secara native oleh Datadog, Langfuse, Arize Phoenix, OpenLLMetry, dan AgentOps. Pelajaran ini menamai atribut yang wajib ada, menelusuri hierarki span (agent → LLM → tool), dan mengirimkan span emitter stdlib yang bisa kamu tancapkan ke OTel exporter mana pun.

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Menamai atribut OTel GenAI yang wajib untuk sebuah span LLM dan sebuah span eksekusi tool.
- Membangun hierarki trace yang mencakup agent loop, panggilan LLM, panggilan tool, dan dispatch klien MCP.
- Memutuskan konten apa yang ditangkap (opt-in) vs diredaksi (default).
- Membangkitkan span ke collector lokal (Jaeger, Langfuse) tanpa menulis ulang kode tool.

## Masalahnya

Sebuah debug dari Februari 2026: pengguna melaporkan "agent saya kadang butuh 30 detik untuk merespons; di waktu lain 3 detik." Tidak ada trace. Log menunjukkan panggilan LLM, tetapi bukan dispatch tool, bukan round-trip server MCP, bukan sub-agent. Kamu menebak. Akhirnya kamu menemukan: satu server MCP kadang menggantung pada cold-start.

Tanpa tracing ujung-ke-ujung, kamu tidak bisa menemukan ini. OTel GenAI memecahkannya.

Convention ini mantap pada 2025-2026 di bawah grup semantic-conventions OpenTelemetry. Ia mendefinisikan nama atribut yang stabil sehingga Datadog, Langfuse, Phoenix, OpenLLMetry, dan AgentOps semuanya mem-parse span yang sama. Instrumentasi sekali; kirim ke backend mana pun.

## Konsepnya

### Hierarki span

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

Seluruhnya bersarang di bawah satu trace id. Span id menautkan hubungan parent-child.

### Atribut yang wajib

Menurut semconv 2025-2026:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — string model yang diminta (mis. `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` — model yang benar-benar melayani.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — id respons provider untuk korelasi.

Untuk span tool:

- `gen_ai.tool.name` — pengenal tool.
- `gen_ai.tool.call.id` — id panggilan yang spesifik.
- `gen_ai.tool.description` — deskripsi tool (opsional).

Untuk span agent:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Jenis span

- `SpanKind.CLIENT` untuk panggilan yang melewati batas proses (provider LLM, server MCP).
- `SpanKind.INTERNAL` untuk langkah loop agent itu sendiri dan eksekusi tool.

### Penangkapan konten opt-in

Secara default, span membawa metrik dan timing — bukan prompt atau completion. Payload besar dan PII nonaktif secara default. Setel `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` dan env var penangkapan konten spesifik untuk menyertakan konten. Tinjau dengan hati-hati sebelum mengaktifkannya di prod.

### Events pada span

Event level-token dapat ditambahkan sebagai span event:

- `gen_ai.content.prompt` — pesan input.
- `gen_ai.content.completion` — pesan output.
- `gen_ai.content.tool_call` — panggilan tool sebagaimana direkam.

Event diurutkan waktu di dalam sebuah span untuk replay terperinci.

### Exporters

Span OTel diekspor ke:

- **Jaeger / Tempo.** OSS, on-prem.
- **Langfuse.** Khusus observabilitas LLM; memvisualisasikan penggunaan token.
- **Arize Phoenix.** Evals + tracing digabung.
- **Datadog.** Komersial; mem-parse atribut `gen_ai.*` secara native.
- **Honeycomb.** Berorientasi kolom; ramah query.

Semuanya berbicara OTLP, format wire-mu. Kodenya tidak peduli.

### Propagasi lintas MCP

Ketika klien MCP memanggil server, suntikkan header traceparent W3C ke permintaan. Streamable HTTP mendukung header standar. Stdio tidak membawa header HTTP secara native; roadmap spesifikasi 2026 membahas penambahan field `_meta.traceparent` pada panggilan JSON-RPC.

Sampai itu dirilis: sertakan traceparent di `_meta` setiap permintaan secara manual. Server mencatat trace id.

### Metrik

Bersama span, semconv GenAI mendefinisikan metrik:

- `gen_ai.client.token.usage` — histogram.
- `gen_ai.client.operation.duration` — histogram.
- `gen_ai.tool.execution.duration` — histogram.

Gunakan ini untuk dashboard yang tidak membutuhkan detail per-panggilan.

### Lapisan AgentOps

AgentOps (didirikan 2024) berspesialisasi dalam observabilitas GenAI. Ia membungkus framework populer (LangGraph, Pydantic AI, CrewAI) untuk memancarkan span OTel secara otomatis. Berguna jika stack-mu memakai framework yang didukung; jika tidak, gunakan instrumentasi manual.

```figure
t3-span-waterfall
```

## Gunakan

`code/main.py` memancarkan span berbentuk OTel ke stdout (dalam format mirip OTLP-JSON) untuk sebuah agent yang memanggil sebuah LLM, meneruskan dua tool, dan melakukan satu round-trip MCP. Tidak ada exporter sungguhan — pelajaran berfokus pada bentuk span dan himpunan atribut. Tempel output ke viewer yang kompatif OTLP atau cukup baca langsung.

Yang perlu diperhatikan:

- Trace id dibagikan ke semua span.
- Link parent-child dikodekan via `parentSpanId`.
- Atribut wajib `gen_ai.*` terisi.
- Penangkapan konten nonaktif secara default; satu skenario menyalakannya lewat env var.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-otel-genai-instrumentation.md`. Diberi codebase agent, skill ini menghasilkan rencana instrumentasi: di mana menambahkan span, atribut mana yang harus diisi, dan exporter mana yang harus dibidik.

## Latihan

1. Jalankan `code/main.py`. Hitung span-nya dan identifikasi mana yang CLIENT vs INTERNAL.

2. Nyalakan penangkapan konten (env var) dan konfirmasikan event `gen_ai.content.prompt` dan `gen_ai.content.completion` muncul. Catat implikasinya bagi PII.

3. Tambahkan metrik eksekusi tool `gen_ai.tool.execution.duration` dan pancarkan sebagai sampel histogram per panggilan.

4. Propagasikan sebuah traceparent dari span agent induk ke field `_meta.traceparent` pada permintaan MCP. Verifikasi bahwa server MCP akan melihat trace id yang sama.

5. Baca spesifikasi semconv OTel GenAI. Identifikasi satu atribut yang terdaftar di semconv tetapi TIDAK dipancarkan oleh kode pelajaran ini. Tambahkan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | Open standard for traces, metrics, logs |
| GenAI semconv | "GenAI semantic conventions" | Stable attribute names for LLM / tool / agent spans |
| `gen_ai.*` | "The attribute namespace" | All GenAI attributes share this prefix |
| Span | "Timed operation" | A unit of work with a start, end, and attributes |
| Trace | "Cross-span ancestry" | Tree of spans sharing a trace id |
| SpanKind | "CLIENT / SERVER / INTERNAL" | Hints about span direction |
| OTLP | "OpenTelemetry Line Protocol" | Wire format for exporters |
| Opt-in content | "Prompt / completion capture" | Off by default; env var to enable |
| traceparent | "W3C header" | Propagates trace context across services |
| Exporter | "Backend-specific shipper" | Component that sends spans to Jaeger / Datadog / etc. |

## Bacaan Lanjutan

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — convention kanonik untuk span, metrik, dan event GenAI
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — daftar atribut span LLM dan eksekusi tool
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — span `invoke_agent` level-agent
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — sumber kebenaran yang di-host GitHub
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — walkthrough integrasi production
