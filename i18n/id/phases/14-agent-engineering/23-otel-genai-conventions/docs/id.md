# OpenTelemetry GenAI Semantic Conventions

> OpenTelemetry's GenAI SIG (launched April 2024) defines standard schema untuk agen telemetry. Span names, attributes, dan content-capture rules converge lintas vendors so agen traces mean sama hal di Datadog, Grafana, Jaeger, dan Honeycomb.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Name GenAI span categories: model/klien, agen, tool.
- Distinguish `invoke_agent` KLIEN vs INTERNAL spans dan ketika setiap applies.
- List top-level GenAI attributes: provider name, request model, data-source ID.
- Jelaskan content-capture kontrak: opt-di, `OTEL_SEMCONV_STABILITY_OPT_IN`, eksternal-reference recommendation.

## Masalahnya

Setiap vendor invents mereka own span names. Ops teams berakhir up building per-framework dashboards. OpenTelemetry's GenAI SIG fixes ini oleh defining satu standard seluruh ecosystem targets.

## Konsepnya

### Span categories

1. **Model / client spans.** Cover raw LLM memanggil. Emitted oleh provider SDKs (Anthropic, OpenAI, Bedrock) dan framework model adapters.
2. **Agent spans.** `create_agent` (ketika agen itu adalah constructed) dan `invoke_agent` (ketika itu berjalan).
3. **Tool spans.** Satu per tool invocation; connected ke agen itu span oleh parent-child relation.

### Agen span naming

- Span name: `invoke_agent {gen_ai.agent.name}` jika named; fallback ke `invoke_agent`.
- Span kind:
  - **CLIENT** — untuk remote agen services (OpenAI Assistants API, Bedrock Agen).
  - **INTERNAL** — untuk di-process agen framework (LangChain, CrewAI, lokal ReAct).

### Kunci attributes

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — model ID.
- `gen_ai.response.model` — resolved model (dapat differ dari request due ke routing).
- `gen_ai.agent.name` — agen identifier.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — untuk RAG: yang corpus atau store adalah consulted.

Technology-specific conventions exist untuk Anthropic, Azure AI Inference, AWS Bedrock, OpenAI.

### Content capture

 default aturan: instrumentations SEBAIKNYA TIDAK capture inputs/outputs oleh default. Capture adalah opt-di via:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

Recommended produksi pola: store content externally (S3, kamu log store), record references pada spans (pointer IDs, tidak prose). Ini adalah Pelajaran 27 content-poisoning pertahanan wired menjadi observabilitas.

### Stability

Sebagian besar conventions adalah experimental sebagai dari March 2026. Opt di ke stable preview dengan:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ maps GenAI attributes natively menjadi miliknya LLM Observabilitas schema. Other backends (Grafana, Honeycomb, Jaeger) dukung raw attributes.

### Di mana ini pola goes salah

- **Capturing full prompts in spans.** PII, secrets, customer data di traces itu ops bisa baca. Store externally.
- **No `gen_ai.provider.name`.** Multi-provider dashboards break ketika attribution adalah missing.
- **Spans without parent links.** Orphaned tool spans. Selalu propagate context.
- **Not setting stability opt-in.** Kamu attributes dapat dapatkan renamed pada backend upgrade.

```figure
ae-genai-span-tree
```

## Bangun Sendiri

`code/main.py` implements sebuah stdlib span emitter matching GenAI conventions:

- `Span` dengan GenAI attribute schema.
- `Tracer` dengan `start_span`, nested contexts.
- SEBUAH scripted agen jalankan itu meng-emit: `create_agent`, `invoke_agent` (INTERNAL), per-tool spans, `chat` spans untuk LLM memanggil.
- SEBUAH content-capture mode itu stores prompts externally dan records IDs pada spans.

Jalankan:

```
python3 code/main.py
```

Output: sebuah span tree dengan semua required GenAI attributes, dan sebuah "eksternal store" showing opt-di content references.

## Gunakan

- **Datadog LLM Observability** (v1.37+) maps attributes natively.
- **Langfuse / Phoenix / Opik** (Pelajaran 24) — auto-instrument ecosystem.
- **Jaeger / Honeycomb / Grafana Tempo** — raw OTel traces; build dashboards dari GenAI attributes.
- **Self-hosted** — jalankan OTel Collector dengan sebuah GenAI processor.

## Rilis

`outputs/skill-otel-genai.md` wires OTel GenAI spans menjadi sebuah existing agen dengan content-capture defaults dan eksternal-reference storage.

## Latihan

1. Instrument kamu Pelajaran 01 ReAct loop dengan `invoke_agent` (INTERNAL) + per-tool spans. Kirim ke sebuah Jaeger instance.
2. Add content capture di "references hanya" mode: prompts ke SQLite, span attributes carry hanya row IDs.
3. Baca spec untuk `gen_ai.data_source.id`. Wire itu menjadi kamu Pelajaran 09 Mem0 pencarian.
4. Set `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` dan verifikasi kamu attributes don't dapatkan renamed oleh collector.
5. Build sebuah dashboard: "yang tool error correlate dengan yang model" dari GenAI attributes alone.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI group" | OTel working group defining the schema |
| invoke_agent | "Agent span" | Name of the span representing an agent run |
| CLIENT span | "Remote call" | Span for a call to a remote agent service |
| INTERNAL span | "In-process" | Span for an in-process agent run |
| gen_ai.provider.name | "Provider" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG source" | Which corpus/store a retrieval hit |
| Content capture | "Prompt logging" | Opt-in capture of messages; store externally in prod |
| Stability opt-in | "Preview mode" | Env var to pin experimental conventions |

## Bacaan Lanjutan

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — spec
- [OpenAI Agen SDK](https://openai.github.io/openai-agents-python/) — GenAI spans oleh default
- [AutoGen v0.4 (Microsoft Riset)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — OTel spans built di
- [Claude Agen SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context propagation
