# Produksi Runtimes: Queue, Event, Cron

> Produksi agen jalankan pada six runtime shapes: request-response, streaming, durable execution, queue-based background, event-driven, dan scheduled. Pick bentuk sebelum kamu pick framework. Observabilitas adalah penopang beban pada setiap bentuk.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Name six produksi runtime shapes dan match setiap ke sebuah framework / product pola.
- Jelaskan mengapa durable execution (LangGraph) matters untuk long-horizon tugas.
- Deskripsikan event-driven runtime dan ketika Claude Managed Agen fits.
- Jelaskan observabilitas-sebagai-penopang beban claim untuk multi-langkah agen.

## Masalahnya

Produksi agen gagal di ways sebuah Jupyter notebook doesn't surface: network timeouts pada langkah 37, user hangs up mid-suara panggil, cron job dies pada machine reboot, background worker berjalan out dari memori. runtime bentuk determines yang kegagalan adalah survivable.

## Konsepnya

### Request-response

- Synchronous HTTP. User waits untuk completion.
- Hanya viable untuk short tugas (<30s).
- Stacks: Agno (Python + FastAPI), Mastra (TypeScript + Express/Hono/Fastify/Koa).
- Observabilitas: standard HTTP access logs + OTel spans.

### Streaming

- SSE atau WebSocket untuk progressive output.
- LiveKit Perluass ini ke WebRTC untuk suara/video (Pelajaran 22).
- Stacks: apa pun framework dengan streaming dukung + sebuah frontend itu menangani SSE/WS.
- Observabilitas: per-chunk timing, pertama-token latency, tail latency.

### Durable execution

- State checkpointed setelah setiap langkah; auto-resumes pada kegagalan.
- AutoGen v0.4 aktor model isolates kegagalan ke satu agen (Pelajaran 14).
- LangGraph's core differentiator (Pelajaran 13).
- Essential ketika langkah count adalah unknown dan recovery cost adalah high.

### Queue-based / background

- Job enters sebuah queue, workers pick up, hasil flow back via webhooks atau pub/sub.
- Essential untuk long-horizon agen (puluhan hingga ratusan langkah per tugas, per Anthropic's komputer pakai announcement).
- Stacks: Celery (Python), BullMQ (Node), SQS + Lambda (AWS), custom.
- Observabilitas: queue depth, per-job latency distribution, DLQ size.

### Event-driven

- Agen subscribe ke triggers: baru email, PR opened, cron fire.
- Claude Managed Agen covers ini out dari box (Pelajaran 17).
- CrewAI Flow (Pelajaran 15) structures event-driven deterministik workflow.
- Observabilitas: trigger source, event-ke-mulai latency, agen latency.

### Scheduled

- Cron-shaped agen itu jalankan periodically.
- Combine dengan durable execution so sebuah failing nightly jalankan resumes berikut tick.
- Stacks: Kubernetes CronJob + sebuah durable framework; hosted (Render cron, Vercel cron).

### 2026 deployment polas

- **CrewAI Flows** untuk event-driven produksi.
- **Agno** stateless FastAPI untuk Python microservices.
- **Mastra** server adapters (Express, Hono, Fastify, Koa) untuk embedding.
- **Pipecat Cloud / LiveKit Cloud** untuk managed suara (Pelajaran 22).
- **Claude Managed Agents** untuk hosted long-berjalan async.

### Observabilitas adalah penopang beban

Tanpa OpenTelemetry GenAI spans (Pelajaran 23) plus sebuah Langfuse/Phoenix/Opik backend (Pelajaran 24), kamu cannot debug sebuah multi-langkah agen itu failed pada langkah 40. Ini adalah tidak optional untuk produksi. Itu's difference antara "kita debug cepat" dan "kita replay dari nol dengan more logging."

### Di mana produksi runtimes gagal

- **Wrong shape choice.** Picking request-response untuk sebuah 5-minute tugas. Users hang up; workers pile up; retries compound.
- **No DLQ.** Queue workers tanpa dead-letter. Failed jobs vanish.
- **Opaque background work.** Background agen berjalan tanpa trace export. Kegagalan adalah invisible sampai user reports mereka.
- **Skipping durable state.** Apa pun jalankan > 30 seconds di mana kamu bisa't afford ke restart membutuhkan durable execution.

```figure
wb-runtime-shapes
```

## Bangun Sendiri

`code/main.py` adalah sebuah stdlib multi-bentuk demo:

- Request-response endpoint (plain function).
- Streaming handler (generator).
- Queue-based worker dengan DLQ.
- Event trigger registry.
- Cron-shaped scheduler.

Jalankan:

```bash
python3 code/main.py
```

Output: lima traces showing setiap bentuk's perilaku pada sama tugas. Sama agen logic, berbeda outer shells. Durable execution ( sixth bentuk) adalah intentionally covered di Pelajaran 13 dengan LangGraph checkpointing.

## Gunakan

- **Request-response** untuk chat-style UX.
- **Streaming** untuk progressive responses.
- **Durable** untuk long-horizon tugas.
- **Queue** untuk batch / async / long-berjalan.
- **Event** untuk agen reactivity.
- **Cron** untuk housekeeping (memori consolidation, eval, cost reports).

## Rilis

`outputs/skill-runtime-shape.md` picks sebuah runtime bentuk untuk sebuah tugas dan wires observabilitas requirements.

## Latihan

1. Port kamu Pelajaran 01 ReAct loop ke semua six shapes di kamu stack. Yang bentuk fits yang product surface?
2. Tambahkan sebuah DLQ ke queue-based demo. Simulate 10% job kegagalan; surface DLQ size.
3. Tulis sebuah cron-triggered eval agen itu berjalan nightly terhadap kamu top 20 traces dari day.
4. Implementasikan streaming dengan backpressure: jika klien adalah lambat, pause agen itu. Bagaimana ini interact dengan sebuah bujet giliran?
5. Baca Claude Managed Agen docs. Ketika akan kamu move sebuah mandiri-hosted long-horizon agen ke managed?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Request-response | "Synchronous" | User waits; short tasks only |
| Streaming | "SSE / WS" | Progressive output; better UX; latency observable per chunk |
| Durable execution | "Resume from failure" | Checkpointed state; restart at last step |
| Queue-based | "Background jobs" | Producer / worker pool / DLQ |
| Event-driven | "Trigger-based" | Agent reacts to external events |
| DLQ | "Dead-letter queue" | Parking lot for failed jobs |
| Claude Managed Agents | "Hosted harness" | Anthropic-hosted long-running async with caching + compaction |

## Bacaan Lanjutan

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — durable execution details
- [Claude Managed Agen overview](https://platform.claude.com/docs/en/managed-agents/overview) — hosted long-berjalan async
- [Anthropic, Introducing komputer pakai](https://www.anthropic.com/news/3-5-models-and-computer-use) — "puluhan hingga ratusan langkah per tugas"
- [AutoGen v0.4 (Microsoft Riset)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — aktor-model fault isolasi
