# Produksi Agen Runtimes — Cepat Instantiation dan Typed Workflow

> SEBUAH produksi agen runtime optimizes apa prototyping framework ignore: instantiation cost, typed workflow surfaces, dan sebuah serving-ready backend. 2026 pairing: Agno (Python) aims pada microsecond agen instantiation dan stateless FastAPI backends. Mastra ships agen, tool, workflow, unified model routing, dan composite storage pada Vercel AI SDK substrate.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## Tujuan Pembelajaran

- Identifikasi Agno's performa targets dan ketika mereka matter.
- Name Mastra's tiga primitives — Agen, Tool, Workflow — dan supported server adapters.
- Jelaskan mengapa sebuah stateless sesi-scoped FastAPI backend adalah recommended Agno produksi path.
- Pick Agno vs Mastra untuk sebuah given stack (Python-pertama vs TypeScript-pertama).

## Masalahnya

LangGraph, AutoGen, CrewAI adalah framework-heavy. Teams itu inginkan "just agen itu loop, cepat, di my runtime" reach untuk Agno (Python) atau Mastra (TypeScript). Keduanya trade beberapa dari framework-owned primitives untuk raw speed dan sebuah tighter fit ke surrounding stack.

## Konsepnya

### Agno

- Python runtime, formerly Phi-data.
- "Tidak graphs, chains, atau convoluted polas — just pure python."
- Performa targets dari mereka docs: ~2μs agen instantiation, ~3.75 KiB memori per agen, ~23 model providers.
- Produksi path: stateless sesi-scoped FastAPI backend. Setiap request memulai sebuah fresh agen; sesi state lives di sebuah DB.
- Native multimodal (text, image, audio, video, file) dan agentic RAG.

 speed targets matter ketika kamu punya thousands dari short-lived agen per kedua (chat fan-di, evaluasi pipelines). Mereka matter less ketika satu agen berjalan untuk 10 minutes.

### Mastra

- TypeScript, built pada Vercel AI SDK.
- Tiga primitives: **Agen**, **Tool** (Zod-typed), **Workflow**.
- Unified Model Router — 3,300+ model lintas 94 providers (March 2026).
- Composite storage: memori, workflow, observabilitas ke berbeda backends; ClickHouse recommended untuk observabilitas pada scale.
- Apache 2.0 dengan `ee/` directories di bawah source-available enterprise license.
- Server adapters untuk Express, Hono, Fastify, Koa; pertama-class Berikut.js dan Astro integration.
- Ships Mastra Studio (localhost:4111) untuk debugging.
- 22k+ GitHub stars, 300k+ weekly npm downloads pada 1.0 (Jan 2026).

### Positioning

Neither adalah trying ke menjadi LangGraph. Mereka compete pada:

- **Language fit.** Agno untuk Python-pertama teams; Mastra untuk TypeScript-pertama.
- **Runtime ergonomics.** Agno = near-zero overhead; Mastra = integrated dengan Vercel ecosystem.
- **Observability.** Keduanya integrate dengan Langfuse/Phoenix/Opik (Pelajaran 24) but Mastra Studio adalah pertama-party.

### Ketika ke pick setiap

- **Agno** — Python backend, banyak short-lived agen, strong perf requirements, FastAPI shop.
- **Mastra** — TypeScript backend, Berikut.js / Vercel deploy, unified multi-provider model routing, Zod-typed tool.
- **LangGraph** (Pelajaran 13) — ketika durable state dan eksplisit graf penalaran matter more than raw speed.
- **OpenAI / Claude Agent SDK** — ketika kamu inginkan provider's productized bentuk (Pelajaran 16–17).

### Di mana ini pola goes salah

- **Perf-for-perf's-sake.** Picking Agno karena "2μs" sounds baik ketika workload adalah satu lambat agen panggil per request. Overhead adalah tidak bottleneck.
- **Ecosystem lock-in.** Mastra's Vercel-flavored integration adalah sebuah plus pada Vercel, sebuah minus elsewhere.
- **Enterprise license confusion.** Mastra's `ee/` directories adalah source-available, tidak Apache 2.0. Baca licenses jika kamu're perencanaan ke fork.

```figure
wb-runtime-spawn
```

## Bangun Sendiri

Ini pelajaran adalah primarily comparative — tidak tunggal kode artefak akan lakukan keduanya framework justice. See `code/main.py` untuk sebuah side-oleh-side toy: sebuah minimal "jalankan sebuah agen, stream output, persist sesi" flow implemented twice (once Agno-shaped, once Mastra-shaped).

Jalankan:

```
python3 code/main.py
```

Dua structurally berbeda but functionally equivalent traces.

## Gunakan

- **Agno** — Python backend itu membutuhkan speed dan FastAPI bentuk.
- **Mastra** — TypeScript backend dengan banyak providers dan workflow primitives.
- Keduanya ship pertama-party observabilitas hooks. Keduanya integrate dengan Langfuse.

## Rilis

`outputs/skill-runtime-picker.md` picks Agno, Mastra, LangGraph, atau sebuah provider SDK based pada stack, latency budget, dan bentuk operasional.

## Latihan

1. Baca Agno's docs. Port stdlib ReAct loop (Pelajaran 01) ke Agno. Apa disappeared? Apa stayed?
2. Baca Mastra's docs. Port sama loop ke Mastra. Apa changed di tool typing (Zod vs nothing)?
3. Benchmark: measure agen instantiation latency pada kamu stack. Melakukan Agno's 2μs matter ke kamu workload?
4. Design sebuah migration: jika kamu've menjadi berjalan CrewAI di Python, Apa yang rusak jika kamu move ke Agno?
5. Baca Mastra's `ee/` license terms. Apa restrictions akan affect sebuah buka-source fork?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | Single client for 3,300+ models across 94 providers |
| Composite storage | "Multiple backends" | Memory/workflows/observability each to a different store |
| Mastra Studio | "Local debugger" | localhost:4111 UI for introspecting agents |
| Source-available | "Not OSS" | License permits source reading but restricts commercial use |

## Bacaan Lanjutan

- [Agno Agen Framework docs](https://www.agno.com/agent-framework) — performa targets, FastAPI integration
- [Mastra docs](https://mastra.ai/docs) — primitives, server adapters, Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — stateful-graf alternative
- [Comet Opik](https://www.comet.com/site/products/opik/) — observabilitas comparisons cited oleh Mastra integrations
