# OpenAI Agen SDK: Handoffs, Guardrail, Tracing

> OpenAI Agen SDK adalah lightweight multi-agen framework built pada Responses API. Lima primitives: Agen, Handoff, Guardrail, Sesi, Tracing. Handoffs adalah tool named `transfer_to_<agent>`. Guardrail trip pada input atau output. Tracing adalah pada oleh default.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Name lima primitives dari OpenAI Agen SDK.
- Jelaskan handoffs: mengapa mereka adalah modeled sebagai tool, apa name bentuk model sees, dan bagaimana context transfers.
- Distinguish input guardrail, output guardrail, dan tool guardrail; jelaskan `run_in_parallel` vs blocking mode.
- Implementasikan sebuah stdlib runtime dengan handoffs + guardrail + span-style tracing.

## Masalahnya

Agen itu cannot delegasikan cleanly berakhir up stuffing everything menjadi satu prompt. Agen tanpa guardrail ship PII, kebijakan-violating output, atau loop forever. OpenAI's SDK codifies tiga primitives itu buat multi-agen kerja tractable.

## Konsepnya

### Lima primitives

1. **Agent.** LLM + instructions + tool + handoffs.
2. **Handoff.** Delegasi ke another agen. Represented ke model sebagai sebuah tool named `transfer_to_<agent_name>`.
3. **Guardrail.** Validation pada input (pertama agen hanya), output (terakhir agen hanya), atau tool invocation (per function tool).
4. **Session.** Automatic conversation history lintas giliran.
5. **Tracing.** Bawaan spans untuk LLM generations, pemanggilan tool, handoffs, guardrail.

### Handoffs sebagai tool

 model sees `transfer_to_billing_agent` di miliknya tool list. Pemanggilan itu signals runtime ke:

1. Copy conversation context (atau collapse itu via `nest_handoff_history` beta).
2. Initialize target agen dengan miliknya instructions.
3. Lanjutkan jalankan dengan target agen.

Ini adalah supervisor pola (Pelajaran 13 / Pelajaran 28) productized.

### Guardrail

Tiga flavors:

- **Input guardrails.** Jalankan pada pertama agen's input. Reject unsafe atau out-dari-scope requests sebelum apa pun LLM panggil.
- **Output guardrails.** Jalankan pada terakhir agen's output. Catch PII leaks, kebijakan violations, malformed responses.
- **Tool guardrails.** Jalankan per-function-tool. Validate arguments, check permissions, audit execution.

Mode:

- **Parallel** (default). Guardrail LLM berjalan alongside utama LLM. Lower tail latency. Jika tripped, utama LLM's kerja adalah discarded (token waste).
- **Blocking** (`run_in_parallel=False`). Guardrail LLM berjalan pertama. Jika tripped, tidak token wasted pada utama panggil.

Tripwires raise `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`.

### Tracing

Pada oleh default. Setiap LLM generation, pemanggilan tool, handoff, dan guardrail meng-emit sebuah span. `OPENAI_AGENTS_DISABLE_TRACING=1` opts out. `add_trace_processor(processor)` fans spans ke kamu own backend alongside OpenAI's.

### Sesi

`Session` stores conversation history di sebuah backend (SQLite, Redis, custom). `Runner.run(agent, input, session=session)` auto-loads dan appends.

### Di mana ini pola goes salah

- **Handoff drift.** Agen SEBUAH hands off ke Agen B yang hands back ke Agen SEBUAH. Tambahkan sebuah hop counter.
- **Guardrail bypass.** Tool guardrail hanya fire pada function tool; bawaan tool (file reader, web fetch) butuhkan terpisah kebijakan.
- **Over-tracing.** Sensitive content di spans. Pair dengan OTel GenAI content-capture rules (Pelajaran 23) — store externally, reference oleh ID.

```figure
ae-agent-handoff
```

## Bangun Sendiri

`code/main.py` implements SDK bentuk di stdlib:

- `Agent`, `FunctionTool`, `Handoff` (sebagai sebuah function tool dengan transfer semantics).
- `Runner` dengan input/output/tool guardrail, handoff dispatch, dan hop counter.
- SEBUAH sederhana span emitter ke tunjukkan trace bentuk.
- SEBUAH triage agen itu hands off ke billing atau dukung based pada user's query; pemicu guardrail pada satu input.

Jalankan:

```
python3 code/main.py
```

 trace menunjukkan dua successful handoffs, satu input guardrail trip, dan sebuah span tree mirroring apa nyata SDK meng-emit.

## Gunakan

- **OpenAI Agents SDK** untuk OpenAI-pertama products.
- **Claude Agent SDK** (Pelajaran 17) untuk Claude-pertama products.
- **LangGraph** (Pelajaran 13) ketika kamu inginkan eksplisit state dan durable resume.
- **Custom** ketika kamu butuhkan exact kontrol (suara, multi-provider, federated deployments).

## Rilis

`outputs/skill-agents-sdk-scaffold.md` scaffolds sebuah agens SDK app dengan sebuah triage agen, handoffs, input/output/tool guardrail, sesi store, dan sebuah trace processor.

## Latihan

1. Tambahkan sebuah handoff hop counter: refuse setelah N transfers. Trace perilaku.
2. Implementasikan `nest_handoff_history` sebagai sebuah option — collapse prior messages menjadi satu summary sebelum transferring.
3. Tulis sebuah blocking output guardrail. Bandingkan latency pada prompts itu akan trip itu vs ones itu pass.
4. Wire `add_trace_processor` ke sebuah JSON logger. Apa bentuk melakukan itu emit per span?
5. Baca SDK docs. Port kamu stdlib toy ke `openai-agents-python`. Apa melakukan kamu model salah?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "LLM + instructions" | Agent type in the SDK; owns tools and handoffs |
| Handoff | "Transfer" | Tool the model calls to delegate to another agent |
| Guardrail | "Policy check" | Validation on input / output / tool invocation |
| Tripwire | "Guardrail trip" | Exception raised when guardrail rejects |
| Session | "History store" | Conversation memory persisted between runs |
| Tracing | "Spans" | Built-in observability over LLM + tool + handoff + guardrail |
| Blocking guardrail | "Sequential check" | Guardrail runs first; no token waste on trip |
| Parallel guardrail | "Concurrent check" | Guardrail runs alongside; lower latency, wastes tokens on trip |

## Bacaan Lanjutan

- [OpenAI Agen SDK docs](https://openai.github.io/openai-agents-python/) — primitives, handoffs, guardrail, tracing
- [Claude Agen SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude-flavored counterpart
- [Anthropic, Building Efektif Agen](https://www.anthropic.com/research/building-effective-agents) — ketika ke reach untuk handoffs pada semua
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — standard Agen SDK spans map ke
