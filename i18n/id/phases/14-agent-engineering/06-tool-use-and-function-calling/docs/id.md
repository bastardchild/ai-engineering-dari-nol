# Tool Pakai dan Function Pemanggilan

> Toolformer (Schick et al., 2023) started mandiri-supervised tool annotation. Berkeley Function Pemanggilan Leaderboard V4 (Patil et al., 2025) sets 2026 bar: 40% agentic, 30% multi-giliran, 10% live, 10% non-live, 10% hallucination. Tunggal-giliran adalah solved. Memori, dynamic decision-making, dan long-horizon tool chains adalah tidak.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan Toolformer's mandiri-supervised training signal: keep tool annotations hanya ketika execution mengurangi berikut-token loss.
- Name BFCL V4's lima evaluasi categories dan apa setiap measures.
- Implementasikan sebuah stdlib registri tool dengan schema validation, argument coercion, dan execution sandboxing.
- Diagnose tiga 2026 buka problems: long-horizon tool chaining, dynamic decision-making, dan memori.

## Masalahnya

Early tool pakai asked: bisa model predict sebuah benar function panggil? Modern tool pakai bertanya: bisa model chain tool lintas 40 langkah, dengan memori, dengan partial observabilitas, dengan recovery dari tool kegagalan, tanpa hallucinating tool itu lakukan tidak exist?

Toolformer established baseline: model bisa pelajari ketika ke panggil tool dengan mandiri-supervision. BFCL V4 defines 2026 evaluasi target. gap antara mereka adalah space produksi agen live di.

## Konsepnya

### Toolformer (Schick et al., NeurIPS 2023)

Ide: let model annotate miliknya own pretraining corpus dengan candidate API memanggil. Untuk setiap candidate, execute itu. Keep annotation hanya jika termasuk tool hasil mengurangi loss pada berikut token. Fine-tune pada filtered corpus.

Tool covered: calculator, QA sistem, pencarian engines, translator, calendar. mandiri-supervision signal adalah purely tentang whether tool membantu predict text — tidak manusia labels.

Scale hasil: tool pakai emerges pada scale. Smaller model hurt dari tool annotations; larger model gain. Ini adalah mengapa 2026 frontier model memiliki strong tool pakai baked di sementara sebagian besar 7B model butuhkan tool eksplisit-pakai fine-tuning ke menjadi reliable.

### Berkeley Function Pemanggilan Leaderboard V4 (Patil et al., ICML 2025)

BFCL adalah 2026 de facto evaluasi. V4 composition:

- **Agentic (40%)** — penuh agen trajectories: memori, multi-giliran, dynamic decisions.
- **Multi-Turn (30%)** — interactive conversations dengan tool chains.
- **Live (10%)** — user-submitted nyata prompts (harder distribution).
- **Non-Live (10%)** — synthetic tes cases.
- **Hallucination (10%)** — detect ketika tidak tool sebaiknya menjadi called.

V3 memperkenalkan state-based evaluasi: setelah sebuah tool sequence, check API's actual state (e.g. "adalah file created?") rather than match AST dari pemanggilan tool. V4 added web pencarian, memori, dan format sensitivity categories.

Kunci 2026 finding: tunggal-giliran function pemanggilan adalah near-solved. Kegagalan concentrate di memori (carrying context lintas giliran), dynamic decision-making (choosing tool based pada prior hasil), long-horizon chains (drift setelah 20+ langkah), dan hallucination detection (refusing ke panggil ketika tidak tool fits).

### Tool schema

Setiap provider memiliki sebuah schema. Mereka differ di details but share sama bentuk:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic memakai `input_schema` directly. OpenAI memakai `function.parameters`. Keduanya accept JSON Schema. Descriptions adalah penopang beban — model membaca mereka ke pick right tool. Buruk tool descriptions adalah #1 root cause dari salah-tool-picked kegagalan.

### Argument validation

Trust tidak pemanggilan tool. Validate:

1. **Type coercion.** Model dapat kembalikan sebuah string "5" di mana schema menyatakan int. Coerce jika unambiguous; reject jika tidak.
2. **Enum validation.** Jika schema menyatakan `status in {"open", "closed"}` dan model meng-emit `"in_progress"`, reject dengan sebuah descriptive error.
3. **Required fields.** Missing required field -> immediate error observasi back ke model, bukan sebuah crash.
4. **Format validation.** Dates, emails, URLs — validate dengan concrete parsers, tidak regex.

Setiap validation kegagalan sebaiknya kembalikan sebuah structured observasi so model bisa retry dengan benar bentuk.

### Paralel pemanggilan tool

Modern providers dukung paralel pemanggilan tool di satu assistant giliran. loop:

1. Model meng-emit 3 pemanggilan tool dengan distinct `tool_use_id`s.
2. Runtime executes mereka (di paralel jika independent).
3. Setiap hasil goes back sebagai sebuah `tool_result` block correlated oleh `tool_use_id`.

Engineering aturan: treat correlation IDs sebagai penopang beban. Tukarm dan kamu dapatkan salah-tool-ke-salah-hasil routing.

### Sandboxing

Tool execution adalah sandbox boundary. See Pelajaran 09 untuk detail. Short version: setiap tool sebaiknya specify baca/tulis surface, network access, timeout, memori cap. Generic `run_shell(cmd)` adalah sebuah red flag; specific `git_status()` adalah safer.

```figure
tool-routing
```

## Bangun Sendiri

`code/main.py` implements sebuah produksi-bentuk registri tool:

- JSON Schema subset validator (stdlib hanya).
- Tool registration dengan description, input schema, timeout, dan executor.
- Argument coercion dan enum validation.
- Paralel tool dispatch dengan correlation IDs.
- Error observasi sebagai structured strings.

Jalankan:

```
python3 code/main.py
```

 trace menunjukkan sebuah mini agen pemanggilan tiga tool di satu giliran, dengan satu deliberately malformed panggil itu adalah rejected dengan sebuah descriptive error model bisa bertindak pada.

## Gunakan

Setiap provider memiliki miliknya own tool schema — Anthropic, OpenAI, Gemini, Bedrock. Pakai sebuah translation layer (OpenAI Agen SDK, Vercel AI SDK, LangChain tool adapter) jika kamu butuhkan multi-provider. BFCL adalah reference benchmark — jalankan terhadap kamu agen sebelum shipping jika tool pakai adalah pusat ke product.

## Rilis

`outputs/skill-tool-registry.md` menghasilkan sebuah tool catalog, schema, dan registry untuk sebuah given tugas domain. Mencakup description-quality checks (melakukan setiap tool's description tell model ketika ke pakai itu?).

## Latihan

1. Tambahkan sebuah "tidak-op" tool itu lets model explicitly refuse ke pakai apa pun other tool. Measure pada sebuah BFCL-like hallucination tes.
2. Implementasikan sebuahrgument coercion untuk int-sebagai-string dan float-sebagai-string. Di mana melakukan coercion mulai ke hide nyata bugs?
3. Tambahkan sebuah per-tool timeout dan sebuah circuit breaker (refuse tool untuk 60s setelah 3 consecutive kegagalan). Apa melakukan ini change tentang bagaimana model pulih?
4. Baca BFCL V4 description. Pick satu category (e.g. "multi-giliran") dan jalankan 10 contoh prompts melalui kamu agen. Report pass rate.
5. Port stdlib validator ke Pydantic atau Zod. Apa melakukan Pydantic/Zod catch itu toy missed?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | "Tool use" | Structured-output tool invocation with validated schema |
| Toolformer | "Self-supervised tool annotation" | Schick 2023 — keep tool calls whose results reduce next-token loss |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 benchmark: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "Function signature for the model" | name, description, JSON Schema of arguments |
| tool_use_id | "Correlation ID" | Ties a tool call to its result; essential for parallel dispatch |
| Hallucination detection | "Know when not to call" | V4 category: refuse to call when no tool fits |
| Argument coercion | "String-to-int repair" | Narrow fixes for predictable schema-mismatch; reject if ambiguous |
| Sandboxing | "Tool execution boundary" | Per-tool read/write surface, network, timeout, memory cap |

## Bacaan Lanjutan

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — mandiri-supervised tool annotation
- [Berkeley Function Pemanggilan Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 2026 eval benchmark
- [Anthropic, Tool pakai dokumentasi](https://platform.claude.com/docs/en/agent-sdk/overview) — produksi tool schema di Claude Agen SDK
- [OpenAI Agen SDK docs](https://openai.github.io/openai-agents-python/) — function tool type dan Guardrail
