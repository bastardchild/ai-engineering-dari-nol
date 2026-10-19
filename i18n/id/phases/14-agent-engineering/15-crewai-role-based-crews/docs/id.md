# Peran-Based Agen Teams — Roles, Tugas, Processes

> Empat primitives: Agen, Tugas, Kru, Process. Dua top-level shapes: Kru (otonom, peran-based collaboration) dan Flow (event-driven, deterministik). CrewAI adalah 2026 implementasi referensi, dan miliknya docs adalah blunt: "untuk apa pun produksi-ready application, mulai dengan sebuah Flow."

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Name CrewAI's empat primitives (Agen, Tugas, Kru, Process) dan apa setiap owns.
- Distinguish Sequential, Hierarchical, dan planned Consensus process; pick satu per workload.
- Distinguish Kru (otonom peran-based) dari Flow (event-driven deterministik), dan jelaskan docs' produksi recommendation.
- Wire tool dengan `@tool` decorator dan `BaseTool` subclass; reason tentang structured outputs vs free text.
- Name empat CrewAI memori types dan ketika setiap pays off.
- Implementasikan sebuah stdlib tiga-agen kru (researcher, writer, editor) itu produces sebuah brief.
- Spot tiga CrewAI mode kegagalan: prompt-bloat, manager-LLM tax, brittle handoffs.

## Masalahnya

Teams adopting multi-agen framework hit sama wall. "Otonom collaboration" sounds great di sebuah demo. Maka sebuah customer file sebuah bug dan kamu butuhkan deterministik replay. Atau finance bertanya bagaimana much sebuah LLM-routed kru costs per jalankan. Atau pada-panggil membutuhkan ke know yang agen stalled pada 3 AM.

Free-form LLM-routed kru jawaban none dari itu cleanly. Pure DAGs jawaban mereka semua but lose exploratory bentuk sebuah brainstorming agen membutuhkan.

CrewAI's split adalah honest tentang trade. Kru untuk collaborative, peran-based, exploratory kerja. Flow untuk event-driven, kode-owned, auditable produksi. Sama framework, dua shapes, pick per surface.

## Konsepnya

### Empat primitives

CrewAI's surface adalah kecil. Memorize ini dan rest adalah config.

- **Agent.** `role + goal + backstory + tools + (optional) llm`. backstory adalah penopang beban. Itu shapes tone, judgment, ketika agen itu berhenti. Tool adalah functions agen itu bisa panggil (more di bawah).
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`. SEBUAH reusable unit dari kerja. `expected_output` adalah kontrak. `context` lists upstream tugas yang outputs adalah passed di. `output_pydantic` forces sebuah structured bentuk.
- **Crew.** Container. Owns list dari `agents`, list dari `tasks`, `process`, dan optional `memory` + `verbose` + `manager_llm` settings.
- **Process.** Execution strategi. Sequential, Hierarchical, Consensus (planned). Picks bentuk dari jalankan.

Agen lakukan tidak see setiap other directly. Tugas reference agen. Kru sequences tugas. Process memutuskan yang picks berikut tugas. Itu adalah seluruh mental model.

> **Validated terhadap** CrewAI 0.86 (2026-05). Newer versions dapat rename atau merge process types; check [CrewAI Processes docs](https://docs.crewai.com/concepts/processes) sebelum relying pada sebuah specific bentuk.

### Sequential vs Hierarchical vs Consensus

- **Sequential.** Tugas jalankan di declaration order. Output dari tugas N adalah available sebagai `context` ke tugas N+1. Lowest cost. Sebagian besar predictable. Pakai ketika order adalah fixed.
- **Hierarchical.** SEBUAH manager Agen (terpisah LLM panggil) routes antara specialists. CrewAI spawns manager either dari kamu `manager_llm` config atau sebuah default. manager picks berikut tugas setiap round dan bisa refuse atau re-route. Pakai ketika kamu punya empat atau more specialists dan order genuinely depends pada prior output.
- **Consensus.** Planned, tidak currently implemented di public API. docs reserve name untuk sebuah future voting-based process. Lakukan tidak rely pada itu today.

Hierarchical adds sebuah per-round LLM panggil ( manager) di atas setiap specialist panggil. Token cost bisa triple pada sebuah lima-langkah jalankan. Pay untuk itu hanya ketika kamu butuhkan routing.

### Kru vs Flow

Ini adalah framing docs lead dengan di 2026.

- **Crew.** LLM-driven autonomy. framework picks bentuk pada runtime. Baik untuk: riset, brainstorming, pertama drafts, anywhere path adalah bagian dari jawaban. Hard ke replay. Hard ke tes. Cheap ke prototype.
- **Flow.** Event-driven graf kamu own. `@start` marks entry. `@listen(topic)` marks sebuah langkah itu fires ketika another langkah meng-emit itu topic. Setiap langkah adalah plain Python (bisa panggil sebuah Kru internally). Baik untuk: produksi. Observable. Testable. Deterministik.

 docs' 2026 produksi recommendation: mulai dengan sebuah Flow. Fold Kru di sebagai `Crew.kickoff()` memanggil dari di dalam Flow langkah ketika autonomy earns miliknya cost. Flow memberi kamu audit trail, Kru memberi kamu exploration. Compose, lakukan tidak pick.

### Tool integration

Tiga ways ke beri sebuah agen sebuah tool. Pick simplest satu itu fits.

1. **`@tool` decorator.** Pure functions become tool. Signature adalah schema; docstring adalah description LLM sees. Best untuk satu-off helpers.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass.** Class-based tool dengan eksplisit args schema, async dukung, retries. Pakai ketika tool memiliki state (sebuah klien, sebuah cache) atau membutuhkan structured args.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkits.** CrewAI ships pertama-party adapters: `SerperDevTool`, `FileReadTool`, `DirectoryReadTool`, `CodeInterpreterTool`, `RagTool`, `WebsiteSearchTool`. Wired dengan satu import.

Structured outputs pakai Pydantic. Pass `output_pydantic=MyModel` pada Tugas. CrewAI validates LLM response terhadap model dan either coerces atau retries. Pair ini dengan sebuah tight `expected_output` string. Free-text outputs adalah fine untuk drafts; structured outputs adalah apa downstream Flow bisa consume.

### Memori hooks

CrewAI ships empat memori types out dari box. Mereka compose: sebuah Kru bisa memungkinkan semua empat pada once.

> **Validated terhadap** CrewAI 0.86 (2026-05). Recent releases route everything melalui sebuah unified `Memory` sistem itu wraps ini empat stores. conceptual model di bawah tetap holds, but public class surface dapat collapse ke sebuah tunggal `Memory` entry-inti di newer versions; check [CrewAI memori docs](https://docs.crewai.com/concepts/memory) untuk current API.

- **Short-term.** Conversation buffer di dalam sebuah tunggal jalankan. Wiped pada berakhir.
- **Long-term.** Persisted lintas berjalan. Stored di sebuah vector DB (Chroma oleh default, swappable). Retrieved oleh similarity ke current tugas.
- **Entity.** Per-entity facts. "Customer X adalah pada enterprise rencana." Keyed oleh entity, tidak oleh similarity. Survives lintas berjalan.
- **Contextual.** Assembly-time retrieval. Pulls relevant memori pada moment agen itu membutuhkan itu, tidak preloaded.

Memungkinkan pada Kru dengan `memory=True` atau per-type config. Backed oleh sebuah embeddings provider kamu configure (defaults ke OpenAI, swappable ke lokal). Memori adalah satu dari places CrewAI earns miliknya keep terhadap thinner framework; pure LangGraph membutuhkan kamu ke wire setiap dari ini yourself.

### Ketika peran-based teams fit

- Tiga ke six agen dengan named roles dan sebuah collaborative workflow. Drafting, reviewing, perencanaan, brainstorming.
- Routing di mana LLM's judgment tentang berikut langkah adalah bagian dari value (Hierarchical).
- Anywhere tim adalah happier reading `role + goal + backstory` than reading sebuah graf definition.

### Ketika mereka lakukan tidak

- Deterministik DAGs dengan strict ordering. Pakai LangGraph (Pelajaran 13). graf bentuk adalah right abstraction; CrewAI's peran framing adalah friction.
- Sub-kedua latency budgets. Hierarchical adds round trips. Even Sequential serializes prompts itu sertakan backstories dan prior outputs.
- Tunggal-agen loop. Skip framework; sebuah agen loop (Pelajaran 1) plus sebuah registri tool adalah shorter.

Pelajaran 17 (Agen Framework Tradeoffs) lays ini out di sebuah matrix. short version: CrewAI sits di "collaborative peran-based" corner.

### Dependency bentuk

Independent dari LangChain. Python 3.10 ke 3.13. Memakai `uv`. Star count: see [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) (snapshot sebagai dari 2026-05). AWS Bedrock integration adalah documented; vendor benchmark report sebuah substantial speedup vs LangGraph pada QA workloads, but methodology (dataset, hardware, evaluasi metrik) adalah tidak published, so treat framework-vendor numbers sebagai directional hanya.

### Di mana ini pola goes salah

- **Prompt-bloat from backstories.** SEBUAH 2000-word backstory per agen dan sebuah lima-agen kru burns context budget sebelum pertama pemanggilan tool. Keep backstories di bawah 200 words. Reuse phrases lintas agen; lakukan tidak repeat house style lima times.
- **Manager-LLM token tax.** Hierarchical process adds sebuah manager LLM panggil sebelum setiap specialist panggil. Pada sebuah lima-tugas kru itu adalah six LLM memanggil alih-alih lima, dan manager panggil carries penuh tugas list plus prior outputs. Switch ke Sequential unless routing depends pada output.
- **Brittle handoffs.** Tugas N's `expected_output` adalah "sebuah outline". Tugas N+1 membaca itu sebagai `context` dan tries ke parse tiga sections. LLM produced empat. downstream Agen ad-libs. Fix dengan `output_pydantic` pada Tugas N so Tugas N+1 membaca sebuah typed object, tidak free text.
- **Crew-as-prod.** Free-form Kru shipped ke produksi tanpa sebuah Flow wrapper. Output variability adalah high; replay adalah impossible; pada-panggil cannot diff sebuah buruk jalankan terhadap sebuah baik satu. Wrap dengan sebuah Flow.

```figure
ae-crew-vs-flow
```

## Bangun Sendiri

`code/main.py` implements stdlib versions dari keduanya shapes plus sebuah tiga-agen kru.

Bentuk:

- `Agent`, `Task` dataclasses matching CrewAI's surface.
- `SequentialCrew.kickoff(inputs)` berjalan tugas di declaration order, threading outputs sebagai `context`.
- `HierarchicalCrew.kickoff(topic)` adds sebuah manager Agen picking berikut specialist setiap round, berhenti pada "selesai".
- `Flow` dengan `@start` dan `@listen(topic)` decorators, sebuah tiny event loop, dan sebuah trace.
- `tool(name)` decorator mirroring CrewAI's `@tool` bentuk.
- `Memory` dengan `short_term`, `long_term`, `entity` stores; mocked similarity memakai numpy.
- Mock LLM responses adalah hardcoded strings keyed off peran plus input prefix. Tidak network. Deterministik.

Concrete demo: researcher, writer, editor kru producing sebuah brief pada "agen engineering 2026". Researcher pulls (mocked) sources. Writer drafts. Editor tightens. Sama kru berjalan melalui sebuah Flow ke tunjukkan deterministik bentuk.

Jalankan:

```bash
python3 code/main.py
```

Trace covers: sequential kru threading outputs melalui `context`, hierarchical kru dengan manager picks (researcher, writer, editor, maka "selesai"), flow berjalan sama tiga langkah dengan eksplisit topics (`researched`, `drafted`, `edited`), pemanggilan tool routed melalui `@tool`, dan long-term memori surviving lintas dua kickoffs.

 Kru trace adalah fluid; manager bisa di prinsip re-order. Flow trace adalah fixed. Itu choice adalah pelajaran.

## Gunakan

- **CrewAI Flow** untuk produksi. Even ketika Flow adalah satu langkah itu memanggil `Crew.kickoff()`. Flow memberi audit boundary.
- **CrewAI Crew (Sequential)** untuk clear-ordering collaborative kerja, especially pertama drafts dan review loop.
- **CrewAI Crew (Hierarchical)** ketika routing depends pada output dan kamu punya empat atau more specialists.
- **LangGraph** (Pelajaran 13) untuk eksplisit state machines, durable resume, strict ordering.
- **AutoGen v0.4** (Pelajaran 14) untuk aktor-model concurrency dan fault isolasi.
- **OpenAI Agents SDK** (Pelajaran 16) untuk OpenAI-pertama products dengan handoffs dan guardrail.
- **Claude Agent SDK** (Pelajaran 17) untuk Claude-pertama products dengan subagents dan sesi store.

## Rilis

`outputs/skill-crew-or-flow.md` picks Kru vs Flow untuk sebuah tugas dan scaffolds minimal implementasi. Hard rejects pada Kru-tanpa-backstory, Flow-tanpa-eksplisit-topics, Hierarchical dengan di bawah tiga specialists.

## Pitfalls

- **Backstory as flavor.** Itu shapes outputs. Tes tiga variants per agen; variance adalah nyata. Pick satu, freeze itu.
- **Skipping `expected_output`.** Tanpa sebuah kontrak per tugas, downstream tugas pick up whatever LLM produced. Kru berjalan; audit gagal.
- **Memory always-on.** Long-term menulis setiap jalankan. Vector DB yang terus bertambah. Retrieval mendapatkan noisy. Scope menulis ke tugas di mana fact adalah persistent.
- **Manager prompt drift.** Hierarchical's manager prompt adalah implisit. Jika routing mendapatkan weird, dump itu di verbose mode dan baca.
- **Tool side effects in Crews.** SEBUAH Kru bisa panggil sebuah tool more times than expected. POST, DELETE, payment belong di sebuah Flow langkah, tidak pernah sebuah Kru tool.

## Latihan

1. Convert Sequential kru ke sebuah Flow. Count touchpoints di mana variability drops. Note di mana readability dropped.
2. Add entity memori ke kru: facts tentang sebuah customer persist lintas kickoffs. Verifikasi retrieval pulls right entity.
3. Implementasikan sebuah Hierarchical process di mana manager refuses ke route ke editor sampai writer's output memiliki pada least tiga paragraphs. Trace retry.
4. Wire sebuah `BaseTool` subclass untuk sebuah (mocked) web pencarian. Bandingkan trace bentuk vs `@tool` decorator version.
5. Add `output_pydantic=Brief` ke editor tugas, di mana `Brief` memiliki `title`, `summary`, `sections`. Buat writer tugas output malformed JSON once; verifikasi CrewAI's retry perilaku di trace.
6. Baca CrewAI's docs intro. Port toy ke nyata `crewai` API. Yang guarantees melakukan stdlib version skip?
7. Wire AgentOps atau Langfuse (Pelajaran 24) ke sebuah nyata jalankan. Yang traces melakukan kamu miss di stdlib version?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "Persona" | Role + goal + backstory + tools |
| Task | "Unit of work" | Description + expected output + assignee + optional structured output |
| Crew | "Agent team" | Container for Agents + Tasks + Process |
| Process | "Execution strategy" | Sequential / Hierarchical / Consensus (planned) |
| Flow | "Deterministic workflow" | Event-driven, code-owned, testable |
| Backstory | "Persona prompt" | Tone and judgment shaper for the Agent |
| `@tool` | "Function tool" | Decorator that turns a function into a tool the Agent can call |
| `BaseTool` | "Class tool" | Class-based tool with args schema, retries, async support |
| Entity memory | "Per-entity facts" | Memory scoped to a customer / account / issue |
| Long-term memory | "Cross-run memory" | Vector-backed memory that survives between kickoffs |
| Contextual memory | "Just-in-time retrieval" | Memory pulled at the moment the Agent needs it |
| Manager LLM | "Router agent" | Extra LLM in Hierarchical process that picks the next task |
| `expected_output` | "Task contract" | String that tells the Agent (and audit) what shape to return |

## Bacaan Lanjutan

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): concepts dan recommended produksi path
- [CrewAI Flow panduan](https://docs.crewai.com/en/concepts/flows): event-driven bentuk, `@start`, `@listen`
- [CrewAI tool reference](https://docs.crewai.com/en/concepts/tools): `@tool`, `BaseTool`, bawaan toolkits
- [CrewAI memori](https://docs.crewai.com/en/concepts/memory): short-term, long-term, entity, contextual
- [Anthropic, Building Efektif Agen](https://www.anthropic.com/research/building-effective-agents): ketika multi-agen membantu dan ketika itu melakukan tidak
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): state-machine alternative
