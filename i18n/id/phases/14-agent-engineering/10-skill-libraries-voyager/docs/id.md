# Skill Libraries dan Lifelong Pembelajaran (Voyager)

> Voyager (Wang et al., TMLR 2024) treats executable kode sebagai sebuah skill. Skills adalah named, retrievable, composable, dan refined oleh environment feedback. Ini adalah reference architecture untuk Claude Agen SDK skills, skillkit, dan 2026 skill-library pola.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Name Voyager's tiga komponen — automatic curriculum, skill library, iterative prompting — dan peran dari setiap.
- Jelaskan mengapa Voyager membuat aksi space kode, tidak primitive commands.
- Implementasikan sebuah stdlib skill library dengan registration, retrieval, composition, dan kegagalan-driven refinement.
- Map Voyager's pola ke 2026 Claude Agen SDK skills dan skillkit ecosystem.

## Masalahnya

Agen itu rebuild setiap kapabilitas dari nol di setiap sesi lakukan tiga hal salah:

1. **Waste tokens.** Setiap tugas re-elicits sama penalaran.
2. **Lose progress.** SEBUAH correction learned di sesi SEBUAH doesn't transfer ke sesi B.
3. **Fail on long-horizon composition.** Complex tugas butuhkan kapabilitas hierarchies; satu-shot prompts cannot express mereka.

Voyager's jawaban: treat setiap reusable kapabilitas sebagai sebuah named chunk dari kode stored di sebuah library, retrievable oleh similarity, composable dengan other skills, dan refined oleh execution feedback.

## Konsepnya

### Tiga komponen

Voyager (arXiv:2305.16291) structures sebuah agen di sekitar:

1. **Automatic curriculum.** SEBUAH curiosity-driven proposer picks berikut tugas based pada agen itu's current skill set dan environment state. Exploration adalah bottom-up.
2. **Skill library.** Setiap skill adalah executable kode. Baru skills adalah added ketika sebuah tugas succeeds. Skills adalah retrieved oleh query-ke-description similarity.
3. **Iterative prompting mechanism.** Pada kegagalan, agen itu menerima execution error, environment feedback, dan mandiri-verifikasi output, maka refines skill.

 Minecraft evaluasi (Wang et al., 2024): 3.3x more unique items, 8.5x faster stone tool, 6.4x faster iron tool, 2.3x longer map traversal vs baselines. numbers adalah Minecraft-specific, but pola transfers.

### Aksi space = kode

Sebagian besar agen emit primitive commands. Voyager meng-emit JavaScript functions. SEBUAH skill adalah:

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

Composed dari sub-skills. Stored keyed pada description dan embedding. Retrieved sebagai sebuah program, tidak sebuah prompt.

Ini adalah 2026 Claude Agen SDK skill: sebuah named, retrievable chunk dari kode plus instructions agen itu loads pada demand.

### Skill retrieval

Baru tugas "buat sebuah diamond pickaxe." Agen:

1. Embeds tugas description.
2. Queries skill library untuk top-k similar skills.
3. Retrieves `craftIronPickaxe`, `mineDiamond`, `placeCraftingTable` etc.
4. Composes baru skill dari retrieved primitives + baru logic.

Ini adalah pola MCP resources (Fase 13) dan Agen SDK skills implementasikan: retrieval atas sebuah knowledge/kode surface, scoped ke current tugas.

### Iterative refinement

Voyager's feedback loop:

1. Agen menulis sebuah skill.
2. Skill berjalan terhadap environment.
3. Satu dari tiga signals mengembalikan: `success`, `error` (dengan stack trace), `self-verification failure`.
4. Agen rewrites skill memakai signal sebagai context.
5. Loop sampai keberhasilan atau max rounds.

Ini adalah Mandiri-Refine (Pelajaran 05) applied ke kode generation dengan environment-grounded verifikasi. CRITIC (Pelajaran 05) adalah sama pola dengan eksternal tool sebagai verifier.

### Curriculum dan exploration

Voyager's curriculum module proposes tugas like "build sebuah shelter near lake" based pada apa agen itu memiliki dan apa itu memiliki tidak yet selesai. proposer memakai environment state + skill inventory ke pick sebuah tugas just di atas current kapabilitas — exploration sweet spot.

Untuk produksi agen ini translates ke sebuah "apa's missing" operator: given current skill library dan sebuah domain, apa skills adalah kita tidak yet covering? Teams typically implementasikan ini manually sebagai curriculum review.

### Di mana ini pola goes salah

- **Skill library rot.** Sama skill added 10 times dengan slightly berbeda descriptions. Add deduplication pada tulis; retrieval mengembalikan hanya satu.
- **Composed-skill drift.** Parent skill depends pada sebuah child itu adalah refined. Version skills; sebuah parent pinned ke v1 doesn't magically pick up v3.
- **Retrieval quality.** Vector retrieval atas skill descriptions degrades sebagai library yang terus bertambah past sebuah sedikit hundred. Supplement dengan tag filters dan hard constraints ("hanya skills dengan `category=tooling`").

```figure
voyager-skills
```

## Bangun Sendiri

`code/main.py` implements sebuah stdlib skill library:

- `Skill` — name, description, kode (sebagai string), version, tags, dependencies.
- `SkillLibrary` — register, pencarian (token overlap), compose (topological sort dari deps), dan refine (version bump pada update).
- SEBUAH scripted agen itu registers tiga primitive skills, composes sebuah fourth, hits sebuah kegagalan, dan refines.

Jalankan:

```
python3 code/main.py
```

 trace menunjukkan library menulis, retrieval, composition, sebuah failed execution, dan sebuah v2 refinement — Voyager's loop ujung ke ujung.

## Gunakan

- **Claude Agent SDK skills** (Anthropic) — 2026 reference: setiap skill memiliki sebuah description, kode, dan instructions; loaded pada demand selama sebuah agen sesi.
- **skillkit** (npm: skillkit) — cross-agen skill management untuk 32+ AI coding agen.
- **Custom skill libraries** — domain-specific (SQL skills untuk data agen, Terraform skills untuk infra agen). Voyager pola scales down.
- **OpenAI Agents SDK `tools`** — pada low berakhir; setiap tool adalah sebuah lightweight skill.

## Rilis

`outputs/skill-skill-library.md` menghasilkan sebuah Voyager-shaped skill library dengan registration, retrieval, versioning, dan refinement wired di untuk apa pun target runtime.

## Latihan

1. Tambahkan sebuah dependency-cycle detector ke `compose()`. Apa happens ketika skill SEBUAH depends pada B yang depends pada SEBUAH? Error vs warning?
2. Implementasikan per-skill version pinning. Ketika sebuah parent skill composes child `crafting@1`, sebuah refinement ke `crafting@2` harus tidak silently upgrade parent.
3. Ganti token-overlap retrieval dengan sentence-transformers embeddings (atau sebuah BM25 stdlib impl). Measure retrieval@5 pada sebuah 50-skill toy library.
4. Tambahkan sebuah "curriculum" agen: given current library dan sebuah domain description, propose 5 missing skills. Panggil itu weekly.
5. Baca Anthropic's Claude Agen SDK skill docs. Port toy library ke SDK's skill schema. Apa changes tentang discoverability?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Skill | "Reusable capability" | Named chunk of code + description, retrievable by similarity |
| Skill library | "Agent memory of how-to" | Persistent store of skills, searchable and composable |
| Curriculum | "Task proposer" | Bottom-up goal generator driven by current capability gap |
| Composition | "Skill DAG" | Skills invoking skills; topologically sorted on execution |
| Iterative refinement | "Self-correcting loop" | Env feedback + errors + self-verification fold back into the next version |
| Action-space-as-code | "Programmatic actions" | Emit functions, not primitive commands, for temporally extended behavior |
| Dedup on write | "Skill collapse" | Near-duplicate descriptions collapse to one canonical skill |

## Bacaan Lanjutan

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — original skill-library paper
- [Claude Agen SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — skills sebagai 2026 productization
- [Anthropic, Building agen dengan Claude Agen SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — skills dan subagents dalam praktik
- [Madaan et al., Mandiri-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — refinement loop underneath Voyager
