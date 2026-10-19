# Memori Blocks dan Sleep-Time Compute

> Discrete functional memori blocks model bisa edit directly, dan sebuah sleep-time agen itu consolidates memori asynchronously sementara primary agen adalah idle. Ini dua ideas adalah bagaimana kamu scale memori melampaui satu conversation.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Name tiga memori tiers Letta memakai (core, recall, archival) dan peran dari setiap.
- Jelaskan memori-block pola: Manusia block, Persona block, dan user-defined blocks sebagai pertama-class typed objects.
- Deskripsikan apa sleep-time compute adalah, mengapa itu sits off kritis path, dan mengapa itu bisa jalankan sebuah stronger model than primary agen.
- Implementasikan sebuah scripted dua-agen loop di mana sebuah primary agen serves responses dan sebuah sleep-time agen consolidates blocks antara giliran.

## Masalahnya

MemGPT (Pelajaran 07) solved virtual-memori alur kontrol. Tiga produksi problems emerged:

1. **Latency.** Setiap memori operation sits pada kritis path. Jika agen itu memiliki ke prune, summarize, atau reconcile sementara user waits, tail latency blows up.
2. **Memory rot.** Menulis accumulate. Contradicted facts stay. Retrieval drowns di stale content.
3. **Structure loss.** SEBUAH flat archival store cannot express " Manusia block adalah selalu di prompt; Persona block adalah selalu di prompt; Tugas block swaps per sesi."

Letta (letta.com) adalah platform name original MemGPT project adopted di 2024 — paper's pola keeps MemGPT name — dan 2026 Letta V1 rewrite adalah sebuah later, terpisah langkah. Memori blocks buat structure eksplisit; sleep-time compute moves consolidation off kritis path.

## Konsepnya

### Tiga tiers

| Tier | Scope | Where it lives | Written by |
|------|-------|----------------|------------|
| Core | Always visible | Inside the main prompt | Agent tool call + sleep-time rewrites |
| Recall | Conversation history | Retrievable | Automatic turn logging |
| Archival | Arbitrary facts | Vector + KV + graph | Agent tool call + sleep-time ingest |

Core adalah MemGPT core. Recall adalah conversation buffer dengan miliknya evicted tail. Archival adalah eksternal store. split cleans up MemGPT's dua-tier overloading.

### Memori blocks

SEBUAH block adalah sebuah typed, persistent, editable bagian dari core tier. original MemGPT paper defined dua:

- **Human block** — facts tentang user (name, peran, preferences, goals).
- **Persona block** — agen itu's mandiri-konsep (identity, tone, constraints).

Letta generalizes ke arbitrary user-defined blocks: sebuah `Task` block untuk current tujuan, sebuah `Project` block untuk codebase facts, sebuah `Safety` block untuk hard constraints. Setiap block memiliki sebuah `id`, `label`, `value`, `limit` (character cap), `description` (so model knows ketika ke edit itu).

Blocks adalah editable via tool surface:

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — condense sebuah block itu adalah near miliknya limit.

### Sleep-time compute

 2025 Letta addition: jalankan sebuah kedua agen di background, off kritis path. Sleep-time agen process conversation transcripts dan codebase context, tulis `learned_context` menjadi shared blocks, dan consolidate atau invalidate archival records.

Properties itu fall out:

- **No latency cost.** Primary responses lakukan tidak wait untuk memori ops.
- **Stronger model allowed.**  sleep-time agen bisa menjadi sebuah more expensive, slower model karena itu adalah tidak latency-constrained.
- **Natural consolidation window.** Dedup, summarize, invalidate contradicted facts ketika user adalah tidak waiting.

 bentuk matches bagaimana humans kerja: kamu lakukan tugas, kamu sleep pada itu, long-term memori settles overnight.

### Native penalaran

Letta V1 (`letta_v1_agent`, 2026) deprecates `send_message`/heartbeat dan inline `Thought:` token di favor dari native penalaran. Responses API (OpenAI) dan Messages API dengan Perluased berpikir (Anthropic) emit penalaran pada sebuah terpisah kanal, diteruskan antar giliran (encrypted lintas providers di produksi). kontrol loop adalah tetap ReAct. thought trace adalah structural, tidak prompt-shaped.

### Di mana ini pola goes salah

- **Block bloat.** Infinite `block_append` hits limit cepat. Wire sebuah block summarizer sebelum tulis itu pushes atas cap.
- **Silent drift.** Sleep-time agen rewrites sebuah block dan primary agen tidak pernah notices. Version blocks dan surface diffs di trace.
- **Poisoned consolidation.** Sleep-time agen processes attacker-reachable content menjadi core. Pelajaran 27 applies ke sleep-time surface too.

```figure
memory-blocks
```

## Bangun Sendiri

`code/main.py` implements:

- `Block` — id, label, value, limit, description.
- `BlockStore` — CRUD + `near_limit(label)` helper.
- Dua scripted agen — `PrimaryAgent` serves sebuah giliran, `SleepTimeAgent` consolidates antara giliran.
- SEBUAH trace itu menunjukkan sebuah tiga-giliran conversation dengan block menulis, plus sebuah sleep-time pass itu summarizes sebuah block dan invalidates sebuah stale fact.

Jalankan:

```
python3 code/main.py
```

 transkrip menunjukkan split: primary giliran adalah cepat dan produce raw menulis; sleep pass compacts dan cleans up.

## Gunakan

- **Letta** (letta.com) untuk implementasi referensi. Mandiri-host atau managed cloud.
- **Claude Agent SDK skills** sebagai block-shaped knowledge — sebuah skill adalah sebuah named, versioned, retrievable block dari instructions agen itu loads pada demand.
- **Custom builds** untuk teams itu inginkan kontrol atas storage backend. Pakai Letta API kontrak so kamu bisa migrate later.

## Rilis

`outputs/skill-memory-blocks.md` menghasilkan sebuah Letta-shaped block sistem dengan sleep-time hooks untuk apa pun runtime, termasuk keamanan rules dan citation wiring.

## Latihan

1. Tambahkan sebuah `block_summarize` tool itu Gantis block value dengan sebuah model-generated summary ketika `near_limit` mengembalikan true. Yang trigger threshold minimizes keduanya summarization memanggil dan block overflow?
2. Implementasikan sleep-time dedup atas archival: dua records yang text memiliki >90% token overlap collapse ke satu. Lakukan itu hanya di sleep pass, tidak pernah pada kritis path.
3. Version blocks. Pada setiap tulis record lama value dan sebuah diff. Expose `block_history(label)` so operators bisa debug "mengapa melakukan agen itu forget X."
4. Treat sleep-time agen sebagai untrusted writers. Ketika mereka touch Persona atau Keamanan block, butuhkan sebuah kedua-agen review sebelum committing.
5. Port contoh ke pakai Letta API (`letta_v1_agent`). Apa yang berubah di block schema, dan Bagaimana native penalaran alter trace bentuk?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Memory block | "Editable prompt section" | Typed, persistent, LLM-editable segment of core memory |
| Human block | "User memory" | Facts about the user, pinned in core |
| Persona block | "Agent identity" | Self-concept, tone, constraints, pinned in core |
| Sleep-time compute | "Async memory work" | Second agent doing consolidation off the critical path |
| Core / Recall / Archival | "Tiers" | Three-layer memory split: always-visible / conversation / external |
| Block limit | "Cap" | Character limit per block; forces summarization |
| Native reasoning | "Thinking channel" | Provider-level reasoning output, not prompt-level `Thought:` |
| Learned context | "Sleep output" | Facts the sleep-time agent writes into shared blocks |

## Bacaan Lanjutan

- [Letta, Memori Blocks blog](https://www.letta.com/blog/memory-blocks) — block pola
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — async consolidation
- [Letta, Rearchitecting agen itu Loop](https://www.letta.com/blog/letta-v1-agent) — native penalaran rewrite
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — origin
