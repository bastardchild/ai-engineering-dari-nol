# Skala Produksi — Queue, Checkpoint, Durability

> Menskalakan sistem multi-agent ke ribuan run serentak menuntut **durable execution** — work queue plus checkpoint, supaya worker mana pun bisa melanjutkan run mana pun setelah crash apa pun, dengan syarat penanganan lease, side effect idempotent, dan replay deterministik sudah ada. Runtime LangGraph adalah contoh referensinya: ia menulis checkpoint setelah tiap super-step dengan kunci `thread_id` (Postgres secara default); crash worker melepas lease dan worker lain melanjutkan. Agen bisa tidur tanpa batas menunggu input manusia. **MegaAgent** (arXiv:2408.09955) menjalankan producer-consumer queue per-agen dengan tiga state (Idle / Processing / Response) dan koordinasi dua lapis (intra-group chat + inter-group admin chat). **Fiber/async** mengalahkan thread-per-job untuk streaming LLM: thread menganggur 99% waktunya menunggu token, fiber yielding secara kooperatif di I/O. Kontra-argumen: "Scaling Agentic Software" dari Ashpreet Bedi memperjuangkan **FastAPI + Postgres + tidak ada yang lain** sampai beban membuktikan sebaliknya — arsitektur sederhana melaju lebih jauh dari perkiraan. Pelajaran ini membangun log checkpoint yang tahan lama, work queue per-agen dengan transisi state, demo async-vs-thread, dan mengukir aturan pragmatis "mulai dari yang sederhana".

**Type:** Learn + Build
**Languages:** Python (stdlib, `asyncio`, `sqlite3`)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## Masalah

Prototipe sistem multi-agent berjalan di satu laptop dengan tiga agen pada event loop in-memory. Kamu pindah ke produksi:

- Agen kadang berjalan berjam-jam (riset panjang, tunggu human-in-the-loop).
- Proses worker crash. Me-restart kehilangan state.
- Beban puncak 10x rata-rata; kamu butuh horizontal scaling.
- Pengguna membayar per agent-run; kamu butuh semantik exactly-once untuk penagihan.

Event loop in-memory tidak melakukan satupun dari ini. Kamu butuh lapisan durable execution di bawahnya. Opsi kanonis 2026:

1. Workflow engine dengan checkpoint (Temporal, runtime LangGraph).
2. Message queue dengan state store (Postgres + SQS/RabbitMQ).
3. Framework model-actor (producer-consumer per-agen milik MegaAgent).
4. FastAPI + Postgres buatan sendiri (argumen Bedi).

Pelajaran ini membangun miniatur dari masing-masing.

## Konsep

### Durable execution, polanya

Mesin durable-execution menyimpan seluruh program state setelah tiap "step" (super-step, dalam bahasa LangGraph). Saat crash:

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

Syarat agar ini bekerja:

- **State yang dapat diserialisasi.** Semua state agen harus bisa disimpan. Function closure dengan koneksi database aktif tidak akan bertahan.
- **Resume deterministik.** Dengan state dan input yang sama, agen menghasilkan aksi yang sama (atau menunda ke oracle deterministik eksternal untuk panggilan LLM).
- **Side effect idempotent.** Panggilan eksternal (tool call, pembayaran) harus idempotent atau memakai kunci deduplication.

LangGraph menulis checkpoint setelah tiap super-step; Temporal menulis setelah tiap aktivitas; Restate memakai journal event-sourced. Ketiganya mengimplementasikan pola yang sama.

### Runtime checkpoint-per-step

Runtime LangGraph adalah contoh yang sudah dikerjakan: tiap agen punya `thread_id`; state adalah dict bertipe; tiap super-step menulis satu baris ke tabel checkpoints. Saat resume, runtime replay dari checkpoint terakhir, bukan dari nol. Agen bisa `interrupt()` menunggu input manusia; runtime menyimpan dan melepas worker. Ketika input tiba, worker mana pun bisa melanjutkan.

Ini desain produksi referensi per April 2026.

### Queue per-agen milik MegaAgent

arXiv:2408.09955 menggambarkan eksperimen skala: ribuan agen serentak dalam satu cluster. Arsitekturnya:

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

Koordinasi dua lapis memungkinkan percakapan intra-group berjalan padat sementara inter-group tetap jarang — pola yang dipakai untuk menjaga biaya linear pada ribuan agen.

### Async vs thread-per-job

Panggilan LLM dibatasi I/O. Thread yang menunggu token berikutnya menganggur 99% waktunya. Thread masing-masing berbiaya ~1MB RAM; pada 10.000 panggilan serentak, itu 10GB hanya untuk stack.

Fiber (`asyncio` Python, goroutine Go, `tokio` Rust) yielding secara kooperatif di I/O. 10.000 panggilan yang sama muat nyaman dalam satu proses. Pada skala LLM-agent, async bukan optimisasi — itu arsitekturnya.

Pengecualian: post-processing yang dibatasi CPU (embedding, trik tokenizer) tetap menginginkan thread atau proses. Pisahkan lapisan I/O dari lapisan CPU-mu.

### Kontra-argumen dari Bedi

"Scaling Agentic Software" (Ashpreet Bedi, 2026) berargumen bahwa kebanyakan tim over-engineering sebelum mengukur beban. Default pragmatisnya:

- FastAPI + Postgres.
- Tiap agent run adalah satu baris; state diperbarui in-place dengan optimistic concurrency.
- Background job lewat `pg_notify` atau Celery worker sederhana.
- Kebijakan retry di kode aplikasi.

Untuk beban di bawah ~100 agent-run serentak pada tugas yang bisa dikelola, ini sering kali sudah cukup. Naik kelas ketika kamu mengukurnya gagal.

Aturannya: adopsi framework durable-execution saat kamu menemui masalah konkret yang tidak bisa diselesaikan arsitektur sederhana. Adopsi terlalu dulu membakar waktu pada seremoni yang tidak berbaloi.

### Semantik exactly-once

Untuk agent run berbayar, kamu butuh "exactly-once efektif" (pengiriman at-least-once + konsumen idempotent). Langkah engineering-nya:

- **Dedup key per run.** Sertakan di setiap panggilan side effect.
- **Outbox pattern.** Side effect menulis ke tabel lebih dulu, lalu proses terpisah mengeksekusinya. Kedua langkah idempotent.
- **Compensating transaction.** Saat side effect berhasil tapi tulisan tracking-nya gagal, jadwalkan kompensasi.

Ini pola rekayasa basis data, bukan khusus LLM. Pajak LLM hanya bahwa panggilan LLM lambat; sisanya adalah distributed systems standar.

### Rainbow deployment

Sistem riset multi-agent Anthropic memakai "rainbow deployment": beberapa versi runtime agen berjalan serentak sehingga agen yang berjalan lama tidak perlu dibunuh di setiap deploy kode. Canary versi baru pada sebagian trafik; pensiunkan versi lama ketika agen mereka selesai.

Ini standar untuk sistem stateful yang berjalan lama; adaptasi 2026-nya adalah agen bisa hidup berjam-jam, jadi siklus deployment harus mengakomodasinya.

### Checklist produksi kanonis

- State tahan lama (checkpoint, snapshot, atau outbox + log yang bisa di-replay).
- Side effect idempotent.
- Lapisan async I/O untuk panggilan LLM.
- Pengiriman at-least-once dengan dedup.
- Deployment rainbow/canary untuk beban kerja stateful.
- Observability: jejak per-agen, audit super-step, penghitung retry.

```figure
sw-checkpoint-replay
```

## Bangun Sendiri

`code/main.py` mengimplementasikan:

- `CheckpointStore` — log checkpoint berbasis SQLite dengan kunci thread-id. Tiap super-step menambahkan satu baris.
- `run_with_checkpoint(agent, thread_id)` — mensimulasikan crash di tengah run; worker kedua melanjutkan dari checkpoint terakhir.
- `AgentQueue` — state machine Idle / Processing / Response per-agen dengan work queue kecil.
- `demo_async_vs_threads()` — menjalankan 500 "panggilan LLM" serentak yang disimulasikan via asyncio dan via thread; melaporkan wall-clock dan puncak memori (diperkirakan).

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan: resume checkpoint berhasil setelah crash yang disimulasikan; versi async menangani 500 panggilan serentak dalam < 1 detik; versi thread butuh beberapa detik dan memakai memori per unit serentak yang lebih besar beberapa order of magnitude.

## Gunakan

`outputs/skill-scaling-advisor.md` menasihati pilihan durable-execution: FastAPI + Postgres, runtime LangGraph, Temporal, atau custom. Dikalibrasi oleh beban, kebutuhan retensi state, dan frekuensi deploy.

## Rilis

Penguatan produksi kanonis:

- **Mulai dari yang sederhana (aturan Bedi).** FastAPI + Postgres sampai kamu mengukurnya gagal.
- **Instrumentasikan semuanya sebelum mengoptimalkan.** Histogram latensi per-run, waktu per-step, jumlah retry, kategorisasi kegagalan.
- **Outbox pattern untuk side effect.** Terutama pembayaran dan panggilan API eksternal.
- **Deploy rainbow.** Jangan pernah membunuh agent run yang sedang berjalan saat deploy.
- **Adopsi mesin durable-execution (Temporal / LangGraph / Restate) ketika** kamu menemui masalah spesifik: tunggu human-in-the-loop berjam-jam, koordinasi lintas-region, kebijakan retry/kompensasi yang kompleks.
- **Async untuk lapisan I/O.** Thread hanya untuk post-processing yang dibatasi CPU.

## Latihan

1. Jalankan `code/main.py`. Pastikan resume checkpoint bekerja; ukur selisih konkurensi async vs thread.
2. Implementasikan tabel **outbox**: setiap tool call menulis ke outbox lebih dulu, lalu goroutine/task terpisah mengeksekusinya. Verifikasi idempotensi dengan menjalankan tool call dua kali.
3. Simulasikan **rainbow deploy**: dua versi runtime serentak; arahkan separuh thread_id baru ke masing-masing; pastikan thread yang sedang berjalan di versi lama tidak terputus.
4. Baca dokumen runtime LangGraph (tautan di bawah). Identifikasi fitur runtime mana yang paling lama direplikasi dalam versi FastAPI + Postgres buatan sendiri. Apakah itu alasan untuk mengadopsi, atau bisa kamu tunda?
5. Baca MegaAgent (arXiv:2408.09955) Bagian 3. Koordinasi dua lapis (intra-group + inter-group admin chat) dinyatakan eksplisit. Sketsakan bagaimana kamu memetakannya ke message queue dengan dua keluarga queue.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Durable execution | "Persist the program state" | Engine writes state after each super-step; crash recovery is deterministic. |
| Super-step | "Transactional boundary" | Unit of work between checkpoints. LangGraph term. |
| thread_id | "Agent run identifier" | Key that binds checkpoints and resume logic. |
| Idempotency | "Safe to retry" | Repeating a side effect produces the same result as one attempt. |
| Outbox pattern | "Decouple side effects" | Write intent to a table; a separate executor performs and marks done. |
| At-least-once delivery | "Possible duplicates" | Message queue semantics; dedup key makes consumer effective-once. |
| Rainbow deploy | "Overlapping versions" | Multiple runtime versions concurrent during long-running workloads. |
| Async fiber | "Cooperative yielding" | User-mode concurrency; cheap compared to threads for I/O-bound loads. |
| Checkpoint | "State snapshot" | Serialized state at a super-step boundary; key for resume. |

## Bacaan Lanjutan

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph runtime design
- [MegaAgent](https://arxiv.org/abs/2408.09955) — per-agent producer-consumer queue; two-layer coordination at thousands of concurrent agents
- [Matrix](https://arxiv.org/abs/2511.21686) — decentralized framework with message queues as the coordination substrate
- [Temporal docs](https://docs.temporal.io/) — the reference workflow engine for durable execution
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — production lessons including rainbow deployment
