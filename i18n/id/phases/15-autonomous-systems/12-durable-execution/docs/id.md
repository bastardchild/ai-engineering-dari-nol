# Agen Latar Berjalan-Lama: Eksekusi Durable

> Agen horizon-panjang produksi tidak berjalan dalam `while True`. Setiap panggilan LLM menjadi activity dengan checkpoint, retry, dan replay. Integrasi OpenAI Agents SDK dari Temporal went GA Maret 2026. Claude Code Routines (Anthropic) menjalankan invokasi Claude Code terjadwal tanpa proses lokal persisten. Sesi berhenti pada input-manusia, selamat dari deploy, dan lanjut dari checkpoint terbaru dengan kunci `thread_id`. Di balik ergonomi baru duduk pola lama — orkestrasi workflow — dengan satu input baru: panggilan LLM sebagai activity non-deterministik yang harus di-replay secara deterministik saat pemulihan.

**Type:** Learn
**Languages:** Python (stdlib, minimal durable-execution state machine)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~60 minutes

## Masalahnya

Bayangkan agen yang berjalan empat jam. Ia memanggil tiga tool, mem-prompt pengguna dua kali, dan melakukan empat puluh panggilan LLM. Di tengah jalan, host tempatnya berjalan reboot. Apa yang terjadi?

- Dalam loop `while True` yang naif: semuanya hilang. Proses mulai ulang dari nol. Ketiga panggilan tool (dengan efek samping nyata) dieksekusi lagi. Pengguna di-prompt lagi untuk hal yang sudah mereka setujui. Empat puluh panggilan LLM ditagih ulang.
- Dengan eksekusi durable: proses lanjut dari checkpoint terbaru. Activity yang sudah selesai tidak dieksekusi ulang; hasilnya di-replay dari log durable. Pengguna tidak menyetujui ulang hal yang sudah mereka setujui. Panggilan LLM yang sudah dilakukan tidak ditagih ulang.

Ini pola yang sama yang dikirim mesin workflow selama satu dekade (Temporal, Cadence, Cherami Uber). Yang baru adalah panggilan LLM kini sejenis activity — non-deterministik, mahal, dengan efek samping — dan cocok bersih dengan pola ini.

Tema berjalan pelajaran ini: reliabilitas horizon-panjang meluruh (METR mengamati "degradasi 35-menit" — angka keberhasilan turun kira-kira kuadratis dengan horizon). Eksekusi durable memungkinkan proses yang lebih panjang dari profil reliabilitas, yang merupakan cara baru untuk gagal secara aman jika desainnya benar dan tidak aman jika desainnya salah.

## Konsepnya

### Activity, workflow, dan replay

- **Workflow**: kode orkestrasi deterministik. Mendefinisikan rangkaian activity, cabang, penantian. Harus deterministik sehingga bisa di-replay dari log event tanpa divergensi yang mengejutkan.
- **Activity**: unit kerja non-deterministik yang bisa gagal. Panggilan LLM, panggilan tool, penulisan file, permintaan HTTP. Setiap activity di-log dengan input-nya dan (setelah selesai) output-nya.
- **Log event**: penyimpanan backing yang durable. Setiap mulai, selesai, gagal, retry activity, dan setiap keputusan workflow dicatat.
- **Replay**: saat pemulihan, kode workflow berjalan ulang dari awal; setiap activity yang sudah selesai mengembalikan hasil yang di-log tanpa mengeksekusi ulang. Hanya activity yang belum selesai yang benar-benar dijalankan.

Ini bentuk yang sama dengan React me-render ulang terhadap virtual DOM, atau Git membangun ulang working tree dari commit. Determinisme dalam orkestrator inilah yang membuat durability murah.

### Mengapa panggilan LLM cocok dengan polanya

Panggilan LLM adalah:
- Non-deterministik (temperature > 0; bahkan temperature 0 melenceng lintas versi model).
- Mahal (uang dan latensi).
- Bisa gagal (rate limit, timeout).
- Side-effectful (jika memanggil tool).

Ini tepat profil activity. Membungkus setiap panggilan LLM sebagai activity memberimu retry dengan backoff eksponensial, checkpoint lintas restart, dan trace yang bisa di-replay untuk debugging.

### Checkpoint dengan kunci `thread_id`

LangGraph, Microsoft Agent Framework, Cloudflare Durable Objects, dan Claude Code Routines semuanya konvergen pada bentuk API yang sama: `thread_id` (atau padanannya) mengidentifikasi sesi; setiap transisi status dipersisten ke backend (default PostgreSQL, SQLite untuk dev, Redis untuk cache); resume membaca checkpoint terbaru.

Pilihan backend penting:

- **PostgreSQL**: durable, bisa di-query, selamat dari deploy. Default untuk LangGraph.
- **SQLite**: hanya dev-lokal; kehilangan data lintas host.
- **Redis**: cepat tetapi efemeral kecuali AOF/snapshot dikonfigurasi.
- **Cloudflare Durable Objects**: terdistribusi secara transparan; di-scope dengan kunci unik; bertahan berjam-jam hingga berminggu-minggu.

### Input-manusia sebagai status kelas-satu

Propose-then-commit (Pelajaran 15) membutuhkan status "menunggu manusia" yang durable. Workflow berhenti, antrean eksternal menahan permintaan yang tertunda, dan persetujuan lanjut dari titik itu tepat. Tanpa durability ini best-effort; dengannya, persetujuan semalam tiba dan workflow berlanjut di pagi hari.

### Degradasi 35-menit

METR mengamati bahwa setiap kelas agen yang diukur menunjukkan peluruhan reliabilitas di luar ~35 menit operasi kontinu. Menggandakan durasi tugas kira-kira melipatgandakan angka kegagalan. Eksekusi durable tidak memperbaikinya; ia membiarkanmu berjalan lebih lama dari profil reliabilitas. Pola amannya adalah menggabungkan durability dengan checkpoint yang membutuhkan HITL segar saat masuk kembali, dan dengan kill switch anggaran (Pelajaran 13) yang membatasi total komputasi terlepas dari wall-clock time.

### Ketika eksekusi durable adalah jawaban yang salah

- Proses lebih pendek dari beberapa menit tanpa input manusia. Overhead > manfaat.
- Pengambilan informasi read-only yang ketat.
- Tugas tempat kebenaran membutuhkan ujung-ke-ujung dalam satu context window (sebagian tugas penalaran; sebagian generasi sekali-jadi).

```figure
memory-consolidation
```

## Gunakan

`code/main.py` mengimplementasikan mesin eksekusi-durable minimal dalam Python stdlib. Ia mendukung:

- Dekorator `@activity` yang meng-log input dan output ke log event JSON.
- Fungsi workflow yang merangkai activity.
- Fungsi `run_or_replay(workflow, event_log)` yang me-replay activity yang selesai tanpa mengeksekusinya ulang.

Driver mensimulasikan workflow tiga-activity, crash di tengah jalan, dan menunjukkan (a) retry naif mengeksekusi ulang semuanya versus (b) replay hanya menjalankan activity yang hilang.

## Rilis

`outputs/skill-durable-execution-review.md` mereviu deployment agen berjalan-lama yang diusulkan untuk bentuk eksekusi-durable yang benar: activity, determinisme, backend checkpoint, status input-manusia, dan kebijakan HITL-on-resume.

## Latihan

1. Jalankan `code/main.py`. Amati perbedaan jumlah eksekusi-activity antara retry naif dan replay. Ubah titik crash dan tunjukkan jumlah replay berubah sesuai itu.

2. Konversi mesin mainan untuk memakai `thread_id` secara eksplisit. Simulasikan dua sesi konkuren yang berbagi mesin dan konfirmasi log event-nya tidak bertabrakan.

3. Ambil satu activity dalam mesin mainan. Perkenalkan non-determinisme (timestamp wall-clock di dalam keputusan workflow). Demonstrasikan divergensi saat replay. Jelaskan bagaimana mesin nyata menangani ini (registrasi efek samping, API `Workflow.now()`).

4. Baca postingan LangChain "Runtime behind production deep agents". Daftarkan setiap status yang dipersisten runtime dan sebutkan mode kegagalan mana yang dicakup masing-masing.

5. Rancang kebijakan checkpoint untuk tugas koding otonom 6 jam. Di mana kamu checkpoint? Seperti apa resume-on-crash? Apa yang membutuhkan HITL segar?

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Workflow | "Agent's script" | Deterministic orchestration code; replayable from event log |
| Activity | "A step" | Non-deterministic unit (LLM call, tool call); logged before and after |
| Event log | "The backing store" | Durable record of every state transition |
| Replay | "Resume" | Re-run workflow; completed activities return logged results without re-execution |
| Checkpoint | "Save point" | Persisted state keyed by thread_id; latest-wins on resume |
| thread_id | "Session key" | Identifier that scopes durable state |
| 35-minute degradation | "Reliability decay" | METR: success rate drops ~quadratically with horizon |
| Non-determinism | "Drift on replay" | Wall clock, random, LLM output; must be registered as side effect |

## Further Reading

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — budget, turns, and resume semantics.
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent shape.
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — concrete runtime requirements.
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — activity shape for LLM calls.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — the 35-minute degradation reference.
