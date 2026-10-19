# Arsitektur Paralel / Swarm / Berjejaring

> Berbeda dengan supervisor: tanpa pemusus sentral. Agent membaca shared event bus, mengambil pekerjaan secara asinkron, menulis hasil kembali. LangGraph secara eksplisit mendukung "Swarm Architecture" untuk lingkungan terdesentralisasi dan dinamis. Matrix (arXiv:2511.21686) merepresentasikan baik alur kontrol maupun alur data sebagai pesan terserialisasi yang dilewatkan melalui antrean terdistribusi untuk menghilangkan bottleneck orchestrator. Tradeoff-nya eksplisit: determinisme dan keterlacakan untuk skalabilitas. Swarm cocok untuk tugas dengan banyak sub-masalah independen; ia tidak cocok untuk tugas yang membutuhkan satu rencana yang kohesif.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`, `queue`)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern), Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## Masalahnya

Supervisor menskalakan ke segelintir worker. Bagaimana dengan ratusan? Supervisor itu sendiri menjadi bottleneck: setiap keputusan tentang siapa mengerjakan apa menyempit melalui satu agent. Satu langkah rencana yang lambat menghentikan seluruh sistem.

Arsitektur swarm membalik desainnya. Alih-alih seorang planner sentral yang mengirim pekerjaan, worker mengambil pekerjaan dari antrean bersama. "Koordinasi" itu dibakar ke dalam semantik event bus. Tanpa orchestrator; sistem menskalakan sampai antreannya yang tidak.

## Konsepnya

### Bentuknya

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

Tanpa orchestrator. Setiap worker mengulangi: tarik sebuah tugas, proses, tulis hasilnya (dan opsional masukkan tindak lanjut ke antrean).

### Kapan swarm cocok

- **Banyak tugas independen.** Scraping, transformasi, klasifikasi. Tugas tidak saling bergantung.
- **Pekerjaan berdurasi bervariasi.** Jika beberapa tugas memakan 100ms dan yang lain 10s, sebuah swarm menyeimbangkan beban secara otomatis — worker cepat menarik pekerjaan berikutnya. Seorang supervisor harus mengantisipasi durasi.
- **Throughput di atas determinisme.** Kamu pedulikan total waktu penyelesaian, bukan pengurutan yang ketat.

### Kapan swarm gagal

- **Alur kerja terurut.** Jika langkah 3 membutuhkan keluaran langkah 2, swarm berisiko langkah 3 menyala sebelum langkah 2 selesai.
- **Tugas rencana-global.** Pertanyaan riset kompleks mendapat manfaat dari seorang planner. Sebuah swarm researcher menghasilkan fakta independen, bukan laporan yang kohesif.
- **Debugging.** Tanpa log sentral dan pekerjaan asinkron, mereproduksi sebuah bug mahal.

### Matrix (arXiv:2511.21686)

Matrix adalah paper 2025 yang membawa swarm ke kesimpulan alaminya: baik alur kontrol maupun alur data adalah pesan terserialisasi pada antrean terdistribusi. Tanpa koordinator sentral. Toleransi kegagalan datang dari ketahanan pesan. Skalabilitas adalah masalah message broker, bukan sistemnya.

Kontribusi: model pemrograman di mana koordinasi multi-agent adalah "topik pesan apa yang di-subscribe agent ini?" alih-alih "agent mana yang dipilih supervisor berikutnya?" Ini membuat sistemnya terlihat seperti mesh event pub/sub.

### Swarm di framework graph

Dokumentasi LangGraph 2025 secara eksplisit menggambarkan "Swarm Architecture" sebagai salah satu pola multi-agent: agent adalah node, tetapi edge membentuk graph berarah dengan siklus dan node mana pun dapat diaktifkan dari pool. Seorang worker memilih dari pekerjaan yang tersedia berdasarkan kondisi, bukan berdasarkan penetapan supervisor.

### Mode kegagalan: starvation dan hot-spotting

Jika semua worker menarik tugas yang tersedia tercepat, tugas berdurasi panjang tidak pernah dipilih sampai menjadi satu-satunya yang tersisa. Starvation antrean klasik.

Mitigasi:
- Antrean prioritas dengan aging eksplisit (tingkatkan prioritas dengan waktu tunggu).
- Spesialisasi worker: beberapa worker hanya mengambil tugas "panjang."
- Back-pressure: batasi berapa banyak tugas cepat yang masuk ke antrean.

### Tautan routing berbasis konten

Swarm berpasangan alami dengan routing berbasis konten (Lesson 22). Alih-alih antrean generik, buat satu antrean per tipe pesan. Worker spesialis hanya subscribe ke tipe mereka. Ini adalah dasar untuk arsitektur message bus yang menskalakan ke ribuan agent.

```figure
sw-work-stealing
```

## Bangun

`code/main.py` mengimplementasikan swarm dari 4 thread worker yang menarik dari `queue.Queue` bersama. Tugas memiliki durasi bervariasi (sebagian cepat, sebagian lambat). Demo membandingkan:

- **Baseline sekuensial:** satu worker memproses semua tugas secara serial.
- **Penetapan tetap:** setiap tugas ditetapkan sebelumnya ke worker tertentu (gaya supervisor).
- **Swarm:** worker menarik dari antrean bersama.

Swarm menyeimbangkan beban secara otomatis; penetapan tetap membiarkan worker menganggur ketika tugas yang ditugaskan kepada mereka lambat.

Jalankan:

```
python3 code/main.py
```

Keluaran menunjukkan jumlah tugas per worker (swarm mendistribusikan secara tidak merata tetapi optimal) dan waktu wall-clock.

## Gunakan

`outputs/skill-swarm-fit.md` mengevaluasi apakah sebuah tugas harus menggunakan swarm vs supervisor. Input: independensi tugas, varians durasi, kebutuhan pengurutan, kebutuhan debuggability.

## Rilis

Checklist:

- **Antrean prioritas dengan aging.** Cegah starvation tugas panjang.
- **Idempotensi worker.** Sebuah tugas bisa ditarik lebih dari sekali jika sebuah worker crash di tengah run. Worker harus idempoten.
- **Antrean tahan lama.** Gunakan Kafka, Redis Streams, atau antrean berbasis database untuk produksi. `queue.Queue` hanya in-memory.
- **Observabilitas per tugas.** Setiap tugas memiliki trace ID; setiap worker mencatat mulai/akhir dengannya.
- **Back-pressure.** Jika antrean tumbuh lebih cepat dari worker mengurasnya, perlambat produsennya.

## Latihan

1. Jalankan `code/main.py`. Seberapa cepat swarm dibanding sekuensial pada beban kerja durasi-bervariasi? Seberapa cepat dibanding penetapan tetap?
2. Tambahkan varian antrean prioritas (gunakan `queue.PriorityQueue`). Tetapkan prioritas berdasarkan field "importance" tugas. Amati apakah tugas prioritas rendah pernah starvation di bawah beban berkelanjutan.
3. Implementasikan detektor hot-spot: catat ketika worker mana pun memproses 3× lebih banyak tugas daripada worker terlambat. Apa yang ditunjukkan itu tentang distribusi durasi-tugas?
4. Baca abstrak paper Matrix (arXiv:2511.21686) dan Bagian 3. Identifikasi satu tradeoff spesifik yang diterima Matrix (peningkatan skalabilitas) dan satu yang ia korbankan (keterlacakan, determinisme).
5. Konversi demo swarm untuk menggunakan `queue.Queue` dari tuple (task_type, payload), dengan worker yang hanya subscribe ke tipe tertentu. Aturan routing mana yang masuk akal ketika tugasnya heterogen?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Swarm architecture | "Decentralized agents" | Workers pull from shared queue; no central orchestrator. |
| Event bus | "Agents subscribe to topics" | Message broker that routes tasks to workers by type or content. |
| Starvation | "Task never runs" | Low-priority task never gets picked because higher-priority work arrives continuously. |
| Hot-spotting | "One worker drowns" | Load imbalance where one worker gets most tasks. |
| Back-pressure | "Slow down the producer" | Mechanism that signals upstream to stop producing when the queue fills up. |
| Idempotent worker | "Safe to re-run" | A task processed twice produces the same result. Required because workers may crash mid-run. |
| Durable queue | "Survives crashes" | Queue backed by disk or replicated storage; tasks are not lost when a worker crashes. |
| Matrix framework | "Full message-passing swarm" | Both data and control flow are serialized messages on distributed queues. |

## Bacaan Lanjutan

- [LangGraph workflows and agents — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — explicit swarm support
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) — full message-passing swarm
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system) — why a specific production system explicitly chose supervisor over swarm
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/) — the event-driven actor rewrite, closer to swarm than v0.2's GroupChat
