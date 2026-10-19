# Ekonomi Prompt Caching dan Semantic Caching

> **Snapshot harga bertanggal 2026-04.** Klaim numerik di bawah mencerminkan rate card vendor yang ditangkap saat publikasi pelajaran ini; verifikasi terhadap dokumen tertaut sebelum mengutipnya lebih lanjut.

> Caching terjadi pada dua lapisan. Caching prompt/prefix L2 (level-provider) memakai ulang KV attention untuk prefix berulang — dokumen prompt-caching Anthropic mengiklankan reduksi biaya hingga 90% dan reduksi latensi 85% pada prompt panjang; untuk Claude 3.5 Sonnet cache read $0,30/M vs $3,00/M segar dengan TTL 5-menit dan premi tulis 2x untuk opsi TTL 1-jam (docs.anthropic.com, 2026-04). Prompt caching OpenAI berlaku otomatis untuk prompt ≥1024 token dan menghargai input ter-cache sekitar diskon 90% vs segar (platform.openai.com, 2026-04); tarif ter-cache per-model persis bergantung pada rate card live. Semantic caching L1 (level-aplikasi) melewati LLM sepenuhnya pada hit kemiripan embedding. "Akurasi 95%" vendor merujuk kebenaran kecocokan, bukan hit rate — hit rate produksi yang dilaporkan berkisar dari 10% (chat terbuka) hingga 70% (FAQ terstruktur); tak ada provider yang mempublikasikan baseline resmi, jadi perlakukan ini sebagai telemetri komunitas, bukan garansi. Jebakan produksi: paralelisasi membunuh caching (N request paralel yang diterbitkan sebelum tulis cache pertama bisa menggelembungkan spend berlipat-lipat), dan konten dinamis di dalam prefix mencegah cache hit sepenuhnya. ProjectDiscovery melaporkan berpindah dari hit rate 7% menjadi 74% (2025-11) dengan memindahkan teks dinamis keluar dari prefix yang bisa di-cache.

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Bedakan caching prompt/prefix L2 (reuse KV pada provider) dari semantic caching L1 (bypass LLM pada prompt mirip).
- Jelaskan penandaan eksplisit `cache_control` Anthropic dan dua opsi TTL (5-menit vs 1-jam) beserta pengali harganya.
- Hitung hemat bulanan ekspektasi dengan hit rate, campuran prompt/respons, dan harga token tertentu.
- Sebutkan antipola paralelisasi yang menggelembungkan tagihan 5-10x dan antipola konten-dinamis yang meruntuhkan hit rate.

## Masalahnya

Kamu menambah prompt caching ke layanan RAG-mu. Tagihan tetap datar. Kamu mengukur hit rate; 7%. Prompt-mu tampak statis tetapi sebenarnya tidak — system prompt menyertakan tanggal kini yang diformat per menit, ID request, dan reorder contoh acak demi diversitas. Setiap request menulis entri cache baru, membaca nol.

Terpisah, agenmu menjalankan sepuluh panggilan tool paralel per pertanyaan pengguna. Kesepuluhnya tiba di provider sebelum tulis cache pertama selesai. Sepuluh tulis, nol baca. Tagihanmu 5-10x dari yang seharusnya "dengan caching".

Caching adalah protokol, bukan flag. Dua lapisan, dua mode kegagalan berbeda.

## Konsepnya

### L2 — caching prompt/prefix provider

Provider menyimpan KV attention untuk prefix yang bisa di-cache dan memakainya ulang pada request berikutnya yang cocok dengan prefix. Kamu membayar biaya tulis sekali, baca nyaris gratis.

**Anthropic (seri Claude 3.5 / 3.7 / 4)**: marker `cache_control` eksplisit dalam request. Kamu menandai blok mana yang bisa di-cache. TTL: 5-menit (tulis berharga 1,25x dasar) atau 1-jam (tulis berharga 2x dasar). Cache read: $0,30/M pada Claude 3.5 Sonnet vs $3,00/M segar — 10x lebih murah (docs.anthropic.com, per 2026-04). Tarif berbeda per model (Opus/Haiku dipublikasikan terpisah); selalu cek-silangan halaman harga live.

**OpenAI**: caching otomatis untuk prompt ≥1024 token (platform.openai.com, 2026-04). Tanpa flag eksplisit. Input ter-cache sekitar 10x lebih murah daripada segar pada rate card gpt-4o/gpt-5 kini. Dokumen maupun catatan rilis tidak mempublikasikan baseline hit-rate resmi; laporan komunitas mengelompok sekitar 30–60% dengan desain prompt hati-hati. Pantau `usage.cached_tokens` untuk mengukur milikmu sendiri.

**Google (Gemini)**: context caching lewat API eksplisit; konteks 1M-token berarti caching lebih menguntungkan.

**Self-hosted (vLLM, SGLang)**: Fase 17 · 06 membahas RadixAttention — pola sama pada komputasimu sendiri.

### L1 — semantic caching level-aplikasi

Sebelum memanggil LLM sama sekali, hash prompt, embed, dan cari request ter-cache yang mirip (kemiripan kosinus di atas ambang, biasanya 0,95+). Pada hit, kembalikan respons ter-cache. Pada miss, panggil LLM dan cache hasilnya.

Open-source: Redis Vector Similarity, GPTCache, Qdrant. Komersial: Portkey Cache, Helicone Cache.

Klaim akurasi vendor merujuk seberapa sering respons ter-cache yang dikembalikan sesuai secara semantik — bukan seberapa sering kamu hit. Hit rate produksi:

- Chat terbuka: 10-15%.
- FAQ / support terstruktur: 40-70%.
- Pertanyaan kode: 20-30% (varian kecil membunuh hit).
- Agen suara mengulang prompt: 50-80% (himpunan tetap normalisasi suara).

### Antipola paralelisasi

Agenmu membuat 10 panggilan tool paralel. Kesepuluhnya punya system prompt 4K-token yang sama. Tulis cache Anthropic bersifat per-request; tulis-cache pertama selesai sekitar 300 ms setelah provider melihat prompt. Request 2-10 tiba dalam window milidetik yang sama dan masing-masing melihat cache miss. Kamu membayar 10 premi tulis, 0 diskon baca.

Perbaikan: batch dengan sequential-first — buat request 1 sendirian, lalu tembakkan 2-10 begitu cache milik 1 terisi. Menambah 300 ms ke panggilan tool pertama; menghemat 5-10x tagihan.

### Antipola konten dinamis

System prompt-mu tampak seperti:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

Setiap request unik. Setiap request menulis. Nol hit.

Perbaikan: pindahkan semua yang benar-benar statis ke prefix yang bisa di-cache; tambahkan konten dinamis setelah batas cache:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery berpindah dari hit rate 7% menjadi 74% dengan cara ini dan mempublikasikan anatominya.

### Tumpuk batch + cache untuk workload semalaman

Batch API (Fase 17 · 15) memberi diskon 50% pada turnaround 24-jam. Input ter-cache di atasnya memberimu ~10x tambahan. Workload klasifikasi, pelabelan, dan generasi-laporan semalaman bisa turun ke ~10% dari biaya sinkron-tanpa-cache dengan menumpuk.

### Angka yang perlu kamu ingat

Titik harga ditangkap 2026-04 dari dokumen vendor tertaut dan bergeser tiap beberapa bulan — cek ulang sebelum mengandalkannya.

- Cache read Anthropic: $0,30/M pada Claude 3.5 Sonnet, sekitar 10x lebih murah daripada input segar (docs.anthropic.com).
- Premi tulis cache Anthropic: 1,25x (TTL 5-menit) atau 2x (TTL 1-jam).
- Auto-cache OpenAI: berlaku untuk prompt ≥1024 token; input ter-cache dihargai sekitar 10% dari input segar pada rate card kini (platform.openai.com).
- Hit rate semantic cache (dilaporkan-komunitas): ~10% chat terbuka; hingga ~70% FAQ terstruktur. Bukan baseline terdokumentasi-vendor.
- ProjectDiscovery: hit rate 7% → 74% dengan memindahkan dinamis keluar dari prefix (blog proyek, 2025-11).
- Antipola paralelisasi: laporan tipikal inflasi tagihan 5–10x saat N request paralel melewatkan tulis cache pertama.

```figure
semantic-cache-hit
```

## Gunakan

`code/main.py` mensimulasikan caching L1 + L2 pada workload campuran. Melaporkan hit rate, tagihan, dan menunjukkan penalti paralelisasi.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-cache-auditor.md`. Dengan template prompt dan trafik, ia mengaudit cacheability dan merekomendasikan restrukturisasi.

## Latihan

1. Jalankan `code/main.py`. Toggle flag paralelisasi. Berapa perubahan tagihannya?
2. System prompt-mu punya tanggal. Pindahkan keluar. Tunjukkan matematika hit rate sebelum/sesudah.
3. Hitung titik-impas TTL 1-jam (tulis 2x) vs TTL 5-menit (tulis 1,25x) dengan arrival rate request-mu.
4. Semantic cache pada ambang 0,95 mencapai 20%. Pada 0,85 mencapai 50% tetapi kamu melihat respons ter-cache yang salah. Pilih ambang tepat dan justifikasi.
5. Kamu mem-batch 10 sub-query paralel per pertanyaan pengguna. Tulis ulang agar ramah-cache tanpa menambah latensi end-to-end.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| L2 prompt cache | "prefix cache" | Provider stores KV for repeated prefix |
| `cache_control` | "Anthropic cache marker" | Explicit attribute marking cacheable blocks |
| Cache write premium | "write tax" | Extra cost for first miss-to-cache (1.25x or 2x) |
| L1 semantic cache | "embedding cache" | App-level hash-and-embed before calling LLM |
| GPTCache | "LLM caching lib" | Popular OSS L1 cache library |
| Cache hit rate | "hits / total" | Fraction of requests served from cache |
| Parallelization anti-pattern | "the N-write trap" | N parallel requests miss cache N times |
| Dynamic content trap | "the time-in-prompt trap" | Dynamic bytes in prefix kill hit rate |
| RadixAttention | "intra-replica cache" | SGLang's prefix-cache implementation |

## Further Reading

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — official `cache_control` semantics and TTLs.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — automatic caching behavior and eligibility.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
