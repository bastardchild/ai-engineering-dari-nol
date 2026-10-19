# Batch API — Diskon 50% sebagai Standar Industri

> Setiap provider besar merilis batch API asinkron dengan diskon 50% dan turnaround ~24-jam. OpenAI, Anthropic, Google, dan kebanyakan platform inferensi (tier batch Fireworks, batch Together) mengimplementasikan pola sama. Tumpuk batch dengan prompt caching dan pipeline semalaman turun ke ~10% dari biaya sinkron-tanpa-cache. Aturannya brutal sederhana: kalau tidak interaktif, ia milik batch. Pipeline generasi konten, klasifikasi dokumen, ekstraksi data, generasi laporan, pelabelan massal, tagging katalog — apa pun yang toleran terhadap latensi 24-jam adalah uang yang tertinggal di meja sampai pindah ke batch. Pola produksi 2026 adalah mentriase setiap workload LLM baru ke tiga jalur: interaktif (sinkron dengan caching), semi-interaktif (antrean async dengan fallback), batch (semalaman, input ter-cache ditumpuk). Workload yang berpura-pura interaktif tetapi toleran terhadap latensi bermenit-menit paling boros.

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## Tujuan Pembelajaran

- Sebutkan tiga batch API provider (OpenAI, Anthropic, Google) dan garansi umum diskon 50% + turnaround 24 jam.
- Hitung biaya penumpukan batch + input-ter-cache pada workload klasifikasi semalaman dan bandingkan dengan baseline sinkron-tanpa-cache.
- Triase workload ke interaktif / semi-interaktif / batch dan justifikasi jalurnya.
- Sebutkan dua jebakan: interaktivitas parsial (pengguna berharap lebih cepat dari 24 jam) dan drift skema-output (format file batch berbeda per provider).

## Masalahnya

Timmu merilis pipeline generasi laporan malam hari. 50.000 dokumen, ringkas masing-masing, klasterkan ringkasan, susun brief eksekutif. Berjalan sinkron butuh 4 jam seharga $2.000/malam. Kamu mendengar tentang batch API.

Batch memberimu 50% off. Kamu juga menyalakan prompt caching pada system prompt (dibagi lintas semua 50k panggilan). Ditumpuk, tagihan turun ke $180/malam — ~9% dari baseline. Pipeline sama, tiga perubahan konfigurasi.

Batch adalah tuas termurah dalam toolkit biaya LLM yang tak ditarik siapa pun. Alasannya kebanyakan organisasional: tim berpikir "real-time" padahal SLA sesungguhnya "sebelum pagi." Pelajaran ini tentang tidak menyisakan 90% tagihan di meja.

## Konsepnya

### Tiga batch API

**OpenAI Batch API**: unggah file JSONL berisi daftar request. Dijanjikan turnaround 24-jam (biasanya ~2-8 jam dalam praktik). Diskon 50% pada token input dan output. Endpoint `/v1/batches`. Input yang layak-cache juga mendapat harga input-ter-cache di atasnya.

**Anthropic Message Batches**: unggah JSONL. Turnaround 24-jam. Diskon 50%. Mendukung `cache_control` — tulis cache eksplisit, baca terjadi otomatis dalam batch.

**Google Vertex AI Batch Prediction**: input BigQuery atau GCS. Diskon 50% serupa untuk Gemini. Terintegrasi dengan pipeline Vertex.

### Semantik: asinkron, bukan lambat

Batch adalah "aku berjanji kembali dalam 24 jam" — bukan "ini akan makan 24 jam." P50 tipikal 2-6 jam. Provider menjadwalkan batch-mu pada window off-peak saat inventaris GPU kurang dimanfaatkan.

### Tumpuk dengan caching

Ringkasan 50k-dokumen dengan system prompt 4K-token yang sama:

- Sinkron tanpa-cache: 50000 × ($input × 4000 + $output × 200) pada tarif penuh.
- Sinkron ter-cache: system prompt ter-cache setelah tulis pertama; 49999 sisanya mendapat input 10x lebih murah.
- Batch ter-cache: semua di atas plus diskon 50% pada baca maupun tulis.

Tumpukannya: batch + cache = ~10% dari tagihan sinkron tanpa-cache. Workload apa pun yang berjalan semalaman dan punya system prompt bersama harus memakai ini.

### Triase workload

**Interaktif** — pengguna menunggu respons. TTFT penting. Panggilan sinkron dengan prompt caching. Tidak bisa batch.

**Semi-interaktif** — pengguna mengirim task, mengecek kembali dalam hitungan menit. Antrean async dengan fallback ke sinkron kalau batch tak tersedia. Pikirkan indexing RAG volume-sedang.

**Batch** — pengguna berharap hasil "sebelum pagi" atau "jam berikutnya." Pipeline konten, klasifikasi skala besar, analisis offline. Selalu batch, selalu tumpuk caching.

Kesalahan umum: mengklasifikasikan semuanya sebagai interaktif karena pipeline-nya produksi. Produksi bukan spesifikasi latensi — SLA adalah.

### Jebakan interaktivitas-parsial

Sebagian fitur tampak interaktif tetapi toleran 5-10 menit. Contoh: laporan kesehatan pelanggan malam hari dengan tombol "refresh". Pengguna mengklik refresh; menunggu 10 menit tidak masalah. Tim merilisnya sebagai sinkron. 50 refresh konkuren berharga 10x dibanding batch-lalu-kirim-via-email.

Pertanyaan yang perlu ditanyakan: "Apa arti 24-jam bagi pengguna ini?" Kalau jawabannya "mereka tidak akan sadar," batch-kan.

### Jebakan skema-output

Format file batch berbeda per provider:

- OpenAI: JSONL, satu request per baris.
- Anthropic: JSONL, satu message per baris; format respons tertanam.
- Vertex: tabel BigQuery atau prefix GCS dengan TFRecord.

Menulis "satu klien batch" lintas provider berarti kode adapter per provider. Gateway yang mengiklankan batch multi-provider (Portkey, LiteLLM beberapa tier) tetap membungkus tipis format mentah.

### Angka yang perlu kamu ingat

- Diskon batch lintas provider: 50% flat pada input + output.
- SLA turnaround: 24 jam dijamin, 2-6 jam P50 tipikal.
- Batch + input ter-cache ditumpuk: ~10% dari biaya sinkron tanpa-cache.
- Aturan triase workload: kalau latensi 24 jam dapat diterima, selalu batch.

```figure
batch-lane-triage
```

## Gunakan

`code/main.py` menghitung biaya lintas sinkron, sinkron+cache, batch, dan batch+cache untuk workload 50k-dokumen. Melaporkan hemat dalam $ dan persen.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-batch-triager.md`. Dengan karakteristik workload, ia mentriase ke interaktif/semi/batch dan mengestimasi hemat.

## Latihan

1. Jalankan `code/main.py`. Untuk pipeline 100k-dok dengan system prompt 3K-token dan output 500-token, hitung hemat tumpukan penuh (batch + cache) vs baseline sinkron.
2. Pilih tiga fitur pada produk nyata yang kamu kenal. Triase masing-masing ke interaktif/semi/batch.
3. Pengguna mengeluh laporannya makan 3 jam. Apakah itu salah-triase batch atau interaktif legit? Tulis kriteria keputusannya.
4. SLA return batch API-mu 24 jam tetapi P99 20 jam. Bagaimana kamu mengomunikasikannya ke pengguna — bagaimana perilaku sistem hilir pada edge case?
5. Hitung titik-impas: pada panjang prefix-bersama berapa batch + cache menjadi lebih murah daripada berjalan semalaman pada GPU reserved milikmu sendiri?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Batch API | "async discount" | 50% off with 24h turnaround |
| JSONL | "batch format" | One JSON request per line; OpenAI/Anthropic standard |
| Message Batches | "Anthropic batch" | Anthropic's batch API product name |
| Batch prediction | "Vertex batch" | Vertex AI's batch API product |
| Turnaround SLA | "24h promise" | Guarantee, not typical; typical is 2-6h |
| Workload triage | "interactivity decision" | Interactive / semi / batch routing decision |
| Output schema | "response format" | Per-provider JSONL layout; not portable |
| Stacked discount | "batch + cache" | ~10% of uncached sync bill when both apply |

## Further Reading

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL format and `/v1/batches` semantics.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch format and `cache_control` interaction.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) — Gemini batch semantics.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
