# Pemahaman Video Panjang pada Konteks Jutaan Token

> Video 4K 1 jam pada 24 FPS, di-patch dan di-embed, menghasilkan urutan sekitar 60 juta token. Episode podcast 2 jam yang ditranskrip berjumlah 30.000 token. Film Blu-ray penuh, bahkan dikompres dengan pooling agresif, berjumlah ratusan ribu token. Gemini 1.5 milik Google (Maret 2024) membuka era ini dengan konteks 10 juta token, melakukan recall needle-in-a-haystack yang andal di atas video panjang-jam. LWM (Liu dkk., Februari 2024) menunjukkan jalur skala ring attention. LongVILA dan Video-XL menskalakan ingestion lebih jauh. VideoAgent menukar konteks mentah dengan retrieval agentic. Setiap pendekatan adalah trade-off berbeda pada komputasi, recall, dan kompleksitas rekayasa. Pelajaran ini membacanya berdampingan.

**Type:** Build
**Languages:** Python (stdlib, needle-in-haystack simulator + agentic-retrieval router)
**Prerequisites:** Phase 12 · 17 (video temporal tokens)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Menghitung jumlah total token visual untuk video berdurasi panjang pada berbagai FPS dan pooling.
- Menjelaskan tiga jalur skala: brute context (Gemini 1.5), ring attention (LWM), kompresi token (LongVILA / Video-XL).
- Membandingkan video VLM konteks-mentah vs video VLM retrieval-agentic (VideoAgent) dari sisi akurasi dan latensi.
- Merancang uji needle-in-a-haystack untuk video 30 menit dan mengukur recall pada menit tertentu.

## Masalahnya

Satu frame berukuran patch gaya Qwen2.5-VL pada resolusi native 384 berjumlah ~729 token. Pada pooling 3x3 itu 81 token per frame. Klip 30 menit pada 1 FPS = 1800 frame = 145.800 token. Bisa ditangani VLM open 2025, tapi ketat. Pada 2 FPS, 291.600 token — hanya konteks terbesar yang muat.

Film 2 jam pada 1 FPS berjumlah 583k token. Di luar kebanyakan model open 2026; memerlukan Gemini 2.5 Pro atau pooling yang lebih agresif.

Tiga jalur skala muncul.

## Konsepnya

### Jalur 1: Brute context (Gemini 1.5, Claude Opus)

Lempar hardware ke masalahnya. Skalakan konteks ke jutaan token, proses semuanya dalam satu forward pass.

Gemini 1.5 Pro diluncurkan dengan 1M token; Gemini 1.5 Ultra hingga 10M; Gemini 2.5 Pro pada 2026 menangani berjam-jam video secara andal. Paper (arXiv:2403.05530) mendokumentasikan recall needle-in-a-haystack 99,7% hingga ~9.5M token.

Rekayasa: implementasi attention kustom dengan hierarki memori (lokal + global + sparse) plus routing expert MoE untuk efisiensi konteks-panjang. Tidak dipublikasikan secara detail penuh. Tidak open-source.

### Jalur 2: Ring attention (LWM, LongVILA)

Ring attention mendistribusikan sekuens panjang ke perangkat dalam sebuah "ring" di mana setiap perangkat memegang satu chunk. Attention di atas sekuens penuh terjadi dengan setiap perangkat mengirim chunk-nya ke perangkat berikutnya dalam pola ring, menghitung attention parsial, dan mengagregasi.

LWM (Liu dkk., 2024) melatih model konteks 1M token dengan cara ini. Komputasi training berskala linear terhadap konteks, bukan kuadratik — pukulan kuadratik pada attention diamortisasi di seluruh perangkat-perangkat ring.

LongVILA (arXiv:2408.10188) mengadaptasi pola ini ke VLM. Video 1400 frame pada 192 token per frame = konteks 268k, dilatih dengan ring attention di seberang paralelisme 8-way.

### Jalur 3: Kompresi token (Video-XL, LongVA)

Lebih murah daripada brute context: kompres secara agresif sebelum LLM melihat sekuens.

Video-XL (arXiv:2409.14485) memakai token ringkasan visual: setiap klip N frame menghasilkan satu token "ringkasan" yang meng-attend ke atas N itu. Saat inference, LLM melihat satu token ringkasan per klip, yang mengecilitkan konteks secara drastis.

LongVA memperluas konteks LLM dari 200k ke 2M dengan teknik "long context transfer". Latih pada teks konteks-panjang, transfer ke video konteks-panjang via representasi bersama.

Kompresi token menukar recall pada timestamp spesifik dengan skalabilitas. Model tahu secara umum apa yang terjadi tapi kadang melewatkan frame yang tepat.

### Jalur 4: Retrieval agentic (VideoAgent)

Jangan berikan video penuh ke LLM. Sebaliknya, perlakukan video sebagai database dan pakai LLM untuk men-query-nya.

VideoAgent (arXiv:2403.10517):

1. LLM membaca pertanyaan.
2. LLM meminta tool retrieval klip yang relevan ("tunjukkan segmen dengan kucing").
3. Tool mengembalikan timestamp klip yang cocok.
4. LLM membaca klip-klip itu via sebuah VLM.
5. LLM menyusun jawaban atau meminta query lanjutan.

Ini adalah pola LLM-as-agent yang diterapkan ke video panjang. Inference lebih murah (hanya klip relevan yang di-encode), rekayasa lebih sulit (kualitas retrieval menjadi bottleneck).

### Benchmark needle-in-a-haystack

Uji konteks-panjang standar: sisipkan marker visual atau teks unik di titik acak dalam video, lalu ajukan query yang memerlukan recall-nya.

Metrik: Recall@k di seberang panjang video dan posisi marker.

Gemini 2.5 Pro memperoleh recall >99% pada video hingga 90 menit. Model open 72B (Qwen2.5-VL-72B, InternVL3-78B) memperoleh ~85-90% pada 30 menit dan merosot melewati 60.

VideoAgent dapat menyamai atau mengalahkan model konteks-mentah pada 2+ jam karena retrieval mengenai needle-nya jika tool-nya bagus.

### Jalur mana yang dipilih

Untuk klip 15 menit dengan akurasi frontier: open 72B + konteks native biasanya bekerja. Pilih Qwen2.5-VL-72B.

Untuk konten 30 menit hingga 1 jam: LongVILA atau Video-XL untuk open; Gemini 2.5 Pro untuk closed. Ambang kualitas penting — untuk frontier beralih ke yang closed.

Untuk konten 2+ jam: VideoAgent atau pola retrieval serupa. Alternatifnya, ringkas ke chunk yang lebih kecil dan berikan ringkasan hierarkis.

### Pola produksi 2026

Dalam praktiknya, pipeline video-panjang produksi bersifat hybrid:

1. Jalankan sampling dynamic-FPS + pooling agresif pada seluruh video (dapatkan representasi global 100k-token).
2. Berikan ke VLM 72B untuk ringkasan global.
3. Jika pengguna mengajukan pertanyaan detail, jalankan retrieval agentic memakai ringkasan itu sebagai indeks.

Ini menggabungkan brute-context untuk pemahaman global dan retrieval untuk detail lokal.

```figure
mm-video-token-budget
```

## Gunakan

`code/main.py`:

- Menghitung anggaran token untuk video dari 1 menit hingga 3 jam pada berbagai FPS + pooling.
- Mensimulasikan satu run needle-in-a-haystack: menyuntikkan marker pada timestamp acak, mengajukan pertanyaan, menskor recall.
- Memuat simulator router retrieval-agentic yang memilih klip spesifik untuk diberikan ke VLM hilir.

Jalankan tabel anggarannya dan rasakan jarak skalanya.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-long-video-strategy-planner.md`. Diberi durasi video dan kompleksitas query, ia memilih antara brute-context, kompresi, dan retrieval agentic, serta menghitung ekspektasi latensi + kualitas.

## Latihan

1. Kuliah 45 menit pada 1 FPS, 81 token per frame. Total token? Muat di konteks model mana?

2. Rancang uji needle-in-a-haystack: pada menit ke berapa kamu menyuntikkan marker, dan apa format query yang tepat?

3. Bandingkan Qwen2.5-VL-72B brute-context (konteks 80k) dengan VideoAgent (Claude 3.5 + retrieval) pada video 1 jam. Mana yang menang di recall? Mana yang menang di latensi?

4. Biaya memori ring attention berskala linear terhadap panjang sekuens dan linear terhadap jumlah perangkat. Jelaskan mengapa dan apa yang gagal jika kamu membuang fase rotasi-ring-nya.

5. Baca Bagian 5 Gemini 1.5 tentang needle-in-a-haystack. Apa yang ditemukan paper tentang recall di batas 1M vs 10M token?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Brute context | "Just more tokens" | Scale LLM context to millions of tokens; process everything in one pass |
| Ring attention | "LWM-style parallel" | Distributed attention pattern where each device holds a chunk and rotates |
| Token compression | "Summary tokens" | Reduce per-clip tokens via a learned compressor before the LLM |
| Needle-in-haystack | "NIH test" | Insert a unique marker at a random point, ask model to recall it at test time |
| Agentic retrieval | "LLM as query planner" | LLM asks a retrieval tool for relevant clips, reads them via a VLM, composes answer |
| VideoAgent | "Retrieval pattern for video" | Canonical agentic-retrieval design: question -> tool -> clip -> answer |

## Bacaan Lanjutan

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
