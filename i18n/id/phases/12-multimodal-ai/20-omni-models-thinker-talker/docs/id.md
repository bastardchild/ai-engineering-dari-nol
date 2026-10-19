# Omni Models: Qwen2.5-Omni dan Pemisahan Thinker-Talker

> Demo produk GPT-4o pada Mei 2024 menggemparkan bukan karena model dasarnya, melainkan karena bentuk produknya — antarmuka suara di mana kamu berbicara, model melihat apa yang dilihat kamera, dan ia membalas dalam waktu kurang dari 250ms. Ekosistem open source menghabiskan sisa 2024 dan 2025 untuk berlomba mencapai permukaan produk itu. Qwen2.5-Omni (Maret 2025) adalah desain open referensi: sebuah Thinker (transformer besar pembangkit teks) plus Talker (transformer paralel pembangkit ucapan), dihubungkan oleh speech token streaming. Mini-Omni menyederhanakannya, Moshi menyamai latensinya, GLM-4-Voice memperluasnya ke Mandarin. Pelajaran ini membaca arsitektur Thinker-Talker dan anggaran latensi yang membuat dialog real-time streaming menjadi nyata.

**Type:** Build
**Languages:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Memisahkan pipeline inferensi menjadi Thinker (penalaran teks) dan Talker (sintesis ucapan) dan menjelaskan mengapa streaming paralel bekerja.
- Menghitung anggaran time-to-first-audio-byte (TTFAB) untuk sebuah interaksi percakapan, komponen per komponen.
- Mendeskripsikan posisi time-aligned TMRoPE pada vision, audio, dan teks di dalam Thinker.
- Menyebutkan tiga pola percakapan real-time: half-duplex, turn-taking, full-duplex.

## Masalahnya

Asisten suara real-time harus melakukan banyak hal, dengan cepat:

1. Mendengarkan pengguna. Tokenisasi ucapan real-time, voice activity detection (VAD) untuk tahu kapan mereka selesai berbicara.
2. Secara opsional melihat. Input kamera pada 2-4 FPS, di-stream ke Thinker bersama audio.
3. Berpikir. Menyusun respons yang dibingkai oleh riwayat percakapan.
4. Berbicara. Mensintesis token ucapan, decode ke gelombang, stream ke speaker pengguna.

Setiap langkah menambah latensi. Nuansa percakapan menuntut round-trip total < 500ms — di bawah itu, pengguna berhenti menyadari jeda. GPT-4o mengklaim ~250ms. Moshi ~160ms. Qwen2.5-Omni ~350-500ms.

Setiap komponen harus streaming. Tidak ada yang boleh berupa "batch semuanya lalu decode."

## Konsepnya

### Thinker dan Talker

Dekomposisi Qwen2.5-Omni:

- Thinker: transformer pembangkit teks 7B-80B. Mengonsumsi token teks + gambar + audio yang berselang-seling. Mengeluarkan token teks yang mewakili apa yang harus dikatakan.
- Talker: transformer pembangkit ucapan yang lebih kecil (200M-1B). Mengonsumsi token output teks Thinker plus token konteks-ucapan terkini. Mengeluarkan token ucapan diskrit (indeks residual-VQ).
- Speech decoder: decoder gelombang streaming (keluarga SNAC, MoVQGAN) yang mengubah token ucapan menjadi sampel audio secara real-time.

Pemisahan ini penting. Thinker harus besar agar penalarannya bagus. Talker bisa kecil karena tugasnya bersifat lokal — mengubah teks menjadi token ucapan. Talker yang lebih besar tidak lebih ekspresif; ia lebih lambat.

Menjalankan keduanya secara paralel:

1. Thinker mengeluarkan token teks t_i.
2. Talker mengonsumsi t_i (via streaming) dan mengeluarkan token ucapan s_i, s_{i+1}, ..., s_{i+k}.
3. Speech decoder mengonsumsi token ucapan saat datang dan mengeluarkan sampel audio.
4. Pada saat Thinker berada di token teks t_{i+3}, Talker sudah meng-stream audio untuk t_0..t_{i+2}.

### TMRoPE — posisi multimodal yang time-aligned

Thinker perlu mengintegrasikan frame gambar (datang pada, misalnya, 4 FPS), frame audio (datang pada 50 frame/detik), dan teks dari riwayat percakapan. Urutan sekuens yang naif (semua gambar, lalu semua audio, lalu teks) kehilangan keselarasan temporal.

TMRoPE menetapkan timestamp absolut ke setiap token. Token vision pada t=2,3 detik. Token audio pada t=2,32 detik. Token teks dari pengguna "stop" pada t=2,35 detik. RoPE memutar attention berdasarkan timestamp; model melihatnya sebagai sesuatu yang bersifat temporal bersamaan.

Ini adalah infrastruktur yang membuat "ia melambaikan tangan sambil mengucapkan halo" bekerja — model melihat frame video dan audio pada momen konseptual yang sama.

### Sintesis ucapan streaming

Token ucapan harus streaming. Mini-Omni (Xie & Wu, 2024) memperkenalkan "language models can hear, talk while thinking in streaming": token output Thinker dan token output Talker berselang-seling dalam satu sekuens. Talker menembak segera setelah Thinker mengunci token teks berikutnya. Tidak ada batas batch.

Moshi (Défossez et al., Oktober 2024) adalah implementasi open tercepat. TTFAB 160ms pada satu A100. Arsitektur: satu transformer 7B yang mengeluarkan token teks dan ucapan pada posisi berselang-seling, dengan "monolog batin" yang memisahkan aliran berpikir dari aliran berbicara. Ini pada dasarnya Thinker + Talker yang dilebur menjadi satu model dengan pelatihan yang cermat.

### VAD dan turn-taking

Voice activity detection berjalan di sisi input. Dua pola:

- Half-duplex: pengguna berbicara, model mendengarkan. Model berbicara, pengguna mendengarkan. Serah-terima yang jelas via deteksi keheningan VAD (~200ms).
- Full-duplex: keduanya bisa berbicara bersamaan. Model bisa memberi tanggapan belakang ("he-eh") atau menyela. Jauh lebih sulit. Moshi mendukung ini.

Qwen2.5-Omni mendukung half-duplex secara bawaan, dengan turn-taking lewat ambang keheningan. Full-duplex membutuhkan penanganan di layer aplikasi.

### Qwen3-Omni (November 2025)

Penerusnya. Thinker Qwen3-80B, Talker lebih besar, TMRoPE-v2 yang lebih baik. Latensi mendekati 250ms milik GPT-4o. Bobot terbuka. Benchmark pada OmniBench sebanding dengan Gemini 2.0 Live.

### Anggaran latensi produksi

Untuk interaksi streaming tipikal:

- Mikrofon -> token audio: 40-80ms.
- Prefill (prompt + riwayat): 100-200ms pada 7B, jauh lebih besar pada 70B.
- Token teks Thinker pertama: 40ms.
- Talker memproses token teks pertama: 20ms.
- Token ucapan pertama terkunci: 40ms.
- Decode residual-VQ: 30ms.
- Decode gelombang ucapan: 50-80ms.

TTFAB total: 320-510ms pada 7B, 600-900ms pada 70B. Kualitas frontier biasanya berarti 70B+; karena itu pula ada kesenjangan latensi pada frontier.

### Matematika token-rate

Pada ucapan 16kHz dengan token ucapan dasar 50 Hz, kamu membutuhkan 50 token ucapan per detik output. Talker harus mengeluarkan ≥50 tok/s untuk mengikuti. Pada throughput LLM tipikal 30-80 tok/s di H100, Talker kecil (200-300M) sudah cukup cepat; Talker 7B akan tertinggal.

Inilah alasan model Talker khusus yang kecil ada, alih-alih "pakai saja model utama."

```figure
l5-thinker-talker
```

## Gunakan

`code/main.py`:

- Mensimulasikan pipeline Thinker-Talker dengan laju emisi token mock.
- Menghitung TTFAB untuk ukuran model dan laju sampel mikrofon yang dapat dikonfigurasi.
- Mendemonstrasikan turn-taking half-duplex dengan ambang keheningan VAD.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-omni-streaming-budget.md`. Diberi target TTFAB dan set fitur produk suara real-time (vision-in, bilingual, full-duplex), memilih Qwen2.5-Omni, Qwen3-Omni, Moshi, atau Mini-Omni dan mengukur ukuran Thinker/Talker.

## Latihan

1. Target TTFAB-mu adalah 300ms. Pada Thinker 7B dan Talker 300M, tuliskan latensi setiap komponen.

2. Qwen2.5-Omni memakai TMRoPE. Jelaskan apa yang dilihat model untuk prompt di mana pengguna mulai berbicara pada t=1 detik dan kamera menangkap sebuah gestur pada t=1,2 detik.

3. Dukungan full-duplex menuntut model mengeluarkan audio sambil mendengarkan. Usulkan format data training yang mengajarkan ini.

4. Baca Bagian 4 paper Moshi. Jelaskan pemisahan "monolog batin" itu dan mengapa ia menghindari pemisahan Thinker-Talker.

5. Hitung anggaran throughput: seberapa cepat Talker harus mengeluarkan token untuk mengikuti ucapan 16kHz pada 50 token layer-dasar per detik?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Thinker | "Reasoning brain" | Large text-generating transformer producing what to say |
| Talker | "Speech-generating mouth" | Small transformer producing discrete speech tokens from Thinker's text |
| TTFAB | "Latency budget" | Time-to-first-audio-byte: from user speech end to first audio sample out |
| TMRoPE | "Time-aligned RoPE" | Position encoding using absolute timestamps across vision, audio, text |
| Half-duplex | "Turn-taking" | User and model alternate; VAD silence detects user-done |
| Full-duplex | "Simultaneous" | Model can speak and listen at the same time; backchannel capable |
| Inner monologue | "Moshi separation" | Single-model design where thinking-stream and speaking-stream interleave |

## Bacaan Lanjutan

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
