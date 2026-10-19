# Model Audio-Language: Busur dari Whisper ke Audio Flamingo 3

> Whisper (Radford dkk., Desember 2022) menyelesaikan pengenalan bicara — 680k jam bicara multilingual weakly-supervised, transformer encoder-decoder sederhana, sebuah benchmark yang membuat setiap rilis ASR berikutnya mengutipnya. Tapi pengenalan bukan penalaran. Menanyakan "instrumen apa yang ada dalam rekaman ini" atau "emosi apa yang diekspresikan pembicara" atau "apa yang terjadi pada menit 3" memerlukan pemahaman audio, bukan transkripsi. Qwen-Audio, SALMONN, LTU, dan Audio Flamingo 3 milik NVIDIA (AF3, Juli 2025) membangun stack itu secara bertahap: pertahankan encoder kelas-Whisper, pasang Q-former, latih pada data instruksi teks-audio, tambahkan penalaran chain-of-thought. Pelajaran ini membahas busur itu.

**Type:** Build
**Languages:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Menghitung log-Mel spectrogram dari gelombang: windowing, FFT, filter bank, transform log.
- Membandingkan opsi encoder: encoder Whisper, BEATs, hibrida AF-Whisper. Kapan masing-masing menang.
- Membangun audio Q-former: N query yang dapat dipelajari cross-attend ke patch spectrogram.
- Menjelaskan pelatihan berjenjang (Whisper-lalu-LLM) vs audio-LLM end-to-end: mengapa end-to-end berskala lebih baik untuk penalaran.

## Masalahnya

Pengenalan bicara diselesaikan oleh Whisper. OCR-of-audio adalah komoditas. Tapi "komoditas" berhenti di transkripsi. Jika model tidak bisa menalar atas apa yang ia dengar — timing, pembicara, emosi, struktur musik, suara lingkungan — transkripsi saja tidak bisa menggerakkan fitur produk.

Tiga jalur yang jelas:

1. Cascade: Whisper men-transkrip, LLM menalar di atas transkrip itu. Berfungsi untuk skenario bicara-murni. Gagal untuk musik, audio lingkungan, tumpang-tindih multi-pembicara, emosi.

2. Audio-LLM end-to-end: encoder audio memberi makan token audio langsung ke sebuah LLM, melewati transkripsi. Melestarikan informasi akustik (emosi, pembicara, lingkungan). Memerlukan data training baru.

3. Hybrid: encoder audio + decoder teks yang bisa baik men-transkrip maupun menalar. Qwen-Audio dan Audio Flamingo memilih jalur ini.

## Konsepnya

### Log-Mel spectrogram: fitur input

Setiap encoder audio dimulai dari fitur yang sama: log-Mel spectrogram.

1. Resample ke 16 kHz.
2. Short-time Fourier transform dengan jendela 25ms, hop 10ms.
3. Ambil magnitudo dari hasil FFT.
4. Terapkan filter bank Mel (biasanya 80 filter ber-spasi-log 0-8000 Hz) untuk membengkokkan ke frekuensi perseptual.
5. Kompres log (log(1 + x)) untuk dynamic range.

Hasil: array 2D berbentuk (T, 80) di mana T adalah jumlah frame waktu. Untuk klip 30 detik pada frame rate 100 Hz: (3000, 80).

### Encoder Whisper

Encoder Whisper adalah transformer gaya-ViT 12 layer yang memproses log-Mel spectrogram sebagai sekuens frame waktu. Output: satu vektor hidden-state per frame waktu.

Untuk ASR, decoder Whisper adalah transformer cross-attention yang menghasilkan token teks yang di-kondisikan pada output encoder. Encoder-decoder standar.

Untuk ALM (audio-LLM), kamu menginginkan output encoder sebagai input ke LLM yang berbeda. Polanya: encoder Whisper dibekukan, Q-former dapat dilatih, LLM dibekukan atau di-tune.

### BEATs dan encoder khusus-audio

Whisper dilatih pada data yang didominasi bicara. Ia lebih lemah untuk musik dan audio lingkungan.

BEATs (Chen dkk., 2022) adalah transformer self-supervised yang dilatih pada AudioSet. Menangkap musik dan suara lingkungan lebih baik daripada Whisper pada jumlah parameter yang sama.

AF-Whisper (hibrida milik Audio Flamingo 3): concat fitur Whisper + BEATs sebagai input audio. Whisper membawa sinyal linguistik, BEATs membawa sinyal akustik.

### Audio Q-former

Pola yang sama dengan visual Q-former milik BLIP-2. Jumlah query yang dapat dipelajari yang tetap (sering 32 atau 64) cross-attend ke atas frame output encoder audio. Query-query itu menjadi token audio yang dikonsumsi LLM.

Tahap alignment training: Q-former saja, loss contrastive + captioning pada pasangan teks-audio (AudioCaps, Clotho). Tahap instruksi: end-to-end, unfreeze LLM, latih pada data instruksi.

### Busurnya — SALMONN, Qwen-Audio, AF3

SALMONN (Tang dkk., 2023): Whisper + BEATs + Q-former + LLaMA. Audio-LLM open pertama dengan kemampuan penalaran serius. Benchmark di MMAU menunjukkan komposit ~0.55.

Qwen-Audio (Chu dkk., 2023): arsitektur serupa, dilatih pada dataset lebih kaya, di-tune untuk dialog multi-turn. MMAU ~0.60.

LTU — Listen, Think, Understand (Gong dkk., 2023): data penalaran eksplisit, fokus pada chain-of-thought di atas klip audio. Lebih kecil tapi lebih fokus.

Audio Flamingo 3 (Goel dkk., Juli 2025): SOTA open saat ini. Backbone LLM 8B (Qwen2 7B), encoder Whisper-large concat BEATs, Q-former 64-query, training pada 1M+ pasangan instruksi teks-audio. MMAU 0.72, menyamai proprietary frontier pada beberapa sub-tugas.

AF3 juga memperkenalkan chain-of-thought on-demand untuk audio: model dapat secara opsional memancarkan token berpikir ("biar aku identifikasi instrumennya dulu: ...") sebelum jawaban akhir. Akurasi pada tugas penalaran kompleks terangkat 3-5 poin saat thinking diaktifkan.

### Berjenjang vs end-to-end

Pipeline berjenjang:

1. Whisper men-transkrip audio → teks.
2. LLM menalar di atas teks.

Berfungsi sempurna untuk "rangkum podcast ini." Gagal untuk:
- "Apa mood lagu ini?" — mood ada di bunyinya, bukan di kata-katanya.
- "Siapa yang berbicara, Alice atau Bob?" — memerlukan identifikasi pembicara.
- "Pada detik ke berapa ledakan terjadi?" — temporal grounding hilang dalam teks.
- "Apakah ini audio asli atau hasil generate?" — deteksi deepfake memerlukan fitur akustik.

End-to-end melestarikan sinyal akustik. Qwen-Audio dan AF3 menangani musik, lingkungan, dan emosi secara native.

### Resep produksi 2026

Untuk produk pemahaman-audio baru:

- Berjenjang jika: transkripsi adalah tujuannya, tanpa musik, tanpa inferensi emosi.
- Keluarga AF3 / Qwen-Audio jika: musik, emosi, multi-pembicara, atau penalaran audio kompleks.

Berjenjang lebih murah dan lebih sederhana. End-to-end lebih berkemampuan.

### MMAU — benchmark penalaran audio

MMAU (Massive Multimodal Audio Understanding) adalah benchmark penalaran audio 2024-2025:

- 10.000 pasangan QA teks-audio di seberang bicara, musik, suara lingkungan.
- Mencakup klasifikasi, penalaran temporal, penalaran kausal, QA terbuka.
- Menguji apa yang secara sistematis dilewatkan pipeline berjenjang.

SOTA open (AF3) di 0.72; proprietary frontier ~0.78 (Gemini 2.5 Pro, Claude Opus 4.7). Jaraknya lebih kecil daripada delta open-vs-closed VideoMME, menandakan audio-LLM sedang matang.

```figure
audio-text-ctc
```

## Gunakan

`code/main.py`:

- Mengimplementasikan perhitungan log-Mel spectrogram dalam stdlib: windowing, DFT naif, filter-bank Mel.
- Skeleton audio Q-former: diberi frame output encoder, hitung Q, K, V, attention, dan pancarkan N token.
- Perbandingan berjenjang-vs-end-to-end pada tugas mainan.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-audio-llm-pipeline-picker.md`. Diberi tugas audio (transkripsi, tagging musik, inferensi emosi, diarization multi-pembicara, klasifikasi lingkungan), ia memilih berjenjang, AF3 end-to-end, atau hibrida.

## Latihan

1. Hitung dimensi log-Mel spectrogram untuk klip 30 detik pada 16kHz, jendela 25ms, hop 10ms, 80 bin Mel. Bagaimana ini berubah pada 48kHz?

2. Mengapa Whisper berkinerja buruk pada musik? Fitur audio apa yang ditangkap BEATs yang tidak ditangkap Whisper?

3. Audio Q-former dengan 64 query vs 32: pada kompleksitas tugas apa 64 terbayar? 32 menghemat komputasi untuk apa?

4. Baca Bagian 4 AF3 tentang on-demand thinking. Usulkan tiga tugas audio di mana chain-of-thought paling membantu.

5. Implementasikan pipeline diarization minimal memakai output AF3. Bagaimana kamu menandai pergantian pembicara?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Log-Mel spectrogram | "Mel features" | 2D (time, frequency) array of log-magnitude values after Mel filter banks |
| Audio Q-former | "Audio Perceiver" | Cross-attention bottleneck from audio encoder output to fixed-length queries feeding the LLM |
| Cascaded | "ASR-then-LLM" | Pipeline where Whisper transcribes and a text LLM reasons; loses acoustic information |
| End-to-end | "Audio-LLM" | Audio features enter the LLM directly via Q-former; preserves acoustic signal |
| BEATs | "Audio AudioSet encoder" | SSL transformer trained on AudioSet; strong on music + environmental sounds |
| MMAU | "Audio reasoning bench" | 10k QA pairs across speech, music, environment; 2024 eval standard |
| On-demand thinking | "Audio CoT" | Model can optionally emit reasoning tokens before final answer, lifts accuracy 3-5 pts |

## Bacaan Lanjutan

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
