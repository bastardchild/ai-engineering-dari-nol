# MIO dan Model Multimodal Streaming Any-to-Any

> GPT-4o mengirim produk yang kebanyakan model open tidak bisa tirukan: agen yang mendengar suara, melihat video, dan berbicara balik secara real-time. Jawaban ekosistem-open pada akhir 2024 adalah MIO (Wang dkk., September 2024). MIO mem-tokenisasi teks, gambar, bicara, dan musik, melatih satu transformer kausal di atas sekuens yang selang-seling, dan menghasilkan modalitas apa pun ke modalitas apa pun. AnyGPT (Zhan dkk., Februari 2024) adalah proof of concept; MIO adalah scale-up-nya; Unified-IO 2 (Allen AI, Desember 2023) adalah kerabatnya dengan vision + action grounding. Pelajaran ini membaca pola any-to-any — empat tokenizer, satu transformer, decode yang ramah-streaming.

**Type:** Learn
**Languages:** Python (stdlib, four-modality token allocator + streaming decode loop)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 6 (Speech and Audio)
**Time:** ~120 minutes

## Tujuan Pembelajaran

- Merancang vocabulary bersama yang menampung token teks, gambar, bicara, dan musik tanpa tabrakan.
- Membandingkan SEED-Tokenizer (gambar) dan residual-VQ SpeechTokenizer (bicara) dari sisi trade-off kompresi + rekonstruksi.
- Menjelaskan kurikulum empat tahap yang membangun generasi any-to-any.
- Menyebut tiga resep any-to-any open dan trade-off utamanya: MIO, AnyGPT, Unified-IO 2.

## Masalahnya

Model multimodal terpadu mudah diklaim tapi sulit dibangun berskala besar. Sebagian besar sistem "any-to-any" sampai 2024 berupa pipeline: model visi → representasi teks → model bicara → audio. Setiap lompatan kehilangan informasi, menambah latensi, dan mempersulit training. Demo video GPT-4o menunjukkan alternatif single-model dengan respons sub-detik; sistem open tertinggal berbulan-bulan.

Tantangan rekayasa:

- Tokenizer harus ada untuk setiap modalitas, mengompres cukup tanpa-loss untuk rekonstruksi, dan menghasilkan token pada laju yang bisa dikonsumsi transformer.
- Satu vocabulary harus mengalokasikan ruang untuk teks (32k+), gambar (16k+), bicara (4k+), musik (8k+). Minimal empat puluh ribu entri lebih.
- Data training harus mencakup setiap pasangan input-output (text→image, image→speech, speech→image, dll.) atau model harus bisa mengkomposisi.
- Inference harus men-stream output token cukup cepat untuk latensi percakapan (<500ms time-to-first-audio-byte).

## Konsepnya

### Empat tokenizer untuk empat modalitas

Stack tokenizer MIO:

- Teks: BPE standar, vocab ~32000.
- Gambar: SEED-Tokenizer (2023) — VAE terkuantisasi dengan codebook diskrit, 4096 entri, 32x32 token per gambar.
- Bicara: residual-VQ SpeechTokenizer (2023) — meng-encode gelombang 16kHz ke dalam 8 codebook hierarkis; level pertama adalah konten kasar, level berikutnya menambah prosodi dan identitas pembicara.
- Musik: residual-VQ serupa (keluarga MusicGen / Encodec milik Meta), 4-8 codebook.

Setiap modalitas menghasilkan token bilangan bulat. Token-token itu mendapat rentang ID yang saling lepas dalam vocabulary bersama:

```
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

Total: vocabulary ~48k. Embedding input dan proyeksi output membentang ke seluruhnya.

### Decode streaming

Generasi bicara memakai residual-VQ. Transformer memprediksi token bicara base (layer 0); sebuah quantizer yang didecode-parallel memprediksi layer berikutnya. Setiap token layer 0 kira-kira 50ms audio pada 16kHz.

Pola streaming:

1. Pengguna berbicara ke mikrofon; tokenizer audio real-time memancarkan token bicara setiap 50ms.
2. MIO mengonsumsi token saat tiba (prefill prompt + forward inkremental).
3. Token output men-stream saat dihasilkan; decoder bicara parallel mengubahnya menjadi sampel audio dengan latensi ~50-150ms.
4. Time-to-first-audio-byte: ~300-500ms di paper MIO, mendekati ~250ms milik GPT-4o.

Mini-Omni (arXiv:2408.16725), GLM-4-Voice (arXiv:2412.02612), dan Moshi (arXiv:2410.00037) adalah desain streaming speech-LLM yang saling melengkapi. Moshi khususnya mencapai round-trip 160ms pada satu GPU.

### Kurikulum empat tahap

Kurikulum training MIO:

1. Tahap 1 — alignment. Korpus pasangan-modalitas skala besar: teks-gambar, teks-bicara, teks-musik. Setiap pasangan memakai segmen vocabulary token-nya sendiri. Melatih vocabulary bersama.
2. Tahap 2 — selang-seling. Dokumen selang-seling multi-modalitas (blog dengan gambar + video, podcast dengan transkrip, dll.). Melatih konteks lintas-modalitas.
3. Tahap 3 — speech-enhanced. Data audio ekstra untuk mengangkat kualitas bicara tanpa kehilangan kemampuan teks.
4. Tahap 4 — SFT. Instruction tuning lintas modalitas: VQA, captioning, narasi, dialog speech-to-speech.

Melewatkan satu tahap merusak kemampuan spesifik: lewati tahap 2 dan model kehilangan konteks lintas-modalitas; lewati tahap 3 dan bicaranya buruk.

### Chain-of-visual-thought

MIO memperkenalkan chain-of-visual-thought: model memancarkan token gambar intermediate sebagai langkah penalaran. Untuk "apakah kucing memanjat pohon?", model:

1. Memancarkan token `<image>` yang me-render adegan (dari gambar input atau sebuah sketsa).
2. Memancarkan teks yang menganalisis sketsa itu.
3. Memancarkan jawaban akhir.

Gambar intermediate yang di-render berfungsi sebagai scratchpad. Benchmark membaik pada tugas penalaran spasial. Idenya mencerminkan chain-of-thought untuk penalaran teks.

### Kompetitor any-to-any

- AnyGPT (arXiv:2402.12226): 4 modalitas (teks, gambar, bicara, musik), desain serupa.
- Unified-IO 2 (arXiv:2312.17172): menambah output aksi visi, kedalaman, normal. Keragaman tugas lebih besar, skala lebih kecil.
- NExT-GPT (arXiv:2309.05519): LLM + decoder diffusion spesifik-modalitas. Bukan pendekatan single-model.
- CoDi (arXiv:2305.11846): diffusion komposabel; any-to-any via latent bersama.

MIO adalah yang paling mendekati any-to-any berbasis token murni. AnyGPT adalah leluhur konseptualnya.

### Anggaran latensi

Untuk produk percakapan, latensi setiap komponen penting:

- Mikrofon ke token audio: ~50ms.
- Prefill (token audio + riwayat): ~100ms pada model 8B.
- Token output pertama: ~50ms.
- Residual-VQ parallel + decoder bicara: ~100-150ms.

Total time-to-first-audio-byte: minimal ~300ms. GPT-4o mengklaim ~250ms. Moshi mengklaim 160ms. MIO/AnyGPT berada di kisaran 400-600ms menurut benchmark publik.

### Mengapa any-to-any tetap sulit

Bahkan pada 2026, model any-to-any open tertinggal dari yang tertutup pada dua sumbu:

- Kualitas bicara. Tokenizer residual-VQ itu lossy; bicara percakapan terdengar robotik dibanding suara kelas ElevenLabs.
- Penalaran lintas-modalitas. Meminta model "nyanyikan tentang apa yang kamu lihat" masih lebih sering gagal daripada tugas visi-murni.

Ini adalah masalah riset yang masih terbuka. Qwen3-Omni (Pelajaran 12.20) adalah upaya open paling canggih pada 2025.

```figure
any-to-any-stream
```

## Gunakan

`code/main.py`:

- Mendefinisikan alokasi vocabulary empat-modalitas dan mencetaknya.
- Merutekan daftar input multimodal (teks, gambar, audio-clip, musik) melalui router tokenizer.
- Mensimulasikan decode streaming untuk respons text-to-speech dengan penghitungan latensi.
- Menghitung time-to-first-audio-byte yang diharapkan mengingat latensi encoder, prefill, dan decoder.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-any-to-any-pipeline-auditor.md`. Diberi spesifikasi produk percakapan (modalitas masuk, modalitas keluar, target latensi), ia mengaudit pilihan desain keluarga MIO dan menghitung anggaran latensinya.

## Latihan

1. Produkmu menerima input bicara dan mengembalikan output bicara. Berapa target anggaran latensi ujung-ke-ujung? Daftarkan komponen yang menghabiskan waktu.

2. Residual-VQ SpeechTokenizer memakai 8 codebook. Usulkan mengapa decode-parallel level residual diperlukan (vs sekuensial) dan berapa penghematan latensi yang dibawanya.

3. Vocabulermu punya 32k teks + 4k gambar + 4k bicara. Tambahkan 8k musik dan ~10 separator. Berapa biaya parameter matriks embedding pada hidden dim 4096?

4. Chain-of-visual-thought memancarkan gambar intermediate. Pertanyaan jenis apa yang diuntungkan? Jenis apa yang dirugikan oleh token tambahan itu?

5. Baca Moshi (arXiv:2410.00037). Jelaskan teknik "inner monologue"-nya dan bandingkan dengan chain-of-visual-thought milik MIO.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Any-to-any | "Multimodal in/out" | A single model that accepts and emits text, image, speech, and music in any direction |
| Residual-VQ | "Speech tokenizer stack" | Multi-codebook tokenization where each layer adds information; base layer is content, later layers are prosody |
| SEED-Tokenizer | "Image codes" | Discrete image tokenizer with 4096-entry codebook used by MIO |
| Chain-of-visual-thought | "Visual scratchpad" | The model generates an intermediate image as a reasoning step before its final answer |
| Time-to-first-audio-byte | "TTFAB" | Latency from user voice to first audio output; <500ms for conversational feel |
| Four-stage curriculum | "Training recipe" | Alignment -> interleaved -> speech-enhanced -> SFT, in that order |

## Bacaan Lanjutan

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)
