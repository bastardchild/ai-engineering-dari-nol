# Speech-to-Speech Streaming — Moshi, Hibiki, dan Dialog Full-Duplex

> 2024-2026 mendefinisikan ulang voice AI. Moshi mengirim satu model yang mendengar dan berbicara secara bersamaan dengan latensi 200 ms. Hibiki menerjemahkan speech-to-speech potongan demi potongan. Keduanya meninggalkan pipeline ASR → LLM → TTS demi arsitektur full-duplex terpadu di atas token codec Mimi. Inilah rancangan referensi yang baru.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Masalahnya

Setiap voice agent yang dibangun dari Pelajaran 11 + 12 punya lantai latensi mendasar di kisaran 300-500 ms: VAD menyala, STT memproses, LLM berpikir, TTS menghasilkan. Setiap tahap punya latensi minimum-nya sendiri. Kamu bisa menyetel dan mem-paralel, tetapi bentuk pipeline-nya membatasi kamu.

Moshi (Kyutai, 2024-2026) mengajukan pertanyaan berbeda: bagaimana kalau tidak ada pipeline? Bagaimana kalau satu model menerima audio dan memancarkan audio secara langsung, terus-menerus, dengan teks sebagai "monolog batin" sementara alih-alih tahap yang wajib ada?

Jawabannya adalah **full-duplex speech-to-speech**. Latensi teoretis 160 ms (frame Mimi 80 ms + delay akustik 80 ms). Latensi praktis 200 ms pada satu GPU L4. Itu setengah dari yang dicapai voice agent pipeline kelas terbaik.

## Konsepnya

![Arsitektur Moshi: dua stream Mimi paralel + teks monolog batin](../assets/moshi-hibiki.svg)

### Arsitektur Moshi

**Input.** Dua stream codec Mimi, keduanya pada 12.5 Hz × 8 codebook:

- Stream 1: audio pengguna (di-encode Mimi, terus-menerus datang)
- Stream 2: audio Moshi sendiri (dihasilkan oleh Moshi)

**Transformernya.** Sebuah Temporal Transformer berparameter 7B memproses kedua stream dan satu stream teks "monolog batin". Pada setiap langkah 80 ms, ia:

1. Mengonsumsi token Mimi pengguna terbaru (8 codebook).
2. Mengonsumsi token Mimi Moshi terbaru (8 codebook, sebagaimana dihasilkan).
3. Menghasilkan token teks Moshi berikutnya (monolog batin).
4. Menghasilkan token Mimi Moshi berikutnya (8 codebook lewat sebuah Depth Transformer kecil).

Ketiga stream — audio pengguna, audio Moshi, teks Moshi — berjalan paralel. Moshi bisa mendengar pengguna sambil berbicara; bisa memotong dirinya sendiri saat pengguna menyela; bisa memberi back-channel ("mhm") tanpa memecah ucapan utamanya.

**Depth transformernya.** Di dalam satu frame, 8 codebook itu tidak diprediksi secara paralel — mereka punya ketergantungan antar-codebook. Sebuah "depth transformer" 2-lapis kecil memprediksi mereka secara sekuensial dalam 80 ms. Ini adalah faktorisasi standar untuk codec LM berbasis AR (juga dipakai VALL-E, VibeVoice).

### Mengapa teks monolog batin membantu

Tanpa teks eksplisit, model harus memodelkan bahasa secara implisit di dalam stream akustiknya. Wawasan Moshi: paksa ia mengeluarkan token teks di samping audio. Stream teks pada dasarnya adalah transkrip dari apa yang Moshi ucapkan. Ini meningkatkan koherensi semantik, mempermudah penggantian kepala model bahasa, dan memberimu transkrip secara gratis.

### Hibiki: terjemahan speech-to-speech streaming

Arsitektur sama, dilatih pada pasangan terjemahan. Audio sumber masuk, audio bahasa target keluar, terus-menerus. Hibiki-Zero (Feb 2026) menghilangkan kebutuhan akan data training sejajar level-kata — memakai data level-kalimat + reinforcement learning GRPO untuk optimisasi latensi.

Empat pasangan bahasa didukung pada awalnya; bisa diadaptasi ke bahasa baru dengan ≈1000 jam.

### Stack Kyutai yang lebih luas (2026)

- **Moshi** — dialog full-duplex (Prancis dulu, Inggris didukung baik)
- **Hibiki / Hibiki-Zero** — terjemahan bicara bersamaan
- **Kyutai STT** — ASR streaming (look-ahead 500 ms atau 2.5 s)
- **Kyutai Pocket TTS** — TTS berparameter 100M berjalan di CPU (Jan 2026)
- **Unmute** — pipeline penuh yang menggabungkan ini di server publik

Throughput pada GPU L40S: 64 sesi bersamaan pada 3× waktu nyata.

### Sesame CSM — saudaranya

Sesame CSM (2025) memakai ide serupa — tulang punggung Llama-3 dengan kepala codec Mimi. Tetapi CSM bersifat satu-arah (menerima konteks + teks, menghasilkan bicara) alih-alih full-duplex. Ini adalah TTS "voice presence" terbaik di pasaran; tidak persis sama dengan kemampuan full-duplex Moshi.

### Angka performa 2026

| Model | Latency | Use case | License |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | full-duplex English / French dialogue | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | French ↔ English streaming translation | CC-BY 4.0 |
| Hibiki-Zero | same | 5 language-pairs, no aligned data | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | context-conditioned TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | closed, OpenAI API | commercial |
| Gemini 2.5 Live | ~350 ms | closed, Google API | commercial |

```figure
sp-fullduplex
```

## Bangun Sendiri

### Langkah 1: antarmukanya

Moshi mengekspos sebuah server WebSocket yang menerima potongan audio ber-encode Mimi 80 ms dan mengembalikan potongan audio ber-encode Mimi 80 ms. Dua arah. Terus-menerus.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Langkah 2: loop full-duplex

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

Kedua arah berjalan bersamaan. asyncio Python atau futures Rust adalah transportasi standar.

### Langkah 3: objektif training (konseptual)

Untuk setiap frame 80 ms `t`:

- Input: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- Prediksi: `moshi_text[t]`, lalu `moshi_mimi[t, codebook_0..7]`

Teks diprediksi sebelum audio (monolog batin); audio diprediksi sekuensial-codebook di dalam depth transformer.

### Langkah 4: di mana Moshi menang dan di mana tidak

Moshi menang:

- Sub-250 ms end-to-end pada hardware murah.
- Back-channel dan interupsi yang alami.
- Tanpa kode lem pipeline.

Moshi tidak menang:

- Tool calling (tidak dilatih untuk itu; kamu butuh jalur LLM terpisah).
- Reasoning panjang (Moshi adalah model dialog berukuran ~8B, bukan Claude/GPT-4).
- Akurasi faktual pada topik niche.
- Kebanyakan use case enterprise produksi (2026 masih memakai pipeline).

## Gunakan

| Situation | Pick |
|-----------|------|
| Lowest-latency voice companion | Moshi |
| Live translation call | Hibiki |
| Voice demo / research | Moshi, CSM |
| Enterprise agent with tools | Pipeline (Lesson 12), not Moshi |
| Custom-voice TTS in context | Sesame CSM |
| Speech-to-speech, any languages | GPT-4o Realtime or Gemini 2.5 Live (commercial) |

## Jebakan

- **Tool calling terbatas.** Moshi adalah model dialog, bukan framework agent. Gabungkan dengan pipeline untuk tool.
- **Kondisioning suara spesifik.** Moshi memakai satu persona terlatih; cloning adalah run training terpisah.
- **Cakupan bahasa.** Prancis + Inggris sangat baik; lainnya terbatas. Hibiki-Zero membantu, tetapi kamu tetap butuh data training.
- **Biaya sumber daya.** Sesi Moshi penuh memegang satu slot GPU; bukan pola deploy shared-tenant yang murah.

## Rilis

Simpan sebagai `outputs/skill-duplex-pipeline.md`. Pilih pipeline vs arsitektur full-duplex untuk sebuah beban kerja voice-agent, beserta alasannya.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ia mensimulasikan arsitektur dua-stream + monolog batin secara simbolis.
2. **Sedang.** Tarik Moshi dari HuggingFace, jalankan server-nya, uji satu percakapan. Ukur latensi wall-clock dari akhir bicara-pengguna ke awal respons-Moshi.
3. **Sulit.** Ambil pipeline agent Pelajaran 12-mu dan bandingkan latensi P50 dengan Moshi pada 20 ucapan uji yang cocok. Tulis kapan sebuah pipeline menang secara arsitektural.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Full-duplex | Hear-and-speak at once | Two audio streams active simultaneously on the same model. |
| Inner monologue | Model's text stream | Moshi emits text tokens alongside its audio output. |
| Depth transformer | Inter-codebook predictor | Small transformer that predicts 8 codebooks within one 80 ms frame. |
| Mimi | Kyutai's codec | 12.5 Hz × 8 codebooks; semantic+acoustic; powers Moshi. |
| Streaming S2S | Audio → audio live | Chunk-by-chunk translation/dialogue, no pipeline stages. |
| Back-channeling | "Mhm" reactions | Moshi can emit small acknowledgments without breaking its turn. |

## Bacaan Lanjutan

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — the paper.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — streaming translation without aligned data.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM spec.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — install + server.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — closed commercial peer.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — the STT/TTS framework under the hood.
