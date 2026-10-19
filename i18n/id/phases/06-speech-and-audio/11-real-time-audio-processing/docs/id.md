# Pemrosesan Audio Real-Time

> Pipa batch memproses satu berkas. Pipa real-time memproses 20 milidetik berikutnya sebelum 20 berikutnya tiba. Setiap AI percakapan, studio siaran, dan bot telepon hidup dan mati oleh anggaran latensi ini.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## Masalahnya

Kamu ingin asisten suara yang terasa hidup. Latensi giliran-mengambil percakapan manusia ~230 ms (hening-ke-jawaban). Apa pun di atas 500 ms terasa robotik; di atas 1500 ms terasa rusak. Anggaran untuk satu putaran penuh **dengar → pahami → respon → bicara** pada 2026 adalah:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi (Kyutai, 2024) mencatat 200 ms full-duplex. GPT-4o-realtime (2024) mencatat ~320 ms. Pipa kaskade pada 2022 dirilis di 2500 ms. Peningkatan 10× datang dari tiga teknik: (1) streaming di mana-mana, (2) pipelining asinkron dengan hasil parsial, (3) generasi yang dapat disela.

## Konsepnya

![Pipa audio streaming dengan ring buffer, gerbang VAD, interupsi](../assets/real-time.svg)

**Frame / chunk / window.** Audio real-time mengalir sebagai blok berukuran tetap. Pilihan umum: 20 ms (320 sampel pada 16 kHz). Segala sesuatu di hilir harus mengikuti ketukan ini.

**Ring buffer.** Buffer sirkular berukuran tetap. Thread producer menulis frame baru, thread consumer membaca. Mencegah alokasi di hot path. Ukuran ≈ latensi-maksimum × laju-sampel; ring 16 kHz selama 2 detik = 32.000 sampel.

**VAD (Voice Activity Detection).** Membuka-tutup kerja di hilir saat tak ada yang berbicara. Silero VAD 4.0 (2024) berjalan <1 ms per frame 30 ms di CPU. `webrtcvad` adalah alternatif yang lebih lama.

**Streaming ASR.** Model yang memancarkan transkrip parsial saat audio masuk. Parakeet-CTC-0.6B dalam mode streaming (NeMo, 2024) mencapai WER 2–5% dengan latensi 320 ms. Whisper-Streaming (Macháček et al., 2023) meng-chunk Whisper agar nyaris-streaming dengan latensi ~2 detik.

**Interruption.** Saat pengguna bicara sementara asisten sedang berbicara, kamu harus (a) mendeteksi barge-in, (b) menghentikan TTS, (c) membuang sisa output LLM. Semuanya dalam 100 ms, atau pengguna merasa asistennya tuli.

**Transport Opus WebRTC.** Frame 20 ms, 48 kHz, bitrate adaptif 8–128 kbps. Standar untuk browser dan seluler. LiveKit, Daily.co, Pion adalah stack 2026 untuk membangun aplikasi suara.

**Jitter buffer.** Paket jaringan datang tidak berurutan / terlambat. Jitter buffer mengurutkan dan menghaluskan; terlalu kecil → jeda yang terdengar, terlalu besar → latensi. 60–80 ms tipikal.

### Jebakan umum

- **Contention thread.** GIL Python + model berat dapat mematikan rasa thread audio. Pakai library audio berbasis callback C (sounddevice, PortAudio) dan jauhkan Python dari hot path.
- **Latensi konversi laju-sampel.** Resampling di dalam pipa menambah 5–20 ms. Pilih resample di muka atau pakai resampler nol-latensi (PolyPhase, `soxr_hq`).
- **Priming TTS.** Bahkan TTS cepat seperti Kokoro punya warm-up 100–200 ms pada permintaan pertama. Cache model + hangatkan dengan run dummy sebelum giliran nyata pertama.
- **Echo cancellation.** Tanpa AEC, output TTS masuk kembali ke mikrofon dan memicu ASR pada suara bot itu sendiri. WebRTC AEC3 adalah default open source.

```figure
nyquist-aliasing
```

## Bangun Sendiri

### Langkah 1: ring buffer

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Kapasitas menentukan latensi buffering maksimum. 32.000 sampel pada 16 kHz = 2 detik.

### Langkah 2: gerbang VAD

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Ganti dengan Silero VAD di produksi:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Langkah 3: streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Langkah 4: penangan interupsi

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Bertumpu pada I/O async dan streaming TTS yang dapat dibatalkan. WebRTC peerconnection.stop() pada track audio adalah cara yang kanonis.

## Gunakan

Stack 2026:

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## Jebakan

- **Buffering 500 ms untuk aman.** Buffer *itu* adalah lantai latensimu. Perkecil.
- **Tidak men-pin thread.** Callback audio di thread dengan prioritas lebih rendah dari UI = glitch saat beban tinggi.
- **Chunk TTS terlalu kecil.** Chunk di bawah 200 ms membuat artefak vocoder terdengar. Chunk 320 ms adalah titik manisnya.
- **Tanpa jitter buffer.** Jaringan nyata penuh jitter; tanpa penghalusan kamu dapat pop.
- **Penanganan error single-shot.** Pipa audio harus tahan-crash. Satu eksepsi membunuh sesi.

## Rilis

Simpan sebagai `outputs/skill-realtime-designer.md`. Rancang pipa audio real-time dengan anggaran latensi konkret per tahap.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Mensimulasikan ring buffer + VAD energi; mencetak latensi tahap untuk stream 10 detik palsu.
2. **Sedang.** Dengan `sounddevice`, bangun loop passthrough yang memproses mikrofonmu dalam frame 20 ms dan mencetak status VAD di setiap frame.
3. **Sulit.** Bangun tes echo full-duplex dengan `aiortc`: browser → WebRTC → Python → WebRTC → browser. Ukur latensi glass-to-glass dengan pulsa 1 kHz.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |

## Bacaan Lanjutan

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — chunked near-streaming Whisper.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — full-duplex 200 ms latency.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — production audio agent orchestration.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — sub-1 ms VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — echo cancellation under open source.
