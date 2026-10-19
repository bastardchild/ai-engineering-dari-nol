# Membangun Pipeline Voice Assistant — Capstone Fase 6

> Semua dari pelajaran 01-11, disatukan. Bangun asisten suara yang mendengar, bernalar, dan membalas. Di tahun 2026 itu adalah masalah teknik yang sudah solved, bukan masalah riset — tapi detail integrasinya yang menentukan apakah produknya benar-benar shipping.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 minutes

## Masalahnya

Bangun asisten end-to-end:

1. Menangkap input mic (16 kHz mono).
2. Mendeteksi awal/akhir bicara pengguna.
3. Transkripsi streaming.
4. Melempar transkrip ke LLM yang bisa memanggil tool (timer, cuaca, kalender).
5. Streaming teks LLM ke **TTS**.
6. Memutar audio kembali ke pengguna.
7. Berhenti jika pengguna menyela di tengah respons.

Target latensi: byte audio **TTS** pertama dalam 800 ms setelah pengguna selesai berbicara di CPU laptop. Target kualitas: tidak ada kata terlewat, tidak ada subtitle hasil halusinasi di keheningan, tidak ada kebocoran voice cloning, tidak ada prompt injection yang berhasil.

## Konsepnya

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### Tujuh komponennya

1. **Audio capture.** Mic → 16 kHz mono → chunk 20 ms. Biasanya `sounddevice` di Python atau AudioUnit/ALSA/WASAPI native di produksi.
2. **VAD (Pelajaran 11).** Silero VAD @ threshold 0.5, min speech 250 ms, silence hang-over 500 ms. Memberi sinyal "mulai" dan "selesai."
3. **Streaming STT (Pelajaran 4-5).** Whisper-streaming, Parakeet-TDT, atau Deepgram Nova-3 (API). Transkrip parsial + final.
4. **LLM dengan tool calling.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. Skema JSON untuk tool. Streaming token.
5. **Streaming TTS (Pelajaran 7).** Kokoro-82M (open tercepat) atau Cartesia Sonic (komersial). Mulai **TTS** setelah 20 token LLM.
6. **Playback.** Keluaran speaker; encode opus untuk jaringan ber-bandwidth rendah.
7. **Interruption handler.** Jika **VAD** terpicu saat playback **TTS**, hentikan playback, batalkan LLM, mulai ulang **STT**.

### Tiga mode kegagalan yang akan kamu hadapi

1. **First-word clip.** **VAD** mulai sedetik terlambat. Kata "hai" dari pengguna hilang. Mulai dengan threshold 0.3, bukan 0.5.
2. **Kebingungan interrupt di tengah respons.** LLM terus menghasilkan setelah pengguna menyela; asisten bicara menimpa pengguna. Sambungkan **VAD** → cancel-LLM.
3. **Halusinasi karena keheningan.** Whisper mengeluarkan "Thanks for watching" pada frame pemanasan yang sunyi. Selalu pasang gerbang **VAD**.

### Stack referensi produksi 2026

| Stack | Latency | License | Notes |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | Industry default 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | DIY-friendly |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Single-model; different architecture, lesson 15 |
| Vapi / Retell (managed) | 300-500 ms | commercial | Fastest to launch; limited customization |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | Privacy / edge |

```figure
v4-voice-latency
```

## Bangun Sendiri

### Langkah 1: capture mic dengan chunking (pseudokode)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Langkah 2: capture giliran bergerbang **VAD**

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Langkah 3: streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Langkah 4: tool calling di dalam loop LLM

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Langkah 5: penanganan interupsi

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Gunakan

Lihat `code/main.py` untuk simulasi yang bisa dijalankan dan menyambungkan ketujuh komponen dengan model stub, sehingga kamu bisa melihat bentuk pipeline-nya bahkan tanpa hardware. Untuk implementasi nyata, ganti stub dengan:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` atau `openai-whisper`
- `openai` (`gpt-4o`) atau `anthropic`
- `kokoro` atau `cartesia`
- `sounddevice` untuk I/O

## Jebakan

- **Mencatat PII selamanya.** Audio giliran penuh adalah PII di sebagian besar yurisdiksi. Retensi 30 hari, dienkripsi saat at-rest.
- **Tanpa barge-in.** Pengguna pasti akan menyela. Asistenmu harus berhenti bicara.
- **TTS yang memblokir.** **TTS** sinkron memblokir event loop. Pakai async atau thread terpisah.
- **Tanpa error handling tool-call.** Tool bisa gagal. LLM harus menerima kembali error + retry sekali, lalu degrade dengan anggun.
- **Filter halusinasi yang terlalu agresif.** Over-filter dan asisten mengulang "I can't help with that." Under-filter dan ia mengatakan apa saja. Kalibrasi pada held-out set.
- **Tanpa opsi wake-word.** Always-listening adalah liabilitas privasi. Tambahkan gerbang wake-word (Porcupine atau openWakeWord).

## Rilis

Simpan sebagai `outputs/skill-voice-assistant-architect.md`. Diberikan batasan budget + skala + bahasa + compliance, hasilkan spesifikasi stack lengkap.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ia mensimulasikan satu giliran penuh end-to-end dengan modul stub dan mencetak latensi per tahap.
2. **Sedang.** Ganti stub **STT** dengan model Whisper nyata pada file `.wav` yang sudah direkam. Ukur WER dan latensi end-to-end.
3. **Sulit.** Tambahkan tool calling: implementasikan `get_weather` (API apa pun) dan `set_timer`. Arahkan LLM melalui tool dan verifikasi bahwa saat pengguna mengatakan "set a 5 minute timer" fungsi yang tepat terpicu dan balasan suara mengonfirmasinya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Turn | A user + assistant round-trip | One VAD-bounded user speech + one LLM-TTS response. |
| Barge-in | Interruption | User speaks while assistant talks; assistant stops. |
| Wake word | "Hey assistant" | Short keyword detector; Porcupine, Snowboy, openWakeWord. |
| End-pointing | Turn ending | VAD + min-silence decision that user has finished. |
| Pre-roll | Pre-speech buffer | Keep 200-400 ms of audio before VAD fires to avoid first-word clip. |
| Tool call | Function invocation | LLM emits JSON; runtime dispatches; result feeds back in-loop. |

## Bacaan Lanjutan

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — production-grade reference.
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY-friendly framework.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — the managed voice-native path.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — full-duplex reference (Lesson 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — wake-word gating.
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM function calling.
