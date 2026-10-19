# Suara Agen: Pipecat dan LiveKit

> Suara agen adalah sebuah pertama-class produksi category di 2026. Pipecat memberi kamu sebuah Python frame-based pipeline (VAD → STT → LLM → TTS → transport). LiveKit Agen bridges AI model ke users atas WebRTC. Produksi latency targets land pada 450–600ms ujung-ke-ujung untuk premium stacks.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Deskripsikan Pipecat's frame-based pipeline: DOWNSTREAM (source→sink) dan UPSTREAM (kontrol).
- Name kanonikal suara pipeline stages dan yang transports Pipecat mendukung.
- Jelaskan LiveKit Agen' dua suara agen classes (MultimodalAgent, VoicePipelineAgent) dan ketika setiap fits.
- Summarize 2026 produksi latency expectations dan bagaimana mereka drive architecture choices.

## Masalahnya

Suara agen adalah tidak sebuah text loop dengan TTS bolted pada. Latency budgets adalah brutal (~600ms), partial audio adalah default, giliran detection adalah sebuah model, dan transports range dari telephony SIP ke WebRTC. Either kamu build sebuah frame-based pipeline (Pipecat) atau kamu lean pada sebuah platform (LiveKit).

## Konsepnya

### Pipecat (pipecat-ai/pipecat)

- Python frame-based pipeline framework.
- `Frame` → `FrameProcessor` chain.
- Dua flow directions:
  - **DOWNSTREAM** — source → sink (audio di, TTS out).
  - **UPSTREAM** — feedback dan kontrol (cancellation, metrik, barge-di).
- `PipelineTask` mengelola lifecycle dengan events (`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`) dan observers untuk metrik/tracing/RTVI.

Typical pipeline:

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transports: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flow adds structured conversations (state machines). Pipecat Cloud adalah managed runtime.

### LiveKit Agen (livekit/agen)

- Bridges AI model ke users atas WebRTC.
- Kunci concepts: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- Dua suara agen classes:
  - **MultimodalAgent** — direct audio via OpenAI Realtime atau equivalent.
  - **VoicePipelineAgent** — STT → LLM → TTS cascade; memberi text-level kontrol.
- Semantic giliran detection via sebuah transformer model.
- Native MCP integration.
- Telephony via SIP.
- 50+ model dengan tidak API keys via LiveKit Inference; 200+ more via plugins.

### Commercial platforms

Vapi (~450–600ms pada sebuah optimized premium stack) dan Retell (~600ms ujung-ke-ujung lintas 180 tes memanggil) build di atas ini. Pick sebuah platform ketika kamu inginkan sebuah managed suara stack tanpa sebuah WebRTC tim.

### Di mana ini pola goes salah

- **No barge-in handling.** User interrupts; agen keeps talking. Membutuhkan UPSTREAM cancel frames di Pipecat, equivalent di LiveKit.
- **STT confidence ignored.** Low-confidence transcripts fed ke LLM sebagai jika gospel. Gate pada confidence atau request confirmation.
- **TTS mid-sentence cutoff.** Ketika pipeline cancels mid-utterance, TTS membutuhkan ke know atau cut audio.
- **Latency budget ignored.** Setiap component adds 50–200ms. Sum kamu chain sebelum shipping.

### Typical 2026 latencies

- VAD: 20–60ms
- STT partial: 100–250ms
- LLM pertama token: 150–400ms
- TTS pertama audio: 100–200ms
- Transport RTT: 30–80ms

Ujung-ke-ujung 450–600ms adalah premium. 800–1200ms adalah umum. Anything > 1500ms feels broken.

```figure
voice-pipeline
```

## Bangun Sendiri

`code/main.py` adalah sebuah frame-based toy pipeline dengan:

- `Frame` types (audio, transkrip, text, tts_audio, kontrol).
- `Processor` interface dengan `process(frame)`.
- SEBUAH lima-stage pipeline (VAD → STT → LLM → TTS → transport) sebagai scripted processors.
- Sebuah UPSTREAM cancel frame ke demonstrasikan barge-di.

Jalankan:

```
python3 code/main.py
```

 trace menunjukkan normal flow dan sebuah barge-di cancel itu berhenti TTS mid-utterance.

## Gunakan

- **Pipecat** untuk penuh kontrol — custom processors, Python-pertama, pluggable providers.
- **LiveKit Agents** untuk WebRTC-pertama deployments dan telephony.
- **Vapi / Retell** untuk hosted suara agen tanpa sebuah WebRTC tim.
- **OpenAI Realtime / Gemini Live** untuk direct audio-di/audio-out (MultimodalAgent).

## Rilis

`outputs/skill-voice-pipeline.md` scaffolds sebuah Pipecat-shaped suara pipeline dengan VAD + STT + LLM + TTS + transport plus barge-di handling.

## Latihan

1. Tambahkan sebuah metrik observer ke kamu toy pipeline: count frames per stage per kedua. Di mana melakukan latency accumulate?
2. Implementasikan confidence-gated STT: di bawah threshold, request "bisa kamu repeat itu?"
3. Add semantic giliran detection: sederhana aturan — jika transkrip berakhir dengan "?", berakhir dari giliran.
4. Baca Pipecat's transport docs. Tukar stdlib transport untuk SmallWebRTCTransport config (stub).
5. Measure sebuah OpenAI Realtime vs STT+LLM+TTS cascade pada sama query. Apa latency cost melakukan text-level kontrol carry?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Frame | "Event" | Typed unit of data in the pipeline (audio, transcript, text, control) |
| Processor | "Pipeline stage" | Handler with process(frame) |
| DOWNSTREAM | "Forward flow" | Source to sink: audio in, speech out |
| UPSTREAM | "Feedback flow" | Control: cancel, metrics, barge-in |
| VAD | "Voice activity detection" | Detects when user is speaking |
| Semantic turn detection | "Smart end-of-turn" | Model-based decision that the user is done |
| MultimodalAgent | "Direct audio agent" | Audio in, audio out; no text in the middle |
| VoicePipelineAgent | "Cascade agent" | STT + LLM + TTS; text-level control |

## Bacaan Lanjutan

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — frame-based pipeline, processors, transports
- [LiveKit Agen docs](https://docs.livekit.io/agents/) — WebRTC + suara primitives
- [Vapi](https://vapi.ai/) — managed suara platform
- [Retell AI](https://www.retellai.com/) — managed suara, latency-benchmarked
