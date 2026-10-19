# Capstone 03 — Asisten Suara Real-Time (ASR ke LLM ke TTS)

> Agen suara yang terasa tepat punya latensi ujung-ke-ujung di bawah 800ms, tahu kapan kamu berhenti bicara, menangani barge-in, dan bisa memanggil tool tanpa tersendat. Retell, Vapi, LiveKit Agents, dan Pipecat semuanya mencapai standar itu pada 2026. Mereka melakukannya dengan bentuk yang sama: sebuah ASR streaming, sebuah turn-detector, sebuah LLM streaming, dan sebuah TTS streaming, semuanya disambungkan lewat WebRTC dengan anggaran latensi agresif di setiap hop. Bangun satu, ukur WER dan MOS serta tingkat false-cutoff, lalu jalankan di bawah packet loss.

**Type:** Capstone
**Languages:** Python (agent + pipeline), TypeScript (web client)
**Prerequisites:** Phase 6 (speech and audio), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 17 (infrastructure)
**Phases exercised:** P6 · P7 · P11 · P13 · P14 · P17
**Time:** 30 hours

## Masalahnya

Suara adalah kategori UX AI yang bergerak paling cepat pada 2025-2026. Plafon teknisnya turun setiap kuartal. OpenAI Realtime API, Gemini 2.5 Live, Cartesia Sonic-2, ElevenLabs Flash v3, LiveKit Agents 1.0, dan Pipecat 0.0.70 semuanya menempatkan first-audio-out sub-800ms dalam jangkauan. Standarnya bukan latensi semata. Yang diukur adalah rasa interaksinya: tidak memotong pengguna, tidak terpotong, pulih dari interupsi di tengah kalimat, memanggil tool di tengah percakapan tanpa membuat audio tersendat, dan bertahan di jaringan mobile yang jittery.

Kamu tidak bisa sampai ke sana dengan menyambung tiga panggilan REST. Arsitekturnya adalah streaming ber-pipeline dari ujung ke ujung. Bangun itu dan mode kegagalannya menjadi terlihat: sebuah VAD yang disetel untuk audio telepon memicu pada TV latar belakang, sebuah turn-detector yang menunggu tanda baca yang tak pernah datang, sebuah TTS yang men-buffer 400ms sebelum mengeluarkan audio. Tugas capstone ini adalah memperbaikinya satu per satu di bawah beban dan menerbitkan laporan latensi-dan-kualitas.

## Konsepnya

Pipeline punya lima tahap streaming: **audio masuk** (WebRTC dari browser atau PSTN), **ASR** (transkrip parsial streaming dari Deepgram Nova-3 atau faster-whisper), **deteksi giliran** (VAD plus sebuah model turn-detector kecil yang membaca transkrip parsial untuk petanda penyelesaian), **LLM** (token streaming begitu giliran dianggap selesai), **TTS** (audio keluar streaming dalam ~200ms setelah token LLM pertama).

Tiga kekhawatiran lintas-potong. **Barge-in**: saat pengguna mulai bicara ketika agen sedang bicara, TTS dibatalkan dan ASR langsung menyambung. **Tool use**: panggilan fungsi di tengah percakapan (cuaca, kalender) harus berjalan di kanal sampingan tanpa membuat audio tersendat; agen mengisi awal sebuah token pengakuan ("one second...") bila latensi melebihi 300ms. **Backpressure**: saat packet loss terjadi, transkrip parsial ditahan, VAD menaikkan ambang speech-gate, dan agen menghindari bicara di atas pesan yang belum diakui.

Tolok ukurnya kuantitatif. WER di bawah 8% pada benchmark VAD Hamming di SNR 15 dB. First-audio-out p50 di bawah 800ms pada 100 panggilan terukur. Tingkat false-cutoff di bawah 3%. MOS di atas 4.2 untuk TTS. 50 panggilan konkuren pada satu g5.xlarge. Angka-angka inilah deliverable-nya.

## Arsitektur

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## Stack

- Transport: LiveKit Agents 1.0 (WebRTC) plus gateway PSTN Twilio; Pipecat 0.0.70 sebagai framework alternatif
- ASR: Deepgram Nova-3 (streaming, first partial sub-300ms) atau faster-whisper Whisper-v3-turbo self-hosted
- VAD: Silero VAD v5 plus turn-detector LiveKit (transformer kecil yang membaca transkrip parsial)
- LLM: OpenAI GPT-4o-realtime untuk integrasi ketat, Gemini 2.5 Flash Live, atau Claude Haiku 4.5 secara kaskade (completion streaming, jalur audio terpisah)
- TTS: Cartesia Sonic-2 (first-byte terendah), ElevenLabs Flash v3, atau Orpheus open-source untuk self-host
- Tools: FastMCP kanal sampingan untuk cuaca/kalender/booking; agen memancarkan filler lebih dulu bila tool memakan >300ms
- Observability: span suara OpenTelemetry, trace suara Langfuse dengan replay audio
- Deployment: satu g5.xlarge (VRAM 24GB) untuk Whisper + Orpheus self-hosted; API hosted untuk latensi terendah

```figure
ce-voice-latency
```

## Bangun Sendiri

1. **Sesi WebRTC.** Dirikan sebuah room LiveKit dan web client yang men-stream audio mikrofon. Di sisi server, pasang sebuah agent worker yang bergabung ke room.

2. **Streaming ASR.** Berikan frame PCM 20ms ke Deepgram Nova-3 (atau faster-whisper di GPU). Berlangganan transkrip parsial dan final. Catat latensi per-parsial.

3. **VAD dan turn detector.** Jalankan Silero VAD v5 pada stream frame. Pada event speech-end, picu turn-detector LiveKit terhadap transkrip parsial terbaru. Baru commit ke "turn complete" saat VAD menyatakan hening selama 500ms dan turn-detector menilai penyelesaian > 0.6.

4. **Stream LLM.** Saat giliran selesai, mulai panggilan LLM dengan percakapan berjalan plus transkrip final. Stream token keluar. Pada token pertama, serahkan ke TTS.

5. **Stream TTS.** Cartesia Sonic-2 men-stream balik potongan audio. Potongan pertama harus meninggalkan server dalam 200ms setelah token LLM pertama. Pancarkan potongan ke room LiveKit; client memutar lewat jitter buffer WebRTC.

6. **Barge-in.** Saat VAD mendeteksi suara pengguna baru ketika TTS sedang diputar, batalkan stream TTS segera, buang sisa output LLM, dan aktifkan kembali ASR. Terbitkan span `tts_canceled`.

7. **Kanal sampingan tool.** Daftarkan cuaca dan kalender sebagai tool function-calling. Saat dipanggil, jalankan panggilannya secara konkuren; bila belum selesai dalam 300ms, minta LLM mengeluarkan "one second, let me check" sebagai filler; lanjutkan setelah tool mengembalikan hasil.

8. **Harness eval.** Rekam 100 panggilan. Hitung WER (terhadap transkrip held-out), tingkat false-cutoff (TTS dibatalkan saat pengguna di tengah kalimat), first-audio-out p50, MOS TTS (manusia atau NISQA), dan tes jitter-loss (jatuhkan 3% paket).

9. **Load test.** Dorong 50 panggilan konkuren pada satu g5.xlarge dengan penelepon sintetis. Ukur first-audio-out p95 yang berkelanjutan.

## Gunakan

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## Rilis

`outputs/skill-voice-agent.md` adalah deliverable-nya. Diberi sebuah domain (customer support, penjadwalan, atau kiosk), skill ini mendirikan agen LiveKit dengan pipeline ASR/VAD/LLM/TTS yang disetel ke tolok ukur pengukuran. Rubrik:

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | End-to-end latency | p50 first-audio-out under 800ms across 100 recorded calls |
| 20 | Turn-taking quality | False-cutoff rate under 3% on the Hamming VAD benchmark |
| 20 | Tool-use correctness | Mid-conversation tool calls that return the right data without stalling audio |
| 20 | Reliability under packet loss | WER and turn-taking stability with 3% packet drop injected |
| 15 | Eval harness completeness | Reproducible measurements with public config |
| **100** | | |

## Latihan

1. Tukar Deepgram Nova-3 dengan faster-whisper v3 turbo di atas g5.xlarge. Ukur jarak latensi dan WER-nya. Identifikasi di mana keputusan CPU-vs-GPU menjadi menentukan.

2. Tambahkan kebijakan arbitrase interupsi: apa yang dilakukan agen saat pengguna menyela di tengah panggilan tool? Bandingkan tiga kebijakan (batal keras, selesaikan-tool-lalu-berhenti, antrekan giliran berikutnya).

3. Jalankan tes turn-detector adversarial: berikan jeda panjang kepada pengguna di tengah kalimat. Setel ambang hening VAD dan ambang skor turn-detector untuk false-cutoff terendah tanpa melewati 900ms.

4. Deploy agen yang sama di PSTN lewat Twilio. Bandingkan first-audio-out PSTN dengan WebRTC. Jelaskan perbedaan jitter buffer dan codec-nya.

5. Tambahkan deteksi aktivitas suara untuk bahasa non-Inggris (Jepang, Spanyol). Ukur tingkat false-trigger Silero VAD v5 dibanding fine-tune khusus bahasa.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Turn detection | "End of utterance" | Classifier that, given VAD silence and a partial transcript, decides the user is done speaking |
| Barge-in | "Interruption handling" | Canceling TTS mid-playback when VAD detects new user speech |
| First-audio-out | "Latency" | Time from user stops speaking to the first audio packet leaving the server |
| VAD | "Speech gate" | Model classifying audio frames as speech vs silence; Silero VAD v5 is the 2026 default |
| Jitter buffer | "Audio smoothing" | Client-side buffer that holds packets briefly to absorb network variance |
| Filler | "Acknowledgment token" | Short phrase the agent emits to avoid silence when a tool is slow |
| MOS | "Mean opinion score" | Perceptual speech quality rating; NISQA is the automated proxy |

## Bacaan Lanjutan

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — framework agen WebRTC referensi
- [Pipecat](https://github.com/pipecat-ai/pipecat) — framework agen streaming alternatif berbasis Python
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — referensi untuk model speech terintegrasi
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — referensi ASR streaming
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — model referensi VAD
- [Cartesia Sonic-2](https://docs.cartesia.ai) — referensi TTS latensi rendah
- [Retell AI architecture](https://docs.retellai.com) — arsitektur agen suara produksi
- [Vapi.ai production stack](https://docs.vapi.ai) — referensi produksi alternatif
