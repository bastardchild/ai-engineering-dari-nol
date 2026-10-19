# Voice Activity Detection & Turn-Taking — Silero, Cobra, dan Trik Flush

> Setiap voice agent hidup atau mati pada dua keputusan: apakah pengguna sedang berbicara sekarang, dan apakah mereka sudah selesai? **VAD** menjawab yang pertama. Turn-detection (**VAD** + silence-hangover + semantic endpoint model) menjawab yang kedua. Salah satu saja dan asistenmu akan memotong pengguna atau tidak pernah berhenti bicara.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 minutes

## Masalahnya

Tiga keputusan berbeda yang diambil voice agent pada setiap chunk 20 ms:

1. **Apakah frame ini suara?** — **VAD**. Binernya, per-frame.
2. **Apakah pengguna sudah memulai ucapan baru?** — onset detection.
3. **Apakah pengguna sudah selesai?** — end-pointing (turn-end).

Jawaban naif (threshold energi) gagal pada derau apa pun — lalu lintas, keyboard, kicauan kerumunan. Jawaban 2026: Silero VAD (open, deep-learned) + model turn-detection (semantic endpointing) + silence hangover yang dikalibrasi **VAD**.

## Konsepnya

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### Kaskade **VAD** tiga tingkat

**Tingkat 1: gerbang energi.** Termurah. Threshold RMS pada -40 dBFS. Menyaring keheningan yang jelas tapi terpicu pada derau apa pun di atas threshold.

**Tingkat 2: Silero VAD** (2020-2026, MIT). 1M parameter. Dilatih pada 6000+ bahasa. Berjalan dalam ~1 ms per chunk 30 ms pada satu thread CPU. TPR 87.7% pada FPR 5%. Default open-source.

**Tingkat 3: semantic turn detector.** Model turn-detection milik LiveKit (2024-2026) atau classifier kecilmu sendiri. Membedakan "jeda di tengah kalimat" dari "selesai bicara." Memakai konteks linguistik (intonasi + kata-kata terkini), bukan cuma keheningan.

### Parameter kunci dan default-nya

- **Threshold.** Silero mengeluarkan probabilitas; klasifikasikan sebagai suara pada &gt; 0.5 (default) atau &gt; 0.3 (sensitif). Threshold lebih rendah = fewer first-word clips, lebih banyak false positive.
- **Durasi bicara minimum.** Tolak suara yang lebih pendek dari 250 ms — biasanya batuk atau derau kursi.
- **Silence hangover (end-pointing).** Setelah **VAD** kembali ke 0, tunggu 500-800 ms sebelum menyatakan end-of-turn. Terlalu pendek → menyela pengguna. Terlalu panjang → terasa lamban.
- **Pre-roll buffer.** Simpan 300-500 ms audio sebelum **VAD** terpicu. Mencegah kata "hai" terpotong.

### Trik flush (Kyutai 2025)

Model **STT** streaming punya delay look-ahead (500 ms untuk Kyutai STT-1B, 2.5 s untuk STT-2.6B). Biasanya kamu menunggu sepanjang itu setelah akhir-suara untuk mendapatkan transkrip. Trik flush: saat **VAD** memberi sinyal akhir-suara, **kirim sinyal flush ke STT** yang memaksa keluaran segera. **STT** memproses pada ~4× realtime, jadi buffer 500 ms selesai dalam ~125 ms.

End-to-end: **VAD** 125 ms + flush **STT** = latensi konversasional.

### Perbandingan **VAD** 2026

| VAD | TPR @ 5% FPR | Latency | License |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero adalah default yang tepat. Cobra adalah upgrade compliance / akurasi. **VAD** berbasis energi saja tidak punya tempat di produksi 2026.

```figure
sp-vad-cascade
```

## Bangun Sendiri

### Langkah 1: gerbang energi

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Langkah 2: Silero VAD di Python

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Langkah 3: state machine turn-end

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Langkah 4: kerangka trik flush

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

**STT** (Kyutai, Deepgram, AssemblyAI) harus mendukung flush agar ini berfungsi. Whisper streaming tidak — ia berbasis block dan selalu menunggu chunk.

## Gunakan

| Situation | VAD choice |
|-----------|-----------|
| Open, fast, general | Silero VAD |
| Commercial call center | Cobra VAD |
| On-device (phone) | Silero VAD ONNX |
| Research / diarization | pyannote segmentation |
| Zero-dependency fallback | WebRTC VAD (legacy) |
| Need turn-ending quality | Silero + LiveKit turn-detector layered |

Aturan praktis: jangan pernah shipping **VAD** berbasis energi saja kecuali kamu benar-benar tak punya pilihan lain.

## Jebakan

- **Threshold tetap.** Bekerja di tempat sepi, gagal di tempat berisik. Kalibrasi di perangkat atau beralih ke Silero.
- **Silence hangover terlalu pendek.** Agen menyela di tengah kalimat. 500-800 ms adalah titik manis untuk percakapan.
- **Hangover terlalu panjang.** Terasa lamban. Uji A/B dengan pengguna target.
- **Tanpa pre-roll buffer.** 200-300 ms pertama audio pengguna hilang. Selalu simpan pre-roll bergulir.
- **Mengabaikan semantic endpointing.** "Hmm, let me think..." mengandung jeda panjang. Pengguna benci dipotong di tengah pikiran. Pakai turn-detector milik LiveKit atau sejenisnya.

## Rilis

Simpan sebagai `outputs/skill-vad-tuner.md`. Pilih model **VAD**, threshold, hangover, pre-roll, dan strategi turn-detection untuk satu beban kerja.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ia mensimulasikan sekuens suara + hening + suara + batuk dan menguji tiga tingkat **VAD**.
2. **Sedang.** Install `silero-vad`, proses rekaman 5 menit, atur threshold untuk meminimalkan baik first-word clip maupun false trigger. Laporkan precision/recall.
3. **Sulit.** Bangun mini turn-detector: Silero VAD + MLP 3-layer di atas embedding 10 kata terakhir (pakai sentence-transformers). Latih pada dataset turn-end berlabel manual. Kalahkan Silero saja sebesar 10% F1.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| VAD | Voice detector | Binary per-frame: is this speech? |
| Turn detection | End-pointing | VAD + silence-hangover + semantic endpoint. |
| Silence hangover | Wait-after-speech | Time to wait before declaring turn end; 500-800 ms. |
| Pre-roll | Pre-speech buffer | Keep 300-500 ms audio before VAD fires. |
| Flush trick | Kyutai hack | VAD → flush-STT → 125 ms instead of 500 ms delay. |
| Semantic endpoint | "Did they mean to stop?" | ML classifier that looks at words, not just silence. |
| TPR @ FPR 5% | ROC point | Standard VAD benchmark; 87.7% for Silero, 50% WebRTC. |

## Bacaan Lanjutan

- [Silero VAD](https://github.com/snakers4/silero-vad) — the reference open VAD.
- [Picovoice Cobra VAD](https://picovoice.ai/products/voice/voice-activity-detection/) — commercial accuracy leader.
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — the sub-200 ms engineering trick.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — semantic endpointing in production.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — the legacy baseline.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — diarization-grade segmentation.
