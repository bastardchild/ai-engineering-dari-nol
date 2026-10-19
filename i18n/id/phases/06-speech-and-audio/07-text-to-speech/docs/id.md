# Text-to-Speech (TTS) — Dari Tacotron ke F5 dan Kokoro

> ASR membalik ucapan menjadi teks; TTS membalik teks menjadi ucapan. Stack 2026 punya tiga bagian: teks → token, token → mel, mel → waveform. Setiap bagian punya model default yang muat di laptop.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Masalahnya

Kamu punya sebuah string: "Please remind me to water the plants at 6 pm." Kamu butuh klip audio 3 detik yang terdengar natural, punya prosodi yang benar (jeda, penekanan), mengucapkan "plants" dengan vokal yang tepat, dan berjalan di bawah 300 ms di CPU untuk asisten suara langsung. Kamu juga perlu menukar suara, menangani input code-switched ("remind me at 6 pm, daijoubu?"), dan tidak memalukan diri sendiri pada nama.

Pipeline TTS modern terlihat seperti ini:

1. **Text frontend.** Menormalisasi teks (tanggal, angka, email), mengonversi ke fonem atau token subword, memprediksi fitur prosodi.
2. **Acoustic model.** Teks → mel spectrogram. Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
3. **Vocoder.** Mel → waveform. WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), neural codec vocoder di 2024+.

Pada 2026, pemisahan acoustic + vocoder menjadi kabur dengan model diffusion end-to-end dan flow-matching. Tapi model mental tiga bagian masih berlaku untuk debugging.

## Konsepnya

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017).** Seq2seq: char-embedding → encoder BiLSTM → attention location-sensitive → decoder LSTM autoregressive memancarkan frame mel. Lambat (AR), goyah pada teks panjang. Masih dikutip sebagai baseline.

**FastSpeech 2 (2020).** Non-autoregressive. Duration predictor mengeluarkan berapa frame mel yang diterima setiap fonem. 1-pass, 10× lebih cepat dari Tacotron. Kehilangan beberapa naturalness (alignment monotonic) tapi dipakai di mana-mana.

**VITS (2021).** Melatih secara bersama encoder + duration berbasis-flow + vocoder HiFi-GAN end-to-end dengan inferensi variasional. Kualitas tinggi, satu model. Mendominasi TTS open-source 2022–2024. Variannya: YourTTS (multi-speaker zero-shot), XTTS v2 (2024, Coqui).

**F5-TTS (2024).** Diffusion transformer di atas flow matching. Prosodi natural, kloning suara zero-shot dengan 5 detik audio referensi. Puncak leaderboard TTS open-source 2026. 335M parameter.

**Kokoro (2024).** Kecil (82M), bisa jalan di CPU, TTS Inggris kelas-terbaik untuk penggunaan real-time. English-only dengan vocabulary tertutup, apache-2.0.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** State of the art komersial. Tag emosi ElevenLabs v2.5 ("[whispered]", "[laughing]") dan suara karakter mendominasi produksi audiobook pada 2026.

### Evolusi vocoder

| Era | Vocoder | Latency | Quality |
|-----|---------|---------|---------|
| 2016 | WaveNet | offline only | SOTA at release |
| 2018 | WaveRNN | ~realtime | good |
| 2020 | HiFi-GAN | 100× realtime | near-human |
| 2022 | BigVGAN | 50× realtime | generalizes across speakers/langs |
| 2024 | SNAC, DAC (neural codecs) | integrated with AR models | discrete tokens, bit-efficient |

Sampai 2026 sebagian besar model "TTS" bersifat end-to-end dari teks ke waveform; mel spectrogram adalah representasi internal.

### Evaluasi

- **MOS (Mean Opinion Score).** Skala 1–5, berbasis crowdsourcing. Masih standar emas; menyakitkan lambat.
- **CMOS (Comparative MOS).** Preferensi A-vs-B. Interval keyakinan lebih ketat per anotasi.
- **UTMOS, DNSMOS.** Prediktor MOS neural tanpa referensi. Dipakai untuk leaderboard.
- **CER (Character Error Rate) via ASR.** Jalankan output TTS melalui Whisper, hitung CER terhadap teks input. Proxy untuk intelligibility.
- **SECS (Speaker Embedding Cosine Similarity).** Kualitas kloning suara.

Angka 2026 pada LibriTTS test-clean:

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

```figure
sp-tts-stack
```

## Bangun Sendiri

### Langkah 1: fonemisasi input

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Phoneme adalah jembatan universal. Hindari memberi teks mentah ke apa pun di bawah kualitas level VITS.

### Langkah 2: jalankan Kokoro (default CPU 2026)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

Berjalan offline, satu file, 82M parameter.

### Langkah 3: jalankan F5-TTS dengan voice cloning

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

Berikan klip referensi 5 detik + transkripsinya; F5 mengkloning prosodi dan timbre.

### Langkah 4: vocoder HiFi-GAN dari nol

Terlalu besar untuk dimuat dalam skrip tutorial, tapi bentuknya seperti ini:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

Training: adversarial (discriminator pada jendela pendek) + loss rekonstruksi mel-spectrogram + feature-matching loss. Sudah komoditisasi — pakai checkpoint pretrained dari repo `hifi-gan` atau nvidia-NeMo.

### Langkah 5: pipeline penuh (pseudocode)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Gunakan

Stack 2026:

| Situation | Pick |
|-----------|------|
| Real-time English voice assistant | Kokoro (CPU) or XTTS v2 (GPU) |
| Voice cloning from 5 s reference | F5-TTS |
| Commercial character voices | ElevenLabs v2.5 |
| Audiobook narration | ElevenLabs v2.5 or XTTS v2 + fine-tune |
| Low-resource language | Train VITS on 5–20 h target-lang data |
| Expressive / emotion tags | ElevenLabs v2.5 or StyleTTS 2 fine-tune |

Pemimpin open-source per 2026: **F5-TTS untuk kualitas, Kokoro untuk efisiensi**. Jangan menjangkau Tacotron kecuali kamu seorang sejarawan.

## Jebakan

- **Tanpa text normalizer.** "Dr. Smith" dibaca sebagai "Doctor" atau "Drive"? "2026" sebagai "twenty twenty six" atau "two zero two six"? Normalisasi SEBELUM phonemizer.
- **Proper noun OOV.** "Ghumare" → "ghyu-mair"? Sediakan model fallback grapheme-to-phoneme untuk token yang tidak dikenal.
- **Clipping.** Output vocoder jarang clip, tapi ketidakcocokan skala mel saat inference bisa melebihi ±1.0. Selalu `np.clip(wav, -1, 1)`.
- **Sample-rate mismatch.** Kokoro mengeluarkan 24 kHz; pipeline downstream-mu mengharapkan 16 kHz → resample atau dapatkan aliasing.

## Rilis

Simpan sebagai `outputs/skill-tts-designer.md`. Rancang pipeline TTS untuk target suara, latensi, dan bahasa tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Membangun dictionary fonem dari vocab mainan, memperkirakan durasi per fonem, dan mencetak jadwal "mel" palsu.
2. **Sedang.** Install Kokoro, sintesis kalimat yang sama dengan suara `af_bella` dan `am_adam`. Bandingkan durasi audio dan kualitas subjektif.
3. **Sulit.** Rekam klip referensi 5 detik dari dirimu sendiri. Gunakan F5-TTS untuk mengkloningnya. Laporkan SECS antara referensi dan output hasil kloningan.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Phoneme | Sound unit | Abstract sound class; 39 in English (ARPABet). |
| Duration predictor | How long each phoneme lasts | Non-AR model output; integer frames per phoneme. |
| Vocoder | Mel → waveform | Neural net mapping mel-spec to raw samples. |
| HiFi-GAN | Standard vocoder | GAN-based; dominant 2020–2024. |
| MOS | Subjective quality | 1–5 mean opinion score from human raters. |
| SECS | Voice-clone metric | Cosine similarity between target and output speaker embedding. |
| F5-TTS | 2024 open-source SOTA | Flow-matching diffusion; zero-shot cloning. |
| Kokoro | CPU English leader | 82M-param model, Apache 2.0. |

## Bacaan Lanjutan

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — the seq2seq baseline.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — end-to-end flow-based.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — current open-source SOTA.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — the vocoder that still ships in 2026.
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 CPU-friendly English TTS.
