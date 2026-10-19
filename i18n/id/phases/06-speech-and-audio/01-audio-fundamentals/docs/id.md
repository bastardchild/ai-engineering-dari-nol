# Dasar Audio — Waveform, Sampling, Transformasi Fourier

> Waveform adalah sinyal mentah. Spektrogram adalah representasinya. Fitur Mel adalah bentuk ramah-ML. Setiap pipeline ASR dan TTS modern menaiki tangga ini, dan anak tangga pertama adalah memahami sampling dan Fourier.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45 minutes

## Masalahnya

Mikrofon menghasilkan sinyal tekanan-vs-waktu. Neural net-mu mengonsumsi tensor. Di antaranya duduk tumpukan konvensi yang, ketika dilanggar, menghasilkan bug diam: model berlatih baik-baik saja tetapi WER berlipat, atau TTS mengirim desis, atau sistem kloning suara menghafal mikrofon alih-alih pembicara.

Setiap bug dalam sistem speech berakar pada satu dari tiga pertanyaan:

1. Pada sample rate berapa data direkam, dan apa yang diharapkan model?
2. Apakah sinyal ter-alias?
3. Apakah kamu beroperasi pada sampel mentah atau pada representasi frekuensi?

Benarkan ini dan sisa Fase 6 terkendali. Salahkan dan bahkan Whisper-Large-v4 menghasilkan sampah.

## Konsepnya

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**Waveform.** Array satu-dimensi float dalam `[-1.0, 1.0]`. Diindeks nomor sampel. Untuk konversi ke detik, bagi dengan sample rate: `t = n / sr`. Klip 10-detik pada 16 kHz adalah array 160.000 float.

**Sample rate (sr).** Berapa sampel per detik. Rate umum di 2026:

| Rate | Use |
|------|-----|
| 8 kHz | Telephony, legacy VOIP. Nyquist at 4 kHz kills consonants. Avoid for ASR. |
| 16 kHz | ASR standard. Whisper, Parakeet, SeamlessM4T v2 all consume 16 kHz. |
| 22.05 kHz | TTS vocoder training for older models. |
| 24 kHz | Modern TTS (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | CD audio, music. |
| 48 kHz | Film, pro audio, high-fidelity TTS (VALL-E 2, NaturalSpeech 3). |

**Nyquist-Shannon.** Sample rate `sr` bisa merepresentasikan frekuensi sampai `sr/2` secara tak ambigu. Batas `sr/2` adalah *frekuensi Nyquist*. Energi di atas Nyquist ter-*alias* — terlipat turun ke frekuensi lebih rendah — dan mengorupsi sinyal. Selalu low-pass filter sebelum downsampling.

**Bit depth.** PCM 16-bit (signed int16, rentang ±32,767) adalah format pertukaran universal. 24-bit untuk musik, float 32-bit untuk DSP internal. Pustaka seperti `soundfile` membaca int16 tetapi mengekspos array float32 dalam `[-1, 1]`.

**Transformasi Fourier.** Sinyal berhingga apa pun adalah jumlah sinusoid pada frekuensi berbeda. Discrete Fourier Transform (DFT) menghitung, untuk `N` sampel, `N` koefisien kompleks — satu per bin frekuensi. `bin k` dipetakan ke frekuensi `k · sr / N` Hz. Magnitudo adalah amplitudo pada frekuensi itu, sudut adalah fase.

**FFT.** Fast Fourier Transform: algoritme `O(N log N)` untuk DFT ketika `N` adalah pangkat 2. Setiap pustaka audio memakai FFT di balik layar. FFT 1024-sampel pada 16 kHz memberi 512 bin frekuensi usable merentang 0–8 kHz pada resolusi 15.6 Hz.

**Framing + window.** Kita tidak mem-FFT seluruh klip. Kita memotongnya menjadi *frame* yang overlap (umumnya 25 ms dengan hop 10 ms), mengalikan tiap frame dengan fungsi window (Hann, Hamming) untuk membunuh diskontinuitas tepi, lalu FFT tiap frame. Inilah Short-Time Fourier Transform (STFT). Pelajaran 02 melanjutkan dari sini.

```figure
mel-scale
```

## Bangun Sendiri

### Langkah 1: baca klip dan plot waveform

`code/main.py` hanya memakai modul stdlib `wave` agar demo bebas-dependensi. Untuk produksi kamu akan memakai `soundfile` atau `torchaudio.load` (keduanya mengembalikan tuple `(waveform, sr)`):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### Langkah 2: sintesis gelombang sinus dari prinsip pertama

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

Sinus 440 Hz (concert A) pada 16 kHz selama 1 detik adalah 16.000 float. Tulis dengan `wave.open(..., "wb")` memakai encoding PCM 16-bit.

### Langkah 3: hitung DFT dengan tangan

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — baik untuk `N=256` guna mengonfirmasi kebenaran, tak berguna untuk audio nyata. Kode nyata memanggil `numpy.fft.rfft` atau `torch.fft.rfft`.

### Langkah 4: temukan frekuensi dominan

Puncak magnitudo indeks `k_star` dipetakan ke frekuensi `k_star * sr / N`. Menjalankannya pada sinus 440 Hz harus mengembalikan puncak pada bin `440 * N / sr`.

### Langkah 5: demonstrasikan aliasing

Sampel sinus 7 kHz pada 10 kHz (Nyquist = 5 kHz). Nada 7 kHz di atas Nyquist dan terlipat ke `10 − 7 = 3 kHz`. Puncak FFT muncul pada 3 kHz. Ini demo aliasing klasik dan alasan setiap DAC/ADC dikirim dengan low-pass filter brick-wall.

## Gunakan

Stack yang benar-benar kamu kirim di 2026:

| Task | Library | Why |
|------|---------|-----|
| Read/write WAV/FLAC/OGG | `soundfile` (libsndfile wrapper) | Fastest, stable, returns float32. |
| Resample | `torchaudio.transforms.Resample` or `librosa.resample` | Correct anti-aliasing built in. |
| STFT / Mel | `torchaudio` or `librosa` | GPU-friendly; PyTorch ecosystem. |
| Real-time streaming | `sounddevice` or `pyaudio` | Cross-platform PortAudio bindings. |
| Inspect a file | `ffprobe` or `soxi` | CLI, fast, reports sr/channels/codec. |

Aturan keputusan: **cocokkan sample rate sebelum mencocokkan apa pun**. Whisper mengharapkan 16 kHz mono float32. Beri ia 44.1 kHz stereo dan kamu akan mendapat sampah yang tampak seperti bug model.

## Rilis

Simpan sebagai `outputs/skill-audio-loader.md`. Skill ini membantumu memeriksa bahwa input audio cocok dengan ekspektasi model downstream dan me-resample dengan benar ketika tidak cocok.

## Latihan

1. **Mudah.** Sintesis campuran 1-detik 220 Hz + 440 Hz + 880 Hz pada 16 kHz. Jalankan DFT. Konfirmasi tiga puncak pada bin ekspektasi.
2. **Sedang.** Rekam WAV suaramu 3-detik pada 48 kHz. Downsample ke 16 kHz memakai `torchaudio.transforms.Resample` (dengan anti-aliasing), lalu ke 16 kHz memakai desimasi naif (setiap sampel ketiga). FFT keduanya. Di mana aliasing muncul?
3. **Sulit.** Bangun STFT dari nol hanya memakai `math` dan DFT dari Langkah 3. Ukuran frame 400, hop 160, window Hann. Plot magnitudo dengan `matplotlib.pyplot.imshow`. Ini spektrogram Pelajaran 02.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sample rate | How many samples per second | Frequency in Hz at which the ADC measures the signal. |
| Nyquist | The max frequency you can represent | `sr/2`; energy above it aliases back down. |
| Bit depth | Resolution of each sample | `int16` = 65,536 levels; `float32` = 24-bit precision in `[-1, 1]`. |
| DFT | The Fourier transform for sequences | `N` samples → `N` complex frequency coefficients. |
| FFT | The fast DFT | `O(N log N)` algorithm requiring `N` = power of 2. |
| Bin | Frequency column | `k · sr / N` Hz; resolution = `sr / N`. |
| STFT | Spectrogram under the hood | Framed + windowed FFT over time. |
| Aliasing | Weird frequency ghosts | Energy above Nyquist mirroring down to lower bins. |

## Bacaan Lanjutan

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — the paper behind the sampling theorem.
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — free, canonical DSP textbook.
- [librosa docs — audio primer](https://librosa.org/doc/latest/auto_tutorials/index.html) — practical walkthrough with code.
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.taylorfrancis.com/books/mono/10.1201/9781315372150/room-acoustics-heinrich-kuttruff) — reference for why real-world audio is not a clean sinusoid.
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — frequency bin intuition cleared up in 10 minutes.
