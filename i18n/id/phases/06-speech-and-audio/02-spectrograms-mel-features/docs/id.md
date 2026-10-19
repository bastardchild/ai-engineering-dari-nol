# Spektrogram, Skala Mel & Fitur Audio

> Neural net tidak mengonsumsi waveform mentah dengan baik. Mereka mengonsumsi spektrogram. Mereka mengonsumsi spektrogram mel bahkan lebih baik. Setiap pengklasifikasi ASR, TTS, dan audio di 2026 hidup-mati oleh satu pilihan pra-pemrosesan ini.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 01 (Audio Fundamentals)
**Time:** ~45 minutes

## Masalahnya

Ambil klip 10-detik 16 kHz. Itu 160.000 float, semuanya dalam `[-1, 1]`, hampir sempurna tak berkorelasi dengan label "dog barking" atau "kata cat". Waveform mentah punya informasinya tetapi dalam bentuk yang sulit diekstrak model. Dua fonem identik yang diucapkan 100 ms berjauhan punya sampel mentah yang sama sekali berbeda.

Spektrogram memperbaikinya. Ia meruntuhkan detail temporal di mana persepsi manusia mengabaikannya (jitter mikrosekon) dan menjaga struktur di mana persepsi memperhatikan (frekuensi mana yang energetik, atas window waktu ~10–25 ms).

Spektrogram mel mendorong lebih jauh. Manusia mempersepsikan pitch secara logaritmis: 100 Hz vs 200 Hz terdengar "berjarak sama" seperti 1000 Hz vs 2000 Hz. Skala mel melengkungkan sumbu frekuensi agar cocok. Spektrogram berskala-mel adalah fitur tunggal terpenting dalam ML speech dari 2010 sampai 2026.

## Konsepnya

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT (Short-Time Fourier Transform).** Iris waveform menjadi frame yang overlap (tipikal: window 25 ms, hop 10 ms = 400 sampel / 160 sampel pada 16 kHz). Kalikan tiap frame dengan fungsi window (Hann adalah default; Hamming tradeoff sedikit berbeda). FFT tiap frame. Tumpuk spektrum magnitudo menjadi matriks berbentuk `(n_frames, n_freq_bins)`. Itulah spektrogrammu.

**Log-magnitudo.** Magnitudo mentah merentang 5-6 orde magnitudo. Ambil `log(|X| + 1e-6)` atau `20 * log10(|X|)` untuk memampatkan rentang dinamis. Setiap pipeline produksi memakai log-magnitudo, bukan magnitudo mentah.

**Skala Mel.** Frekuensi `f` dalam Hz dipetakan ke mel `m` via `m = 2595 * log10(1 + f / 700)`. Pemetaan kira-kira linear di bawah 1 kHz dan kira-kira logaritmis di atasnya. 80 bin mel mencakup 0–8 kHz adalah input ASR standar.

**Filterbank Mel.** Himpunan filter segitiga berjarak sama pada skala mel. Tiap filter adalah jumlah berbobot bin FFT bersebelahan. Mengalikan magnitudo STFT dengan matriks filterbank menghasilkan spektrogram mel dalam satu matmul.

**Spektrogram log-mel.** `log(mel_spec + 1e-10)`. Input Whisper. Input Parakeet. Input SeamlessM4T. Frontend audio universal 2026.

**MFCC.** Ambil spektrogram log-mel, terapkan DCT (tipe II), simpan 13 koefisien pertama. Mendekorelasi fitur dan memampatkan lebih jauh. Fitur dominan sampai sekitar 2015 ketika CNN/Transformer pada log-mel mentah menyusul. Masih dipakai dalam pengenalan pembicara (x-vector, ECAPA).

**Tradeoff resolusi.** FFT lebih besar = resolusi frekuensi lebih baik tetapi resolusi waktu lebih buruk. 25 ms / 10 ms adalah default audio-ML; 50 ms / 12.5 ms untuk musik; 5 ms / 2 ms untuk deteksi transien (pukulan drum, plosif).

```figure
spectrogram-window
```

## Bangun Sendiri

### Langkah 1: frame waveform

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

Klip 10-detik 16 kHz dengan `frame_len=400, hop=160` menghasilkan 998 frame.

### Langkah 2: window Hann

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

Kalikan element-wise sebelum FFT. Menghapus kebocoran spektral akibat pemotongan pada endpoint tak-nol.

### Langkah 3: magnitudo STFT

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

Produksi memakai `torch.stft` atau `librosa.stft` (didukung-FFT, tervektorisasi). Loop di sini pedagogis; ia berjalan pada klip pendek di `code/main.py`.

### Langkah 4: filterbank mel

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 mel mencakup 0–8 kHz dengan `n_fft=400` memberi matriks `(80, 201)`. Kalikan magnitudo STFT `(n_frames, 201)` dengan transpose-nya untuk mendapat spektrogram mel `(n_frames, 80)`.

### Langkah 5: log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

Alternatif umum: `librosa.power_to_db` (dB ternormalisasi-referensi), `10 * log10(power + eps)`. Whisper memakai rutin clip + normalize yang lebih rumit (lihat `log_mel_spectrogram` Whisper).

### Langkah 6: MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

Terapkan DCT ke tiap frame log-mel, simpan 13 koefisien pertama. Itulah matriks MFCC-mu. Koefisien pertama biasanya dibuang (ia meng-encode energi keseluruhan).

## Gunakan

Stack 2026:

| Task | Features |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 log-mels, 10 ms hop, 25 ms window |
| TTS acoustic model (VITS, F5-TTS, Kokoro) | 80 mels, 5–12 ms hop for fine temporal control |
| Audio classification (AST, PANNs, BEATs) | 128 log-mels, 10 ms hop |
| Speaker embedding (ECAPA-TDNN, WavLM) | 80 log-mels or raw-waveform SSL |
| Music (MusicGen, Stable Audio 2) | EnCodec discrete tokens (not mels) |
| Keyword spotting | 40 MFCCs for tiny devices |

Aturan praktis: **jika kamu tidak mengerjakan musik, mulai dengan 80 log-mel.** Beban bukti ada pada penyimpangan apa pun.

## Jebakan yang masih lolos di 2026

- **Mismatch jumlah mel.** Training dengan 80 mel, inference dengan 128 mel. Kegagalan diam. Log shape fitur di kedua ujung.
- **Mismatch sample-rate di hulu.** Mel yang dihitung pada 22.05 kHz tampak berbeda dari 16 kHz. Perbaiki SR *sebelum* featurisasi.
- **dB vs log.** Whisper mengharapkan log-mel, bukan dB-mel. Sebagian pipeline HF mendeteksi otomatis; kode kustommu tidak.
- **Drift normalisasi.** Normalisasi per-ujaran saat training, normalisasi global saat inference. Bug produksi yang melipatgandakan WER.
- **Kebocoran dari padding.** Zero-padding ujung klip menghasilkan spektrum datar pada frame ekor. Pad simetris atau replikasi.

## Rilis

Simpan sebagai `outputs/skill-feature-extractor.md`. Skill ini memilih tipe fitur, jumlah mel, frame/hop, dan normalisasi untuk target model tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ia mensintesis chirp (frekuensi disapu 200 → 4000 Hz) dan mencetak bin mel argmax per frame. Plot (opsional) dan konfirmasi cocok dengan sapuan.
2. **Sedang.** Jalankan ulang dengan `n_mels` dalam `{40, 80, 128}` dan `frame_len` dalam `{200, 400, 800}`. Ukur bandwidth puncak-tajam lintas sumbu waktu. Kombinasi mana yang me-resolve chirp terbaik?
3. **Sulit.** Implementasikan `power_to_db` dan bandingkan akurasi ASR pengklasifikasi CNN mungil pada AudioMNIST memakai (a) log-mel mentah, (b) dB-mel dengan `ref=max`, (c) MFCC-13 + delta + delta-delta. Laporkan akurasi top-1.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Frame | A slice | 25 ms chunk of waveform fed to one FFT. |
| Hop | Stride | Samples between consecutive frames; 10 ms is ASR default. |
| Window | Hann/Hamming thing | Point-wise multiplier that tapers the frame edges to zero. |
| STFT | Spectrogram generator | Framed + windowed FFT; yields time × frequency matrix. |
| Mel | Warped frequency | Log-perception scale; `m = 2595·log10(1 + f/700)`. |
| Filterbank | The matrix | Triangular filters that project STFT onto mel bins. |
| Log-mel | Whisper's input | `log(mel_spec + eps)`; standardized in 2026. |
| MFCC | Old-school feature | DCT of log-mel; 13 coeffs, decorrelated. |

## Bacaan Lanjutan

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — the MFCC paper.
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — the original mel scale.
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — read the reference implementation.
- [librosa feature extraction docs](https://librosa.org/doc/latest/api/feature.html) — reference for `mfcc`, `melspectrogram`, and hop/window.
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — production-scale pipeline for Parakeet + Canary models.
