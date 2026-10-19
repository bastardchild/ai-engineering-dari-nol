# Neural Audio Codec — EnCodec, SNAC, Mimi, DAC dan Pemisahan Semantik-Akustik

> Generasi audio 2026 hampir semuanya berbasis token. EnCodec, SNAC, Mimi, dan DAC mengubah gelombang kontinu menjadi sekuens diskrit yang bisa diprediksi transformer. Pemisahan token semantik-vs-akustik — codebook pertama sebagai semantik, sisanya sebagai akustik — adalah pergeseran arsitektur paling penting sejak Transformer untuk audio.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## Masalahnya

Model bahasa bekerja pada token diskrit. Audio itu kontinu. Jika kamu ingin model bergaya LLM untuk suara / musik — MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus — kamu lebih dulu butuh **neural audio codec**: encoder terpelajari yang mendiskretkan audio ke dalam vocabulary token kecil, dan decoder pasangan yang merekonstruksi gelombang.

Dua keluarga telah muncul:

1. **Codec reconstruction-first** — EnCodec, DAC. Mengoptimalkan kualitas audio perseptual. Token bersifat "akustik" — mereka menangkap semuanya termasuk identitas pembicara, timbre, dan derau latar.
2. **Codec semantic-first** — Mimi (Kyutai), SpeechTokenizer. Memaksa codebook pertama untuk mengkodekan konten linguistik / fonetik (sering dengan distilasi dari WavLM). Codebook berikutnya adalah detail akustik.

Wawasan 2024-2026: **codec reconstruction murni memberimu suara buram saat kamu mencoba menghasilkan dari teks.** LLM di atas token codec harus mempelajari struktur bahasa DAN struktur akustik dalam codebook yang sama, dan itu tidak skalabel. Memisahkannya — codebook semantik 0, codebook akustik 1-N — adalah yang membuat Moshi dan Sesame CSM bekerja.

## Konsepnya

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### Trik inti: Residual Vector Quantization (RVQ)

Alih-alih satu codebook besar (yang butuh jutaan kode untuk kualitas baik), semua audio codec modern memakai **RVQ**: kaskade dari codebook kecil. Codebook pertama mengkuantisasi keluaran encoder; yang kedua mengkuantisasi residual; dan seterusnya. Setiap codebook berisi 1024 kode. 8 codebook = vocabulary efektif 1024^8 = 10^24.

Saat inference, menjumlahkan semua kode terpilih per frame untuk merekonstruksi.

### Empat codec yang penting di 2026

**EnCodec (Meta, 2022).** Baseline-nya. Encoder-decoder di atas gelombang, bottleneck RVQ. 24 kHz, 32 codebook mungkin, default 4 codebook @ 1.5 kbps. Memakai arsitektur `1D conv + transformer + 1D conv`. Dipakai oleh MusicGen.

**DAC (Descript, 2023).** RVQ dengan codebook ber-normalisasi L2, fungsi aktivasi periodik, loss yang diperbaiki. Fidelsitas rekonstruksi tertinggi di antara codec open mana pun — kadang tak terbedakan dari suara asli dengan 12 codebook. 44.1 kHz full-band.

**SNAC (Hubert Siuzdak, 2024).** RVQ multi-skala — codebook kasar beroperasi pada frame rate lebih rendah daripada yang halus. Secara efektif memodelkan audio secara hierarkis: "sketsa" kasar di ~12 Hz plus detail di 50 Hz. Dipakai oleh Orpheus-3B karena struktur hierarkisnya terpetakan baik ke generasi berbasis LM.

**Mimi (Kyutai, 2024).** Pembeda permainan di 2026. Frame rate 12.5 Hz (sangat rendah), 8 codebook @ 4.4 kbps. Codebook 0 **didistilasi dari WavLM** — dilatih untuk memprediksi fitur konten-suarà WavLM. Codebook 1-7 adalah residual akustik. Pemisahan ini yang menggerakkan Moshi (Pelajaran 15) dan Sesame CSM.

### Frame rate penting untuk language modeling

Frame rate lebih rendah = sekuens lebih pendek = LM lebih cepat.

| Codec | Frame rate | 1 s = N frames | Good for |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | music, general audio |
| DAC-44.1k | 86 Hz | 86 | high-fidelity music |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM efficient |
| Mimi | 12.5 Hz | 12.5 | streaming speech |

Pada 12.5 Hz, satu ucapan 10 detik hanya 125 frame codec — transformer bisa dengan mudah memprediksinya.

### Token semantik vs akustik

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Token semantik (codebook 0 di Mimi).** Mengkodekan apa yang dikatakan — fonem, kata, konten. Didistilasi dari WavLM melalui loss prediksi bantu.
- **Token akustik (codebook 1-7).** Mengkodekan timbre, identitas pembicara, prosodi, derau latar, detail halus.

Sebuah **AR LM** memprediksi token semantik lebih dulu (dikondisikan teks), lalu memprediksi token akustik (dikondisikan semantik + referensi pembicara). Faktorisasi inilah alasan **TTS** modern bisa melakukan zero-shot voice cloning: model semantik menangani konten; model akustik menangani timbre.

### Kualitas rekonstruksi 2026 (bits per sec, bitrate lebih rendah itu lebih baik)

| Codec | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Codec tradisional seperti Opus masih menang per bit dalam kualitas perseptual. Codec neural menang pada **token diskrit** (yang tidak dihasilkan Opus) dan **kualitas model generatif** (apa yang bisa dilakukan LM dengan token tersebut).

```figure
rvq-codec-cascade
```

## Bangun Sendiri

### Langkah 1: encode dengan EnCodec

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

`n_codebooks=8` pada 6 kbps. Setiap kode adalah 0-1023 (10-bit).

### Langkah 2: decode dan ukur rekonstruksi

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Langkah 3: pemisahan semantik-akustik (gaya Mimi)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Codebook semantik 0 selaras dengan WavLM. Kamu bisa melatih transformer teks-ke-semantik — vocabulary-nya jauh lebih kecil daripada langsung ke-audio. Lalu decoder akustik-ke-gelombang terpisah mengkondisikan pada referensi pembicara.

### Langkah 4: mengapa AR LM di atas token codec bekerja

Untuk klip suara 10 detik pada 12.5 Hz Mimi × 8 codebook:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 token adalah konteks yang sepele bagi transformer. Transformer berparameter 256M bisa menghasilkan 10 detik suara dalam hitungan milidetik di GPU modern.

## Gunakan

Petakan masalah → codec:

| Task | Codec |
|------|-------|
| General music generation | EnCodec-24k |
| Highest-fidelity reconstruction | DAC-44.1k |
| AR LM over speech (TTS) | SNAC or Mimi |
| Streaming full-duplex speech | Mimi (12.5 Hz) |
| Sound-effect library with text | EnCodec + T5 condition |
| Fine-grained audio editing | DAC + inpainting |

Aturan praktis: **jika kamu membangun model generatif, mulailah dengan Mimi atau SNAC. Jika kamu membangun pipeline kompresi, pakai Opus.**

## Jebakan

- **Terlalu banyak codebook.** Menambah codebook menaikkan fidelsitas secara linear tapi panjang sekuens LM juga linear. Berhenti di 8-12.
- **Ketidakcocokan frame rate.** Melatih LM pada Mimi 12.5 Hz lalu fine-tuning pada EnCodec 50 Hz gagal tanpa suara.
- **Menganggap semua codebook setara.** Di Mimi, codebook 0 membawa konten; kehilangannya menghancurkan intelligibility. Kehilangan codebook 7 nyaris tak terasa.
- **Memakai kualitas rekonstruksi sebagai satu-satunya metrik.** Sebuah codec bisa punya rekonstruksi hebat tapi tak berguna untuk generasi berbasis LM jika struktur semantiknya buruk.

## Rilis

Simpan sebagai `outputs/skill-codec-picker.md`. Pilih codec untuk tugas generatif atau kompresi tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ia mengimplementasikan kuantizer skalar + residual mainan dan mengukur error rekonstruksi saat kamu menambah codebook.
2. **Sedang.** Install `encodec` dan bandingkan 1, 4, 8, 32 codebook pada klip suara held-out. Plot PESQ atau MSE terhadap bitrate.
3. **Sulit.** Muat Mimi. Encode sebuah klip. Ganti codebook 0 dengan bilangan acak; decode. Lalu ganti codebook 7 dengan cara yang sama. Bandingkan dua kerusakan tersebut — kerusakan codebook 0 seharusnya menghancurkan intelligibility; kerusakan codebook 7 nyaris tak mengubah apa pun.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| RVQ | Residual quantization | Cascade of small codebooks; each quantizes the previous residual. |
| Frame rate | Codec speed | How many token-frames per second. Lower = faster LM. |
| Semantic codebook | Codebook 0 (Mimi) | Codebook distilled from SSL features; encodes content. |
| Acoustic codebooks | Everything else | Timbre, prosody, noise, fine detail. |
| PESQ / ViSQOL | Perceptual quality | Objective metrics correlating with MOS. |
| EnCodec | Meta codec | The RVQ baseline; used by MusicGen. |
| Mimi | Kyutai codec | 12.5 Hz frame rate; semantic-acoustic split; powers Moshi. |

## Bacaan Lanjutan

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — the RVQ baseline.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — highest-fidelity open.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — multi-scale RVQ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — semantic-acoustic split, WavLM distillation.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — the two-stage semantic/acoustic paradigm.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — the original streamable RVQ codec.
