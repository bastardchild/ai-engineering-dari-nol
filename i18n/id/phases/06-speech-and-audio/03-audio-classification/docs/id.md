# Klasifikasi Audio — Dari k-NN pada MFCC hingga AST dan BEATs

> Semuanya dari "anjing menggonggong vs sirene" hingga "ini bahasa apa" adalah klasifikasi audio. Fiturnya adalah mels. Arsitekturnya berpindah tiap dekade. Evaluasinya tetap AUC, F1, dan recall per kelas.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 minutes

## Masalahnya

Kamu dapat klip 10 detik. Kamu ingin tahu: "ini apa?" Suara kota (sirene, bor, anjing), perintah suara (ya/tidak/berhenti), identifikasi bahasa (en/es/ar), emosi pembicara (marah/netral), atau suara lingkungan (indoor/outdoor, keramaian). Semua ini adalah *klasifikasi audio*, dan pada 2026 arsitektur baseline-nya sudah matang: log-mel → CNN atau Transformer → softmax.

Inti kesulitannya bukan jaringannya. Melainkan data. **Dataset** audio punya ketidakseimbangan kelas yang kejam, pergeseran domain yang kuat (bersih vs berisik), dan noise label (siapa yang memutuskan "keramaian kota" vs "suara restoran"?). 80% masalahnya adalah kurasi, augmentasi, dan evaluasi, bukan bertukar CNN dengan Transformer.

## Konsepnya

![Tangga klasifikasi audio: k-NN pada MFCC hingga AST hingga BEATs](../assets/audio-classification.svg)

**k-NN pada MFCC (baseline 1990-an).** Ratakan MFCC per klip, hitung kemiripan cosine ke bank berlabel, kembalikan voting mayoritas dari top K. Mengejutkan kuat pada dataset kecil yang bersih (Speech Commands, ESC-50). Berjalan tanpa GPU.

**2D CNN pada log-mels (2015-2019).** Perlakukan log-mel berbentuk `(T, n_mels)` sebagai gambar. Terapkan ResNet-18 atau gaya VGG. Global mean pool pada sumbu waktu. Softmax di atas kelas. Masih menjadi baseline di kebanyakan kompetisi kaggle 2026.

**Audio Spectrogram Transformer, AST (2021-2024).** Patchify log-mel (misal patch 16×16), tambahkan position embedding, masukkan ke **ViT**. State of the art pada AudioSet (mAP 0.485) untuk supervised learning.

**BEATs dan WavLM-base (2024-2026).** Pretraining self-supervised pada jutaan jam. Fine-tune pada tugasmu dengan 1-10% data supervised yang seharusnya kamu butuhkan. Pada 2026 ini adalah titik awal default untuk audio non-pertuturan. BEATs-iter3 mengalahkan AST dengan selisih 1-2 mAP pada AudioSet sambil memakai 1/4 komputasi.

**Encoder Whisper sebagai backbone beku (2024).** Ambil encoder **Whisper**, buang decoder, pasang classifier linear. Hampir-SOTA pada identifikasi bahasa dan klasifikasi event sederhana dengan nol augmentasi audio. Baseline "makan siang gratis".

### Ketidakseimbangan kelas adalah tantangan sebenarnya

ESC-50: 50 kelas, 40 klip masing-masing — seimbang, mudah. UrbanSound8K: 10 kelas, tidak seimbang 10:1. AudioSet: 632 kelas dengan long tail 100.000:1. Teknik yang bekerja:

- Sampling seimbang selama training (bukan saat evaluasi).
- Mixup: interpolasi linear dua klip (dan labelnya) sebagai augmentasi.
- SpecAugment: mask band waktu dan frekuensi acak. Sederhana; krusial.

### Evaluasi

- Multiclass eksklusif (Speech Commands): akurasi top-1, akurasi top-5.
- Multiclass multi-label (AudioSet, gaya UrbanSound): mean average precision (mAP).
- Sangat tidak seimbang: recall per kelas + F1 makro.

Angka 2026 yang perlu kamu tahu:

| Benchmark | Baseline | SOTA 2026 | Source |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

```figure
mfcc-pipeline
```

## Bangun Sendiri

### Langkah 1: featurisasi

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Langkah 2: ringkasan panjang-tetap

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Sederhana tapi kuat: mean + varians sepanjang waktu memberi embedding tetap 26-dim untuk MFCC 13-koefisien. Berjalan instan. Mengalahkan baseline NN state-of-the-art pada ESC-50 sesingkat 2017.

### Langkah 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Langkah 4: naik ke CNN pada log-mels

Di PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3 juta parameter. Training ~10 menit di ESC-50 dengan satu RTX 4090. Akurasi 80%+.

### Langkah 5: fine-tune transformer audio terlatih-pra (AST ditampilkan)

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

Contoh ini melakukan fine-tune AST dari Hub. BEATs, pilihan default 2026, tidak ada di Hugging Face Hub: unduh checkpoint dari [rilis BEATs di microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beats) dan muat dengan kelas `BEATs` dan `BEATsConfig` dari repo tersebut; loop fine-tuning-nya tetap berbentuk sama.

## Gunakan

Stack 2026:

| Situation | Start with |
|-----------|-----------|
| Tiny dataset (<1000 clips) | k-NN on MFCC means (your baseline) + audio augmentation |
| Medium dataset (1K–100K) | BEATs or AST fine-tune |
| Large dataset (>100K) | Train from scratch or fine-tune Whisper-encoder |
| Real-time, edge | 40-MFCC CNN, quantized to int8 (KWS-style) |
| Multi-label (AudioSet) | BEATs-iter3 with BCE loss + mixup + SpecAugment |
| Language ID | MMS-LID, SpeechBrain VoxLingua107 baseline |

Aturan keputusan: **mulai dengan backbone beku, bukan model baru**. Fine-tune kepala BEATs memberimu 95% SOTA dalam hitungan jam, bukan minggu.

## Rilis

Simpan sebagai `outputs/skill-classifier-designer.md`. Pilih arsitektur, augmentasi, strategi keseimbangan kelas, dan metrik eval untuk tugas klasifikasi audio tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Ia melatih baseline k-NN MFCC pada dataset sintetis 4 kelas (nada murni dengan pitch berbeda). Laporkan confusion matrix.
2. **Sedang.** Ganti `summarize` dengan [mean, var, skew, kurtosis]. Apakah pooling 4-moment mengalahkan mean+var pada dataset sintetis yang sama?
3. **Sulit.** Dengan `torchaudio`, latih 2D CNN pada ESC-50 fold 1. Laporkan akurasi cross-validation 5-fold. Tambahkan SpecAugment (time mask = 20, freq mask = 10) dan laporkan delta-nya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| AudioSet | The ImageNet of audio | Google's 2M-clip, 632-class weakly-labeled YouTube dataset. |
| ESC-50 | Small classification benchmark | 50 classes × 40 clips of environmental sounds. |
| AST | Audio Spectrogram Transformer | ViT on log-mel patches; 2021 SOTA. |
| BEATs | Self-supervised audio | Microsoft model, iter3 leads AudioSet as of 2026. |
| Mixup | Pair augmentation | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | Mask-based augmentation | Zero-out random time and frequency bands of the spectrogram. |
| mAP | Main multi-label metric | Mean average precision across classes and thresholds. |

## Bacaan Lanjutan

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — arsitektur rujukan dari 2021–2024.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — default 2024+.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — augmentasi audio yang dominan.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — benchmark 50 kelas yang masih hidup.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — taksonomi YouTube 632 kelas; masih standar emas.
