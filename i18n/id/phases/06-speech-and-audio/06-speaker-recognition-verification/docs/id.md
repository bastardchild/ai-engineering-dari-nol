# Pengenalan & Verifikasi Pembicara

> ASR bertanya "apa yang mereka katakan?" Pengenalan pembicara bertanya "siapa yang mengatakannya?" Matematikanya terlihat sama — embedding plus cosine — tapi setiap keputusan produksi bertumpu pada satu angka EER.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 minutes

## Masalahnya

Seorang pengguna mengucapkan sebuah passphrase. Kamu ingin tahu: apakah ini orang yang dia klaim (*verification*, 1:1), atau justru orang pertama di bank enrollment-mu (*identification*, 1:N)? Atau tidak keduanya — apakah ini pembicara tak dikenal (*open-set*)?

Sebelum 2018: GMM-UBM + i-vectors. EER cukup masuk akal tapi rapuh terhadap pergeseran kanal (ponsel vs laptop) dan emosi. 2018–2022: x-vectors (backbone TDNN yang dilatih dengan angular margin). 2022+: embedding ECAPA-TDNN dan WavLM-large. Menjelang 2026, bidang ini didominasi tiga model dan satu metrik.

Metrik itu adalah **EER** — Equal Error Rate. Atur threshold keputusanmu sehingga False Accept Rate = False Reject Rate. Titik potongnya adalah EER. Dipakai di setiap paper, setiap leaderboard, setiap panggilan procurement.

## Konsepnya

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**Pipeline-nya.** Enrollment: rekam 5–30 detik pembicara target; hitung embedding berdimensi tetap (192-d untuk ECAPA-TDNN, 256-d untuk WavLM-large). Verification: ambil embedding dari ucapan uji; hitung kemiripan cosine; bandingkan dengan sebuah threshold.

**ECAPA-TDNN (2020, masih dominan di 2026).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network: blok konvolusi 1D dengan squeeze-excitation, attention pooling multi-head, diikuti layer linier ke 192-d. Dilatih pada VoxCeleb 1+2 (2.700 pembicara, 1,1 juta ucapan) dengan loss Additive Angular Margin (AAM-softmax).

**WavLM-SV (2022+).** Fine-tune backbone SSL WavLM-large yang sudah pretrain dengan loss AAM. Kualitas lebih tinggi tapi lebih lambat — 300+ MB vs 15 MB.

**x-vector (baseline).** TDNN + statistics pooling. Klasik; masih berguna di CPU / edge.

**AAM-softmax.** Softmax standar dengan margin tambahan `m` di ruang angular: `cos(θ + m)` untuk kelas yang benar. Memaksa pemisahan angular antar kelas. Tipikal `m=0.2`, scale `s=30`.

### Skoring

- **Cosine** antara embedding enrollment dan test. Keputusan berbasis threshold.
- **PLDA (Probabilistic LDA).** Proyeksikan embedding ke ruang laten tempat same-speaker vs different-speaker punya likelihood ratio tertutup (*closed-form*). Ditambahkan di atas cosine untuk penurunan EER +10–20%. Standar sebelum 2020; kini hanya dipakai pada setup closed-set.
- **Normalisasi skor.** `S-norm` atau `AS-norm`: normalkan setiap skor terhadap kohort mean dan std impostor. Penting untuk evaluasi lintas-domain.

### Angka yang perlu kamu ketahui (2026)

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Diarisasi

"Siapa bicara kapan" dalam klip multi-pembicara. Pipeline: VAD → segmentasi → embed tiap segmen → klasterisasi (agglomerative atau spektral) → haluskan batas. Stack modern: `pyannote.audio` 3.1, yang membundel segmentasi + embedding + klasterisasi pembicara di balik satu panggilan. DER SOTA 2026 di AMI sekitar 15% (turun dari 23% pada 2022).

```figure
sp-eer-crossover
```

## Bangun Sendiri

### Langkah 1: embedding mainan dari statistik MFCC

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

Bukan SOTA sama sekali — hanya untuk pengajaran. `code/main.py` memakai ini sebagai proof-of-concept pada data pembicara sintetis.

### Langkah 2: kemiripan cosine + threshold

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Langkah 3: EER dari pasangan kemiripan

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

Mengembalikan (eer, threshold_at_eer). Laporkan keduanya.

### Langkah 4: produksi dengan SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### Langkah 5: diarisasi dengan pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Gunakan

Stack 2026:

| Situation | Pick |
|-----------|------|
| Closed-set 1:1 verification, edge | ECAPA-TDNN + cosine threshold |
| Open-set verification, cloud | WavLM-SV + AS-norm |
| Diarization (meetings, podcasts) | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing (replay / deepfake detection) | AASIST or RawNet2 |
| Tiny embedded (KWS + enrollment) | Titanet-Small (NeMo) |

## Jebakan

- **Ketidakcocokan kanal.** Model yang dilatih pada VoxCeleb (video web) ≠ audio panggilan telepon. Selalu evaluasi pada kanal target.
- **Ucapan pendek.** EER memburuk tajam di bawah 3 detik audio uji.
- **Enrollment dengan noise.** Satu enrollment berisik meracuni anchor. Pakai ≥3 sampel bersih dan rata-ratakan.
- **Threshold tetap lintas kondisi.** Selalu tune threshold pada dev set held-out dari domain target.
- **Cosine pada embedding yang tak ternormalisasi.** Normalisasi L2 dulu; kalau tidak, magnitudo yang mendominasi.

## Rilis

Simpan sebagai `outputs/skill-speaker-verifier.md`. Pilih model, protokol enrollment, rencana tuning threshold, dan safeguard anti-fraud.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Membangun "pembicara" sintetis (profil nada berbeda), mendaftarkan, lalu menghitung EER pada daftar uji berisi 100 pasangan.
2. **Sedang.** Pakai SpeechBrain ECAPA pada 30 ucapan VoxCeleb1 (5 pembicara × 6 masing-masing). Hitung EER dengan cosine vs PLDA.
3. **Sulit.** Bangun pipeline lengkap enroll → diarize → verify dengan `pyannote.audio`. Evaluasi DER pada dev set AMI.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| EER | The headline metric | Threshold where False Accept = False Reject. |
| Verification | 1:1 | "Is this Alice?" |
| Identification | 1:N | "Who is speaking?" |
| Open-set | Unknown possible | Test set can contain unenrolled speakers. |
| Enrollment | Registering | Computing a speaker's reference embedding. |
| AAM-softmax | The loss | Softmax with additive angular margin; forces cluster separation. |
| PLDA | Classic scoring | Probabilistic LDA; likelihood-ratio scoring on top of embeddings. |
| DER | Diarization metric | Diarization Error Rate — miss + false alarm + confusion. |

## Bacaan Lanjutan

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — makalah deep-embedding klasiknya.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — arsitektur dominan 2020–2026.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — backbone SSL untuk SV dan diarisasi.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — stack produksi untuk diarisasi + embedding.
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — peringkat EER terkini lintas model.
