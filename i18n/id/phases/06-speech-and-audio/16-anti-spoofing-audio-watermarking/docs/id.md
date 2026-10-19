# Anti-Spoofing Suara & Watermarking Audio — ASVspoof 5, AudioSeal, WaveVerify

> Voice cloning tiba lebih cepat daripada pertahanannya. Sistem suara produksi 2026 butuh dua hal: sebuah detektor (AASIST, RawNet2) yang mengklasifikasikan bicara asli vs palsu, dan sebuah watermark (AudioSeal) yang bertahan terhadap kompresi dan penyuntingan. Kirim keduanya atau jangan kirim voice cloning.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## Masalahnya

Tiga pertahanan terkait:

1. **Anti-spoofing / deteksi deepfake.** Diberi sebuah klip audio, apakah itu sintetis atau asli? Benchmark ASVspoof (ASVspoof 2019 → 2021 → 5) adalah standar emasnya.
2. **Watermarking audio.** Tanamkan sinyal tak-terdeteksi di dalam audio hasil generasi yang nanti bisa diekstrak oleh sebuah detektor. AudioSeal (Meta) dan WavMark adalah pilihan open-source-nya.
3. **Provenance terautentikasi.** Penandaan kriptografis atas file audio + metadata. C2PA / Content Authenticity Initiative.

Deteksi menangani lawan yang tidak kooperatif. Watermarking menangani kepatuhan — audio buatan AI harus bisa diidentifikasi sebagai demikian. Keduanya wajib ada pada 2026.

## Konsepnya

![Anti-spoofing vs watermarking vs provenance — tiga lapis pertahanan](../assets/spoofing-watermark.svg)

### ASVspoof 5 — benchmark 2024-2025

Perubahan terbesar dari edisi sebelumnya:

- **Data crowdsourced** (bukan bersih studio) — kondisi realistis.
- **~2000 pembicara** (vs ~100 sebelumnya).
- **32 algoritma serangan.** TTS + voice conversion + gangguan adversarial.
- **Dua track.** Countermeasure (CM) deteksi mandiri; Spoofing-robust ASV (SASV) untuk sistem biometrik.

State-of-the-art pada ASVspoof 5: EER ~7.23%. Pada ASVspoof 2019 LA yang lebih lama: EER 0.42%. Deployment dunia-nyata: harapkan EER 5-10% pada klip in-the-wild.

### AASIST dan RawNet2 — keluarga model deteksi

**AASIST** (2021, diperbarui hingga 2026). Graph-attention pada fitur spektral. SOTA saat ini pada tugas countermeasure ASVspoof 5.

**RawNet2.** Front-end konvolusi di atas gelombang mentah + tulang punggung TDNN. Baseline lebih sederhana; tetap kompetitif dengan fine-tuning.

**NeXt-TDNN + fitur SSL.** Varian 2025: ECAPA-style + fitur WavLM + focal loss. Mencapai EER 0.42% pada ASVspoof 2019 LA.

### AudioSeal — default watermark 2024

**AudioSeal** milik Meta (Jan 2024, v0.2 Des 2024). Rancangan kuncinya:

- **Terlokalisasi.** Mendeteksi watermark per-frame pada resolusi sampel 16 kHz (1/16000 s).
- **Generator + detektor dilatih bersama.** Generator belajar menanam sinyal tak-terdengar; detektor belajar menemukannya lewat augmentasi.
- **Robust.** Bertahan terhadap kompresi MP3 / AAC, EQ, speed-shift ±10%, campuran noise +10 dB SNR.
- **Cepat.** Detektor berjalan pada 485× realtime; 1000× lebih cepat dari WavMark.
- **Kapasitas.** Payload 16-bit (bisa mengkode ID model, timestamp generasi, user ID) yang bisa ditanam di setiap ucapan.

### WavMark

Baseline open-source sebelum-AudioSeal. Jaringan neural yang dapat diinvert, 32 bits/detik. Masalahnya:

- Sinkronisasi brute-force lambat.
- Bisa dihapus oleh noise Gaussian atau kompresi MP3.
- Tidak ramah real-time.

### WaveVerify (Juli 2025)

Menangani kelemahan AudioSeal — khususnya manipulasi temporal (reversal, kecepatan). Memakai generator berbasis FiLM + detektor Mixture-of-Experts. Kompetitif dengan AudioSeal pada serangan standar; menangani edit temporal.

### Celah yang dimanfaatkan para penyerang

Dari AudioMarkBench: "di bawah pitch shift, semua watermark menunjukkan Bit Recovery Accuracy di bawah 0.6, yang menandakan penghapusan nyaris-lengkap." **Pitch-shift adalah serangan universal.** Tidak ada watermark 2026 yang sepenuhnya robust terhadap modifikasi pitch agresif. Inilah mengapa kamu butuh deteksi (AASIST) di samping watermarking.

### C2PA / Content Authenticity Initiative

Bukan teknik ML — sebuah format manifest. File audio membawa metadata yang ditandai secara kriptografis tentang alat pembuatan, penulis, tanggal. Audobox / Seamless memakainya. Bagus untuk provenance; tidak berdaya jika aktor jahat meng-encode-ulang dan men-strip metadata.

```figure
v4-audio-watermark
```

## Bangun Sendiri

### Langkah 1: detektor fitur-spektral sederhana (mainan)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Bicara sintetis sering punya energi frekuensi-tinggi yang datar tak biasa. Detektor produksi memakai AASIST, bukan ini. Tetapi intuisinya tetap berlaku.

### Langkah 2: embed + deteksi AudioSeal

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### Langkah 3: evaluasi — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Langkah 4: integrasi produksi

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

Setiap generasi mengirim: (1) watermark, (2) manifest bertanda-tangan, (3) log audit yang patuh pada kebijakan retensi.

## Gunakan

| Use case | Defense |
|----------|---------|
| Shipping TTS / voice cloning | AudioSeal embed on every output (non-negotiable) |
| Biometric voice unlock | AASIST + ECAPA ensemble; liveness challenge |
| Call-center fraud detection | AASIST on 20% sample of incoming calls |
| Podcast authenticity | C2PA signing on upload, AudioSeal if AI-generated |
| Research / training detectors | ASVspoof 5 train/dev/eval sets |

## Jebakan

- **Watermark yang detektor-nya tak pernah dijalankan.** Sia-sia. Jalankan detektor di CI-mu.
- **Deteksi tanpa kalibrasi.** AASIST yang dilatih pada ASVspoof LA overfit; akurasi dunia-nyata turun. Kalibrasi pada domain-mu.
- **Celah pitch-shift.** Pitch shift agresif menghapus sebagian besar watermark. Sediakan fallback deteksi.
- **Strip-and-rehost metadata.** C2PA bisa dibobol dengan mudah lewat re-encode. Selalu tambahkan pertahanan kriptografis + perseptual (watermark) secara bersamaan.
- **Liveness sebagai deteksi.** Minta pengguna mengucapkan frasa acak. Mencegah serangan replay tetapi bukan cloning real-time.

## Rilis

Simpan sebagai `outputs/skill-spoof-defender.md`. Pilih model deteksi, watermark, manifest provenance, dan playbook operasional untuk sebuah deployment voice-gen.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Detektor mainan + embed/deteksi watermark mainan pada audio sintetis.
2. **Sedang.** Pasang `audioseal`, tanamkan payload 16-bit dalam output TTS, dekode-ulang. Rusakkan audio dengan noise dan ukur Bit Recovery Accuracy.
3. **Sulit.** Fine-tune RawNet2 atau AASIST pada ASVspoof 2019 LA. Ukur EER. Uji pada held-out set klip hasil F5-TTS — lihat bagaimana deteksi OOD merosot.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ASVspoof | The benchmark | Biennial challenge; 2024 = ASVspoof 5. |
| CM (countermeasure) | Detector | Classifier: real speech vs synthetic / converted. |
| SASV | Speaker verif + CM | Integrated biometric + spoof detection. |
| AudioSeal | Meta watermark | Localized, 16-bit payload, 485× faster than WavMark. |
| Bit Recovery Accuracy | Watermark survival | Fraction of payload bits recovered after attack. |
| C2PA | Provenance manifest | Cryptographic metadata about creation / authorship. |
| AASIST | Detector family | Graph-attention-based anti-spoofing SOTA. |

## Bacaan Lanjutan

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — the current benchmark.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — the watermark default.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — MoE detector for temporal attacks.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — the SOTA detection backbone.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — robustness evaluation.
- [C2PA specification](https://spec.c2pa.org/specifications/specifications/2.4/index.html) — provenance manifest format.
