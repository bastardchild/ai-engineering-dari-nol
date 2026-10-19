# Model Bahasa-Audio — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> Model bahasa-audio 2026 bernalar di atas bicara + suara lingkungan + musik. Qwen2.5-Omni-7B menyetarakan GPT-4o Audio di MMAU-Pro. Audio Flamingo Next mengalahkan Gemini 2.5 Pro di LongAudioBench. Jarak antara terbuka dan tertutup pada dasarnya sudah tertutup — kecuali pada tugas multi-audio, di mana semua orang nyaris acak.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 minutes

## Masalahnya

Kamu punya audio 5 detik: anjing menggonggong, seseorang berteriak "berhenti!", lalu hening. Pertanyaan berguna membentang di banyak sumbu:

- **Transkripsi.** "Apa yang diucapkan?" — wilayah ASR.
- **Penalaran semantik.** "Apakah orang itu dalam bahaya?" — menuntut pemahaman gabungan atas gonggongan + teriakan + keheningan.
- **Penalaran musik.** "Alat musik apa yang memainkan melodi itu?"
- **Retrieval audio panjang.** "Di mana dalam kuliah 90 menit ini instruktur menjelaskan gradient descent?"

Satu model yang menjawab semuanya dengan satu prompt adalah **model bahasa-audio** (LALM / ALM). Terpisah dari ASR murni: LALM menghasilkan jawaban bahasa alami bentuk-bebas, bukan sekadar transkrip.

## Konsepnya

![Model bahasa-audio: encoder audio + projector + decoder LLM](../assets/alm-architecture.svg)

### Templat tiga komponen

Setiap LALM 2026 punya kerangka yang sama:

1. **Encoder audio.** Whisper encoder · BEATs · CLAP · WavLM · atau encoder kustom per model.
2. **Projector.** Linear atau MLP yang menjembatani fitur encoder-audio ke ruang embedding token milik LLM.
3. **LLM.** Decoder berbasis Llama / Qwen / Gemma. Menerima token teks + audio berselang-seling; menghasilkan teks.

Training:

- **Tahap 1.** Bekukan encoder + LLM; latih hanya projector pada data ASR / captioning.
- **Tahap 2.** Fine-tuning penuh / LoRA pada tugas audio instruction-following (QA, penalaran, pemahaman musik).
- **Tahap 3 (opsional).** Voice-in / voice-out menambahkan decoder bicara. Qwen2.5-Omni dan AF3-Chat melakukan ini.

### Peta model 2026

| Model | Backbone | Audio encoder | Output modality | Access |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### Realitas benchmark (2026)

**MMAU-Pro.** 1800 pasang QA mencakup bicara / suara / musik / campuran. Subset multi-audio disertakan.

| Model | Overall | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA on LongAudioBench | — | — | — | — |

**Kolom multi-audio menghakimi semuanya.** Peluang acak pada pilihan ganda 4 opsi = 25%; kebanyakan model skornya di sekitar situ. LALM masih kesulitan membandingkan dua cuplikan.

### Di mana LALM berguna pada 2026

- **Audit kepatuhan rekaman pusat panggilan.** "Apakah agen menyebutkan pengungkapan yang diwajibkan?"
- **Aksesibilitas.** Mendeskripsikan peristiwa suara untuk pengguna tuli (bukan sekadar transkripsi).
- **Moderasi konten.** Mendeteksi bahasa kekerasan + nada mengancam + konteks latar.
- **Chaptering podcast / rapat.** Ringkasan semantik, bukan sekadar giliran pembicara.
- **Analisis katalog musik.** "Temukan semua trek dengan perpindahan kunci bagian-B."

### Di mana mereka belum berguna

- Teori musik halus (di bawah level akor).
- Penalaran atribusi-pembicara atas percakapan panjang (merosot setelah 10 menit).
- Perbandingan multi-audio (22-26% nyaris di atas acak).
- Penalaran streaming real-time (kebanyakan adalah inferensi batch offline).

```figure
v4-alm-tokens
```

## Bangun Sendiri

### Langkah 1: query Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Langkah 2: pola projector

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

Sesederhana itu. Projector biasanya 1-3 layer linear. Melatihnya pada pasangan ASR (audio → transkrip) adalah tugas pretext Tahap-1.

### Langkah 3: benchmark MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("gamma-lab-umd/MMAU-Pro", split="test")
mcq = mmau.filter(lambda item: len(item["choices"] or []) > 1)

correct = 0
for item in mcq:
    answer = call_model(item["audio_path"], item["question"], item["choices"])
    if answer == item["answer"]:
        correct += 1
print(f"Accuracy: {correct / len(mcq):.3f}")
```

`audio_path` menunjuk ke `data.zip` di repo dataset (sekitar 47 GB), jadi unduh dan buka zip sebelum melakukan skoring. Loop exact-match ini adalah sanity check, bukan skorer benchmark, sehingga angkanya tidak sebanding dengan hasil MMAU-Pro yang dipublikasikan. Evaluator resmi mencocokkan jawaban pilihan ganda lewat kemiripan embedding (NV-Embed-v2), menilai jawaban open-ended dengan LLM judge, dan memeriksa jawaban instruction-following dengan aturan regex: tulis prediksi ke kolom `model_output` dan jalankan `evaluate_mmau_pro_comprehensive.py` dari [repo MMAU-Pro](https://github.com/sonalkum/MMAUPro). Laporkan setiap `category` (speech, sound, music, multi, dan sisanya) secara terpisah. Angka agregat menyembunyikan di mana model gagal.

## Gunakan

| Task | 2026 pick |
|------|-----------|
| Free-form audio QA (open) | Qwen2.5-Omni-7B |
| Best open on long audio | Audio Flamingo Next |
| Best closed | Gemini 2.5 Pro |
| Voice-in / voice-out agent | Qwen2.5-Omni or GPT-4o Audio |
| Music reasoning | Audio Flamingo 3 or 2 (music-specialized AF-CLAP) |
| Call-center audit | Gemini 2.5 Pro via API, with RAG over your policy docs |

## Jebakan

- **Kepercayaan berlebih pada multi-audio.** Jika tugasmu butuh "cuplikan mana yang punya X," performa setara peluang-acak adalah nyata.
- **Degradasi audio panjang.** Lewat 10 menit, atribusi pembicara kebanyakan model rusak. Lakukan diarization dulu (Pelajaran 6), lalu rangkum.
- **Halusinasi pada keheningan.** Masalah gaya Whisper yang sama diwarisi LALM yang memakai encoder Whisper. Pasang gerbang VAD.
- **Cherry-picking benchmark.** Postingan blog vendor menyoroti kategori kasus terbaik. Jalankan sendiri subset multi-audio MMAU-Pro.

## Rilis

Simpan sebagai `outputs/skill-alm-picker.md`. Pilih LALM + subset benchmark + modalitas-output (teks vs bicara) untuk tugas pemahaman-audio tertentu.

## Latihan

1. **Mudah.** Jalankan `code/main.py` untuk melihat pola projector mainan + routing LALM palsu dari (audio-embedding, token-teks) → token output.
2. **Sedang.** Skorkan Qwen2.5-Omni-7B pada 100 butir bicara MMAU-Pro. Bandingkan dengan angka yang dilaporkan makalahnya.
3. **Sulit.** Bangun baseline audio-captioning minimal: encoder BEATs + projector 2-layer + Llama-3.2-1B yang dibekukan. Fine-tune hanya projector-nya di AudioCaps. Bandingkan dengan SALMONN di Clotho-AQA.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder. |
| Projector | Adapter | Small MLP mapping audio features into LLM embedding space. |
| MMAU | The benchmark | 10k audio-QA pairs across speech, sound, music. |
| MMAU-Pro | Harder MMAU | 1800 multi-audio / reasoning-heavy questions. |
| LongAudioBench | Long-form eval | Multi-minute clips with semantic queries. |
| Voice-in / voice-out | Speech-native | Model ingests speech and emits speech without text detour. |

## Bacaan Lanjutan

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — reference architecture.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — speech-in-speech-out.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — the open long-audio leader.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — dual-encoder pioneer.
- [MMAU-Pro leaderboard](https://sonalkum.github.io/mmau-pro/) — live 2026 rankings.
