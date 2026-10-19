# Show-o dan Model Unified Discrete-Diffusion

> Transfusion mencampur representasi continuous dan diskrit. Show-o (Xie dkk., Agustus 2024) mengambil jalan sebaliknya: token teks memakai next-token prediction kausal, token gambar memakai masked discrete diffusion semangat MaskGIT. Keduanya duduk di dalam satu transformer dengan attention mask hibrida. Hasilnya menyatukan VQA, text-to-image, inpainting, dan generasi modalitas-campuran pada satu backbone, satu tokenizer per modalitas, satu formulasi loss (next-token diperluas ke masked prediction). Pelajaran ini membahas desain Show-o — mengapa masked discrete diffusion adalah generator gambar parallel ber-langkah-sedikit — dan membandingkannya dengan Transfusion dan Emu3.

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 minutes

## Tujuan Pembelajaran

- Menjelaskan masked discrete diffusion: schedule yang men-mask token secara seragam lalu meminta transformer memulihkannya.
- Membandingkan decoding gambar parallel (Show-o, MaskGIT) dengan decoding gambar autoregressive (Chameleon, Emu3) dari sisi kecepatan dan kualitas.
- Menyebut tiga tugas yang ditangani Show-o dalam satu checkpoint: T2I, VQA, image inpainting.
- Memilih schedule masking (cosine, linear, terpotong) dan bernalar tentang efeknya terhadap kualitas sampel.

## Masalahnya

Training dua-loss Transfusion bekerja tapi dinamikanya lebih rumit — loss diffusion continuous hidup pada skala numerik yang berbeda dari loss NTP diskrit. Menyeimbangkan bobot loss adalah pencarian hyperparameter. Arsitekturnya efektif tapi kompleks.

Jawaban Show-o: pertahankan kedua modalitas secara diskrit (seperti Chameleon), tapi hasilkan gambar secara parallel lewat masked discrete diffusion alih-alih secara sekuensial. Training objective menjadi satu masked-token-prediction yang menggeneralisasi next-token-prediction secara alami.

## Konsepnya

### Masked discrete diffusion (MaskGIT)

Trik MaskGIT asli Chang dkk. (2022) itu elegan. Mulai dari gambar yang sepenuhnya di-mask (setiap token adalah id khusus `<MASK>`). Pada setiap langkah, prediksi semua token yang di-mask secara parallel, lalu simpan prediksi top-K yang paling yakin dan mask ulang sisanya. Setelah ~8-16 iterasi, semua token terisi. Schedule berapa banyak token yang di-unmask per langkah di-tuning — schedule cosine bekerja baik.

Training sederhana: sampelkan rasio masking secara seragam dari [0, 1], terapkan pada VQ tokens gambar, latih transformer untuk memulihkan yang di-mask. Persis seperti yang BERT lakukan untuk teks, diskalakan ke generasi gambar.

### Show-o: satu transformer, mask hibrida

Show-o menaruh MaskGIT di dalam transformer causal-language-model. Attention mask-nya:

- Token teks: kausal (LLM standar).
- Token gambar: bidirectional penuh dalam blok gambar (sehingga token yang di-mask dapat melihat setiap token gambar lain saat prediksi).
- Text-to-image: teks menghadiri gambar sebelumnya, gambar menghadiri teks sebelumnya.

Training berselang-seling antara:
1. NTP standar pada sekuens teks.
2. Sampel T2I: teks → gambar dengan token gambar yang di-mask, loss masked-token-prediction.
3. Sampel VQA: gambar → teks dengan token teks yang di-mask (sebenarnya hanya NTP).

Loss terpadu adalah cross-entropy pada token `<MASK>`, yang mencakup baik NTP teks (hanya token terakhir yang "di-mask") maupun masked-diffusion gambar (subset acak yang di-mask).

### Sampling parallel

Show-o menghasilkan gambar dalam ~16 langkah alih-alih ~1000 (autoregressive per token) atau ~20 (diffusion). Pada setiap langkah, prediksi semua token yang di-mask secara parallel; commit yang top-K yakin; ulangi.

Bandingkan:
- Chameleon / Emu3 (autoregressive atas token): N_tokens forward pass, biasanya 1024-4096 per gambar.
- Transfusion (diffusion continuous): ~20 langkah, masing-masing satu full transformer pass.
- Show-o (masked discrete diffusion): ~16 langkah, masing-masing satu full transformer pass.

Show-o lebih cepat daripada Chameleon pada model skala serupa, kira-kira cocok dengan jumlah langkah Transfusion dengan biaya per-langkah lebih rendah (logits vocab diskrit vs loss MSE continuous).

### Tugas dalam satu checkpoint

Show-o mendukung empat tugas saat inference, dipilih oleh format prompt:

- Generasi teks: output teks autoregressive standar.
- VQA: gambar masuk, teks keluar.
- T2I: teks masuk, gambar keluar via masked discrete diffusion.
- Inpainting: gambar dengan sebagian token di-mask, isi kembali.

Kemampuan inpainting datang gratis dari training masked-prediction. Mask sebuah region pada grid VQ-token, berikan sisanya plus prompt teks, prediksi token yang di-mask.

### Schedule masking

Schedule berapa banyak token yang di-unmask per langkah membentuk kualitas. Show-o merekomendasikan cosine:

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

Pada langkah 0, semua token di-mask (rasio 1.0). Pada langkah T, tidak ada yang di-mask. Cosine memusatkan massa pada rasio rentang-menengah di mana prediksi paling informatif. Schedule linear juga bekerja tapi berplatok lebih cepat.

### Show-o2

Show-o2 (lanjutan 2025, arXiv 2506.15564) menskalakan Show-o: basis LLM lebih besar, tokenizer lebih baik, schedule masking lebih diperbaiki. Pola arsitektur yang sama.

### Di mana Show-o berada

Dalam taksonomi 2026:

- Token diskrit + NTP: Chameleon, Emu3. Sederhana tapi inference lambat.
- Token diskrit + masked diffusion: Show-o, MaskGIT, LlamaGen, Muse. Sampling parallel, masih lossy oleh tokenizer.
- Continuous + diffusion: Transfusion, MMDiT, DiT. Kualitas tertinggi, training lebih kompleks.
- Continuous + flow matching dalam VLM: JanusFlow, InternVL-U. Paling baru.

Pilih berdasarkan tugas: Show-o saat kamu ingin T2I + inpainting + VQA dalam satu model open dengan kecepatan wajar; Transfusion saat kualitas sangat penting dan kamu mampu membiayai plumbing dua-loss.

```figure
masked-diffusion-unmask
```

## Gunakan

`code/main.py` mensimulasikan sampling Show-o:

- Grid mainan 16 token VQ.
- "Transformer" tiruan yang memprediksi logits berdasarkan prompt dan token yang sedang tidak di-mask.
- Sampling masked parallel selama 8 langkah dengan schedule cosine.
- Mencetak status intermediate (evolusi pola mask) dan token akhir.

Jalankan, dan amati mask larut langkah demi langkah.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-unified-gen-model-picker.md`. Diberi produk yang membutuhkan baik pemahaman (VQA, captioning) maupun generasi (T2I, inpainting) dengan konstrain bobot-terbuka, ia memilih antara keluarga Show-o, keluarga Transfusion/MMDiT, dan keluarga Emu3 / Chameleon dengan trade-off konkret.

## Latihan

1. Masked discrete diffusion menghasilkan sampel dalam ~16 langkah. Mengapa tidak 1? Apa yang rusak jika kamu meng-unmask semuanya pada langkah 0?

2. Inpainting gratis dengan masked diffusion. Usulkan satu use case produk (nyata atau hipotetis) di mana inpainting Show-o mengalahkan model spesialis.

3. Schedule cosine vs schedule linear: telusuri jumlah token yang di-unmask per langkah untuk T=8. Mana yang lebih seimbang?

4. Gambar Show-o 512x512 berjumlah 1024 token. Dengan vocab K=16384, model memancarkan 1024 * log2(16384) = 14,336 bit (~1.75 KiB) data. Keluaran Stable Diffusion menghasilkan 512*512*24 bit = 6,291,456 bit (~768 KiB) piksel mentah. Berapa rasio kompresi dan kualitas apa yang dibelinya?

5. Baca LlamaGen (arXiv:2406.06525). Bagaimana model gambar autoregressive kelas-kondisional milik LlamaGen berbeda dari pendekatan masked Show-o?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Masked discrete diffusion | "MaskGIT-style" | Training to predict masked tokens; at inference, iteratively unmask the most-confident predictions |
| Cosine schedule | "Unmask schedule" | Decay of mask ratio over inference steps; concentrates confidence growth at mid-range |
| Parallel decoding | "All tokens at once" | Every step predicts the full sequence of masked tokens in one forward pass, then commits top-K |
| Hybrid attention | "Causal + bidirectional" | Mask that is causal over text tokens and bidirectional within image blocks |
| Inpainting | "Fill-in generation" | Condition on an image with some tokens masked, predict the missing ones; free from the training objective |
| Commitment rate | "Top-K per step" | How many tokens are declared "done" per iteration; controls inference vs quality trade-off |

## Bacaan Lanjutan

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
