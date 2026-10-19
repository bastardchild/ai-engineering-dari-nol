# Resep VLM Open-Weight: Apa yang Benar-Benar Penting

> Literatur VLM open-weight 2024-2026 adalah hutan tabel ablation. MM1 Apple menguji 13 kombinasi image encoder, connector, dan data mix. Molmo Allen AI membuktikan caption manusia terperinci mengalahkan distilasi GPT-4V. Cambrian-1 menjalankan 20+ perbandingan encoder. Idefics2 memformalkan ruang desink lima sumbu. Prismatic VLMs membandingkan 27 resep training pada benchmark terkontrol. Dari semua kebisingan itu, segelintir hasil bertahan lintas paper: image encoder lebih penting daripada arsitektur connector, campuran data lebih penting dari keduanya, dan caption manusia terperinci mengalahkan data sintetis terdistilasi. Pelajaran ini membaca tabel-tabel itu supaya kamu tidak perlu.

**Type:** Learn + lab
**Languages:** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites:** Phase 12 · 05 (LLaVA baseline)
**Time:** ~180 minutes

## Tujuan Pembelajaran

- Menyebutkan ruang desain VLM lima sumbu: image encoder, connector, LLM, data mix, resolution schedule.
- Membaca tabel ablation MM1 / Idefics2 / Cambrian-1 dan memprediksi knob mana yang menggeser benchmark tertentu.
- Memilih resep (encoder, connector, data, resolusi) untuk VLM baru diberi budget komputasi dan campuran tugas.
- Menjelaskan mengapa caption manusia terperinci mengalahkan distilasi GPT-4V pada jumlah token yang sama.

## Masalahnya

Ratusan VLM open-weight ada. Sebagian besar kesenjangan antara "bagus" dan "state-of-the-art" bukan arsitektur. Itu data, resolution schedule, dan pilihan encoder. Mengetahui knob mana yang diputar lebih dulu saat modelmu underperform menyelamatkanmu dari kesalahan 5-juta-GPU-hour.

Gelombang 2023 (LLaVA-1.5, InstructBLIP, MiniGPT-4) berjalan pada pretraining caption-pair + LLaVA-Instruct-150k. Baseline bagus. Berpuncak sekitar MMMU 35%.

Gelombang 2024 (MM1, Idefics2, Molmo, Cambrian-1, Prismatic VLMs) menjalankan ablation eksaustif. Hasilnya mengejutkan dan praktis.

## Konsepnya

### Ruang desain lima sumbu

Idefics2 (Laurençon et al., 2024) menyebutkan sumbu-sumbu itu:

1. Image encoder. CLIP ViT-L/14, SigLIP SO400m/14, DINOv2 ViT-g/14, InternViT-6B. Encoder berbeda dalam patch size, resolusi, dan objective pretraining.
2. Connector. MLP (2-4 layer), Q-Former (32 queries + cross-attn), Perceiver Resampler (64 queries), C-Abstractor (convolutional + bilinear pooling).
3. Model bahasa. Llama-3 8B / 70B, Mistral 7B, Phi-3, Gemma-2, Qwen2.5. Ukuran LLM adalah biaya parameter dominan.
4. Data training. Caption pair (CC3M, LAION), interleaved (OBELICS, MMC4), instruction (LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron).
5. Resolution schedule. Tetap 224/336/448, AnyRes, native dinamis. Dinaikkan selama training atau konstan.

Setiap VLM produksi membuat pilihan pada setiap sumbu. Sebagian besar variasi skor MMMU dijelaskan oleh sumbu 1, 4, dan 5 — bukan oleh connector mana yang kamu pilih.

### Sumbu 1: encoder > connector

MM1 Bagian 3.2 menunjukkan: bertukar dari CLIP ViT-L/14 ke SigLIP SO400m/14 menambah 3+ poin MMMU. Bertukar connector dari MLP ke Perceiver Resampler menambah kurang dari 1 poin. Idefics2 mereplikasi: SigLIP > CLIP, Q-Former ≈ MLP ≈ Perceiver pada jumlah token yang sama.

"Cambrian Vision Encoders Match-Up" Cambrian-1 (Tong et al., 2024) menjalankan 20+ encoder pada benchmark vision-centric (CV-Bench). Papan peringkat atas adalah campuran DINOv2 dan SigLIP; CLIP di tengah; ImageBind dan ViT-MAE lebih rendah. Kesenjangan dari CLIP ViT-L ke DINOv2 ViT-g/14 sekitar ~5-7 poin pada CV-Bench.

Encoder default 2026 untuk VLM terbuka adalah SigLIP 2 SO400m/14 untuk fitur semantik + dense, kadang digabung dengan fitur DINOv2 ViT-g/14 ("Spatial Vision Aggregator" Cambrian melakukan ini).

### Sumbu 2: desain connector seri imbang

MM1, Idefics2, Prismatic, dan MM-Interleaved semuanya mencapai kesimpulan yang sama: pada jumlah visual-token tetap, arsitektur connector nyaris tidak penting. MLP 2-layer pada patch mean-pooled berkinerja dalam 1 poin dari Q-Former 32-query pada token budget yang sama.

Yang penting adalah jumlah token. Lebih banyak visual token = lebih banyak komputasi LLM = kinerja lebih baik sampai titik tertentu, lalu pengembalian berkurang. 64 token per gambar terlalu sedikit untuk OCR. 576-1024 token adalah sweet spot untuk kebanyakan VLM terbuka. 2048+ hanya membantu untuk dokumen dan grafik.

Q-Former vs MLP adalah pertanyaan biaya, bukan kualitas: Q-Former membatasi token pada 32-64 terlepas dari resolusi gambar; MLP memancarkan semua patch token. Untuk input resolusi tinggi, Q-Former menghemat konteks LLM; untuk resolusi rendah, perbedaannya adalah noise.

### Sumbu 3: ukuran LLM menetapkan plafon

Menggandakan LLM dari 7B ke 13B secara andal menambah 2-4 poin pada MMMU di setiap paper VLM. Pada 70B kamu jenuh di sebagian besar benchmark. Plafon penalaran multimodal VLM adalah plafon penalaran teks LLM — visual encoder hanya bisa memberinya makanan, bukan bernalar untuknya.

Inilah mengapa Qwen2.5-VL-72B dan Claude Opus 4.7 menghancurkan MMMU-Pro dan ScreenSpot-Pro: otak bahasanya sangat besar. VLM 7B tidak bisa menggantikan VLM 70B melalui desain connector yang cerdik.

### Sumbu 4: data — caption manusia terperinci mengalahkan distilasi

Molmo + PixMo (Deitke et al., 2024) adalah hasil 2024 yang harus dibaca semua orang. Allen AI meminta anotator manusia mendeskripsikan gambar dalam pass speech-to-text padat 1-3 menit, menghasilkan 712K gambar dengan caption padat. Tidak ada distilasi GPT-4V di mana pun dalam data training.

Molmo-72B mengalahkan Llama-3.2-90B-Vision pada 11 dari 11 benchmark. Delta itu bukan arsitektur — itu kualitas caption. Caption manusia terperinci berisi 5-10x lebih banyak informasi per gambar daripada caption web pendek dan tetap berdasar faktual di mana distilasi GPT-4V berhalusinasi.

ShareGPT4V (Chen et al., 2023) dan Cauldron (Idefics2) mengikuti playbook yang sama dengan caption manusia + GPT-4V campuran. Trensnya jelas: untuk frontier 2026, kepadatan caption > jumlah caption > kenyamanan distilasi.

### Sumbu 5: resolusi dan jadwalnya

Ablation Idefics2: 384 -> 448 menambah 1-2 poin. 448 -> 980 dengan pemisahan gambar (AnyRes) menambah 3-5 lagi pada benchmark OCR. Training resolusi datar mendatar pada akurasi sedang; peningkatan resolusi (mulai 224, selesai 448 atau native) training lebih cepat dan berakhir lebih tinggi.

Cambrian-1 menjalankan trade-off resolusi vs token: pada komputasi tetap, kamu bisa memiliki lebih banyak token pada resolusi lebih rendah atau lebih sedikit token pada resolusi lebih tinggi. Resolusi lebih tinggi menang untuk OCR; resolusi-lebih-rendah-token-lebih-banyak menang untuk pemahaman scene umum.

Resep produksi 2026: latih Tahap 1 pada 384 tetap, Tahap 2 dengan resolusi dinamis hingga 1280 untuk tugas berat-OCR.

### Perbandingan terkontrol Prismatic

Prismatic VLMs (Karamcheti et al., 2024) adalah paper yang mengontrol semua sumbu. LLM 13B yang sama, data instruction yang sama, evaluasi yang sama — hanya satu sumbu yang bervariasi sekaligus. Hasilnya:

- Jumlah visual-token per-gambar menjelaskan ~60% variasi.
- Pilihan encoder menjelaskan ~20%.
- Arsitektur connector menjelaskan ~5%.
- Sisanya (campuran data, scheduler, LR) sisa ~15%.

Ini dekomposisi kasar, tetapi itu jawapan paling bersih untuk "apa yang harus saya ablate lebih dulu" dalam literatur.

### Pemilih untuk 2026

Dengan bukti tersebut, resep VLM-terbuka default untuk proyek baru pada 2026:

- Encoder: SigLIP 2 SO400m/14 pada resolusi asli dengan NaFlex, digabung dengan DINOv2 ViT-g/14 untuk fitur dense jika kamu butuh segmentasi/grounding.
- Connector: MLP 2-layer pada patch token. Lewati Q-Former kecuali kamu terbatas token.
- LLM: Qwen2.5 / Llama-3.1 / Gemma 2, 7B untuk biaya, 70B untuk kualitas, dipilih berdasarkan latensi target.
- Data: PixMo + ShareGPT4V + Cauldron, ditambah data instruction spesifik-tugas.
- Resolusi: dinamis (min 256, max 1280 piksel per sisi panjang).
- Schedule: Tahap 1 alignment (hanya projector), Tahap 2 full fine-tune, Tahap 3 fine-tune spesifik-tugas.

Setiap default itu tertelusur kembali ke ablation terukur dalam paper yang dikutip di akhir pelajaran ini.

```figure
l5-vlm-recipe-knobs
```

## Gunakan

`code/main.py` adalah parser tabel ablation dan pemilih resep. Ia meng-encode tabel ablation MM1 dan Idefics2 (diringkas) dan memungkinkanmu bertanya:

- "Diberi budget X dan tugas Y, resep mana yang menang?"
- "Jika saya bertukar SigLIP untuk CLIP pada Llama 7B, berapa delta MMMU yang diharapkan?"
- "Sumbu mana yang harus saya ablate lebih dulu untuk jawaban dengan keyakinan 80%?"

Outputnya adalah daftar resep terperingkat dengan delta benchmark yang diharapkan dan rekomendasi "ablate lebih dulu".

## Rilis

Pelajaran ini menghasilkan `outputs/skill-vlm-recipe-picker.md`. Diberi campuran tugas target, budget komputasi, dan target latensi, ia menghasilkan resep lengkap (encoder, connector, LLM, campuran data, resolution schedule) dengan kutipan ke ablation yang membenarkan setiap pilihan. Mencegah engineer menemukan ulang tabel ablation Idefics2 setiap kali proyek VLM baru dimulai.

## Latihan

1. Baca MM1 Bagian 3.2. Untuk LLM 2B tetap pada budget 50 juta gambar, encoder mana yang menang? Apakah jawabannya berbalik pada LLM 13B? Mengapa?

2. Cambrian-1 menemukan bahwa menggabungkan DINOv2 + SigLIP mengungguli masing-masing sendirian pada benchmark vision-centric tetapi tidak menambah sinyal pada MMMU. Prediksi benchmark mana yang naik dan mana yang tetap datar.

3. Targetmu adalah agen UI ponsel pada LLM 2B. Pilih encoder, connector, resolusi, dan campuran data. Benarkan setiap pilihan dengan tabel ablation spesifik.

4. Molmo mengirim model 4B dan 72B. Yang 4B kompetitif dengan VLM tertutup 7B; yang 72B mengalahkan Llama-3.2-90B-Vision pada 11/11 benchmark. Apa itu memberitahumu tentang hipotesis plafon-ukuran-LLM?

5. Rancang tabel ablation untuk mengisolasi kualitas campuran-data dari kualitas encoder pada VLM 7B. Berapa minimal pelatihan? Usulkan empat pengaturan sumbu.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Ablation | "Turning one knob" | Training multiple runs that differ in exactly one design-space axis, holding everything else constant |
| Connector | "Bridge" / "projector" | Trainable module that maps vision encoder output into the LLM's token space (MLP, Q-Former, Perceiver) |
| Detailed human caption | "Dense caption" | A multi-sentence human-written description (typically 80-300 tokens) richer than a web alt text |
| Distillation | "GPT-4V captions" | Training data generated by a stronger proprietary VLM; convenient but prone to inherited hallucination |
| AnyRes / dynamic res | "High-res path" | Strategy to feed images larger than the encoder's native resolution via tiling or M-RoPE |
| Resolution ramp | "Curriculum" | Training schedule that starts low-resolution and increases, speeding alignment learning |
| Vision-centric bench | "CV-Bench / BLINK" | Evaluation that stresses fine-grained visual perception rather than language-heavy reasoning |
| PixMo | "Molmo's data" | Allen AI's 712K densely-captioned image dataset; human speech transcribed into dense captions |

## Bacaan Lanjutan

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
