# Hukum Scaling

> Makalah Kaplan 2020 berkata: model lebih besar, loss lebih rendah. Makalah Hoffmann 2022 berkata: kamu kurang melatih. Komputasi masuk ke dua ember — parameter dan token — dan pembagiannya tidak jelas.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## Masalahnya

Ketika kamu punya komputasi training C FLOPs dan menginginkan model terbaik, kamu menghadapi dua knob:

1. **Berapa banyak parameter (N)?** Model lebih besar, kapasitas lebih tinggi.
2. **Berapa banyak token training (D)?** Lebih banyak data, pemakaian kapasitas lebih baik.

FLOPs berskala kira-kira `6 × N × D`. Kamu bisa menaikkan N dan menurunkan D, atau menaikkan D dan menurunkan N. Mana yang lebih baik?

Sebelum 2022, jawabannya adalah "dorong N keras." GPT-3 (2020) adalah 175B parameter yang dilatih pada ~300B token. Rasio sekitar 1.7 token per parameter. Hukum scaling Kaplan mendukung ini.

Hoffmann et al. (2022), melatih keluarga kecil model bernama Chinchilla, menemukan sesuatu yang berbeda: rasio optimal lebih dekat ke **20 token per parameter**. GPT-3 10× kurang-terlatih. Chinchilla (70B param, 1.4T token) mengalahkan GPT-3 (175B, 300B token) di setiap benchmark dengan biaya inference 2.5× lebih rendah.

2026 adalah dunia Chinchilla — dengan satu twist penting. Llama 3 8B dilatih pada 15 triliun token, rasio 1,875 token per parameter. Sembilan puluh empat kali melewati Chinchilla-optimal. Biaya inference lebih penting daripada biaya training untuk model yang akan dipakai pada skala besar, jadi over-training (melewati Chinchilla) demi footprint deployable yang lebih kecil adalah bawaan 2026.

## Konsepnya

![Chinchilla curves: loss vs compute at various N/D ratios](../assets/scaling-laws.svg)

### Hukum Hoffmann

Dari makalah Chinchilla, loss mengikuti:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = parameter (non-embedding).
- `D` = token training.
- `α ≈ 0.34`, `β ≈ 0.28` (kira-kira simetris).
- `E ≈ 1.69`, ceiling loss yang tak tereduksi.
- `A ≈ 406`, `B ≈ 411`.

Dua suku saling bertukar saat kamu berskala. Ambil turunan thd. `N` pada komputasi tetap (C = 6ND) dan selesaikan:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Compute-optimal: 20 token per parameter.

### Mengapa tetap over-training

Chinchilla-optimal meminimalkan loss training per FLOP training. Tapi kamu membayar biaya training sekali; biaya inference selamanya.

Untuk chatbot yang melayani satu triliun token per bulan, inference mendominasi biaya total. Pendekatan Llama: latih lebih kecil, lebih lama. 8B pada 15T token sangat dioptimasi-inference:

- Muat di GPU konsumen.
- Latensi hanya sepersekian dari 70B Chinchilla-optimal.
- Kualitas cukup dekat untuk sebagian besar task.

Makalah DeepMind 2024 ("Over-training is the new optimal") memformalkannya. Untuk workload yang didominasi-inference, rasio yang tepat lebih dekat ke 100–500 token per parameter tergantung volume serving.

### Emergence vs smoothness

Klaim: kemampuan tertentu (aritmetika, penalaran multi-langkah, mengikuti chain-of-thought) "muncul" tiba-tiba pada suatu skala.

Schaeffer et al. (2023) berargumen ini adalah artefak pengukuran: metrik emergent memakai skoring diskontinu (exact match, akurasi pada ambang) yang menyembunyikan perbaikan mulus pada logit yang mendasari. Metrik kontinu (cross-entropy) menunjukkan kurva mulus.

Pada 2026 konsensusnya: prediksi via loss kontinu bisa diandalkan. Lompatan benchmark sering kali artefak scorer. Rencanakan budget terhadap metrik kontinu.

### Gambaran 2026

Hukum scaling masih bekerja, tapi:

| Factor | Changed how |
|--------|-------------|
| Data quality | Curating "good" tokens (Phi-style) shifts curves by >2× effective compute |
| MoE | Total params decouple from active FLOPs; scaling laws per-active-FLOP |
| Post-training | Some capabilities (instruction following, code) shift with SFT+RLHF more than pretraining |
| Multimodality | Image + text tokens scale together; separate curves per modality |
| Synthetic data | Models generate training data; effective compute can compound |

Optimizer Muon (Kimi Moonlight, 2024) menunjukkan kenaikan effective-compute ~2× atas AdamW pada data sebanding. Sebagian run training 2026 memakai Muon secara default. Mengubah konstanta absolut dalam hukum scaling, bukan bentuknya.

```figure
scaling-laws
```

## Bangun Sendiri

Lihat `code/main.py`. Kita mengimplementasikan persamaan loss Chinchilla dan memecahkan `(N, D)` compute-optimal pada masing-masing dari beberapa budget komputasi.

### Langkah 1: loss Chinchilla

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

Plot `L` sebagai kontur atas `(N, D)` pada `C = 6ND` tetap. Temukan minimumnya.

### Langkah 2: frontier compute-optimal

Untuk budget komputasi dari `1e17` sampai `1e25` FLOPs, temukan `(N, D)` yang meminimalkan loss dengan kendala `6ND = C`. Verifikasi rasio `D/N ≈ 20`.

### Langkah 3: biaya over-training

Hitung loss ekstra yang kamu bayar untuk melatih model 10× lebih kecil (1/10 dari N optimal, 10× D optimal). Laporkan penghematan FLOP inference (proporsional terhadap N) sebagai gantinya.

### Langkah 4: bandingkan dengan model nyata

Masukkan pasangan `(N, D)` yang diketahui untuk GPT-3, Chinchilla, Llama 3 8B, DeepSeek-V3 (param aktif), dan bandingkan loss prediksi vs yang dilaporkan.

## Gunakan

Kamu kecil kemungkinan melatih model frontier sendiri. Tapi hukum scaling memberi tahu:

1. **Apakah fine-tune-mu punya cukup data.** Kalau data khusus-task-mu di bawah 20 token per param model dasar, harapkan saturasi pada suatu lantai loss.
2. **Apakah memilih model dasar yang lebih besar.** Kalau kamu menghabiskan seluruh budget untuk inference, pilih model yang lebih kecil dan dilatih-lebih-lama.
3. **Di mana return mengecil.** Melampaui 1000× Chinchilla-optimal, perubahan log-loss menjadi noise.

**Trayektori riset pada 2026:**

- **Rezim data-constrained.** Web punya jumlah token berkualitas-tinggi yang terbatas (~5–10 triliun bahasa Inggris setelah filtering). Pretraining frontier mendekati plafon ini. Data sintetis, multibahasa, multimodal, dan fine-tuning berskala-RLHF adalah tuas berikutnya.
- **Trik pengali-komputasi.** Optimizer Muon, MoE, kurasi data yang lebih baik — masing-masing menggeser konstanta absolut, bukan asimtotnya.
- **Hukum scaling untuk RL.** Pertanyaan terbuka. Bukti awal menunjukkan power-law dalam sampel RL tapi dengan eksponen sangat berbeda dari pretraining.

## Rilis

Lihat `outputs/skill-training-budget-estimator.md`. Skill ini memilih `(N, D, jam, GPU)` untuk run training baru berdasarkan budget komputasi, batasan deployment, dan loss target.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Cetak `(N, D)` Chinchilla-optimal untuk budget komputasi `1e20`, `1e22`, `1e24`. Bandingkan dengan tabel model nyata.
2. **Sedang.** Implementasikan kurva loss-sebagai-fungsi-komputasi Hoffmann. Plot loss vs `log10(C)` untuk frontier compute-optimal. Identifikasi kapan hukum memprediksi kita butuh `>10^28` FLOPs untuk 0,1 reduksi berikutnya dalam cross-entropy.
3. **Sulit.** Fit hukum scaling-mu sendiri pada 5 model mungil (100K sampai 10M param) yang dilatih pada dataset yang sama. Estimasi `α` dan `E`. Seberapa cocok eksponenmu dengan yang dipublikasikan?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Parameters (N) | "Model size" | Non-embedding weight count; determines capacity. |
| Tokens (D) | "Training data" | Number of training tokens seen; determines how well the parameters get used. |
| Compute (C) | "FLOPs spent" | Approximately `6 × N × D` for a standard transformer. |
| Chinchilla-optimal | "D/N ≈ 20" | Ratio that minimizes loss per FLOP of pretraining. |
| Over-training | "Past Chinchilla" | Spend extra training FLOPs to save inference FLOPs; D/N >> 20. |
| Irreducible loss | "The floor" | The `E` term in the scaling law; the entropy of the data itself. |
| Emergent capability | "Sudden jumps at scale" | Often a scorer artifact; continuous loss is smooth. |
| Effective compute | "Training-efficiency multiplier" | Better data / optimizer / architecture multiplies how far a FLOP goes. |

## Bacaan Lanjutan

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — makalah hukum scaling pertama; kurang-terlatih.
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla.
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — emergence sebagai artefak pengukuran.
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — mengapa over-training Llama tepat untuk workload-nya.
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — pengali komputasi 2×.
