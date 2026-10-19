# ControlNet, LoRA & Conditioning

> Teks saja adalah sinyal kontrol yang kikuk. ControlNet membuatmu mengkloning model diffusion terlatih dan menyetirnya dengan peta depth, kerangka pose, coretan, atau gambar tepi. LoRA membuatmu fine-tune model 2B-parameter dengan melatih 10 juta parameter. Bersama mereka mengubah Stable Diffusion dari mainan menjadi pipeline gambar 2026 yang dikirim setiap agensi.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — for LoRA foundation)
**Time:** ~75 minutes

## Masalahnya

Prompt seperti "a woman in a red dress walking a dog on a busy street" tidak memberi model info soal *di mana* anjingnya, *pose apa* wanitanya, atau *perspektif* jalannya. Teks menentukan sekitar 10% dari yang dibutuhkan untuk menspesifikasi gambar. Sisanya visual dan tidak bisa dideskripsikan efisien dengan kata.

Melatih model kondisional baru dari nol untuk setiap sinyal (pose, depth, canny, segmentasi) itu prohibitif. Kamu mau menjaga backbone SDXL 2.6B-param tetap beku, menempelkan side-network kecil yang membaca conditioning, dan membuatnya mendorong fitur intermediate backbone. Itulah ControlNet.

Kamu juga mau mengajarkan model konsep baru (wajahmu, produkmu, style-mu) tanpa retraining model penuh. Kamu mau delta 100x lebih kecil. Itulah LoRA — adapter low-rank yang dicolokkan ke bobot attention yang ada.

ControlNet + LoRA + teks = toolkit praktisi 2026. Sebagian besar pipeline gambar produksi melapis 2-5 LoRA, 1-3 ControlNet, dan IP-Adapter di atas basis SDXL / SD3 / Flux.

## Konsepnya

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

Ambil SD terlatih. *Kloning* separuh encoder U-Net. Bekukan yang orisinal. Latih kloningan untuk menerima input conditioning ekstra (tepi, depth, pose). Sambungkan kloningan kembali ke separuh decoder yang orisinal dengan skip connection *zero-convolution* (conv 1×1 yang diinisialisasi nol — mulai sebagai no-op, pelajari delta).

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Init zero-conv berarti ControlNet mulai sebagai identitas — tanpa bahaya bahkan sebelum training. Latih pada 1M triple (prompt, condition, gambar) dengan loss diffusion standar.

ControlNet per-modalitas dikirim sebagai model sisi kecil (~360M untuk SDXL, ~70M untuk SD 1.5). Kamu bisa mengomposisinya saat inference:

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

Untuk lapis linear apa pun `W ∈ R^{d×d}` dalam model, bekukan `W` dan tambahkan delta low-rank:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

dengan `r << d`. Rank 4-16 standar untuk attention, rank 64-128 untuk fine-tune berat. Jumlah parameter baru: `2 · d · r` alih-alih `d²`. Untuk attention SDXL dengan `d=640`, `r=16`: 20k param per adapter alih-alih 410k — reduksi 20x. Di seluruh model: LoRA biasanya 20-200MB vs basis 5GB.

Saat inference kamu bisa menskalakan LoRA: `W' = W + α · B @ A`. `α = 0.5-1.5` normal. Banyak LoRA bertumpuk aditif (dengan caveat biasa bahwa mereka berinteraksi secara non-linear).

### IP-Adapter (Ye et al., 2023)

Adapter mungil yang menerima *gambar* sebagai conditioning (di samping teks). Memakai encoder gambar CLIP untuk menghasilkan token gambar, menyuntikkannya ke cross-attention di samping token teks. ~20MB per model dasar. Membuatmu bisa "membangkitkan gambar dengan style referensi ini" tanpa LoRA.

## Matriks komposabilitas

| Tool | What it controls | Size | When to use |
|------|------------------|------|-------------|
| ControlNet | Spatial structure (pose, depth, edges) | 70-360MB | Exact layout, composition |
| LoRA | Style, subject, concept | 20-200MB | Personalization, style |
| IP-Adapter | Style or subject from reference image | 20MB | No text can describe the look |
| Textual Inversion | Single concept as a new token | 10KB | Legacy, mostly replaced by LoRA |
| DreamBooth | Full fine-tune on a subject | 2-5GB | Strong identity, high compute |
| T2I-Adapter | Lighter ControlNet alternative | 70MB | Edge devices, inference budget |

ControlNet ≈ spasial. LoRA ≈ semantik. Pakai keduanya.

```figure
v4-controlnet-zero
```

## Bangun Sendiri

`code/main.py` mensimulasikan kedua mekanisme pada 1-D:

1. **LoRA.** Lapis linear terlatih `W`. Bekukan. Latih `B @ A` low-rank sehingga `W + BA` cocok dengan lapis linear target. Tunjukkan bahwa `r = 1` cukup untuk mempelajari koreksi rank-1 dengan sempurna.

2. **ControlNet-lite.** Prediktor "basis beku" dan "side network" yang membaca sinyal ekstra. Output side network di-gate dengan skalar terpelajari yang diinisialisasi nol (versi zero-conv kita). Latih dan perhatikan gate naik.

### Langkah 1: matematika LoRA

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Langkah 2: side network ber-init-nol

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

Pada langkah 0 output identik dengan basis. Training awal mengupdate `gate` perlahan — tanpa drift katastropik.

## Jebakan

- **Over-scaling LoRA.** `α = 2` atau `α = 3` adalah hack umum "buat lebih kuat" yang menghasilkan output over-stylized / rusak. Jaga `α ≤ 1.5`.
- **Konflik bobot ControlNet.** Memakai ControlNet Pose pada bobot 1.0 dan ControlNet Depth pada bobot 1.0 biasanya overshoot. Jumlah bobot ≈ 1.0 adalah bawaan aman.
- **LoRA pada basis yang salah.** LoRA SDXL diam-diam no-op pada SD 1.5 karena dimensi attention tidak cocok. Diffusers akan memperingatkan pada 0.30+.
- **Drift Textual Inversion.** Token yang dilatih pada satu checkpoint melayang parah pada yang lain. LoRA lebih portabel.
- **Penggabungan-bobot dan penyimpanan LoRA.** Kamu bisa memanggang LoRA ke bobot model dasar untuk inference lebih cepat (tanpa penjumlahan runtime), tapi kamu kehilangan kemampuan menskalakan `α` saat runtime. Simpan kedua versi.

## Gunakan

| Goal | 2026 pipeline |
|------|---------------|
| Reproduce a brand's art style | LoRA trained on ~30 curated images at rank 32 |
| Put my face in a generated image | DreamBooth or LoRA + IP-Adapter-FaceID |
| Specific pose + prompt | ControlNet-Openpose + SDXL + text |
| Depth-aware composition | ControlNet-Depth + SD3 |
| Reference + prompt | IP-Adapter + text |
| Exact layout | ControlNet-Scribble or ControlNet-Canny |
| Background replace | ControlNet-Seg + Inpainting (Lesson 09) |
| Fast 1-step style | LCM-LoRA on SDXL-Turbo |

## Rilis

Simpan `outputs/skill-sd-toolkit-composer.md`. Skill menerima task (aset input: prompt, gambar referensi opsional, pose opsional, depth opsional, coretan opsional) dan mengeluarkan stack tool, bobot, dan protokol seed yang bisa direproduksi.

## Latihan

1. **Mudah.** Dalam `code/main.py`, variasikan rank LoRA `r` dari 1 ke 4. Pada rank berapa LoRA cocok persis dengan delta target rank-2?
2. **Sedang.** Latih dua LoRA terpisah pada dua transform target. Muat keduanya bersama dan tunjukkan interaksi aditifnya. Kapan interaksi merusak linearitas?
3. **Sulit.** Pakai diffusers untuk menumpuk: SDXL-base + Canny-ControlNet (bobot 0.8) + LoRA style (α 0.8) + IP-Adapter (bobot 0.6). Ukur trade-off FID-vs-kepatuhan-prompt saat bobot stack bervariasi.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ControlNet | "Spatial control" | Cloned encoder + zero-conv skips; reads a conditioning image. |
| Zero convolution | "Starts as identity" | 1×1 conv initialized to zero; ControlNet starts as no-op. |
| LoRA | "Low-rank adapter" | `W + B @ A`, `r << d`; 100x fewer params than a full fine-tune. |
| rank r | "The knob" | LoRA compression; 4-16 typical, 64+ for heavy personalization. |
| α | "LoRA strength" | Runtime scaling of the LoRA delta. |
| IP-Adapter | "Reference image" | Small image-conditioning adapter via CLIP-image tokens. |
| DreamBooth | "Full subject fine-tune" | Train the full model on ~30 images of a subject. |
| Textual Inversion | "New token" | Learn a new word embedding only; legacy, mostly replaced. |

## Catatan produksi: swap LoRA, lane ControlNet, serving multi-tenant

SaaS teks-ke-gambar nyata melayani ratusan LoRA dan selusin ControlNet atas checkpoint basis yang sama. Problem serving-nya sangat mirip multi-tenancy LLM (literatur produksi membahas kasus LLM di bawah continuous batching dan LoRAX / S-LoRA):

- **Hot-swap LoRA, jangan merge.** Menggabungkan `W' = W + α·B·A` ke basis memberi inference per-langkah ~3-5% lebih cepat tapi membekukan `α` dan basis. Jaga LoRA tetap panas di VRAM sebagai delta rank-r; diffusers mengekspos `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` untuk aktivasi per-request. Biaya swap adalah bobot `2 · d · r · num_layers` — skala-MB, sub-detik.
- **ControlNet sebagai lane attention kedua.** Encoder kloningan berjalan paralel dengan basis. Dua ControlNet masing-masing berbobot 1.0 = dua forward pass ekstra per langkah, bukan satu pass tergabung. Ruang-kepala batch-size turun kuadratik. Budget-kan ~1.5× biaya langkah per ControlNet aktif.
- **LoRA terkuantisasi juga.** Kalau kamu mengkuantisasi basis (lihat Pelajaran 07, Flux pada 8GB), delta LoRA juga terkuantisasi bersih ke 8-bit atau 4-bit. Loading ala-QLoRA membuatmu menumpuk 5-10 LoRA di atas basis Flux 4-bit tanpa meledakkan memori.

Khusus-Flux: notebook Flux-on-8GB milik Niels mengkuantisasi basis ke 4-bit; menumpuk LoRA style (`pipe.load_lora_weights("user/style-lora")`) pada basis terkuantisasi itu pada `weight_name="pytorch_lora_weights.safetensors"` tetap bekerja. Inilah resep yang dikirim sebagian besar agensi SaaS pada 2026.

## Bacaan Lanjutan

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA (orisinal untuk LLM; port ke diffusion).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — alternatif ControlNet yang lebih ringan.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — pipeline referensi.
