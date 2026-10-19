# Vision Transformers (ViT)

> Gambar adalah grid patch. Kalimat adalah grid token. Transformer yang sama memakan keduanya.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time:** ~45 minutes

## Masalahnya

Sebelum 2020, computer vision berarti convolution. Setiap SOTA pada ImageNet, COCO, dan benchmark deteksi memakai backbone CNN. Transformer untuk bahasa.

Dosovitskiy et al. (2020) — "An Image is Worth 16x16 Words" — menunjukkan kamu bisa membuang convolution sepenuhnya. Iris gambar menjadi patch berukuran tetap, proyeksikan setiap patch secara linear menjadi embedding, umpankan sequence ke encoder transformer vanilla. Pada skala cukup (pretraining ImageNet-21k atau lebih besar), ViT menyamai atau mengalahkan model berbasis ResNet.

ViT adalah awal pola yang lebih luas pada 2026: satu arsitektur, banyak modalitas. Whisper mentokenisasi audio. ViT mentokenisasi gambar. Token aksi untuk robotika. Token piksel untuk video. Transformer tidak peduli — beri ia sequence dan ia belajar.

Pada 2026, ViT dan keturunannya (DeiT, Swin, DINOv2, ViT-22B, SAM 3) memiliki sebagian besar vision. CNN masih menang pada perangkat edge dan task sensitif-latensi. Sisanya punya ViT di suatu tempat dalam stack.

## Konsepnya

![Image → patches → tokens → transformer](../assets/vit.svg)

### Langkah 1 — patchify

Bagi gambar `H × W × C` menjadi sequence flat patch `N × (P·P·C)`. Setup tipikal: gambar `224 × 224`, patch `16 × 16` → 196 patch masing-masing 768 nilai.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Ukuran patch adalah tuasnya. Patch lebih kecil = lebih banyak token, resolusi lebih baik, biaya attention kuadratik. Patch lebih besar = lebih kasar, lebih murah.

### Langkah 2 — embedding linear

Satu matriks terpelajari memproyeksikan setiap flat patch ke `d_model`. Setara dengan convolution ber-kernel `P` dan stride `P`. Dalam PyTorch ini harfiah `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` — implementasi 2 baris.

### Langkah 3 — tempelkan token `[CLS]`, tambahkan positional embedding

- Tempelkan token `[CLS]` yang bisa dipelajari. Hidden state finalnya adalah representasi gambar yang dipakai untuk klasifikasi.
- Tambahkan positional embedding terpelajari (ViT-orisinal) atau sinusoidal 2D (varian belakangan).
- Pada 2024+ RoPE diperluas ke 2D untuk posisi, kadang tanpa embedding eksplisit.

### Langkah 4 — encoder transformer standar

Tumpuk L block `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`. Identik dengan BERT. Tanpa lapisan khusus-vision. Inilah punchline pedagogis makalahnya.

### Langkah 5 — head

Untuk klasifikasi: ambil hidden state `[CLS]` → linear → softmax. Untuk DINOv2 atau SAM, buang `[CLS]`, pakai embedding patch secara langsung.

### Varian yang penting

| Model | Year | Change |
|-------|------|--------|
| ViT | 2020 | The original. Fixed patch size, full global attention. |
| DeiT | 2021 | Distillation; trainable on ImageNet-1k only. |
| Swin | 2021 | Hierarchical with shifted windows. Fixed sub-quadratic cost. |
| DINOv2 | 2023 | Self-supervised (no labels). Best general vision features. |
| ViT-22B | 2023 | 22B params; scaling laws apply. |
| SigLIP | 2023 | ViT + language pair, sigmoid contrastive loss. |
| SAM 3 | 2025 | Segment anything; ViT-Large + promptable mask decoder. |

### Mengapa butuh waktu lama

ViT butuh data *sangat banyak* untuk menyamai CNN karena ia tidak punya inductive bias CNN (invariansi translasi, lokalitas). Tanpa >100M gambar berlabel atau pretraining self-supervised yang kuat, CNN masih menang pada komputasi sebanding. DeiT memperbaikinya pada 2021 dengan trik distilasi; DINOv2 memperbaikinya permanen pada 2023 dengan self-supervision.

```figure
n5-patch-stream
```

## Bangun Sendiri

Lihat `code/main.py`. Patchify + embedding linear + sanity check Python-stdlib murni. Tanpa training — ViT pada skala realistis apa pun butuh PyTorch dan GPU berjam-jam.

### Langkah 1: gambar palsu

Gambar RGB 24 × 24 sebagai daftar baris berisi tuple `(R, G, B)`. Kita memakai patch 6×6 → 16 patch, masing-masing vektor embedding 108-d.

### Langkah 2: patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Urutan raster: row-major melintasi grid. Setiap ViT memakai urutan ini.

### Langkah 3: embed linear

Kalikan setiap flat patch dengan matriks acak `(patch_flat_size, d_model)`. Verifikasi shape output adalah `(N_patches + 1, d_model)` setelah menempelkan `[CLS]`.

### Langkah 4: hitung parameter untuk ViT realistis

Cetak jumlah param untuk ViT-Base: 12 lapis, 12 head, d=768, patch=16. Bandingkan dengan ResNet-50 (~25M). ViT-Base jatuh pada ~86M. ViT-Large ~307M. ViT-Huge ~632M.

## Gunakan

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**Embedding DINOv2 adalah bawaan 2026 untuk fitur gambar.** Bekukan backbone, latih head mungil. Bekerja untuk klasifikasi, retrieval, deteksi, captioning. Checkpoint DINOv2 milik Meta mengungguli CLIP pada setiap task vision non-teks.

**Pemilihan patch-size.** Model kecil memakai 16×16 (ViT-B/16). Prediksi padat (segmentasi) memakai 8×8 atau 14×14 (SAM, DINOv2). Model sangat besar memakai 14×14.

## Rilis

Lihat `outputs/skill-vit-configurator.md`. Skill ini memilih varian ViT dan ukuran patch untuk task vision baru berdasarkan ukuran dataset, resolusi, dan budget komputasi.

## Latihan

1. **Mudah.** Jalankan `code/main.py`. Verifikasi jumlah patch sama dengan `(H/P) * (W/P)` dan dimensi flat patch sama dengan `P*P*C`.
2. **Sedang.** Implementasikan positional embedding sinusoidal 2D — dua kode sinusoidal independen untuk `row` dan `col` setiap patch, digabungkan. Umpankan ke ViT PyTorch mungil dan bandingkan akurasi vs positional embedding terpelajari pada CIFAR-10.
3. **Sulit.** Bangun ViT 3 lapis (PyTorch), latih pada 1.000 gambar MNIST dengan patch 4×4. Ukur akurasi tes. Kini tambahkan pretraining DINOv2 pada 1.000 gambar yang sama (disederhanakan: latih saja encoder untuk memprediksi embedding patch dari patch ter-mask). Apakah akurasi membaik?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Patch | "The vision-transformer token" | Flat vector of pixel values for a `P × P × C` region of the image. |
| Patchify | "Chop + flatten" | Slice image into non-overlapping patches, flatten each to a vector. |
| `[CLS]` token | "The image summary" | Prepended learnable token; its final embedding is the image representation. |
| Inductive bias | "What the model assumes" | ViT has fewer priors than CNNs; needs more data to make up the gap. |
| DINOv2 | "Self-supervised ViT" | Trained without labels using image augmentation + momentum teacher. Best general image features in 2026. |
| SigLIP | "CLIP's successor" | ViT + text encoder trained with sigmoid contrastive loss; better than CLIP on matched compute. |
| Swin | "Windowed ViT" | Hierarchical ViT with local attention + shifted windows; sub-quadratic. |
| Register tokens | "2023 trick" | A few extra learnable tokens that soak up attention sinks; improves DINOv2 features. |

## Bacaan Lanjutan

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — makalah ViT.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — perbaikan register-token untuk DINOv2.
