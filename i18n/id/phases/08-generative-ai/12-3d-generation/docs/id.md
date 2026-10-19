# Pembangkitan 3D

> 3D adalah modalitas di mana leverage 2D-ke-3D paling kuat. Terobosan 2023 adalah 3D Gaussian Splatting. Dorongan generatif 2024-2026 melapiskan diffusion multi-view + rekonstruksi 3D di atasnya untuk menghasilkan objek dan scene dari satu prompt atau foto.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**Time:** ~45 minutes

## Masalahnya

Konten 3D itu menyakitkan:

- **Representasi.** Mesh, point cloud, grid voxel, signed distance field (SDF), neural radiance field (NeRF), Gaussian 3D. Masing-masing punya trade-off.
- **Kelangkaan data.** ImageNet punya 14M gambar. Dataset 3D bersih terbesar (Objaverse-XL, 2023) punya ~10M objek, sebagian besar kualitas rendah.
- **Memori.** Grid voxel 512³ adalah 128M voxel; NeRF scene yang berguna butuh 1M sampel/ray. Pembangkitan lebih sulit daripada rekonstruksi.
- **Supervisi.** Untuk gambar 2D kamu punya pikselnya. Untuk 3D kamu biasanya punya segelintir view 2D dan harus mengangkat ke 3D.

Stack 2026 memisahkan kedua problem. Pertama, bangkitkan *gambar multi-view 2D* dengan model diffusion. Kedua, fit *representasi 3D* (biasanya Gaussian splatting) ke gambar itu.

## Konsepnya

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### Representasi: 3D Gaussian Splatting (Kerbl et al., 2023)

Representasikan scene sebagai awan ~1M Gaussian 3D. Masing-masing punya 59 parameter: posisi (3), kovariansi (6, atau quaternion 4 + skala 3), opasitas (1), warna spherical-harmonics (48 pada derajat 3, 3 pada derajat 0).

Rendering = proyeksi + alpha-compositing. Cepat (~100 fps pada 1080p di 4090). Terdiferensiasi. Fit dengan gradient descent terhadap foto ground-truth. Scene muat dalam 5-30 menit pada GPU konsumen.

Dua inovasi 2023-2024 di atasnya:
- **Splat Gaussian generatif.** Model seperti LGM, LRM, InstantMesh memprediksi awan Gaussian langsung dari satu atau beberapa gambar.
- **Gaussian Splatting 4D.** Gaussian dengan offset per-frame untuk scene dinamis.

### Diffusion multi-view

Fine-tune model diffusion gambar terlatih untuk membangkitkan banyak view konsisten dari objek yang sama dari prompt teks atau gambar tunggal. Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024). Biasanya mengeluarkan 4-16 view mengelilingi objek, diangkat ke 3D via Gaussian splatting atau NeRF.

### Pipeline teks-ke-3D

| Model | Input | Output | Time |
|-------|-------|--------|------|
| DreamFusion (2022) | text | NeRF via SDS | ~1 hour per asset |
| Magic3D | text | mesh + texture | ~40 min |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1 min |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30 min |
| LRM (Meta, 2023) | image | triplane | ~5 s |
| InstantMesh (2024) | image | mesh | ~10 s |
| SV3D (Stability, 2024) | image | novel views | ~2 min |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1 min |
| TripoSR (2024) | image | mesh | ~1 s |
| Meshy 4 (2025) | text + image | PBR mesh | ~30 s |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60 s |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30 s |

Arah 2025-2026: model teks-ke-mesh langsung dengan material PBR yang cocok untuk game engine. Langkah intermediate diffusion multi-view masih resep berkinerja-terbaik untuk objek umum.

### NeRF (untuk konteks)

Neural Radiance Field (Mildenhall et al., 2020). MLP mungil mengambil `(x, y, z, arah view)` dan mengeluarkan `(warna, densitas)`. Render dengan mengintegralkan sepanjang ray. Mengalahkan sintesis novel-view berbasis-mesh dalam kualitas tapi 100-1000x lebih lambat dirender. Digantikan Gaussian splatting untuk sebagian besar pemakaian real-time tapi masih dominan dalam riset.

```figure
v4-3d-multiview
```

## Bangun Sendiri

`code/main.py` mengimplementasikan fit "Gaussian splatting" 2D mainan: representasikan gambar target sintetis (gradien mulus) sebagai jumlah splat Gaussian 2D. Optimasi posisi, warna, dan kovariansi dengan gradient descent agar cocok dengan target. Kamu melihat dua operasi inti: render forward (splat + alpha-composite) dan fit dengan gradient descent.

### Langkah 1: splat Gaussian 2D

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Langkah 2: render dengan menjumlah splat

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

Gaussian Splatting 3D nyata mengurutkan Gaussian berdasarkan depth dan alpha-composite berurutan. Mainan 2D kita hanya menjumlah.

### Langkah 3: fit dengan gradient descent

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Jebakan

- **Inkonsistensi view.** Kalau kamu membangkitkan 4 view independen dan mereka tidak sepakat soal struktur objek, fit 3D-nya blur. Perbaiki: diffusion multi-view dengan attention bersama.
- **Halusinasi sisi-belakang.** Tunggal-gambar → 3D harus mengarang sisi tak terlihat. Kualitas sangat bervariasi.
- **Ledakan splat Gaussian.** Training tak terkendala tumbuh menjadi 10M splat dan overfit. Heuristik densifikasi + pruning (dari makalah 3D-GS orisinal) esensial.
- **Masalah topologi.** Mesh dari field implisit (SDF) sering punya lubang atau self-intersection. Jalankan remesher (mis. voxel remesh blender) sebelum mengirim.
- **Lisensi data training.** Objaverse punya lisensi campuran; pemakaian komersial bervariasi per model.

## Gunakan

| Task | 2026 pick |
|------|-----------|
| Scene reconstruction from photos | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| Text-to-3D object for games | Meshy 4 or Rodin Gen-1.5 (PBR output) |
| Image-to-3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| Novel-view synthesis from few images | CAT3D, SV3D |
| Dynamic scene reconstruction | 4D Gaussian Splatting |
| Avatar / clothed human | Gaussian Avatar, HUGS |
| Research / SOTA | Whatever dropped last week |

Untuk mengirim 3D produksi dalam pipeline game atau e-commerce: Meshy 4 atau Rodin Gen-1.5 mengeluarkan mesh PBR yang langsung masuk Unity / Unreal.

## Rilis

Simpan `outputs/skill-3d-pipeline.md`. Skill menerima brief 3D (input: teks / satu gambar / beberapa gambar; output: mesh / splat / NeRF; pemakaian: render / game / VR) dan mengeluarkan: pipeline (diffusion multi-view + fit, atau model mesh langsung), model dasar, budget iterasi, post-processing topologi, channel material yang dibutuhkan.

## Latihan

1. **Mudah.** Jalankan `code/main.py` dengan 4, 16, 64 Gaussian. Laporkan MSE final vs target.
2. **Sedang.** Perluas ke Gaussian warna (RGB). Konfirmasi rekonstruksi cocok dengan pola warna target.
3. **Sulit.** Pakai gsplat atau Nerfstudio, rekonstruksi objek nyata dari capture 50 foto. Laporkan waktu fit dan SSIM final pada view held-out.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | Scene as a cloud of 3D Gaussians; differentiable alpha-composite render. |
| NeRF | "Neural radiance field" | MLP that outputs color + density at a 3D point; render by ray integration. |
| Triplane | "Three 2-D planes" | Factor 3D into three 2-D axis-aligned feature grids; cheaper than volumetric. |
| SDS | "Score distillation sampling" | Train 3D model by using 2D-diffusion score as pseudo-gradient. |
| Multi-view diffusion | "Many views at once" | Diffusion model that outputs a batch of consistent camera views. |
| PBR | "Physically-based rendering" | Material with albedo, roughness, metallic, normal channels. |
| Densification | "Grow splats" | 3DGS training heuristic: split / clone splats in high-gradient regions. |

## Catatan produksi: 3D belum punya substrat bersama

Tidak seperti gambar (latent diffusion + DiT) dan video (DiT spatiotemporal), 3D tidak punya satu runtime dominan pada 2026. Pohon keputusan produksi bercabang pada representasi:

- **NeRF / triplane.** Inference adalah ray-marching + forward MLP per sampel. Render 512² membutuhkan jutaan forward MLP. Batch sampel ray secara agresif; SDPA/xformers berlaku.
- **Diffusion multi-view + rekonstruksi LRM.** Pipeline dua tahap. Tahap 1 (DiT multi-view) adalah server diffusion persis seperti Pelajaran 07. Tahap 2 (transformer LRM) adalah satu forward pass atas view. Profil latensi keseluruhan adalah "diffusion + sekali-jalan" — pilih primitif serving per-tahap dengan sesuai.
- **SDS / DreamFusion.** Optimasi per-aset, bukan inference. Job build, bukan handler request.

Untuk sebagian besar produk 2026, jawaban yang tepat adalah "jalankan model diffusion multi-view on request, rekonstruksi ke 3DGS secara asinkron, sajikan 3DGS untuk viewing real-time". Ini membagi workload bersih antara server GPU-inference (cepat) dan optimizer offline (lambat).

## Bacaan Lanjutan

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — diffusion multi-view.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d-novel-multi-view-synthesis-and-3d-generation-from-a-single-image-using-latent-video-diffusion) — SV3D.
