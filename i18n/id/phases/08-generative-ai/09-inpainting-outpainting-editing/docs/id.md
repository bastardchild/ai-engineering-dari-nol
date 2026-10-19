# Inpainting, Outpainting & Editing Gambar

> Teks-ke-gambar membuat hal baru. Inpainting memperbaiki yang lama. Di produksi, 70% pekerjaan gambar berbayar adalah editing — tukar latar, hapus logo, perluas kanvas, regenerasi tangan. Inpainting adalah tempat diffusion membuktikan jasanya.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Time:** ~75 minutes

## Masalahnya

Klien mengirim foto produk sempurna dengan papan mengganggu di latar. Kamu mau menghapus papan dan membiarkan sisanya pixel-identik. Kamu tidak bisa menjalankan teks-ke-gambar dari nol — hasilnya akan punya warna berbeda, pencahayaan berbeda, sudut produk berbeda. Kamu mau meregenerasi *hanya* wilayah ter-mask, dan kamu mau regenerasinya menghormati konteks sekitar.

Itulah inpainting. Varian:

- **Inpainting.** Regenerasi di dalam mask, jaga piksel luar.
- **Outpainting.** Regenerasi di luar mask (atau melampaui kanvas), jaga yang dalam.
- **Editing gambar.** Regenerasi seluruh gambar tapi jaga fidelitas semantik atau struktural ke yang orisinal (SDEdit, InstructPix2Pix).

Setiap pipeline diffusion pada 2026 menyertakan mode inpainting. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit. Mereka bekerja pada prinsip yang sama.

## Konsepnya

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### Pendekatan naif (dan mengapa ia salah)

Jalankan teks-ke-gambar standar dengan mask. Di setiap langkah sampling, ganti wilayah tak-ter-mask dari laten ber-noise dengan gambar bersih yang di-forward-diffuse. Ia bekerja... buruk. Artefak batas merembes karena model tidak punya info soal apa yang ada di wilayah ter-mask.

### Model inpainting yang benar

Latih U-Net termodifikasi yang mengambil 9 channel input alih-alih 4:

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

Channel ekstra adalah salinan gambar source terenkode-VAE plus mask satu-channel. Saat training, kamu memask wilayah gambar secara acak dan melatih model untuk mendenoise hanya wilayah ter-mask sementara wilayah tak-ter-mask diberikan sebagai sinyal conditioning bersih. Saat inference, model bisa "melihat" apa yang mengelilingi wilayah ter-mask dan menghasilkan completion yang koheren.

SD-Inpaint, SDXL-Inpaint, Flux-Fill semuanya memakai input 9-channel (atau analog) ini. `StableDiffusionInpaintPipeline`, `FluxFillPipeline` milik Diffusers.

### SDEdit (Meng et al., 2022) — editing gratis

Tambahkan noise ke gambar source sampai `t` intermediate, lalu jalankan rantai reverse dari `t` turun ke 0 dengan prompt baru. Tanpa retraining. Pilihan `t` awal menukar fidelitas dengan kebebasan kreatif:

- `t/T = 0.3` → nyaris identik dengan source, perubahan stilistis kecil
- `t/T = 0.6` → edit sedang, menjaga struktur kasar
- `t/T = 0.9` → dibangkitkan dari dekat-noise, preservasi source minimal

### InstructPix2Pix (Brooks et al., 2023)

Fine-tune model diffusion pada triple `(input_image, instruction, output_image)`. Saat inference, kondisikan pada gambar input maupun instruksi teks ("make it sunset", "add a dragon"). Dua skala CFG: skala gambar dan skala teks.

### RePaint (Lugmayr et al., 2022)

Pertahankan model diffusion tanpa-syarat standar. Di setiap langkah reverse, resample — sesekali lompat kembali ke state lebih-ber-noise dan regenerasi. Menghindari artefak batas. Dipakai ketika kamu tidak punya model inpainting terlatih.

```figure
inpaint-mask-reinject
```

## Bangun Sendiri

`code/main.py` mengimplementasikan skema inpainting 1-D mainan pada data 5 dimensi. Kita melatih DDPM pada data campuran 5-D di mana setiap sampel adalah 5 float dari satu dari dua cluster. Saat inference, kita "memask" 2 dari 5 dimensi, menyuntikkan versi noisy-forward dari tiga yang tak-ter-mask di setiap langkah, dan meregenerasi hanya dimensi ter-mask.

### Langkah 1: data DDPM 5-D

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Langkah 2: latih denoiser atas semua 5 dim

DDPM standar. Net mengeluarkan prediksi noise 5-D untuk input ber-noise 5-D.

### Langkah 3: saat inference, reverse yang sadar-mask

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

Ini adalah pendekatan naif dan ia bekerja pada data 1-D mainan. Inpainting gambar nyata memakai input 9-channel karena koherensi tekstur lebih penting.

### Langkah 4: outpainting

Outpainting adalah inpainting dengan mask dibalik: mask kanvas baru (yang sebelumnya tidak ada), isi sisanya dengan yang orisinal. Objektif training identik.

## Jebakan

- **Seam.** Pendekatan naif meninggalkan batas terlihat karena info gradien tidak mengalir melintasi mask. Perbaiki: dilatasi mask 8-16 piksel, atau pakai model inpainting yang benar.
- **Kebocoran mask.** Kalau wilayah tak-ter-mask gambar conditioning berkualitas-rendah atau ber-noise, ia mengotori pembangkitan di dalam mask. Denoise atau blur sedikit.
- **CFG berinteraksi dengan ukuran mask.** CFG tinggi pada mask kecil = patch saturasi. Turunkan CFG untuk edit kecil.
- **Tebing fidelitas SDEdit.** Berpindah dari `t/T = 0.5` ke `t/T = 0.6` bisa kehilangan identitas subjek. Sweep dan checkpoint.
- **Prompt mismatch.** Prompt seharusnya mendeskripsikan *seluruh* gambar, bukan hanya konten baru. "A cat sitting on a chair" bukan "a cat".

## Gunakan

| Task | Pipeline |
|------|----------|
| Remove object, small mask | SD-Inpaint or Flux-Fill, standard prompt |
| Replace sky | SD-Inpaint + "blue sky at sunset" |
| Extend canvas | SDXL outpaint mode (8px feather) or Flux-Fill with outpaint mask |
| Regenerate hand / face | SD-Inpaint with prompt re-describing the subject + ControlNet-Openpose |
| Change style of one region | SDEdit at `t/T=0.5` on masked region |
| "Make it sunset" | InstructPix2Pix or Flux-Kontext |
| Background replacement | SAM mask → SD-Inpaint |
| Ultra-high-fidelity | Flux-Fill or GPT-Image (hosted) for hardest cases |

SAM (Segment Anything milik Meta, 2023) + inpaint diffusion adalah pipeline penghapus-latar 2026. SAM 2 (2024) bekerja pada video.

## Rilis

Simpan `outputs/skill-editing-pipeline.md`. Skill menerima gambar orisinal + deskripsi edit + mask opsional (atau prompt SAM) dan mengeluarkan: pendekatan pembangkitan-mask, model dasar, skala CFG (gambar + teks), SDEdit-t atau mode inpainting, dan checklist QA.

## Latihan

1. **Mudah.** Dalam `code/main.py`, variasikan fraksi dimensi ter-mask dari 0.2 ke 0.8. Pada fraksi berapa kualitas inpaint (residual di dim ter-mask) sama dengan pembangkitan tanpa-syarat?
2. **Sedang.** Implementasikan RePaint: di setiap langkah reverse ke-10, lompat kembali 5 langkah (tambahkan noise) dan denoise ulang. Ukur apakah ia mengurangi residual batas di tepi mask.
3. **Sulit.** Pakai Hugging Face diffusers untuk membandingkan: SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill pada 20 task regenerasi-wajah. Skor kepatuhan pose dan preservasi identitas secara terpisah.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Inpainting | "Fill the hole" | Regenerate inside a mask; keep outside pixels. |
| Outpainting | "Extend the canvas" | Regenerate outside the canvas; keep inside. |
| 9-channel U-Net | "Proper inpainting model" | U-Net with `noisy \| encoded-source \| mask` as input. |
| SDEdit | "Img2img with noise level" | Noise to time `t`, denoise with new prompt. |
| InstructPix2Pix | "Text-only edits" | Fine-tuned diffusion on (image, instruction, output) triples. |
| RePaint | "No retraining" | Re-noise periodically during reverse to reduce seams. |
| SAM | "Segment Anything" | Mask generator by clicks or boxes; pairs with inpaint. |
| Flux-Kontext | "Edit with context" | Flux variant that accepts a reference image + instruction for edits. |

## Catatan produksi: pipeline edit sensitif-latensi

Pengguna yang mengedit gambar mengharapkan round trip sub-5-detik. SDXL-Inpaint 30 langkah pada 1024² adalah 3-4 dtk pada L4, plus pembangkitan mask SAM (~200 ms) dan enkode/decode VAE (~500 ms gabungan). Dalam pembingkaian produksi, ini TTFT-bound alih-alih throughput-bound — batch 1, konkurensi rendah, minimalkan setiap tahap:

- **SAM-H yang lambat.** SAM-H pada 1024² adalah ~200 ms; SAM-ViT-B adalah ~40 ms dengan kehilangan kualitas minor. SAM 2 (video) menambah overhead temporal; jangan pakai untuk edit gambar-tunggal.
- **Lewati enkode bila memungkinkan.** `pipe.image_processor.preprocess(img)` mengenkode ke laten. Kalau kamu punya laten dari pembangkitan sebelumnya (tipikal dalam UI edit-iteratif), teruskan langsung via `latents=...` untuk melewatkan satu enkode VAE.
- **Dilatasi mask penting untuk throughput juga.** Mask kecil berarti sebagian besar forward pass U-Net terbuang (piksel tak-ter-mask di-clamp anyway). `StableDiffusionInpaintPipeline` milik `diffusers` menjalankan U-Net penuh terlepas; hanya varian inpaint-9-channel yang benar mengeksploitasi komputasi ter-mask.
- **Flux-Kontext adalah jawaban 2025.** Satu forward pass atas `(source_image, instruction)` — tanpa mask terpisah, tanpa sweep noise SDEdit. Pada H100 ia mengirim edit dalam ~1.5 dtk. Pelajaran arsitekturalnya: runtuhkan tahap-tahapnya.

## Bacaan Lanjutan

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — inpainting tanpa-training.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — editing instruksi-teks.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, sumber mask.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — SAM video.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — editing level-attention.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — tooling 2024.
