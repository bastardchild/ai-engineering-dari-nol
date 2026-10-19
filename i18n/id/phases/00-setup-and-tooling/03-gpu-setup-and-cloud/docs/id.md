# Setup GPU & Cloud

> Training di CPU cukup untuk belajar. Training yang sesungguhnya butuh GPU.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## Tujuan Pembelajaran

- Verifikasi ketersediaan GPU lokal memakai `nvidia-smi` dan API CUDA PyTorch
- Konfigurasikan Google Colab dengan GPU T4 untuk eksperimen gratis berbasis cloud
- Benchmark perkalian matriks di CPU vs GPU dan ukur percepatannya
- Estimasi model terbesar yang muat di VRAM-mu memakai aturan praktis fp16

## Masalahnya

Kebanyakan pelajaran di fase 1-3 berjalan baik di CPU. Tapi begitu kamu mulai melatih CNN, Transformer, atau LLM (fase 4+), kamu butuh akselerasi GPU. Training yang memakan 8 jam di CPU hanya memakan 10 menit di GPU.

Kamu punya tiga opsi: GPU lokal, GPU cloud, atau Google Colab (gratis).

## Konsepnya

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## Bangun Sendiri

### Opsi 1: GPU NVIDIA lokal

Cek apakah kamu punya:

```bash
nvidia-smi
```

Instal PyTorch dengan CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Opsi 2: Google Colab

1. Buka [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. Jalankan `!nvidia-smi` untuk verifikasi

Unggah notebook dari kursus ini langsung ke Colab.

### Opsi 3: GPU Cloud

Untuk Lambda Labs, RunPod, atau Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Tidak punya GPU? Tidak masalah.

Kebanyakan pelajaran berjalan di CPU. Pelajaran yang butuh GPU akan mengatakannya dan menyertakan link Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Bangun Sendiri: benchmark GPU vs CPU

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Latihan

1. Jalankan benchmark di atas dan bandingkan waktu CPU vs GPU
2. Kalau kamu tidak punya GPU, jalankan di Google Colab dan bandingkan
3. Cek berapa memori GPU yang kamu punya dan estimasi model terbesar yang bisa muat (aturan praktis: 2 byte per parameter untuk fp16)

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform that lets you run code on the GPU |
| VRAM | "GPU memory" | Video RAM on the GPU, separate from system RAM. Limits model size. |
| fp16 | "Half precision" | 16-bit floating point, uses half the memory of fp32 with minimal accuracy loss |
| Tensor Core | "Fast matrix hardware" | Specialized GPU cores for matrix multiplication, 4-8x faster than regular cores |
