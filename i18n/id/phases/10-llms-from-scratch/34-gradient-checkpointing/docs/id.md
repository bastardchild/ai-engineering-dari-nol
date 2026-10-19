# Pemeriksaan Gradien dan Perhitungan Ulang Aktivasi

> Backprop menyimpan setiap aktivasi perantara. Pada parameter 70B dan konteks 128K, itu berarti 3 TB aktivasi per peringkat. Pos pemeriksaan memperdagangkan FLOP untuk memori: menghitung ulang alih-alih menyimpan. Pertanyaannya adalah segmen mana yang harus dihilangkan, dan jawabannya bukanlah “semuanya”.

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## Masalahnya

Pelatihan transformator menyimpan, untuk setiap lapisan, masukan untuk setiap operasi yang dibedakan ke belakang: masukan perhatian, masukanQ/K/Vproyeksi, keluaran softmax,FFNmasukan, keluaran norma, dan aliran sisa. Untuk lapisan dengan ukuran tersembunyi`d`, panjang urutan`L`, kumpulan`B`, ini sesuai urutan`12 * B * L * d`mengapung per lapisan.

Untuk`d=8192, L=8192, B=1`, itu 800MB/layerdi dalamBF16. Model 64 lapisan memerlukan aktivasi sebesar 51 GB — dan itu sebelum kamu mengalikannya dengan ukuran microbatch, sebelum kamu menambahkan perantara perhatian-softmax (`L^2`per kepala), dan sebelum kamu memfaktorkan salinan parsial tensor-paralel.

Tagihan dua sisi:BF16bobot plus status pengoptimal mungkin muat dalam 80 GB, namun aktivasi membuat kamu kewalahan. Pos pemeriksaan gradien (alias penghitungan ulang aktivasi) adalah perbaikan standar. Hilangkan sebagian besar aktivasi; ulangi maju selama mundur untuk mendapatkannya kembali. Biaya: FLOP tambahan. Manfaat: memori turun sebesar rasio segmen pos pemeriksaan terhadap total lapisan.

Jika dilakukan secara naif, pos pemeriksaan membutuhkan biaya FLOP forward-pass sekitar 33% lebih banyak per langkah. Dilakukan dengan baik - pos pemeriksaan selektif sesuai dengan "pilihan cerdas" Korthikanti dkk. — kamu menghemat memori 5x dengan harga di bawah 5%FLOPdi atas kepala. Dan denganFP8matmul,FSDPoffload, dan MoE paralel ahli ini sangat penting: kamu tidak mampu membeli memori atau komputasi yang terbuang.

## Konsepnya

### Apa yang Sebenarnya Dibutuhkan Mundur

`output = layer(input)`. Keinginan mundur`grad_input`dan`grad_params`. Untuk menghitungnya diperlukan:

- `input`(untuk menghitung`grad_params = input.T @ grad_output`untuk lapisan linier)
- beberapa zat antara turunan aktivasi (turunan dariReLU/GELU/softmaxtergantung pada nilai aktivasi)

Forward pass menyimpannya secara otomatis di grafik autograd. Setiap`tensor.retain_grad()`dan setiap operasi yang membutuhkan masukannya tetap memiliki referensi.

### Pos Pemeriksaan Penuh yang Naif

Bagi jaringan menjadi`N`segmen. Selama penerusan, simpan hanya *input* ke setiap segmen. Ketika ke belakang membutuhkan zat perantara, jalankan kembali umpan ke depan segmen tersebut untuk mewujudkannya, lalu bedakan.

Contoh: Trafo 32 lapis dipecah menjadi 32 segmen yang masing-masing 1 lapis.

- Memori: 32 input lapisan (kecil) vs 32 * (volume aktivasi per lapisan) (besar).
- Komputasi ekstra: 1 maju ekstra per segmen, yaitu, ~33% lebih banyak total FLOP maju (karena mundur adalah 2x maju, langkah penuh menjadi 1 + 1 + 2 = 4 unit, bukan 1 + 2 = 3).

Ini adalah Chen dkk yang asli. Resep 2016: setiap pos pemeriksaan`sqrt(L)`lapisan untuk menyeimbangkan memori dan komputasi. Untuk L=64, itu berarti 8 pos pemeriksaan.

### Pos Pemeriksaan Selektif (Korthikanti 2022)

Tidak semua aktivasi dikenakan biaya yang sama. Output softmax perhatiannya adalah`B*L*L*heads`dan tumbuh *secara kuadrat* dengan panjang urutan. ItuFFNaktivasi tersembunyi adalah`B*L*4d`dan tumbuh secara linear. Untuk urutan yang panjang softmax mendominasi.

Pos pemeriksaan selektif menyimpan aktivasi yang murah (proyeksi linier, residu) dan hanya menghitung ulang aktivasi yang mahal (perhatian). kamu membayar FLOP minimal untuk menghitung ulang tetapi menghemat memori O(L^2).

Megatron-Core mengimplementasikan ini sebagai penghitungan ulang aktivasi "selektif". Digunakan di sebagian besar latihan perbatasan tahun 2024+.

### Pembongkaran

Alternatif untuk menghitung ulang: mengirimkan aktivasi keCPU RAMantara maju dan mundur. Membutuhkan bandwidth PCIe; bermanfaat ketika bandwidth menganggur melebihi biaya rematerialisasi. Strategi campuran adalah hal yang umum: memeriksa beberapa lapisan, membongkar yang lain.

FSDP2mengirimkan pembongkaran sebagai opsi kelas satu. Offload bersinar ketikaGPUmengalami hambatan pada memori tetapiCPU-GPUtransfer memiliki ruang kepala.

### Hitung Ulang Model Biaya

FLOP per langkah dengan setiap pos pemeriksaan yang naif`k`lapisan keluar`L`:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Dengan pos pemeriksaan selektif kamu hanya menghitung ulang kernel perhatian, bukan seluruh lapisan:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Model Penghematan Memori

Volume aktivasi per lapisan:`A`. Untuk`L`lapisan, total memori aktivasi:`L * A`.

Pos pemeriksaan penuh (ukuran segmen 1): hanya toko`L * input_volume`(~`L * 1/10 A`untuk trafo standar). Menghemat ~`9 * L * A * 1/10`.

Pos pemeriksaan setiap`k`lapisan: simpan`L/k * A`ditambah`k-1`nilai lapisan dalam segmen aktif.

Di`k = sqrt(L)`, memori dan perhitungan ulang biaya keduanya berskala`sqrt(L)`— tradeoff optimal untuk lapisan biaya seragam.

### Kapan Tidak ke Pos Pemeriksaan

- Lapisan paling dalam dari tahap saluran pipa sudah dalam penerbangan. Bagaimanapun mereka harus menyelesaikannya.
- Lapisan pertama dan terakhir mendominasi komputasi panggung (jarang terjadi pada transformator).
- Kernel perhatian sudah menggunakan FlashAttention — Flash sudah menghitung ulang softmax dengan cepat, sehingga pos pemeriksaan tingkat lapisan tambahan menambahkan sedikit tambahan.

### Pola Implementasi

1. **Pembungkus fungsi:** membungkus segmen`torch.utils.checkpoint.checkpoint(fn, input)`. Hanya toko PyTorch`input`, menghitung ulang semuanya secara terbalik.

2. **Berbasis dekorator:** memberi label lapisan sebagai titik pemeriksaan; pelatih memutuskan pada waktu konfigurasi segmen mana yang akan dibungkus.

3. **Penghitungan ulang eksplisit secara manual:** tulis sendiri backward passnya, panggil custom`recompute_forward`yang menduplikasi penerusan dengan input yang disimpan.

Ketiganya memberikan hasil fungsional yang sama. Pembungkus adalah idiom standar.

### Interaksi dengan TP/PP/FP8

- **Tensor paralel:** input pos pemeriksaan harus dikumpulkan atau disebarkan ulang saat dihitung ulang; menangani biaya komunikasi.
- **Pipeline parallel:** Pola umumnya adalah memeriksa maju setiap tahap pipeline sehingga microbatch urutan terbalik dapat menggunakan kembali memori aktivasi.
- **FP8menghitung ulang:** riwayat amax yang diperbarui selama penghitungan ulang harus cocok dengan penerus asli, atauFP8pergeseran skala. Sebagian besar kerangka kerja menggambarkan skalanya.

```figure
activation-recompute
```

## Bangun Sendiri

### Langkah 1: Model Mainan Dengan Segmen

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Langkah 2: Naif Mundur Membutuhkan Semua Aktivasi

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Langkah 3: Memori Pos Pemeriksaan-Setiap-k

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Langkah 4: Model Biaya

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Langkah 5: Penaksir Memori

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Langkah 6: Ukuran Segmen Optimal

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Langkah 7: Keputusan Pos Pemeriksaan Selektif

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Gunakan

- **torch.utils.pos pemeriksaan**:`from torch.utils.checkpoint import checkpoint`— pembungkus kanonik di PyTorch. Membungkus suatu fungsi; hanya menyimpan input, menghitung ulang secara mundur.
- **Penghitungan ulang aktivasi Megatron-Core**: mendukung`selective`,`full`, dan`block`mode. Standar pada pelatihan perbatasan tahun 2024+.
- **FSDP2pembongkaran**:`module.to_empty(device="cpu")`dengan`offload_policy`di dalamFSDP2aktivasi pecahan keCPUalih-alih menghitung ulang.
- **Pembongkaran ZeRO Kecepatan Dalam**:CPUpembongkaran untuk status pengoptimal dan aktivasi, melengkapi pos pemeriksaan.

## Rilis

Pelajaran ini menghasilkan`outputs/prompt-activation-recompute-policy.md`— prompt yang mengambil konfigurasi model kamu (lapisan, tersembunyi, seq, batch) dan tersediaGPUmemori dan mengeluarkan kebijakan penghitungan ulang per lapisan (tidak ada/selektif/penuh/offload).

## Latihan

1. Verifikasi kebenarannya. Jalankan`model_forward`+`model_backward`(aktivasi penuh) vs`model_forward_checkpointed`+`model_backward_checkpointed`(segmen). Gradien parameter harus identik dengan presisi mesin.

2. Ukuran segmen sapuan`k`dari 1 sampai`L`. MerencanakanFLOPoverhead dan memori. Temukan lutut kurva.

3. Terapkan pos pemeriksaan selektif: simpan masukan modul perhatian tetapi bukan perantaranya. UkurFLOPpos pemeriksaan overhead vs lapisan penuh untuk model 32 lapisan di seq=8192.

4. Tambahkan pembongkaran. Simpan masukan segmen ke simulasi "CPUbuffer" (daftar terpisah). Ukur "bandwidth PCIe" sebagaibytes/timedan temukan titik impas antara pembongkaran dan penghitungan ulang.

5. Tolok ukur trafo PyTorch asli dengan dan tanpa`torch.utils.checkpoint`. Ukur memori (melalui`torch.cuda.max_memory_allocated`) dan waktu langkah.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Gradient checkpointing | "Save memory by redoing forward" | Store segment inputs only; recompute intermediates during backward to get gradient-support tensors |
| Activation recomputation | "Same as checkpointing" | The HPC-flavored name for the same technique |
| Segment size (k) | "How many layers per checkpoint" | Number of layers whose intermediates are dropped and rematerialized together |
| Selective checkpointing | "Korthikanti's trick" | Recompute only expensive-to-store activations (attention softmax); keep cheap ones |
| Full checkpointing | "The naive version" | Recompute every layer's intermediates in every segment |
| Block checkpointing | "Coarse-grained" | Checkpoint whole transformer blocks; largest granularity |
| FLOP overhead | "The compute tax" | Extra FLOPs per step = (recompute FLOPs) / (fwd + bwd FLOPs); 33% naive, 5% selective |
| Activation offload | "Ship to CPU" | Move activations to CPU RAM across forward->backward; alternative to recompute |
| sqrt-L rule | "The classical optimum" | For uniform-cost layers, optimal checkpoint spacing is sqrt(L) layers |
| Attention-softmax volume | "The O(L^2) problem" | L^2 * heads * batch floats; dominates activation memory at long contexts |

## Bacaan Lebih Lanjut

- [Chen dkk., 2016 -- "Melatih Jaringan Dalam dengan Biaya Memori Sublinear"](https://arxiv.org/abs/1604.06174) -- makalah asli yang memformalkan pos pemeriksaan gradien
- [Korthikanti dkk., 2022 -- "Mengurangi Perhitungan Ulang Aktivasi pada Model Transformator Besar"](https://arxiv.org/abs/2205.05198) -- perhitungan ulang aktivasi selektif dan analisis biaya formal
- [Pudipeddi et al., 2020 -- "Melatih Jaringan Neural Besar dengan Memori Konstan menggunakan Algoritma Eksekusi Baru"](https://arxiv.org/abs/2002.05645) -- pendekatan memori konstan alternatif melalui rematerialisasi mode terbalik
- [Ren et al., 2021 -- "ZeRO-Offload: Demokratisasi Pelatihan Model Berskala Miliar"](https://arxiv.org/abs/2101.06840) -- aktivasi offload dalam skala besar
- [dokumen PyTorch torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html) -- standarAPI
- [Dokumentasi penghitungan ulang aktivasi Jembatan Megatron](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html) -- mode selektif, penuh, dan blok
