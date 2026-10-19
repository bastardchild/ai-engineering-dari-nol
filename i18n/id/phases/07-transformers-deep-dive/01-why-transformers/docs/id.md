# Mengapa Transformer — Masalah-Masalah pada RNN

> RNN memproses token satu per satu. Transformer memproses semua token sekaligus. Satu taruhan arsitektural itu mengubah setiap kurva scaling dalam deep learning setelah 2017.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## Masalahnya

Sebelum 2017, setiap model sequence state-of-the-art di planet ini — bahasa, terjemahan, speech — adalah recurrent neural network. LSTM dan GRU memenangkan benchmark terjemahan setara ImageNet selama setengah dekade. Mereka adalah satu-satunya tool yang dimiliki siapa pun.

Mereka punya tiga kelemahan fatal. Komputasi sekuensial berarti kamu tidak bisa memparalelkan sepanjang sumbu waktu: token `t+1` membutuhkan hidden state dari token `t`. Sequence 1.024 token berarti 1.024 langkah serial pada GPU yang bisa melakukan 1.000.000 operasi floating-point per siklus. Waktu wall-clock training berskala linear terhadap panjang sequence pada hardware yang dirancang untuk paralelisme.

Vanishing gradient berarti informasi 50 token ke belakang sudah dikompresi melalui 50 non-linearitas. Gated recurrent unit (LSTM, GRU) melunakkan himpitan itu tapi tidak pernah menghilangkannya. Dependensi jarak jauh — "the book I read last summer on a plane to Kyoto was…" — rutin gagal.

Fixed-width hidden state berarti encoder memeras seluruh source sequence menjadi satu vektor sebelum decoder melihat apa pun. Tidak peduli source-nya 5 token atau 500; bottleneck-nya punya shape yang sama.

Makalah 2017 "Attention Is All You Need" mengusulkan sesuatu yang radikal: buang recurrence sepenuhnya. Biarkan setiap posisi attend ke setiap posisi lain secara paralel. Latih dalam satu perkalian matriks besar alih-alih 1.024 perkalian sekuensial.

Hasilnya mendominasi setiap modalitas pada 2026. Bahasa (GPT-5, Claude 4, Llama 4), vision (ViT, DINOv2, SAM 3), audio (Whisper), biologi (AlphaFold 3), robotika (RT-2). Block yang sama, input berbeda.

## Konsepnya

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence sebagai bottleneck.** RNN menghitung `h_t = f(h_{t-1}, x_t)`. Setiap langkah bergantung pada langkah sebelumnya. Kamu tidak bisa menghitung `h_5` sebelum `h_4`. Pada GPU modern dengan 10.000+ core paralel, ini menyia-nyiakan 99% silikon pada sequence panjang.

**Attention sebagai broadcast.** Self-attention menghitung `output_i = sum_j(a_ij * v_j)` untuk setiap pasangan `(i, j)` secara simultan. Seluruh matriks attention N×N terisi dalam satu batched matmul. Tidak ada langkah yang bergantung pada langkah lain. GPU menyukainya.

**Kecepatannya bukan konstanta.** Ini adalah perbedaan antara serial depth `O(N)` dan serial depth `O(1)`. Dalam praktiknya, transformer berlatih 5–10× lebih cepat per epoch pada hardware yang sebanding pada N=512, dan kesenjangannya melebar seiring panjang sequence sampai kamu menabrak memory wall `O(N²)` dari attention (yang kemudian diperbaiki oleh Flash Attention — lihat Pelajaran 12).

**Harga yang dibayar transformer.** Memori attention berskala `O(N²)`. Untuk konteks 2K, baik-baik saja. Untuk konteks 128K, kamu butuh sliding window, ekstrapolasi RoPE, tiling Flash Attention, atau varian linear attention. Recurrence itu `O(N)` baik dalam waktu maupun memori; transformer menukar waktu dengan memori lalu memenangkan kembali waktunya lewat paralelisme.

**Pergeseran inductive bias.** RNN mengasumsikan lokalitas dan recency. Transformer tidak mengasumsikan apa pun — setiap pasangan adalah kandidat untuk attention. Itulah mengapa transformer butuh lebih banyak data untuk berlatih dengan baik tapi berskala lebih jauh begitu memilikinya. Chinchilla (2022) memformalkannya: dengan token yang cukup, transformer selalu mengalahkan RNN dengan jumlah parameter yang sama.

```figure
rnn-vs-parallel
```

## Bangun Sendiri

Tidak ada neural network di sini — kita mensimulasikan bottleneck inti secara numerik supaya kamu merasakan kesenjangannya di laptopmu.

### Langkah 1: ukur serial depth

Lihat `code/main.py`. Kita membangun dua fungsi. Satu mengenkode sequence sebagai rantai penjumlahan (serial, seperti RNN). Satu mengenkodenya sebagai reduksi paralel (broadcast, seperti attention). Matematika yang sama, dependency graph berbeda.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Kita mengukur waktu keduanya pada sequence sampai 100.000 elemen. Versi RNN adalah O(N) dan satu pipeline CPU. Bahkan dalam Python murni, reduksi ala attention mengalahkannya pada panjang ≥ 1.000 karena `sum()` Python diimplementasikan dalam C dan beriterasi tanpa overhead interpreter per langkah.

### Langkah 2: hitung operasi teoretis

Kedua algoritma melakukan N penjumlahan. Perbedaannya adalah *dependency depth*: berapa banyak operasi yang harus terjadi berurutan sebelum operasi berikutnya bisa mulai. Depth RNN = N. Depth attention = log(N) dengan reduksi tree, atau 1 dengan parallel scan. Depth, bukan jumlah op, yang menentukan waktu GPU.

### Langkah 3: scaling empiris pada sequence panjang

Kita mencetak tabel pewaktuan yang membuat kesenjangan O(N) terlihat. Pada laptop Mac 2026, sequence di bawah 1.000 elemen terlalu cepat untuk diukur. Sequence 100.000 menunjukkan linear scan yang bersih. Skalakan itu ke transformer 16.384 token dengan LSTM 12 lapis yang setara dan kamu akan paham mengapa waktu wall-clock training menjadi penghambat pada 2016.

## Gunakan

Kapan masih memilih RNN pada 2026:

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

State-space model (SSM) seperti Mamba pada dasarnya adalah RNN dengan parameterisasi terstruktur yang memberi mereka yang terbaik dari keduanya: memori scan `O(N)`, training paralel via selective scan. Mereka memulihkan 90% kualitas transformer dengan scaling konteks panjang yang lebih baik. Pada 2026 sebagian besar lab terdepan melatih model hibrida SSM+transformer (mis. Jamba, Samba) — recurrence tidak mati, ia menjadi komponen.

## Rilis

Lihat `outputs/skill-architecture-picker.md`. Skill ini memilih arsitektur untuk problem sequence baru berdasarkan batasan panjang, throughput, dan budget training. Ia harus selalu menolak merekomendasikan RNN murni untuk run training di atas 1B token tanpa menyatakan trade-off-nya.

## Latihan

1. **Mudah.** Ambil `rnn_style` dari `code/main.py` dan ganti scalar hidden state dengan vektor hidden state sepanjang 64. Ukur ulang. Seberapa besar overhead serial tumbuh terhadap dimensi hidden state?
2. **Sedang.** Implementasikan prefix-sum paralel (Hillis-Steele scan) dalam Python murni. Verifikasi bahwa ia menghasilkan output numerik yang sama dengan serial scan pada panjang 1024. Hitung depth-nya.
3. **Sulit.** Port reduksi ala attention ke PyTorch di GPU. Ukur waktu keduanya selagi kamu menyapu panjang sequence dari 64 ke 65.536. Plot dan jelaskan bentuk kurvanya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | Computation where step `t` depends on step `t-1`, forcing serial execution along the time axis. |
| Serial depth | "How deep the graph is" | Longest chain of dependent ops; bounds wall-clock even on infinite hardware. |
| Attention | "Let tokens look at each other" | Weighted sum `sum_j a_ij v_j` where `a_ij` comes from a similarity score between positions i and j. |
| Context window | "How much the model sees" | Number of positions an attention layer can take as input; quadratic memory cost scales here. |
| Inductive bias | "Assumptions baked into the architecture" | Prior about what the data looks like; CNNs assume translation invariance, RNNs assume recency. |
| State-space model | "RNN with algebra behind it" | Recurrence parameterized for parallel training via structured state-space matrices. |
| Quadratic bottleneck | "Why context costs so much" | Attention memory = `O(N²)` in sequence length; Flash Attention hides the constants, not the scaling. |

## Bacaan Lanjutan

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — makalah yang mengakhiri recurrence di NLP arus utama.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — tempat attention lahir, ditempelkan pada RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — makalah LSTM orisinal, sebagai catatan.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — jawaban recurrent modern terhadap transformer.
