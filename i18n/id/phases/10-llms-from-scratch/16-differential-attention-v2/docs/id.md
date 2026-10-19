# Perhatian Diferensial (V2)

> Perhatian Softmax menyebarkan sejumlah kecil kemungkinan ke setiap token yang tidak cocok. Lebih dari 100 ribu token yang menambah kebisingan dan menenggelamkan sinyal. Transformator Diferensial (Ye et al.,ICLR2025) memperbaikinya dengan menghitung perhatian sebagai selisih dua softmax, mengurangi tingkat kebisingan bersama.DIFFV2 (Microsoft, Januari 2026) adalah penulisan ulang tumpukan produksi: mencocokkan latensi dekode dengan Transformer dasar, tanpa kernel khusus, kompatibel dengan FlashAttention. Pelajaran ini adalah V1 hingga V2 end-to-end, dengan implementasi mainan yang berfungsi dari operasi perbedaan yang dapat kamu jalankan di stdlib Python.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Nyatakan dengan tepat mengapa perhatian softmax memiliki tingkat kebisingan dan mengapa perhatian tersebut berkembang seiring dengan panjangnya konteks.
- Turunkan rumus perhatian diferensial dan jelaskan mengapa pengurangan membatalkan komponen kebisingan bersama sambil mempertahankan sinyal.
- Telusuri perbedaan V1-ke-V2: apa yang menjadi lebih cepat, apa yang menjadi lebih sederhana, apa yang menjadi lebih stabil, dan mengapa setiap perubahan diperlukan untuk pra-pelatihan produksi.
- Terapkan perhatian diferensial dari awal dengan Python murni dan verifikasi secara empiris properti peredam bising pada kueri sinyal-plus-kebisingan sintetis.

## Masalahnya

Perhatian softmax standar memiliki properti matematika yang berubah menjadi sakit kepala operasional dalam skala besar. Untuk sebuah pertanyaan`q`, bobot perhatiannya adalah`softmax(qK^T / sqrt(d))`. Softmax tidak pernah bisa menghasilkan angka nol yang tepat — setiap token yang tidak cocok mendapat massa positif. Massa sisa tersebut adalah kebisingan, dan skalanya bergantung pada panjang konteks. Dengan 128 ribu token, meskipun setiap token yang tidak cocok hanya mendapat probabilitas 0,001%, gabungan 127.999 token tersebut berkontribusi sekitar 12% dari total. Model harus belajar mengatasi dasar kebisingan yang berkembang seiring dengan konteks.

Secara empiris hal ini muncul sebagai gangguan perhatian-kepala: kutipan halusinasi dalam konteks yang panjangRAG, kegagalan yang hilang di tengah-tengah pada tugas pengambilan 100 ribu token, dan penurunan akurasi yang halus pada tolok ukur needle-in-haystack yang melewati 32 ribu. Kertas Transformator Diferensial (arXiv:2410.05258,ICLR2025) mengukur kesenjangannya:DIFFTransformers mencapai tingkat kebingungan yang lebih rendah, akurasi konteks panjang yang lebih tinggi, dan halusinasi yang lebih sedikit dibandingkan garis dasar berukuran sama.

DIFFV1 memiliki tiga masalah yang membuatnya tidak masuk dalam jalur pipa pra-pelatihan. Cache nilainya harus dimuat dua kali per langkah decode, itu memerlukan kustomCUDAkernel yang merusak kompatibilitas FlashAttention, dan RMSNorm per kepala mengganggu kestabilan pelatihan jangka panjang pada skala 70 miliar lebih.DIFFV2 (Blog Microsoft unilm, 20 Januari 2026) memperbaiki ketiganya. Pelajaran ini memandu kedua versi, membuat operator pembeda, dan menentukan tolok ukur peredam bising pada kueri mainan.

## Konsepnya

### Lantai kebisingan softmax

Untuk sebuah pertanyaan`q`dan kunci`K = [k_1, ..., k_N]`, bobot perhatian adalah:

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

Tidak`w_i`selalu nol. Jika`k_i`sama sekali tidak ada hubungannya dengan`q`, skornya`q . k_i`bukan 0 — ia berfluktuasi di sekitar nol dengan varians`||q||^2 / d`. Setelah normalisasi softmax, setiap token yang tidak terkait masih berkontribusi`O(1/N)`ke jumlah tertimbang. Total kontribusi token yang tidak terkait adalah`O((N-1)/N) = O(1)`— bukan jumlah yang sedikit.

Apa yang diinginkan model adalah sesuatu seperti hard top-k: bobot tinggi pada token yang cocok, bobot mendekati nol di tempat lain. Softmax terlalu halus untuk melakukannya secara langsung.

### Ide diferensial

Bagi proyeksi Q dan K masing-masing kepala menjadi dua: Q = (Q_1,Q_2) dan K = (K_1,K_2). Hitung dua peta perhatian:

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

Keluaran:

```
DiffAttn = (A_1 - lambda * A_2) V
```

Pengurangan tersebut membatalkan distribusi kebisingan apa pun yang dimiliki kedua peta. Jika kedua peta memiliki bobot yang kira-kira seragam pada 127 ribu token yang tidak terkait (yang akan terjadi, pada inisialisasi acak), peta tersebut akan dibatalkan. Sinyal - yang mencapai bobot puncak pada beberapa token yang benar-benar relevan - hanya dibatalkan jika muncul di kedua peta dengan magnitudo yang sama, yang tidak akan terjadi setelah model dilatih.

`lambda`adalah skalar yang dapat dipelajari per kepala, diparameterisasi sebagai`lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`. Ini bisa menjadi negatif.`lambda_init`defaultnya adalah angka positif kecil seperti 0,8.

### Mengapa ini cocok dengan peredam bising

Bayangkan dua mikrofon berisik yang merekam suara yang sama. Keduanya menangkap speaker ditambah kebisingan latar belakang yang berkorelasi. Kurangi satu dari yang lain dan kebisingan yang dibagikan akan hilang. Suara tersebut bertahan karena kedua sinyal memiliki perbedaan fase atau amplitudo yang cukup untuk mencegah pembatalan penuh. Per-kepala`lambda`mempelajari dengan tepat keseimbangan ini.

### V1 vs V2: perbedaannya

V1 menjaga jumlah parameter tetap sama dengan Transformer dasar. Untuk mendapatkan dua kueri per kepala, dimensi kepala dibelah dua. Hal ini memerlukan ekspresi kepala dan — yang lebih menyakitkan lagi — mengurangi separuh nilai cache per kepala. Dekode harus memuat cache nilai dua kali per langkah (sekali per cabang softmax). Hasil: decode lebih lambat dari baseline meskipun jumlah parameter cocok.

V2 menggandakan jumlah kepala kueri dan menjaga kepala KV tetap sama (meminjam parameter dari proyeksi ke atas). Dimensi kepala tetap sama dengan garis dasar. Setelah pengurangan, dimensi ekstra diproyeksikan kembali ke bawah agar sesuai dengan garis dasar TransformerO_Wproyeksi. Tiga hal terjadi sekaligus:

1. Kecepatan dekode sesuai dengan kecepatan dasar (cache KV dimuat satu kali).
2. FlashAttention berjalan tidak berubah (tidak ada kernel khusus).
3. Intensitas aritmatika saat dekode meningkat (lebih banyak komputasi per byte yang dimuatHBM).

V2 juga menghapus RMSNorm per kepala yang digunakan V1 untuk menstabilkan pengurangan. Pada skala pra-pelatihan kelas 70B, RMSNorm tersebut mengganggu kestabilan pelatihan yang terlambat. V2 menggantikannya dengan skema inisialisasi sederhana yang menjaga pelatihan tetap stabil tanpa modul tambahan.

### Kapan harus meraihnya

| Workload | Benefit |
|----------|---------|
| Long-context RAG (64k+) | Cleaner attention maps, fewer hallucinated citations |
| Needle-in-haystack benchmarks | Substantial accuracy lift past 32k |
| Multi-document QA | Less cross-document interference |
| Code completion at 8k | Marginal, not worth the architecture change |
| Short chat (< 4k) | Essentially indistinguishable from baseline |

Nilainya bertambah seiring dengan panjangnya konteks. Pada token 4k, tingkat kebisingannya cukup kecil sehingga perhatian standar baik-baik saja. Pada 128k itu merugikan kamu.

### Bagaimana susunannya dengan kenop 2026 lainnya

| Feature | Compatible with DIFF V2? |
|---------|------------------------|
| GQA | Yes (V2 increases Q heads, not KV heads) |
| MLA (DeepSeek) | Yes in principle, no published paper combining them |
| MoE | Yes (attention is independent of MLP block) |
| RoPE | Yes (unchanged) |
| YaRN / long-context scaling | Yes (exactly where DIFF helps most) |
| FlashAttention | Yes in V2 (was no in V1) |
| Speculative decoding | Yes (attention change is invisible to the spec-decode loop) |

```figure
differential-attention
```

## Bangun Sendiri

`code/main.py`mengimplementasikan perhatian diferensial dengan Python murni. Kueri mainan dengan struktur sinyal-plus-kebisingan yang diketahui memungkinkan kamu mengukur rasio peredam bising secara langsung.

### Langkah 1: perhatian softmax standar

Operasi matriks Stdlib: daftar daftar, matmul manual, softmax dengan pengurangan stabilitas numerik maks.

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Langkah 2: membagi Q, K menjadi dua bagian

Gaya V1: membagi dua dimensi kepala. Gaya V2: pertahankan dimensi kepala dan gandakan jumlah kepala. Implementasi mainan menggunakan V1 untuk kejelasan pedagogis — matematikanya sama, hanya pembukuannya saja yang berbeda.

### Langkah 3: dua cabang softmax + pengurangan

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

Catatan: bobot keluaran bisa negatif. Tidak apa-apa — cache nilai masih menangani kontribusi yang ditandatangani. Proyeksi V berikutnya menyerap tanda tersebut.

### Langkah 4: pengukuran pembatalan kebisingan

Buatlah rangkaian sintetis dengan panjang 1024. Tempatkan token sinyal pada posisi yang diketahui, isi sisanya dengan noise. Hitung (a) bobot perhatian softmax standar pada posisi sinyal dan (b) bobot perhatian diferensial. Ukur rasio sinyal terhadap kebisingan di masing-masingnya.DIFFperhatian secara andal menghasilkan rasio signal-to-noise yang lebih tinggi dengan faktor 3x-10x tergantung pada seberapa besar perbedaan kedua cabang yang telah dilatih.

### Langkah 5: Akuntansi parameter V1 vs V2

Diberikan konfigurasi (tersembunyi=4096, kepala=32, d_head=128), cetak:

- Transformator Dasar: Q, K, V setiap ukuran`hidden * hidden`,MLPdi 4 * tersembunyi.
- DIFFV1: Q, K setiap ukuran`hidden * hidden`, ukuran V`hidden * hidden`(tidak berubah), kepala redup dibelah dua secara internal. Menambahkan per kepala`lambda`parameter (O(kepala * d_head)).
- DIFFV2: ukuran Q`2 * hidden * hidden`, ukuran K`hidden * hidden`, ukuran V`hidden * hidden`. Ekstra redup diproyeksikan kembali ke bawah sebelumnyaO_W. Menambahkan sama`lambda`parameter.

Mainan tersebut mengukur biaya parameter tambahan untuk V2 (kira-kira`hidden * hidden`ekstra per blok perhatian) dan mencetaknya.

## Gunakan

DIFFV2 belum dikirimkan di setiap server inferensi produksi pada April 2026, namun integrasi sedang berlangsung di vLLM dan SGLang. Sementara itu polanya muncul pada:

- Model produksi konteks panjang internal Microsoft.
- Replikasi penelitian di beberapa pelatihan model terbuka berjalan dengan target konteks lebih dari 256 ribu.
- Arsitektur hybrid yang menggabungkanDIFFperhatian dengan perhatian jendela geser pada lapisan alternatif.

Kapan kamu akan mencapai ini pada tahun 2026:

- Melatih model baru dari awal dengan menargetkan 64 ribu lebih konteks efektif. Tambahkan perhatian yang berbeda dari awal; pelatihan ulang nanti mahal.
- Menyempurnakan model konteks panjang di mana kegagalan yang terjadi di tengah-tengah mendominasi evaluasi kamu. LoRA pada proyeksi Q dapat memperkirakanDIFFstruktur.

Bila kamu tidak mau:

- kamu menayangkan model padat terlatih dengan performa konteks panjang yang stabil. Biaya pelatihan ulang jarang sebanding dengan beban yang ada.
- Konteks kamu selalu di bawah 16k. Lantai kebisingan dapat diabaikan.

## Rilis

Pelajaran ini menghasilkan`outputs/skill-diff-attention-integrator.md`. Dengan adanya arsitektur model, panjang konteks target, profil halusinasi, dan anggaran pelatihan, hal ini menghasilkan rencana integrasi untuk menambahkan perhatian yang berbeda pada proses pra-pelatihan baru atau penyempurnaan LoRA.

## Latihan

1. Jalankan`code/main.py`. Pastikan rasio sinyal terhadap kebisingan yang dilaporkan untuk perhatian diferensial lebih tinggi dari perhatian softmax standar pada kueri sintetis. Variasikan amplitudo kebisingan dan tunjukkan titik persilangan di mana perhatian standar menjadi tidak dapat digunakan.

2. Hitung delta jumlah parameter dari garis dasar keDIFFV1 dan dari awal hinggaDIFFV2 untuk model kelas 7B (tersembunyi=4096, kepala=32, d_head=128, 32 lapisan). Tunjukkan komponen mana yang memperoleh parameter dan mana yang tetap sama.

3. Baca Bagian 3 dariDIFFMakalah V1 (arXiv:2410.05258) dan Bagian 2 dariDIFFV2 Blog Memeluk Wajah. Dalam dua kalimat, jelaskan mengapa RMSNorm per kepala V1 diperlukan dan mengapa V2 dapat menghapusnya tanpa menyebabkan perbedaan pelatihan.

4. Menerapkan ablasi: hitung perhatian diferensial dengan`lambda = 0`(softmax murni pertama) dan`lambda = 1`(pengurangan penuh). Pada kueri sintetis, ukur bagaimana perubahan signal-to-noise di seluruh sapuan. Identifikasi`lambda`yang memaksimalkan signal-to-noise.

5. Perluas mainan itu keGQA+DIFFV2. Pilih 8 kepala KV dan 32 kepala Q. Tunjukkan bahwa ukuran cache KV cocok dengan garis dasarGQAmodel dengan konfigurasi (8, 32) yang sama.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Differential attention | "Two softmaxes minus each other" | Split Q, K into two halves, compute two softmax maps, subtract the second (scaled by lambda) from the first, then multiply by V |
| Noise floor | "The non-zero tail of softmax" | The O(1/N) weight softmax puts on every unrelated token, which sums to O(1) across long contexts |
| lambda | "The subtraction scale" | Per-head learnable scalar parameterized as `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`; can be negative |
| DIFF V1 | "The ICLR 2025 version" | Original Differential Transformer; halves head dim to preserve parameter count, needs custom kernel, slower decode |
| DIFF V2 | "The January 2026 fix" | Doubles Q heads keeping KV heads; matches baseline decode speed and works with FlashAttention |
| Per-head RMSNorm | "The V1 stabilizer" | Extra norm V1 applied after the difference; V2 removed it to prevent late-training instability |
| Signal-to-noise ratio | "How much attention is wasted" | Ratio of weight on the true signal position to average weight on unrelated positions |
| Lost in the middle | "Long-context failure mode" | Empirical phenomenon where retrieval accuracy dips for documents in the middle of a long context — DIFF attention reduces this |
| Arithmetic intensity | "FLOPs per byte loaded" | Ratio V2 increased at decode by doubling queries per KV load; important for memory-bound decode |

## Bacaan Lebih Lanjut

- [Kamu dkk. — Transformator Diferensial (arXiv:2410.05258,ICLR2025)](https://arxiv.org/abs/2410.05258) — makalah asli dengan teori peredam bising dan ablasi konteks panjang
- [Microsoft unilm — Differential Transformer V2 (blog Hugging Face, Januari 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — penulisan ulang tumpukan produksi, dekode dasar yang cocok, kompatibel dengan FlashAttention
- [Memahami Perhatian Diri yang Telah Dilatih Diferensial Transformator (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — analisis teoretis tentang mengapa pengurangan memulihkan struktur perhatian yang telah dilatih sebelumnya
- [DibagikanDIFFTransformator (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — varian berbagi parameter
- [Vaswani dkk. — Hanya Perhatian yang kamu Butuhkan (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — Transformator dasarDIFFmengurangi dari
- [Liu dkk. — Tersesat di Tengah (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — tolok ukur konteks panjangDIFFsasaran perhatian
