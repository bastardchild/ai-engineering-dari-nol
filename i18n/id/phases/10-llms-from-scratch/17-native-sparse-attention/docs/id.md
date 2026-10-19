# Perhatian Jarang Asli (DeepSeekNSA)

> Pada 64 ribu token, perhatian memakan 70-80% latensi dekode. Setiap laboratorium model terbuka mempunyai rencana untuk memperbaikinya. milik DeepSeekNSA(ACLmakalah terbaik tahun 2025) adalah salah satu yang macet: tiga cabang perhatian paralel — token berbutir kasar terkompresi, token berbutir halus yang dipertahankan secara selektif, dan jendela geser untuk konteks lokal — digabungkan melalui gerbang pembelajaran. Ini selaras dengan perangkat keras (ramah kernel), dapat dilatih secara asli (berfungsi dalam pra-pelatihan, tidak dipasang pada inferensi), dan pada dekode 64k, ini berjalan lebih cepat daripada FlashAttention sambil mencocokkan atau mengalahkan kualitas perhatian penuh. Pelajaran ini membangun tiga cabang secara end-to-end dan menunjukkan mengapa ketersebaran dapat dibedakan secara end-to-end.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Sebutkan ketiganyaNSAcabang perhatian dan apa yang ditangkap masing-masing cabang.
- Jelaskan alasannyaNSAadalah "dapat dilatih secara asli" di mana metode perhatian jarang sebelumnya hanya bersifat inferensi.
- Hitung perhatian, hitung penghematanNSAversus perhatian penuh pada konteks 64k sebagai fungsi ukuran blok kompresi dan pemilihan top-k.
- Terapkan kombinasi tiga cabang di stdlib Python pada urutan sintetis pendek dan verifikasi perilaku bobot gating.

## Masalahnya

Perhatian penuh pada panjang urutan biaya N`O(N^2)`waktu dan`O(N)`Cache KV per lapisan. Pada 64 ribu token, jumlah bandwidth komputasi dan memori sangat besar. Estimasi teoretis yang diukur dariNSAmakalah: perhatian menyumbang 70-80% dari total latensi decode pada 64k. Semuanya di hilir —TTFT,tokens/sec, biaya per juta token — didominasi oleh biaya perhatian.

Sedikit perhatian adalah jawaban yang jelas. Upaya sebelumnya terbagi dalam dua kelompok. Ketersebaran pola tetap (jendela geser, langkah, blok-lokal) membuang informasi dan gagal dalam tugas penarikan kembali jangka panjang. Ketersebaran waktu inferensi (pemangkasan cache KV,H2O, StreamingLLM) diterapkan pada model yang telah dilatih sebelumnya tentang perhatian padat dan hanya memulihkan sebagian kecil dari potensi percepatan karena model tidak pernah diminta untuk mengarahkan informasi melalui pola renggang.

Perhatian Jarang Asli (Yuan et al., DeepSeek +PKU+ UW,ACLMakalah terbaik tahun 2025, arXiv:2502.11089) melakukan keduanya: pola ketersebaran yang dipelajari model selama pra-pelatihan, diimplementasikan sebagai algoritme selaras kernel yang benar-benar memberikan penghematan komputasi pada inferensi. Dua tahun dari sekarang,NSAatau keturunan langsung adalah perhatian default pada setiap model konteks panjang perbatasan.

## Konsepnya

### Tiga cabang paralel

Untuk setiap kueri,NSAmenjalankan perhatian tiga kali, terhadap tiga tampilan berbeda dari cache KV:

1. **Cabang terkompresi.** Token dikelompokkan ke dalam blok berdasarkan ukuran`l`(biasanya 32 atau 64). Setiap blok dikompresi menjadi satu token ringkasan melalui pembelajaran kecilMLP. Kueri menangani token terkompresi ini, mendapatkan gambaran kasar dari keseluruhan urutan.

2. **Cabang yang dipilih.** Menggunakan skor perhatian dari cabang terkompresi, k blok teratas yang paling relevan dengan kueri saat ini diidentifikasi. Token terperinci (tidak terkompresi) dari blok tersebut dibaca dan kueri menangani semuanya. Pikirkan perhatian cabang terkompresi sebagai sinyal perutean untuk pemilihan.

3. **Cabang jendela geser.** Kueri ditujukan ke yang terbaru`W`token (biasanya 512) untuk konteks lokal. Cabang ini menangkap pola jangka pendek yang banyak strukturnya (sintaks, inti referensi lokal) yang mungkin terlewatkan oleh dua cabang lainnya.

Tiga keluaran cabang digabungkan melalui gerbang per posisi yang dipelajari:

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win`adalah bobot gerbang dari yang kecilMLPpada kueri. Jumlahnya tidak harus 1 — mereka dapat menimbang cabang secara mandiri.

### Mengapa ini "dapat dilatih secara asli"

Langkah pemilihan (blok k atas) bersifat diskrit. Operasi diskrit memutus aliran gradien. Pekerjaan dengan perhatian jarang sebelumnya melewatkan backprop melalui seleksi (membatasi pelatihan) atau menggunakan relaksasi terus menerus yang tidak memberikan ketersebaran nyata dalam inferensi.

NSAmenghindari hal ini: perhatian cabang terkompresi ADALAH perhatian kasar yang dapat dibedakan pada keseluruhan rangkaian. Operasi top-k hanya menggunakan kembali skor perhatian tertinggi dari cabang terkompresi untuk memilih blok halus mana yang akan dimuat. Gradien mengalir melalui skor cabang terkompresi (yang memengaruhi keluaran terkompresiANDlogika pemilihan), dan kontribusi blok yang dipilih terhadap hasil akhir juga dapat dibedakan. Yang tidak dapat dibedakan`top_k`Operasi ini tidak boleh dilakukan pada grafik komputasi maju - ini hanya mengontrol blok mana yang dimuat dari memori.

Inilah alasannyaNSAdapat digunakan dalam pra-pelatihan secara menyeluruh. Model tersebut belajar mengarahkan informasi melalui tiga cabang secara bersamaan, menghasilkan pola renggang yang pada inferensi benar-benar memberikan percepatan yang dijanjikan.

### Kernel yang selaras dengan perangkat keras

NSAKernel dirancang untuk modernGPUhierarki memori. Kernel memuat kueri berdasarkanGQAgrup (lingkaran luar), mengambil blok KV renggang yang sesuai per grup (lingkaran dalam), dan menjalankan perhatianSRAM. Karena setiap grup kueri melihat blok terpilih yang sama (pemilihan dilakukan per grup kueri, bukan per kepala kueri), beban KV diamortisasi ke seluruh grup. Intensitas aritmatika tetap tinggi.

Makalah ini melaporkan kernel Triton berjalan 9x lebih cepat daripada FlashAttention pada dekode 64k, dengan rasio kecepatan bertambah seiring dengan panjang urutan. Kernel maju dan mundur keduanya disediakan.

### Anggaran komputasi

Biarkan`N`menjadi panjang urutan,`l`ukuran blok kompresi,`k`jumlah pilihan k teratas,`w`jendela geser,`b`ukuran blok yang dipilih (biasanya sama`l`).

- Cabang terkompresi:`O(N/l)`kunci per permintaan, jadi`O(N * N / l)`jumlah.
- Cabang yang dipilih:`O(k * b)`kunci per permintaan, jadi`O(N * k * b)`.
- Cabang geser:`O(w)`kunci per permintaan, jadi`O(N * w)`.

Jumlah:`O(N * (N/l + k*b + w))`.

Dengan`N = 64k, l = 64, k = 16, b = 64, w = 512`: biaya per permintaan adalah`1000 + 1024 + 512 = 2536 keys`. Perhatian penuh adalah`64000 keys`. Pengurangan komputasi 25x.

Dengan`N = 128k, l = 64, k = 16, b = 64, w = 512`: biaya per permintaan adalah`2000 + 1024 + 512 = 3536 keys`. Perhatian penuh adalah`128000 keys`. Pengurangan 36x. Manfaatnya bertambah seiring dengan panjangnya urutan, yang merupakan inti keseluruhannya.

### Bagaimana perbandingannya

| Method | Differentiable | Real inference speedup | Long-range recall |
|--------|---------------|----------------------|-------------------|
| Sliding window only | yes | yes | fails |
| Strided / block-sparse | yes | yes | partial |
| KV pruning (H2O, StreamingLLM) | N/A (inference-time) | yes | partial |
| MoBA (Moonshot) | partial | yes | good |
| NSA | yes (natively) | yes (9x at 64k) | matches full attention |

MoBA (Moonshot, arXiv:2502.13189) diterbitkan secara bersamaan dan menggunakan pendekatan tiga-lebih-lebih-dari-satu, dengan menerapkan prinsip MoE pada blok perhatian.NSAdan MoBA adalah dua arsitektur yang perlu diketahui untuk pra-pelatihan konteks panjang tahun 2026.

```figure
sliding-window-attention
```

## Bangun Sendiri

`code/main.py`mengimplementasikan tiga cabang pada urutan sintetik pendek dan menunjukkan:

- KompresiMLP(garis dasar mean-pool yang sederhana digunakan untuk kejelasan pedagogi; yang sebenarnyaNSAmenggunakan yang dipelajariMLP).
- Pemilihan blok k teratas didorong oleh skor cabang terkompresi.
- Perhatian jendela geser pada yang terakhir`w`token.
- Kombinasi yang terjaga keamanannya.
- Hasil cetakan hitungan komputasi dibandingkan dengan perhatian penuh.

### Langkah 1: kompres token menjadi blok

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Langkah 2: perhatian cabang terkompresi

Jalankan perhatian softmax dari kueri terhadap kunci terkompresi. Skor cabang terkompresi berfungsi ganda sebagai sinyal untuk pemilihan k teratas.

### Langkah 3: pemilihan blok top-k

Pilih indeks`k`blok terkompresi dengan skor tertinggi. Muat token asli yang tidak terkompresi dari blok tersebut dan perhatikan token tersebut.

### Langkah 4: perhatian jendela geser

Ambil yang terakhir`w`token dan jalankan perhatian standar terhadap mereka.

### Langkah 5: gerbang + gabungan

KecilMLPpada kueri menghasilkan tiga bobot gerbang. Output akhir adalah jumlah tertimbang dari tiga output cabang.

### Langkah 6: menghitung penghitungan

Cetak jumlah kunci yang dihadiri per kueri untuk setiap cabang dan totalnya. Bandingkan dengan`N`(perhatian penuh). Pada sintetis 1024-token dengan`l = 32, k = 4, w = 128`,NSAmelihat`32 + 128 + 128 = 288`kunci per kueri versus 1024 untuk perhatian penuh — 3,5x lebih sedikit.

## Gunakan

NSAdikirimkan melalui jalur pra-pelatihan konteks panjang milik DeepSeek. Status integrasi dalam tumpukan inferensi publik pada April 2026:

- **DeepSeek internal**: penggunaan bobot asli yang dipublikasikanNSAatau penerusnyaDSA(Mencari Lebih Dalam Perhatian Jarang).
- **vLLM**: eksperimentalNSAdukungan dalam pengembangan untuk bobot DeepSeek-V3.x.
- **SGLang**:NSAtolok ukur yang dipublikasikan; jalur produksi mengikuti vLLM.
- **llama.cpp /CPU**: tidak didukung; overhead dekomposisi kernel tidak sepadanCPUkeluaran.

Kapan harus meraihnyaNSA:

- Pra-pelatihan atau pelatihan lanjutan dijalankan dengan menargetkan konteks 64 ribu lebih dengan anggaran komputasi yang besar.
- Kesimpulan dari pos pemeriksaan konteks panjang DeepSeek sendiri. Bobotnya adalahNSA-asli.

Kapan tidak:

- Melayani model terlatih dengan perhatian padat yang sudah ada. kamu tidak dapat melakukan retrofitNSAtanpa pelatihan lanjutan.
- Konteks di bawah 16k. Biaya overhead tiga cabang mendominasi penghematan.
- Obrolan interaktif Batch-1. Manfaat dekode yang sensitif terhadap latensi, tetapi hanya pada konteks yang panjang.

## Rilis

Pelajaran ini menghasilkan`outputs/skill-nsa-integrator.md`. Mengingat spesifikasi proses pra-pelatihan dalam konteks panjang, ini menghasilkanNSArencana integrasi: ukuran blok kompresi, top-k, jendela geser, gerbangMLPlebar, pilihan kernel, dan evaluasi konteks panjang spesifik yang membenarkan perubahan arsitektur.

## Latihan

1. Jalankan`code/main.py`pada sintetis 1024-token. Menyapu`(l, k, w)`di tiga preset dan mencetak jumlah komputasi. Identifikasi preset yang mencapai jumlah kunci terendah per kueri sekaligus mempertahankan 95% ingatan dibandingkan perhatian penuh pada pengujian yang rumit.

2. Ganti kompresor mean-pool dengan pembelajaran kecilMLP(2 lapis, tersembunyi 32). Latihlah pada tugas sintetis yang sinyalnya adalah rata-rata sebuah blok. Ukur kesenjangan kebingungan terhadap garis dasar kumpulan rata-rata pada data yang disimpan.

3. Terapkan gerbangMLP. Dibutuhkan query sebagai input dan output tiga skalar. Tunjukkan bahwa gerbang berperilaku wajar: pembobotan yang hampir seragam pada kueri acak, bobot yang berat pada cabang yang dipilih saat kueri mencapai blok jauh di belakang.

4. Hitung anggaran memori cache KV untukNSA-mengaktifkan model 70B pada konteks 128k. Head KV ada 8, head redup 128,BF16. Bandingkan dengan perhatian penuh danMLA(Fase 10 · 14 ditampilkanMLAnomornya). Identifikasi panjang barisan dimanaNSACache KV cabang yang terperinci sama dengan perhatian penuh.

5. Baca Bagian 4 dariNSAmakalah (arXiv:2502.11089) dan jelaskan dalam tiga kalimat mengapa skor perhatian cabang terkompresi digunakan kembali untuk pemilihan k teratas daripada menghitung skor perutean terpisah. Ikat jawabannya dengan aliran gradien.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Compressed branch | "Coarse view" | Attention over block-averaged keys that provides global context in O(N/l) keys per query |
| Selected branch | "Top-k blocks" | Fine-grained attention over the `k` blocks with highest compressed-branch scores |
| Sliding window | "Local context" | Attention over the last `W` tokens for short-range patterns |
| Native trainability | "Pre-train with the sparsity on" | The sparsity pattern is learned during pre-training, not bolted on at inference |
| Compression block size l | "Group size for coarse view" | How many tokens get merged into one summary; 32-64 typical |
| Top-k | "Blocks to keep" | Number of compressed blocks whose uncompressed tokens get read; 16 typical |
| Sliding window W | "Local attention radius" | Typically 512; shorter hurts local coherence, longer wastes compute |
| Branch gate | "How to mix the three" | Per-position MLP output that weights the three branches' contributions |
| Hardware alignment | "Kernel-friendly sparsity" | Sparse pattern chosen so that the actual GPU kernel achieves the theoretical speedup |
| DSA | "NSA's successor" | Deepseek Sparse Attention, the architecture that followed NSA in DeepSeek's lineage |

## Bacaan Lebih Lanjut

- [Yuan dkk. — Perhatian Jarang Asli: Perhatian Jarang yang Sesuai dengan Perangkat Keras dan Dapat Dilatih Secara Asli (arXiv:2502.11089,ACLMakalah Terbaik 2025)](https://arxiv.org/abs/2502.11089) — kertas
- [Laporan Teknis DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — keluarga arsitekturNSAtarget
- [Moonshot AI — MoBA: Campuran Block Attention untuk LLM Konteks Panjang (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — pekerjaan bersamaan, perhatian gaya MoE pada blok
- [Beltagy dkk. — Longformer: Transformator Dokumen Panjang (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — asal jendela geser
- [Xiao dkk. — StreamingLLM: Model Bahasa Streaming yang Efisien dengan Penyerap Perhatian (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — garis dasar ketersebaran waktu inferensiNSAmembaik
- [Dao dkk. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — garis dasar perhatian penuhNSAkernel mengalahkan pada 64k
