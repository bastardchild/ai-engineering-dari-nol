# Prediksi Multi-Token (MTP)

> Setiap autoregresifLLMdariGPT-2 ke Llama 3 berlatih dengan satu kerugian per posisi: prediksi token berikutnya. DeepSeek-V3 menambahkan kerugian kedua per posisi: prediksi token setelah itu. Parameter tambahan 14B (pada model 671B) disaring kembali ke model utama melalui aliran gradien, dan yang dilatihMTPkepala digunakan kembali pada inferensi sebagai perancang penguraian kode spekulatif dengan penerimaan 80%+. Throughput generasi 1,8× hadir secara gratis. Pelajaran ini membangun yang berurutanMTPmodul dari laporan teknis DeepSeek, menghitung kerugian dan tata letak parameter shared-head, dan menjelaskan alasannyaMTPmempertahankan rantai sebab akibat sementara paralel asli Gloeckle dkkMTPmemecahkannya.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pre-training a mini GPT), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- NyatakanMTPtujuan pelatihan dan mendapatkan kerugian gabungan di seluruh kedalaman prediksi.
- Jelaskan perbedaan antara paralel Gloeckle dkkMTPkepala (2024) dan sekuensial DeepSeek-V3MTPmodul dan mengapa desain sekuensial mempertahankan rantai sebab akibat.
- Hitung parameter dan overhead memori penambahanMTPmodul untuk menjalankan pra-pelatihan.
- Terapkan satuMTPmodul dari awal: penyematan bersama, blok transformator per kedalaman, proyeksi, dan kepala keluaran bersama.

## Masalahnya

Prediksi token berikutnya adalah standarnyaLLMtujuan pelatihan. Setiap keadaan tersembunyi diawasi untuk memprediksi satu hal: token berikutnya. Ini adalah sinyal yang sangat lemah. Sebagian besar informasi dalam suatu rangkaian melampaui satu token — struktur, koherensi, faktualitas, aliran aritmatika. Model harus mempelajarinya dengan mengumpulkan banyak sinyal satu token dalam triliunan token.

MTPbertanya: bagaimana jika setiap negara bagian tersembunyi diawasi untuk memprediksi beberapa token masa depan sekaligus? Gloeckle dkk. (Meta, 2024) menunjukkan bahwa hal ini membantu. Implementasinya menempatkan beberapa output head independen di atas backbone, masing-masing memprediksi offset yang berbeda. Paralel, sederhana, tetapi kepala-kepala tersebut melihat keadaan tersembunyi yang sama tanpa penyempurnaan hierarki apa pun — dan prediksi tersebut tidak berantai secara kausal, sehingga tidak dapat digunakan untuk penguraian kode spekulatif.

DeepSeek-V3 (Desember 2024) didesain ulangMTPsebagai modul berurutan yang menjaga rantai sebab akibat pada setiap kedalaman prediksi. Model tersebut memprediksi`t+1`dari`h_i^(0)`, lalu memprediksi`t+2`dari keadaan tersembunyi baru`h_i^(1)`yang digabungkan`h_i^(0)`dengan`E(t+1)`penyematan, dan sebagainya. Setiap kedalaman adalah blok trafo kecilnya sendiri. Penyematan bersama dan kepala keluaran bersama menjaga overhead parameter tetap sederhana. Pada skala DeepSeek-V3, terdapat 14 miliar parameter tambahanMTPmodul di atas bobot model utama 671B. Overhead 2% itu membeli sinyal pelatihan yang lebih padatANDrancangan penguraian kode spekulatif yang sudah jadi pada inferensi.

Pelajaran ini membangun satuMTPmodul dan kehilangan kedalaman D dari awal. Perhitungannya rapi. Implementasinya adalah 150 baris.

## Konsepnya

### Yang berurutanMTPresep

DeepSeek-V3 menambahkan`D` MTPmodul di atas model utama. Setiap modul`k`(untuk`k = 1..D`) memprediksi token secara mendalam`k`— yaitu,`t_{i+k}`diberi awalan melalui posisi`i`.

Modul`k`terdiri dari:

- Blok transformator`T_k`dengan perhatiannya sendiri danMLP.
- Matriks proyeksi`M_k`yang menggabungkan keadaan tersembunyi pada kedalaman sebelumnya dengan penyematan token kebenaran dasar pada kedalaman berikutnya.
- Penyematan bersama`E`(sama dengan model utama).
- Kepala keluaran bersama`Out`(sama dengan model utama).

Saat pelatihan, untuk awalan melalui posisi`i`, keadaan tersembunyi per kedalaman adalah:

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

Prediksi per kedalaman adalah:

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

Kerugian per kedalaman adalah entropi silang terhadap kebenaran dasar`t_{i+k}`:

```
L_k = CE(logits_{i+k}, t_{i+k})
```

Hilangnya sendi di seluruh kedalaman:

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda`adalah faktor bobot yang kecil — DeepSeek-V3 menggunakan 0,3 untuk 10% pelatihan pertama dan 0,1 setelahnya. Total kerugian pelatihan adalah`L_main + L_MTP`.

### Kenapa berurutan, bukan paralel

Paralel asli GloeckleMTPmemiliki kepala keluaran D, masing-masing langsung diterapkan`h_i^(0)`. Setiap kepala memprediksi`t_{i+k}`dari keadaan tersembunyi tulang punggung yang sama. Itu berlatih dengan baik, tetapi prediksinya tidak dikondisikan satu sama lain. kamu tidak dapat menggunakan`head_1`keluaran untuk membantu`head_2`— kepala menembak secara paralel.

Desain sekuensial DeepSeek-V3 dibuat`h_i^(k)`dari`h_i^(k-1)`ditambah penyematan token berikutnya yang sebenarnya`E(t_{i+k})`. Hal ini menjaga rantai sebab akibat: untuk memprediksi`t_{i+k+1}`, modul secara mendalam`k+1`melihat apa yang terjadi`t_{i+k}`. Hal ini secara struktural identik dengan bagaimana dekoder autoregresif menggunakan keluarannya sendiri — menghasilkanMTPmodul yang dapat langsung digunakan sebagai perancang decoding spekulatif.

Kesimpulannya: umpan`h_i^(k-1)`dan rancangannya`t_{i+k}`ke dalam modul`k+1`, dapatkan prediksinya`t_{i+k+1}`. Mengulang. Itu tepatnya sebuahEAGLE-style draft, menggunakan yang terlatihMTPmodul sebagai rancangan jaringan. DeepSeek-V3 melaporkan 80%+ penerimaan pada yang pertamaMTPmodul dan kecepatan ~1,8×.

### Akuntansi parameter

Untuk model dengan tersembunyi`h`dan kosa kata`V`:

- Model utama: miliaran parameter, ditambah satu ukuran kepala keluaran`V * h`.
- Kepala keluaran bersama: gunakan kembali kepala model utama. Tidak ada parameter tambahan.
- Penyematan bersama: gunakan kembali penyematan model utama. Tidak ada parameter tambahan.
- Per-MTPmodul:
  - Proyeksi`M_k`:`(2h) * h = 2h^2`.
  - Blok transformator`T_k`: perhatian (`4h^2`untukMHA) ditambahMLP(biasanya`8h^2`untuk SwiGLU dengan rasio8/3). Tentang`12h^2`per blok.

Total tambahan per modul:`~14h^2`. Untuk DeepSeek-V3`h = 7168`, D = 1 modul:`~14 * 7168^2 = ~720M`parameter di atas kertas. DeepSeek-V3 melaporkan 14B — perbedaannya adalah sebagian besar lapisan ahli adalah MoE di dalamnyaMTPmodul juga.

### Imbalan penguraian spekulatif

Selama pra-pelatihan,MTPmodul memperlambat pelatihan sekitar 10% (komputasi lebih maju, kerugian ekstra). Imbalannya ada dua:

1. Sinyal pelatihan lebih padat. Setiap negara bagian yang tersembunyi melihat target pengawasan H+1. Efek terukur padaMMLU,GSM8K,MATH, HumanEval: peningkatan beberapa poin persentase yang konsisten dalam ablasi DeepSeek-V3.

2. Draf decoding spekulatif gratis pada inferensi. ItuMTPmodul sudah dilatih untuk memprediksi beberapa token berikutnya. Digunakan kembali sebagai jaringan rancangan, jaringan ini memberikan tingkat penerimaan 80%+. Pada level tersebut, decoding spesifikasi N=3 atau N=5 memberikan throughput 1,8×. Biaya waktu pelatihan sebesar 10% terbayar saat pertama kali kamu menjalankan inferensi.

### Kaitannya denganEAGLE

EAGLEmelatih model draf kecilSEPARATELYsetelah pra-pelatihan.MTPmemasukkan draf tersebut ke dalam pra-pelatihan. Kedua pendekatan ini bertemu pada tingkat penerimaan yang serupa tetapi melalui jalur yang berbeda:

| Dimension | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| When trained | Post-pre-training | During pre-training |
| Backward-compatible with existing weights | Yes | No (need to re-train) |
| Draft params | 1-2 transformer layers | 1 transformer block + projection |
| Acceptance rate | 0.88-0.92 | 0.80+ at depth 1 |
| Benefit beyond speedup | Speculative decoding only | Denser training signal + speedup |

```figure
multi-token-predict
```

## Bangun Sendiri

`code/main.py`membangun satuMTPmodul ujung ke ujung: penyematan bersama, proyeksi, blok transformator, kepala keluaran bersama. Ia kemudian menghitung kerugian entropi silang per kedalaman pada urutan sintetik pendek dan mencetak jumlah parameter berdasarkan komponen. Kosakata mainan yang terdiri dari 32 token membuat angka-angkanya tetap terbaca.

### Langkah 1: tabel penyematan bersama

Satu`vocab_size x hidden`tabel digunakan oleh model utamaANDoleh setiapMTPmodul di setiap kedalaman. Bukan salinan kedua — secara harfiah merupakan tensor yang sama.

### Langkah 2: kombinasi per kedalaman

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

Real DeepSeek-V3 menggabungkan dua vektor RMSNormed menjadi`[2h]`dan proyek dengan`h x 2h`matriks. Mainan ini menggunakan penambahan vektor untuk singkatnya stdlib.

### Langkah 3: blok transformator pada kedalaman k

Ditambah perhatian diriMLP. Di dalam mainan, blok perhatian linier satu lapis dan SwiGLUMLPmenjaga struktur tetap terlihat tanpa numpy.

### Langkah 4: kepala keluaran bersama

Gunakan kembali proyeksi keluaran model utama. Logit atas kosakata.

### Langkah 5: kehilangan per kedalaman

Entropi silang softmax(logits) terhadap token kebenaran dasar pada offset`k`. Agregat melintasi kedalaman dengan`lambda / D`faktor skala.

### Langkah 6: akuntansi parameter

Cetak jumlah total parameter, jumlah bersama (penyematan, head), dan jumlah tambahan per modul. Tunjukkan rasioMTPekstra ke ukuran model utama.

## Gunakan

MTPdiintegrasikan ke dalam DeepSeek-V3 (Desember 2024) dan seri DeepSeek-R1. Pada inferensi:

- Tumpukan penyajian DeepSeek sendiri akan habisMTPmodul sebagai decoder spekulatif di luar kotak.
- vLLM dan SGLang memiliki jalur integrasi untuk DeepSeek-V3MTPper April 2026.
- AMDTutorial ROCm SGLang menunjukkan secara spesifikMTPkonfigurasi decoding spekulatif dengan kecepatan terukur 1,8× pada pos pemeriksaan V3.

Kapan harus digunakanMTPdalam menjalankan pra-pelatihan baru:

- kamu mengontrol seluruh jalur pra-pelatihan dan ingin menerima sinyal pelatihan yang lebih padat.
- kamu tahu bahwa kamu akan menyajikan model dalam skala besar dan menginginkan decoding spekulatif secara gratis.
- Ukuran tersembunyi kamu setidaknya 4096. Pada skala 1B, overhead lebih merugikan daripada keuntungan yang membantu.

Kapan tidak:

- Menyempurnakan model padat terlatih yang sudah ada. ItuMTPmodul tidak dilatih.
- Teliti model yang ingin kamu bandingkan dengan dasar yang jelas.MTPmengubah arsitektur.

## Rilis

Pelajaran ini menghasilkan`outputs/skill-mtp-planner.md`. Mengingat spesifikasi proses pra-pelatihan (ukuran model, data, komputasi), ia mengembalikan rencana untuk integrasiMTP: jumlah kedalaman D,`lambda`jadwal, overhead memori, dan kabel decoding spekulatif waktu inferensi.

## Latihan

1. Jalankan`code/main.py`. Tunjukkan kerugian per kedalaman menurun secara monoton seiring dengan menguatnya sinyal sintetik. Ubah sintetik untuk menggunakan pola tetap dan pastikan kehilangan kedalaman-1 dan kedalaman-2 menyatu.

2. Hitung parameter overhead untuk model 70B padat (tersembunyi 8192, 80 lapisan) dengan D=1MTPmodul. Bandingkan dengan overhead 14B yang dilaporkan DeepSeek-V3. Jelaskan mengapa angka DeepSeek lebih tinggi: theMTPblok transformator mewarisi struktur MoE yang sama, meningkatkan jumlah parameter per modul.

3. Implementasikan D=2 pada mainan: tambahkan satu detikMTPmodul yang mengambil h^(1) dan memprediksi`t_{i+2}`. Verifikasi kerugian gabungan dan akuntansi parameter sesuai dengan persamaan makalah DeepSeek 19-21.

4. Ganti mainan menjadi paralelMTP(Gaya Gloeckle): tambahkan kepala keluaran D di atas keadaan tersembunyi utama, masing-masing memprediksi offset yang berbeda. Ukur bagaimana kerugian per kedalaman dibandingkan dengan versi sekuensial pada sinyal sintetik yang sama. Versi sekuensial harus menghasilkan kerugian kedalaman-k yang lebih rendah untuk k > 1 karena mengkondisikan prediksi perantara.

5. Gunakan yang terlatihMTPmodul sebagaiEAGLE-style draft: panggil modul k untuk melamar`t_{i+k}`di inferensi. Ukur tingkat penerimaan token draf ini terhadap prediksi model utama pada urutan yang telah ditentukan. Jika kamu mencapai 50%+ pada mainan tersebut, kamu telah mereproduksi empirisnyaMTP-sebagai properti draft.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MTP module | "Extra loss block" | A small transformer block plus projection that predicts a token `k` positions ahead of the main model |
| Prediction depth | "Which offset" | The integer `k` such that module `k` predicts `t_{i+k}` from prefix through position `i` |
| Parallel MTP | "Gloeckle-style" | D independent heads on the same backbone hidden state, no conditional chain |
| Sequential MTP | "DeepSeek-V3 style" | Each module conditions on the previous depth's hidden state plus the next token's embedding; preserves causal chain |
| Shared output head | "Reuse the main head" | The MTP modules call the main model's LM head, not a separate output projection |
| Shared embedding | "Reuse the main table" | Same vocabulary embedding table is used everywhere; no duplicate parameters |
| Projection matrix M_k | "Combine hidden + next-token" | An `h x 2h` linear layer that folds the previous hidden state and the target-token embedding into the next depth's input |
| Joint loss L_MTP | "Averaged extra losses" | Arithmetic mean of per-depth cross-entropy losses, scaled by `lambda` |
| Acceptance rate at depth 1 | "How often MTP draft is right" | The rate at which the D=1 MTP module's top-1 prediction equals the main model's top-1 prediction; 80%+ on DeepSeek-V3 |
| Lambda weighting | "Extra-loss importance" | Per-depth scaling factor; 0.3 at start of training, 0.1 later on DeepSeek-V3 |

## Bacaan Lebih Lanjut

- [DeepSeek-AI — Laporan Teknis DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — urutan penuhMTPdeskripsi (Bagian 2.2), termasuk persamaan kerugian gabungan dan percepatan 1,8× pada inferensi
- [Gloeckle dkk. — Model Bahasa Besar yang Lebih Baik & Lebih Cepat melalui Prediksi Multi-token (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — paralelMTPdesain dasar DeepSeek membaik
- [Kartu model DeepSeek-V3 pada Wajah Memeluk](https://huggingface.co/deepseek-ai/DeepSeek-V3) — Total 685B (671B utama + 14BMTP), catatan penerapan
- [Leviathan dkk. — Inferensi Cepat dari Transformers melalui Penguraian Kode Spekulatif (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — kerangka penguraian kode spekulatifMTPcocok
- [Li dkk. —EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) —EAGLErancangan arsitektur tahun 2025, mitranyaMTPbersaing dengan
