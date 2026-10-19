# Decoding Spekulatif danEAGLE-3

> Fase 7 · Pelajaran 16 membuktikan perhitungan matematikanya: aturan penolakan Leviathan mempertahankan distribusi verifikator dengan tepat. Pelajaran ini adalah tampilan tumpukan pelatihan dari decoding spekulatif produksi tahun 2026.EAGLE-3 mengubah rancangan model dari perkiraan murah menjadi jaringan kecil yang dibuat khusus dan dilatih pada status tersembunyi milik pemverifikasi, lalu menambahkan loop pengujian waktu pelatihan yang menyelaraskan distribusi rangkaian dan inferensinya. Hasil: percepatan end-to-end 3× hingga 6,5×, tarif per token yang diterima di atas 0,9 pada chat, tidak ada tradeoff distribusi. Setiap tumpukan inferensi produksi pada tahun 2026 mengirimkannya secara default.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16 (speculative decoding math), Phase 10 · 12 (inference optimization)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Nyatakan teorema Leviathan dalam satu kalimat dan buktikan bahwa loop spekulatif menghasilkan sampel yang didistribusikan secara identik ke verifikator.
- Ikuti perkembangan dua tahun dari decoding spesifikasi vanilla (Leviathan 2023) hinggaEAGLE,EAGLE-2, danEAGLE-3 dan sebutkan batasan pasti setiap langkah yang dihapus.
- Hitung percepatan yang diharapkan dari tingkat penerimaan`α`dan rasio biaya draft-to-verifier`c`, dan pilih panjang draf yang optimal`N`untuk setiap rezim.
- Terapkan putaran spekulatif penuh dari awal: draf, verifikasi, tolak sampel dari sisa, kembalikan cache KV saat ditolak, keluarkan token bonus saat diterima penuh.

## Masalahnya

Penguraian kode autoregresif pada model 70B berjalan pada kecepatan mungkin 35 token per detik pada sebuahH100. ItuGPUtidak mendekati jenuh. Bandwidth memori adalah batas tertinggi: setiap token memuat bobot 70BHBM, melakukan satu langkah aritmatika, dan menghasilkan satu pelampung. Unit komputasi sebagian besar tidak digunakan.

Penguraian kode spekulatif mengubahnya menjadi masalah throughput yang sebenarnya dapat kamu pecahkan. Draf murah mengusulkan`N`token masuk`N`umpan-umpan kecil ke depan. Verifikator berjalan satu kali pada awalan ditambah semua`N`konsep. Jika distribusi verifikator pada posisi`i`setuju dengan rancangan tersebut (secara statistik akan kami buat secara tepat), kami menerima; jika tidak, kami menolak dan mengambil sampel koreksi dari distribusi sisa. Seorang penyerang model besar menghasilkan hingga`N+1`token yang diterima, bukan satu.

Teorema yang penting adalah Leviathan, Kalman, Matias (ICML2023): distribusi keluaran sama dengan hasil pengambilan sampel dari verifikator secara langsung. Tidak kira-kira. Secara identik. Inilah alasan mengapa decoding spekulatif dapat diterima dalam produksi — ini adalah optimasi latensi murni tanpa pengorbanan kualitas.

Apa yang diberikan Fase 7 · Pelajaran 16 kepada kamu adalah matematika. Apa yang diberikan pelajaran ini kepada kamu adalah tumpukan pelatihan. Draf yang bagus bernilai kecepatan 2x lebih banyak daripada draf murah.EAGLE,EAGLE-2, danEAGLE-3 (Li et al., 2024–2025) mengubah "draft = versi lebih kecil dari model yang sama" menjadi disiplin teknik presisi. Server inferensi produksi 2026 defaultnya adalahEAGLE-3.

## Konsepnya

### Invarian: pengambilan sampel penolakan Leviathan

Biarkan`p(t)`menjadi pendistribusian draf untuk token selanjutnya yang diberi awalan, dan`q(t)`menjadi pihak yang memverifikasi. Contoh draf token`d ~ p`. Terima dengan kemungkinan`min(1, q(d) / p(d))`. Saat ditolak, sampel dari distribusi sisa`(q - p)_+ / ||(q - p)_+||_1`. Sampel yang dihasilkan didistribusikan menurut`q`. Ini benar, tidak peduli seberapa buruknya`p`adalah — semakin buruk, semakin sering kamu menolaknya, namun hasilnya tetap sama persis.

Tumpukan`N`dari panggilan ini bolak-balik menggunakan satu penerusan verifikasi`prefix + d_1 + ... + d_N`. Verifikator kembali`q_1, q_2, ..., q_{N+1}`secara bersamaan. Berjalan dari kiri ke kanan. Pada penolakan pertama pada posisi`j`, sampel dari`residual(q_j, p_j)`dan berhenti. Pada penerimaan penuh, cicipi satu token bonus dari`q_{N+1}`.

### Yang menentukan percepatan

Biarkan`α`menjadi tingkat penerimaan yang diharapkan per token yang dirancang. Biarkan`c = cost(draft) / cost(verifier)`menjadi rasio biaya. Jumlah token yang diterima per verifikator yang diharapkan adalah:

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

Total waktu dinding yang diharapkan per token yang diterima adalah`(N * c + 1) / E[accepted]`. Minimalkan itu sehubungan dengan`N`dan kamu mendapatkan sweet spot. Untuk`α = 0.8, c = 0.05`: optimal`N`adalah sekitar 5–7, kecepatannya 3,2×. Untuk`α = 0.95, c = 0.02`: optimal`N`adalah sekitar 8–10, percepatan meningkat 5×.

Pengungkit terbesar adalah`α`. Berasal dari`α = 0.6`(draf vanila) ke`α = 0.9`(EAGLE-3) pada keadaan tetap`N = 5`membawa kamu dari 2.2 token yang diharapkan diterima per verifikator ke 4.1. Throughput hampir 2× lebih banyak dari pemverifikasi yang sama.

### Kemajuan dua tahun

**Vanilla spekulatif (Leviathan, 2023).** Model draf adalah model kecil yang dilatih secara mandiriLLMdari keluarga yang sama. Mudah untuk dihubungkan,`α ≈ 0.6`, percepat sekitar 2× paling banter.

**EAGLE-1 (Li dkk., 2024).** Draf adalah transformator kecil — biasanya satu atau dua lapisan — yang mengambil status tersembunyi lapisan terakhir pemverifikasi sebagai masukan dan memprediksi token berikutnya secara langsung. Karena draf melihat representasi fitur verifikator, maka distribusinya lebih dekat dengan verifikator.`α`naik menjadi 0,7–0,8.

**EAGLE-2 (Li dkk., 2024).** Menambahkan pohon draf dinamis: alih-alih mengusulkan satu rangkaian`N`token, mengusulkan pohon kecil kandidat, menilai masing-masing dengan verifikator dalam satu langkah ke depan (pohon perhatian), dan berjalan di jalur dengan probabilitas tertinggi. Panjang draf menjadi adaptif per langkah.`α`per token jalur yang diterima naik di atas 0,85.

**EAGLE-3 (Li et al., 2025, NeurIPS).** Dua perubahan lagi. Pertama, hilangkan seluruh kerugian prediksi fitur —EAGLE-1/2melatih draf agar sesuai dengan status tersembunyi pemverifikasi, yang membatasi jumlah bantuan data.EAGLE-3 berlatih langsung pada prediksi token. Kedua, tes waktu pelatihan (TTT): selama pelatihan draf, masukkan kembali prediksi draf sebelumnya sebagai masukan melalui beberapa langkah, dengan cara yang sama saat beroperasi pada inferensi. Hal ini menyelaraskan distribusi pelatihan dan pengujian serta menghentikan akumulasi kesalahan. Kecepatan terukur: hingga 6,5× pada obrolan, peningkatan throughput 38% pada batch 64 di SGLang aktifH100.

### Pengembalian cache KV

Verifikasi memperluas cache KV pemverifikasi sebesar`N`entri dalam satu pass. Jika penolakan terjadi pada posisi`j`, isi cache dari posisi sebelumnya`j-1`sekarang salah. Dua implementasi umum: menulis ke buffer awal dan melakukan penerimaan (vLLM, TensorRT-LLM), atau simpan cache KV fisik ditambah panjang logis dan potong saat ditolak. Apa pun yang terjadi, biaya rollback adalah byte per lapisan per head, yang dapat diabaikan jika dibandingkan dengan biaya forward-pass.

UntukEAGLE-2 pencarian pohon, pemverifikasi menjalankan perhatian dengan topeng non-kausal yang menghormati topologi pohon. Rekayasanya rumit tetapi perhitungannya adalah panggilan perhatian kilat standar dengan topeng khusus.

### Draf arsitektur pada tahun 2026

| Strategy | Draft type | `α` | Speedup | Training cost |
|----------|-----------|-----|---------|---------------|
| Vanilla | Separate small LLM | 0.55-0.70 | 1.8-2.3× | None (reuse existing small model) |
| Medusa | Extra LM heads on verifier | 0.65-0.75 | 2-3× | ~1B SFT tokens |
| EAGLE-1 | 1-layer transformer on hidden states | 0.70-0.80 | 2.5-3× | ~60B tokens |
| EAGLE-2 | EAGLE-1 + dynamic draft tree | 0.80-0.88 | 3-4× | ~60B tokens |
| EAGLE-3 | Multi-layer feature fusion + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B tokens |
| Lookahead | No draft (Jacobi iteration) | N/A | 1.3-1.6× | None |

Pada produksi tahun 2026: vLLM dan SGLang defaultnyaEAGLE-3 bila tersedia,EAGLE-2 sebaliknya. TensorRT-LLMmemiliki jalur Medusa tercepat untuk Meta danNVIDIAmodel publik. llama.cpp mengirimkan draft vanilla untukCPUpenerapan.

```figure
l5-spec-decode-eagle
```

## Bangun Sendiri

Lihat`code/main.py`. Ini adalah loop spekulatif Leviathan lengkap dengan semua bagiannya: draft-of-N, verifikasi paralel pass, penolakan per posisi, pengambilan sampel sisa, token bonus, pengembalian KV, dan verifikasi empiris bahwa distribusi keluaran cocok dengan pengambilan sampel langsung dari`q`.

### Langkah 1: aturan penolakan

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### Langkah 2: distribusi sisa

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### Langkah 3: langkah spekulatif penuh

Itu`spec_step`draf fungsi`N`token dari`p`, lalu verifikasi semuanya dalam satu paralel`q`evaluasi. Untuk setiap token yang dirancang, ia menerapkan aturan penolakan, dan pada penolakan pertama, ia mengambil sampel koreksi dari sisa. Jika semuanya menerima, ia mengeluarkan token bonus`q_{N+1}`.

### Langkah 4: Pembukuan rollback KV

Simulator melacak secara logis`kv_length`per pekerja. Tentang penerimaan`k`konsep,`kv_length += k`. Tentang penolakan pada posisi`j`, cache sudah ditulis sebelumnya`j`, tetapi panjang logisnya disetel ke`prefix_length + j + 1`— satu melewati token koreksi. Bacaan selanjutnya terpotong ke panjang logis.

### Langkah 5: cek Leviathan

Jalankan 50.000 langkah spekulatif. Hitung distribusi empiris dari token yang diterima. Bandingkan dengan 50.000 sampel langsung dari`q`. Statistik chi-kuadrat harus berada di bawah nilai kritis. Teorema ini lolos dalam praktik.

### Langkah 6: percepatan vs

Sapu kualitas draf dengan mengganggu`p`jauh dari`q`pada amplitudo yang berbeda. Ukur`α`, lalu plot token yang diharapkan per panggilan verifikator sebagai fungsi dari`α`dan`N`. Kode mencetak tabel yang menunjukkan caranyaEAGLE-Kualitas draf kelas 3 (`α ≈ 0.9`) membuka 4–5 token per panggilan verifikator.

## Gunakan

Tingkat produksi`vllm serve`denganEAGLE-3:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang denganEAGLE-3 pada angkatan 64 aktifH100: kira-kira 1,38× lebih banyak throughput dibandingkan decoding vanilla batch-64, menurutEAGLE-3 kertas.

Kapan harus melakukan decoding spekulatif:

- Beban kerja obrolan interaktif apa pun yang latensi p50 lebih penting daripada throughput puncak.
- Pembuatan kode dan keluaran terstruktur (JSON,SQL).`α`berada di atas 0,9 karena distribusi target sangat dapat diprediksi.
- Generasi jangka panjang (ribuan token). Percepatan yang diamortisasi terus membuahkan hasil.

Kapan tidak:

- Model yang sangat kecil (< 3B). Drafnya tidak jauh lebih murah dibandingkan verifikator.
- Kecil batch-1CPUpenerapan. Overhead memori pada model draf mungkin tidak sepadan.
- Pengambilan sampel materi iklan bersuhu sangat tinggi di mana`α`runtuh.

## Rilis

Pelajaran ini menghasilkan`outputs/skill-eagle3-tuner.md`. Mengingat beban kerja inferensi (model, ukuran batch, latensi target, profil tugas), ini merekomendasikan strategi penguraian kode spekulatif dan parameter penyetelan (keluarga draf,`N`, kedalaman pohon, peralihan sadar suhu).

## Latihan

1. Jalankan`code/main.py`. Konfirmasikan statistik chi-kuadrat pada pemeriksaan distribusi Leviathan tetap di bawah nilai kritis 95% pada 50.000 sampel.

2. Menyapu`N`dari 1 sampai 10 dengan`α`diadakan pada 0,9 dan`c`diadakan pada 0,04. Plot token yang diharapkan per panggilan verifikator dan waktu dinding aktual per token. Temukan`N`yang meminimalkan waktu dinding. Jelaskan bentuk kurva tersebut!

3. Ubah kode untuk disimulasikanEAGLE-2 Pencarian pohon: pada setiap langkah, rancangan mengusulkan bentuk pohon`[2, 2, 2]`(delapan jalur kandidat). Pemverifikasi berjalan satu kali, dan jalur yang diterima dengan probabilitas tertinggi menang. Hitung`α`per daun dan total token per panggilan verifikator. Bandingkan dengan decoding spesifikasi rantai linier pada komputasi yang setara.

4. Terapkan simulator rollback KV batch untuk dua urutan bersamaan. Urutan A telah menerima semua draf; urutan B ditolak pada posisi 2. Tunjukkan bahwa benar`kv_length`diperbarui per urutan dan tidak ada pekerjaan yang terbuang.

5. BacaEAGLE-3 makalah Bagian 4 (Tes Waktu Pelatihan). Jelaskan dalam dua kalimat mengapa rancangan pelatihan naif tanpaTTTmenderita bias eksposur, dan mengapa memasukkan prediksinya sendiri ke dalam draf selama pelatihan akan memperbaikinya. Hubungkan ini dengan literatur pengambilan sampel terjadwal di seq2seq.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Leviathan rule | "min(1, q over p)" | Bernoulli accept/reject with probability `min(1, q(d)/p(d))`, preserves the verifier distribution exactly when you sample from the residual on rejection |
| Residual distribution | "(q minus p) plus, normalized" | `(q - p)_+` clamped at zero and renormalized — the correct distribution to sample from on rejection |
| Acceptance rate α | "how often the draft is right" | Expected per-token Bernoulli-success probability under the rejection rule; governs all speedup math |
| EAGLE-1 | "hidden-state draft" | Tiny transformer draft conditioned on the verifier's last-layer hidden state (Li et al., 2024) |
| EAGLE-2 | "dynamic draft tree" | EAGLE-1 plus a tree of candidate continuations scored with tree attention in one verifier pass |
| EAGLE-3 | "training-time test" | Drops the feature-prediction loss, trains on direct token prediction with the draft fed its own outputs during training |
| Training-time test (TTT) | "exposure bias fix" | Run the draft autoregressively during training so train and test input distributions match — the direct analog of scheduled sampling |
| KV rollback | "undo rejected drafts" | Bookkeeping that resets the verifier's KV cache to the accepted-prefix length after a rejection |
| Bonus token | "the free one" | When all `N` drafts accept, sample one extra from `q_{N+1}` at no additional verifier cost |
| Tree attention | "verify many candidates at once" | Attention with a non-causal mask that respects the topology of a draft tree; computes `q_i` for every node in the tree in one forward pass |

## Bacaan Lebih Lanjut

- [Leviathan, Kalman, Matias - Inferensi Cepat dari Transformers melalui Decoding Spekulatif (arXiv:2211.17192,ICML2023)](https://arxiv.org/abs/2211.17192) — makalah dasar dan teorema kesetaraan
- [Chen dkk. — Mempercepat Dekode Model Bahasa Besar dengan Pengambilan Sampel Spekulatif (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — pengenalan independen secara bersamaan dengan bukti bersih
- [Li dkk. —EAGLE: Pengambilan Sampel Spekulatif Memerlukan Pemikiran Ulang Ketidakpastian Fitur (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) —EAGLE-1, rancangan kondisi tersembunyi
- [Li dkk. —EAGLE-2: Inferensi Model Bahasa yang Lebih Cepat dengan Pohon Draf Dinamis (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — pencarian pohon dinamis
- [Li dkk. —EAGLE-3: Meningkatkan Akselerasi Inferensi melalui Tes Waktu Pelatihan (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — default produksi tahun 2026
- [Cai dkk. — Medusa: Beberapa Kepala Penguraian Kode (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — pendekatan alternatif tanpa rancangan
- [dokumentasi Dekode Spekulatif vLLM](https://docs.vllm.ai/en/latest/features/spec_decode.html) — referensi produksi kanonik dengan semua strategi yang terhubung
