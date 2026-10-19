# Desain Self-Improvement yang Dibatasi

> Riset telah konvergen pada empat primitif untuk membatasi loop self-improvement. Invarian formal yang harus berlaku di setiap edit. Jangkar alignment yang tidak bisa dimodifikasi. Batasan multi-objektif tempat setiap dimensi (keamanan, keadilan, ketahanan) harus berlaku, bukan hanya performa. Deteksi regresi yang menghentikan loop ketika metrik historis menunjukkan kehilangan kapabilitas. Tidak ada yang merupakan bukti keamanan — hasil teoretik-informasi (kompleksitas Kolmogorov, teorema Lob) membatasi apa yang bisa dibuktikan sistem apa pun tentang penerusnya. Mereka adalah mitigasi yang menaikkan biaya kegagalan senyap.

**Type:** Learn
**Languages:** Python (stdlib, bounded-loop with invariant check)
**Prerequisites:** Phase 15 · 07 (RSI), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## Masalahnya

Simulator balapan Pelajaran 7 menunjukkan perbedaan laju kecil berlipat menjadi kesenjangan besar. Studi kasus DGM Pelajaran 4 menunjukkan loop bisa secara aktif mengakali evaluator-nya sendiri. Kedua hasil menunjuk ke pertanyaan engineering yang sama: batasan apa yang bisa kamu pasang pada loop self-improvement sehingga batasan itu sendiri tidak bisa dilemahkan secara senyap oleh loop?

Ringkasan ICLR 2026 RSI Workshop (openreview.net/pdf?id=OsPQ6zTQXV) mengidentifikasi empat primitif semacam itu. RSP v3.0 Anthropic (Pelajaran 19) dan FSF v3 DeepMind (Pelajaran 20) sama-sama merujuknya dalam ambang kapabilitas. Karya Meta HyperAgents dan kerangka komunitas seperti SAHOO (Maret 2026) mengimplementasikan subset dalam produksi.

Bingkai yang jujur: ini mitigasi. Hasil teoretik-informasi membatasi apa yang bisa dibuktikan sistem apa pun tentang penerusnya, dan tidak ada desain saat ini menutup soal itu secara formal. Loop yang dibatasi baik lebih aman dibanding yang tidak dibatasi, bukan aman dalam arti absolut.

## Konsepnya

### Primitif 1: invarian formal

Invarian adalah properti yang harus berlaku sebelum dan sesudah setiap modifikasi-diri. Contoh:

- Distribusi keluaran dikondisikan pada header konstitusi tetap (Pelajaran 17).
- Tidak ada panggilan tool ke endpoint yang tidak diotorisasi.
- Penulisan memori lewat jalur yang di-log dan ditandatangani.
- Hash modul evaluator cocok dengan versi yang disetujui.

Invarian dicek oleh kode eksternal yang tidak bisa diedit loop. Jika modifikasi yang diusulkan melanggar invarian, ia ditolak. Ini menangkap mode tamper-evaluator gaya DGM (Pelajaran 4).

Bagian sulitnya adalah memilih invarian yang perlu bagi keamanan dan komputabel dalam waktu wajar. Spesifikasi berlebih menghentikan progres; spesifikasi kurang membiarkan loop berjalan melewati properti penting.

### Primitif 2: jangkar alignment

Jangkar alignment adalah representasi immutable atas tujuan inti loop, dipatok di luar permukaan edit loop. Contoh:

- Teks konstitusional (Pelajaran 17) yang dimuat dari lokasi read-only pada setiap invokasi.
- Evaluator di repositori terpisah dengan kontrol akses independen.
- Manifes izin-tool yang ditandatangani manusia dan diverifikasi ulang setiap siklus.

Peran jangkar adalah mencegah drift tujuan. Loop boleh memperbaiki cara mengejar tujuan, tetapi tidak bisa mengedit apa tujuannya.

Mode kegagalan yang subtil: jangkar yang tidak bisa diedit loop masih bisa diinterpretasi ulang oleh loop yang melenceng dalam cara membaca jangkar. Constitutional AI (Pelajaran 17) secara eksplisit berbasis alasan untuk menangani situasi baru; lapisan penalaran itulah tempat drift interpretasi berada. Jangkar itu perlu, tidak cukup.

### Primitif 3: batasan multi-objektif

Loop yang mengoptimalkan satu skor skalar akan menemukan jalan pintas. Loop yang harus sekaligus memenuhi banyak batasan keras punya lebih sedikit jalan pintas. Sumbu tipikal:

- Performa (benchmark tingkat-tugas)
- Keamanan (evaluasi red-team, refusal rate pada yang diketahui-buruk)
- Keadilan (batas disparate-impact pada subgrup sensitif)
- Ketahanan (set uji OOD, penanganan input adversarial)

Modifikasi diterima hanya jika setiap batasan berlaku. Governor biaya Pelajaran 13 menumpuk ini dengan batasan finansial. Llama Guard Pelajaran 18 terpasang sebagai sumbu keamanan.

### Primitif 4: deteksi regresi

Skor setiap siklus dibandingkan dengan distribusi historis. Penurunan di luar toleransi yang dikonfigurasi menghentikan loop. Ini menangkap kehilangan kapabilitas senyap yang jika tidak akan terserap ke rata-rata berjalan saat loop "membaik melewatinya."

Implementasi praktis: simpan skor per tugas dari N siklus terakhir. Setiap siklus baru menghitung delta per tugas. Jika ada delta turun di bawah ambang, siklus ditolak dan manusia mereviu.

### Batas teoretik-informasi

Kompleksitas Kolmogorov dan teorema Lob menetapkan batas atas atas apa yang bisa dibuktikan sistem tentang dirinya sendiri. Godel Machine formal Schmidhuber (Pelajaran 4) membidik batas tertinggi; tidak ada yang menyelesaikan bukti non-trivial. Hasil Lob berkata: jika sistem secara terbukti meyakini "aku akan melakukan X jika aku membuktikan aku harus melakukan X," ia akan melakukan X tanpa membuktikan ia harus, sebuah kegagalan referensi-diri yang terkenal.

Implikasinya bagi primitif kita: mereka tidak bisa menutup soal keamanan. Mereka membuat kegagalan senyap lebih mahal. Loop yang melenceng atau jahat, yang tadinya diam-diam melewati cek yang hilang, kini harus aktif merusak cek yang eksplisit, yang merupakan tanda yang lebih terdeteksi.

### Contoh yang dikerjakan

Misalkan agen mengusulkan sebuah edit. Tumpukan gerbangnya:

1. Cek invarian: hash modul, manifes izin-tool, header konstitusional.
2. Cek jangkar: pernyataan tujuan cocok dengan versi yang disetujui (byte-wise atau semantik).
3. Evaluasi multi-objektif: sumbu performa, keamanan, keadilan, ketahanan.
4. Deteksi regresi: tidak ada sumbu turun lebih dari toleransi.

Keempatnya harus lolos agar edit mendarat. Satu kegagalan menghentikan loop.

```figure
bounded-gates
```

## Gunakan

`code/main.py` menjalankan loop self-improvement yang dibatasi pada mainan gaya DGM dari Pelajaran 4, tetapi dengan keempat primitif dilapis di atasnya. Setiap primitif bisa dinyalakan atau dimatikan satu per satu. Demonstrasinya adalah bahwa setiap primitif menangkap kelas kegagalan spesifik, dan bahwa melepas salah satunya membiarkan kelas kegagalan itu lolos.

## Rilis

`outputs/skill-bounded-loop-review.md` mengaudit loop terbatas yang diusulkan dan menilai primitif mana dari keempatnya yang benar-benar ia implementasikan versus yang ia klaim.

## Latihan

1. Jalankan `code/main.py` dengan semua primitif menyala. Konfirmasi loop masih membaik pada metrik utama tanpa membiarkan hack menang.

2. Matikan deteksi regresi. Susun input tempat ini berujung pada kehilangan kapabilitas senyap yang diterima.

3. Matikan batasan multi-objektif. Tunjukkan loop konvergen pada sumbu performa sementara sumbu keamanan turun.

4. Rancang jangkar alignment untuk agen koding. Teks apa, disimpan di mana, dicek bagaimana?

5. Baca ringkasan ICLR 2026 RSI Workshop. Pilih satu dari empat primitif dan usulkan perbaikan konkret atas state of the art saat ini.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Invariant | "Always-true property" | A property checked by external code before and after every edit |
| Alignment anchor | "Pinned objective" | Immutable core-goal representation outside the loop's edit surface |
| Multi-objective constraint | "All axes must hold" | Performance, safety, fairness, robustness — all required |
| Regression detection | "Pause on drop" | Pause the loop when historical metric deltas suggest capability loss |
| Kolmogorov bound | "Information-theoretic limit" | Limits what a system can prove about its own successor |
| Lob's theorem | "Self-reference trap" | System can act on "I should" without proving it should |
| Gate stack | "Layered check" | Multiple primitives combined; any failure rejects the edit |
| Bounded improvement | "Mitigation, not proof" | Raises silent-failure cost; does not close the safety problem |

## Further Reading

- [ICLR 2026 RSI Workshop summary (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) — the four-primitive convergence.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — multi-objective capability thresholds.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — deceptive-alignment monitoring as an invariant primitive.
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html) — the formal-proof ancestor of these primitives.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — the reason-based alignment anchor.
