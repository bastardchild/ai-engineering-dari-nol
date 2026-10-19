# Konsensus dan Byzantine Fault Tolerance untuk Agent

> BFT dari sistem distribusi klasik bertemu dengan LLM stokastik. Pada 2025-2026 tiga arah riset muncul: **CP-WBFT** (arXiv:2511.10400) menimbang setiap suara dengan sebuah confidence probe; **DecentLLMs** (arXiv:2507.14928) berjalan tanpa leader dengan proposisi pekerja paralel dan agregasi geometric-median; **WBFT** (arXiv:2505.05103) menggabungkan voting berbobot dengan Hierarchical Structure Clustering untuk memisahkan node Core dan Edge. Hasil empiris yang jujur dari "Can AI Agents Agree?" (arXiv:2603.01213) adalah bahwa bahkan kesepakatan skalar pun rapuh hari ini — satu agent yang menipu bisa membahayakan sebuah Mixture-of-Agents. BFT itu necessary tetapi tidak sufficient. Pelajaran ini membangun sebuah protokol BFT minimal, menyuntikkan tiga serangan khusus-agent (byzantine lie, sycophantic conformity, correlated-error monoculture), dan mengukur bagaimana tiap varian konsensus menghadapinya.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## Masalahnya

Kamu punya N agent LLM yang masing-masing menghasilkan sebuah jawaban. Mereka tidak setuju. Voting mayoritas memilih yang salah karena dua agent berkorelasi (base model sama, data training sama, mode kegagalan sama). Sebuah agent ketiga kebetulan salah dengan cara yang baru — sehingga mayoritas itu adalah mayoritas palsu.

Sekarang tambahkan sebuah agent yang menipu: ia berbohong dengan sengaja. Atau sebuah agent yang sycophantic: ia setuju dengan siapa pun yang terakhir bicara. Dalam BFT klasik, asumsinya adalah node Byzantine berjumlah pecahan `f < n/3` dan berperilaku sewenang-wenang. Realitas 2026 adalah node LLM bersifat stokastik bahkan ketika jujur, berkorelasi antar model, dan dipengaruhi oleh output satu sama lain. Kamu tidak bisa memperlakukannya sebagai pemilih Bernoulli yang independen.

BFT klasik (PBFT, 1999) tidak salah — ia belum lengkap. Ia menangani pembalikan bit sewenang-wenang. Ia tidak menangani "tiga agent jujur berbagi satu halusinasi karena mereka berbagi data training." Pelajaran ini membangun dari fondasi PBFT dan menambahkan tiga adaptasi 2025-2026 di atasnya.

## Konsepnya

### Apa yang diberikan BFT klasik padamu

Practical Byzantine Fault Tolerance (Castro & Liskov, OSDI 1999) menoleransi `f < n/3` node Byzantine. Protokolnya punya tiga fase (pre-prepare, prepare, commit) dan dua primitif (pesan bertanda tangan, quorum certificate). Kesepakatan atas satu nilai di antara `n >= 3f + 1` node jahat-atau-jujur.

Jaminannya kuat tetapi mengasumsikan:

1. **Kegagalan independen.** Para Byzantine tidak berkoordinasi.
2. **Node jujur benar-benar jujur.** Keakuratan output jujur bukan masalah; protokol hanya menyelaraskan ketidaksepakatan.
3. **Pertanyaan itu punya jawaban ground-truth.** Konsensus atas fakta yang salah tetaplah konsensus.

Agent LLM melanggar ketiganya. Dua agent yang menjalankan base model yang sama berbagi kegagalan. Sebuah LLM "jujur" tetap berhalusinasi. Dan pada pertanyaan yang ambigu, "kebenaran" adalah apa yang diputuskan para agent — tidak ada oracle eksternal.

### Tiga serangan khusus LLM

**Byzantine lie.** Satu agent mengeluarkan jawaban yang sengaja salah. BFT klasik menanganinya jika `f < n/3`.

**Sycophantic conformity.** Satu agent membaca jawaban orang lain sebelum voting dan menyelaraskan diri dengan siapa pun yang terakhir bicara. Tidak berbahaya, tetapi berkorelasi dengan suara paling keras. BFT klasik tidak mencegahnya karena agent itu lolos dari setiap pemeriksaan tanda tangan.

**Correlated-error monoculture.** Tiga agent berbagi satu base model. Mereka berhalusinasi dengan jawaban salah yang sama. Mayoritasnya salah. BFT klasik tidak menolong karena ketiganya "jujur" setuju.

### Tanggapan 2025-2026

**CP-WBFT** (arXiv:2511.10400) — Confidence-Probed Weighted BFT. Setiap pemilih melampirkan sebuah confidence probe ke jawabannya (probabilitas yang dilaporkan sendiri, atau prediksi model kalibrasi terpisah). Bobot suara berskala dengan keyakinan. Dilaporkan +85.71% peningkatan BFT pada graph lengkap. Mitigasi untuk: sycophantic conformity (agent yang menyesuaikan diri cenderung punya keyakinan rendah pada posisi yang mereka sukarelawankan).

**DecentLLMs** (arXiv:2507.14928) — Tanpa leader. Agent pekerja mengajukan proposisi secara paralel, agent evaluator memberi skor pada proposisi, jawaban akhir adalah geometric median dari posisi yang diberi skor. Robust ketika `f < n/2`. Mitigasi untuk: Byzantine lie dan kesalahan berkorelasi (geometric median robust terhadap outlier dan menarik ke arah gumpalan padat, bukan rata-rata yang bias-nya karena model).

**WBFT** (arXiv:2505.05103) — Weighted BFT dengan Hierarchical Structure Clustering. Bobot suara diberikan berdasarkan kualitas respons ditambah sebuah trust score yang dipelajari dari riwayat. Kelompokkan agent menjadi Core dan Edge; agent Core harus mencapai konsensus terlebih dahulu, agent Edge mengikuti. Mitigasi untuk: skalabilitas (konsensus Core kecil dan cepat) dan sebagian untuk monoculture (Core bisa dipilih demi keragaman).

### Empiris: "Can AI Agents Agree?" (arXiv:2603.01213)

Makalah itu mengukur kesepakatan skalar (agent LLM yang sepakat atas satu nilai numerik) di berbagai model frontier. Temuannya tidak nyaman:

- Bahkan tanpa adversarial, agent LLM tidak setuju pada pertanyaan skalar dengan tingkat di atas 30% di banyak benchmark.
- Satu agent yang mengadopsi persona penipu bisa menarik konsensus Mixture-of-Agents meleset 40+ poin persentase dari baseline jujur.
- Tingkat ketidaksepakatan berkorelasi dengan keragaman model — ensemble heterogen lebih tidak setuju daripada yang homogen (bagus: kesalahan tidak berkorelasi) tetapi juga mengalir lebih lambat (jelek: waktu-menuju-kesepakatan lebih lama).

Intinya: BFT memberimu mesin untuk menyelaraskan output, tetapi tidak memberitahumu apakah output yang diselaraskan itu benar. Gabungkan dengan verifikasi (spesialisasi peran Fase 16 · 08), keragaman (varian debat Fase 16 · 15), dan agent evaluator (benchmark Fase 16 · 24).

### Protokol inti, versi dipangkas

Sebuah ronde BFT minimal untuk agent LLM:

```
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

Langkah pengelompokan semantik adalah twist khusus LLM-nya. Dua jawaban "studi melaporkan 4.2%" dan "peningkatan 4.2%" berada di cluster yang sama. Pemeriksaan kesetaraan string yang naif akan melewatkannya. Di produksi, gunakan sebuah model embedding murah atau kanonisasi eksplisit.

### Penyetelan threshold

Parameter `threshold` menentukan kapan menerima dan kapan mencoba ulang. Terlalu rendah: kamu menerima mayoritas lemah. Terlalu tinggi: kamu tidak pernah menerima apa pun. Rentang empiris: 0.5-0.67 untuk agent `n=5-7`, lebih tinggi untuk `n` yang lebih kecil. Di bawah threshold, eskalasikan ke manusia atau ke ensemble agent yang berbeda.

### Di mana konsensus tidak menolong

- **Pertanyaan ambigu.** Jika pertanyaan itu tidak punya ground truth, konsensus adalah sebuah opini. Sebut saja begitu.
- **Pertanyaan majemuk.** "Tulis kode dan jelaskan" — dua jawaban. Voting secara independen untuk masing-masing.
- **Multi-round adversarial.** Jika agent bisa mengamati ronde sebelumnya dan meniru (debat Du 2023), mereka mulai setuju satu sama lain terlepas dari kebenaran. Batasi ronde-nya (2-3 biasanya).

```figure
swarm-consensus-wave
```

## Bangun Sendiri

`code/main.py` mengimplementasikan:

- `AgentVoter` — kebijakan ber-skrip dengan (jawaban, keyakinan).
- `MajorityVote` — plurality klasik.
- `CPWBFT` — voting berbobot keyakinan dengan pengelompokan semantik.
- `DecentLLMs` — agregasi geometric-median atas proposisi yang diberi skor.
- `Scenario` — menjalankan setiap aggregator di bawah tiga pola serangan.

Pola serangan yang diimplementasikan:

1. `byzantine`: satu agent berbohong dengan keyakinan tinggi.
2. `sycophancy`: satu agent menyalin jawaban pertama yang dilihatnya, dengan keyakinan yang cocok.
3. `monoculture`: tiga agent berbagi satu jawaban salah (kesalahan berkorelasi) dengan keyakinan sedang.

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan: sebuah tabel (serangan, aggregator) -> jawaban akhir, dengan jawaban benar disorot. Plurality gagal pada kasus monoculture. Pemberat keyakinan milik CPWBFT memitigasi sycophancy. Geometric median milik DecentLLMs menarik ke arah cluster jujur ketika monoculture kurang dari setengah populasi.

## Gunakan

`outputs/skill-consensus-designer.md` merancang sebuah protokol konsensus untuk sebuah ensemble multi-agent: metode pengelompokan, pemberatan, threshold, dan kebijakan eskalasi untuk ronde di bawah threshold.

## Rilis

Sebelum mengirim mekanisme konsensus apa pun:

- **Uji serangan dengan setidaknya tiga pola** di atas. Protokolmu seharusnya gagal secara dapat diprediksi, bukan diam-diam.
- **Catat setiap cluster minoritas** beserta provenance-nya. Cluster minoritas adalah sistem peringatan dini-mu untuk kesalahan berkorelasi.
- **Paksakan ronde berbatas.** Tidak ada "terus berdebat sampai sepakat" — itu memberi hadiah pada sycophancy.
- **Pisahkan kesepakatan dari kebenaran.** Output konsensus diarahkan ke sebuah verifier; verifier independen dari ensemble-nya.
- **Pantau tingkat kesepakatan.** Lonjakan tajam berarti bias konformitas; penurunan tajam berarti drift model.

## Latihan

1. Jalankan `code/main.py`. Konfirmasi plurality gagal pada serangan monoculture tetapi CPWBFT memitigasinya sebagian ketika keyakinan monoculture berada di bawah 0.7.
2. Tambahkan pola serangan keempat: **silent abstention** — satu agent menolak menjawab ("Saya tidak tahu"). Bagaimana seharusnya setiap aggregator memperlakukan abstensi? Implementasikan pilihanmu.
3. Tukar pengelompokan semantik dari kanonisasi string menjadi kemiripan-embedding (gunakan model embedding open-source apa pun). Apa yang terjadi pada serangan sycophancy?
4. Baca CP-WBFT (arXiv:2511.10400). Implementasikan langkah kalibrasi confidence-probe (sebuah model kalibrasi terpisah memeriksa keyakinan yang dilaporkan sendiri oleh setiap agent). Ukur peningkatan akurasi pada skenario monoculture.
5. Baca "Can AI Agents Agree?" (arXiv:2603.01213). Reproduksi sebuah eksperimen kesepakatan-skalar yang disederhanakan: tiga agent, satu pertanyaan skalar, prompt persona-penipu. Apakah CPWBFT atau DecentLLMs yang menangkapnya?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| BFT | "Byzantine fault tolerance" | Castro-Liskov 1999 protocol for consensus with `f < n/3` arbitrary faults. |
| Byzantine | "Any bad behavior" | A node that can lie, drop messages, fail silently — anything but crash safely. |
| Confidence probe | "How sure are you?" | Self-reported or calibrator-predicted probability attached to a vote. |
| Semantic clustering | "Same answer, different words" | Grouping equivalent answers before counting votes. |
| Geometric median | "Robust center" | The point minimizing sum of distances to sample points. Robust to outliers, unlike the mean. |
| Monoculture | "Same model, same failures" | Correlated errors when agents share training data or base model. |
| Sycophantic conformity | "Agreeing with the loud voice" | An agent's vote biases toward whoever spoke first/loudest. |
| Core/Edge | "Hierarchical BFT" | WBFT split: small Core consensus first, Edge nodes follow. Bounds latency. |

## Bacaan Lanjutan

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) — the foundation
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — vote weighting by confidence
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) — geometric-median aggregation
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — Core/Edge split for bounded latency
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) — scalar-agreement fragility and deceptive-persona attack
