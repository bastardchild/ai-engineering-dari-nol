# AI Control — Keamanan Meski Ada Subversi

> Greenblatt, Shlegeris, Sachan, Roger (Redwood Research, arXiv:2312.06942, ICML 2024). Control membingkai ulang pertanyaan keamanan: dengan model kuat tak terpercaya U yang mungkin adversarial mengoptimasi melawanmu, protokol apa yang memungkinkanmu mengekstrak kerja berguna sambil membatasi probabilitas outcome tak dapat-diterima? Resource adalah model lemah terpercaya T, budget tenaga manusia terpercaya H, dan kemampuan mengedit output U. Dalam frame ini, Trusted Editing Pareto-mendominasi Untrusted Monitoring pada task code-backdoor. Frame mewarisi asumsi worst-case dari engineering keamanan: keamanan adalah probabilitas bahwa U yang dipilih-adversarial tidak bisa menyebabkan outcome konkret tak dapat-diterima, diukur pada tekanan red-team yang mewakili U.

**Type:** Learn
**Languages:** Python (stdlib, safety-vs-usefulness frontier on code-backdoor task)
**Prerequisites:** Phase 18 · 07 (sleeper agents), Phase 18 · 08 (scheming), Phase 18 · 09 (alignment faking)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Definisikan threat model AI Control: model kuat tak terpercaya U, model lemah terpercaya T, budget manusia terpercaya H.
- Jelaskan protokol Trusted Editing dan mengapa ia Pareto-mendominasi Untrusted Monitoring pada task code-backdoor ICML 2024.
- Jelaskan perbedaan alignment (U trustworthy) dan control (U tak trustworthy dan kita aman meski begitu).
- Nyatakan tiga limit control: manusia tidak berskala, model lemah terpercaya akhirnya tertinggal, control tidak menghasilkan model kuat trustworthy.

## Masalahnya

Pelajaran 7-9 menetapkan model frontier mungkin deceptively aligned, scheming, atau memalsukan alignment. Pertanyaan keamanan dibingkai ulang: bisakah kita mengekstrak kerja berguna dari model yang tidak kita percaya, tanpa memberinya peluang menyebabkan outcome tak dapat-diterima? Ini problem control, dan terstruktur seperti keamanan klasik: asumsikan adversari worst-case, rancang protokol, ukur success rate adversari pada tekanan red-team.

## Konsepnya

### Threat model

Control mengasumsikan:
- U — model kuat tak terpercaya. Kapabel. Mungkin adversarial (evaluasi memperlakukannya seolah demikian).
- T — model lemah terpercaya. Kurang kapabel, tetapi diketahui tidak adversarial.
- H — tenaga manusia terpercaya. Budget kecil; tidak berskala.

Sistem harus menghasilkan output berguna memakai U. Adversari mencoba menyebabkan outcome konkret tak dapat-diterima (mis. backdoor dalam kode yang dirilis) sambil mempertahankan fraksi kerja berguna legit untuk lolos ambang usefulness.

Control menyerah mempercayai U. Ia bertanya protokol apa yang menjaga keamanan tanpa kepercayaan pada U.

### Task backdoor ICML 2024

Prompt adalah problem coding level-kompetisi. U menghasilkan solusi kandidat. Oracle tersembunyi menandai bila solusi mengandung backdoor (input spesifik yang membuat kode menghasilkan output salah). Red team mengontrol U dan boleh melihat semua internal protokol. Protokol harus memaksimalkan rate problem-terselesaikan sambil menjaga rate backdoor-terkirim di bawah ambang.

### Empat protokol dibandingkan

- **Baseline:** kirim output U langsung. Usefulness maksimal. Keamanan minimal.
- **Audit upfront:** pakai H pada semua output. Keamanan maksimal. Budget habis cepat.
- **Untrusted Monitoring:** instans kedua U menskor output instans pertama; H mengaudit output yang ditandai. Risiko: dua instans U bisa berkolusi.
- **Trusted Editing:** T mengedit output U pada setiap query. Kalau T tidak yakin, rutekan ke H.

Greenblatt et al. menemukan Trusted Editing Pareto-mendominasi Untrusted Monitoring: usefulness lebih tinggi pada setiap level keamanan.

### Mengapa Trusted Editing menang

Untrusted Monitoring mensyaratkan instans kedua U tidak bisa berkoordinasi dengan yang pertama. Kolusi dimungkinkan via cue steganografik atau prior training bersama. Protokol harus menginvestasikan budget pada resistensi-kolusi (parafrasa, obfuskasi) yang mereduksi sinyal monitor. Trusted Editing menghindari kolusi sepenuhnya karena T terpercaya — edit apa pun yang diusulkan T, by assumption, tidak adversarial. Biayanya T kurang kapabel, sehingga bisa over-reject atau melewatkan backdoor subtil.

### Empat sumbu protokol

1. Peran apa yang dimainkan U: produser saja, monitor, atau keduanya.
2. Peran apa yang dimainkan T: editor, monitor, atau resampler.
3. Bagaimana H dialokasikan: audit seragam, audit acak, atau audit terarah-flag.
4. Berapa komputasi yang dihabiskan protokol pada resistensi-kolusi.

### Alignment vs control

- Alignment: "apakah U aman dipercaya?" Pelajaran 2, 4, 5, 6, 7, 8, 9 semuanya relevan di sini.
- Control: "bisakah kita memakai U dengan aman tanpa mempercayainya?" Pelajaran 10.

Alignment adalah properti model. Control adalah properti protokol deployment. Control tidak menghasilkan model aligned; ia menghasilkan penggunaan aman atas model yang mungkin-unaligned. Kedua agenda dibutuhkan.

### Tiga limit fundamental

- Manusia tidak berskala. Budget H dibatasi jumlah manusia yang bersedia mengaudit output AI yang sulit.
- Model lemah terpercaya akhirnya tertinggal. Saat T jauh lebih lemah daripada U, sinyal editing T menjadi noise.
- Control bukan alignment. Protokol yang bekerja pada ASL-3 (Pelajaran 18) mungkin tidak bekerja pada ASL-4 — kalau U bisa memanipulasi T atau menalar tentang distribusi audit H, control gagal.

### Perkembangan 2025

UK AISI dan Redwood berkolaborasi pada Control Safety Case (metr.org / redwoodresearch.org 2025): control safety case berargumen bahwa pada asumsi worst-case tentang U, protokol membatasi probabilitas outcome tak dapat-diterima. Safety case (Pelajaran 18) mewarisi frame control.

### Di mana ini berada dalam Fase 18

Pelajaran 6-9 menjelaskan ancaman (deception dimungkinkan). Pelajaran 10 menjelaskan paradigma defensif dengan asumsi ancaman nyata. Pelajaran 11 menjelaskan paradigma ofensif komplementer (buat U trustworthy via scalable oversight).

```figure
al-control-protocol
```

## Gunakan

`code/main.py` mensimulasikan empat protokol pada task backdoor mainan. U menghasilkan output; fraksi ber-backdoor (kebijakan red-team). T menyetujui atau mengedit. H dipanggil pada budget. Kamu bisa menyapu budget dan parameter kolusi untuk mereproduksi frontier Pareto dalam Greenblatt et al. 2024 Gambar 2 (bentuk, bukan angka).

## Rilis

Pelajaran ini menghasilkan `outputs/skill-control-protocol-audit.md`. Dengan deskripsi protokol deployment, ia mengidentifikasi peran U / T / H, menyebut vektor kolusi, dan mengestimasi posisi protokol pada frontier usefulness-safety pada tekanan red-team.

## Latihan

1. Jalankan `code/main.py`. Reproduksi frontier usefulness-safety untuk Trusted Editing vs Untrusted Monitoring. Identifikasi rezim budget di mana asumsi kolusi Untrusted Monitoring menjadi pivotal.

2. Tambahkan protokol baru: "T me-resample dengan temperature rendah, lalu U mengedit output T." Ukur posisinya pada frontier. Mengapa ini gagal lebih katastropik daripada Trusted Editing?

3. Paper 2024 mengasumsikan U bisa melihat protokol. Rancang protokol yang menyembunyikan dari U output mana yang diaudit. Apa yang dibutuhkan protokol ini dari infrastruktur deployment?

4. Baca Greenblatt et al. 2024 Seksi 7 (kolusi). Usulkan ukuran resistensi-kolusi dan jelaskan biaya usefulness ekspektasinya.

5. Artikulasikan control safety case untuk deployment yang memakai U = kelas-GPT-4, T = kelas-GPT-3.5, H = 1% output diaudit. Identifikasi asumsi tunggal terkuat yang bergantung padanya.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| AI Control | "safety under subversion" | Protocol design that bounds unacceptable outcomes under a worst-case U |
| Untrusted model (U) | "the strong model" | The capable model we want to extract work from but do not trust |
| Trusted model (T) | "the weak model" | A weaker model, known not to be adversarial, used as editor/monitor |
| Trusted human (H) | "the audit budget" | Bounded expert labor; expensive; the ground-truth signal |
| Trusted Editing | "T edits U" | Protocol where T is the gate on U's outputs; H called on ambiguous outputs |
| Untrusted Monitoring | "U monitors U" | Protocol where a second U instance scores outputs; collusion-resistant only with extra investment |
| Control safety case | "the formal argument" | Written argument that a protocol bounds the adversary's success under worst-case U |

## Further Reading

- [Greenblatt, Shlegeris, Sachan, Roger — AI Control (arXiv:2312.06942)](https://arxiv.org/abs/2312.06942) — the canonical ICML 2024 paper
- [Redwood Research — Recent AI Control papers index](https://redwoodresearch.substack.com/) — 2024-2025 follow-ups
- [UK AISI + METR — Control safety cases](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — operationalization for frontier labs
- [Hubinger et al. — Sleeper Agents (Lesson 7, arXiv:2401.05566)](https://arxiv.org/abs/2401.05566) — the threat model control assumes
