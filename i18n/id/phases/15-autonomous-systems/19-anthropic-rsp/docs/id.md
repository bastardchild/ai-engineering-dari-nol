# Responsible Scaling Policy v3.0 Anthropic

> RSP v3.0 berlaku 24 Februari 2026, menggantikan kebijakan 2023. Mitigasi dua-tier: apa yang akan dilakukan Anthropic secara unilateral vs apa yang dibingkai sebagai rekomendasi industri-luas (termasuk standar keamanan RAND SL-4). Menambahkan Frontier Safety Roadmap dan Risk Report sebagai dokumen tetap ketimbang deliverable sekali-jadi. Menghapus komitmen pause 2023. Memperkenalkan ambang AI R&D-4: setelah dilewati, Anthropic harus menerbitkan affirmative case yang mengidentifikasi risiko misalignment dan mitigasinya. Claude Opus 4.6 tidak melewatinya. Anthropic menyatakan dalam pengumuman v3.0 bahwa "confidently ruling this out is becoming difficult." SaferAI menilai RSP 2023 pada 2,2; mereka menurunkan v3.0 menjadi 1,9, menempatkan Anthropic dalam kategori RSP "lemah" bersama OpenAI dan DeepMind. Ambang kualitatif menggantikan komitmen kuantitatif 2023; penghapusan klausul pause adalah regresi paling tajam.

**Type:** Learn
**Languages:** Python (stdlib, RSP threshold decision engine)
**Prerequisites:** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**Time:** ~45 minutes

## Masalahnya

Lab frontier menerbitkan kebijakan scaling yang sebagian dokumen teknis, sebagian dokumen tata kelola, dan sebagian sinyal ke regulator. RSP v3.0 adalah dokumen Anthropic saat ini. Membacanya cermat penting bukan karena kepatuhan padanya mengikat (tidak), tetapi karena bingkainya membentuk bagaimana lab memahami risiko katastropik dan bagaimana mereka mengomunikasikan tradeoff ke publik.

Diff v3.0 vs v2.0 adalah unit yang berguna. Apa yang ditambah: Frontier Safety Roadmap, Risk Report, ambang AI R&D-4. Apa yang dihapus: komitmen pause 2023. Apa yang dibingkai ulang: jadwal mitigasi dua-tier yang terbagi antara unilateral-Anthropic dan rekomendasi-industri. Reviu eksternal — SaferAI — menurunkan skor dari 2,2 (v2) menjadi 1,9 (v3.0). Beginilah kebijakan scaling bisa menjadi kurang ketat sambil tampak lebih dipoles.

## Konsepnya

### Jadwal mitigasi dua-tier

- **Aksi unilateral Anthropic**: apa yang akan dilakukan Anthropic terlepas dari apa yang dilakukan lab lain. Berhenti training di atas ambang, tindakan keamanan spesifik, gerbang deployment spesifik.
- **Rekomendasi industri-luas**: apa yang menurut Anthropic harus dilakukan industri secara kolektif. Termasuk standar keamanan RAND SL-4. Ini bukan komitmen di sisi Anthropic; ini advokasi kebijakan.

Struktur dua-tier tidak ada di v2. Artinya pembaca perlu melihat di kolom mana setiap komitmen berada. Tindakan keamanan di kolom "rekomendasi industri-luas" bukan janji Anthropic; itu harapan Anthropic.

### Ambang AI R&D-4

Ini level kapabilitas yang dinamai RSP v3.0 sebagai ambang penting berikutnya. Spesifiknya: model yang bisa mengotomatiskan fraksi substansial riset AI dengan biaya kompetitif. Setelah Anthropic meyakini sebuah model melewatinya, mereka harus menerbitkan affirmative case yang mengidentifikasi risiko misalignment dan mitigasi sebelum melanjutkan scaling.

Claude Opus 4.6 tidak melewatinya menurut pengumuman v3.0. Dokumen menambahkan: "confidently ruling this out is becoming difficult." Frasa itu penting; ia mengakui bahwa ambang cukup dekat untuk menjadi masalah yang hidup, bukan batas spekulatif.

Pelajaran 6 (Automated Alignment Research) dan Pelajaran 7 (Recursive Self-Improvement) bermuara langsung ke ambang ini. Peneliti alignment otomatis yang melewati bar kualitas-riset adalah bukti bahwa ambang AI R&D-4 mendekat.

### Frontier Safety Roadmap dan Risk Report

v3.0 menaikkan dua jenis artefak menjadi dokumen tetap:

- **Frontier Safety Roadmap**: dokumen berwawasan-depan yang menjelaskan kerja keamanan yang direncanakan, ekspektasi kapabilitas, dan riset mitigasi.
- **Risk Report**: dokumen retrospektif atas model spesifik setelah rilis, menjelaskan kapabilitas yang teramati dan risiko residual.

Keduanya publik. Keduanya diperbarui pada kadens yang dideklarasikan. Utilitasnya adalah: pembaca bisa melacak bagaimana apa yang dikatakan Anthropic akan mereka lakukan dalam Roadmap dibandingkan dengan apa yang mereka laporkan dalam Risk Report.

### Menghapus klausul pause

RSP 2023 mencakup komitmen pause eksplisit: jika model melewati ambang kapabilitas spesifik, training akan berhenti sampai mitigasi ada. v3.0 mengganti pause eksplisit dengan formulasi yang lebih lembut (terbitkan affirmative case, lanjutkan jika mitigasi memadai). SaferAI dan analis lain menyebut ini langsung sebagai regresi terkuat dalam dokumen baru.

Argumen kebijakan untuk perubahan: ambang kuantitatif 2023 ternyata tidak terjangkau oleh benchmark kapabilitas era-2026 karena benchmark sendiri di-skala ulang. Argumen kontra: klausul pause dalam kebijakan scaling adalah perangkat komitmen; menghapusnya menghapus kredibilitas kebijakan.

### Penurunan SaferAI

SaferAI adalah organisasi independen yang menilai dokumen gaya-RSP. Peringkat publik mereka: RSP Anthropic 2023 mencetak 2,2 (dari skala tempat 4,0 adalah RSP terbaik saat ini dan 1,0 nominal). v3.0 mencetak 1,9. Ini memindahkan Anthropic dari "moderat" ke "lemah," bergabung dengan OpenAI dan DeepMind dalam kategori lemah.

Faktor penurunan menurut SaferAI:
- Ambang kualitatif menggantikan yang kuantitatif.
- Komitmen pause dihapus.
- Mitigasi ambang AI R&D-4 dideskripsikan sebagai "affirmative case" ketimbang tindakan spesifik.
- Mekanisme reviu bergantung pada Safety Advisory Group Anthropic, dengan pengawasan independen yang terbatas.

### Apa yang bukan pelajaran ini

Ini bukan pelajaran kepatuhan. RSP v3.0 bukan regulasi; tidak ada yang memaksa Anthropic mengikutinya. Pelajarannya adalah membaca dokumen dengan spesifisitas dan skeptisisme yang layak. Kebijakan scaling adalah sinyal publik utama yang dipancarkan lab frontier tentang postur risiko-katastropik. Membacanya dengan baik adalah keterampilan praktis bagi siapa pun yang pekerjaannya bergantung pada kapabilitas frontier.

```figure
a5-rsp-ladder
```

## Gunakan

`code/main.py` mengimplementasikan mesin keputusan kecil yang mencerminkan bentuk evaluasi-ambang RSP: dengan kandidat model dan set pengukuran kapabilitas, kembalikan apakah ambang AI R&D-4 dilewati, bagian affirmative-case yang disyaratkan, dan apakah deployment bisa lanjut. Ia sengaja sederhana; intinya membuat logika dokumen eksplisit.

## Rilis

`outputs/skill-scaling-policy-review.md` mereviu kebijakan scaling (Anthropic, OpenAI, DeepMind, atau internal) terhadap referensi v3.0: struktur dua-tier, ambang, komitmen pause, reviu independen.

## Latihan

1. Jalankan `code/main.py`. Masukkan tiga model sintetis pada level kapabilitas berbeda. Konfirmasi evaluator ambang berperilaku seperti diharapkan dan menghasilkan templat affirmative-case yang benar.

2. Baca RSP v3.0 penuh (32 halaman). Identifikasi setiap komitmen yang berada di tier "rekomendasi industri-luas". Komitmen mana yang tadinya "unilateral Anthropic" di v2?

3. Baca metodologi penilaian RSP SaferAI. Reproduksi skor 1,9 mereka untuk v3.0 dengan menerapkan rubriknya pada dokumen. Baris rubrik mana yang paling mendorong penurunan?

4. Komitmen pause 2023 dihapus. Usulkan komitmen pengganti yang menjaga kredibilitas kebijakan sambil mengakui soal skala-ulang benchmark 2026.

5. Bandingkan RSP v3.0 dengan Preparedness Framework v2 OpenAI (Pelajaran 20). Pilih satu area tempat v3.0 lebih kuat. Pilih satu area tempat Preparedness Framework lebih kuat.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| RSP | "Anthropic's scaling policy" | Responsible Scaling Policy; v3.0 effective Feb 24, 2026 |
| AI R&D-4 | "Research-automation threshold" | Capability to automate substantial AI research at competitive cost |
| Affirmative case | "Safety justification" | Published argument that risks are identified and mitigations adequate |
| Frontier Safety Roadmap | "Forward plan" | Standing document on planned safety work and expected capabilities |
| Risk Report | "Retrospective on a model" | Standing document on observed capability and residual risk after release |
| Two-tier mitigation | "Unilateral vs industry" | Anthropic commitments vs industry recommendations, separated |
| Pause commitment | "2023 clause" | Explicit promise to pause training; removed in v3.0 |
| SaferAI rating | "Independent RSP grade" | Third-party rubric; v3.0 scored 1.9 (v2 was 2.2) |

## Further Reading

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — the full 32-page policy.
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — summary of changes from v2.
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/responsible-scaling-policy/roadmap) — standing document linked from RSP v3.0.
- [Anthropic — Risk Report: February 2026](https://www.anthropic.com/feb-2026-risk-report) — retrospective on Claude Opus 4.6, the frontier model when it was published.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — connects AI R&D-4 to measured autonomy.
