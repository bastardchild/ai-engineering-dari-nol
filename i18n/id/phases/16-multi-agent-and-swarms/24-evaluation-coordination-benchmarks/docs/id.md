# Benchmark Evaluasi dan Koordinasi

> Lima benchmark 2025-2026 mencakup ruang evaluasi multi-agent. **MultiAgentBench / MARBLE** (ACL 2025, arXiv:2503.01935) mengevaluasi topologi star/chain/tree/graph dengan KPI berbasis milestone; **graph terbaik untuk riset**, cognitive planning menambah ~3% pencapaian milestone. **COMMA** mengevaluasi koordinasi multimodal informasi-asimetris; model state-of-the-art termasuk GPT-4o kesulitan mengalahkan baseline acak. **MedAgentBoard** (arXiv:2505.12371) mencakup empat kategori tugas medis dan sering menemukan multi-agent tidak mendominasi LLM tunggal. **AgentArch** (arXiv:2509.10769) menbenchmark arsitektur agen perusahaan yang menggabungkan tool-use + memory + orkestrasi. **SWE-bench Pro** ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)) punya 1865 masalah di 41 repo yang mencakup aplikasi bisnis, layanan B2B, dan alat pengembang; model frontier memperoleh skor ~23% di Pro versus 70%+ di Verified — sebuah reality check atas kontaminasi. Claude Opus 4.7 (April 2026) dilaporkan **64,3%** di Pro dengan koordinasi agent-teams eksplisit (sumber primer Anthropic belum diterbitkan — perlakukan sebagai preliminary); Verdent (agent scaffold) mencapai **76,1% pass@1** di Verified ([laporan teknis Verdent](https://www.verdent.ai/blog/swe-bench-verified-technical-report)). **AAAI 2026 Bridge Program WMAC** (https://multiagents.org/2026/) adalah titik fokus komunitas 2026. Pelajaran ini dibangun di atas metrik MARBLE, menjalankan sweep topologi-vs-metrik, dan mengukir aturan "hanya lulus SWE-bench Verified bukan bukti generalisasi".

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 15 (Voting and Debate Topology), Phase 16 · 23 (Failure Modes)
**Time:** ~75 minutes

## Masalah

Ketika sebuah paper mengklaim "sistem multi-agent kami lebih baik," pertanyaannya: lebih baik dari apa, pada apa, diukur bagaimana? Era 2023-2024 evaluasi multi-agent adalah kekacauan — semua orang memilih metrik, baseline, dan set tugasnya sendiri. Benchmark 2025-2026 memaksakan struktur.

Tanpa benchmark bersama, kamu tidak bisa membandingkan dua sistem multi-agent secara bermakna. Lebih buruk lagi, tanpa benchmark hold-out, model frontier bisa terkontaminasi. SWE-bench Verified menjadi sebagian terkontaminasi di korpus training pada pertengahan 2025; skor frontier membengkak; Pro dirancang sebagai reality check yang belum terkontaminasi.

Pelajaran ini mengenumerasi lima benchmark kanonis 2026, menyebut apa yang diukur tiap benchmark, dan mengajarmamu membaca klaim benchmark secara skeptis.

## Konsep

### MultiAgentBench (MARBLE) — ACL 2025

arXiv:2503.01935. Mengevaluasi empat topologi koordinasi (star, chain, tree, graph) pada tugas riset, coding, dan perencanaan. KPI berbasis milestone melacak kemajuan sebagian, bukan hanya keberhasilan akhir.

Hasil terukur:

- Topologi **Graph** terbaik untuk skenario riset; mendukung kritik any-to-any.
- **Chain** terbaik untuk coding refinement bertahap.
- **Star** terbaik untuk konsolidasi cepat-faktual.
- **Coordination tax** muncul melewati ~4 agen pada graph.
- **Cognitive planning** menambah ~3% pencapaian milestone di semua topologi.

Gunakan ketika: kamu ingin membandingkan topologi koordinasi secara apples-to-apples. Repo MARBLE (https://github.com/ulab-uiuc/MARBLE) menyediakan evaluator-nya.

### COMMA — informasi multimodal asimetris

Mencakup tugas di mana agen punya modalitas observasi berbeda dan harus berkoordinasi tanpa berbagi informasi penuh. Hasil yang dilaporkan tidak nyaman: model frontier termasuk GPT-4o kesulitan mengalahkan **baseline acak** pada kolaborasi agen-agen di COMMA. Sinyalnya adalah modalitas multi-agent kurang dilatih dan kurang dievaluasi — LLM menangani kerja sama single-modality secara wajar; koordinasi multi-modality runtuh.

Gunakan ketika: sistem kamu punya koordinasi multimodal atau informasi-asimetris. Hasil nol dari COMMA adalah peringatan untuk mengukur sebelum mengklaim.

### MedAgentBoard — uji tekan domain

arXiv:2505.12371. Empat kategori tugas medis: diagnosis, treatment planning, pembuatan laporan, komunikasi pasien. Membandingkan multi-agent vs LLM tunggal vs sistem berbasis aturan konvensional.

Temuan: multi-agent TIDAK mendominasi LLM tunggal di sebagian besar kategori. Keunggulan multi-agent sempit — dekomposisi tugas membantu ketika subtugasnya jelas terpisah (diagnosis + perawatan); merugikan ketika overhead koordinasi melebihi keuntungan spesialisasi (pembuatan laporan).

Gunakan ketika: domain kamu punya baseline LLM-tunggal yang jelas. Jika pelajaran MedAgentBoard tergeneralisasi, banyak sistem multi-agent yang diusulkan over-engineered.

### AgentArch — arsitektur perusahaan

arXiv:2509.10769. Setting perusahaan dengan tool use, memory, dan orkestrasi berlapis bersamaan. Benchmark mengisolasi kontribusi tiap lapisan: seberapa besar penambahan tools membantu? Menambah memory? Menambah orkestrasi multi-agent?

Gunakan ketika: kamu merancang stack agen perusahaan dan perlu membenarkan tiap lapisan. AgentArch membantu menghindari membeli fitur yang nilainya tidak bisa kamu ukur.

### SWE-bench Pro — reality check-nya

arXiv:2509.16941. 1865 masalah di 41 repositori yang mencakup aplikasi bisnis, layanan B2B, dan alat pengembang. Dirancang untuk **belum terkontaminasi** pada cutoff training yang lebih lambat. Model frontier memperoleh skor ~23% di Pro versus 70%+ di Verified. Selisihnya adalah sinyal kontaminasi.

Skor April 2026:
- Claude Opus 4.7 di Pro: **64,3%** (dilaporkan dengan koordinasi agent-teams eksplisit; sumber primer Anthropic belum diterbitkan — perlakukan sebagai preliminary).
- Verdent (agent scaffold) di Verified: **76,1% pass@1** ([laporan teknis](https://www.verdent.ai/blog/swe-bench-verified-technical-report)).
- Skor mentah frontier di Pro tanpa agent scaffolding: ~23-35% ([paper SWE-bench Pro](https://arxiv.org/abs/2509.16941)).

Intinya: "kami mengalahkan SWE-bench Verified" bukan lagi bukti kemampuan. Pro adalah tes pengungkapan saat ini. Agent-team scaffolding menghasilkan keuntungan yang terukur di Pro (selisih ~30-40 poin), yang merupakan salah satu argumen empiris terkuat untuk koordinasi multi-agent pada 2026.

### AAAI 2026 WMAC

AAAI 2026 Bridge Program — Workshop on Multi-Agent Coordination (https://multiagents.org/2026/). Titik fokus komunitas riset AI multi-agent 2026. Paper yang diterima dan prosiding workshop adalah wahana kanonis untuk mengevaluasi metode baru; utamakan klaim yang diterima WMAC di atas preprint arXiv untuk keputusan produksi.

### Baca klaim benchmark secara skeptis — checklist 2026

Ketika seseorang mengklaim sebuah hasil multi-agent:

1. **Benchmark mana, split mana?** SWE-bench Verified vs Pro sangat berbeda. Angka yang dilaporkan pada split yang salah tidak berguna.
2. **Cek kontaminasi.** Apakah benchmark dirilis setelah cutoff training modelnya? Jika tidak, perlakukan dengan hati-hati.
3. **Perbandingan baseline.** Terhadap baseline LLM-tunggal, terhadap acak, terhadap kerja multi-agent sebelumnya. Bukan "terhadap versi tanpa-tuning dari sistem yang sama."
4. **Signifikansi statistik.** N percobaan, p-value, confidence interval. Model frontier variansnya tinggi; satu kali run menyesatkan.
5. **Keragaman tugas.** Satu tugas atau banyak? Generalisasi penting untuk produksi.
6. **Pengungkapan biaya.** Token per tugas, wall-clock. Solusi 90% dengan biaya 20x adalah keputusan bisnis, bukan klaim kemampuan.

### Yang tidak diukur baik oleh benchmark mana pun

- **Koordinasi horizon panjang.** Interaksi wall-clock selama berhari-hari. Semua benchmark saat ini berjalan pendek.
- **Ketahanan advesarial.** Apa yang terjadi ketika satu agen berbahaya atau terkompromikan?
- **Drift saat deployment.** Benchmark bersifat statis; distribusi produksi bergeser.
- **Performa ternormalisasi-biaya.** Sebagian besar benchmark melaporkan akurasi mentah, bukan akurasi-per-dolar.

Membangun benchmark internal sendiri untuk sumbu yang benar-benar kamu pedulikan sering kali adalah langkah yang tepat.

```figure
a5-bench-gap
```

## Bangun Sendiri

`code/main.py` adalah walk-through non-interaktif:

- Mensimulasikan 3 sistem multi-agent pada tugas mainan.
- Menghitung metrik milestone bergaya MARBLE untuk tiap sistem.
- Menjalankan cek kontaminasi dengan menahan tugas dari set "training".
- Membandingkan secara eksplisit dengan baseline acak.
- Mencetak scorecard klaim-benchmark.

Jalankan:

```bash
python3 code/main.py
```

Output yang diharapkan: scorecard sistem dengan akurasi mentah, pencapaian milestone, biaya-per-tugas, delta terhadap baseline acak, dan catatan cek-kontaminasi.

## Gunakan

`outputs/skill-benchmark-reader.md` membaca klaim benchmark multi-agent apa pun dan menerapkan checklist kewaspadaan. Output: sebuah grade dan caveat.

## Rilis

Disiplin evaluasi produksi:

- **Bangun benchmark internal** yang mencerminkan distribusi produksi nyatamu. Benchmark publik menginformasi tapi tidak menggantikan.
- **Sertakan baseline acak** di setiap perbandingan. Jika kamu tidak bisa mengalahkan acak dengan margin besar pada tugas koordinasi, tugasnya mungkin tidak well-posed.
- **Laporkan biaya bersama akurasi.** Biaya token dan wall-clock. Tim ops butuh keduanya.
- **Bangun ulang benchmark tiap kuartal.** Distribusi produksi bergeser; benchmark basi menyesatkan.
- **Hindari overfitting ke benchmark terpublikasi.** Jika timmu mengoptimalkan khusus untuk angka SWE-bench Pro, kamu akan regres di produksi.

## Latihan

1. Jalankan `code/main.py`. Identifikasi sistem mana dari tiga sistem simulasi yang punya biaya-per-milestone terbaik. Apakah itu cocok dengan sistem berakurasi-mentah tertinggi?
2. Baca MultiAgentBench (arXiv:2503.01935). Untuk domain tasmu sendiri, tentukan topologi mana dari keempatnya yang akan direkomendasikan MARBLE. Benarkan berdasarkan hasil paper-nya.
3. Baca paper SWE-bench Pro. Apa secara spesifik yang membuatnya tahan-kontaminasi? Apakah teknik yang sama bisa diterapkan pada benchmark lain yang kamu pedulikan?
4. Baca temuan COMMA tentang koordinasi multimodal. Rancang satu tugas koordinasi multimodal sederhana yang bisa kamu tambahkan ke benchmark internal. Apa yang dihitung sebagai sinyal yang berguna?
5. Terapkan checklist klaim-benchmark pada hasil utama satu paper multi-agent baru-baru ini. Grade apa yang akan kamu berikan pada klaim itu?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MARBLE | "MultiAgentBench" | ACL 2025; star/chain/tree/graph topologies with milestone KPIs. |
| COMMA | "Multimodal benchmark" | Multimodal asymmetric-info coordination; frontier models struggle vs random. |
| MedAgentBoard | "Domain stress test" | Four medical categories; often finds multi-agent does not dominate single-LLM. |
| AgentArch | "Enterprise benchmark" | Tools + memory + orchestration layered. |
| SWE-bench Pro | "Contamination-resistant" | 1865 problems, 41 repos; ~23% vs 70%+ on Verified (the contamination signal). |
| Milestone achievement | "Partial credit" | Benchmarks that reward progress, not only final success. |
| Contamination | "Benchmark leaked into training" | Post-release, benchmarks drift into training corpora; scores inflate. |
| WMAC | "AAAI 2026 Bridge Program" | Workshop on Multi-Agent Coordination; community focal point. |

## Bacaan Lanjutan

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — topology benchmark with milestone KPIs
- [MARBLE repository](https://github.com/ulab-uiuc/MARBLE) — reference implementation
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — domain stress test; multi-agent often does not dominate
- [AgentArch](https://arxiv.org/abs/2509.10769) — enterprise agent architectures
- [SWE-bench leaderboards](https://www.swebench.com/) — Verified and Pro scores for frontier models
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — the 2026 community focal point
