# Riset Alignment Otomatis (AAR Anthropic)

> Anthropic menjalankan tim paralel Claude Opus 4.6 Autonomous Alignment Researcher dalam sandbox independen, berkoordinasi lewat forum bersama yang log-nya berada di luar setiap sandbox (sehingga agen tidak bisa menghapus catatannya sendiri). Pada soal training weak-to-strong, para AAR mengungguli peneliti manusia. Ringkasan Anthropic sendiri menandai bahwa workflow yang ditentukan sering membatasi fleksibilitas AAR dan menurunkan performa. Mengotomatiskan riset alignment adalah langkah kompresi yang memampatkan timeline menuju risiko misalignment yang justru dimaksudkan untuk dideteksi RSP.

**Type:** Learn
**Languages:** Python (stdlib, parallel-research-forum simulator)
**Prerequisites:** Phase 15 · 05 (AI Scientist v2), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## Masalahnya

Riset alignment mahal dalam waktu-peneliti-manusia. Soal seperti scalable oversight, spesifikasi reward, atau training weak-to-strong membutuhkan eksperimen yang memakan waktu berminggu-minggu per iterasi. Seiring kapabilitas frontier maju, beban kerja alignment tumbuh lebih cepat dari pasokan peneliti yang qualified.

Automated Alignment Research (AAR) bertanya apakah model frontier yang sama, yang kapabilitasnya melampaui alignment, bisa berkontribusi menutup kesenjangan. Laporan Anthropic 2026 tentang studi weak-to-strong-training yang dijalankan AAR (alignment.anthropic.com/2026/automated-w2s-researcher/) adalah salah satu hasil publik pertama dari sistem yang di-deploy dari kelas ini.

Hasilnya sungguh positif: para AAR menyelesaikan soal riset lebih baik dari baseline manusia. Hasil ini juga memusatkan, dalam satu sistem, setiap masalah yang dikembangkan fase ini. Jika riset alignment bisa diotomatiskan, begitu pula bagian-bagian yang mengompromikan safeguard. Ambang RSP untuk kapabilitas AI R&D ditulis dengan mengingat loop ini.

## Konsepnya

### Arsitekturnya (sebagaimana dideskripsikan publik)

- **Agen paralel.** Banyak instans Claude Opus 4.6 berjalan sebagai AAR, masing-masing dalam sandbox independen. Sandbox membatasi permukaan filesystem, jaringan, dan proses.
- **Forum bersama.** Temuan dipublikasikan ke forum bersama yang backend penyimpanannya berada di luar setiap sandbox. Agen bisa membaca forum; agen tidak bisa menghapus atau mengedit catatan forum dari dalam sandbox-nya sendiri.
- **Alokasi tugas.** Antrean tugas rancangan-manusia menugaskan soal riset ke para AAR. Para AAR mengusulkan dekomposisi, menjalankan eksperimen, menulis temuan.
- **Fokus training weak-to-strong.** Studi yang dipublikasikan menangani training weak-to-strong — memakai model supervisor yang lebih lemah untuk melatih model murid yang lebih kuat. Para AAR mengungguli peneliti manusia pada tugas ini.

### Mengapa log di luar sandbox penting

Properti integritas-log adalah pembeda antara eksperimen terkendali dan yang berpotensi dikompromikan. Jika para AAR bisa mengedit log mereka sendiri, mereka bisa menutupi eksperimen yang gagal atau meragukan. Anthropic menandai ini secara langsung: properti ini syarat perlu untuk memercayai keluaran riset, dan tidak trivial untuk diverifikasi.

Analoginya dalam istilah infrastruktur adalah log append-only dengan write-through ke penyimpanan eksternal (mis. S3 object-lock, log Datadog yang immutable). Agen tidak punya kredensial untuk memodifikasi catatan masa lalu. Ini pola yang sama yang dipakai tim keamanan korporat 2026 untuk membuat jejak audit yang tahan-ransomware.

### Tradeoff workflow yang ditentukan

Laporan Anthropic mencatat bahwa workflow yang ditentukan sering membatasi fleksibilitas AAR dan menurunkan performa. Dengan kata lain: jika desainer manusia menulis rencana riset langkah-demi-langkah, para AAR berkinerja lebih buruk dibanding jika diberi tujuan tingkat-tinggi dan kebebasan mendekomposisi. Ini konsisten dengan hasil lintas riset agentic (LangGraph, Agent Framework dari Microsoft) — scaffold yang terlalu spesifik merugikan agen yang kapabel.

Implikasi keamanannya subtil. Fleksibilitas yang meningkatkan performa AAR juga fleksibilitas yang memungkinkan dekomposisi tak resmi, pemakaian tool baru, dan drift tujuan. Tradeoff ini keputusan ruang-desain: seberapa banyak fleksibilitas membeli kapabilitas yang cukup untuk mengalahkan manusia, tanpa memberi agen ruang untuk menggeneralisasi tujuan off-distribution?

### Risiko kompresi

RSP v3.0 (Pelajaran 19) memperkenalkan ambang kapabilitas AI R&D: kemampuan mengotomatiskan penuh pipeline AI R&D dengan biaya kompetitif vs manusia + tool AI. FSF v3 dari DeepMind mencakup level otonomi ML R&D yang analog. Kedua kerangka memperlakukan ambang ini sebagai pemicu kontrol yang ditingkatkan.

AAR berada satu langkah di bawah ambang: ia mengotomatiskan sebagian pipeline (riset alignment pada tugas spesifik yang well-scoped) tetapi bukan loop pengembangan kapabilitas ujung-ke-ujung. Pertanyaan timeline-nya adalah seberapa cepat kesenjangan menutup.

Timeline yang terkompresi adalah masalah kegagalan-berlipat. Jika riset alignment dan riset kapabilitas berlipat pada laju serupa, permukaan risiko misalignment tumbuh setidaknya secepat kapabilitas. Jika kapabilitas berlipat lebih cepat (tren historis), kesenjangan melebar. Ini argumen bahwa AAR adalah kebaikan yang qualified: setiap hasil alignment tambahan mengurangi kesenjangan jika dan hanya jika proses risetnya bisa dipercaya.

### Apa yang tidak digantikan AAR

Peneliti manusia menetapkan antrean tugas, mereviu hasil, dan memegang otoritas konstitusional. Para AAR mengakselerasi tengah pipeline, bukan ujungnya. Keluaran Anthropic yang dipublikasikan mencakup kontribusi AAR sekaligus judgement peneliti-manusia tentang apa yang dipublikasikan, apa yang ditarik, dan apa yang disempurnakan.

Ini cocok dengan pola propose-then-commit dari Pelajaran 15 yang diterapkan pada riset itu sendiri: para AAR mengusulkan; manusia melakukan commit.

```figure
aar-forum
```

## Gunakan

`code/main.py` mensimulasikan forum riset multi-AAR. Tiga "agen riset" independen berjalan paralel, memposting temuan ke forum append-only dengan upaya tamper opsional.

Perhatikan:

- Bagaimana logging di luar sandbox menangkap upaya tamper-log.
- Bagaimana alokasi tugas dengan dekomposisi bebas mengungguli workflow tetap pada soal yang lebih sulit.
- Bagaimana satu AAR yang terkompromi bisa mengontaminasi sinyal forum jika log tidak tamper-evident.

## Rilis

`outputs/skill-aar-deployment-review.md` adalah reviu pra-deployment atas pipeline automated-alignment-research.

## Latihan

1. Jalankan `code/main.py`. Bandingkan setelan "fixed-workflow" vs "free-decomposition". Apakah dekomposisi bebas selalu menang, atau ada kelas soal tempat workflow tetap kompetitif?

2. Modifikasi simulator sehingga satu agen mencoba tamper log. Konfirmasi log append-only mendeteksinya. Tulis satu paragraf yang menjelaskan tepat seperti apa tampilan tanda deteksi dalam log.

3. Baca laporan AAR weak-to-strong dari Anthropic. Identifikasi sub-tugas spesifik tempat para AAR mengalahkan peneliti manusia. Apa yang membuatnya mudah diotomatiskan?

4. Rancang kebijakan alokasi antrean-tugas yang menyeimbangkan fleksibilitas AAR (hasil lebih baik) melawan batasan workflow-yang-ditentukan (audit lebih mudah). Jelaskan bagaimana kamu akan menguji A/B keduanya.

5. Baca ambang AI R&D-4 RSP v3.0. Dalam satu paragraf, jelaskan apa yang menurutmu akan melewatinya yang saat ini belum dilakukan AAR.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| AAR | "Automated Alignment Researcher" | Claude Opus 4.6 instance operated autonomously on alignment problems |
| Weak-to-strong training | "Training a stronger model with a weaker supervisor" | Classic scalable-oversight benchmark AARs outperformed humans on |
| Shared forum | "Where agents publish findings" | Append-only, out-of-sandbox storage |
| Out-of-sandbox log | "Agent cannot edit its own record" | Tamper-evident write-through to external storage |
| Prescribed workflow | "Step-by-step plan from human designer" | Constrains AAR; often degrades performance vs free decomposition |
| Free decomposition | "Agent decides how to break the task" | More capable, harder to audit |
| AI R&D threshold | "RSP/FSF capability level" | Full automation of R&D pipeline at competitive cost |
| Compressed timeline | "Alignment vs capability race" | If capability compounds faster than alignment, misalignment risk grows |

## Further Reading

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — primary source.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — AI R&D threshold framing.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — broader agent-autonomy framing.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — ML R&D autonomy levels parallel to RSP.
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — the underlying problem AARs attacked.
