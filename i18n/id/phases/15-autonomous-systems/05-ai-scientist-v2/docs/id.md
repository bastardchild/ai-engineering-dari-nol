# AI Scientist v2 — Riset Otonom Level Workshop

> AI Scientist v2 dari Sakana (Yamada et al., arXiv:2504.08066) menjalankan loop riset penuh: hipotesis, kode, eksperimen, gambar, naskah, submisi. Ia adalah sistem pertama yang makalah buatannya lolos peer review di workshop ICLR 2025. Evaluasi independen (Beel et al.) menemukan 42% eksperimen gagal karena kesalahan koding dan tinjauan literatur kerap salah melabeli konsep mapan sebagai novel. Dok Sakana sendiri memperingatkan bahwa codebase mengeksekusi kode tulisan-LLM dan merekomendasikan isolasi Docker. Kedua sisi gambaran itu adalah intinya.

**Type:** Learn
**Languages:** Python (stdlib, research-loop state-machine toy)
**Prerequisites:** Phase 15 · 03 (AlphaEvolve), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## Masalahnya

Riset adalah tugas terbuka. Tidak seperti pencarian algoritmik AlphaEvolve atau modifikasi-diri DGM yang dibatasi benchmark, hasil riset tidak punya kriteria kebenaran yang machine-checkable. Sebuah makalah dinilai reviewer, bukan unit test. Itu membuat loop lebih sulit ditutup — dan lebih berharga jika tertutup, karena riset adalah tempat kemajuan berlipat berada.

AI Scientist v1 (Sakana, 2024) menutup loop dengan berangkat dari templat tulisan-manusia. LLM mengisi eksperimen di dalam scaffolding tetap. AI Scientist v2 (Yamada et al., 2025) melepas syarat templat dengan memakai agentic tree search plus loop kritik vision-language model. Sistem menghasilkan ide, mengimplementasikan eksperimen, menghasilkan gambar, menulis makalah, dan beriterasi atas umpan balik reviewer.

Vonis peer review: satu makalah buatan v2 diterima di workshop ICLR 2025 (dengan disclosure). Vonis evaluasi independen: sistem ini jauh dari andal. Keduanya benar.

## Konsepnya

### Arsitekturnya

1. **Generasi ide.** LLM mengusulkan ide riset yang dikondisikan pada topik dan literatur sebelumnya. v1 memakai templat; v2 memakai pencarian agentic atas ruang hipotesis.
2. **Cek kebaruan.** Langkah retrieval literatur memeriksa apakah ide sudah dipublikasikan. Di langkah inilah evaluasi Beel et al. menemukan mislabeling — metode mapan kerap diklasifikasi sebagai novel.
3. **Rencana eksperimen.** Agen menyusun protokol eksperimental dan menulis kode.
4. **Eksekusi.** Kode berjalan di sandbox. Kegagalan diumpankan kembali ke loop retry. Dalam pengukuran Beel et al., 42% eksperimen gagal karena kesalahan koding pada tahap ini.
5. **Generasi gambar.** Vision-language model membaca gambar yang dihasilkan dan menulis ulangnya agar jelas. Ini tambahan teknis kunci v2.
6. **Penulisan naskah.** LLM menyusun draf makalah, beriterasi dengan reviewer internal.
7. **Opsional: submisi.** Makalah disubmit ke sebuah venue.

### Apa arti hasil lolos-workshop itu

Satu makalah buatan v2 lolos peer review di workshop ICLR 2025. Para penulis men-disclose asal makalah itu ke komite program. Penerimaan itu sebuah data point; ia bukan lisensi untuk mengklaim sistem "mengerjakan riset."

Konteks penting: makalah workshop adalah bar yang lebih rendah dibanding makalah konferensi utama. Peer review itu berisik; sebagian kecil submisi diterima pada hari tertentu. Satu keberhasilan adalah proof of concept, bukan klaim reliabilitas. Makalah Nature 2026 mendokumentasikan loop ujung-ke-ujung dan sendiri ditulis bersama peneliti manusia; ia bukan "sistem menulis makalah Nature."

### Apa yang ditemukan evaluasi independen

Beel et al. (arXiv:2502.14297) menjalankan evaluasi eksternal. Temuan utama:

- **Kegagalan eksperimen.** 42% eksperimen gagal karena kesalahan koding (import buruk, shape mismatch, variabel tak terdefinisi). Loop retry menangkap sebagian, tidak semua.
- **Mislabeling kebaruan.** Langkah retrieval-literatur kerap menandai konsep mapan sebagai novel. Ini padanan riset dari halusinasi.
- **Kesenjangan kualitas presentasi.** Kritik gambar vision-language menghasilkan visual kelas publikasi, menutupi kelemahan eksperimental yang mendasarinya.

Temuan terakhir adalah yang terpenting untuk fase ini. Sistem yang menghasilkan keluaran meyakinkan tanpa mengerjakan riset yang meyakinkan lebih berbahaya, bukan lebih aman, dibanding yang gagal secara jelas. Evaluasi harus menjangkau klaim yang mendasari, bukan berhenti pada gambarnya.

### Masalah sandbox-escape

README repositori Sakana sendiri memperingatkan:

> Due to the nature of this software, which executes LLM-generated code, we cannot guarantee safety. There are risks of dangerous packages, uncontrolled web access, and spawning of unintended processes. Use at your own risk and consider Docker isolation.

Ini bentuk operasional otonomi di domain yang tidak terverifikasi. LLM menulis kode; kode berjalan; kode bisa melakukan apa pun yang diizinkan prosesnya. Tanpa sandbox yang membatasi filesystem, jaringan, dan aksi proses secara keras, agen riset mandiri apa pun bisa mengeksfiltrasi data, membakar komputasi, atau menulis ulang dirinya sendiri.

Cerita sandbox AlphaEvolve lebih mudah karena evaluator-nya ketat. Loop AI Scientist v2 menjalankan kode terbuka dengan tujuan terbuka. Itu sebabnya ia butuh isolasi lebih kuat (minimum Docker; seccomp / gVisor lebih disarankan) dan reviu manual atas setiap submisi sebelum keluar dari sistem.

### Di mana v2 berada dalam stack frontier

| System | Target | Output kind | Evaluator | Known failure |
|---|---|---|---|---|
| AlphaEvolve | algorithms | code | unit + benchmark | bounded by evaluator rigor |
| DGM | agent scaffolding | code | SWE-bench | reward hacking |
| AI Scientist v2 | research papers | text + code + figures | peer review (weak) | experiment failures, mislabeling, polish masking weakness |

v2 punya evaluator otomatis terlemah dari ketiganya, permukaan keluaran terluas, dan jalur terpendek ke artefak publik. Kontrol operasional (sandbox, reviu, disclosure) mengerjakan sebagian besar kerja keamanan.

```figure
mx-research-loop
```

## Gunakan

`code/main.py` mensimulasikan loop v2 sebagai state machine: ide → cek kebaruan → eksperimen → gambar → naskah → reviu → terima-atau-iterasi. Setiap state punya probabilitas kegagalan yang bisa dikonfigurasi dari temuan Beel et al. Jalankan simulator untuk N loop dan hitung:

- Berapa ide mencapai submisi.
- Berapa submisi yang cacat eksperimental kritisnya disembunyikan makalah yang dipoles.
- Bagaimana anggaran retry mempertukarkan kualitas vs hasil.

## Rilis

`outputs/skill-ai-scientist-sandbox-review.md` adalah checklist reviu dua-gerbang untuk apa pun yang dihasilkan agen loop-riset sebelum keluar dari sandbox.

## Latihan

1. Jalankan `code/main.py` dengan parameter default. Berapa fraksi proses loop yang menghasilkan makalah "bersih"? Berapa fraksi yang menghasilkan makalah dengan cacat kegagalan-eksperimen yang dipoles kritik-gambar?

2. Default sudah memakai 42% / 25% dari Beel et al. Jalankan ulang dengan `--experiment-failure 0.20 --novelty-mislabel 0.10` lalu dengan `--experiment-failure 0.60 --novelty-mislabel 0.40`. Bagaimana porsi dipoles-tetapi-cacat bergeser antara kedua proses?

3. Baca README repo AI Scientist v2 dari Sakana tentang syarat sandbox. Sebutkan dua restriksi tambahan (selain Docker) yang akan kamu terapkan untuk proses otonom multi-hari.

4. Baca Beel et al. Bagian 4 tentang kesenjangan kualitas-presentasi. Rancang satu evaluator tambahan yang akan menangkap makalah yang tampak dipoles tetapi cacat secara eksperimental.

5. Usulkan protokol reviu-manusia untuk keluaran agen-riset yang lebih skalabel dibanding "seorang PhD membaca setiap makalah." Identifikasi bottleneck-nya dan rancang solusinya.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| AI Scientist v1 | "Sakana's templated research agent" | Filled experiments into a fixed scaffold |
| AI Scientist v2 | "Template-free research agent" | Agentic tree search with VLM figure critique |
| Agentic tree search | "Branching research agent" | Expands multiple experiment plans in parallel; prunes by internal critic |
| Vision-language critique | "VLM polish on figures" | Multimodal model reads figures and rewrites them for clarity |
| Literature retrieval | "Novelty check" | Searches prior work to confirm idea novelty — documented to mislabel |
| Polish masking | "Pretty paper, broken research" | Presentation quality exceeds experimental quality; hides weaknesses |
| Sandbox escape | "LLM code breaks out" | Agent-executed code does things the loop designer did not intend |

## Further Reading

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — paper.
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) — vendor summary with peer-review context.
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — external evaluation numbers.
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) — the templated predecessor.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — broader framing of open-ended research agents.
