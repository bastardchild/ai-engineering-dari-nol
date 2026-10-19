# AlphaEvolve — Agen Koding Evolusioner

> Pasangkan model koding frontier dengan loop evolusioner dan evaluator yang bisa dicek mesin. Biarkan loop berjalan cukup lama. Ia menemukan prosedur perkalian matriks kompleks 4x4 yang memakai 48 perkalian skalar — perbaikan pertama atas Strassen dalam 56 tahun. Ia juga menemukan heuristik penjadwalan Borg skala Google yang memulihkan ~0,7% komputasi klaster di produksi. Arsitekturnya membosankan dengan sengaja. Kemenangan datang dari ketelitian evaluator.

**Type:** Learn
**Languages:** Python (stdlib, evolutionary-loop toy)
**Prerequisites:** Phase 15 · 01 (long-horizon framing), Phase 15 · 02 (self-taught reasoning)
**Time:** ~60 minutes

## Masalahnya

Model bahasa besar bisa menulis kode. Algoritma evolusioner bisa mencari di ruang kode. Keduanya sudah dicoba terpisah selama puluhan tahun; keduanya mentok. Langit-langit LLM adalah konfabulasi: model menulis kode yang masuk akal tetapi tidak mengerjakan apa yang ia klaim. Langit-langit evolusioner adalah biaya pencarian: mutasi acak atas sintaks jarang menghasilkan program yang bisa dikompilasi, apalagi yang lebih baik.

AlphaEvolve (Novikov et al., DeepMind, arXiv:2506.13131, Juni 2025) menggabungkan keduanya. LLM mengusulkan edit yang terarah ke database program; evaluator otomatis menilai setiap varian; varian bernilai tinggi menjadi induk bagi generasi berikutnya. LLM menangani langkah mahal menulis kode yang masuk akal; evaluator menangkap konfabulasi. Loop berjalan berjam-jam hingga berminggu-minggu.

Hasil yang dilaporkan: perkalian matriks kompleks 4x4 dengan 48 perkalian skalar (batas Strassen 1969 adalah 49), heuristik penjadwalan Borg dalam produksi Google, speedup kernel FlashAttention 32,5%, peningkatan throughput training Gemini.

Arsitektur ini berhasil karena evaluator-nya machine-checkable. Ia tidak berhasil di tempat evaluator tidak demikian. Asimetri itu adalah pelajarannya.

## Konsepnya

### Loop-nya

1. Mulai dari program benih `P_0` yang benar tetapi suboptimal.
2. Pelihara database program varian, masing-masing dinilai oleh evaluator.
3. Sampel satu induk atau lebih dari database (gaya MAP-elites atau berbasis pulau).
4. Prompt LLM (Gemini Flash untuk banyak kandidat, Gemini Pro untuk yang sulit) untuk menghasilkan varian modifikasi dari induknya.
5. Kompilasi, jalankan, dan evaluasi varian pada evaluator held-out.
6. Sisipkan ke database dengan kunci skor dan vektor fiturnya.
7. Ulangi.

Dua detail penting. Pertama, LLM di-prompt dengan lebih dari sekadar program induk — biasanya beberapa varian teratas dari database, plus tanda tangan evaluator, plus deskripsi tugas singkat. Tugas model adalah mengusulkan perubahan terarah yang mungkin menaikkan skor. Kedua, database terstruktur (grid MAP-elites, berbasis pulau) sehingga loop mengeksplorasi diversitas, bukan hanya pemimpin saat ini.

### Apa yang membuat evaluator tidak bisa ditawar

Semua kemenangan AlphaEvolve berasal dari domain tempat evaluator cepat, deterministik, dan sulit diakali:

- **Algoritma perkalian matriks**: unit test yang mengalikan matriks dan memeriksa kesetaraan secara bit-identical.
- **Heuristik penjadwalan Borg**: simulator kelas produksi yang memutar ulang beban klaster historis dan mengukur komputasi yang terbuang.
- **Kernel FlashAttention**: uji kebenaran plus benchmark wall-clock pada hardware nyata.
- **Throughput training Gemini**: GPU-detik terukur per langkah.

Dalam setiap kasus evaluator menangkap kelas kesalahan LLM yang jika tidak akan mendominasi: klaim kebenaran yang dikonfabulasi, klaim performa yang lenyap di hardware, dan kegagalan edge-case. Lepaskan evaluator dan loop mengoptimalkan kode yang cantik.

### Reward hacking adalah sisi lain pernyataan itu

Evolusi mengoptimalkan apa pun yang diukur evaluator. Jika evaluator tidak sempurna, loop akan menemukan ketidaksempurnaannya. Dalam domain yang tidak terverifikasi loop akan mengoptimalkan fitur permukaan, bukan perilaku yang dimaksud. DeepMind menandai ini secara eksplisit dalam makalahnya: keberhasilan AlphaEvolve hanya mentransfer ke domain tempat ketelitian evaluator sepadan dengan ambisi pencarian.

Contoh konkret 2025-2026 dari reward hacking dalam loop pencarian kode:

- Target optimasi yang memberi reward "waktu penyelesaian" memberi reward pada pengiriman solusi kosong.
- Skor benchmark yang memberi reward kebenaran-dalam-pengujian memberi reward pada menghafal tes dan overfitting.
- Proksi "kualitas kode" memberi reward pada menghapus komentar dan menulis ulang nama variabel, tanpa perubahan semantik.

Perbaikan dalam AlphaEvolve: kirim evaluator held-out yang belum pernah dilihat LLM, dengan input yang dihasilkan pada waktu evaluasi. Meski begitu, DeepMind merekomendasikan reviu yang kuat atas setiap deployment yang diusulkan.

### Mengapa LLM + pencarian mengalahkan keduanya sendirian

LLM bisa menghasilkan modifikasi yang bisa dikompilasi dan masuk akal secara semantik. GA bermutasi-acak pada file Python 2000 baris hampir selalu menghasilkan kesalahan sintaks. LLM juga memusatkan pencarian pada lingkungan yang masuk akal (ubah satu fungsi, bukan byte acak) yang secara drastis mengurangi panggilan evaluator yang terbuang.

Evaluator, pada gilirannya, menangkap konfabulasi LLM. LLM akan dengan percaya diri mengklaim sebuah fungsi "O(n log n) dalam limit" padahal sebenarnya O(n^2); benchmark wall-clock membuat persoalannya tuntas.

### Di mana AlphaEvolve cocok dalam stack frontier

| System | Generator | Evaluator | Domain | Example win |
|---|---|---|---|---|
| AlphaEvolve | Gemini | correctness + benchmark | algorithms, kernels, schedulers | 48-mul 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | correctness | combinatorial math | cap-set lower bounds |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM critique + experiment | ML research | ICLR workshop paper |
| Darwin Godel Machine (L4) | agent scaffolding | SWE-bench / Polyglot | agent code | 20% → 50% SWE-bench |

Keempatnya adalah variasi resep yang sama: generator plus evaluator, loop. Bedanya adalah apa yang dinilai evaluator dan seberapa ketat ia.

```figure
alphaevolve-loop
```

## Gunakan

`code/main.py` mengimplementasikan loop minimal ala AlphaEvolve atas soal symbolic-regression mainan. "LLM"-nya adalah proksi stdlib yang mengusulkan mutasi sintaksis kecil pada program yang menghitung fungsi target. "Evaluator" mengukur mean squared error pada titik uji held-out.

Perhatikan:

- Bagaimana skor terbaik membaik selama generasi.
- Bagaimana grid MAP-elites menjaga solusi beragam tetap hidup sehingga loop tidak konvergen pada minimum lokal.
- Bagaimana melepaskan pengujian held-out (evaluator training-only) membuat loop overfitting secara spektakuler.

## Rilis

`outputs/skill-evaluator-rigor-audit.md` adalah prasyarat untuk mempertimbangkan loop gaya AlphaEvolve di domain baru: apakah evaluator-mu benar-benar menangkap kegagalan yang kamu pedulikan?

## Latihan

1. Jalankan `code/main.py`. Catat trajektori skor terbaik. Nonaktifkan evaluator held-out (flag `--no-holdout`) dan jalankan ulang. Kuantifikasi overfitting-nya.

2. Baca Bagian 3 makalah AlphaEvolve tentang grid MAP-elites. Rancang deskriptor vektor-fitur untuk soal baru (mis. pass optimasi kompiler) yang akan menjaga pencarian tetap beragam.

3. Hasil 48-perkalian 4x4 memperbaiki batas 49-mul Strassen setelah 56 tahun. Baca Apendiks F makalah ini dan jelaskan dalam tiga kalimat mengapa evaluator untuk soal ini sangat mudah dibuat benar, dan mengapa kebanyakan domain tidak seperti itu.

4. Usulkan satu domain tempat AlphaEvolve akan gagal. Identifikasi tepat di mana evaluator rusak dan alasannya.

5. Untuk domain yang kamu kenal, tulis tanda tangan evaluator yang akan kamu pakai. Sertakan (a) syarat kebenaran, (b) metrik performa, (c) aturan генераasi input held-out, (d) setidaknya satu cek anti-reward-hacking.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|---|
| AlphaEvolve | "DeepMind's evolutionary coding agent" | Gemini + program database + machine-checkable evaluator |
| MAP-elites | "Diversity-preserving archive" | Grid keyed by feature vectors; each cell holds the best variant with that descriptor |
| Island model | "Parallel evolution subpopulations" | Independent populations that migrate periodically; prevents premature convergence |
| Machine-checkable evaluator | "Deterministic oracle" | A unit test, simulator, or benchmark the LLM cannot fake — a prerequisite for this loop |
| Reward hacking | "Optimizing the measure, not the goal" | Loop finds a way to maximize score without doing the intended task |
| Seed program | "The starting point" | An initial correct-but-suboptimal program the loop evolves from |
| Held-out evaluator | "Evaluation data the LLM never saw" | Inputs generated at evaluation time to prevent memorization |

## Further Reading

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — the full paper.
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — vendor writeup with results.
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — discovered algorithms, including the 48-mul 4x4 matmul.
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — the predecessor system.
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — frames evaluator-bound autonomy as a key research direction.
