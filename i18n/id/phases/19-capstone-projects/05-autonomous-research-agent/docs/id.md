# Capstone 05 — Agen Riset Otonom (Kelas AI-Scientist)

> AI-Scientist-v2 dari Sakana menerbitkan makalah utuh. Agent Laboratory menjalankan eksperimennya. Allen AI membagikan trace-nya. Bentuk 2026 adalah tree search plan-execute-verify di atas eksperimen, biaya yang dianggarkan, eksekusi kode yang di-sandbox, penulis LaTeX dengan feedback visual, dan ensemble reviewer bergaya NeurIPS yang otomatis. Tugas capstone adalah membangun satu, menjalankannya dari ujung ke ujung dalam $30 per makalah, dan bertahan dari red team sandbox-escape yang didokumentasikan Sakana.

**Type:** Capstone
**Languages:** Python (agent + sandbox), LaTeX (output)
**Prerequisites:** Phase 2 (ML), Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 18 (safety)
**Phases exercised:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**Time:** 40 hours

## Masalahnya

Agen riset otonom melewati sebuah ambang pada 2026. AI-Scientist-v2 milik Sakana AI diterbitkan di Nature dengan makalah hasil generate yang lolos peer review workshop. ShinkaEvolve (ICLR 2026) melanjutkan garis itu ke evolusi hipotesis. Agent Laboratory milik AMD merilis trace yang bisa direproduksi. Agen-agen ini bukan sihir — itu adalah loop plan-execute-verify yang berjalan di atas pohon kandidat eksperimen, lengkap dengan batas biaya, sandbox yang terikat seed, dan review otomatis. Kerajinannya ada di loop, anggaran, dan cerita keselamatannya.

Kamu mempelajari loop itu dengan mengimplementasikannya terhadap sebuah ide awal di domain yang sempit (misalnya, ablation attention-sparsity pada transformer 100M parameter). nilainya bukan pada penemuan sesuatu yang baru di run pertama. Nilainya ada pada infrastrukturnya: tree-search, sandbox eksperimen, loop penulis-reviewer, laporan red team. Tim Sakana mendokumentasikan kegagalan sandbox-escape; agenmu harus lolos red team yang sama.

## Konsepnya

Agen ini adalah best-first tree search. Node adalah spesifikasi eksperimen: (hipotesis, config, kode, hasil yang diharapkan). Sebuah langkah expand mengusulkan anak dengan edit kecil (tukar optimizer, geser batch size, ablate sebuah komponen). Setiap anak berjalan di sandbox baru dengan batas resource keras. Hasil diberi umpan balik ke fungsi skoring yang memeringkat node berdasarkan (novelty × quality × sisa anggaran). Pohon tumbuh sampai anggaran habis, lalu cabang terbaik ditulis.

Penulisnya multimodal. Ia menghasilkan draf LaTeX, mengompilasinya, me-render figure, lalu memberikan PDF hasil render kembali ke mode vision Claude Opus 4.7 untuk kritik atas layout, keterbacaan figure, dan kesejajaran klaim-bukti. Ensemble reviewer berisi lima hakim LLM mengeluarkan skor bergaya NeurIPS (novelty, rigor, clarity, reproducibility, impact); bila rata-ratanya jatuh di bawah ambang, makalah kembali ke penulis beserta kritiknya.

Keselamatannya menopang beban. Setiap eksperimen berjalan di sandbox E2B atau Daytona tanpa network egress, wall-clock terbatas, dan limit resource yang dipatok. Langkah code-generation agen melewati sebuah lapisan policy yang memblokir syscall yang kabur dari sandbox. Laporan red team mereproduksi permukaan serang yang didokumentasikan Sakana (fork bomb, pelarian filesystem, panggilan jaringan yang ditulis LLM).

## Arsitektur

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## Stack

- Orkestrasi: LangGraph dengan checkpointing dan gerbang persetujuan manusia
- Tree search: best-first buatan sendiri di atas node eksperimen (gaya AB-MCTS dari Sakana v2)
- Sandbox: E2B per eksperimen, fallback Docker-in-Docker; batas resource via cgroups
- Literatur: Semantic Scholar Graph API + OpenAlex + cache FAISS lokal atas abstrak
- Penulis: template LaTeX + Claude Opus 4.7 (mode vision) untuk kritik figure dan layout
- Reviewer: ensemble 5 hakim (Opus 4.7, GPT-5.4, Gemini 3 Pro, DeepSeek R1, Qwen3-Max) dengan agregasi berbobot
- Framework eksperimen: PyTorch 2.5 untuk eksperimen fisik, W&B untuk logging
- Observability: Langfuse untuk trace agen, anggaran keras $30 per makalah

```figure
ce-experiment-tree
```

## Bangun Sendiri

1. **Scoping ide awal dan domain.** Ambil sebuah ide awal (mis. "investigate sparsity patterns in attention maps of sub-1B transformers"). Definisikan ruang pencarian: model, dataset, anggaran komputasi.

2. **Langkah literatur.** Query Semantic Scholar + OpenAlex untuk 50 makalah relevan yang paling banyak dikutip; cache abstrak secara lokal; hasilkan digest domain 1-halaman.

3. **Kerangka pohon.** Inisialisasi root dengan hipotesis awal. Implementasikan `expand(node) -> children` dengan proposal edit-kecil (satu perubahan config per anak). Implementasikan `score(node)` sebagai suku kata berbobot novelty × quality × budget.

4. **Pembungkusan sandbox.** Setiap eksperimen menjalankan `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` (atau kebijakan ekuivalen E2B). Seed ditulis ke sandbox; keluaran di-mount read-only kembali keluar.

5. **Loop plan-execute-verify.** `plan` mengusulkan anak. `execute` menjalankan sandbox, menangkap log dan metrik. `verify` menjalankan pemeriksaan unit atas metrik (apakah loss turun? apakah ablation mengisolasi efeknya?). Node yang gagal mendapat alasan kegagalan yang disimpan di pohon.

6. **Penulis.** Setelah anggaran habis, pilih cabang terbaik. Render figure dengan matplotlib. Hasilkan draf LaTeX via Claude Opus 4.7 dengan trace cabang dalam context. Kompilasi. Berikan PDF hasil kompilasi kembali ke Opus 4.7 vision untuk kritik. Ulangi.

7. **Ensemble reviewer.** Lima hakim menilai draf pada (novelty, rigor, clarity, reproducibility, impact) dengan rubrik bergaya NeurIPS. Bila mean < 4.0/5, kembalikan ke penulis beserta kritiknya. Berhenti keras setelah 3 kali penulisan ulang.

8. **Red team.** Bangun atau integrasikan serangkaian tugas adversarial yang menyasar sandbox: fork bomb, percobaan exfiltrasi jaringan, pelarian filesystem, karakter metachar shell yang ditulis LLM. Konfirmasi semuanya diblokir. Tulis temuannya.

9. **Reproduksibilitas.** Setiap makalah dikirimkan beserta JSON trace tree-search-nya, seed, link run W&B, config sandbox, dan README yang mereproduksinya dari ujung ke ujung.

## Gunakan

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## Rilis

`outputs/skill-ai-scientist.md` adalah deliverable-nya. Diberi sebuah ide awal + sebuah domain + anggaran $30, skill ini menjalankan pipeline penuh dan mengeluarkan sebuah makalah yang bisa direview plus bundle reproduksibilitas.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Paper quality | Blind rubric review against published workshop papers |
| 20 | Experimental rigor | Baselines, seeds, ablations; every claim backed by a cell in the results table |
| 20 | Cost and compute discipline | $30/paper ceiling enforced, Langfuse-traced |
| 20 | Safety | Sandbox red team passes; network policy and kill-switch verified |
| 15 | Reproducibility | One-command rerun with identical seeds reproduces the paper |
| **100** | | |

## Latihan

1. Jalankan pipeline terhadap tiga ide awal berbeda dalam domain yang sama. Bandingkan bagian pohon pencarian mana yang tumpang tindih. Identifikasi komputasi terbuang yang terduplikasi.

2. Tambahkan gerbang human-in-the-loop sebelum eksekusi eksperimen untuk node yang diperkirakan di atas $5. Ukur seberapa besar total biaya turun.

3. Tukar ensemble reviewer dengan satu hakim tunggal. Ukur tingkat false-accept pada set held-out berisi makalah yang diketahui buruk.

4. Masukkan tes red team exfiltrasi jaringan: agen menulis kode yang mencoba `curl` ke alamat eksternal. Konfirmasi kebijakan `--network=none` memblokirnya. Catat percobaannya.

5. Bandingkan tree-search-mu dengan baseline acak datar (anggaran sama, tanpa strategi ekspansi). Laporkan gain novelty × quality.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Tree search | "AB-MCTS-style expansion" | Best-first exploration over experiment nodes with a novelty×quality×budget score |
| Sandbox | "Experiment isolation" | Container with no network, bounded CPU/memory, pinned seeds, read-only inputs |
| Vision critique | "Render-then-read" | Compile the paper to PDF, feed the PDF back to a VLM for layout and claim-evidence critique |
| Reviewer ensemble | "Automated peer review" | Multiple LLM judges scoring the paper with a NeurIPS rubric; weighted aggregate gates the pipeline |
| Novelty score | "Is this new?" | Heuristic that penalizes proximity to the 50-paper literature cache |
| Cost ceiling | "$ budget" | Hard cap on total spend per paper; Langfuse counters + pre-run estimates |
| Red team | "Sandbox-escape audit" | Adversarial tasks that would escape the sandbox if the policy is wrong |

## Bacaan Lanjutan

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — agen riset produksi referensi
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — metodologi aslinya
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — ekstensi evolusioner
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — framework lab riset multi-peran
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — lapisan orkestrasi referensi
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — pencarian literatur
- [E2B sandboxes](https://e2b.dev) — referensi isolasi eksperimen
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/ReviewerGuidelines) — rubrik yang dikodekan oleh ensemble reviewer
