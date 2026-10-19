# Lanskap Agen Koding Otonom (2026)

> SWE-bench Verified naik dari 4% menjadi 80,9% dalam kurang dari tiga tahun. Claude Sonnet 4.5 yang sama mencetak 43,2% pada SWE-agent v1 dan 59,8% pada Cline autonomous — scaffolding di sekitar model kini sama pentingnya dengan model itu sendiri. OpenHands (dulu OpenDevin) adalah platform berlisensi MIT paling aktif dan loop CodeAct-nya mengeksekusi aksi Python langsung dalam sandbox alih-alih panggilan tool JSON. Angka utama menyembunyikan isu metodologis: 161 dari 500 tugas SWE-bench Verified hanya membutuhkan perubahan 1–2 baris, dan SWE-bench Pro (tugas 10+ baris) berada pada 23–59% untuk model frontier yang sama.

**Type:** Learn
**Languages:** Python (stdlib, CodeAct vs JSON tool-call comparison)
**Prerequisites:** Phase 14 · 07 (Tool use), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## Masalahnya

"Agen koding mana yang terbaik" adalah pertanyaan yang salah. Pertanyaan yang benar adalah: pada distribusi tugas yang cocok dengan pekerjaanmu, dengan scaffolding yang akan kamu jalankan di produksi, reliabilitas ujung-ke-ujung apa yang kamu dapat?

Antara 2022 dan 2026 bidang ini belajar bahwa scaffolding — lapisan retrieval, perencana, sandbox, loop edit-verify, format umpan balik — bersifat load-bearing. Claude Sonnet 4.5 pada SWE-agent v1 mencetak 43,2% pada SWE-bench Verified; model yang sama di dalam scaffolding autonomous Cline mencetak 59,8%. Selisih 16,6 poin absolut, bobot sama. Model dasar adalah komponen; loop adalah produk.

Soal pendampingnya adalah saturasi benchmark menyembunyikan regresi. SWE-bench Verified nyaris jenuh, dan ekor tugas-mudah (161 dari 500 tugas membutuhkan ≤2 baris) menarik skor teratas ke atas. Kualitas dunia-nyata lebih baik diukur pada distribusi seperti SWE-bench Pro (perubahan 10+ baris), tempat para pemimpin yang sama masih berada pada 23–59%.

## Konsepnya

### SWE-bench, satu paragraf

SWE-bench (Jimenez et al.) mengambil isu GitHub nyata dengan patch ground-truth dan meminta agen menghasilkan patch yang membuat rangkaian pengujian lolos. SWE-bench Verified (OpenAI, 2024) adalah subset 500 tugas kurasi-manusia dengan tugas ambigu dan rusak dibuang. SWE-bench Pro adalah penerus yang lebih sulit — tugas yang membutuhkan perubahan 10+ baris, tempat agen frontier saat ini berada pada 23–59%.

### Apa yang sebenarnya ditunjukkan kurva 2022 → 2026

- **2022**: model riset pada ~4% pada SWE-bench mentah.
- **2024**: GPT-4 + scaffolding gaya Devin pada ~14%; SWE-agent pada ~12%.
- **2025**: Claude 3.5/3.7 Sonnet di dalam Aider dan SWE-agent mendorong ke rentang 40–55%.
- **2026**: Claude Sonnet 4.5 dan kompetitor frontier pada 70–80%+ pada SWE-bench Verified. Leaderboard Epoch AI melacak ini secara live.

Kemiringan datang dari tiga sumber yang berlipat: model dasar yang lebih baik, scaffolding yang lebih baik (CodeAct, refleksi, loop verifier), dan benchmark yang lebih baik (Verified membuang noise).

### CodeAct vs panggilan tool JSON

OpenHands (All-Hands-AI, arXiv:2407.16741, dulu OpenDevin) mengambil taruhan arsitektural spesifik: alih-alih model memancarkan panggilan tool JSON yang di-decode dan dieksekusi host, model memancarkan kode Python dan kernel gaya Jupyter menjalankannya dalam sandbox. Agen bisa meng-loop berkas, merantai tool, dan menangkap exception-nya sendiri di dalam satu aksi.

Trade-off-nya:

- **Panggilan tool JSON**: setiap aksi satu giliran; mudah diaudit; komposisionalitas terbatas; aman secara default karena setiap panggilan lewat validator eksplisit.
- **CodeAct**: satu aksi bisa berupa program utuh; komposisional; membutuhkan sandbox yang di-hardening (OpenHands memakai isolasi Docker); mode kegagalan mencakup apa pun yang diizinkan runtime sandbox.

Kedua arsitektur ada di produksi. CodeAct dominan di platform terbuka (OpenHands, smolagents). Panggilan tool JSON tetap dominan di layanan terkelola (Anthropic Managed Agents, OpenAI Assistants) tempat penyedia mengendalikan eksekutor.

### Scaffold dalam lanskap 2026

| Scaffold | License | Execution model | Notable property |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | CodeAct in Docker | Most active open platform; event-stream replayable |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | First end-to-end SWE-bench scaffold |
| Aider | Apache-2 | edit-via-diff in local repo | Minimal scaffold, strong regression stability |
| Cline | Apache-2 | VS Code agent with tool policy | Highest-scoring open scaffold on Sonnet 4.5 |
| Devin (Cognition) | Proprietary | Managed VM + planner | First "AI software engineer" product category |
| Claude Code | Proprietary | Permission modes + routines | Lesson 10 covers the agent loop in detail |

### Mengapa scaffolding mendominasi

Proses koding adalah trajektori horizon-panjang (Pelajaran 1). Reliabilitas berlipat lintas langkah. Tiga tempat scaffolding membeli poin:

1. **Retrieval**: menemukan file yang tepat untuk dibaca adalah bottleneck senyap. ACI SWE-agent, indeks-file OpenHands, dan repo-map Aider semuanya menyerang ini.
2. **Loop verifier**: menjalankan pengujian, membaca stack trace, dan mencoba ulang adalah delta 10+ poin pada SWE-bench.
3. **Penahanan kegagalan**: sandbox yang rollback saat error mencegah kerusakan berlipat. Model yang sama dengan dan tanpa loop verifier tampak seperti dua produk berbeda.

### Saturasi benchmark dan distribusi yang nyata

Para penulis OpenHands dan Epoch AI sama-sama menandai bahwa SWE-bench Verified punya ekor mudah: 161 dari 500 tugas hanya butuh perubahan 1–2 baris. Skor tinggi sebagian didorong ekor ini. SWE-bench Pro membatasi pada perubahan 10+ baris dan mengembalikan skor pada rentang 23–59% bahkan untuk sistem frontier. Distribusi produksimu hampir pasti lebih dekat ke Pro dibanding ke Verified.

Implikasi memilih agen: jalankan subset ala-Pro dari backlog bug-mu sendiri. Skor yang penting adalah skor pada tugas yang representatif terhadap apa yang kamu kirim.

```figure
a5-scaffold-delta
```

## Gunakan

`code/main.py` membandingkan dua scaffold agen mainan pada distribusi tugas-mini tetap:

1. Scaffold **panggilan tool JSON** yang mengambil satu aksi per giliran.
2. Scaffold **CodeAct** yang bisa memancarkan cuplikan Python kecil per aksi.

Keduanya memakai "model" rintisan (aturan deterministik) sehingga perbandingan mengisolasi scaffold dari kualitas model. Keluarannya menunjukkan scaffold CodeAct menyelesaikan lebih banyak tugas dalam giliran lebih sedikit dengan biaya blast radius per aksi yang lebih besar.

## Rilis

`outputs/skill-scaffold-audit.md` membantumu mengaudit scaffold agen-koding yang diusulkan sebelum adopsi: kualitas retrieval, keberadaan verifier, isolasi sandbox, dan kesesuaian benchmark-ke-distribusi.

## Latihan

1. Jalankan `code/main.py`. Berapa giliran yang dipakai setiap scaffold pada set tugas yang sama? Berapa blast radius per aksi masing-masing?

2. Baca makalah OpenHands (arXiv:2407.16741). Makalah berargumen CodeAct mengalahkan panggilan tool JSON pada tugas kompleks. Identifikasi satu mode kegagalan yang diakui makalah dan tulis satu kalimat tentang kapan mode itu akan mendominasi di produksi.

3. Pilih satu tugas dari backlog bug-mu yang membutuhkan perubahan 10+ baris lintas dua file. Estimasi probabilitas keberhasilan ujung-ke-ujung untuk model frontier pada (a) panggilan tool JSON dan (b) CodeAct. Justifikasi kesenjangannya.

4. SWE-bench Verified punya 161 tugas satu-file 1–2 baris. Susun skor yang mengecualikannya. Bagaimana leaderboard berubah?

5. Baca "Introducing SWE-bench Verified" (OpenAI). Jelaskan metodologi spesifik yang dipakai untuk membuang tugas ambigu, dan sebutkan satu kategori yang akan terlewat kurasi.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| SWE-bench | "Coding benchmark" | Real GitHub issues with ground-truth patches and test suites |
| SWE-bench Verified | "Cleaned subset" | 500 human-curated tasks, easier-tail present |
| SWE-bench Pro | "Harder subset" | 10+ line changes; frontier sits at 23–59% |
| CodeAct | "Code-as-action" | Agent emits Python; Jupyter-style kernel executes in sandbox |
| JSON tool call | "Function calling" | Each action is a structured JSON payload validated before execution |
| Scaffold | "Agent framework" | Retrieval + planner + executor + verifier loop around the base model |
| ACI (Agent-Computer Interface) | "SWE-agent's format" | Command set designed for LLM ergonomics, not human shells |
| Verifier loop | "Test-and-retry" | Run tests, read output, revise patch; biggest non-model reliability gain |

## Further Reading

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — the original benchmark and methodology.
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — how the curated subset was built.
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct architecture and event-stream design.
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — live-tracked scores.
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon coding-agent reliability framing.
