# Spesialisasi Peran — Planner, Critic, Executor, Verifier

> Dekomposisi multi-agent paling umum di 2026: satu agent merencanakan, satu mengeksekusi, satu mengkritik atau memverifikasi. MetaGPT (arXiv:2308.00352) memformalkannya sebagai SOP yang dikodekan ke dalam prompt peran — Product Manager, Architect, Project Manager, Engineer, QA Engineer — mengikuti `Code = SOP(Team)`. ChatDev (arXiv:2307.07924) menautkan designer, programmer, reviewer, tester melalui "chat chain" dengan "communicative dehallucination" (agent secara eksplisit meminta detail yang hilang). Verifier adalah penanggung beban: Cemri et al. (MAST, arXiv:2503.13657) menunjukkan setiap kegagalan multi-agent dapat ditelusuri ke verifikasi yang hilang atau rusak. PwC melaporkan peningkatan akurasi 7× (10% → 70%) dari loop validasi terstruktur di CrewAI.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**Time:** ~60 minutes

## Masalahnya

Sistem multi-agent generik menghasilkan keluaran generik. Tiga coder dalam group chat menulis tiga varian dari kode mediocre yang sama. Kamu bisa menambah lebih banyak agent, menambah lebih banyak putaran, dan tetap tidak melewati ambang kualitas.

Perbaikannya bukan lebih banyak agent — melainkan agent yang *berbeda*. Tugaskan peran yang berbeda. Beri critic tool yang tidak dimiliki planner. Beri verifier test suite yang objektif. Sekarang sistem memiliki ketidaksepakatan internal dengan koreksi yang berdasar, bukan sekadar menebak paralel.

## Konsepnya

### Empat peran kanonis

**Planner.** Membaca tujuan, menghasilkan daftar langkah atau sebuah spesifikasi. Tool: retrieval pengetahuan, dokumentasi. Keluaran: rencana terstruktur.

**Executor.** Membaca satu langkah rencana pada satu waktu, menghasilkan artefak. Tool: tool pekerjaan yang sebenarnya (compiler kode, shell, API client). Keluaran: artefaknya.

**Critic.** Membaca keluaran executor terhadap intent planner. Tool: akses read-only ke artefak, analisis statis. Keluaran: terima/tolak dengan alasan.

**Verifier.** Membaca artefak dan menjalankan pemeriksaan deterministik. Tool: test runner, type checker, schema validator. Keluaran: lulus/gagal dengan bukti.

Critic bersifat subjektif, berpendapat, sering berbasis LLM. Verifier bersifat objektif, deterministik, sering berbasis kode. Keduanya bukan peran yang sama.

### Pola SOP MetaGPT

MetaGPT (arXiv:2308.00352) mengodekan SOP rekayasa perangkat lunak sebagai prompt peran:

- **Product Manager** menulis PRD-nya.
- **Architect** menghasilkan desain sistem.
- **Project Manager** memecah tugas.
- **Engineer** mengimplementasikan.
- **QA Engineer** menjalankan test.

Setiap peran memiliki schema input/output yang ketat. Prompt peran menyatakan apa yang peran itu *adalah* dan apa yang *harus dihasilkannya*. Formulasi `Code = SOP(Team)` — SOP deterministik mengubah tim LLM menjadi pipeline yang dapat diprediksi.

### Communicative dehallucination ChatDev

ChatDev menambahkan sebuah langkah kunci: ketika sebuah executor membutuhkan detail spesifik yang tidak ada dalam rencananya, ia secara eksplisit bertanya kepada designer sebelum melanjutkan. Ini mencegah kegagalan LLM klasik yang mengarang detail secara meyakinkan.

Implementasi: prompt peran menyertakan "ketika kamu membutuhkan informasi spesifik yang tidak diberikan kepadamu, tanyalah peran yang relevan dengan nama sebelum menghasilkan keluaran."

### Mengapa verifier paling penting

Cemri et al. (MAST) menelusuri 1642 kegagalan eksekusi multi-agent. 21,3% adalah celah verifikasi — sistem merilis jawaban yang tidak diperiksa siapa pun. Sisa 79% sering kembali ke "ada pemeriksaan yang gagal diam-diam atau tidak pernah dijalankan." Verifikasi adalah peran penanggung beban.

PwC melaporkan (deployment CrewAI, 2025) bahwa menambahkan loop validasi terstruktur memindahkan akurasi dari 10% ke 70%. Peningkatan 7× dari satu peran.

### Critic vs verifier

- Seorang critic adalah LLM yang meninjau sebuah artefak untuk kualitas. Subjektif. Bisa ditipu oleh prosa yang meyakinkan.
- Seorang verifier adalah program deterministik yang berjalan di atas artefak. Objektif. Memberikan lulus/gagal dengan bukti.

Gunakan keduanya. Critic menangkap masalah selera yang tidak bisa diartikulasikan verifier. Verifier menangkap bug yang tidak bisa dilihat critic karena hanya muncul saat runtime.

### Anti-pola

Setiap peran di sistem-mu adalah LLM dan setiap keluaran peran adalah "looks good to me." Mode kegagalan MAST klasik. Tambahkan setidaknya satu verifier yang lulus/gagal-nya diputuskan oleh kode, bukan oleh LLM.

### Pemetaan framework

- **CrewAI** — `Agent(role, goal, backstory)` adalah permukaan spesialisasi teksbook.
- **LangGraph** — node bisa memiliki prompt terspesialisasi; edge menegakkan pipeline-nya.
- **AutoGen** — ConversableAgent spesifik-peran dengan nama satu-kata dalam GroupChat.
- **OpenAI Agents SDK** — tool handoff antara Agent yang terspesialisasi-peran.

```figure
swarm-roles
```

## Bangun

`code/main.py` mengimplementasikan pipeline 4-peran yang membangun sebuah fungsi Python sederhana:

- **Planner** menghasilkan sebuah spesifikasi.
- **Executor** menghasilkan string kode.
- **Critic** (disimulasikan LLM) menandai masalah yang jelas.
- **Verifier** menjalankan kode yang dihasilkan dalam sandbox (`exec`) terhadap sebuah test case.

Demo berjalan dua kali: sekali di mana executor menghasilkan kode yang benar (critic + verifier keduanya lulus), sekali di mana executor menghasilkan kode di luar spesifikasi (critic melewatkan bug karena terlihat meyakinkan, verifier menangkapnya karena test gagal).

Jalankan:

```
python3 code/main.py
```

## Gunakan

`outputs/skill-role-designer.md` mengambil sebuah tugas dan menghasilkan daftar peran (3-5 peran), schema input/output per peran, dan pemeriksaan verifier. Gunakan ini sebelum menyambungkan agent ke framework.

## Rilis

Checklist:

- **Setidaknya satu verifier deterministik.** Jangan pernah semua-LLM.
- **Schema I/O eksplisit per peran.** Planner mengembalikan spesifikasi, bukan prosa; executor membaca schema itu.
- **Communicative dehallucination.** Executor harus bertanya kepada planner ketika info hilang; jangan pernah mengarangnya.
- **Urutan critic/verifier.** Jalankan critic terlebih dahulu (murah, menangkap masalah desain), verifier kedua (lambat, menangkap bug).
- **Anggaran loop.** Maksimal 2 putaran revisi critic-executor sebelum eskalasi ke manusia.

## Latihan

1. Jalankan `code/main.py` dan amati bagaimana verifier menangkap bug yang critic lewatkan. Tambahkan pemeriksaan analisis statis (hitung kemunculan `return`) sebagai verifier tambahan. Apa yang ditangkapnya yang dilewatkan test runtime?
2. Tambahkan peran ke-5: "analis persyaratan" yang menerjemahkan keinginan pengguna menjadi spesifikasi siap-planner. Permintaan communicative dehallucination apa yang harus mengalir naik kepadanya?
3. Baca MetaGPT Bagian 3 ("Agents"). Daftarkan schema input/output dari masing-masing 5 peran MetaGPT.
4. Baca diagram chat-chain ChatDev (arXiv:2307.07924 Gambar 3). Identifikasi di mana communicative dehallucination mematahkan loop yang sebaliknya akan tak terhingga.
5. Peningkatan akurasi 7× PwC datang dari loop verifikasi. Hipotesiskan tiga tugas di mana menambahkan verifier tidak akan membantu — di mana pemeriksaan deterministik atas kebenaran tidak mungkin atau terlalu mahal.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Role specialization | "Different agents, different jobs" | Distinct system prompts tuned for planner/executor/critic/verifier roles. |
| SOP pattern | "Encoded standard operating procedure" | MetaGPT's framing: strict I/O schemas per role turn a team into a pipeline. |
| Communicative dehallucination | "Ask before inventing" | ChatDev pattern: executor asks planner when a detail is missing rather than making one up. |
| Critic | "LLM reviewer" | Subjective, opinionated reviewer. Catches taste issues. Can be fooled by plausible prose. |
| Verifier | "Deterministic check" | Code-based pass/fail. Test runner, type checker, schema validator. Cannot be fooled. |
| Verification gap | "No one checked" | 21.3% of MAST failures. Answer shipped without a check that would have caught the bug. |
| Revision loop | "Critic sends it back" | Critic rejection triggers executor re-run with feedback. Needs a budget. |
| All-LLM anti-pattern | "Looks good to me" | Every role is an LLM, no deterministic check. Classic MAST failure. |

## Bacaan Lanjutan

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — the SOP-as-role-prompt reference paper
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — chat chain + communicative dehallucination
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy; verification gaps are 21.3% of failures
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction) — production role specification surface
