# Arsitektur Hierarkis dan Mode Kegagalannya

> Hierarkis adalah supervisor bersarang. Agent manager di atas sub-manager di atas worker. `Process.hierarchical` CrewAI adalah versi teksbook: sebuah `manager_llm` mendelegasikan tugas secara dinamis dan memvalidasi keluaran. Padanan LangGraph-nya adalah `create_supervisor(create_supervisor(...))`. Ini adalah pola alami ketika tugasnya benar-benar bagan organisasi. Ini juga pola yang paling mungkin runtuh menjadi looping manajerial — agent manager menugaskan pekerjaan dengan buruk, salah menafsirkan sub-keluaran, atau gagal mencapai konsensus. Sequential sering mengalahkannya.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern)
**Time:** ~60 minutes

## Masalahnya

Setelah pola supervisor klik, langkah alami berikutnya adalah "bagaimana jika para worker-nya sendiri adalah supervisor?" Tim memiliki sub-tim; perusahaan memiliki departemen dari departemen. Arsitektur hierarkis mencerminkan hal itu.

Masalahnya: manager LLM bukanlah manajer manusia. Seorang manajer manusia punya prior stabil tentang apa yang diketahui bawahan mereka. Seorang manager LLM menalar ulang organisasi setiap giliran dari apa pun yang ada di context-nya. Sedikit drift di context itu, dan seluruh pohon salah mengalokasikan pekerjaan.

## Konsepnya

### Bentuknya

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

Setiap node internal merencanakan, mendelegasikan, dan mensintesis. Hanya daun yang mengerjakan pekerjaan.

### Di mana pola ini bersinar

- **Pemetaan organisasi yang jelas.** Jika tugas yang sebenarnya bersifat departemental ("legal meninjau dokumen, finance meninjau dokumen, engineering meninjau dokumen, lalu rangkum untuk eksekutif"), hierarkinya eksplisit.
- **Ringkasan lokal.** Setiap sub-manager mensintesis keluaran timnya sebelum manager teratas melihatnya. Manager teratas melihat tiga ringkasan sub-manager, bukan lima belas keluaran worker.

### Di mana pola ini rusak

Tiga mode kegagalan yang terus ditemukan post-mortem 2026:

1. **Kesalahan penugasan tugas.** Manager membaca tujuan, menghallusinasi sebuah dekomposisi, dan mendelegasikan ke sub-manager yang salah. Karena sub-manager dengan taat mengerjakan apa yang ia terima, kesalahan itu hanya muncul di sintesis teratas — satu level jauh dari tempat manusia bisa menangkapnya.
2. **Salah tafsir keluaran.** Sub-manager mengembalikan "tidak dapat memverifikasi klaim X." Manager teratas merangkum sebagai "klaim X tidak terkonfirmasi." Makna menyimpang di setiap level.
3. **Loop konsensus.** Dua sub-manager tidak setuju; manager teratas meminta mereka merekonsiliasi; mereka mendelegasikan ulang ke bawah; worker menjalankan ulang; sub-manager mengembalikan jawaban yang sedikit berbeda; loop. `Process.hierarchical` CrewAI menjaga dari ini dengan batas langkah, tetapi batas itu sendiri sekarang adalah hyperparameter.

### Pertanyaan penentu

Sequential (pipeline linear) vs hierarkis: apakah tugasmu benar-benar memiliki sub-tim independen, atau hanya satu alur linear yang menyamar sebagai pohon? Jika yang terakhir, gunakan sequential. Jika yang pertama, gunakan hierarkis tetapi anggarkan aturan rekonsiliasi eksplisit.

### Implementasi framework-peran

`Process.hierarchical` CrewAI menyambungkan sebuah LLM manager di atas kru spesialis. Manager tersebut:

- menerima tugas tingkat atas,
- menugaskan subtugas ke kru,
- mengevaluasi keluaran kru,
- memutuskan untuk menerima, mendelegasikan ulang, atau mengiterasi.

Dokumentasi: https://docs.crewai.com/en/introduction (cari "Hierarchical Process" di bawah Core Concepts).

### Implementasi framework-graph

LangGraph menggunakan panggilan `create_supervisor` bersarang. Supervisor inner punya graph-nya sendiri; supervisor outer memperlakukan graph inner sebagai node opak. Ini lebih bersih daripada CrewAI untuk debugging (kamu bisa melangkah melalui setiap graph secara terpisah) tetapi lebih sulit untuk mengekspresikan reshaping dinamis pohon.

Referensi: https://reference.langchain.com/python/langgraph-supervisor.

```figure
swarm-hierarchy-token
```

## Bangun

`code/main.py` menjalankan hierarki 3 level:

- manager teratas: memecah sebuah tugas menjadi cabang "engineering" dan "legal",
- sub-manager engineering: memecah menjadi worker "frontend" dan "backend",
- sub-manager legal: satu worker.

Demo membandingkan jalur happy path (semua setuju) terhadap sebuah **jalur terganggu** di mana dekomposisi manager teratas salah melabeli "legal" sebagai "finance" dan menyaksikan kesalahan itu bercabang — sub-manager dengan taat mengerjakan pekerjaan finance, sintesis teratas melaporkan temuan finance, pertanyaan legal asli tidak terjawab.

Jalankan:

```
python3 code/main.py
```

Keluaran menunjukkan kedua jalur dengan perbandingan berdampingan yang jelas dari "apa yang diminta" vs "apa yang dikirimkan."

## Gunakan

`outputs/skill-hierarchy-fitness.md` mengevaluasi apakah sebuah tugas tertentu harus menggunakan supervisor hierarkis, sequential, atau datar. Input: deskripsi tugas, struktur organisasi, anggaran rekonsiliasi. Output: rekomendasi pola dengan mode kegagalan spesifik yang harus dijaga.

## Rilis

Jika kamu merilis hierarkis:

- **Cap kedalaman pohon di 2.** Tiga level sudah menyembunyikan sebagian besar kesalahan dari observabilitas.
- **Anggaran rekonsiliasi eksplisit.** Tetapkan maksimum putaran sebelum manager teratas harus commit. Biasanya 2.
- **Provenance di setiap sintesis.** Ringkasan setiap node harus mengutip keluaran daun mana yang menghasilkannya.
- **Peringatan pada drift dekomposisi.** Log dekomposisi manager per langkah; diff terhadap query pengguna. Jika dekomposisi tidak lagi mencakup query, picu peringatan.

## Latihan

1. Jalankan `code/main.py` dan bandingkan happy vs terganggu. Berapa banyak level hand-off manager yang diperlukan sebelum keluaran teratas sepenuhnya menyimpang dari pertanyaan pengguna?
2. Tambahkan level ketiga (teratas → sub → sub-sub → worker). Ukur seberapa sering jalur terganggu memperbaiki dirinya sendiri vs menyimpang sepenuhnya seiring kedalaman bertambah.
3. Implementasikan worker "canary" di setiap sub-manager yang selalu ditanyai pertanyaan pengguna asli tanpa diubah. Gunakan jawaban canary untuk mendeteksi drift dekomposisi. Bagaimana manager bereaksi ketika canary tidak setuju dengan jawaban yang disintesis?
4. Baca dokumen `Process.hierarchical` CrewAI. Identifikasi satu guardrail konkret yang diterapkan CrewAI (batas langkah, batasan manager_llm) dan jelaskan mode kegagalan apa yang menjadi targetnya.
5. Bandingkan supervisor LangGraph bersarang dengan hierarkis CrewAI. Mana yang membuat loop rekonsiliasi lebih murah untuk dideteksi?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Hierarchical | "Org chart pattern" | Supervisors over supervisors; only leaves do work. |
| Manager LLM | "The boss" | The LLM that decomposes, assigns, and validates at an internal node. |
| Decomposition drift | "The boss lost the plot" | Top manager's split no longer covers the original question. |
| Reconciliation loop | "Endless meetings" | Sub-managers disagree; top re-delegates; workers re-run; loop until budget exhausted. |
| Depth-2 ceiling | "Don't go deeper than 2 levels" | Empirical guardrail: 3+ levels collapses observability. |
| Canary question | "Ground truth at every level" | A worker that is always asked the original query unchanged, to detect drift. |
| Provenance chain | "Who said what" | Trace from each synthesis back to the leaf outputs that produced it. |

## Bacaan Lanjutan

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction) — textbook hierarchical with a manager LLM
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — nested supervisor via `create_supervisor`
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system) — why Anthropic deliberately chose flat supervisor over hierarchical
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy; section on coordination failures documents decomposition drift
