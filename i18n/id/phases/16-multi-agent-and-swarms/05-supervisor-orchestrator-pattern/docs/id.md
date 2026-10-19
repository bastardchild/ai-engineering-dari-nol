# Pola Supervisor / Orchestrator-Worker

> Satu agent lead merencanakan dan mendelegasikan; worker terspesialisasi mengeksekusi di context paralel dan melapor kembali. Inilah pola di balik sistem Research Anthropic (Claude Opus 4 sebagai lead, Sonnet 4 sebagai subagent), yang terukur +90,2% di atas Opus 4 single-agent pada eval riset internal. Pos teknik Anthropic melaporkan bahwa 80% varians pada BrowseComp dijelaskan oleh penggunaan token saja — multi-agent menang terutama karena setiap subagent mendapat context window yang segar. Pelajaran ini membangun pola supervisor dari primitif-primitif dan membahas pelajaran teknik 2026 dari deployment produksi.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## Masalahnya

Riset adalah tugas prototipikal yang gagal ditangani sistem single-agent. Kamu bertanya "apa yang berubah di sistem multi-agent antara 2023 dan 2026?" Sebuah agent membaca lima paper secara sekuensial, mengisi setengah context-nya dengan teks mereka, lalu harus menalar tentang semuanya sekaligus. Ia lupa paper pertama pada saat mencapai yang kelima. Ia tidak bisa memparalelkan.

Pola supervisor memperbaiki ini: satu agent lead merencanakan pencarian, mendelegasikan setiap sub-pertanyaan ke seorang worker, dan mensintesis. Setiap worker mendapat window 200k-token-nya sendiri untuk satu pertanyaan yang sempit. Lead tidak pernah melihat paper mentah — hanya ringkasan worker.

Sistem Research produksi Anthropic melaporkan +90,2% pada eval riset internal vs satu Opus 4. Pos yang sama mencatat bahwa 80% varians BrowseComp dijelaskan oleh *penggunaan token saja*. Context segar per subagent adalah mekanisme utamanya.

## Konsepnya

### Polanya

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

Lead tidak pernah membaca materi mentah. Para worker tidak pernah melihat pekerjaan satu sama lain sampai lead mensintesis. Setiap panah adalah sebuah handoff dengan artefak yang sempit.

### Mengapa pola ini menang

Tiga mekanisme:

1. **Context segar per subagent.** Seorang worker yang mengeksplorasi "warisan FIPA-ACL" tidak membawa 40k token yang dihabiskan lead untuk merencanakan. Ia mendapat window 200k untuk satu pertanyaan.
2. **Spesialisasi via prompt.** Prompt lead adalah "dekomposisi dan sintesis," bukan "riset." Setiap prompt worker sempit: "temukan apa yang berubah di X." Prompt yang fokus menghasilkan keluaran yang fokus.
3. **Paralelisme.** Worker berjalan bersamaan. Waktu wall-clock kira-kira `max(worker_times) + plan + synthesis`, bukan `sum(worker_times)`.

### Pelajaran teknik (Anthropic 2025)

Pos Anthropic mencantumkan beberapa pelajaran produksi yang masih relevan untuk 2026:

- **Skalakan upaya ke kompleksitas query.** Query sederhana: satu agent, 3-10 tool call. Query kompleks: 10+ agent. Lead yang harus mengevaluasi ini, bukan pemanggil.
- **Luas lalu sempit.** Dekomposisikan ke sub-pertanyaan luas terlebih dahulu, lalu lahirkan lebih banyak worker per sub-pertanyaan jika jawabannya menjamin kedalaman.
- **Deploy rainbow.** Agent bersifat jangka panjang dan stateful. Blue-green tradisional tidak bekerja. Anthropic menggunakan rainbow: rollout bertahap versi baru sambil versi lama dikuras.
- **Penggunaan token mendominasi.** Multi-agent adalah ~15× token dari single-agent. Jalankan hanya ketika nilai tugasnya menjustifikasi biayanya.

### Giliran graph-native

LangGraph awalnya merilis library `langgraph-supervisor` dengan helper tingkat tinggi `create_supervisor`. Pada 2025 LangChain memindahkan rekomendasinya ke implementasi pola supervisor via tool-calling secara langsung, karena tool call memberikan kontrol lebih besar atas *apa yang dilihat supervisor* (context engineering). Library-nya masih bekerja; dokumentasinya sekarang merekomendasikan bentuk tool-calling.

### Mode kegagalan

- **Lead menghallusinasi rencana.** Jika lead menghasilkan sub-pertanyaan yang tidak mendekomposisi pertanyaan yang sebenarnya, worker melakukan riset presisi pada target yang salah.
- **Worker over-explore.** Tanpa batas lingkup yang eksplisit, worker menyimpang melampaui sub-pertanyaan yang ditugaskan dan mencemari tahap sintesis.
- **Konflik sintesis.** Dua worker mengembalikan fakta yang saling bertentangan. Lead harus meminta ulang (menambahkan satu putaran) atau mencatat ketidaksepakatan secara eksplisit. Pemilihan diam-diam salah satu pihak adalah kegagalan terburuk: pengguna tidak pernah tahu ada ketidaksepakatan.

### Kapan supervisor salah

- **Tugas sekuensial.** Jika langkah 2 secara harfiah membutuhkan keluaran langkah 1, paralelisme tidak membeli apa pun. Gunakan pipeline (CrewAI Sequential, graph linear LangGraph).
- **Query sederhana.** Single-agent menanganinya lebih cepat dan lebih murah. Gunakan pemeriksaan "skalakan upaya" lead sebelum melahirkan worker.
- **Determinisme ketat.** Supervisor menggunakan delegasi yang dipilih LLM. Graph statis lebih baik ketika audit/replay lebih penting daripada adaptabilitas.

```figure
supervisor-hierarchy
```

## Bangun

`code/main.py` mengimplementasikan seorang supervisor dari tiga worker paralel menggunakan `threading`. Lead mendekomposisi sebuah query menjadi sub-pertanyaan, worker berjalan bersamaan pada setiap sub-pertanyaan, dan lead mensintesis. Tanpa LLM nyata — para worker di-skrip untuk mensimulasikan fetch-dan-ringkas.

Struktur kunci:

- `Lead.plan(query)` memecah sebuah query menjadi 3 sub-pertanyaan.
- `Worker.run(sub_q)` mengembalikan ringkasan palsu (bisa berupa agent ber-tool apa pun dalam produksi).
- `Lead.run(query)` memulai worker di thread, join, dan mensintesis.

Jalankan:

```
python3 code/main.py
```

Keluaran menunjukkan rencana, jejak worker paralel dengan timestamp mulai/akhir, dan sintesis akhir. Kamu bisa melihat kemenangan wall-clock: tiga worker 0,3 detik berjalan dalam ~0,35 detik, bukan 0,9.

## Gunakan

`outputs/skill-supervisor-designer.md` mengambil query pengguna dan menghasilkan desain pola supervisor: system prompt lead, peran worker, aturan dekomposisi sub-pertanyaan, dan template sintesis. Gunakan ini sebelum membangun sistem agent bergaya riset baru.

## Rilis

Checklist sebelum mendeploys pola supervisor:

- **Pemasangan model.** Lead pada model tingkat penalaran (kelas Opus, kelas `o3`). Worker pada model yang lebih cepat dan lebih murah (Sonnet, `o4-mini`).
- **Timeout worker.** Worker mana pun yang melebihi 2× median runtime dibunuh; lead melahirkan ulang dengan lingkup lebih sempit atau melanjutkan tanpanya.
- **Cap token per worker.** Batas keras (katakan 10× input sintesis yang diharapkan) mencegah worker liar meledakkan anggaran.
- **Observabilitas.** Lacak rencana lead, tool call setiap worker, dan sintesis. Ini adalah dasar untuk debugging pasca-apapun.
- **Rollout rainbow.** Agent stateful jangka panjang memerlukan transisi versi bertahap, bukan hot swap.

## Latihan

1. Jalankan `code/main.py`, lalu modifikasi lead untuk melahirkan 5 worker alih-alih 3. Amati efek wall-clock. Pada jumlah worker berapa overhead spawn melebihi penghematan paralel dalam demo ini?
2. Implementasikan timeout worker: bunuh worker mana pun yang berjalan lebih lama dari 0,5 detik dan minta lead mensintesis sisa hasilnya. Observabilitas apa yang kamu butuhkan untuk mengetahui sebuah worker dipotong?
3. Tambahkan langkah deteksi konflik ke sintesis lead: jika dua worker mengembalikan jawaban yang bertentangan, lead mencatat ketidaksepakatan alih-alih memilih satu. Bagaimana kamu mendeteksi kontradiksi tanpa memanggil LLM?
4. Baca pos teknik sistem Research Anthropic. Daftarkan tiga praktik yang perlu diadopsi demo main-main ini untuk berjalan di produksi.
5. Bandingkan `create_supervisor` LangGraph (legacy) vs rekomendasi tool-calling yang baru. Mana yang memberimu kontrol lebih baik atas apa yang dilihat supervisor? Mengapa Anthropic secara eksplisit meneruskan hanya sub-jawaban dan bukan context worker mentah ke sintesis?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Supervisor | "Lead agent" | An orchestrator agent that plans, delegates, and synthesizes. Does not do the work itself. |
| Worker | "Subagent" | A focused agent invoked by the supervisor with narrow scope and its own context window. |
| Orchestrator-worker | "Supervisor pattern" | Same thing, different name. The 2026 literature uses both. |
| Fresh context | "Clean window" | A worker's context starts from its system prompt and assigned question, not the lead's history. |
| Rainbow deployment | "Gradual rollout" | Long-running stateful agents need versioned drain-and-replace, not blue-green. |
| Token dominance | "Context is the variable" | 80% of research-eval variance comes from total tokens used, not model choice, per Anthropic. |
| Scale effort | "Match agent count to complexity" | Lead estimates query difficulty, spawns 1 vs 10+ workers accordingly. |
| Synthesis conflict | "Workers disagree" | Two workers return contradictory facts; the lead must surface disagreement, not silently pick one. |

## Bacaan Lanjutan

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — the production reference for supervisor pattern
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — tool-calling supervisor is now the recommended form
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — the legacy helper, still used in 2026 production
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — handoff-based supervisor variant
