# Pengorbanan Kerangka Agen - Grafik, Peran, dan Orkestrasi Aktor

> Setiap kerangka kerja menjual demo yang sama (agen peneliti membuat laporan) dan menyembunyikan bug yang sama (skema status berkelahi dengan lapisan orkestrasi). Pilih kerangka kerja yang abstraksinya sesuai dengan bentuk masalah kamu; yang lainnya adalah lem yang kamu tulis dua kali.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**Time:** ~45 minutes

## Masalahnya

kamu memiliki tugas yang membutuhkan lebih dari satuLLMpanggilan. Mungkin alur kerja penelitian (rencanakan, cari, rangkum, kutip). Mungkin itu adalah pipa peninjauan kode (parse diff, kritik, patch, validasi). Mungkin asisten multi-turn yang memesan penerbangan, menulis email, dan mengarsipkan laporan pengeluaran. kamu memilih kerangka kerja.

Tiga hari kemudian, kamu menemukan kebocoran abstraksi kerangka kerja. CrewAI memberi kamu peran tetapi melawan kamu ketika "peneliti" perlu menyerahkan rencana terstruktur kepada "penulis". AutoGen memberi kamu obrolan antar agen tetapi tidak memiliki status kelas satu sehingga pos pemeriksaan kamu hanyalah kumpulan log percakapan. LangGraph memberi kamu grafik keadaan tetapi memaksa kamu memberi nama setiap transisi sebelum kamu mengetahui apa yang akan dilakukan agen. Agno memberi kamu abstraksi agen tunggal yang berteriak ketika kamu mencoba menyebar ke tiga pekerja secara bersamaan.

Cara mengatasinya bukanlah "memilih kerangka kerja terbaik". Hal ini untuk mencocokkan abstraksi inti kerangka kerja dengan bentuk masalah kamu. Pelajaran ini menggambarkan peta itu.

## Konsepnya

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

Empat kerangka kerja mendominasi lanskap tahun 2026. Abstraksi intinya tidak sama.

| Framework | Core abstraction | Best fit | Worst fit |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — typed state, nodes, conditional edges, checkpointer. | Workflows with explicit state and human-in-the-loop interrupts; production agents needing time-travel debugging. | Loose, role-driven brainstorming where the topology is unknown. |
| **CrewAI** | `Crew` — roles (goal, backstory), tasks, process (sequential or hierarchical). | Role-playing or persona-driven workflows with a short linear/hierarchical plan. | Anything stateful beyond the crew's turn history; complex branching. |
| **AutoGen** | `ConversableAgent` pair — two or more agents that speak in turns until an exit condition. | Multi-agent *dialogue* (teacher-student, proposer-critic, actor-reviewer) where the thinking emerges from the chat. | Deterministic workflows with a known DAG; anything needing durable state across restarts. |
| **Agno** | `Agent` — a single LLM + tools + memory, composable into teams. | Fast-to-build single agents and lightweight teams; strong multi-modality and built-in storage drivers. | Deep, explicitly-branched graphs with custom reducers. |

### Apa sebenarnya arti "abstraksi".

Abstraksi inti kerangka kerja adalah hal yang kamu gambar di papan tulis saat kamu menampilkan arsitektur.

- **LangGraph** → kamu menggambar grafik. Node adalah langkah, tepian adalah transisi, dan objek keadaan di setiap titik diketik. Model mental adalah mesin negara.
- **CrewAI** → kamu menggambar bagan organisasi. Setiap peran memiliki deskripsi pekerjaan dan manajer mengarahkan tugas. Model mental adalah sebuah tim kecil yang terdiri dari para spesialis.
- **AutoGen** → kamu menggambar Slack DM. Dua agen saling mengirim pesan; sepertiga bergabung jika kamu membutuhkan moderator. Model mentalnya adalah obrolan.
- **Agno** → kamu menggambar satu kotak dengan peralatan yang tergantung di sana. Letakkan kotak di samping satu sama lain untuk sebuah tim. Model mentalnya adalah "agen yang dilengkapi baterai".

### Pertanyaan negara

Negara bagian adalah tempat sebagian besar pilihan kerangka kerja gagal dalam produksi.

- **LangGraph.** Status yang diketik (`TypedDict`atau model Pydantic), reduksi per bidang, checkpointer kelas satu (SQLite/Postgres/Redis). Melanjutkan, menyela, dan melakukan perjalanan waktu gratis. *(Lihat Fase 11 · 16.)*
- **CrewAI.** Status mengalir sebagai string antar tugas melalui`context`lapangan, atau terstruktur melalui`output_pydantic`. Tidak ada penyimpanan per kru yang tahan lama; kamu lari sendiri jika kru harus selamat dari restart.
- **AutoGen.** Status adalah riwayat obrolan dan semua yang ditentukan pengguna`context`. Transkrip percakapan tetap ada; keadaan alur kerja sewenang-wenang tidak terjadi kecuali kamu menulis adaptor.
- **Agno.** Driver penyimpanan internal (SQLite, Postgres, Mongo, Redis, DynamoDB) yang terpasang pada`Agent`melalui`storage=`— sesi percakapan dan kenangan pengguna tetap ada secara otomatis. Bukan checkpointer grafik lengkap; toko sesi.

### Pertanyaan bercabang

Setiap cabang agen non-sepele. Siapa yang memutuskan cabang itu penting.

- **LangGraph** — kamu yang memutuskan, melalui tepi bersyarat. Perutean adalah fungsi Python dengan cabang bernama. Cabang-cabangnya adalah kelas satu dalam grafik yang dikompilasi; pos pemeriksaan mencatat cabang mana yang diambil.
- **CrewAI** — manajer memutuskan dalam mode hierarki; dalam mode berurutan kamu memutuskan pada waktu pembuatan. Perutean tersirat dalam daftar tugas; tidak ada kata "jika" kelas satu di luar perintah manajer.
- **AutoGen** — agen memutuskan melalui obrolan. Percabangan muncul dari siapa yang berbicara selanjutnya.`GroupChatManager`memilih pembicara berikutnya; kamu dapat menulis tangan a`speaker_selection_method`tapi defaultnya adalahLLM-didorong.
- **Agno** — agen memutuskan alat mana yang akan dihubungi selanjutnya. Tim memilikicoordinator/router/collaboratormodus; bercabang lebih dari itu adalah tanggung jawab pengembang.

### Pertanyaan tentang observasi

- **LangGraph** — OpenTelemetry melalui LangSmith atau eksportir OTel mana pun. Setiap transisi node adalah rentang jejak; pos pemeriksaan berfungsi ganda sebagai jejak yang dapat diputar ulang. LangSmith adalah opsi pihak pertama;Langfuse/Phoenixjuga memiliki adaptor.
- **CrewAI** — OpenTelemetry kelas satu sejak akhir tahun 2025; integrasi dengan Langfuse, Phoenix, Opik, AgentOps.
- **AutoGen** — Integrasi OpenTelemetry melalui`autogen-core`; AgentOps dan Opik memiliki konektor. Perincian pelacakan dilakukan per pesan agen, bukan per node.
- **Agno** — bawaan`monitoring=True`bendera plus eksportir OpenTelemetry; integrasi yang erat dengan Langfuse untuk pelacakan sesi.

### Biaya dan latensi

Keempat kerangka kerja menambahkan overhead per panggilan (logika kerangka kerja, validasi, serialisasi). Urutan kasar peningkatan overhead: Agno ≈ LangGraph < CrewAI ≈ AutoGen. Perbedaannya didominasi oleh seberapa banyak tambahannyaLLMmerutekan kerangka kerja tidak. Manajer hierarki CrewAI menggunakan token untuk memutuskan siapa yang berikutnya; AutoGen`GroupChatManager`demikian pula. LangGraph hanya menggunakan token di tempat kamu menulis`llm.invoke`. Jalur agen tunggal Agno tipis.

Jika biaya per proses penting, pilih perutean eksplisit (LangGraph edge, AutoGen`speaker_selection_method`) berakhirLLM-perutean yang dipilih.

### Interoperabilitas

- **LangGraph** ↔ **LangChain** alat, retriever, LLM. Kelas satuMCPadaptor (alat yang diimpor sebagaiMCPserver).
- **CrewAI** ↔ alat yang diwarisi`BaseTool`; Alat LangChain, alat LlamaIndex, danMCPsemua alat beradaptasi. Delegasi kru-ke-kru melalui`allow_delegation=True`.
- **Gen Otomatis** →`FunctionTool`membungkus semua callable Python;MCPadaptor tersedia. Kopling ketat keAG2ekosistem untuk pola agen-ke-agen.
- **Agno** →`@tool`dekorator atau subkelas BaseTool;MCPadaptor; alat dapat dibagikan ke seluruh agen dan tim.

## Keterampilan

> kamu dapat menjelaskan, dalam satu kalimat, mengapa kerangka kerja tertentu tepat untuk masalah agen tertentu.

Daftar periksa pra-pembuatan:

1. **Gambarlah bentuknya.** Apakah ini grafik (keadaan yang diketik, transisi yang diberi nama)? Sebuah permainan peran (para spesialis menyerahkan pekerjaan)? Obrolan (agen berbicara sampai selesai)? Agen tunggal dengan alat?
2. **Tentukan siapa yang mencabang.** Percabangan yang ditentukan pengembang → LangGraph. Diputuskan oleh manajer-agen → Hirarki CrewAI. Muncul obrolan → AutoGen. Alat-panggilan-diputuskan → Agno.
3. **Periksa anggaran negara.** Apakah kamu memerlukan resume dari pos pemeriksaan? Perjalanan waktu? Gangguan manusia di tengah jalan? Jika ya, LangGraph adalah defaultnya; Sesi Agno mencakup keadaan cakupan percakapan.
4. **Periksa anggaran biaya.**LLM-Perutean yang dipilih memerlukan token tambahan per giliran. Jika agen berjalan ribuan kali sehari, pilih perutean eksplisit.
5. **Anggarkan overhead kerangka kerja.** Setiap kerangka kerja merupakan ketergantungan lainnya. Jika tugasnya ada duaLLMpanggilan dan alat, tulis 30 baris Python biasa; tidak ada kerangka kerja lebih murah daripada tidak ada kerangka kerja.

Menolak untuk menggunakan kerangka kerja sebelum kamu dapat menggambar grafik, bagan organisasi, obrolan, atau kotak agen. Tolak untuk memilih salah satu yang memaksa kamu melawan model negaranya demi hal yang sebenarnya kamu butuhkan.

## Matriks Keputusan

| Problem shape | Preferred framework | Why |
|---------------|---------------------|-----|
| Workflow DAG with typed state, human approvals, long-running | LangGraph | First-class state, checkpointer, interrupts, time-travel. |
| Research / writing pipeline with distinct roles | CrewAI (sequential) or LangGraph subgraphs | Role-per-task is cheap to express in CrewAI; scale up with LangGraph when branching gets complex. |
| Proposer-critic or teacher-student dialogue | AutoGen | Two-agent chat is its native shape. |
| Single agent with tools, sessions, memory | Agno | Thinnest setup, built-in storage and memory. |
| Thousands of parallel fanouts with reducers | LangGraph + `Send` | The only one with a first-class parallel-dispatch API. |
| Quick prototype, no framework commitment | Plain Python + provider SDK | No framework is the fastest framework. |

```figure
l5-framework-fit
```

## Latihan

1. **Mudah.** Ambil tugas yang sama — "meneliti kantor pusat Anthropic, menulis ringkasan 200 kata, mengutip sumber" — dan menerapkannya di LangGraph (empat node: rencanakan, cari, tulis, kutip) dan di CrewAI (tiga peran: peneliti, penulis, editor). Laporkan biaya token per proses dan baris kode.
2. **Medium.** Bangun tugas yang sama di AutoGen (obrolan peneliti ↔ penulis, editor bergabung melalui`GroupChat`) dan Agno (agen tunggal dengan`search_tools`dan`write_tools`, ditambah penyimpanan sesi). Beri peringkat keempat implementasi berdasarkan (a) biaya per pengoperasian, (b) kemampuan untuk melanjutkan setelah crash, (c) kemampuan untuk memasukkan persetujuan manusia sebelum langkah penulisan.
3. **Sulit.** Buat skrip pohon keputusan`pick_framework.py`yang membutuhkan deskripsi masalah singkat (JSON:`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`) dan mengembalikan rekomendasi dengan justifikasi satu kalimat. Verifikasikan pada enam kasus yang kamu desain sendiri.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Orchestration | "How the agents coordinate" | The layer that decides which node/role/agent runs next. |
| Durable state | "Resume after a restart" | State that survives process death, attached to a checkpoint or session store. |
| LLM-selected routing | "Let the model decide" | A planner LLM picks the next step each turn; flexible but pays tokens on every decision. |
| Explicit routing | "Developer decides" | A Python function or static edge picks the next step; cheap and auditable. |
| Crew | "A CrewAI team" | Roles + tasks + process (sequential or hierarchical) bound into a single runnable. |
| GroupChat | "AutoGen's multi-agent chat" | A managed conversation between N agents with a speaker selector. |
| Team (Agno) | "Multi-agent Agno" | Route / coordinate / collaborate mode over a set of agents. |
| StateGraph | "LangGraph's graph" | Typed-state, node, conditional-edge, checkpointer abstraction. |

## Bacaan Lebih Lanjut

- [Dokumentasi LangGraph](https://langchain-ai.github.io/langgraph/) — StateGraph, checkpointer, interupsi, perjalanan waktu.
- [Dokumentasi kruAI](https://docs.crewai.com/) — Kru, Arus, Agen, Tugas, Proses.
- [Dokumentasi AutoGen](https://microsoft.github.io/autogen/) — ConversableAgent, GroupChat, tim, alat.
- [Dokumentasi Agno](https://docs.agno.com/) — Agen, Tim, Alur Kerja, penyimpanan, memori.
- [Antropik — Membangun agen yang efektif (Des 2024)](https://www.anthropic.com/research/building-effective-agents) — pustaka pola (rangkaian cepat, perutean, paralelisasi, pekerja orkestra, pengoptimal-evaluator) kerangka kerja-agnostik.
- [Yao dkk., "ReAct: Mensinergikan Penalaran dan Tindakan" (ICLR2023)](https://arxiv.org/abs/2210.03629) — loop yang didandani oleh setiap kerangka kerja.
- [Wu dkk., "AutoGen: Mengaktifkan Generasi BerikutnyaLLMLamaran melalui Percakapan Multi-Agen" (2023)](https://arxiv.org/abs/2308.08155) — Makalah desain AutoGen.
- [Park et al., "Agen Generatif: Simulacra Interaktif Perilaku Manusia" (UIST2023)](https://arxiv.org/abs/2304.03442) — fondasi permainan peran yang dibangun oleh persona gaya CrewAI.
- Fase 11 · 16 (LangGraph) — kerangka kerja yang menjadi tolok ukur pelajaran ini.
- Fase 11 · 19 (Refleksi) — sebuah pola yang terpetakan dengan rapi ke LangGraph tetapi canggung untuk CrewAI.
- Fase 11 · 22 (Kemampuan observasi produksi) — cara melengkapi kerangka kerja mana pun yang kamu pilih.
