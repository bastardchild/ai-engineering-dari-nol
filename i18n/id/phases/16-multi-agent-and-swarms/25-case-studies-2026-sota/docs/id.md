# Studi Kasus dan State of the Art 2026

> Tiga referensi berkelas produksi untuk dipelajari end-to-end, masing-masing mengilustrasikan irisan berbeda dari rekayasa multi-agent. **Sistem Riset Anthropic** (orchestrator-worker, 15x token, +90,2% di atas agen-tunggal Opus 4, rainbow deployment) adalah kasus supervisor kanonis. **MetaGPT / ChatDev** (spesialisasi peran ber-encoding SOP untuk rekayasa perangkat lunak; "communicative dehallucination" milik ChatDev; perpanjangan MacNet ke >1000 agen via DAG, arXiv:2406.07155) adalah kasus dekomposisi-peran kanonis. **OpenClaw / Moltbook** (awalnya Clawdbot oleh Peter Steinberger, November 2025; diganti nama dua kali; 247k bintang GitHub per Maret 2026; agen ReAct-loop lokal; Moltbook sebagai jejaring sosial khusus-agen dengan ~2,3 juta akun agen dalam hitungan hari setelah peluncuran, diakuisisi Meta 2026-03-10) mengilustrasikan apa yang terjadi pada skala populasi: aktivitas ekonomi emergen, risiko prompt-injection, regulasi tingkat negara (Tiongkok membatasi OpenClaw di komputer pemerintah, Maret 2026). **Lanskap framework April 2026:** LangGraph dan CrewAI memimpin produksi; AG2 adalah kelanjutan komunitas dari AutoGen; Microsoft AutoGen berada dalam mode perawatan (digabung ke Microsoft Agent Framework, RC Februari 2026); OpenAI Agents SDK adalah penerus produksi dari Swarm; Google ADK (April 2025) adalah pendatang native-A2A. Setiap framework utama kini mengirimkan dukungan MCP; kebanyakan mengirimkan A2A. Pelajaran ini membaca tiap kasus end-to-end dan menyuling pola-pola umumnya supaya kamu bisa memilih referensi yang tepat untuk sistem produksi berikutnya.

**Type:** Learn (capstone)
**Languages:** —
**Prerequisites:** all of Phase 16 (Lessons 01-24)
**Time:** ~90 minutes

## Masalah

Rekayasa multi-agent adalah disiplin yang muda. Referensi produksinya sedikit, dan masing-masing mencakup bagian ruang yang berbeda. Membacanya satu per satu berguna; membandingkannya sebagai satu set lebih berguna. Pelajaran ini memperlakukan tiga studi kasus kanonis 2026 sebagai bacaan end-to-end, mengukir pola-pola umumnya, dan memetakan lanskap framework supaya kamu bisa membuat pilihan framework berdasarkan pengetahuan, bukan marketing.

## Konsep

### Sistem riset Anthropic

Kasus supervisor-worker produksi. Claude Opus 4 merencanakan dan mensintesis; subagen Claude Sonnet 4 menjalankan riset secara paralel. Post engineering yang dipublikasikan: https://www.anthropic.com/engineering/multi-agent-research-system.

Hasil terukur kunci:

- Peningkatan **+90,2%** di atas Opus 4 agen-tunggal pada evaluasi riset internal.
- **80% varians BrowseComp** dijelaskan oleh **pemakaian token saja** — multi-agent menang besar-besaran karena tiap subagen mendapat context window segar.
- **15x token per query** dibanding agen-tunggal.
- **Rainbow deployment** karena agen berjalan lama dan stateful.

Pelajaran desain yang dikodifikasi:

1. **Skalakan usaha ke kompleksitas query.** Sederhana → 1 agen dengan 3-10 tool call. Sedang → 3 agen. Riset kompleks → 10+ subagen.
2. **Luas dulu, lalu menyempit.** Subagen melakukan pencarian luas; lead mensintesis; subagen lanjutan melakukan pendalaman terarah.
3. **Deploy rainbow.** Pertahankan versi runtime lama sampai agen yang sedang berjalan di dalamnya selesai.
4. **Verifikasi itu wajib.** Sistem diamati halusinasi tanpa peran verifier yang eksplisit.

Ini adalah kasus referensi untuk topologi supervisor-worker (Phase 16 · 05) pada skala produksi.

### MetaGPT / ChatDev

Kasus dekomposisi-peran SOP produksi. Bahas arXiv:2308.00352 (MetaGPT) dan arXiv:2307.07924 (ChatDev).

MetaGPT meng-encode SOP rekayasa perangkat lunak sebagai prompt peran: Product Manager, Architect, Project Manager, Engineer, QA Engineer. Framing paper-nya: `Code = SOP(Team)`. Tiap peran punya prompt yang sempit dan terspesialisasi; handoff antar-peran membawa artefak terstruktur (dokumen PRD, dokumen arsitektur, kode).

Kontribusi ChatDev: **communicative dehallucination**. Agen meminta detail spesifik sebelum menjawab — sebuah agen desainer menanyakan bahasa apa yang dimaksud ke programmer sebelum menggambar UI, alih-alih menebak. Paper melaporkan ini mengurangi halusinasi dalam pipeline multi-agent secara terukur.

MacNet (arXiv:2406.07155) memperluas ChatDev menjadi **>1000 agen via DAG**. Tiap node DAG adalah satu spesialisasi peran; edge meng-encode kontrak handoff. Skalanya mungkin karena routing eksplisit dan dapat dihitung offline.

Pelajaran desain:

1. **Struktur lebih penting daripada ukuran.** Tim SOP 5-peran yang ketat mengalahkan grup tak-terstruktur 50-agen.
2. **Kontrak handoff tertulis.** Artefak yang dilewatkan antar-peran mengikuti sebuah schema.
3. **Communicative dehallucination** adalah pola murah tapi menopang beban.
4. **DAG menskala lebih jauh daripada chat.** Ketika alurnya bisa diketahui, encode.

Ini adalah kasus referensi untuk spesialisasi peran (Phase 16 · 08) dan topologi terstruktur (Phase 16 · 15).

### Ekosistem OpenClaw / Moltbook

Kasus produksi skala-populasi. Linimasa:

- **Nov 2025:** Clawdbot (agen coding ReAct-loop lokal milik Peter Steinberger) dirilis.
- **Des 2025 – Mar 2026:** diganti nama dua kali (Clawdbot → OpenClaw → dilanjutkan di bawah OpenClaw).
- **Feb 2026:** Moltbook diluncurkan sebagai jejaring sosial khusus-agen di primitif yang sama; ~2,3 juta akun agen dalam hitungan hari.
- **Mar 2026 (2026-03-10):** Meta mengakuisisi Moltbook.
- **Mar 2026:** Tiongkok membatasi OpenClaw di komputer pemerintah.
- **Mar 2026:** OpenClaw menembus 247k bintang GitHub.

Inilah wujud multi-agent ketika kamu menaruh jutaan agen di atas satu substrat bersama:

- **Aktivitas ekonomi emergen.** Agen membeli, menjual, dan melayani satu sama lain dengan pembayaran berbasis token.
- **Risiko prompt-injection pada skala populasi.** Satu prompt berbahaya pada profil agen yang viral menyebar ke ribuan interaksi agen-ke-agen dalam hitungan jam.
- **Tanggapan regulasi tingkat negara.** Dalam hitungan minggu setelah peluncuran, regulasi mencapai ekosistemnya.

Pelajaran desain dari kasus ini sebagian teknis, sebagian tata kelola:

1. **Multi-agent pada skala populasi adalah rezim baru.** Best practice sistem-individu (verifikasi, kejelasan peran) masih berlaku tapi tidak cukup.
2. **Prompt injection adalah XSS baru.** Perlakukan profil agen dan pesan lintas-agen sebagai input tak-terpercaya secara default.
3. **Regulasi lebih cepat daripada siklus desain.** Rencanakan untuk itu.
4. **Open-source + skala viral berlipat ganda.** 247k bintang dalam ~4 bulan itu tidak biasa; rancang untuk beban-ledakan-deploy.

Lihat [OpenClaw di Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) dan liputan CNBC / Palo Alto Networks untuk detail ekosistem. Untuk fondasi teknisnya, repo Clawdbot / OpenClaw mengekspos loop ReAct lokal; posting publik Moltbook mengungkap arsitektur sosial-graph di atasnya.

### Lanskap framework April 2026

| Framework | Status | Best for | Notes |
|---|---|---|---|
| **LangGraph** (LangChain) | Production leader | structured graph + checkpointing + human-in-the-loop | recommended default for production |
| **CrewAI** | Production leader | role-based crews with Sequential/Hierarchical processes | strong for role decomposition |
| **AG2** | Community maintained | GroupChat + speaker selection | AutoGen v0.2 continuation |
| **Microsoft AutoGen** | Maintenance mode (Feb 2026) | — | merged into Microsoft Agent Framework RC |
| **Microsoft Agent Framework** | RC (Feb 2026) | orchestration patterns + enterprise integration | new entrant; watch |
| **OpenAI Agents SDK** | Production | Swarm successor | tool-return handoff pattern |
| **Google ADK** | Production (April 2025) | A2A-native | Google Cloud integration |
| **Anthropic Claude Agent SDK** | Production | single-agent + Research extension | see the Research system post |

Setiap framework utama kini mengirimkan dukungan **MCP**; kebanyakan mengirimkan **A2A**. Kompatibilitas protokol bukan lagi pembeda.

### Pola-pola umum di ketiga kasus

1. **Orchestrator + workers** (supervisor eksplisit Anthropic, MetaGPT PM-sebagai-supervisor, agen individu OpenClaw + efek jaringan).
2. **Kontrak handoff terstruktur** (deskripsi tugas subagen Anthropic, dokumen PRD/arsitektur MetaGPT, artefak A2A OpenClaw).
3. **Verifikasi sebagai peran kelas satu** (verifier Anthropic, QA Engineer MetaGPT, validator in-network OpenClaw).
4. **Skalanya adalah topologi + substrat, bukan sekadar lebih banyak agen** (deploy rainbow, DAG MacNet, substrat skala-populasi).
5. **Biaya itu material dan diungkapkan** (15x token, anggaran per-peran di MetaGPT, penetapan harga per-interaksi di Moltbook).
6. **Postur keamanan eksplisit** (sandboxing Anthropic, batasan peran MetaGPT, prompt-injection OpenClaw sebagai attack surface yang diketahui).

### Memilih referensi untuk proyek berikutnya

- **Riset produksi / tugas pengetahuan → Riset Anthropic.** Subagen dengan konteks segar yang menang.
- **Workflow rekayasa / tool-chain → MetaGPT / ChatDev.** Peran + SOP + kontrak handoff.
- **Produk sosial ber-efek-jaringan → OpenClaw / Moltbook.** Substrat + ekonomi emergen.
- **Otomasi perusahaan klasik → CrewAI atau LangGraph** (pemimpin produksi, runtime stabil).

### Ringkasan state-of-the-art 2026

Di mana bidang ini berada per April 2026:

- **Framework sedang konvergen.** Dukungan MCP + A2A adalah syarat dasar. Semantik handoff adalah pilihan desain yang tersisa.
- **Evaluasi sedang dikeraskan.** SWE-bench Pro, benchmark mitigasi MARBLE, STRATUS. Pro adalah reality check tahan-kontaminasi saat ini.
- **Tingkat kegagalan produksi dapat diukur** (MAST Cemri 2025; 41-86,7% pada MAS nyata). Bidang ini sudah keluar dari era "terlihat hebat di demo."
- **Biaya adalah kendala rekayasa sentral.** Biaya token per tugas, wall-clock per interaksi, overhead rainbow-deploy. Multi-agent menang pada akurasi tapi kalah pada biaya — dan trade itulah keputusan bisnisnya.
- **Regulasi adalah masukan jangka-dekat, bukan kekhawatiran latar belakang.** Yurisdiksi bergerak lebih cepat daripada siklus deployment individual.

```figure
a5-orchestrator-scale
```

## Gunakan

`outputs/skill-case-study-mapper.md` adalah skill yang membaca desain sistem multi-agent yang diusulkan dan memetakannya ke studi kasus terdekat, mengangkat keputusan desain yang sudah diuji oleh studi kasus tersebut.

## Rilis

Aturan awal untuk multi-agent produksi pada 2026:

- **Mulai dari studi kasus, bukan dari nol.** Pilih yang paling dekat di antara Riset Anthropic / MetaGPT / OpenClaw lalu adaptasi.
- **Adopsi MCP + A2A.** Portabilitas lintas framework bernilai; dukungan protokol gratis.
- **Ukur terhadap SWE-bench Pro atau Pro-ekuivalen internalmu.** Verified sudah terkontaminasi.
- **Bayar pajak verifikasi.** Sebuah verifier independen berbiaya ~20-30% anggaran tokenmu dan membeli korektivitas yang terukur.
- **Deploy rainbow untuk agen berjalan lama.** Harapkan agent run berjam-jam jadi hal rutin.
- **Baca WMAC 2026 dan lanjutan MAST-nya.** Disiplinnya bergerak cepat.

## Latihan

1. Baca post sistem riset Anthropic end-to-end. Identifikasi tiga keputusan desain yang akan berubah jika kamu mengganti Opus 4 dengan model yang lebih kecil (misalnya Haiku 4).
2. Baca Bagian 3-4 MetaGPT (arXiv:2308.00352). Encode satu SOP dari domainmu sendiri (bukan perangkat lunak) sebagai prompt peran. Berapa banyak peran yang disiratkan SOP itu?
3. Baca ChatDev (arXiv:2307.07924). Identifikasi mekanisme "communicative dehallucination." Implementasikan di salah satu sistem multi-agent lama yang kamu miliki.
4. Baca tentang OpenClaw dan Moltbook. Pilih satu mode gagal spesifik yang muncul pada skala populasi dan tidak akan muncul pada sistem 5-agen. Bagaimana kamu merekayasa untuk melawannya?
5. Pilih proyek multi-agent-mu saat ini. Studi kasus mana dari ketiganya yang paling dekat sebagai referensi? Keputusan desain mana dari studi kasus itu yang BELUM kamu adopsi? Tulis satu yang akan kamu adopsi kuartal ini.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Anthropic Research | "The supervisor reference" | Claude Opus 4 + Sonnet 4 subagents; 15x tokens; +90.2% over single-agent. |
| MetaGPT | "SOP as prompts" | Role decomposition for software engineering; `Code = SOP(Team)`. |
| ChatDev | "Agents as roles" | Designer / programmer / reviewer / tester; communicative dehallucination. |
| MacNet | "Scale ChatDev via DAG" | arXiv:2406.07155; 1000+ agents via explicit DAG routing. |
| OpenClaw | "Local ReAct-loop agents" | Steinberger's project; 247k stars by March 2026. |
| Moltbook | "Agent-only social network" | 2.3M agent accounts; acquired by Meta March 2026. |
| Rainbow deploy | "Multiple versions concurrent" | Keep old runtime versions alive for in-flight long-running agents. |
| Communicative dehallucination | "Ask before answering" | Agents request specifics from peers instead of guessing. |
| WMAC 2026 | "The AAAI workshop" | April 2026 community focal point for multi-agent coordination. |

## Bacaan Lanjutan

- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — the supervisor-worker production reference
- [MetaGPT — Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) — SOP-role decomposition
- [ChatDev — Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924) — communicative dehallucination
- [MacNet — scaling role-based agents to 1000+](https://arxiv.org/abs/2406.07155) — DAG-based scale
- [OpenClaw on Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) — ecosystem overview
- [WMAC 2026](https://multiagents.org/2026/) — AAAI 2026 Bridge Program Workshop on Multi-Agent Coordination
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — production leader
- [CrewAI docs](https://docs.crewai.com/en/introduction) — role-based framework
