# Agent Generatif dan Simulasi Emergen

> Park et al. 2023 (UIST '23, arXiv:2304.03442) mengisi **Smallville**, sebuah sandbox berisi 25 agent, dengan arsitektur tiga-bagian: **memory stream** (log bahasa alami), **reflection** (sintesis tingkat-tinggi yang dihasilkan agent atas stream-nya sendiri), dan **plan** (perilaku tingkat-hari, lalu sub-plan). Hasil landmark-nya adalah munculnya pesta Valentine: satu agent yang diberi benih "ingin mengadakan pesta Valentine," tanpa skrip lebih lanjut, menghasilkan undangan yang menyebar ke seluruh populasi, tanggal yang dikoordinasikan, dan pesta itu terjadi — dari 24 agent yang awalnya tidak tahu apa-apa tentangnya. Ablasi menunjukkan ketiga komponen diperlukan agar terlihat meyakinkan. Kegagalan yang terdokumentasi adalah galat norma-spatia (masuk ke toko yang tutup, memakai kamar mandi satu-orang). Inilah arsitektur rujukan untuk simulasi agent dan evaluasi sosial multi-agent pada 2026.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## Masalahnya

Sebagian besar sistem multi-agent adalah tim yang di-skrip ketat: planner merencanakan, coder coding, reviewer mereview. Itu bekerja untuk tugas yang terdefinisi baik. Itu tidak menangkap perilaku emergen dan tak-ter-skrip yang muncul ketika agent punya memori, prioritas, dan dunia terbuka. Riset, simulasi sosial, dan AI game yang semakin menjamur membutuhkan jenis kedua ini.

Arsitektur Smallville adalah benchmark untuk itu. Sebelum Park 2023, simulasi agent terbaik adalah pengikut-skrip yang dangkal; sesudahnya, pola itu menjadi bawaan untuk agent generatif di dunia terbuka. Jika kamu membangun sebuah simulasi agent pada 2026, kamu sedang memakai tiga komponen Smallville atau secara eksplisit memberi alasan mengapa tidak.

## Konsepnya

### Tiga komponennya

**Memory stream.** Sebuah log append-only atas observasi, aksi, refleksi, dan plan. Setiap entri punya timestamp, sebuah tipe, deskripsi (bahasa alami), dan metadata turunan: **recency**, **importance** (dinilai-sendiri 1-10 oleh agent), dan **relevance** (kemiripan kosinus terhadap query saat ini).

```
[2026-02-14 09:12:03] observation: Isabella Rodriguez asked me if I like jazz
[2026-02-14 09:14:22] reflection:   I enjoy long conversations about music
[2026-02-14 10:05:00] plan:         Attend Isabella's Valentine's Day party tonight
```

Retrieval memori menggabungkan ketiga skor: `score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`. Entri top-k masuk ke prompt saat ini.

**Reflection.** Secara berkala (setiap N memori atau pada event penting), agent menghasilkan sintesis orde-tinggi dari memori terbaru. Entri refleksi kembali ke stream dan bisa diretrieval seperti memori lain. Inilah cara agent membangun "pemahaman" — padanan arsitektur dari keyakinan jangka panjang.

**Plan.** Dekomposisi top-down. Pertama, sebuah rencana tingkat-hari dalam garis besar ("pergi kerja, makan malam dengan Klaus"). Lalu rencana tingkat-jam. Lalu rencana tingkat-aksi. Plan bisa direvisi: ketika sebuah observasi bertentangan dengan sebuah plan, agent membuat-ulang segmen yang terdampak.

### Mengapa ketiganya penting (ablation)

Park et al. menjalankan ablation yang membuang masing-masing observation, reflection, dan plan. Setiap ablation menyakiti kredibilitas:

- Tanpa **observation** agent kehilangan konteks dan bertindak berdasarkan keyakinan yang usang.
- Tanpa **reflection** agent tidak bisa membentuk keyakinan orde-tinggi; interaksi tetap dangkal.
- Tanpa **plan** perilaku menjadi derau reaktif; tujuan terurai.

Skor kredibilitas dari penilai manusia tertinggi dengan ketiganya; membuang salah satu menghasilkan regresi yang terukur.

### Kemunculan Valentine

Satu agent, Isabella Rodriguez, diberi tujuan "ingin mengadakan pesta Valentine di Hobbs Cafe pada 14 Februari pukul 5 sore." 24 agent lainnya tidak menerima benih semacam itu. Selama hari-hari yang disimulasikan:

1. Plan Isabella mencakup mengundang orang.
2. Setiap undangan menjadi sebuah observasi di memory stream seorang tetangga.
3. Refleksi tetangga itu menghasilkan keyakinan: "Isabella mengadakan pesta."
4. Plan tetangga itu memuat "hadiri pesta pada 14 Februari."
5. Tetangga memberi tahu tetangga lain. Undangan menyebar tanpa koordinasi pusat.
6. Pukul 5 sore pada 14 Februari, beberapa agent berkumpul di Hobbs Cafe.

Ini adalah emergence dalam arti teknis: perilaku tingkat-sistem (sebuah pesta) lahir dari interaksi lokal (undangan bilateral + perencanaan individu) tanpa sebuah orchestrator pusat.

### Mode kegagalan yang terdokumentasi

Park et al. secara eksplisit mendokumentasikan:

- **Galat norma spasia.** Agent berjalan masuk ke toko yang tutup. Agent mencoba memakai kamar mandi satu-orang yang sama. Agent makan di ruangan yang tidak dimaksudkan untuk makan. Model tidak menyimpulkan norma sosio-fisik dari lingkungan semata.
- **Meluapnya memori.** Run simulasi dalam membuat biaya retrieval memori meningkat. Remedi praktisnya: kompaksi memori berkala (ringkas-dan-pangkas) dan decay pada entri ber-importance rendah.
- **Halusinasi refleksi.** Refleksi bisa mengarang hubungan yang tidak ada di memory stream. Mitigasinya: sertakan id memori sumber di prompt refleksi dan verifikasi saat retrieval.

Ini adalah mode kegagalan yang relevan-dengan-produksi: simulasi agent 2026 mana pun mewarisi semuanya.

### Aturan implementasi tiga-komponen

1. **Memori bersifat append-only.** Jangan pernah memutasi sebuah entri memori. Koreksi adalah entri baru.
2. **Skor importance itu murah.** Panggil LLM untuk menilai importance 1-10 saat penulisan. Simpan cache skornya.
3. **Retrieval itu berperingkat, bukan penyaringan.** Top-k berdasarkan skor gabungan; jangan memakai filter keras (yang kehilangan konteks).
4. **Reflection berjalan secara berkala.** Picu ketika jumlah importance dari memori yang belum diproses melampaui sebuah threshold (misalnya, 150).
5. **Plan bisa direvisi.** Ketika sebuah observasi baru bertentangan dengan sebuah plan, hasilkan-ulang hanya segmen yang terdampak, bukan seluruh plan.

### Agent generatif melampaui Smallville

Sastra lanjutan 2024-2026 memperluas arsitektur ini:

- **Simulasi sosial multi-agent untuk riset kebijakan / pasar.** Populasi ala Smallville mensimulasikan perilaku pengguna sebagai respons terhadap fitur. Lebih cepat daripada tes A/B; akurasinya diperdebatkan.
- **NPC AI untuk game.** RPG dengan agent Smallville menghasilkan storyline emergen alih-alih quest ber-skrip.
- **Benchmark evaluasi agent generatif.** Alih-alih akurasi tugas, metriknya menjadi kredibilitas + koherensi perilaku selama run panjang.

Arsitekturnya adalah rujukannya. Perpanjangan menukar komponen (vector store untuk memori, reflection yang diperkaya retrieval, plan neurosimbolik) tetapi mempertahankan struktur tiga-bagian.

### Mengapa ini penting bagi engineering multi-agent

Smallville adalah proof of concept bahwa emergence multi-agent itu murah ketika komponennya tepat. Arsitektur ini kini sudah direplikasi pada model open-source (LLM yang lebih kecil kehilangan kredibilitas secara anggun, bukan tajam). Sistem produksi mana pun yang membutuhkan **perilaku sosial emergen** memakai bentuk ini. Sistem mana pun yang membutuhkan **eksekusi tugas yang ketat** memakai pola supervisor / peran / primitif dari awal fase ini.

```figure
a5-memory-reflection
```

## Bangun Sendiri

`code/main.py` mengimplementasikan ketiga komponen dalam Python stdlib dengan kebijakan agent ber-skrip (tanpa LLM sungguhan). Demo-nya mereproduksi kemunculan pesta Valentine dalam skala mini:

- `MemoryStream` — log append-only dengan retrieval recency/importance/relevance.
- `reflect(stream)` — refleksi ber-skrip atas memori terbaru ber-importance tinggi.
- `plan(agent_state)` — rencana tingkat-hari dan tingkat-jam berdasarkan keyakinan saat ini.
- Skenario: 5 agent. Agent 1 diawali dengan "adakan pesta jam 5 sore." Selama tick yang disimulasikan, undangan menyebar dan agent berkumpul.

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan: jejak tick-demi-tick. Pada tick terakhir, setidaknya 3 dari 5 agent memperlihatkan pesta di plan mereka, dan mereka berkumpul di lokasi pesta. Satu benih menghasilkan kedatangan yang terkoordinasi tanpa orchestrator apa pun.

## Gunakan

`outputs/skill-simulation-designer.md` merancang sebuah simulasi agent generatif: jumlah agent, skema memori, irama reflection, horizon plan, dan metrik evaluasi.

## Rilis

Aturan untuk simulasi produksi:

- **Memori adalah basis datanya.** Pilih sebuah store sungguhan (vector DB, Postgres) pada skala. Stdlib in-memory hanya untuk prototipe.
- **Catat jejak retrieval.** Untuk setiap aksi, catat memori top-k yang mendorongnya. Inilah kemampuan debug-mu.
- **Anggarkan token per agent.** Retrieve + reflect + plan per tick bagi tiap agent adalah O(k) panggilan LLM. N agent × T tick × panggilan-per-tick bisa menenggelamkan anggaranmu.
- **Kompakkan memori secara berkala.** Ringkas-dan-pangkas entri ber-importance rendah. Kebijakan retensi adalah sebuah keputusan desain, bukan detail.
- **Deteksi pelanggaran norma spasia / sosial** secara eksplisit. Arsitektur tidak mempelajarinya.

## Latihan

1. Jalankan `code/main.py`. Konfirmasi 3+ agent berkumpul di pesta. Tambah agent menjadi 10 — apakah emergence-nya masih terjadi?
2. Hapus langkah reflection. Seperti apa perilakunya? Petakan ke temuan ablation di Park 2023.
3. Perkenalkan sebuah tujuan berbenih yang bersaing ("Klaus ingin memberikan talk riset jam 5 sore"). Apakah agent terpecah, atau satu tujuan mendominasi? Apa yang menentukannya?
4. Tambahkan kendala spasia: Hobbs Cafe muat paling banyak 4 agent. Apakah simulasi menangani kelebihan itu dengan anggun, atau justru mengenai pola kegagalan "kamar mandi satu-orang"?
5. Baca Park et al. (arXiv:2304.03442) Bagian 6 (eksperimen perilaku emergen). Identifikasi satu perilaku yang tidak bisa direproduksi di versi mini-mu. Komponen arsitektur mana yang perlu kamu tingkatkan?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Memory stream | "The agent's diary" | Append-only log of observations, actions, reflections, plans. |
| Recency | "How new is the memory" | Exponential-decay score by age. |
| Importance | "How much does the agent care" | Self-rated 1-10 at write time. Cached. |
| Relevance | "How related to the current query" | Cosine similarity (embedding-based). |
| Reflection | "Higher-order belief" | Synthesis generated from recent memories, re-ingested as a new memory. |
| Plan | "Day/hour/action decomposition" | Top-down plan tree. Revisable when observations contradict. |
| Smallville | "Park 2023's sandbox" | 25-agent simulation that produced the Valentine's Day emergence. |
| Believability | "The quality metric" | Human-rater score for whether behavior seems like a plausible agent. |

## Bacaan Lanjutan

- [Park et al. — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — the reference architecture
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763) — publication venue
- [Smallville code release](https://github.com/joonspk-research/generative_agents) — reference Python implementation
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) — prior art for structured-memory agents
