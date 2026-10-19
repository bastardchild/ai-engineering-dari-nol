# Voting, Self-Consistency, dan Topologi Debat

> Agregasi termurah: sampel N agent independen, voting mayoritas. Self-consistency Wang et al. 2022 melakukan ini dengan satu model yang di-sampling N kali. Multi-agent memperluasnya dengan agent **heterogen** untuk keluar dari monoculture — model berbeda, prompt berbeda, temperature berbeda, konteks berbeda. Melampaui voting mayoritas, topologi debat itu penting: MultiAgentBench (arXiv:2503.01935, ACL 2025) mengevaluasi koordinasi star / chain / tree / graph dan menemukan **graph terbaik untuk riset**, dengan sebuah "coordination tax" setelah ~4 agent. AgentVerse (ICLR 2024) mendokumentasikan dua pola emergen — perilaku volunteer dan perilaku conformity — dan conformity sekaligus merupakan sebuah fitur (menemukan konsensus) dan sebuah risiko (groupthink, Pelajaran 24). Pelajaran ini memetakan ruang topologi, membangun tiap varian, dan mengukur coordination tax-nya.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 minutes

## Masalahnya

Debat bisa meningkatkan akurasi (Du et al., arXiv:2305.14325). Debat juga bisa menurunkannya. Apakah debat membantu bergantung pada empat pilihan struktural:

1. Siapa bicara dengan siapa (topologi).
2. Berapa banyak ronde (Du 2023: ronde dan agent sama-sama berpengaruh secara independen).
3. Apakah agent-nya heterogen (base model berbeda mematahkan monoculture).
4. Apakah ada suara adversarial (steel-manning vs. straw-manning).

Tim yang menempelkan "jalankan 5 agent dan voting" ke sebuah tugas sering mundur dibanding satu agent tunggal. Kegagalannya bukan acak. Kegagalan itu mengikuti topologi dan heterogenitas. Pelajaran ini adalah peta topologinya.

## Konsepnya

### Self-consistency, baseline satu-model

Wang et al. 2022 ("Self-Consistency Improves Chain of Thought Reasoning") meng-sampling model yang sama N kali pada temperature > 0 dan melakukan voting mayoritas atas jawaban jalur-penalarannya. Hasilnya di GSM8K: gain besar dengan N=40 sampel dibanding satu decode greedy. Self-consistency adalah pendahulu satu-agent dari voting multi-agent.

Batasannya: self-consistency memakai satu base model. Kesalahan berkorelasi secara struktural dari lahirnya. Jika model punya bias sistematis, semua N sampel memilikinya.

### Voting multi-agent, perpanjangan heterogen

Ganti N sampel dengan N agent yang *berbeda*. Base model berbeda (Claude, GPT, Llama), prompt berbeda, akses tools berbeda. Manfaatnya: kesalahan tidak berkorelasi. Biayanya: agent berbeda berbiaya berbeda; menyorotasinya menambah overhead.

Nama kanonis 2026 untuk debat heterogen adalah **A-HMAD** — Adversarial Heterogeneous Multi-Agent Debate. Tidak diadopsi secara universal, tetapi makalah-makalah memakai istilah itu untuk "model-model berbeda berdebat, yang mengurangi kesalahan berkorelasi dari runtuhnya monoculture."

### Empat topologi

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

Star: satu hub, semua sisanya hanya berbicara ke hub. Setara dengan supervisor-worker tanpa back-channel.
Chain: linear, setiap agent melihat output pendahulunya. Mirip pipeline.
Tree: hierarkis, dipakai oleh sistem agent hierarkis (Pelajaran 06).
Graph: siapa-ke-siapa. Termasuk clique fully-connected dan DAG sembarang.

### Coordination tax (MultiAgentBench)

MultiAgentBench (MARBLE, ACL 2025, arXiv:2503.01935) mem-benchmark star, chain, tree, graph pada satu rangkaian tugas yang mencakup riset, coding, dan perencanaan. Hasil terukur utamanya:

- Topologi **Graph** menang pada tugas riset. Informasi mengalir siapa-ke-siapa; agent bisa saling mengkritik.
- **Star** menang pada tugas faktual jawaban-cepat. Hub menyaring dan mengkonsolidasikan.
- **Chain** menang pada pipeline bertahap (penyempurnaan berstase).
- **Coordination tax** muncul setelah ~4 agent pada topologi graph. Wall-clock dan biaya token tumbuh lebih cepat daripada kualitas.

Atap 4-agent itu empiris, bukan fundamental. Ia mencerminkan kapasitas konteks LLM 2026: konteks tiap agent penuh dengan output para peer, dan nilai marginal dari menambahkan agent N+1 turun begitu semua orang bisa melihat semua orang.

### Strategi Debat Multi-Agent ("Should we be going MAD?")

arXiv:2311.17371 adalah survei 2023 tentang strategi MAD. Temuan utama yang direplikasi pihak lain: varian MAD yang *secara struktur mirip* dengan self-consistency (sampling independen + agregasi) sering kalah dari self-consistency ketika memakai anggaran yang sama. MAD paling membantu ketika agent benar-benar heterogen dan debatnya punya struktur adversarial (satu agent membantah).

### Pola emergen AgentVerse

AgentVerse (ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) mendokumentasikan dua perilaku yang muncul dari debat multi-agent bahkan tanpa desain eksplisit:

- **Volunteer.** Sebuah agent menawarkan bantuan ("Saya bisa mengambil langkah berikutnya") tanpa diminta. Berguna: ia mengalokasikan pekerjaan ke agent yang paling mampu untuk sebuah subtask.
- **Conformity.** Sebuah agent menyesuaikan posisinya agar cocok dengan seorang kritikus, bahkan ketika kritikus itu salah. Ini adalah padanan debat dari sycophancy (Pelajaran 14).

Conformity adalah alasan mengapa debat-sampai-sepakat memberi hadiah pada para pembuli. Ronde berbatas dengan hakim terpisah memitigasinya.

### Heterogenitas: tuas yang benar-benar menggeser akurasi

Sebuah pola pada sastra praktis 2024-2026: menukar salah satu dari N agent-mu dengan base model yang berbeda memberi lompatan akurasi lebih besar daripada menambah N sebanyak 1. Intuisinya adalah monoculture — setiap sumber kesalahan-independen baru bernilai lebih daripada satu sampel berkorelasi tambahan.

Dalam batasnya, heterogenitas mengalahkan kuantitas. Tiga model berbeda mengalahkan lima salinan satu model pada sebagian besar tugas yang punya ground truth yang bersih.

### Metode juri

Framework Sibyl (dikutip dalam sastra Minsky-LLM) memformalkan sebuah "jury" — sekumpulan kecil agent spesialis yang menyempurnakan jawaban dengan voting di setiap tahap. Berbeda dengan voting mayoritas biasa, sebuah juri punya peran: satu agent melakukan pemeriksaan-silang, satu menyuplai konteks, satu memberi skor pada plausibilitas. Metode juri adalah titik tengah antara voting biasa (murah, rentan-monoculture) dan MAD penuh (mahal, rentan-conformity).

### Kapan vote-dengan-debat mendominasi

- Pertanyaan itu punya ground truth (fakta, matematika, perilaku kode). Konvergensi voting bermakna.
- Agent bisa mengakses sumber atau tools yang berbeda (heterogenitas tersedia).
- Ronde berbatas (2-3 biasanya) dan ada hakim atau verifier terpisah.
- Anggaran mengizinkan 3-5 agent. Lewat 5-7 pada topologi graph, coordination tax mendominasi.

### Kapan vote-dengan-debat merugikan

- Pertanyaan itu berbentuk opini. Agent konvergen ke jawaban yang terlihat paling yakin, bukan yang paling benar.
- Semua agent berbagi satu base model. Monoculture membuat kensus tidak bermakna.
- Ronde tidak berbatas. Conformity menang setiap saat.
- Tugasnya sederhana. Sebuah agent tunggal dengan self-consistency pada N=5 lebih murah dan sama akuratnya.

```figure
sw-debate-topology
```

## Bangun Sendiri

`code/main.py` mengimplementasikan:

- `run_star(agents, hub, question)` — hub mem-poll setiap pekerja, mengagregasi.
- `run_chain(agents, question)` — penyempurnaan sekuensial.
- `run_tree(root, children, question)` — hierarkis dengan agregasi kedalaman-2.
- `run_graph(agents, question, rounds)` — debat semua-ke-semua, ronde berbatas.
- Sebuah dial heterogenitas ber-skrip: setiap agent punya `error_bias` yang menunjukkan kecenderungan salah sistematisnya.
- Sebuah harness pengukuran yang menjalankan setiap topologi pada N=3, 5, 7 dan melaporkan (akurasi, total_tokens, wallclock_simulated).

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan: sebuah tabel topologi × N → (akurasi, token, latensi). Graph menang pada N=3-5 pada tugas bergaya riset; star menang pada tugas faktual-cepat; graph pada N=7 menampilkan coordination tax (latensi membengkak lebih cepat daripada akurasi).

## Gunakan

`outputs/skill-topology-picker.md` adalah sebuah skill yang membaca deskripsi tugas dan merekomendasikan sebuah topologi (star / chain / tree / graph), sebuah N (jumlah agent), sebuah profil heterogenitas (base model yang dipakai), dan sebuah batas ronde.

## Rilis

Untuk ensemble apa pun:

- Mulai dengan **self-consistency pada N=5** memakai satu base model yang kuat. Itu baseline murahnya.
- Naikkan ke **voting heterogen pada N=3** jika akurasi penting. Ukur delta-nya.
- Naikkan ke **topologi debat** hanya jika tugasnya punya struktur (riset, multi-langkah) dan ronde berbatas memungkinkan.
- Selalu catat cluster minoritas. Ketika sebuah minoritas terus-menerus benar, kamu punya sebuah sinyal keragaman.
- Benchmark wall-clock dan token bersamaan dengan akurasi. "Akurasi lebih baik dengan biaya 10x" adalah sebuah keputusan bisnis.

## Latihan

1. Jalankan `code/main.py`. Plot kurva coordination-tax untuk topologi graph: akurasi vs N, token vs N. Pada N berapa kurva itu berbelok?
2. Implementasikan A-HMAD: tiga agent dengan bias yang sengaja berbeda. Bagaimana baseline dengan bias yang sama semua dibandingkan dengan A-HMAD pada serangan monoculture dari Pelajaran 14?
3. Tambahkan sebuah peran "hakim" ke topologi graph yang tidak voting, hanya memberi skor pada konsensus akhir. Apakah ini mengubah perilaku conformity yang emergen?
4. Baca makalah AgentVerse (ICLR 2024). Identifikasi perilaku emergen mana yang paling kuat ditampilkan implementasimu. Bisakah kamu memunculkan perilaku berlawanan dengan mengubah prompt?
5. Baca MultiAgentBench (arXiv:2503.01935) Bagian 4 (eksperimen topologi). Reproduksi hasil "graph-wins-research" pada satu tugas dari makalah itu memakai harness-mu.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Self-consistency | "Sample N times, vote" | Wang 2022. Single model, N temperature>0 samples, majority vote on reasoning paths. |
| Heterogeneity | "Different models" | Ensemble of different base models or prompt families. Breaks monoculture. |
| MAD | "Multi-agent debate" | Generic term for agents exchanging critiques over rounds. See Du 2023. |
| A-HMAD | "Adversarial Heterogeneous MAD" | MAD variant emphasizing different models + adversarial structure. |
| Topology | "Who talks to whom" | Star, chain, tree, graph. Determines information flow. |
| Coordination tax | "Diminishing returns" | Above ~4 agents on graph, cost grows faster than quality. |
| Volunteer behavior | "Unprompted help" | AgentVerse emergent pattern: an agent offers to take a step. |
| Conformity behavior | "Agreement under pressure" | AgentVerse emergent pattern: an agent aligns with a critic. |
| Jury | "Small specialized panel" | Sibyl-style ensemble with roles (examiner, context, scorer). |

## Bacaan Lanjutan

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — single-model baseline
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — both agents AND rounds matter independently
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — topology benchmark showing graph best for research, chain for pipelines
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — MAD-strategy survey; finds MAD often loses to self-consistency at equal budget
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — volunteer and conformity emergent patterns
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — reference benchmark implementation
