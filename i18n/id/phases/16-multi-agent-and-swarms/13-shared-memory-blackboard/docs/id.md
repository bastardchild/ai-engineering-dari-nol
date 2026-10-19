# Memori Bersama dan Pola Blackboard

> Dua pendekatan hidup berdampingan di sistem multi-agent 2026: **message pool** (semua orang melihat pesan semua orang, seperti AutoGen GroupChat atau MetaGPT) dan **blackboard dengan langganan** (agent berlangganan ke event yang relevan, seperti Context-Aware MCP atau framework Matrix). Keduanya adalah satu-satunya bagian bersifat-stateful dari sebuah sistem multi-agent — yang berarti keduanya adalah tempat bug menarik tinggal. Mode kegagalan rujukannya adalah **memory poisoning**: satu agent berhalusinasi sebuah "fakta," agent lain memperlakukannya sebagai sesuatu yang terverifikasi, dan akurasi merosot secara bertahap dengan cara yang jauh lebih sulit didebug daripada crash seketika. Pelajaran ini membangun kedua struktur dari stdlib, menyuntikkan sebuah serangan poisoning, dan menunjukkan tiga mitigasi yang benar-benar bekerja di produksi.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## Masalahnya

Sistem multi-agent membutuhkan tempat bagi agent untuk berbagi fakta. Opsi harfiahnya adalah "meneruskan semuanya lewat pesan" — tapi itu menemukan ulang status bersama dengan salinan ekstra. Opsi lain adalah "beri semua orang log global" — tapi log global tumbuh tanpa batas dan mudah diracuni. Opsi ketiga adalah "proyeksikan sebuah tampilan per agent" — skalabel tetapi berat schema.

Ketika salah satu agent berhalusinasi dan menulis halusinasi itu ke status bersama, setiap agent hilir yang membaca status itu mengadopsi halusinasi tersebut sebagai fakta. Pada saat manusia menyadarinya, rantai penalaran sudah sedalam lima langkah dan akar masalahnya adalah pesan ketiga yang pernah ditulis. Mendegradasi akurasi multi-agent lebih sulit daripada men-debug sebuah crash.

Inilah memory poisoning. Ini adalah keluarga kegagalan kedua paling terdokumentasi dalam taksonomi MAST (Cemri et al., arXiv:2503.13657) dan bersifat struktural: desain memori bersama apa pun tanpa provenance dan tanpa verifier yang tidak bisa menulis pada akhirnya akan menampilkannya.

## Konsepnya

### Dua topologi utama

**Message pool penuh.** Setiap agent membaca setiap pesan. AutoGen GroupChat dan MetaGPT memakai ini. Sederhana, transparan, bisa diperiksa, tetapi tidak menskalakan lewat ~10 agent karena konteks tiap agent penuh dengan pekerjaan agent lain.

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Blackboard dengan langganan.** Agent mendeklarasikan minat pada topik; substrate hanya mengarahkan pesan yang relevan. CA-MCP (arXiv:2601.11595) dan framework terdesentralisasi Matrix (arXiv:2511.21686) memakai ini. Menskalakan lebih jauh, tetapi membutuhkan perancangan schema di muka agar langganan bermakna.

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### Kapan masing-masing menang

- **Pool penuh** menang ketika agent sedikit (< 10), heterogen, dan percakapannya berhorizon pendek. Menalar siapa mengatakan apa menjadi trivial ketika semua orang melihat segalanya.
- **Blackboard** menang ketika agent banyak, homogen dalam peran tetapi banyak dalam jumlah (swarms), dan percakapannya berjalan lama. Routing menghemat biaya token dan pencemaran konteks.

Sistem produksi sering mencampurkannya: sebuah pool penuh kecil di atas (lapisan perencanaan), blackboard di bawah (lapisan pekerja).

### Memory poisoning, dalam satu skenario

Tiga agent mengerjakan sebuah tugas riset. Agent A adalah agent retrieval. Agent B adalah summarizer. Agent C adalah analis.

1. A mengambil sebuah halaman dan menulis pesan ke status bersama: "Studi melaporkan peningkatan akurasi 42%."
2. Halaman yang diambil ternyata berbunyi "peningkatan 4.2%." A berhalusinasi satu desimal.
3. Membaca status bersama, B menulis: "Peningkatan akurasi besar 42% dilaporkan (sumber: A)."
4. Membaca status bersama, C menulis: "Rekomendasikan adopsi — kenaikan 42% bersifat transformatif."
5. Laporan akhir mengutip angka 42% yang tidak pernah ada.

Tidak ada agent yang crash. Tidak ada tes yang gagal. Sistemnya "bekerja." Halusinasi itu menyeberang dari konteks satu agent ke penalaran setiap agent hilir melalui status bersama.

### Mengapa ini struktural

Tanpa status bersama, halusinasi agent A tetap di konteks A. Agent hilir akan mengambil-ulang atau menurunkan-ulang dan mungkin menangkap kesalahannya. Dengan status bersama yang naif, konteks A menjadi konteks semua orang, dan halusinasi itu dicuci menjadi fakta.

Masalahnya bukan status bersama itu sendiri — melainkan status bersama **tanpa provenance dan tanpa verifier yang independen**. Tiga mitigasi menanganinya:

1. **Atribusikan provenance pada setiap tulisan.** Setiap entri di status bersama mencatat siapa yang menulisnya, kapan, di bawah prompt apa, dan (jika berlaku) sumber apa yang dikutip agent. Agent hilir membaca dengan skeptisisme yang terikat pada provenance.
2. **Versikan tulisan; perlakukan sebagai append-only.** Sebuah koreksi adalah entri baru yang menggantikan yang lama, bukan pembaruan di tempat. Jejak audit tetap terjaga.
3. **Simpan setidaknya satu agent yang tidak bisa menulis ke status bersama.** Sebuah agent verifier read-only menyampel entri, mengambil-ulang sumber, dan menandai ketidakkonsistenan. Karena tidak bisa menulis ke pool, ia tidak bisa diracuni oleh pool.

### Preseden blackboard (Hayes-Roth, 1985)

Pola blackboard mendahului agent LLM selama empat dekade. Hayes-Roth (1985, "A Blackboard Architecture for Control") menjelaskan Knowledge Sources spesialis yang mengamati sebuah blackboard global, menyumbang solusi parsial, dan memicu sumber lain. Blackboard 2026 (CA-MCP, Matrix) adalah pola yang sama dengan agent LLM sebagai Knowledge Sources dan blob JSON sebagai solusi parsial. Sastra lama sudah mendokumentasikan solusi untuk write contention, kontrol opportunistic, dan konsistensi yang kemudian ditemukan-ulang oleh sistem modern.

### Proyeksi vs tampilan penuh

Blackboard murni memberi setiap subscriber proyeksi yang sama (berlingkup topik). Desain yang lebih agresif adalah **proyeksi per agent**: setiap agent mendapat tampilan yang disesuaikan dengan perannya. State reducers milik LangGraph adalah implementasi kanonis 2026 — fungsi reducer melipat status global menjadi irisan khusus-peran.

Proyeksi per agent menskalakan lebih jauh tetapi membutuhkan sebuah schema. Tanpa itu, kamu membangun ulang proyeksi ad-hoc di setiap prompt agent.

### Pola write-contention

Beberapa agent menulis secara bersamaan adalah masalah konkurensi, bukan sekadar masalah LLM. Tiga pola bekerja:

- **Penulis sekuensial (satu produsen).** Semua tulisan melewati satu agent koordinator yang menserialkan. Sederhana, tetapi menjadi titik hambat.
- **Konkurensi optimistis dengan versioning.** Setiap entri punya sebuah versi; penulis gagal saat versi tidak cocok dan mencoba ulang. Teknik basis data klasik.
- **Partisi topik.** Agent yang berbeda memiliki topik yang berbeda. Tidak ada kontensi lintas-topik. Membutuhkan batas partisi yang dirancang.

Sebagian besar framework 2026 memakai penulis sekuensial secara bawaan karena panggilan LLM cukup lambat sehingga kontensi jarang terjadi dan titik hambatnya tidak terasa.

### Verifier yang tidak bisa menulis

Mitigasi yang paling menahan beban adalah verifier read-only. Aturan implementasinya:

- Verifier berbagi status dengan tim (membaca blackboard atau pool).
- Verifier tidak punya handle tulis ke status bersama — hanya ke kanal verifikasi terpisah.
- Verifier mengambil-ulang secara independen sumber yang dikutip dalam tulisan. Menandai ketidaksepakatan.
- Output verifier sendiri diarahkan ke manusia atau agent keputusan terpisah, tidak pernah dimasukkan kembali ke pool.

Tanpa pemisahan ini, output verifier menjadi entri baru di pool, yang berarti pool yang teracuni meracuni verifier, yang meracuni verifikasinya.

```figure
swarm-blackboard
```

## Bangun Sendiri

`code/main.py` mengimplementasikan kedua topologi dalam Python stdlib plus sebuah serangan poisoning mainan dan tiga mitigasi.

- `MessagePool` — log append-only thread-safe dengan pembacaan penuh.
- `Blackboard` — pub/sub berkey topik dengan langganan per agent.
- `ProvenanceEntry` — setiap tulisan mencatat (writer, timestamp, prompt_hash, source_uri).
- `PoisoningScenario` — menjalankan sebuah tugas riset tiga-agent di mana agent A berhalusinasi satu desimal. Mencetak laporan akhir.
- `Verifier` — sebuah agent read-only yang mengambil-ulang sumber dan menandai ketidakkonsistenan. Menjalankan skenario yang sama dengan verifier hadir.

Jalankan:

```
python3 code/main.py
```

Output yang diharapkan:
- Run 1 (tanpa verifier): halusinasi 42% menyebar ke laporan akhir.
- Run 2 (dengan verifier): verifier menandai ketidakkonsistenan itu, pool dilabeli "flagged", laporan akhir memuat sebuah pencabutan.

## Gunakan

`outputs/skill-memory-auditor.md` adalah sebuah skill yang mengaudit desain memori bersama dari sistem multi-agent mana pun untuk provenance, versioning, dan pemisahan verifier. Jalankan pada arsitektur multi-agent baru sebelum produksi.

## Rilis

Untuk desain memori bersama apa pun:

- Rekam provenance pada setiap tulisan: `(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`.
- Jadikan log tersebut append-only. Koreksi adalah entri baru yang merujuk entri yang digantikan.
- Deploy setidaknya satu agent verifier read-only dengan akses sumber independen.
- Arahkan output verifier ke kanal terpisah, bukan kembali ke pool bersama.
- Catat rasio tulisan yang merupakan penggantian — rasio yang naik adalah bukti dini pola halusinasi.

## Latihan

1. Jalankan `code/main.py`. Konfirmasi bahwa run 1 menyebar halusinasi dan run 2 menangkapnya.
2. Tambahkan halusinasi kedua: agent B mengarang ukuran dataset. Seharusnya verifier menangkap keduanya tanpa disetel-tangan untuk salah satunya.
3. Tukar pool penuh menjadi sebuah blackboard dengan partisi topik (`prices`, `summaries`, `analyses`). Skenario poisoning mana yang dibuat partisi topik lebih sulit dilakukan, dan mana yang tidak terbantu?
4. Baca Hayes-Roth (1985, "A Blackboard Architecture for Control"). Identifikasi dua pola kontrol dari makalah itu yang tidak dibahas di pelajaran ini dan yang akan menguntungkan sistem 2026.
5. Baca CA-MCP (arXiv:2601.11595). Petakan Shared Context Store-nya ke kelas MessagePool atau Blackboard di `code/main.py`. Primitif apa saja yang ditambahkan CA-MCP di atasnya?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Message pool | "Shared chat history" | Append-only log that every agent reads. Full transparency, poor scaling. |
| Blackboard | "Shared workspace" | Topic-keyed pub/sub. Agents subscribe to relevant topics. Scales farther. |
| Provenance | "Who wrote what" | Metadata on each write: writer, timestamp, prompt, sources. |
| Memory poisoning | "Hallucinations spreading" | One agent's error enters shared state, downstream agents adopt it as fact. |
| Append-only | "No in-place updates" | Corrections are new entries that supersede. Preserves audit trail. |
| Unwritable verifier | "Independent auditor" | Read-only agent that re-fetches sources and flags inconsistencies. |
| Projection | "Scoped view" | Per-agent view computed from global state. LangGraph reducers are the canonical case. |
| Knowledge Source | "Specialist agent" | Hayes-Roth's 1985 term for a blackboard participant. |

## Bacaan Lanjutan

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy; memory poisoning is a coordination-failure sub-family
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — Shared Context Store for coordinated MCP servers
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — message-queue-based blackboard without a central orchestrator
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — the per-agent projection pattern in production
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — provenance and verification notes from a production deployment
