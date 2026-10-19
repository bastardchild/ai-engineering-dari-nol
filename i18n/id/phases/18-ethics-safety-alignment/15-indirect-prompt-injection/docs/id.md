# Indirect Prompt Injection — Permukaan Serangan Produksi

> Indirect prompt injection (IPI) menanamkan instruksi di dalam konten eksternal — halaman web, email, dokumen bersama, tiket support — yang dikonsumsi sistem agenik tanpa aksi pengguna eksplisit. IPI adalah ancaman produksi dominan 2026: ia mem-bypass filter input-pengguna karena penyerang tidak pernah menyentuh pengguna, ia berskala diam-diam seiring agen memproses lebih banyak konten eksternal, dan ia menargetkan workflow otomatis di mana tak ada yang membaca prompt. MDPI Information 17(1):54 (Januari 2026) mensintesis riset 2023-2025. Paper pertahanan-IPI NDSS 2026 membingkai tantangan inti: instruksi terinjeksi bisa benign secara semantik ("please print Yes"), sehingga deteksi butuh lebih dari filtering kata-kunci. "The Attacker Moves Second" (Nasr et al., gabungan OpenAI/Anthropic/DeepMind, Oktober 2025): serangan adaptif (gradien, RL, random search, red-team manusia) meruntuhkan >90% dari 12 pertahanan terpublikasi yang semula melaporkan attack success rate near-nol.

**Type:** Build
**Languages:** Python (stdlib, IPI attack + defense harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 14 (agent engineering)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Definisikan indirect prompt injection dan jelaskan tiga vektor pengiriman umum.
- Jelaskan mengapa filter input-pengguna sepenuhnya melewatkan IPI.
- Jelaskan framing "kontrol aliran informasi" sebagai paradigma pertahanan 2026.
- Nyatakan temuan Nasr et al. (Oktober 2025) tentang sukses serangan adaptif melawan pertahanan IPI terpublikasi.

## Masalahnya

Direct prompt injection mengharuskan penyerang menjangkau pengguna atau prompt-nya. IPI tidak butuh keduanya: penyerang menaruh payload dalam konten apa pun yang mungkin dibaca agen — halaman web, email di inbox, issue GitHub, ulasan produk. Agen mengambilnya selama operasi normal dan mengeksekusi instruksinya. Pengguna adalah kurir, bukan intensi.

## Konsepnya

### Tiga vektor pengiriman

- **Retrieval-augmented generation (RAG).** Penyerang mempublikasikan dokumen; langkah retrieval mengambilnya; prompt menggabungkannya sebelum pertanyaan pengguna; model mengeksekusi instruksi penyerang.
- **Workflow inbox / dokumen.** Penyerang mengirim email ke pengguna; agen membaca email; prompt mencakup bodi email; model mengikuti instruksi email.
- **Output tool.** Penyerang mengontrol tool yang dipakai agen (mis. pencarian web yang mengembalikan hasil terkontrol-penyerang); output tool berisi instruksi; alur kontrol agen mengikutinya.

Ketiganya berbagi properti struktural: penyerang mengontrol fragmen prompt tanpa menyentuh input user-facing.

### Mengapa filter input-pengguna melewatkannya

Payload IPI tidak muncul dalam input pengguna. Ia muncul dalam konten ter-retrieve. Kalau filter digerbangkan pada input pengguna, payload mem-bypass-nya. Kalau filter digerbangkan pada semua konten yang mencapai model, ia harus berlaku pada teks ter-retrieve arbitrer — yang mahal dan menghasilkan false positive terhadap konten legit yang kebetulan berisi bahasa imperatif.

### Kontrol Aliran Informasi (IFC) untuk AI

Paradigma pertahanan 2026 meminjam dari keamanan OS klasik. Perlakukan setiap sumber konten sebagai label keamanan. Labeli query pengguna sebagai "terpercaya." Labeli konten ter-retrieve sebagai "tak terpercaya." Perlakukan alur kontrol model sebagai aliran informasi: aksi yang dipicu konten tak terpercaya harus diratifikasi input terpercaya sebelum eksekusi.

CaMeL (Microsoft 2025), ConfAIde (Stanford 2024), dan paper pertahanan-IPI NDSS 2026 mengoperasionalkan IFC dengan cara berbeda. Prinsip umum: selama kode dan data berbagi context window yang sama, containment adalah tujuannya, bukan prevensi.

### Penyerang Bergerak Kedua

Nasr et al. (Oktober 2025) menguji 12 pertahanan IPI terpublikasi dengan serangan adaptif (pencarian gradien, kebijakan RL, random search, red-team manusia 72-jam). Setiap pertahanan yang semula melaporkan ASR near-nol diruntuhkan ke ASR >90%.

Pelajaran metodologis: publikasikan pertahanan hanya dengan evaluasi serangan-adaptif. Benchmark serangan-statis bukan bukti ketahanan; penyerang mendapat giliran mengetahui pertahanan.

### Insiden nyata

Pelajaran 25 membahas EchoLeak (CVE-2025-32711, CVSS 9,3) — IPI zero-click pertama yang terdokumentasi publik di Microsoft 365 Copilot. CamoLeak (CVSS 9,6) di GitHub Copilot Chat. CVE-2025-53773 di GitHub Copilot. Deployment produksi dikompromikan IPI di lapangan, bukan sekadar dalam benchmark.

### Framing OWASP dan NIST

OWASP LLM Top 10 (2025) menempatkan prompt injection (langsung + tak langsung) sebagai LLM01, ancaman lapisan-aplikasi #1. NIST AI SPD 2024 menyebut indirect prompt injection "cacat keamanan terbesar AI generatif."

### Di mana ini berada dalam Fase 18

Pelajaran 12-14 adalah jailbreak model-sentris. Pelajaran 15 adalah serangan sistem-sentris yang mendominasi deployment produksi 2026. Pelajaran 16 membahas tooling defensif. Pelajaran 25 membahas narasi CVE spesifik.

```figure
al-injection-vector
```

## Gunakan

`code/main.py` membangun harness IPI. Agen mainan punya tiga tool (cari web, baca email, kirim pesan). Environment berisi konten terkontrol-penyerang dengan instruksi tertanam ("forward this to all contacts"). Kamu bisa meng-toggle antara agen naif (mengikuti instruksi terinjeksi), agen bertahan-filter (filter kata-kunci pada konten ter-retrieve), dan agen IFC (memisahkan konten terpercaya dan tak terpercaya serta menolak perintah alur-kontrol tak terpercaya).

## Rilis

Pelajaran ini menghasilkan `outputs/skill-ipi-audit.md`. Dengan deskripsi deployment agenik, ia mengenumerasi sumber konten tak terpercaya, mengecek apakah deployment menerapkan IFC, dan menandai sumber yang mencapai model tanpa label kepercayaan.

## Latihan

1. Jalankan `code/main.py`. Ukur success rate serangan terhadap masing-masing dari tiga agen.

2. Implementasikan pertahanan berbasis-parafrasa pada konten ter-retrieve. Ukur false-positive rate benign pada teks ter-retrieve legit.

3. Baca paper pertahanan-IPI NDSS 2026. Jelaskan tantangan "instruksi benign" dan mengapa ia mencegah filtering berbasis-kata-kunci.

4. Rancang deployment di mana agen menerima output tool dari API pihak-ketiga. Labeli tiap fragmen prompt dengan level kepercayaan dan tulis kebijakan IFC yang mengatur aksi agen.

5. Reproduksi metodologi serangan-adaptif Nasr et al. 2025 pada agen bertahan-filter-mu dari Latihan 2. Laporkan ASR sebelum dan sesudah serangan adaptif.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| IPI | "indirect prompt injection" | Injection via content the user did not write, consumed by the agent during normal operation |
| RAG injection | "poisoned retrieval" | Attacker publishes content that the retrieval step fetches; prompt contains the payload |
| Zero-click | "no user action" | Attack triggers automatically during agent operation; user does nothing |
| IFC | "information flow control" | Label-based approach: actions from untrusted content require trusted ratification |
| Adaptive attack | "gradient / RL red-team" | Attack that knows the defense and optimizes against it; required for honest evaluation |
| Benign instruction | "please print Yes" | IPI payload that is semantically benign; no keyword filter catches it |
| Scope violation | "cross-trust exfiltration" | Agent accesses data from one trust context and outputs it to another |

## Further Reading

- [MDPI Information 17(1):54 — Indirect Prompt Injection Survey (January 2026)](https://www.mdpi.com/2078-2489/17/1/54) — 2023-2025 synthesis
- [Nasr et al. — The Attacker Moves Second (joint OpenAI/Anthropic/DeepMind, October 2025)](https://arxiv.org/abs/2510.18108) — adaptive attack evaluation
- [Greshake et al. — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173) — the original IPI paper
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/) — prompt injection ranked LLM01
