# Human-in-the-Loop: Propose-Then-Commit

> Konsensus HITL 2026 bersifat spesifik. Ia bukan "agen bertanya, pengguna mengklik Setujui." Ia adalah propose-then-commit: aksi yang diusulkan dipersisten ke penyimpanan durable dengan kunci idempotency; ditampilkan ke reviewer beserta intensi, lineage data, izin yang disentuh, blast radius, dan rencana rollback; di-commit hanya setelah pengakuan positif; diverifikasi setelah eksekusi untuk mengonfirmasi efek samping benar-benar terjadi. `interrupt()` LangGraph plus checkpointing PostgreSQL, `RequestInfoEvent` Microsoft Agent Framework, dan `waitForApproval()` Cloudflare semuanya mengimplementasikan bentuk yang sama. Mode kegagalan kanonisnya adalah persetujuan cap-stempel: "Setujui?" diklik tanpa reviu. Mitigasi yang terdokumentasi adalah challenge-and-response dengan checklist eksplisit.

**Type:** Learn
**Languages:** Python (stdlib, propose-then-commit state machine with idempotency)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 14 (Tripwires)
**Time:** ~60 minutes

## Masalahnya

Agen mengambil sebuah aksi. Pengguna harus memutuskan: setujui atau tidak. Jika keputusan instan, ia mungkin bukan reviu. Jika keputusan terstruktur, ia lambat tetapi bisa dipercaya. Pertanyaan engineering-nya adalah bagaimana membuat reviu terstruktur menjadi jalan yang paling mudah.

Pola HITL era 2023 adalah prompt sinkron: "Agen ingin mengirim email ke X dengan isi Y — setujui?" Pengguna mengklik Setujui. Semua orang merasa sistem aman. Dalam praktik permukaan ini banyak dicap-stempel: pengguna menyetujui cepat, persetujuan sedikit memprediksi, dan ketika agen salah, jejak audit menunjukkan riwayat panjang persetujuan yang tidak bisa diingat pengguna.

Pola 2026 — propose-then-commit — memindahkan HITL ke substrat durable, melampirkan metadata terstruktur, dan membutuhkan commit positif. Setiap SDK agen terkelola mengirim versinya: `interrupt()` LangGraph, `RequestInfoEvent` Microsoft Agent Framework, `waitForApproval()` Cloudflare. Nama API berbeda; bentuknya tidak.

## Konsepnya

### State machine propose-then-commit

1. **Usulkan.** Agen menghasilkan aksi yang diusulkan. Dipersisten ke penyimpanan durable (PostgreSQL, Redis, Durable Object). Mencakup:
   - intensi (mengapa agen melakukan ini)
   - lineage data (sumber apa yang mengarah ke usulan ini)
   - izin yang disentuh (scope / file / endpoint mana)
   - blast radius (apa kasus terburuknya)
   - rencana rollback (jika di-commit, bagaimana kita membatalkannya)
   - kunci idempotency (unik per usulan; pengiriman ulang mengembalikan catatan yang sama)
2. **Tampilkan.** Reviewer melihat usulan beserta semua metadata. Reviewer adalah orang (bukan agen yang mereviu dirinya sendiri).
3. **Commit.** Pengakuan positif. Aksi dieksekusi.
4. **Verifikasi.** Setelah eksekusi, efek samping dibaca kembali dan dikonfirmasi. Jika langkah verify gagal, sistem berada dalam status buruk yang diketahui dan alerting berjalan.

### Kunci idempotency

Tanpa kunci idempotency, retry setelah kegagalan transien bisa mengeksekusi ganda aksi yang disetujui. Contoh konkret: pengguna menyetujui "transfer $100 dari A ke B." Jaringan berkedip. Workflow retry. Pengguna menyetujui sekali tetapi transfer dieksekusi dua kali. Kunci idempotency mengikat persetujuan pada satu efek samping unik; eksekusi kedua menjadi no-op.

Ini pola idempotency yang sama yang dipakai API Stripe dan AWS. Memakainya ulang untuk persetujuan agen eksplisit dalam dok Microsoft Agent Framework.

### Durability: mengapa persetujuan lebih awet dari proses

Ruang tunggu persetujuan adalah potongan status yang tidak dimiliki agen. Workflow berhenti (Pelajaran 12). Ketika persetujuan tiba, workflow lanjut dari titik itu tepat. Itu sebabnya LangGraph memasangkan `interrupt()` dengan checkpointing PostgreSQL dan bukan sekadar status dalam-memori — persetujuan dua hari kemudian masih menemukan workflow utuh.

### Persetujuan cap-stempel dan mitigasi challenge-and-response

UI default untuk HITL (tombol "Setujui" / "Tolak") menghasilkan persetujuan cepat tanpa reviu sejati. Mitigasi yang terdokumentasi: checklist challenge-and-response yang membutuhkan jawaban positif atas pertanyaan spesifik sebelum tombol Setujui aktif. Bentuk konkret:

- "Apakah kamu paham resource apa yang disentuh ini? [ ]"
- "Sudahkah kamu memverifikasi blast radius bisa diterima? [ ]"
- "Apakah kamu punya rencana rollback jika ini gagal? [ ]"

Bukan birokrasi demi birokrasi — sebuah forcing function. Reviewer yang tidak bisa mencentang kotak meminta klarifikasi (eskalasi) atau menolak (default yang aman). Riset keamanan-agen Anthropic secara eksplisit mengutip HITL berbasis-checklist sebagai mitigasi pola persetujuan cap-stempel.

### Apa yang dihitung konsekuensial

Tidak setiap aksi butuh propose-then-commit. Panduan 2026:

- **Aksi konsekuensial** (selalu HITL): penulisan irreversibel, transaksi finansial, komunikasi keluar, perubahan database produksi, operasi filesystem destruktif.
- **Aksi reversibel** (kadang HITL): edit pada file lokal, perubahan env staging, penulisan reversibel dengan rollback yang jelas.
- **Pembacaan dan inspeksi** (tidak pernah HITL): membaca file, me-list resource, memanggil API read-only.

### Verifikasi pasca-aksi

"Commit berjalan" tidak sama dengan "efek samping terjadi." Partisijaringan dan kondisi balapan bisa menghasilkan workflow yang mengira berhasil sementara backend tidak mempersisten. Langkah verify membaca ulang resource target setelah commit untuk mengonfirmasi. Ini pola yang sama dengan transaksi database berklausa `RETURNING` atau `GetObject` AWS setelah `PutObject`.

### EU AI Act Pasal 14

Pasal 14 mewajibkan pengawasan manusia yang efektif untuk sistem AI berisiko-tinggi di EU. "Efektif" bukan hiasan. Bahasa regulasi secara spesifik mengecualikan pola cap-stempel. Propose-then-commit dengan challenge-and-response adalah bentuk yang selamat dari pengawasan Pasal 14 dalam dok kepatuhan Agent Governance Toolkit Microsoft.

```figure
mx-propose-then-commit
```

## Gunakan

`code/main.py` mengimplementasikan state machine propose-then-commit dalam Python stdlib. Penyimpanan durable adalah file JSON. Kunci idempotency adalah hash dari (thread_id, action_signature). Driver mensimulasikan tiga kasus: alur persetujuan bersih, retry setelah kegagalan transien (yang tidak boleh mengeksekusi ganda), dan default cap-stempel versus alur challenge-and-response.

## Rilis

`outputs/skill-hitl-design.md` mereviu alur kerja HITL yang diusulkan untuk bentuk propose-then-commit dan menandai metadata, idempotency, verifikasi, atau lapisan challenge-and-response yang hilang.

## Latihan

1. Jalankan `code/main.py`. Konfirmasi bahwa retry atas usulan yang disetujui memakai catatan durable dan tidak mengeksekusi ulang. Kini ubah kunci idempotency agar mencakup timestamp dan tunjukkan retry mengeksekusi ganda.

2. Perluas catatan usulan dengan field `rollback`. Simulasikan eksekusi yang langkah verify-nya gagal. Tunjukkan rollback terpicu otomatis.

3. Baca dok `RequestInfoEvent` Microsoft Agent Framework. Identifikasi satu field metadata yang disertakan API tetapi hilang dari mesin mainan. Tambahkan dan jelaskan apa yang ia lindungi.

4. Rancang checklist challenge-and-response untuk aksi spesifik (mis. "posting ke akun Twitter publik"). Tiga pertanyaan apa yang harus dijawab reviewer? Mengapa tiga itu?

5. Pilih satu kasus tempat prompt "Setujui?" sinkron sudah cukup (tanpa penyimpanan durable). Jelaskan alasannya, dan sebutkan kelas risiko yang kamu terima.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Propose-then-commit | "Two-phase approval" | Persisted proposal + positive commit + verify |
| Idempotency key | "Retry-safe token" | Unique per proposal; second execution no-ops |
| Data lineage | "Where it came from" | The specific source content that led to the proposal |
| Blast radius | "Worst case" | Scope of effect if the action goes wrong |
| Rubber-stamp | "Fast approval" | "Approve" clicked without genuine review |
| Challenge-and-response | "Forcing checklist" | Reviewer must positively acknowledge specific questions |
| RequestInfoEvent | "MS Agent Framework primitive" | Durable HITL request with structured metadata |
| `interrupt()` / `waitForApproval()` | "Framework primitives" | LangGraph / Cloudflare equivalents of the same shape |

## Further Reading

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`, durable approvals.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` and Durable Objects.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL as a mitigation for long-horizon risk.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — regulatory baseline for high-risk systems.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — constitutional framing around oversight.
