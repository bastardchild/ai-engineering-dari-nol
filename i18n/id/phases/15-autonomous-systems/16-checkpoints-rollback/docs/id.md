# Checkpoint dan Rollback

> Setiap transisi status-graf dipersisten. Ketika worker crash, lease-nya kedaluwarsa dan worker lain mengambil alih pada checkpoint terbaru. Cloudflare Durable Objects memegang status lintas jam atau minggu. Propose-then-commit (Pelajaran 15) mendefinisikan rencana rollback per aksi. Verifikasi pasca-aksi menutup loop. EU AI Act Pasal 14 membuat pengawasan manusia yang efektif wajib untuk sistem berisiko-tinggi — dalam praktik ini berarti checkpoint harus bisa di-query, rollback harus digladi, dan jejak audit harus selamat dari deploy. Mode kegagalan yang tajam: tanpa kunci idempotency dan cek prasyarat, retry setelah kegagalan transien bisa mengeksekusi ganda aksi yang sudah disetujui. Verifikasi pasca-aksi inilah yang menangkapnya.

**Type:** Learn
**Languages:** Python (stdlib, checkpoint and rollback state machine)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 15 (Propose-then-commit)
**Time:** ~60 minutes

## Masalahnya

Eksekusi durable (Pelajaran 12) membuat agen yang crash bisa dilanjutkan. Propose-then-commit (Pelajaran 15) membuat aksi yang disetujui bisa diaudit. Pelajaran ini menggabungkannya: apa yang terjadi ketika aksi yang disetujui dieksekusi sebagian, crash, dan lanjut? Kapan rollback berjalan, dan terhadap status apa?

Sistem nyata merangkainya berbeda:

- **LangGraph** men-checkpoint setiap transisi status-graf ke PostgreSQL. Saat worker crash, lease dilepas dan worker lain lanjut pada checkpoint terbaru. Workflow berhenti pada `interrupt()`, yang sendiri dipersisten.
- **Cloudflare Durable Objects** memegang status per kunci lintas jam atau minggu. Co-locate komputasi dengan penyimpanan untuk aksi yang disetujui.
- **Microsoft Agent Framework** mengekspos primitif `Checkpoint` dalam API workflow; replay plus idempotency mencakup retry.

Dalam setiap kasus, kombinasi yang benar-benar berhasil adalah: kunci idempotency (mencegah eksekusi-ganda) + cek prasyarat (status masih seperti yang disetujui) + verify pasca-aksi (efek samping benar-benar terjadi) + rollback saat verify-gagal.

## Konsepnya

### Setiap transisi dipersisten

Transisi status-graf adalah langkah apa pun yang memindahkan workflow dari satu status bernama ke status lain. Implementasi naif hanya mempersisten pada titik commit spesifik; implementasi produksi mempersisten setiap transisi. Biayanya (beberapa penulisan ekstra) kecil relatif terhadap gain reliabilitas (replay mendarat di mana saja, pemulihan lease presisi).

### Pemulihan lease

Ketika worker crash, workflow tidak hilang; lease (klaim berumur-pendek bahwa worker ini mengeksekusi proses ini) sekadar kedaluwarsa. Worker lain mengambil checkpoint terbaru dan lanjut. Mekanisme lease inilah yang membiarkan sistem produksi selamat dari rolling deploy tanpa kehilangan pekerjaan dalam-penerbangan.

### Idempotency plus prasyarat

Idempotency saja tidak cukup. Bayangkan: workflow disetujui untuk "transfer $100 dari A ke B ketika saldo > $1000." Workflow di-commit, crash di tengah eksekusi, dan lanjut. Jika hanya kunci idempotency dicek, dan eksekusi lanjut, transfer berjalan sekali (benar). Tetapi bayangkan antara crash dan lanjut, saldo A turun menjadi $500 lewat workflow berbeda. Cek idempotency masih lolos; prasyarat tidak. Tanpa cek prasyarat, kita mengirim overdraft.

Setiap aksi konsekuensial butuh keduanya:

- **Kunci idempotency**: mencegah eksekusi-ganda.
- **Cek prasyarat**: mengonfirmasi status masih konsisten dengan yang disetujui.

### Verifikasi pasca-aksi

"Tool mengembalikan 200" bukan verifikasi. Verifikasi nyata membaca ulang status target dan mengonfirmasi efek samping benar-benar terjadi. Polanya:

- Pembaruan database: `UPDATE ... RETURNING *` lalu assert baris yang dikembalikan cocok dengan status yang dimaksud.
- Pengiriman email: cek folder terkirim untuk ID pesan setelah submisi.
- Penulisan file: baca file kembali dan hash.
- Panggilan API: `GET` lanjutan pada resource target.

Jika verify gagal, workflow berada dalam status-buruk yang diketahui. Rollback berjalan.

### Rencana rollback

Setiap aksi konsekuensial dalam propose-then-commit (Pelajaran 15) membawa rencana rollback. Jenisnya:

- **Rollback in-band**: balikkan efek samping secara langsung (`DELETE` setelah `INSERT`, `Send-correction-email` setelah kirim).
- **Transaksi kompensasi**: aksi baru yang menetralkan yang asli (pola SAGA standar).
- **Rollback out-of-band**: alert manusia, hentikan workflow, tinggalkan status buruk untuk investigasi.

Rollback no-op ("kita tidak bisa membatalkan ini") harus dinamai dalam usulan. Aksi tanpa rollback membutuhkan HITL yang lebih kuat pada waktu commit (challenge-and-response Pelajaran 15).

### Pembacaan operasional EU AI Act Pasal 14

Pasal 14 mensyaratkan "pengawasan manusia yang efektif" untuk sistem berisiko-tinggi. Dalam istilah operasional, implementor membacanya sebagai:

- Checkpoint bisa di-query auditor.
- Rollback digladi (diuji ujung-ke-ujung setidaknya sekali).
- Jejak audit selamat dari deploy (backend checkpoint bukan efemeral).
- Verifikasi yang gagal di-alert, bukan diam-diam di-log.

Workflow yang crash di tengah-commit, lanjut, dan menyelesaikan efek samping tanpa jalur verify + rollback tidak selamat dari pengujian Pasal 14.

### Mode kegagalan yang tajam: eksekusi-ganda

Insiden produksi paling umum di ruang ini:

1. Aksi disetujui, kunci idempotency k.
2. Commit mulai, dieksekusi, mengembalikan 200.
3. Workflow crash sebelum mempersisten status "committed".
4. Workflow lanjut; melihat "disetujui tetapi belum di-commit"; mengeksekusi ulang.
5. Efek samping terpicu dua kali.

Mitigasi: persisten intensi "in-flight" sebelum eksekusi, eksekusi dengan kunci idempotency, lalu tandai "committed" hanya setelah verifikasi pasca-aksi berhasil. Jika aksi terpicu dan penulisan status gagal, kamu tahu harus memverifikasi dan (bila perlu) memicu ulang. Jika penulisan status berhasil dan aksi gagal, kamu memverifikasi dan memicu tepat sekali lewat jalur pemulihan.

```figure
checkpoint-replay
```

## Gunakan

`code/main.py` mengimplementasikan workflow yang di-checkpoint dengan idempotency, prasyarat, verify, dan rollback. Driver mensimulasikan empat skenario: proses bersih, retry setelah crash (idempotency menangkap), prasyarat gagal (workflow batal tanpa terpicu), verify gagal (rollback terpicu).

## Rilis

`outputs/skill-rollback-rehearsal.md` merancang pengujian gladi-rollback untuk workflow yang diusulkan dan mengaudit backend checkpoint untuk persistensi jejak-audit.

## Latihan

1. Jalankan `code/main.py`. Verifikasi keempat skenario. Untuk kasus crash-saat-commit, konfirmasi aksi terpicu tepat sekali lintas retry.

2. Modifikasi pola "tandai selesai dulu, lalu kerjakan" sehingga penulisan status terpicu setelah aksi. Jalankan ulang skenario crash. Ukur berapa aksi duplikat terpicu.

3. Rancang rencana rollback untuk aksi produksi spesifik (mis. "posting ke kanal Slack"). Klasifikasikan sebagai in-band, kompensasi, atau out-of-band. Justifikasi pilihannya.

4. Ambil satu workflow yang kamu kenal. Identifikasi setiap transisi status. Tandai masing-masing dengan syarat durability (persisten / jangan persisten). Hitung yang saat ini tidak kamu persisten.

5. Pengujian rollback yang digladi: rancang pengujian ujung-ke-ujung yang menjalankan workflow nyata, meng-crash-nya, dan mengonfirmasi jalur rollback terpicu. Apa yang di-assert pengujian?

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Checkpoint | "Save point" | Every graph-state transition persists to a durable store |
| Lease | "Worker claim" | Short-lived claim that a worker is executing a run; expires on crash |
| Precondition | "State gate" | Assertion that the state is still consistent with the approved action |
| Post-action verify | "Re-read check" | Confirm the side effect actually happened in the target system |
| In-band rollback | "Direct undo" | Reverse the side effect with the inverse operation |
| Compensating transaction | "SAGA undo" | A new action that neutralizes the original |
| Mark-as-done-first | "Status write order" | Persist the committed status before returning from commit |
| Article 14 | "EU AI Act human oversight" | Operational: queryable checkpoints, rehearsed rollbacks, auditable trail |

## Further Reading

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — checkpoint primitives and lease recovery.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Durable Objects as a state substrate.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — regulatory baseline.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — reliability framing for long-horizon workflows.
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — workflow shape for Claude Code Routines.
