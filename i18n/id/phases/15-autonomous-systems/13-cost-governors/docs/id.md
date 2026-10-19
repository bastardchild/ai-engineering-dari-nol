# Anggaran Aksi, Batas Iterasi, dan Governor Biaya

> Biaya LLM bulanan sebuah agen e-commerce menengah melonjak dari $1.200 menjadi $4.800 setelah timnya menyalakan skill "order-tracking". Itu bukan bug harga. Itu agen yang menemukan loop baru dan terus berbelanja di dalamnya. Agent Governance Toolkit dari Microsoft (2 April 2026) mengkodifikasi pertahanan terhadap kelas ini: `max_tokens` per permintaan, anggaran token dan dolar per tugas, batas per hari/bulan, batas iterasi, routing model bertingkat, prompt caching, context windowing, checkpoint HITL pada aksi mahal, kill switch saat anggaran jebol. Claude Code Agent SDK dari Anthropic mengirim primitif yang sama dengan nama berbeda. Batas kecepatan finansial — mis. putus akses pada >$50 dalam 10 menit — menangkap loop lebih cepat dibanding batas bulanan.

**Type:** Learn
**Languages:** Python (stdlib, layered cost-governor simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**Time:** ~60 minutes

## Masalahnya

Agen otonom membelanjakan uang nyata pada setiap giliran. Keluaran buruk chatbot adalah balasan yang buruk; loop buruk agen adalah tagihan. Istilah yang didokumentasikan industri untuk mode kegagalan ini adalah "Denial of Wallet" — agen terus bernalar, terus memanggil tool, terus menagih, dan tidak ada yang menghentikannya karena tidak ada yang dirancang untuk itu.

Perbaikannya bukan satu angka. Ia tumpukan batas pada skala waktu dan granularitas berbeda: per permintaan, per tugas, per jam, per hari, per bulan. Tumpukan yang dirancang baik menangkap loop yang lepas dalam hitungan menit, kebocoran lambat dalam hitungan jam, dan rilis yang buruk dalam sehari. Tumpukan yang sama menjaga anggaran tetap ada ketika agen horizon-panjang dan otonom.

Ini pelajaran engineering: matematikanya trivial, disiplinnya tempat tim gagal. Daftar batas di bawah semuanya dinamai baik dalam dok Agent Governance Toolkit Microsoft maupun dok Claude Code Agent SDK Anthropic.

## Konsepnya

### Tumpukan governor-biaya

1. **`max_tokens` per permintaan.** Sederhana. Mencegah satu panggilan memancarkan completion yang tak terbatas.
2. **Anggaran token per tugas.** Di seluruh proses, jangan lewati N token. Berhenti keras pada batas.
3. **Anggaran dolar per tugas.** Sama dengan token tetapi dalam mata uang. `max_budget_usd` dalam Claude Code.
4. **Batas panggilan per tool.** Tidak lebih dari N panggilan `WebFetch`, N panggilan `shell_exec`, dst.
5. **Batas iterasi (`max_turns`).** Total iterasi loop agen; mencegah loop penalaran tak berujung.
6. **Batas per menit / per jam / per hari / per bulan.** Jendela bergulir. Menangkap kebocoran pada skala waktu berbeda.
7. **Batas kecepatan finansial.** Mis., "jika belanja melewati $50 dalam 10 menit, putus akses." Menangkap burn berbasis-loop sebelum batas bulanan terpicu.
8. **Routing model bertingkat.** Default ke model yang lebih kecil; eskalasi ke yang lebih besar hanya ketika classifier menilai tugas layak.
9. **Prompt caching.** System prompt dan konteks stabil disimpan dalam cache penyedia; biaya token pengiriman ulang nyaris nol.
10. **Context windowing.** Kompaksi / rangkuman untuk menjaga konteks aktif di bawah ambang; pengurangan biaya-token langsung.
11. **Checkpoint HITL pada aksi mahal.** Sebelum aksi yang diketahui mahal (panggilan tool panjang, unduhan besar, upgrade model yang mahal), butuhkan ketukan manusia.
12. **Kill switch saat anggaran jebol.** Sesi dibatalkan ketika batas mana pun terpicu. Batas dicatat; membutuhkan jalur re-enable terpisah.

### Mengapa tumpukan, bukan satu batas

Satu batas bulanan menangkap agen yang lepas hanya setelah dompet habis. Satu batas per permintaan tidak menangkap apa pun pada level sesi. Mode kegagalan berbeda membutuhkan skala waktu berbeda:

- **Loop lepas** (agen macet dalam retry 5 detik): ditangkap batas kecepatan.
- **Kebocoran lambat** (agen mengerjakan ~2x kerja yang diharapkan per tugas): ditangkap batas harian.
- **Rilis buruk** (versi baru memakai 5x token): ditangkap batas mingguan / bulanan.
- **Lonjakan sah** (permintaan nyata, bukan bug): ditangkap batas jam / hari dengan log yang jelas.

### Permukaan anggaran harness

Claude Code Agent SDK mengekspos (dok publik):

- `max_turns` — batas iterasi.
- `max_budget_usd` — batas dolar; sesi dibatalkan saat jebol.
- `allowed_tools` / `disallowed_tools` — allowlist dan denylist tool.
- Titik hook sebelum pemakaian tool untuk akuntansi-biaya kustom.

Gabungkan dengan tangga mode-izin (Pelajaran 10). Sesi `autoMode` tanpa `max_budget_usd` adalah otonomi tanpa governor. Anthropic secara eksplisit membingkai Auto Mode sebagai membutuhkan kontrol anggaran; classifier ortogonal terhadap biaya.

### EU AI Act, OWASP Agentic Top 10

Agent Governance Toolkit Microsoft mencakup OWASP Agentic Top 10 dan syarat EU AI Act Pasal 14 (pengawasan manusia). Untuk produksi di EU, logging dan penegakan batas bukan opsional.

### Kasus $1.200 → $4.800 yang teramati

Kasus nyata dalam dok Microsoft: agen e-commerce yang biaya bulanannya tiga kali lipat setelah tool baru ditambahkan. Tool itu membiarkan agen mem-poll status pesanan selama setiap sesi. Tanpa deteksi loop. Tanpa batas per tool. Tanpa alert atas pertumbuhan minggu-ke-minggu. Perbaikannya adalah batas per tool plus alert pertumbuhan-harian. Ini templat: setiap permukaan tool baru adalah loop potensial baru; setiap tool baru butuh batasnya sendiri dan alert-nya sendiri.

```figure
cost-governor-stack
```

## Gunakan

`code/main.py` mensimulasikan proses agen dengan dan tanpa tumpukan governor-biaya berlapis. Agen simulasi melenceng ke loop polling setelah beberapa giliran; tumpukan berlapis menangkapnya dalam jendela kecepatan sementara satu batas bulanan tidak akan terpicu sampai berhari-hari kemudian.

## Rilis

`outputs/skill-agent-budget-audit.md` mengaudit tumpukan governor-biaya deployment agen yang diusulkan dan menandai lapisan yang hilang.

## Latihan

1. Jalankan `code/main.py`. Konfirmasi batas kecepatan terpicu sebelum batas iterasi pada trajektori loop-polling. Kini matikan batas kecepatan dan ukur berapa yang "dibelanjakan" agen sebelum batas iterasi menangkapnya.

2. Rancang set batas per tool untuk agen browser (Pelajaran 11). Tool mana yang butuh batas paling ketat? Tool mana yang bisa berjalan tak terbatas tanpa risiko?

3. Baca dok Agent Governance Toolkit Microsoft. Daftarkan setiap jenis batas yang dinamai toolkit. Petakan masing-masing ke satu mode kegagalan (loop lepas, kebocoran lambat, rilis buruk, lonjakan).

4. Tetapkan harga proses semalam tanpa pengawasan untuk tugas realistis (mis. "triage 50 isu dalam repo"). Atur `max_budget_usd` pada 2x estimasi titikmu. Justifikasi 2x itu.

5. `max_budget_usd` Claude Code terpicu pada biaya agregat sesi. Rancang batas kecepatan komplementer yang akan kamu tegakkan secara eksternal. Apa yang memicu pemutusan, dan seperti apa re-enable itu?

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Denial of Wallet | "Runaway bill" | Agent loop generating spend with no cap to stop it |
| max_tokens | "Per-request cap" | Ceiling on a single completion's size |
| max_turns | "Iteration cap" | Ceiling on agent loop iterations in a session |
| max_budget_usd | "Dollar kill switch" | Session cost cap; aborts on breach |
| Velocity limit | "Rate cap" | Limit on spend per short window (e.g., $50 / 10 min) |
| Tiered routing | "Small model first" | Cheap model default; escalate only when classifier warrants |
| Prompt caching | "Cached system prompt" | Provider-side cache reduces re-send token cost to near zero |
| HITL checkpoint | "Human approval gate" | Human tap required before expensive action |

## Further Reading

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`, `max_budget_usd`, tool allowlists.
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — cost-governor checkpoints.
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — provider-side cost controls.
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — caching mechanics.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — cost profile for long-horizon agents.
