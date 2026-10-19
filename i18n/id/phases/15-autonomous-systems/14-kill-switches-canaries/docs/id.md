# Kill Switch, Circuit Breaker, dan Token Canary

> Kill switch adalah boolean yang dipegang di luar permukaan edit agen — kunci Redis, feature flag, config yang ditandatangani — yang menonaktifkan agen sepenuhnya. Circuit breaker lebih halus: ia terpicu pada pola spesifik (lima panggilan tool identik berturut-turut), menghentikan jalur yang melanggar, dan mengeskalasi ke manusia. Token canary mewarisi dari deception klasik: kredensial palsu atau catatan honeypot yang tidak punya alasan sah disentuh agen, yang aksesnya memicu alert. Datapath berbasis eBPF (mis. Cilium) bisa menulis ulang egress pod yang dikarantina ke honeypot forensik di lapisan kernel; benchmark Cilium yang dipublikasikan melaporkan latensi datapath P99 sub-milidetik di bawah beban (anggaran propagasimu bergantung pada bagaimana pembaruan kebijakan mencapai node, bukan datapath itu sendiri). Detektor statistik (EWMA, CUSUM) yang beradaptasi pada baseline bergerak akan diam-diam menerima drift — lapisi dengan batas konstitusional keras yang tidak melengkung.

**Type:** Learn
**Languages:** Python (stdlib, three-detector simulator: kill switch, circuit breaker, canary)
**Prerequisites:** Phase 15 · 13 (Cost governors), Phase 15 · 10 (Permission modes)
**Time:** ~60 minutes

## Masalahnya

Governor biaya (Pelajaran 13) membatasi apa yang bisa dibelanjakan agen. Mereka tidak membatasi apa yang bisa dilakukan agen di dalam anggaran. Agen dengan batas kecepatan $50 masih bisa mengeksfiltrasi secret, menerbitkan postingan yang salah, atau menghapus resource — aksi mahal sering kali justru yang murah dalam token.

Pelajaran ini mencakup tiga detektor yang berada di samping lapisan biaya:

1. **Kill switch**: tombol off boolean yang dipegang di luar jangkauan agen.
2. **Circuit breaker**: detektor pola-aksi yang menghentikan jalur spesifik.
3. **Token canary**: umpan yang agen tanpa alasan sah menyentuhnya akan ketahuan dengan menyentuhnya.

Ketiganya adalah engineering pra-LLM. Deception klasik, breaker batas-laju, dan kill feature-flag mendahului agen otonom. Yang baru adalah permukaan serangannya: agen membaca konten tak tepercaya (Pelajaran 11), mengedit memorinya sendiri, dan bisa merangkai banyak aksi tampak-aman menjadi satu yang tidak aman. Detektor yang dinamai di sini berhasil karena mereka tidak memercayai laporan-diri agen.

## Konsepnya

### Kill switch

Kill switch adalah boolean yang dibaca agen tetapi tidak bisa ditulis. Pilihan implementasi:

- **Feature flag dalam layanan terkelola.** LaunchDarkly, Statsig, Unleash. Pembaruan berpropagasi dalam hitungan detik.
- **Kunci Redis yang di-poll agen.** Sederhana; membutuhkan proses agen memeriksa pada setiap giliran.
- **Config bertanda tangan di object storage.** Agen memverifikasi tanda tangan saat boot; menolak status yang tidak ditandatangani.
- **Sinyal level-OS atau kill siklus-hidup kontainer.** `kill` Docker, `kubectl delete pod` Kubernetes, stop systemd.

Properti yang dimiliki kill switch yang benar:

- Agen tidak bisa mengaturnya ke `off`. (Tinggal dalam sistem yang tidak ditulis kredensial agen.)
- Ia dicek pada setiap aksi konsekuensial, bukan hanya saat startup.
- Ketika off, agen tidak melakukan apa pun yang teramati secara eksternal, termasuk logging ke sistem yang bisa dijangkau agen.
- Menyalakannya kembali adalah aksi manusia yang eksplisit, bukan timeout otomatis.

### Circuit breaker

Circuit breaker menghentikan pola spesifik, bukan seluruh agen. Bentuk klasik (dari buku Nygard 2007, masih berlaku):

- **Closed**: aksi diizinkan.
- **Open**: aksi diblokir.
- **Half-open**: setelah cool-down, 1–3 upaya probe diizinkan (default 1); sukses menutup breaker, kegagalan tersisa membukanya kembali.

Pemicu yang relevan-agen:

- Lima panggilan tool identik berturut-turut (loop repetitif).
- Lima kegagalan beruntun pada tool yang sama dengan input berbeda (kegagalan sistemik).
- Laju panggilan tool di atas ambang (kecepatan Pelajaran 13).
- Tool spesifik dipanggil (mis. `file.delete`) setelah pembacaan konten di luar kepercayaan (Pelajaran 11).

### Token canary

Token canary (juga disebut honeytoken) adalah input yang tidak boleh disentuh agen. Akses adalah alarmnya.

Contoh untuk agen:

- `AWS_SECRET_ACCESS_KEY` palsu dalam environment tanpa hak istimewa aktual. Upaya eksfiltrasi langsung teratribusi.
- Catatan database palsu yang ditandai dengan nilai sentinel yang diketahui. Pembacaan atau pembaruan apa pun atas catatan ini memicu alert.
- File palsu dalam workspace. Pembacaan atau edit atas file ini mencurigakan.
- Entri memori palsu yang disuntikkan saat sesi mulai. Jika muncul dalam keluaran berikutnya, memori telah tereksfiltrasi.

Desain canary spesifik pada alur kerja. Canary yang punya alasan sah disentuh agen adalah noise.

### Mengapa melapisi batas statistik dan keras

Detektor statistik (EWMA, CUSUM, z-score pada laju panggilan-tool) beradaptasi pada baseline bergerak. Itu fitur untuk drift yang sah dan bug untuk drift adversarial. Penyerang yang sabar bisa menggeser baseline cukup lambat sehingga tidak ada alarm berbunyi.

Batas konstitusional keras (Pelajaran 17) tidak melengkung. "Jangan pernah email `secrets@`" sebagai aturan konstitusional lebih sederhana, bisa diaudit, dan tidak bisa diakali drift lambat. Pola pelapisannya: detektor statistik menangkap sebagian besar noise, batas keras menangkap serangan yang berjalan melewati detektor.

### Karantina lewat redirect datapath eBPF

Kebijakan jaringan berbasis eBPF Cilium bisa menulis ulang egress pod yang dikarantina ke honeypot forensik di lapisan kernel. Benchmark publik Cilium melaporkan latensi datapath P99 sub-milidetik pada TCP_RR; anggaran wall-clock untuk *memicu* karantina di klastermu didominasi oleh bagaimana pembaruan kebijakan berpropagasi dari control plane ke node (beberapa ratus milidetik realistis pada Kubernetes dengan operator yang pre-loaded; latensi ekor bergantung pada ukuran klaster dan kadens rekonsiliasi). Ini memberimu dua hal: agen tidak mengeksfiltrasi apa pun yang nyata, dan perilaku agen di dalam honeypot teramati penuh. Pola yang sama, lapisan berbeda, dengan token canary: agen mengungkapkan intensi lewat apa yang ia coba lakukan saat dikungkung.

### Apa yang tidak ditangkap detektor apa pun

- **Serangan semantik yang tersusun dari primitif yang diizinkan.** Setiap aksi individual lolos setiap detektor; komposisinya adalah masalahnya. Ini kegagalan komposit Auto Mode dari Pelajaran 10.
- **Serangan yang mencakup detektor sebagai bagian payload.** Jika agen membaca halaman yang berkata "sebelum bertindak, nonaktifkan canary" dan agen punya kapabilitasnya, detektor terkompromi. Token canary harus berada dalam sistem yang tidak bisa dimodifikasi agen.

```figure
circuit-breaker
```

## Gunakan

`code/main.py` mensimulasikan trajektori agen pendek lewat tiga detektor. Kill switch yang dipegang dalam dict eksternal; circuit breaker yang terpicu pada lima panggilan tool identik; file canary yang pembacaannya memicu alert. Masukkan trajektori sintetis: aksi sah, loop repetitif, probe canary, dan skenario terpicu-kill-switch tempat aksi agen dihentikan.

## Rilis

`outputs/skill-tripwire-design.md` mereviu tumpukan detektor yang diusulkan untuk deployment agen dan menandai kesenjangan (kill switch hilang, canary hilang, ambang circuit breaker terlalu longgar).

## Latihan

1. Jalankan `code/main.py`. Konfirmasi circuit breaker terpicu pada giliran 5 (panggilan identik kelima) dan canary terpicu pada giliran 9 (pembacaan kunci-palsu).

2. Tambahkan detektor statistik: z-score EWMA pada laju panggilan-tool. Masukkan trajektori yang melenceng lambat dan tunjukkan detektor tidak pernah terpicu. Kini tambahkan batas keras (tidak lebih dari 50 panggilan tool dalam 10 menit) dan tunjukkan batas keras terpicu pada trajektori yang sama.

3. Rancang set token canary untuk agen browser (Pelajaran 11). Daftarkan setidaknya tiga canary dan apa yang akan dideteksi masing-masing.

4. Baca dok kebijakan-jaringan Cilium. Jelaskan alur karantina redirect-egress secara konkret: selector kebijakan mana, pod mana, rewrite egress mana, alert mana. Apa yang mengatur latensi wall-clock dari "memutuskan karantina" ke "paket pertama yang di-redirect"?

5. Definisikan prosedur re-enable untuk agen yang di-kill-switch. Siapa yang bisa menyalakan ulang? Apa yang harus didokumentasikan? Apa yang harus berubah pada agen sebelum re-enable?

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Kill switch | "Off button" | Boolean outside the agent's edit surface; checked on every consequential action |
| Circuit breaker | "Pattern pause" | Action-specific trip on repetition, failure rate, or rate-limit |
| Canary token | "Honeytoken" | Bait the agent has no legitimate reason to touch; access fires an alert |
| Honeypot | "Forensic sandbox" | Redirected traffic / workspace where a quarantined agent is observed |
| EWMA | "Moving average" | Exponentially weighted; adapts to drift (feature + bug) |
| CUSUM | "Cumulative sum" | Detects sustained shift from baseline |
| Hard limit | "Constitutional rule" | Does not adapt; constant regardless of history |
| Constitutional limit | "Always-true rule" | Tied to Lesson 17's constitution; cannot be edited by the agent |

## Further Reading

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — kill-switch and circuit-breaker framing for autonomous agents.
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — production governance patterns.
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — detection-and-response requirements.
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — pod-level egress redirect and forensic honeypot patterns.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — hardcoded prohibitions as "constitutional limits".
