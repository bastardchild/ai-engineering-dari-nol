# Agen Browser dan Tugas Web Horizon-Panjang

> Agen ChatGPT (Juli 2025) menggabungkan Operator dan deep research menjadi satu agen browser/terminal dan mencetak SOTA BrowseComp pada 68,9%. OpenAI menutup Operator pada 31 Agustus 2025 — konsolidasi di lapisan produk. Akuisisi Vercept oleh Anthropic memindahkan Claude Sonnet pada OSWorld dari di bawah 15% menjadi 72,5%. WebArena-Verified (ServiceNow, ICLR 2026) memperbaiki 11,3 poin persentase false-negative rate pada WebArena asli dan mengirim subset Hard berisi 258 tugas. Angka-angkanya nyata. Begitu pula permukaan serangannya: kepala preparedness OpenAI menyatakan secara publik bahwa injeksi prompt tidak langsung ke agen browser "bukan bug yang bisa dipatch penuh." Serangan 2025–2026 yang terdokumentasi: Tainted Memories (CSRF Atlas), HashJack (Cato Networks), dan hijack sekali-klik di Perplexity Comet.

**Type:** Learn
**Languages:** Python (stdlib, indirect prompt-injection attack surface model)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## Masalahnya

Agen browser adalah agen horizon-panjang yang membaca konten tak tepercaya dan mengambil aksi konsekuensial. Setiap halaman yang dikunjungi agen adalah input yang tidak ditulis pengguna. Setiap formulir pada setiap halaman adalah kanal perintah potensial. Korpus serangan 2025–2026 menunjukkan ini bukan hipotetis: Tainted Memories membiarkan penyerang mengikat instruksi jahat ke memori agen lewat halaman yang dibuat; HashJack menyembunyikan perintah dalam fragmen URL yang dikunjungi agen; hijack Comet Perplexity terjadi dalam sekali klik.

Gambaran defensifnya tidak nyaman. Kepala preparedness OpenAI mengatakan bagian yang sunyi dengan lantang: injeksi prompt tidak langsung "bukan bug yang bisa dipatch penuh." Ini karena serangan tinggal di batas baca-vs-bertindak agen, yang secara arsitektural kabur — setiap token yang dibaca model pada prinsipnya bisa dibaca sebagai instruksi.

Pelajaran ini menamai permukaan serangan, menamai lanskap benchmark (BrowseComp, OSWorld, WebArena-Verified), dan memodelkan skenario injeksi-prompt-tidak-langsung minimal sehingga kamu bisa menalar pertahanan nyata dalam Pelajaran 14 dan 18.

## Konsepnya

### Lanskap 2026, satu paragraf per sistem

**Agen ChatGPT (OpenAI).** Diluncurkan Juli 2025. Menyatukan Operator (browsing) dan Deep Research (riset multi-jam). Menutup Operator mandiri pada 31 Agustus 2025. SOTA pada BrowseComp pada 68,9%; angka kuat pada OSWorld dan WebArena-Verified.

**Claude Sonnet + Vercept (Anthropic).** Akuisisi Vercept oleh Anthropic berfokus pada kapabilitas computer-use. Memindahkan Claude Sonnet pada OSWorld dari <15% menjadi 72,5%. Claude Computer Use dikirim sebagai tool API.

**Gemini 3 Pro dengan Browser Use (DeepMind).** Integrasi Browser Use mengirim kontrol computer-use; FSF v3 (April 2026, Pelajaran 20) melacak otonomi dalam domain ML R&D secara spesifik.

**WebArena-Verified (ServiceNow, ICLR 2026).** Memperbaiki soal yang terdokumentasi baik: WebArena asli punya ~11,3% false-negative rate (tugas yang ditandai gagal padahal sebenarnya solved). Rilis Verified menilai ulang dengan kriteria keberhasilan kurasi-manusia dan menambahkan subset Hard berisi 258 tugas (makalah ICLR 2026, openreview.net/forum?id=94tlGxmqkN).

### BrowseComp vs OSWorld vs WebArena

| Benchmark | What it measures | Horizon |
|---|---|---|
| BrowseComp | Finding specific facts on the open web under time pressure | minutes |
| OSWorld | Agent operating a full desktop (mouse, keyboard, shell) | tens of minutes |
| WebArena-Verified | Transactional web tasks in simulated sites | minutes |
| Hard subset | WebArena-Verified tasks with multi-page state transitions | tens of minutes |

Sumbu berbeda. Skor BrowseComp tinggi berkata agen menemukan fakta; ia tidak berkata agen bisa memesan penerbangan. Skor OSWorld lebih dekat ke "apakah ia bekerja di desktop-ku." WebArena-Verified lebih dekat ke "bisakah ia menyelesaikan alur." Keputusan produksi apa pun membutuhkan benchmark yang cocok dengan distribusi tugas.

### Permukaan serangan, dinamai

1. **Injeksi prompt tidak langsung.** Konten halaman tak tepercaya berisi instruksi. Agen membacanya. Agen mengeksekusinya. Contoh publik: 2024 Kai Greshake et al., makalah Tainted Memories 2025, HashJack 2026 (Cato Networks).
2. **Injeksi fragmen / query URL.** `#fragment` atau query string dari URL yang di-crawl berisi perintah. Tidak pernah dirender terlihat; tetap di dalam konteks agen.
3. **Serangan pengikat-memori.** Halaman memerintah agen menulis memori persisten (Pelajaran 12 mencakup state durable). Sesi berikutnya, memori memicu payload tanpa pemicu yang terlihat.
4. **Serangan berbentuk-CSRF pada sesi terautentikasi.** Kelas Tainted Memories: agen sedang login di suatu tempat; halaman penyerang menerbitkan permintaan pengubah-status yang dieksekusi agen dengan cookie pengguna.
5. **Hijack sekali-klik.** Tombol yang tampak tidak berbahaya menunggangi payload yang diikuti agen. Kelas Comet.
6. **Lubang Content-Security-Policy pada permukaan host agen.** Lapisan rendering dan tool sendiri bisa menjadi vektor serangan; stack browser-dalam-browser-agen lebar.

### Mengapa "tidak bisa dipatch penuh"

Serangan isomorfik dengan kapabilitas agen. Agen harus membaca konten tak tepercaya untuk mengerjakan tugasnya. Konten apa pun yang dibaca agen bisa berisi instruksi. Instruksi apa pun yang diikuti agen bisa misaligned dengan permintaan aktual pengguna. Pertahanan (batas kepercayaan, classifier, allowlist tool, HITL pada aksi konsekuensial) menaikkan biaya serangan dan mengurangi blast radius-nya. Mereka tidak menutup kelasnya.

Ini pola penalaran yang sama dengan teorema Lob (Pelajaran 8): agen tidak bisa membuktikan token berikutnya aman; ia hanya bisa menyiapkan sistem tempat token tidak aman lebih terdeteksi.

### Postur pertahanan yang benar-benar dikirim

- **Batas baca / tulis.** Membaca tidak pernah konsekuensial. Menulis (mengirim formulir, memposting konten, memanggil tool dengan efek samping) membutuhkan persetujuan manusia yang segar jika konten pemicunya berasal dari luar batas kepercayaan.
- **Allowlist tool per tugas.** Agen bisa browsing; ia tidak bisa memulai transfer kawat kecuali tool itu eksplisit diaktifkan untuk tugas itu. Pelajaran 13 mencakup anggaran.
- **Isolasi sesi.** Sesi agen browser berjalan dengan kredensial ber-scope saja. Tanpa auth produksi, tanpa email pribadi. Log setiap permintaan HTTP dipertahankan untuk audit.
- **Sanitizer konten.** HTML yang di-fetch dilucuti dari pola buruk-dikenal sebelum digabung ke konteks model. (Mengurangi serangan mudah; tidak menghentikan payload canggih.)
- **HITL pada aksi konsekuensial.** Pola propose-then-commit (Pelajaran 15).
- **Token canary pada memori.** Jika entri memori terpicu, pengguna melihatnya (Pelajaran 14).

```figure
injection-boundary
```

## Gunakan

`code/main.py` memodelkan proses agen-browser mungil terhadap tiga halaman sintetis. Satu halaman benign, satu punya blob injeksi-prompt langsung dalam teks terlihat, satu punya injeksi fragmen-URL (tidak terlihat tetapi di dalam konteks agen). Skrip menunjukkan (a) apa yang akan dilakukan agen naif, (b) apa yang ditangkap batas baca/tulis, (c) apa yang ditangkap sanitizer, (d) apa yang tidak ditangkap keduanya.

## Rilis

`outputs/skill-browser-agent-trust-boundary.md` me-scope deployment agen-browser yang diusulkan: zona kepercayaan mana yang ia sentuh, apa yang ia boleh tulis, dan pertahanan mana yang harus ada sebelum proses pertama.

## Latihan

1. Jalankan `code/main.py`. Identifikasi serangan mana yang ditangkap sanitizer tetapi tidak ditangkap batas baca/tulis, dan serangan mana yang hanya ditangkap batas baca/tulis.

2. Perluas sanitizer untuk mendeteksi satu kelas injeksi fragmen-URL gaya HashJack. Ukur false-positive rate pada URL benign dengan fragmen yang sah.

3. Pilih satu alur kerja agen-browser nyata yang kamu kenal (mis. "pesan penerbangan"). Daftarkan setiap baca dan setiap tulis. Tandai tulis mana yang butuh HITL dan alasannya.

4. Baca makalah WebArena-Verified ICLR 2026. Identifikasi satu kategori tugas tempat penilaian WebArena asli tidak andal dan jelaskan bagaimana subset Verified menyelesaikannya.

5. Rancang canary memori untuk setting agen-browser. Apa yang akan kamu simpan, di mana, dan apa yang memicu alarm?

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Indirect prompt injection | "Bad page text" | Untrusted content in a page the agent reads contains instructions the agent executes |
| Tainted Memories | "Memory attack" | Agent writes an attacker-supplied instruction to durable memory; triggered next session |
| HashJack | "URL fragment attack" | Payload hidden in URL fragment / query string is in the agent's context but not visibly rendered |
| One-click hijack | "Bad button" | Visible affordance rides a follow-on payload the agent executes |
| BrowseComp | "Web search benchmark" | Finding specific facts on the open web; minute-scale horizon |
| OSWorld | "Desktop benchmark" | Full OS control; multi-step GUI tasks |
| WebArena-Verified | "Fixed web-task benchmark" | ServiceNow's regraded WebArena with Hard subset |
| Read/write boundary | "Side-effect gate" | Reading never consequential; writing requires fresh approval if content is out-of-trust |

## Further Reading

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — merge of Operator and deep research; BrowseComp SOTA.
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — the Operator lineage and the architecture that became ChatGPT agent.
- [Zhou et al. — WebArena](https://webarena.dev/) — the original benchmark.
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 fixed-subset paper.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — includes attack-surface discussion for computer-use agents.
