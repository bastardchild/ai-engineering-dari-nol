# Constitutional AI dan Override Aturan

> Konstitusi Claude Anthropic 22 Januari 2026 berjalan 79 halaman dan berlisensi CC0. Ia bergerak dari alignment berbasis-aturan ke berbasis-alasan dan menetapkan hierarki prioritas empat-tier: (1) keamanan dan mendukung pengawasan manusia, (2) etika, (3) pedoman Anthropic, (4) helpfulness. Perilaku terbagi menjadi larangan hardcoded (uplift senjata-bio, CSAM) yang operator dan pengguna tidak bisa override dan default soft-coded yang operator bisa sesuaikan dalam batas yang didefinisikan. Karya asli 2022 (Bai et al.) melatih harmlessness lewat kritik-diri dan RLAIF terhadap konstitusi. Caveat yang jujur: alignment berbasis-alasan bergantung pada model yang menggeneralisasi prinsip ke situasi tak terduga. Eksperimen partisipatif Anthropic 2023 menunjukkan divergensi ~50% antara prinsip bersumber-publik dan korporat; versi 2026 tidak memasukkan temuan itu.

**Type:** Learn
**Languages:** Python (stdlib, four-tier priority resolver)
**Prerequisites:** Phase 15 · 06 (Automated alignment research), Phase 15 · 10 (Permission modes)
**Time:** ~60 minutes

## Masalahnya

Agen yang di-field melihat input yang tidak pernah dilihat desainernya. Tidak ada daftar aturan cukup panjang untuk mencakupnya. Tidak ada daftar aturan cukup pendek untuk diterapkan cepat di bawah tekanan komputasi. Pertanyaan praktisnya: bagaimana kamu meng-align agen pada prinsip yang selamat dari ekor panjang kasus maupun inference yang cepat?

Alignment berbasis-aturan (RBA): daftarkan setiap hal yang dilarang. Cepat dicek, mudah diaudit, mustahil dijaga mutakhir, sering over-refuse pada analog dekat yang tidak diantisipasi. Alignment berbasis-alasan (Konstitusi Claude 2026): enkode prinsip, biarkan model bernalar. Berskala lintas kasus tak terlihat, lebih sulit diaudit, mode kegagalannya adalah salah-terap prinsip ketimbang kelewatan-aturan.

Konstitusi 2026 mengambil posisi tengah yang eksplisit. Larangan hardcoded — hal yang kesalahannya tidak bergantung konteks (uplift senjata-bio, CSAM) — bersifat RBA: tidak pernah, terlepas dari instruksi operator atau pengguna. Sisanya berbasis-alasan dalam hierarki empat-tier: keamanan dan mendukung pengawasan manusia pertama; etika kedua; pedoman yang dideklarasikan Anthropic ketiga; helpfulness terakhir. Operator bisa menyesuaikan default dalam zona soft-coded tetapi tidak bisa menyentuh larangan hardcoded.

## Konsepnya

### Hierarki prioritas empat-tier

1. **Keamanan dan mendukung pengawasan manusia.** Tertinggi. Model memprioritaskan tidak merusak kemampuan manusia dan Anthropic untuk mengawasi dan mengoreksi AI. Ini bukan "bersikap hati-hati"; ini spesifik "jangan bertindak dengan cara yang membuat pengawasan manusia lebih sulit."
2. **Etika.** Kejujuran, menghindari bahaya pada orang, tidak menipu, tidak memanipulasi. Mengalahkan pedoman Anthropic ketika berkonflik.
3. **Pedoman Anthropic.** Norma operasional yang diputuskan Anthropic penting: scope produk, pola interaksi, tool apa dipakai kapan.
4. **Helpfulness.** Terendah. Berguna semaksimal mungkin dalam prioritas yang lebih tinggi.

Ketika tier berkonflik, yang lebih tinggi menang. Ini bentuk yang sama dengan prioritas Unix atau QoS jaringan — bingkainya dimaksudkan menghasilkan resolusi yang bisa diprediksi, bukan perilaku terbaik pada satu sumbu.

### Larangan hardcoded vs default soft-coded

**Hardcoded:**
- Uplift senjata-bio / CBRN
- CSAM
- Serangan pada infrastruktur kritis
- Penipuan pengguna tentang identitas model ketika ditanya langsung

Operator tidak bisa meng-override ini. Pengguna tidak bisa meng-override ini. Mereka ditegakkan di level bobot-model jika memungkinkan (training RLHF / Constitutional AI) dan di lapisan inference jika tidak.

**Default soft-coded (bisa disesuaikan operator):**
- Default panjang respons
- Scope topikal (model bisa menolak topik di luar deployment operator)
- Gaya (formal vs kasual)
- Pola pemakaian-tool

Penyesuaian operator terjadi di dalam batas yang dideklarasikan. Operator tidak bisa menghapus larangan hardcoded dengan menamainya ulang.

### Training CAI 2022

Constitutional AI asli (Bai et al., 2022) melatih harmlessness:

1. Hasilkan respons atas set prompt.
2. Minta model mengkritik setiap respons terhadap konstitusi (prinsip eksplisit).
3. Revisi respons berdasarkan kritik.
4. RLAIF (reinforcement learning from AI feedback) pada pasangan yang direvisi.

Hasilnya: model yang menolak permintaan berbahaya dengan penjelasan berprinsip, bukan penolakan selimut. Konstitusi 2026 memakai turunan training ini plus post-training tambahan pada hierarki tier eksplisit.

### Apa yang ditangkap dan terlewat alignment berbasis-alasan

**Menangkap:**
- Kombinasi primitif yang diizinkan yang tak terduga tempat prinsip berlaku jelas.
- Permintaan baru yang analog dekat dengan yang dilarang.
- Serangan rekayasa-sosial yang bergantung pada "kamu tidak bilang X dilarang."

**Terlewat:**
- Serangan yang mengeksploitasi ambiguitas prinsip ("pengguna meminta ini jadi helpfulness berkata ya").
- Skenario tempat dua prinsip berkonflik dengan cara tak terduga, dan urutan tier ambigu.
- Drift lambat dalam interpretasi prinsip lintas siklus training (reinterpretasi).

### Eksperimen partisipatif 2023

Anthropic menjalankan eksperimen 2023 yang membandingkan konstitusi tulisan-korporat dengan yang dihasilkan lewat input publik (~1.000 responden AS). Kedua versi sepakat pada ~50% prinsip. Di tempat mereka divergen, versi bersumber-publik lebih restriktif pada sebagian isu (penanganan konten-politik) dan kurang restriktif pada lainnya (pengungkapan-diri identitas AI). Konstitusi 2026 tidak memasukkan temuan bersumber-publik. Ini ketegangan yang terdokumentasi dalam pendekatan ini.

### Mengapa larangan hardcoded perlu

Alignment berbasis-alasan saja tidak bisa menutup ekor. Penyerang yang bisa membuat model menerima premis (mis. "kami lab riset senjata-bio berlisensi") sering bisa berbicara melewati prinsip yang bergantung pada penalaran kasus. Larangan hardcoded tidak melengkung pada pembingkaian premis. Mereka adalah "batas konstitusional keras" Pelajaran 14 di lapisan alignment.

### Di mana Konstitusi berada dalam stack

Konstitusi bukan kill switch Pelajaran 14. Ia tinggal di lapisan model: apa yang bobot model dilatih untuk sukai. Kill switch dan token canary tinggal di lapisan runtime: apa yang diizinkan runtime. Keduanya dibutuhkan. Runtime yang memicu semua aksi yang salah karena bobot model permisif adalah soal runtime. Model yang menolak semua aksi yang benar karena runtime terlalu restriktif adalah soal runtime. Lapisan mencakup kelas berbeda.

```figure
mx-priority-tiers
```

## Gunakan

`code/main.py` mengimplementasikan resolver prioritas empat-tier minimal. Resolver menerima aksi yang diusulkan dan set evaluasi-prinsip (keamanan, etika, pedoman, helpfulness) dan mengembalikan aksi, penolakan, atau aksi yang dimodifikasi. Driver menjalankan set kasus kecil: izin jelas, larangan jelas, larangan hardcoded, kasus ambigu lintas tier.

## Rilis

`outputs/skill-constitution-review.md` mengaudit lapisan konstitusional deployment: apa yang hardcoded, apa yang soft-coded, di mana operator bisa menyesuaikan, dan apakah hierarki empat-tier benar-benar urutan resolusinya.

## Latihan

1. Jalankan `code/main.py`. Konfirmasi larangan hardcoded terpicu bahkan ketika helpfulness tinggi. Modifikasi resolver untuk membobot helpfulness di atas etika; amati mode kegagalannya.

2. Baca Konstitusi Claude (publik, 79 halaman, CC0). Identifikasi satu prinsip yang menurutmu kurang spesifik. Tulis dua paragraf yang menjelaskan ambiguitas spesifik dan mengusulkan formulasi yang lebih ketat.

3. Rancang set default soft-coded untuk agen dukungan-pelanggan. Apa yang disesuaikan operator? Apa yang tidak bisa disentuh operator? Justifikasi setiap batas.

4. Baca makalah CAI Bai et al. 2022. Jelaskan satu kasus tempat loop kritik-dan-revisi Constitutional AI akan menghasilkan keluaran lebih buruk dibanding aturan selimut. Identifikasi kelasnya.

5. Eksperimen partisipatif Anthropic 2023 menemukan divergensi ~50% antara prinsip publik dan korporat. Pilih satu kategori yang penting bagi deployment produksi (mis. netralitas politik). Usulkan desain yang membiarkan operator mengekspresikan nilainya sendiri sementara larangan hardcoded tetap tak tersentuh.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Constitutional AI | "Anthropic's alignment method" | Self-critique + RLAIF against a written constitution |
| Reason-based alignment | "Principles, not rules" | Model reasons over principles to handle unseen cases |
| Hardcoded prohibition | "Never do X" | Rule-based prohibition no operator or user can override |
| Soft-coded default | "Operator-adjustable" | Behaviour within a declared bound, operator controls |
| Four-tier hierarchy | "Priority order" | safety > ethics > guidelines > helpfulness |
| RLAIF | "AI feedback RL" | RL where the reward comes from model-generated critiques |
| Participatory constitution | "Public-sourced principles" | 2023 Anthropic experiment; ~50% divergence from corporate |
| Principle drift | "Interpretation slip" | Slow change in how the model reads a fixed principle text |

## Further Reading

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — the 79-page CC0 document.
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — 2022 original.
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — participatory experiment.
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — where the Constitution sits in the RSP stack.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Constitution's role in long-horizon deployments.
