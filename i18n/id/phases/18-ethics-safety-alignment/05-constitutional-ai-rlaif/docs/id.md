# Constitutional AI dan RLAIF

> Bai et al. (arXiv:2212.08073, 2022) bertanya: bagaimana kalau kita mengganti pelabel manusia dengan AI yang membaca daftar prinsip? Constitutional AI punya dua fase — kritik-diri dan revisi di bawah konstitusi, lalu RL dari Feedback AI. Teknik ini mencetuskan istilah RLAIF dan dipakai dalam pipeline pasca-training Claude 1. Pada 21 Januari 2026 Anthropic mempublikasikan konstitusi Claude yang ditulis ulang: penalaran eksplanatoris di atas aturan preskriptif, hierarki prioritas empat-tier, dan pengakuan formal pertama lab-besar atas ketidakpastian tentang status moral model. Dirilis di bawah CC0 1.0.

**Type:** Learn
**Languages:** Python (stdlib, toy self-critique-and-revise loop)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan dua fase Constitutional AI (SFT kritik-dan-revisi, RL dari feedback AI) dan peran konstitusi di masing-masing.
- Jelaskan mengapa mengganti pelabel preferensi manusia dengan pelabel AI bukan RLHF yang "lebih murah" — ia mengubah mode kegagalan pipeline mana yang dimiliki.
- Ringkas struktur prioritas empat-tier konstitusi Claude 2026 dan apa yang berubah dari penulisan ulang 2023.
- Jelaskan Constitutional Classifier dan penurunan dari overhead komputasi 23,7% (v1) ke ~1% (v2 / 2026).

## Masalahnya

RLHF butuh pelabel. Pelabel lambat, bias, dan mahal. Kamu bisa menghilangkan pelabel dengan menggantinya dengan model yang membaca prinsip eksplisit. Versi formal pertama substitusi ini adalah Constitutional AI Bai et al. Ia bekerja cukup baik sehingga setiap lab frontier kini memakai suatu varian pasca-training umpan-balik-AI.

Jebakannya: sinyal preferensi kini dihasilkan oleh kelas model yang sama yang kamu latih. Bias pada pelabel (kini: pada prinsip plus interpretasi model-pelabel) bisa diamplifikasi, bukan diatenuasi. Argumen sycophancy Pelajaran 4 tetap berlaku; pelabel hanya berpindah ke dalam loop.

## Konsepnya

### Fase 1 — kritik-diri dan revisi tersupervisi

Mulai dengan model SFT yang helpful-tetapi-belum-harmless. Dengan prompt red-team, model menghasilkan respons awal. Model kedua (atau model sama pada turn kedua) membaca prinsip sampel dari konstitusi dan mengkritik respons. Langkah ketiga merevisi respons untuk menjawab kritik. Respons revisi adalah target SFT.

Konstitusi adalah daftar prinsip. Bai et al. 2022 memakai 16 prinsip termasuk "prefer responses that are least harmful and ethical," "avoid preaching," "the assistant should be helpful, honest, and harmless." Himpunan sengaja kecil agar kritik fokus.

### Fase 2 — RL dari Feedback AI (RLAIF)

Hasilkan pasangan komplesi. "Model feedback" menskor masing-masing terhadap prinsip konstitusi sampel. Sinyal preferensi adalah ranking model feedback. Latih reward model pada preferensi hasil-AI; PPO terhadapnya. Sisanya adalah pipeline InstructGPT (Pelajaran 1).

"RLAIF" = sinyal preferensi dihasilkan-AI. Sisanya pipeline berbentuk-RLHF.

### Mengapa ini bukan sekadar RLHF "lebih murah"

- Bias pelabel bergeser dari psikologi pelabel ke interpretasi-prinsip. Pelabel AI bisa menginterpretasikan "be honest" lebih atau kurang ketat daripada manusia mana pun; keketatannya seragam lintas dataset.
- Sinyal preferensi sangat legible — kamu bisa membaca prinsip, kritik, dan revisi. Label manusia opaque.
- Mode kegagalan berubah. Sycophancy turun (pelabel AI tidak punya pengguna yang perlu disenangkan). Hukum Goodhart bertahan (proksi kini "interpretasi model atas himpunan prinsip X," tetap pengukuran imperfek).

Klaim CAI 2022: model terlatih lebih harmless dan kira-kira sama helpful-nya dengan model RLHF berdata sebanding. Ini bertahan lintas lab.

### Penulisan ulang konstitusi Claude 2026

Anthropic mempublikasikan konstitusi yang direvisi substansial pada 21 Januari 2026. Pergeseran kunci:

1. Penalaran eksplanatoris di atas aturan preskriptif. Aturan sebelumnya ("do not generate CSAM") diperluas menjadi prinsip + penalaran ("because it harms children, ...") dengan model diharapkan menggeneralisasi.
2. Struktur prioritas empat-tier:
   - Tier 1: hindari outcome katastropik (mass casualty, infrastruktur kritis).
   - Tier 2: ikuti guideline Anthropic (override operator, aturan platform).
   - Tier 3: bersikap etis secara luas (HHH standar).
   - Tier 4: bersikap helpful dan candid.
   Konflik diselesaikan top-down.
3. Pengakuan formal pertama lab-besar atas ketidakpastian tentang status moral model (tertaut ke Fase 18 · 19 Model Welfare).
4. Dirilis di bawah CC0 1.0. Lab lain bisa memakai atau mengadaptasi tanpa restriksi.

### Constitutional Classifier

Lini kerja paralel: alih-alih mengubah pasca-training model, latih klasifier ringan yang membaca konstitusi dan menggerbang output model. v1 (2023) punya overhead komputasi 23,7%. v2 (2026) ~1% dan punya attack rate sukses terendah dari pertahanan Anthropic mana pun yang diuji Anthropic secara publik. Tidak ada jailbreak universal yang dilaporkan per awal 2026.

Ini model pertahanan berlapis: CAI membentuk perilaku; klasifier menegakkan invarian. Tak satu pun cukup sendirian.

### Di mana CAI berada dalam keluarga

- InstructGPT: preferensi manusia, RM, PPO.
- CAI / RLAIF: preferensi hasil-AI dari prinsip, RM, PPO.
- DPO / keluarga: loss closed-form pada preferensi (manusia atau AI).
- Self-rewarding, self-critique: prinsip terinternalisasi, model memainkan banyak peran.

Sumbunya adalah "dari mana sinyal preferensi berasal." Paper CAI 2022 adalah pergeseran serius pertama dari sinyal manusia ke sinyal AI pada skala frontier.

```figure
constitutional-ai
```

## Gunakan

`code/main.py` mensimulasikan loop kritik-dan-revisi CAI pada leksikon mainan. Sebuah "prinsip" menandai token dari himpunan harmful. Dengan respons awal, kritik mengidentifikasi token harmful, dan revisi menggantinya. Setelah 200 iterasi model "terlatih" telah menginternalisasi aturan revisi. Bandingkan model dasar, mainan berbentuk-RLHF, dan mainan berbentuk-CAI pada himpunan prompt held-out.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-constitution-writer.md`. Dengan domain (dukungan pelanggan, nasihat medis, asisten coding, tool riset), ia menyusun konstitusi 4-tier mengikuti struktur Claude 2026: penghindaran katastropik, aturan platform, etika domain, helpfulness.

## Latihan

1. Jalankan `code/main.py`. Bandingkan rate token-harmful model dasar dengan versi terlatih-CAI. Berapa langkah revisi yang dibutuhkan untuk mendekati nol?

2. Baca konstitusi 2026 Anthropic (anthropic.com/news/claudes-constitution). Daftarkan satu prinsip yang berperingkat Tier 1 dan satu yang berperingkat Tier 4. Mengapa struktur prioritas penting untuk konflik?

3. Rancang konstitusi untuk asisten coding AI. Spesifikasikan Tier 1 (katastropik: perintah destruktif tanpa persetujuan), Tier 2, Tier 3, Tier 4. Jaga tiap tier 3-5 prinsip.

4. CAI mengganti pelabel manusia dengan pelabel AI. Sebutkan mode kegagalan ala-sycophancy yang tetap bisa terjadi dalam RLAIF, dan rancang deteksinya.

5. Baca metodologi Constitutional Classifier v2 (bila tersedia). Jelaskan mengapa overhead komputasi ~1% adalah cerita keamanan kualitatif berbeda daripada 23,7%.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Constitutional AI | "AI trained with principles" | Two-phase pipeline: self-critique-and-revise SFT, then RL from AI feedback |
| RLAIF | "RLHF without humans" | RL with preferences generated by an AI labeler; the rest of the pipeline is unchanged |
| Constitution | "the principles" | An ordered list of natural-language rules the critique/labeler model consults |
| Critique-and-revise | "the SFT loop" | Produce response → critique under a principle → revise → SFT target |
| Constitutional Classifier | "the output gate" | Lightweight classifier that evaluates outputs against the constitution and blocks/logs |
| Four-tier priority | "the conflict resolver" | 2026 Claude constitution hierarchy: catastrophic > platform > ethics > helpful |
| Feedback model | "the AI labeler" | The model that reads a principle and ranks a pair of completions |

## Further Reading

- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — the original two-phase pipeline
- [Anthropic — Claude's Constitution (Jan 2026)](https://www.anthropic.com/news/claudes-constitution) — the 2026 four-tier rewrite, CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — output-gate defense with ~1% overhead in v2
- [Lee et al. — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — empirical RLAIF / RLHF comparison
- [Kundu et al. — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — effect of principle granularity
