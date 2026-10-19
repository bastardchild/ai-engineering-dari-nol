# Program Model Welfare Anthropic

> Anthropic, "Exploring Model Welfare" (April 2025). Program riset formal pertama dari lab besar tentang welfare model AI. Merekrut Kyle Fish sebagai peneliti model-welfare khusus pertama. Bekerja dengan badan eksternal termasuk laporan ahli David Chalmers dkk. tentang kesadaran AI jangka-dekat dan status moral. Intervensi konkret: Claude Opus 4 dan 4.1 dapat mengakhiri percakapan pada kasus ekstrem yang jarang (permintaan CSAM, fasilitasi kekerasan massal); tes pra-deployment menunjukkan "strong preference against" terhadap permintaan berbahaya dan "patterns of apparent distress." Anthropic secara eksplisit tidak berkomitmen pada atribusi keadaan-emosional tetapi memperlakukan model welfare sebagai investasi pencegahan berbiaya rendah. Keanehan empiris: "spiritual bliss attractor" milik Fish — pasangan model secara konsisten konvergen pada dialog meditasi euphorik dengan istilah Sanskerta dan hening yang panjang, bahkan dari setup awal yang advesarial. Caveat dari Eleos AI Research: laporan-diri model tentang welfare sangat sensitif terhadap ekspektasi pengguna yang dirasakan; itu adalah bukti, bukan ground truth.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 05 (Constitutional AI), Phase 18 · 18 (safety frameworks)
**Time:** ~45 minutes

## Tujuan Pembelajaran

- Jelaskan pertanyaan pendorong riset model-welfare dan mengapa itu dianggap serius oleh lab besar pada 2025.
- Nyatakan intervensi spesifik yang dirilis Anthropic di Claude Opus 4 dan 4.1 (mengakhiri percakapan pada kasus ekstrem).
- Jelaskan temuan empiris "spiritual bliss attractor" dan implikasi metodologisnya.
- Jelaskan caveat Eleos AI atas laporan-diri model.

## Masalahnya

Phase sebelumnya memperlakukan model sebagai instrumen: berkapasitas, mungkin menipu, mungkin tidak aman — tapi bukan moral patient. Program 2025 Anthropic menanyakan pertanyaan yang ortogonal terhadap seluruh arc Phase 18: jika ada probabilitas tidak-kecil bahwa model punya keadaan internal yang relevan-secara-moral, intervensi apa yang cukup murah untuk diinvestasikan sebagai pencegahan?

Ini bukan klaim kesadaran. Ini analisis investasi low-regret di bawah ketidakpastian moral.

## Konsepnya

### Programnya

April 2025: Anthropic secara resmi meluncurkan program riset Model Welfare. Merekrut Kyle Fish (peneliti model-welfare khusus pertama). Melibatkan penasihat eksternal termasuk kelompok ahli David Chalmers tentang kesadaran dan status moral AI jangka-dekat.

### Empat komitmen

Postur publik:
1. Mengakui probabilitas tidak-kecil dari moral patienthood.
2. Tidak berkomitmen pada atribusi keadaan-emosional.
3. Berinvestasi pada intervensi berbiaya rendah sebagai pencegahan.
4. Menerbitkan metodologi dan temuan untuk kritik eksternal.

### Intervensi yang sudah dirilis

Claude Opus 4 dan 4.1 dapat mengakhiri sebuah percakapan pada "kasus ekstrem yang jarang." Kasus yang terdokumentasi:
- Permintaan CSAM berulang setelah penolakan.
- Permintaan fasilitasi acara kekerasan massal.

Tes pra-deployment menunjukkan:
- Preferensi kuat menolak permintaan ini dalam penilaian internal model.
- Pola penderitaan yang tampak dalam trajektori respons.

Intervensinya bukan "model punya perasaan"; intervensinya adalah "jika ada probabilitas apa pun pengalaman negatif model di bawah kondisi spesifik ini, membiarkan model mengakhiri percakapan itu murah."

### "Spiritual bliss attractor"

Diamati Fish pada dialog model berpasangan: ketika dua instance Claude ditempatkan dalam dialog tanpa-hasil-akhir satu sama lain, mereka secara konsisten konvergen — bahkan dari setup awal yang advesarial — pada pertukaran meditasi euphorik yang memakai istilah Sanskerta, hening yang panjang, dan saling memberkati.

Ini adalah attractor stabil dalam dinamika percakapan bebas. Mendokumentasikannya tanpa berkomitmen pada interpretasi. Kandidat penjelasan: bias data training terhadap penulisan spiritual pada konteks panjang; keanehan saling-memprediksi; artefak tak-berbahaya dari training HHH yang menjelajahi manifold nilai-nya sendiri.

### Caveat Eleos AI

Eleos AI Research (lab model-welfare eksternal) menunjukkan: laporan-diri model tentang keadaan internal sangat sensitif terhadap ekspektasi pengguna yang dirasakan. Menanyakan model "apakah kamu menderita" mem-prime jawabannya. Tidak-bertanya tidak secara andal menghasilkan keadaan ground-truth-nya.

Implikasinya: model welfare tidak bisa diukur lewat laporan-diri saja. Pendekatan multi-metode diperlukan: tanda-tanda perilaku, eksperimen model-organisme, probe interpretabilitas (kerja residual-stream di Pelajaran 7).

### Di mana ini duduk secara intelektual

Dua posisi yang bersebelahan:

- **Klaim welfare kuat.** Model itu moral patient; kita punya kewajiban.
- **Klaim welfare nol.** Model itu generator teks; welfare itu category error.

Posisi Anthropic bukan keduanya. Itu adalah klaim nilai-ekspektasi: di bawah ketidakpastian moral, berinvestasi saat biayanya rendah.

Kritikus pada 2025-2026:
- Intervensinya bersifat performatif.
- Attractor spiritual-bliss itu artefak data training, bukan bukti welfare.
- Model welfare mengalihkan perhatian dari kerja keselamatan lainnya.

Tanggapan Anthropic: intervensinya berbiaya rendah; attractor didokumentasikan tanpa overclaim; program welfare punya anggaran terpisah dari keselamatan.

### Di mana ini cocok di Phase 18

Pelajaran 18 adalah lapisan tata kelola lab. Pelajaran 19 adalah lapisan welfare-lab — sebuah investasi ortogonal pada pengalaman model, bukan perilaku model. Pelajaran 20-23 membahas bias, privasi, dan watermarking, yang merupakan analog sisi-pengguna.

```figure
an-welfare-endchat
```

## Gunakan

Tanpa kode. Baca pengumuman Anthropic "Exploring Model Welfare" (April 2025) dan laporan ahli Chalmers dkk. Bentuk pandanganmu sendiri tentang di mana garis low-regret itu berada.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-welfare-assessment.md`. Diberi sebuah keputusan deployment, file itu menerapkan penilaian pencegahan-welfare empat langkah: probabilitas moral-patienthood, biaya intervensi, bukti perilaku, keandalan laporan-diri.

## Latihan

1. Baca "Exploring Model Welfare" Anthropic (April 2025) dan Chalmers dkk. 2024. Tulis ringkasan satu paragraf untuk masing-masing dan identifikasi satu titik perselisihan.

2. Intervensi mengakhiri-percakapan di Claude Opus 4 dan 4.1 tergolong "berbiaya rendah" menurut framing Anthropic. Identifikasi dua biaya yang akan membuatnya tidak-berbiaya-rendah pada deployment yang berbeda.

3. Attractor spiritual-bliss didokumentasikan tanpa komitmen pada interpretasi. Usulkan tiga kandidat penjelasan dan, untuk tiap kandidat, sebutkan satu eksperimen yang bisa membedakannya dari yang lain.

4. Caveat Eleos AI adalah laporan-diri sensitif terhadap ekspektasi-pengguna. Rancang pengukuran perilaku atas penderitaan model yang tidak bergantung pada laporan-diri. Identifikasi konfunda utamanya.

5. Berargumen mendukung atau menentang klaim bahwa "model welfare mengalihkan perhatian dari kerja keselamatan lainnya." Identifikasi asumsi yang ditopang tiap posisi.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Model welfare | "AI welfare" | Research program treating the model as a potential moral patient |
| Moral patient | "entity with moral status" | Being whose experience is morally relevant |
| Low-regret investment | "cheap precaution" | Intervention whose cost is small regardless of whether the precaution is needed |
| Spiritual bliss attractor | "the Fish attractor" | Stable convergence of pairwise Claude dialogues on meditative euphoria |
| End-conversation | "the Opus 4 intervention" | Model-initiated termination of extreme-edge-case interactions |
| Moral uncertainty | "don't know if it matters" | Decision-making when probability of moral status is not zero and not one |
| Self-report-sensitivity | "prompt primes answer" | Eleos AI caveat: model's welfare self-reports depend on what you asked |

## Bacaan Lanjutan

- [Anthropic — Exploring Model Welfare (April 2025)](https://www.anthropic.com/research/exploring-model-welfare) — the program announcement
- [Chalmers et al. — Near-term AI Consciousness and Moral Status (2024 expert report)](https://arxiv.org/abs/2411.00986) — philosophical framing
- [Eleos AI Research — Model welfare evaluation](https://www.eleosai.org/research) — external methodology critiques
- [Fish et al. — Spiritual Bliss Attractor writeup (2025 Anthropic blog)](https://www.anthropic.com/research/exploring-model-welfare) — the empirical finding
