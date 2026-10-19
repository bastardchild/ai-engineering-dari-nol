# Speculative Decoding EAGLE-3 di Produksi

> Speculative decoding memasangkan draft model yang cepat dengan model target. Draft mengusulkan K token; model target memverifikasi dalam satu forward; token yang diterima pada dasarnya gratis. Pada 2026, EAGLE-3 adalah varian production-grade — ia melatih draft head pada hidden state model target, bukan pada token mentah, sehingga mendorong acceptance rate alpha ke rentang 0,6-0,8 pada chat umum. Pertanyaan tepat bukan "seberapa cepat draft" melainkan "berapa alpha pada trafikku?" Kalau alpha turun di bawah ~0,55, speculative decoding menjadi net negatif pada konkurensi tinggi karena setiap draft yang ditolak memakan satu forward pass target tambahan. Pelajaran ini mengajarimu mengukur alpha dulu dan menyalakan flag kemudian.

**Type:** Learn
**Languages:** Python (stdlib, toy acceptance-rate simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Sebutkan tiga generasi speculative decoding dan jelaskan apa yang diubah EAGLE-3 dari EAGLE-2 dan dari draft model klasik.
- Definisikan acceptance rate alpha, hitung speedup ekspektasi dari alpha dan K (panjang draft), dan identifikasi alpha titik-impas untuk konkurensi targetmu.
- Jelaskan mengapa speculative decoding bersifat opt-in (bukan default) di vLLM 2026 dan mengapa menyalakannya tanpa mengukur alpha adalah antipola produksi.
- Tulis rencana pengukuran: benchmark mana, distribusi prompt mana, titik konkurensi mana, metrik mana yang menjadi gerbang.

## Masalahnya

Decode bersifat memory-bound. Pada H100 yang menjalankan Llama 3.3 70B FP8, setiap token yang di-decode membaca ~140 GB/detik bobot dan mengemisikan satu token. Komputasi GPU nyaris idle selama decode — bottleneck-nya adalah bandwidth HBM, bukan throughput matmul.

Speculative decoding mengeksploitasi selisih itu. Hasilkan K token kandidat dengan draft model murah, lalu minta model target memverifikasi semuanya dalam satu forward pass. Setiap token terverifikasi pada dasarnya gratis (diamortisasi menjadi forward batch-of-K yang memang harus dilakukan target).

Pendekatan draft-model klasik memakai model lebih kecil dari famili yang sama (Llama 3.2 1B menyusun draft untuk Llama 3.3 70B). Ia bekerja tetapi acceptance rate biasa saja — distribusi model kecil menyimpang dari target. EAGLE, lalu EAGLE-2, lalu EAGLE-3 melatih draft head ringan langsung pada state internal model target, sehingga distribusi draft mengikuti target jauh lebih dekat. Itulah mengapa alpha naik dari 0,4 dengan draft-model menjadi 0,6-0,8 dengan EAGLE-3.

Jebakannya: EAGLE-3 bersifat opt-in di vLLM 2026. `speculative_config` harus diset eksplisit. Tanpa flag, tanpa akselerasi. Tim yang menyalakannya tanpa mengukur alpha pada trafik nyata sering melihat latensi ekor memburuk, bukan membaik.

## Konsepnya

### Apa yang sebenarnya dibeli speculative decoding

Tanpa spec decode, biaya per-token adalah satu forward target. Dengan spec decode pada panjang draft K dan acceptance alpha, ekspektasi token per forward target adalah `1 + K * alpha`. Speedup-nya adalah `(1 + K * alpha) / (1 + epsilon)` di mana epsilon adalah overhead draft-plus-verifikasi. Untuk K=5, alpha=0,7: `(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`. Angka dunia-nyata mengelompok sekitar 2-3x karena alpha jarang setinggi itu pada trafik produksi dan epsilon tumbuh pada batch size besar.

### Mengapa alpha satu-satunya metrik yang penting

Token yang ditolak tidak hilang begitu saja — mereka memaksa forward target kedua untuk token pertama yang ditolak. Pada workload di mana alpha turun ke 0,4, kamu membayar overhead draft plus verifikasi plus roll-ulang. Pada konkurensi tinggi (misal 256 konkuren), batch decode sudah cukup besar sehingga selisih bandwidth-memori antara "target saja" dan "target dengan verifikasi" menyusut. Di bawah alpha 0,55 pada kebanyakan hardware 2026, spec decode menjadi net negatif.

Alpha bervariasi menurut workload. Pada chat umum ala-ShareGPT, EAGLE-3 yang dilatih pada ShareGPT mencapai 0,6-0,8. Pada trafik spesifik-domain (kode, medis, legal) draft head yang dilatih pada data umum turun ke 0,4-0,6. Melatih draft head spesifik-domain memulihkan alpha — ia pekerjaan training ringan dan cepat dibanding finetuning target.

### Generasi EAGLE sekilas

- **Draft model klasik**: model kecil dari famili sama. Alpha 0,3-0,5. Infrastruktur sederhana — dua model dimuat, draft menjalankan K forward per forward target.
- **EAGLE-1 (2024)**: satu draft head dilatih pada hidden state target (lapisan terakhir). Alpha ~0,5-0,6. Overhead param kecil di atas target.
- **EAGLE-2 (2025)**: panjang draft adaptif dan draft berbasis-tree (verifikasi banyak cabang dalam satu pass target). Alpha ~0,6-0,7. Scheduler draft lebih kompleks.
- **EAGLE-3 (2025-2026)**: draft head dilatih pada banyak lapisan target (bukan hanya terakhir), alignment lebih baik. Alpha ~0,6-0,8 pada chat umum.

### Resep produksi 2026

1. Rilis model target polos. Ukur baseline TTFT, ITL, throughput pada konkurensi target.
2. Nyalakan draft EAGLE-3 lewat `speculative_config` vLLM. Ulangi benchmark.
3. Catat acceptance rate alpha. vLLM V1 melaporkannya sebagai `spec_decode_metrics.accepted_tokens_per_request`. Bagi dengan panjang draft yang diminta untuk mendapat alpha.
4. Kalau alpha < 0,55 pada distribusi trafik produksi, matikan spec decode atau latih draft EAGLE-3 spesifik-domain.
5. Pada konkurensi produksi, ulangi. Konfirmasi P99 ITL tidak memburuk.

### Jebakan produksi: ekor P99

Rata-rata ITL turun dengan spec decode. P99 bisa memburuk kalau kamu tidak men-tune. Draft yang ditolak memicu sekuens dua-pass (draft + verifikasi-gagal + roll-ulang). Pada batch penuh, dua pass itu terserialisasi. Awasi P99 ITL, bukan P50.

### Di mana EAGLE-3 sudah di-deploy

Google men-deploy speculative decoding di AI Overviews pada 2025 (kualitas sama, respons lebih cepat). vLLM V1 membawa `speculative_config` sebagai antarmuka terdokumentasi, dan matriks fiturnya menandai speculative decoding kompatibel dengan chunked prefill. SGLang mendukung EAGLE-3 sebagai path draft yang direkomendasikan untuk workload yang kaya-prefix.

### Matematika titik-impas dalam satu baris

Speedup ekspektasi: `S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`. Menetapkan `S = 1` menyelesaikan alpha: `alpha_breakeven = verify_overhead / K`. Untuk verify_overhead tipikal ~0,15 dan K=5: `alpha_breakeven = 0.03`. Tetapi itu matematika decode mentah. Pada konkurensi tinggi overhead verifikasi naik dan batch decode sudah mengamortisasi pembacaan memori lintas sekuens, sehingga alpha_breakeven efektif naik ke ~0,45-0,55 dalam praktik.

### Kapan tidak memakai speculative decoding

- Generasi offline batch-1 di mana latensi tidak penting. Pakai target polos.
- Output sangat pendek (di bawah 50 token). Overhead draft dan biaya verifikasi mendominasi.
- Domain terspesialisasi tanpa draft head yang dilatih-domain. Alpha terlalu rendah.
- Mengasumsikan setiap pasangan fitur bisa dikombinasikan. Cek matriks kompatibilitas vLLM untuk versimu: v0.18.0 menandai speculative decoding kompatibel dengan chunked prefill.

```figure
mx-speculative-tree
```

## Gunakan

`code/main.py` mensimulasikan loop decode dengan dan tanpa speculative decoding pada rentang nilai alpha dan panjang draft K. Ia mencetak alpha titik-impas, speedup terukur, dan perilaku ekor. Jalankan pada beberapa kombinasi (alpha, K) untuk melihat persis di mana speculative decoding berhenti menguntungkan.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-eagle3-rollout.md`. Dengan model target, deskripsi distribusi trafik, dan target konkurensi, ia menghasilkan rencana rollout EAGLE-3 bertahap — benchmark baseline, nyalakan konfigurasi, ukur alpha, gerbang pada alpha >= 0,55, awasi P99 ITL.

## Latihan

1. Jalankan `code/main.py`. Pada K=5, alpha berapa yang kamu butuhkan untuk speedup 2x? Untuk speedup 3x? Seberapa sensitif itu terhadap verify_overhead?
2. Bayangkan trafik produksi terbagi 70% chat umum, 30% kode. Chat umum mencapai alpha 0,7 dengan EAGLE-3 yang dilatih pada ShareGPT; kode mencapai alpha 0,4. Berapa alpha campuran dan apakah spec decode net-positif?
3. Baca dokumentasi `speculative_config` vLLM. Sebutkan tiga mode (draft model, EAGLE, N-gram) dan cek fitur mana yang dikombinasikan masing-masing pada versi vLLM-mu.
4. Kamu melihat rata-rata ITL turun 25% setelah menyalakan EAGLE-3 tetapi P99 ITL naik 15%. Diagnosis dan usulkan mitigasi.
5. Hitung biaya memori draft head EAGLE-3 untuk Llama 3.3 70B. Bagaimana dibanding menjalankan Llama 3.2 1B sebagai draft klasik?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Speculative decoding | "draft plus verify" | Propose K tokens with a cheap model, verify all K in one target forward |
| Acceptance rate alpha | "spec accept rate" | Fraction of draft tokens accepted by the target; the only metric that matters |
| Draft length K | "spec k" | How many tokens the draft proposes per target forward; typical 4-8 |
| Verify overhead epsilon | "spec overhead" | Extra cost to verify-and-reroll vs a plain target forward; grows with batch |
| EAGLE-3 | "latest EAGLE" | 2025-2026 variant; trains draft head on multiple target layers; alpha 0.6-0.8 on general chat |
| `speculative_config` | "vLLM spec config" | The explicit opt-in in vLLM V1; no default means no acceleration |
| N-gram spec decode | "N-gram draft" | GPU-side draft using N-gram lookups in the prompt; chunked-prefill-compatible |
| Break-even alpha | "no-op alpha" | Alpha at which spec decode gives zero speedup; watch this at production concurrency |
| Rejected-draft two-pass | "reroll cost" | Two target forwards when drafts reject; drives P99 tail |

## Further Reading

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — authoritative source on `speculative_config` and chunked-prefill compatibility in V1.
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — the exact field set.
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — original EAGLE draft-head formulation.
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — adaptive drafts and trees.
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — efficient LLM system with speculative decoding.
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — production rollout checklist.
