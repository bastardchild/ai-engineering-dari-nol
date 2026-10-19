# Ekonomi Platform Inferensi — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

> Pasar inferensi 2026 bukan lagi sekadar sewa waktu GPU. Pasar ini terbagi menjadi silikon kustom (Groq, Cerebras, SambaNova), platform GPU (Baseten, Together, Fireworks, Modal), dan marketplace yang mengutamakan API (Replicate, DeepInfra). Fireworks menaikkan harga $1/jam per GPU pada 1 Mei 2026, dan valuasi $4B dengan 10T+ token/hari menunjukkan model berbasis volume berhasil. Baseten menutup Series E $300M pada valuasi $5B pada Januari 2026. Aturan positioning kompetitifnya sederhana: Fireworks mengoptimalkan latensi, Together mengoptimalkan keluasan katalog, Baseten mengoptimalkan polesan enterprise, Modal mengoptimalkan DX yang Python-native, Replicate mengoptimalkan jangkauan multimodal, Anyscale mengoptimalkan Python terdistribusi. Pelajaran ini memberimu matriks yang bisa kamu serahkan ke founder.

**Type:** Learn
**Languages:** Python (stdlib, toy per-call economics comparator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Sebutkan tiga segmen pasar (silikon kustom, platform GPU, API-first) dan petakan setiap vendor ke segmennya.
- Jelaskan mengapa model harga API "per-token" terkompresi mengikuti kurva biaya serving engine, bukan hardware-nya.
- Hitung biaya efektif per request di minimal tiga vendor dan jelaskan kapan per-menit (Baseten, Modal) mengalahkan per-token.
- Identifikasi platform mana yang menjadi default tepat untuk workload tertentu (serverless yang bursty, throughput tinggi yang stabil, varian fine-tuned, multimodal).

## Masalahnya

Kamu sudah mengevaluasi platform hyperscaler terkelola. Kamu memutuskan butuh provider yang lebih sempit dan lebih cepat — Fireworks untuk latensi, Together untuk keluasan, Baseten untuk model kustom fine-tuned. Kini kamu punya enam pilihan nyata dan halaman harganya tidak sejajar. Fireworks menampilkan $/M token; Baseten menampilkan $/menit; Modal menampilkan $/detik; Replicate menampilkan $/prediksi. Kamu tidak bisa membandingkannya head-to-head tanpa memodelkan workload.

Lebih buruk lagi, model bisnis di balik setiap halaman harga berbeda. Fireworks menjalankan engine kustom sendiri (FireAttention) di atas GPU bersama; tarif per-token mencerminkan kurva utilisasinya. Baseten memberimu Truss + GPU khusus; per-menit mencerminkan eksklusivitas. Modal adalah serverless Python sejati — penagihan per-detik dengan cold start sub-detik. Output yang sama (respons LLM), tiga fungsi biaya berbeda.

Pelajaran ini memodelkan keenamnya dan menunjukkan kapan masing-masing menang.

## Konsepnya

### Tiga segmen

**Silikon kustom** — Groq (LPU), Cerebras (WSE), SambaNova (RDU). Biasanya decode 5-10x lebih cepat daripada klaster berbasis GPU pada model yang sama. Harga per-token lebih tinggi (Groq sekitar ~$0,99/M untuk Llama-70B akhir 2025) tetapi tak terkalahkan untuk kasus penggunaan yang sensitif latensi. Groq adalah pilihan produksi untuk agen suara dan terjemahan real-time.

**Platform GPU** — Baseten, Together, Fireworks, Modal, Anyscale. Berjalan di NVIDIA (H100, H200, B200 pada 2026) atau kadang AMD. Lapisan ekonomi antara "sewa GPU mentah" (RunPod, Lambda) dan "layanan terkelola hyperscaler" (Bedrock).

**Marketplace API-first** — Replicate, DeepInfra, OpenRouter, Fal. Katalog luas, bayar-per-prediksi atau bayar-per-detik, menekankan time-to-first-call.

### Fireworks — platform GPU yang dioptimalkan untuk latensi

- Engine FireAttention (kustom); diklaim 4x lebih rendah latensinya daripada vLLM pada konfigurasi setara.
- Tier batch sekitar 50% dari tarif serverless untuk workload non-interaktif.
- Model fine-tuned disajikan dengan tarif yang sama seperti model dasar — pembeda nyata dibanding provider yang mengenakan premi untuk LoRA-mu.
- Pertengahan 2026: menaikkan sewa GPU on-demand $1/jam efektif 1 Mei 2026. Harga volume bisa dinegosiasi pada skala besar.
- Sinyal finansial: valuasi $4B, 10T+ token/hari ditangani.

### Together — dioptimalkan untuk keluasan

- 200+ model termasuk rilis open-source dalam hitungan hari setelah publikasi upstream.
- 50-70% lebih murah daripada Replicate pada model LLM setara — positioning "AI Native Cloud" adalah volume dan katalog.
- Inferensi + fine-tuning + training dalam satu API.

### Baseten — dioptimalkan untuk polesan enterprise

- Framework Truss: pengemasan model berisi dependensi, secret, dan konfigurasi serving dalam satu manifest.
- Rentang GPU dari T4 hingga B200. Penagihan per-menit dengan mitigasi cold-start yang wajar.
- SOC 2 Type II, siap HIPAA. Pilihan umum fintech dan healthcare.
- Valuasi $5B, Series E Januari 2026 ($300M dari CapitalG, IVP, NVIDIA).

### Modal — dioptimalkan untuk Python-native

- Infrastructure-as-code dalam Python murni. Dekorasi fungsi dengan `@modal.function(gpu="A100")` lalu deploy dengan satu perintah.
- Penagihan per-detik. Cold start 2-4 detik dengan pre-warming; <1 detik untuk model kecil.
- Series B $87M pada valuasi $1,1B (2025). Skor developer experience terkuat dalam survei independen.

### Replicate — keluasan multimodal

- Bayar-per-prediksi. Platform default untuk model gambar, video, dan audio.
- Ekosistem integrasi (Zapier, Vercel, plugin CMS).
- Kurang kompetitif pada tarif LLM per-token tetapi menang pada variasi multimodal.

### Anyscale — Ray-native

- Dibangun di atas Ray; RayTurbo adalah engine inferensi proprietary Anyscale (bersaing dengan vLLM).
- Terbaik untuk workload Python terdistribusi di mana langkah inferensi adalah satu node dalam graf yang lebih besar.
- Klaster Ray terkelola; integrasi erat dengan Ray AIR dan Ray Serve.

### Per-token versus per-menit — kapan masing-masing menang

Per-token masuk akal saat workload tidak sensitif latensi dan bursty — kamu hanya membayar yang kamu pakai. Per-menit masuk akal saat utilisasi tinggi dan terprediksi — kamu mengalahkan per-token begitu GPU jenuh.

Aturan kasar: untuk workload di atas ~30% utilisasi berkelanjutan satu GPU khusus, per-menit (Baseten, Modal) mulai mengalahkan per-token (Fireworks, Together). Di bawah itu, per-token menang karena kamu tidak membayar idle.

### Engine kustom adalah moat sesungguhnya

Setiap platform di atas vLLM dan SGLang mengklaim engine kustom. FireAttention, RayTurbo, stack inferensi Baseten. Klaim engine-kustom berbau marketing — framing jujurnya adalah vLLM + SGLang mewakili sekitar 80% inferensi open-source produksi, dan pembeda di lapisan platform adalah DX, atribusi, dan SLA.

### Angka yang perlu kamu ingat

- Sewa GPU Fireworks: kenaikan $1/jam efektif 1 Mei 2026.
- Klaim Fireworks: latensi 4x lebih rendah daripada vLLM pada konfigurasi setara.
- Together: 50-70% lebih murah daripada Replicate untuk LLM.
- Valuasi Baseten: $5B (Series E, Jan 2026, ronde $300M).
- Valuasi Modal: $1,1B (Series B, 2025).
- Per-menit mengalahkan per-token di atas ~30% utilisasi berkelanjutan.

```figure
cost-per-token
```

## Gunakan

`code/main.py` membandingkan enam vendor pada workload sintetis lintas model harga. Melaporkan $/hari dan $/M token efektif. Jalankan untuk menemukan titik impas antara per-token dan per-menit.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-inference-platform-picker.md`. Dengan profil workload, SLA, dan budget, ia memilih platform inferensi utama dan menyebut runner-up-nya.

## Latihan

1. Jalankan `code/main.py`. Pada utilisasi berkelanjutan berapa Baseten (per-menit) mengalahkan Fireworks (per-token) untuk model 70B pada satu H100? Turunkan crossover sendiri dan bandingkan dengan aturan praktisnya.
2. Produkmu menyajikan generasi gambar plus chat plus speech-to-text. Pilih platform untuk tiap modalitas dan sebutkan pola gateway yang menyatukannya.
3. Fireworks menaikkan harga $1/jam pada model utamamu. Modelkan dampak biaya campuran jika 40% trafikmu pindah ke tier batch (50% lebih murah).
4. Pelanggan teregulasi mewajibkan SOC 2 Type II + HIPAA + GPU khusus. Tiga platform mana yang viable dan mana yang menang pada FinOps?
5. Bandingkan biaya per 1.000 prediksi untuk Llama 3.1 70B di Fireworks serverless, Together on-demand, Baseten khusus, dan Replicate API. Mana yang termurah pada 10 prediksi/hari? Pada 10.000?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Custom silicon | "non-GPU chips" | Groq LPU, Cerebras WSE, SambaNova RDU — optimized for decode |
| FireAttention | "Fireworks engine" | Custom attention kernel; marketed at 4x lower latency than vLLM |
| Truss | "Baseten's format" | Model packaging manifest; dependencies + secrets + serving config |
| Per-token | "API pricing" | Charge by tokens consumed; pay for no idle |
| Per-minute | "dedicated pricing" | Charge by wall-clock GPU time; wins at high utilization |
| Per-prediction | "Replicate pricing" | Charge per model invocation; common for image/video |
| RayTurbo | "Anyscale engine" | Proprietary inference on Ray; competes with vLLM on Ray clusters |
| Batch tier | "50% off" | Non-interactive queue at reduced rate; common on Fireworks, OpenAI |
| Fine-tuned at base rate | "Fireworks LoRA" | Charge LoRA-served requests at base model's rate (differentiator) |

## Further Reading

- [Fireworks Pricing](https://fireworks.ai/pricing) — per-token rates, batch tier, GPU rental.
- [Baseten Pricing](https://www.baseten.co/pricing/) — per-minute rates, committed capacity, enterprise tiers.
- [Modal Pricing](https://modal.com/pricing) — per-second GPU rates and free tier.
- [Together AI Pricing](https://www.together.ai/pricing) — model catalog and per-token rates.
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo and managed Ray pricing.
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — comparative assessment.
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — vendor landscape.
