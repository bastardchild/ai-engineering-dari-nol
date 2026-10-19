# Platform LLM Terkelola — Bedrock, Vertex AI, Azure OpenAI

> Tiga hyperscaler, tiga strategi berbeda. AWS Bedrock adalah marketplace model — Claude, Llama, Titan, Stability, Cohere di balik satu API. Azure OpenAI adalah kemitraan eksklusif OpenAI ditambah Provisioned Throughput Units (PTU) untuk kapasitas khusus. Vertex AI mengutamakan Gemini dengan cerita long-context dan multimodal terbaik. Pada 2026 Artificial Analysis mengukur Azure OpenAI sekitar 50 ms median dan Bedrock sekitar 75 ms pada setara Llama 3.1 405B — PTU menjelaskan selisihnya karena kapasitas khusus mengalahkan on-demand bersama. Aturan keputusannya bukan "mana yang tercepat" melainkan "katalog model dan permukaan FinOps mana yang cocok dengan produkku." Pelajaran ini mengajarimu memilih dengan tradeoff yang ditulis eksplisit, bukan sekadar firasat.

**Type:** Learn
**Languages:** Python (stdlib, toy cost-and-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Sebutkan tiga strategi platform (marketplace vs eksklusif vs mengutamakan Gemini) dan pasangkan masing-masing dengan kasus penggunaan produk.
- Jelaskan apa yang kamu dapat dari Provisioned Throughput Units (PTU) di Azure OpenAI dan mengapa Bedrock on-demand biasanya terbaca ~25 ms lebih lambat pada skala 405B.
- Gambarkan permukaan atribusi FinOps untuk setiap platform (Bedrock Application Inference Profiles vs proyek-per-tim di Vertex vs scope Azure + reservasi PTU).
- Tuliskan kebijakan "minimal dua provider" dan jelaskan mengapa lock-in satu vendor adalah kesalahan mahal pada 2026.

## Masalahnya

Kamu memilih Claude 3.7 Sonnet untuk produkmu. Sekarang kamu perlu menyajikannya. Kamu bisa memanggil API Anthropic langsung, atau memanggilnya lewat AWS Bedrock, atau lewat gateway. API langsung adalah yang paling sederhana; Bedrock menambahkan BAA, VPC endpoint, IAM, dan atribusi CloudWatch. Gateway menambahkan failover, penagihan terpadu, dan rate limit lintas provider.

Pertanyaan yang lebih dalam adalah katalog. Kalau kamu butuh Claude dan Llama dan Gemini dalam satu produk, kamu tidak bisa membelinya semua dari satu tempat kecuali tempat itu adalah Bedrock plus Vertex plus Azure OpenAI sekaligus. Para hyperscaler tidak saling menggantikan — masing-masing membuat taruhan berbeda tentang siapa pemilik lapisan model.

Pelajaran ini memetakan tiga taruhan tersebut, selisih latensi, selisih FinOps, dan risiko lock-in.

## Konsepnya

### Tiga strategi

**AWS Bedrock** — marketplace. Claude (Anthropic), Llama (Meta), Titan (first-party AWS), Stability (gambar), Cohere (embedding), Mistral, plus sub-katalog gambar dan embedding. Satu API, satu permukaan IAM, satu ekspor CloudWatch. Taruhan Bedrock adalah pelanggan lebih menginginkan opsionalitas daripada satu model tunggal.

**Azure OpenAI** — kemitraan eksklusif. Kamu mendapat GPT-4 / 4o / 5 / o-series, DALL·E, Whisper, dan fine-tuning model OpenAI di datacenter Azure. Tidak ada model non-OpenAI dalam katalog "Azure OpenAI Service" — semuanya masuk ke Azure AI Foundry (produk terpisah). Taruhan Azure adalah OpenAI tetap menjadi frontier dan pelanggan menginginkan kontrol enterprise atas relasi spesifik itu.

**Vertex AI** — Gemini dulu, sisanya kemudian. Gemini 1.5 / 2.0 / 2.5 Flash dan Pro, plus Model Garden (pihak ketiga). Taruhan Vertex adalah multimodal long-context — konteks Gemini 1M-token adalah pembedanya.

### Selisih latensi pada skala besar

Artificial Analysis menjalankan benchmark berkelanjutan. Pada deployment Llama 3.1 405B yang setara (on-demand bersama), median latensi token-pertama Azure OpenAI sekitar 50 ms; Bedrock sekitar 75 ms. Selisih ini bukan kegagalan AWS — melainkan perbedaan model kapasitas. Azure menjual PTU (Provisioned Throughput Units), yang mencadangkan kapasitas GPU untuk tenant-mu. Bedrock punya padanannya (Provisioned Throughput) tetapi mulai sekitar $21/jam per unit, dan kebanyakan pelanggan tetap memakai on-demand bersama.

Kapasitas bersama on-demand bersaing dengan trafik pelanggan lain. Kapasitas khusus tidak. Kalau SLA produkmu adalah TTFT < 100 ms pada P99, kamu membeli PTU di Azure, membeli Bedrock Provisioned Throughput, atau menerima varians default.

### Ekonomi Provisioned Throughput

Azure PTU: blok komputasi inferensi yang dicadangkan. Hemat hingga ~70% dibanding on-demand untuk workload yang terprediksi. Biaya tetap per jam terlepas dari trafik — kamu membayar reservasi bahkan saat idle. Titik impas biasanya sekitar 40-60% utilisasi berkelanjutan.

Bedrock Provisioned Throughput: $21-$50 per jam tergantung model dan region. Matematikanya mirip — titik impas sekitar setengah utilisasi puncak. Komitmen bulanan diwajibkan.

Kapasitas provisioned Vertex dijual per SKU Gemini; harga bervariasi menurut model dan region dan tidak terlalu dipublikasikan.

### Permukaan FinOps — pembeda sesungguhnya

**Bedrock Application Inference Profiles** adalah atribusi paling rapi di marketplace. Beri tag pada profil dengan `team`, `product`, `feature`; arahkan semua pemanggilan model melaluinya; CloudWatch merinci biaya per profil tanpa pasca-pemrosesan. Ditambahkan 2025, masih menjadi yang paling granular di antara hyperscaler.

**Atribusi Vertex** memakai proyek-per-tim plus label-di-mana-mana. Kamu memodelkan tiap tim sebagai proyek GCP, menempelkan label pada setiap resource, dan memakai BigQuery Billing Export + DataStudio untuk rollup. Lebih banyak kerja, tetapi BigQuery memberimu SQL bebas atas data biaya.

**Azure** mengandalkan scope subscription/resource-group plus tag, dengan reservasi PTU sebagai objek biaya kelas satu. Tag diwarisi dari resource group, bukan dari request, sehingga atribusi per-request membutuhkan metrik kustom Application Insights atau gateway yang membubuhkan header.

Polanya: Bedrock paling rapi secara native, Vertex paling fleksibel lewat BigQuery, Azure paling buram kecuali kamu menginstrumentasinya.

### Lock-in adalah risiko 2026

Komitmen ke satu hyperscaler masih wajar saat satu model mendominasi. Pada 2026 frontier bergerak tiap bulan — Claude 3.7 satu kuartal, Gemini 2.5 kuartal berikutnya, GPT-5 kuartal setelahnya. Terkunci ke satu platform berarti kamu kehilangan akses ke dua pertiga frontier.

Pola yang dipakai tim yang berhasil: minimal dua provider untuk setiap panggilan LLM yang kritis bagi produk. Bedrock plus Azure OpenAI adalah pasangan umum — Claude dari satu sisi, GPT dari sisi lain, failover di antara keduanya, gateway yang sama. Kenaikan biaya bisa diabaikan karena gateway merutekan secara optimal; kenaikan ketersediaan saat outage (seperti insiden Azure OpenAI Januari 2025, outage AWS us-east-1) bersifat menentukan.

### Residensi data, BAA, dan industri teregulasi

Bedrock: BAA di kebanyakan region; VPC endpoint; guardrail. Default umum fintech.
Azure OpenAI: HIPAA, SOC 2, ISO 27001; residensi data UE; default enterprise-teregulasi.
Vertex: HIPAA, GDPR, residensi data per region; stack compliance Google Cloud.

Ketiganya memenuhi checkbox dasar. Perbedaannya ada pada kebijakan retensi data, cara log ditangani, dan apakah pemantauan penyalahgunaan membaca trafikmu (default opt-in di kebanyakan; opt-out tersedia untuk enterprise).

### Angka yang perlu kamu ingat

- Median TTFT Azure OpenAI pada setara Llama 3.1 405B: ~50 ms (dengan PTU).
- Median TTFT Bedrock on-demand: ~75 ms.
- Bedrock Provisioned Throughput: $21-$50/jam per unit.
- Titik impas PTU Azure: ~40-60% utilisasi berkelanjutan.
- Hemat PTU vs on-demand pada utilisasi tinggi: hingga 70%.

```figure
i4-platform-lanes
```

## Gunakan

`code/main.py` membandingkan tiga platform pada workload sintetis — ia memodelkan ekonomi on-demand vs PTU, varians TTFT, dan fidelitas atribusi biaya. Jalankan untuk melihat di mana PTU menguntungkan dan di mana keluasan model marketplace mengalahkan selisih TTFT.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-managed-platform-picker.md`. Dengan profil workload (model yang dibutuhkan, SLA TTFT, volume harian, syarat compliance), ia merekomendasikan platform utama, cadangan, dan rencana instrumentasi FinOps.

## Latihan

1. Jalankan `code/main.py`. Pada utilisasi berkelanjutan berapa PTU Azure mengalahkan on-demand untuk model kelas 70B? Hitung titik impasnya dan bandingkan dengan rentang 40-60% yang diiklankan.
2. Produkmu butuh Claude 3.7 Sonnet dan GPT-4o. Rancang deployment dua provider — mana yang ke hyperscaler mana, gateway apa yang di depan, bagaimana kebijakan failover-nya?
3. Pelanggan healthcare teregulasi mewajibkan BAA, residensi data US-East, dan TTFT P99 di bawah 100 ms. Pilih platform dan justifikasi dengan tiga fitur spesifik.
4. Kamu mendapati tagihan Bedrock naik 4x bulan ini tanpa perubahan trafik. Tanpa Application Inference Profiles, bagaimana kamu menemukan pelakunya? Dengan profil, berapa lama waktu yang dibutuhkan?
5. Baca halaman harga Azure OpenAI dan Bedrock. Untuk workload Claude 100M-token/bulan, mana yang lebih murah — API Anthropic langsung, Bedrock on-demand, atau Bedrock Provisioned Throughput?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Bedrock | "AWS LLM service" | Model marketplace across Claude, Llama, Titan, Mistral, Cohere |
| Azure OpenAI | "Azure's ChatGPT" | Exclusive OpenAI models in Azure datacenters with enterprise controls |
| Vertex AI | "Google's LLM" | Gemini-first platform with Model Garden for third-party models |
| PTU | "dedicated capacity" | Provisioned Throughput Unit — reserved inference GPUs, priced per hour |
| Application Inference Profile | "Bedrock tagging" | Per-product cost/usage profile with tags, CloudWatch-native |
| Model Garden | "Vertex catalog" | Vertex AI's third-party model section, separate from Gemini |
| Two-provider minimum | "LLM redundancy" | Policy of running every critical LLM path across ≥2 hyperscalers |
| BAA | "HIPAA paperwork" | Business Associate Agreement; required for PHI; provided by all three |
| Abuse monitoring | "the log watcher" | Provider-side safety scan on prompts/outputs; opt-out in enterprise |

## Further Reading

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — authoritative rate card and Provisioned Throughput pricing.
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/azure-openai/) — PTU economics and rate cards.
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini tiers and Model Garden surcharges.
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — continuous latency and throughput benchmarks across providers.
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — enterprise decision framework.
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — attribution mechanics side-by-side.
