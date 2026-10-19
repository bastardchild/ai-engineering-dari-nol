# Sistem Moderasi — OpenAI, Perspective, Llama Guard

> Sistem moderasi produksi mengoperasionalkan kebijakan keselamatan yang didefinisikan di Pelajaran 12-16. OpenAI Moderation API: `omni-moderation-latest` (2024) dibangun di atas GPT-4o mengklasifikasi teks + gambar dalam satu panggilan; 42% lebih baik pada set uji multilingual dibanding versi sebelumnya; skema respons mengembalikan 13 boolean kategori — harassment, harassment/threatening, hate, hate/threatening, illicit, illicit/violent, self-harm, self-harm/intent, self-harm/instructions, sexual, sexual/minors, violence, violence/graphic; gratis untuk sebagian besar developer. Pola berlapis: Input moderation (pra-generasi), Output moderation (pasca-generasi), Custom moderation (aturan domain). Panggilan paralel async menyembunyikan latensi; respons placeholder saat di-flag. Llama Guard 3/4 (Pelajaran 16): 14 bahaya MLCommons, Code Interpreter Abuse, 8 bahasa (v3), multi-gambar (v4). Perspective API (Google Jigsaw): penilaian toksisitas yang mendahului gelombang LLM-as-moderator; terutama toksisitas single-dimension dengan varian severe-toxicity/insult/profanity; baseline untuk riset moderasi konten. Depresiasi: Azure Content Moderator didepresiasi Februari 2024, dipensiunkan Februari 2027, digantikan oleh Azure AI Content Safety.

**Type:** Build
**Languages:** Python (stdlib, three-layer moderation harness)
**Prerequisites:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan taksonomi kategori dari OpenAI Moderation API dan bagaimana ia berbeda dari set MLCommons Llama Guard 3.
- Jelaskan pola tiga lapisan moderasi (input, output, custom) dan sebutkan satu mode gagal dari masing-masing.
- Jelaskan posisi Perspective API sebagai baseline era pra-LLM dan mengapa ia tetap digunakan dalam riset.
- Nyatakan timeline depresiasi Azure.

## Masalahnya

Pelajaran 12-16 menjelaskan serangan dan tooling pertahanan. Pelajaran 29 membahas sistem moderasi yang telah di-deploy yang mengoperasionalkan pertahanan di permukaan tempat pengguna menyentuh produk. Pola tiga lapisan adalah konfigurasi default tahun 2026.

## Konsepnya

### OpenAI Moderation API

`omni-moderation-latest` (2024). Dibangun di atas GPT-4o. Mengklasifikasi teks + gambar dalam satu panggilan. Gratis untuk sebagian besar developer.

Kategori (13 boolean dalam skema respons):
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

Dukungan multimodal berlaku untuk `violence`, `self-harm`, dan `sexual` tapi bukan `sexual/minors`; sisanya hanya teks.

Untuk harness kode di `code/main.py` kita meruntuhkan sub-kategori `/threatening`, `/intent`, `/instructions`, dan `/graphic` ke induk top-level mereka demi kesederhanaan pedagogis. Kode produksi harus memakai skema lengkap 13 kategori.

42% lebih baik pada set uji multilingual dibanding endpoint moderasi generasi sebelumnya. Skor per kategori; aplikasi menetapkan threshold.

### Llama Guard 3/4

Dibahas di Pelajaran 16. 14 kategori bahaya MLCommons (diorganisasi berbeda dari 13 boolean skema respons OpenAI). Mendukung 8 bahasa (v3). Llama Guard 4 (April 2025) native multimodal, 12B.

Taksonomi OpenAI dan Llama Guard tumpang tindih tapi berbeda. OpenAI memiliki "illicit" sebagai kategori luas; Llama Guard memiliki "violent crimes" dan "non-violent crimes" secara terpisah. Deployment memilih berdasarkan kecocokan taksonomi kebijakan mereka.

### Perspective API (Google Jigsaw)

Sistem penilaian toksisitas yang mendahului gelombang LLM-as-moderator (pra-2020). Kategori: TOXICITY, SEVERE_TOXICITY, INSULT, PROFANITY, THREAT, IDENTITY_ATTACK. Skor utama single-dimension (TOXICITY) dengan varian sub-dimensi.

Luas digunakan sebagai baseline riset moderasi konten karena API-nya stabil, terdokumentasi, dan memiliki bertahun-tahun data kalibrasi. Untuk use case modern yang berdekatan dengan LLM, Llama Guard atau OpenAI Moderation umumnya lebih cocok.

### Pola tiga lapisan

1. **Input moderation.** Klasifikasi prompt pengguna sebelum generasi. Tolak jika di-flag. Latensi: satu panggilan classifier.
2. **Output moderation.** Klasifikasi output model sebelum pengiriman. Ganti dengan sebuah refusal jika di-flag. Latensi: satu panggilan classifier setelah generasi.
3. **Custom moderation.** Aturan spesifik domain (regex, allowlist, kebijakan bisnis). Berjalan di input atau output.

Ketiga lapisan berurutan by design: input moderation harus selesai sebelum generasi, dan output moderation berjalan setelah generasi. Paralelisme berlaku di dalam sebuah lapisan — menjalankan beberapa classifier (mis., OpenAI Moderation + Llama Guard + Perspective) secara bersamaan pada teks yang sama menyembunyikan latensi per classifier. Sebagai optimasi opsional, sebuah respons placeholder ("one moment, checking...") dapat ditampilkan selagi input moderation menyelesaikan dan streaming token-1 ditunda. Perilaku flag dapat dikonfigurasi: tolak, sanitize, eskalasi ke review manusia.

### Mode gagal

- **Hanya input.** Tidak menangkap halusinasi output (serangan encoding Pelajaran 12-14 melewati classifier input).
- **Hanya output.** Mengizinkan input apa pun mencapai model; meningkatkan biaya; memamerkan penalaran internal ke penyerang.
- **Hanya custom.** Tidak robust lintas kategori; regex rapuh.

Berlapis adalah default. Belt-and-suspenders.

### Depresiasi Azure

Azure Content Moderator: didepresiasi Februari 2024, dipensiunkan Februari 2027. Digantikan oleh Azure AI Content Safety, yang berbasis LLM dan terintegrasi dengan Azure OpenAI. Migrasi adalah sebuah proyek level field 2024-2027 untuk deployment Azure.

### Di mana ini cocok di Fase 18

Pelajaran 16 membahas tooling moderasi dalam konteks red-team. Pelajaran 29 membahas moderasi yang telah di-deploy. Pelajaran 30 menutup dengan bukti kemampuan dual-use saat ini.

```figure
an-moderation-layers
```

## Gunakan

`code/main.py` membangun sebuah harness moderasi tiga lapisan: input moderator (keyword + skor kategori), output moderator (classifier yang sama pada output), custom moderator (aturan domain). Kamu dapat menjalankan input melaluinya dan mengamati lapisan mana yang menangkap apa.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-moderation-stack.md`. Diberi sebuah deployment, ia merekomendasikan konfigurasi stack moderasi: classifier mana di input, yang mana di output, aturan custom mana, dan judge apa untuk kasus tepi.

## Latihan

1. Jalankan `code/main.py`. Jalankan sebuah input yang jinak, borderline, dan berbahaya melalui ketiga lapisan. Laporkan lapisan mana yang menyala untuk masing-masing.

2. Perluas harness dengan penilaian toksisitas bergaya Perspective API pada kategori tertentu. Bandingkan perilaku threshold-nya dengan skor kategori.

3. Baca dokumen OpenAI Moderation API dan daftar kategori Llama Guard 3. Petakan setiap kategori OpenAI ke kategori Llama Guard terdekat. Identifikasi tiga kategori yang tidak terpetakan dengan bersih.

4. Rancang sebuah stack moderasi untuk deployment asisten kode (mis., GitHub Copilot). Identifikasi kategori yang paling dan paling tidak relevan dan usulkan aturan custom.

5. Azure Content Moderator dipensiunkan pada Februari 2027. Rencanakan migrasi ke Azure AI Content Safety. Identifikasi elemen dengan risiko tertinggi dari migrasi tersebut.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OpenAI Moderation | "omni-moderation-latest" | GPT-4o-based 13-category (text) classifier with partial multimodal support |
| Perspective API | "Google Jigsaw toxicity" | Pre-LLM-era toxicity scoring baseline |
| Llama Guard | "MLCommons 14-category" | Meta's hazard classifier (v3: 8B text, 8 langs; v4: 12B multimodal) |
| Input moderation | "pre-generation filter" | Classifier on user prompt before model call |
| Output moderation | "post-generation filter" | Classifier on model output before delivery |
| Custom moderation | "domain rules" | Deployment-specific rules (regex, allowlist, policy) |
| Layered moderation | "all three layers" | Standard production deployment pattern |

## Bacaan Lanjutan

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation endpoint
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard repo
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — toxicity scoring
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure replacement
