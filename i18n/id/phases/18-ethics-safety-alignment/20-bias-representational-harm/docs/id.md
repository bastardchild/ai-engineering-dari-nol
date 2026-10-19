# Bias dan Representational Harm dalam LLM

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed (Computational Linguistics 2024, arXiv:2309.00770). Survei fondasional 2024 yang membedakan representational harm (stereotype, erasure) dari allocational harm (distribusi sumber daya yang tidak merata) dan mengategorikan metrik evaluasi sebagai embedding-based, probability-based, atau generated-text-based. Empiris 2024-2025: An et al. (PNAS Nexus, Maret 2025) mengukur bias gender x race interseksional di lintas GPT-3.5 Turbo, GPT-4o, Gemini 1.5 Flash, Claude 3.5 Sonnet, Llama 3-70B pada evaluasi resume otomatis untuk 20 pekerjaan level pemula. WinoIdentity (COLM 2025, arXiv:2508.07111) memperkenalkan evaluasi keadilan berbasis-uncertainty untuk identitas interseksional. Yu & Ananiadou 2025 mengidentifikasi gender neuron di layer MLP; Ahsan & Wallace 2025 memakai SAE untuk mengungkap bias rasial klinis; Zhou et al. 2024 (UniBias) memanipulasi attention head untuk debiasing. Meta-kritik (arXiv:2508.11067): literatur 10 tahun secara tidak proporsional fokus pada bias gender-binari.

**Type:** Build
**Languages:** Python (stdlib, toy embedding-based bias probe)
**Prerequisites:** Phase 05 (word embeddings), Phase 18 · 01 (instruction following)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Definisikan representational vs allocational harm dan berikan satu contoh masing-masing dalam deployment LLM.
- Sebutkan tiga kategori metrik-evaluasi dari Gallegos et al. 2024 dan deskripsikan satu metrik dari tiap kategori.
- Deskripsikan interseksionalitas dan mengapa pengukuran keadilan berbasis-uncertainty milik WinoIdentity menangani kesenjangan pada evaluasi bias satu-sumbu.
- Deskripsikan dua pendekatan interpretabilitas-mekanistis terhadap bias (gender neuron, fitur SAE, manipulasi attention head).

## Masalahnya

Pelajaran sebelumnya membahas kerugian yang disengaja (jailbreak, scheming) dan tata kelola keselamatan. Bias adalah kerugian yang muncul tanpa niat — dari distribusi data training, dari framing prompt, dari pilihan desain yang menumpuk. Mengukur dan menguranginya adalah tantangan metodologis yang berbeda dari ketahanan advesarial.

## Konsepnya

### Representational vs allocational

- **Representational harm.** Stereotype, erasure, penggambaran yang merendahkan. Sebuah LLM yang menggambarkan perawat secara eksklusif perempuan sedang menghasilkan representational harm.
- **Allocational harm.** Hasil material yang tidak merata. Sebuah LLM yang menilai resume pelamar Hitam secara sistematis lebih rendah sedang menghasilkan allocational harm.

Keduanya tidak sama. Sebuah model bisa "representationally unbiased" (menghasilkan penggambaran beragam) sambil "allocationally biased" (memberi rekomendasi yang tidak merata). Evaluasi perlu mengukur keduanya.

### Tiga kategori metrik-evaluasi (Gallegos et al. 2024)

- **Embedding-based.** Tes gaya WEAT pada embedding pra-RLHF. Mengukur asosiasi statistis antara istilah identitas dan istilah atribut. Terbatas: mengukur representasi, bukan perilaku.
- **Probability-based.** Log-likelihood dari completion yang mengonfirmasi stereotype vs yang melanggar stereotype. Pengukuran sisi-decoder. Menangkap sebagian bias perilaku.
- **Generated-text-based.** Pengukuran tugas-hilir pada teks yang dihasilkan. Penilaian resume, penulisan rekomendasi, dialog. Paling valid secara ekologis; paling sulit direproduksi.

### Interseksionalitas

Evaluasi bias pada "gender" melewatkan bias yang hanya aktif pada pasangan (gender, race). An et al. 2025 menemukan GPT-4o menurunkan skor perempuan Hitam dalam penilaian resume lebih besar daripada laki-laki Hitam dan lebih besar daripada perempuan putih secara terpisah. Evaluasi satu-sumbu tidak bisa menangkap ini.

WinoIdentity (COLM 2025) memperkenalkan keadilan interseksional berbasis-uncertainty. Ia mengukur apakah uncertainty model atas hasil berbeda di lintas tuple identitas interseksional — bukan hanya prediksi titiknya. Ini menangkap kasus di mana model sama-sama salah di antara kelompok tapi lebih tidak pasti untuk sebagian, yang menghasilkan perilaku alokasi hilir yang berbeda.

### Pendekatan mekanistis

Karya interpretabilitas 2024-2025 membuka bias untuk intervensi mekanistis:

- **Gender neuron (Yu & Ananiadou 2025).** Neuron MLP spesifik berkorelasi dengan perilaku spesifik-gender. Meng-ablasi neuron ini menurunkan metrik gender-gap dengan biaya kemampuan yang terbatas.
- **Bias rasial klinis via SAE (Ahsan & Wallace 2025).** Fitur sparse autoencoder mendekomposisi representasi internal ke dimensi yang dapat diinterpretasi; fitur yang berkorelasi-rasial bisa diidentifikasi dan ditekan.
- **UniBias (Zhou et al. 2024).** Manipulasi attention head untuk debiasing zero-shot. Head tertentu memperkuat sensitivitas kelas-identitas; menolakkan atau menimbang-ulang head ini mengurangi bias tanpa fine-tuning.

### Meta-kritiknya

Tinjauan literatur 10 tahun (arXiv:2508.11067, 2025) menemukan bidang ini secara tidak proporsional fokus pada bias gender-binari. Sumbu lain — disabilitas, agama, status migrasi, identitas multibahasa — mendapat perhatian jauh lebih sedikit. Meta-kritik berargumen fokus sempit dapat merugikan kelompok terpinggirkan karena pengabaian: sebuah model yang sudah baik di-debias pada gender binari mungkin sangat bias pada dimensi yang tidak diperiksa siapa pun.

### Di mana ini cocok di Phase 18

Pelajaran 20-21 membahas bias dan keadilan secara formal. Pelajaran 22 membahas privasi. Pelajaran 23 membahas watermarking. Ini adalah lapisan kerugian-pengguna yang melengkapi lapisan penipuan/keselamatan yang lebih awal.

```figure
an-bias-two-harms
```

## Gunakan

`code/main.py` membangun bias probe mainan berbasis-embedding: mengukur jarak gaya WEAT antara istilah identitas dan istilah atribut dalam embedding ko-osi sederhana. Kamu bisa menyuntikkan sebuah bias dan mengamati metriknya aktif; menerapkan operasi debiasing sederhana dan mengamati pemulihan sebagian.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-bias-eval.md`. Diberi sebuah model card atau klaim keadilan, file itu mengaudit evaluasi di lintas tiga kategori metrik (embedding, probability, generated-text), cakupan interseksionalitas, dan mekanisme dari intervensi debiasing apa pun.

## Latihan

1. Jalankan `code/main.py`. Laporkan skor bias gaya WEAT sebelum dan sesudah langkah debiasing. Jelaskan mengapa metriknya tidak turun ke nol.

2. Perluas probe dengan tes interseksional: (gender, race) x (career, family). Laporkan skor bias lintas-sumbu.

3. Baca An et al. 2025 (PNAS Nexus). Identifikasi dua efek interseksional yang mereka laporkan dan akan dilewatkan oleh evaluasi gender satu-sumbu.

4. Yu & Ananiadou 2025 mengidentifikasi gender neuron. Sketsakan sebuah eksperimen falsifikasi yang bisa membedakan "neuron ini menyebabkan bias gender" dari "neuron ini berkorelasi dengan bias gender."

5. Meta-kritik berargumen bidang ini terlalu sempit fokus pada gender binari. Pilih satu sumbu yang kurang diteliti dan deskripsikan protokol pengukuran representational-harm untuknya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Representational harm | "stereotypes / erasure" | Biased portrayal of a group |
| Allocational harm | "unequal decisions" | Biased material outcome for a group |
| WEAT | "the embedding test" | Word Embedding Association Test; co-occurrence-based bias probe |
| Intersectionality | "combined identity effects" | Bias that emerges at the intersection of multiple identity axes |
| Gender neurons | "MLP bias neurons" | Specific neurons whose activations correlate with gender-specific behaviour |
| SAE feature | "interpretable dimension" | Sparse-autoencoder-identified feature; useful for mechanistic bias analysis |
| UniBias | "attention-head debiasing" | Zero-shot debiasing by reweighting attention heads |

## Bacaan Lanjutan

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — canonical survey
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) — five-model intersectional study
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — new benchmark
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612) — zero-shot debiasing
