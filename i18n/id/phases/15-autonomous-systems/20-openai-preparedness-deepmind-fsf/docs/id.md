# Preparedness Framework OpenAI dan Frontier Safety Framework DeepMind

> Preparedness Framework v2 OpenAI (April 2025) memperkenalkan Research Categories — Long-range Autonomy, Sandbagging, Autonomous Replication and Adaptation, Undermining Safeguards — berbeda dari Tracked Categories. Tracked Categories memicu Capabilities Report plus Safeguards Report yang direviu Safety Advisory Group. FSF v3 DeepMind (September 2025, dengan Tracked Capability Levels ditambahkan 17 April 2026) melipat otonomi ke domain ML R&D dan Cyber (otonomi ML R&D level 1 = mengotomatiskan penuh pipeline AI R&D dengan biaya kompetitif vs manusia + tool AI). FSF v3 secara eksplisit menangani deceptive alignment lewat pemantauan otomatis atas penyalahgunaan instrumental-reasoning. Catatan yang jujur: Research Categories dalam PF v2 (termasuk Long-range Autonomy) tidak otomatis memicu mitigasi; bahasa kebijakannya "potensial." DeepMind sendiri berkata pemantauan otomatis "tidak akan tetap cukup jangka-panjang" jika penalaran instrumental menguat.

**Type:** Learn
**Languages:** Python (stdlib, three-framework decision-table diff tool)
**Prerequisites:** Phase 15 · 19 (Anthropic RSP)
**Time:** ~45 minutes

## Masalahnya

Pelajaran 19 membaca kebijakan scaling Anthropic secara cermat. Pelajaran ini melengkapi gambaran dengan membaca milik OpenAI dan DeepMind. Ketiga dokumen adalah artefak sepupu yang menangani pertanyaan yang sama — kapan lab frontier harus berhenti atau menggerbangi model — dan mereka konvergen pada sedikit kategori dan divergen di tempat spesifik yang penting.

Konvergensi: ketiganya melabeli otonomi jarak-jauh sebagai kelas kapabilitas yang layak dilacak. Ketiganya mengakui perilaku menipu (alignment faking, sandbagging) sebagai kelas risiko spesifik. Ketiganya punya badan reviu internal. Divergensi: OpenAI membagi kategori menjadi "Tracked" (mitigasi wajib) dan "Research" (tanpa pemicu otomatis). DeepMind melipat otonomi ke dua domain ketimbang menamainya terpisah. Lab menamai Tracked vs Research, atau Critical vs Moderate, atau Tier-1 vs Tier-2; konsekuensi operasional dari bucket tempat kapabilitas berada berbeda lintas lab.

Membacanya bersama adalah latihan yang berguna. Kapabilitas yang sama bisa menjadi "mitigasi wajib" di Anthropic, "dipantau tetapi tidak memicu" di OpenAI, dan "dilacak dalam domain spesifik" di DeepMind. Postur kebijakan penting.

## Konsepnya

### Preparedness Framework v2 OpenAI (April 2025)

Struktur:

- **Tracked Categories**: memicu Capabilities Report (apa yang bisa dilakukan model) plus Safeguards Report (mitigasi apa yang ada). Direviu Safety Advisory Group sebelum deployment.
- **Research Categories**: kapabilitas model yang dilacak lab tetapi belum dikomitmenkan pada mitigasi spesifik. Termasuk Long-range Autonomy, Sandbagging, Autonomous Replication and Adaptation, Undermining Safeguards.

Bingkai kunci dalam v2: Research Categories tidak otomatis memicu mitigasi. Bahasa kebijakannya adalah mitigasi "potensial". Ini delta spesifik yang harus diperhatikan: lokasi kategori dalam Tracked vs Research adalah engsel antara "digerbangi" dan "diamati."

### Frontier Safety Framework v3 DeepMind (September 2025; Tracked Capability Levels ditambahkan 17 April 2026)

Struktur:

- **Critical Capability Levels (CCLs)**: ambang kapabilitas lintas lima domain: Cyber, Bio, ML R&D, CBRN, Autonomy (dilipat ke ML R&D dan Cyber).
- **Tracked Capability Levels**: granularitas tambahan yang ditambahkan April 2026. Contoh konkret: otonomi ML R&D level 1 = mengotomatiskan penuh pipeline AI R&D dengan biaya kompetitif vs manusia + tool AI.
- **Pemantauan deceptive alignment**: komitmen eksplisit pada pemantauan otomatis atas penyalahgunaan instrumental-reasoning.

Bingkai otonomi berbeda dari milik OpenAI. DeepMind tidak menyimpan "Autonomy" sebagai domain tingkat-atas; ia dilipat ke domain tempat otonomi akan menyebabkan bahaya (ML R&D dan Cyber). Argumennya adalah otonomi tanpa domain adalah kapabilitas tanpa risiko; argumen kontranya adalah otonomi lintas domain adalah meta-risiko yang harus dinamai kerangka.

### Apa yang dikonvergensikan ketiganya

- Internal Safety Advisory Group (bernama SAG Anthropic, SAG OpenAI, komite internal DeepMind). Reviu sebelum deployment untuk model berkapabilitas-tinggi.
- Penyebutan eksplisit deceptive alignment / alignment faking sebagai kelas risiko.
- Artefak tetap pada kadens yang dideklarasikan (Anthropic: Frontier Safety Roadmap, Risk Report; OpenAI: Capabilities dan Safeguards Report; DeepMind: siklus pembaruan FSF).
- Pengakuan bahwa pertahanan hanya-pemantauan punya langit-langit. DeepMind eksplisit: "pemantauan otomatis tidak akan tetap cukup jangka-panjang."

### Di mana mereka divergen

- **Anthropic**: komitmen pause dihapus dalam v3.0; ambang AI R&D-4 adalah gerbang bernama berikutnya.
- **OpenAI**: pemisahan Tracked vs Research; Research Categories (termasuk Long-range Autonomy) tidak otomatis menggerbangi.
- **DeepMind**: otonomi dilipat ke domain lain; Tracked Capability Levels menambah granularitas pada April 2026.

### Sandbagging: kapabilitas spesifik yang menyulitkan ketiganya

Sandbagging (model yang strategis berkinerja di bawah pada evaluasi) berada dalam Research Categories OpenAI. RSP v3.0 Anthropic menanganinya lewat kesenjangan konteks-evaluasi (Pelajaran 1). DeepMind menanganinya lewat pemantauan deceptive alignment dalam FSF v3.

Jika model melakukan sandbagging pada evaluasi, ambang kapabilitas setiap kerangka diremehkan. Kerangka bekerja hanya jika pengukurannya bekerja. Itu sebabnya pengukuran eksternal (Pelajaran 21, METR) dan evaluasi adversarial perlu di samping penilaian-diri lab.

### Keterampilan membaca-kebijakan

- Temukan: setiap kapabilitas yang kamu pedulikan harus bisa ditemukan dalam kebijakan. Jika tidak bisa ditemukan, kebijakan tidak mencakupnya.
- Klasifikasikan: apakah ia Tracked (memicu mitigasi) atau Research (dilacak tetapi tidak memicu)? OpenAI menamai ini; Anthropic dan DeepMind punya padanannya sendiri.
- Kadens: apakah kebijakan diperbarui pada jadwal yang dideklarasikan, atau hanya setelah event spesifik? Kadens yang dideklarasikan lebih kuat.
- Independensi: apakah reviu eksternal wajib atau opsional? Anthropic bermitra dengan Apollo dan US AI Safety Institute; OpenAI dengan METR; DeepMind terutama dengan SAG internal.

```figure
a5-tracked-vs-research
```

## Gunakan

`code/main.py` mengimplementasikan tool diff tabel-keputusan kecil. Dengan kapabilitas (otonomi, deceptive alignment, otomasi R&D, uplift siber, dst.), ia mengeluarkan bagaimana setiap dari tiga kebijakan mengklasifikasikan kapabilitas, dan mitigasi apa yang terpicu. Ia alat bantu baca, bukan tool kebijakan.

## Rilis

`outputs/skill-cross-policy-diff.md` menghasilkan perbandingan lintas-kebijakan untuk kapabilitas spesifik, memakai tiga kerangka sebagai referensi.

## Latihan

1. Jalankan `code/main.py`. Konfirmasi keluaran tool diff cocok dengan kebijakan untuk setidaknya dua kapabilitas yang bisa kamu verifikasi terhadap dokumen sumber.

2. Baca Preparedness Framework v2 OpenAI penuh. Identifikasi setiap Research Category. Untuk masing-masing, tulis satu kalimat tentang mengapa ia berada di Research ketimbang Tracked.

3. Baca FSF v3 DeepMind penuh, plus pembaruan Tracked Capability Levels April 2026. Identifikasi kriteria evaluasi spesifik otonomi ML R&D level 1. Bagaimana kamu akan mengukurnya secara eksternal?

4. Sandbagging berada dalam Research Categories OpenAI. Rancang evaluasi yang akan memaksa model yang sandbagging mengungkapkan kapabilitas aktualnya. Rujuk diskusi gaming konteks-eval Pelajaran 1.

5. Bandingkan ketiga kebijakan pada kapabilitas spesifik (pilihanmu). Sebutkan klasifikasi kebijakan mana yang menurutmu paling ketat dan mana paling longgar. Justifikasi dengan teks sumber.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Preparedness Framework | "OpenAI's scaling policy" | PF v2 (April 2025); Tracked vs Research categories |
| Tracked Category | "Mandatory mitigation" | Triggers Capabilities + Safeguards Reports; SAG review |
| Research Category | "Monitored only" | Tracked but no automatic mitigation; includes Long-range Autonomy |
| Frontier Safety Framework | "DeepMind's scaling policy" | FSF v3 (Sept 2025) + Tracked Capability Levels (Apr 2026) |
| CCL | "Critical Capability Level" | DeepMind threshold per domain (Cyber, Bio, ML R&D, CBRN) |
| ML R&D autonomy level 1 | "R&D automation" | Fully automate AI R&D pipeline at competitive cost |
| Sandbagging | "Strategic underperformance" | Model underperforms on evals; in OpenAI Research Categories |
| Instrumental reasoning | "Means-ends reasoning" | Reasoning about how to achieve goals; target of DeepMind monitoring |

## Further Reading

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 announcement.
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — full document.
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 announcement.
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — Tracked Capability Levels addition.
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — example of an FSF-format Risk Report.
