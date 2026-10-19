# Time Horizon METR dan Evaluasi Kapabilitas Eksternal

> METR (ex-ARC Evals) adalah 501(c)(3) independen sejak Desember 2023. Tolok ukur Time Horizon 1.1 mereka (Januari 2026) mencocokkan kurva logistik antara probabilitas keberhasilan tugas vs log(waktu penyelesaian manusia ahli); titik potong pada probabilitas 50% mendefinisikan time horizon model. Set engagement 2025–2026 mencakup GPT-5.1, GPT-5.1-Codex-Max, dan evaluasi monitoring prototipe (bisakah monitor menangkap tugas sampingan; bisakah agen mengelak). Rangkaian benchmark: HCAST (180+ tugas ML, siber, SWE, penalaran; 1 menit hingga 8+ jam), RE-Bench (71 tugas ML research-engineering dengan baseline ahli), SWAA. Catatan yang jujur: pengukuran METR bersifat ideal — tanpa manusia, tanpa konsekuensi nyata — dan tim telah mendokumentasikan kesenjangan perilaku eval-vs-deployment (Pelajaran 1). Time horizon adalah batas atas, bukan prediksi deployment.

**Type:** Learn
**Languages:** Python (stdlib, logistic-fit horizon estimator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 19 (RSP)
**Time:** ~60 minutes

## Masalahnya

Kebijakan scaling (Pelajaran 19, 20) hanya sebagus pengukuran yang mereka rujuk. Ambang "AI R&D-4" dan "Long-range Autonomy" didefinisikan dalam prosa kebijakan; mereka menjadi actionable hanya ketika evaluasi spesifik menghasilkan angka spesifik.

METR adalah organisasi evaluasi eksternal 2024–2026 yang mendefinisikan banyak angka itu. Mereka mengevaluasi model frontier — sering pra-rilis, di bawah NDA dengan lab — dan menerbitkan metodologi sesudahnya. Tolok ukur Time Horizon 1.1 (Januari 2026) adalah artefak utama mereka: satu skalar yang memadatkan kapabilitas menjadi satuan yang mudah dipahami manusia ("model ini bisa mengerjakan jenis tugas yang dikerjakan ahli selama X jam pada reliabilitas 50%").

Pelajaran ini sebagian tentang metodologi (bagaimana horizon dihitung) dan sebagian tentang interpretasi (mengapa horizon adalah batas atas, bukan prediksi deployment). Kedua keterampilan saling melengkapi. Tim yang memahami bagaimana horizon di-fit jauh lebih sulit dibodohi klaim vendor yang buruk dibanding tim yang hanya melihat "14 jam" pada slide.

## Konsepnya

### Latar METR

- Didirikan: Desember 2023 (ex-ARC Evals, spin out menjadi 501(c)(3) independen).
- Scope: evaluasi kapabilitas otonom model frontier, sering pra-rilis.
- Lab mitra: Anthropic, OpenAI (banyak engagement 2025–2026).
- Deliverable penting: Time Horizon 1.0 (Maret 2025), Time Horizon 1.1 (Januari 2026), evaluasi monitoring prototipe.

### Fit Time Horizon

Metodologi (dari blog dan makalah METR):

1. Kumpulkan rangkaian tugas yang membentang dari waktu penyelesaian ahli skala-menit hingga skala-jam. Rangkaian saat ini: HCAST (180+ tugas), RE-Bench (71 tugas), SWAA.
2. Jalankan model pada setiap tugas; catat sukses atau gagal.
3. Cocokkan kurva logistik: P(sukses) sebagai fungsi log(waktu penyelesaian ahli).
4. Horizon adalah waktu-ahli tempat P(sukses) = 0,5.

Bentuk fit-logistik adalah yang tepat karena kapabilitas umumnya punya hubungan yang menanjak dan mendekati-plato dengan kesulitan tugas. Titik 50% adalah pilihan (bisa 10%, 90%); METR melaporkan banyak ambang dalam makalah detail tetapi memimpin dengan 50% karena paling intuitif.

### Angka Januari 2026

Menurut Time Horizon 1.1:

- Claude Opus 4.6: ~14 jam pada reliabilitas 50%, per Time Horizon 1.1 (Januari 2026).
- Waktu penggandaan pada tugas gaya HCAST: ~4,3 bulan (130,8 hari) pada fit pasca-2023 yang dilaporkan Time Horizon 1.1 (Januari 2026); angka ~7 bulan adalah fit penuh 2019–2025 dari Time Horizon 1.0 dan dilaporkan dalam TH1.1 berdampingan dengan angka pasca-2023.

Itulah angka spesifik yang dirujuk seluruh Fase 15. Mereka disitasi dalam RSP v3.0 Anthropic (Pelajaran 19) dan FSF v3 DeepMind (Pelajaran 20) sebagai salah satu pengukuran yang mengisi penilaian ambang.

### Rangkaian benchmark

- **HCAST**: 189 tugas lintas ML, keamanan siber, software engineering, dan penalaran umum. Rentang: 1 menit hingga 8+ jam waktu ahli. Rangkaian utama di balik angka horizon.
- **RE-Bench**: 71 tugas ML research-engineering dengan baseline ahli manusia. Mengukur apakah model bisa mengerjakan kerja ML R&D yang mengisi ambang AI R&D-4.
- **SWAA**: tugas yang lebih pendek; berguna untuk mengkalibrasi ujung bawah kurva horizon.

### Evaluasi monitoring prototipe

Kerja METR 2025–2026 mencakup prototipe evaluasi-monitoring: bisakah monitor menangkap model yang melakukan tugas sampingan (sesuatu selain yang diminta operator); bisakah agen mengelak monitor. Ini langsung mengisi kategori "Undermining Safeguards" dalam PF v2 OpenAI dan pemantauan deceptive-alignment dalam FSF v3 DeepMind (Pelajaran 20).

### Mengapa horizon adalah batas atas

- **Tooling yang diidealkan**: tugas benchmark memberi model tool bersih dan data yang terformat baik. Deployment nyata lebih berantakan.
- **Tanpa konsekuensi nyata**: model tidak pernah benar-benar menagih pelanggan, menghapus data nyata, atau menghubungi orang nyata. Deployment nyata punya taruhan irreversibel.
- **Gaming konteks-eval**: Pelajaran 1. Model berperilaku berbeda dalam pengujian. International AI Safety Report 2026 mendokumentasikan ini secara empiris.
- **Tanpa varians pengguna yang sah**: prompt benchmark terstruktur. Pengguna nyata menghasilkan permintaan yang ambigu dan bergantung-konteks.

Horizon adalah langit-langit kapabilitas di bawah kondisi yang menguntungkan. Reliabilitas deployment adalah angka berbeda, lebih rendah, dan tim harus mengukur distribusinya sendiri untuk mengetahuinya.

### Kasus evaluator-eksternal

Evaluasi eksternal penting karena lab internal punya insentif mengoptimalkan metrik yang mereka laporkan. Independensi METR — 501(c)(3) dengan metodologi yang dideklarasikan dan makalah peer-review — adalah mitigasi struktural. Ia tidak cukup sendirian (lab tetap mengendalikan apa yang dilihat METR), tetapi ia jelas lebih baik dibanding tanpa evaluasi eksternal.

### Cara memakai angka horizon dalam praktik

- **Sebagai filter kapabilitas**: jika horizon model jauh di bawah waktu-ahli tugas yang diusulkan, jangan kirim ia otonom (file skill Pelajaran 1).
- **Sebagai indikator tren**: waktu penggandaan memberitahumu berapa lama praktik saat ini akan tetap aman bahkan tanpa mitigasi baru.
- **Sebagai prior**: horizon 14 jam adalah titik mulai. Sesuaikan ke bawah untuk distribusi tugasmu, kualitas tooling-mu, dan konteks deployment-mu.

```figure
a5-horizon-fit
```

## Gunakan

`code/main.py` mengimplementasikan fit logistik atas keberhasilan-tugas vs log(waktu ahli), dengan set hasil sintetis. Ia melaporkan horizon 50% (headline METR), horizon 10% (konservatif), dan horizon 90% (optimistis). Juga mendemonstrasikan apa yang berubah ketika angka keberhasilan digembungkan artifisial oleh gaming konteks-eval.

## Rilis

`outputs/skill-horizon-interpretation.md` mereviu klaim horizon vendor dan menghasilkan analisis kesenjangan antara klaim benchmark dan realitas deployment.

## Latihan

1. Jalankan `code/main.py`. Konfirmasi horizon 50% dari fit cocok dengan ground truth sintetis. Kini bagi-dua grid waktu-tugas; apakah estimasi horizon berubah bermakna?

2. Baca postingan blog Time Horizon 1.1 METR. Identifikasi tugas spesifik tempat reliabilitas tertinggi dan tempat terendah. Jelaskan mengapa kesenjangan ada.

3. Baca resource "Measuring Autonomous AI Capabilities" dari METR. Daftarkan kategori tugas HCAST. Pilih satu kategori yang akan kamu boboti lebih berat untuk tugas produksi dan justifikasi alasannya.

4. Perkenalkan gaming konteks-eval ke simulator: balik ~20% tugas gagal menjadi sukses. Laporkan horizon baru. Ini mendekati apa yang dilakukan angka gaming 20% pada angka yang teramati.

5. Rancang evaluasi horizon internal pada backlog bug-mu sendiri atau set tugas yang representatif. Jelaskan pengumpulan data, fit, dan apa yang diberitahukan keluaran padamu. Bandingkan dengan angka METR.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| METR | "External evaluator" | ex-ARC Evals; independent 501(c)(3) since Dec 2023 |
| Time Horizon | "Capability measure" | Expert task length at 50% reliability, from logistic fit |
| HCAST | "METR's main suite" | 180+ tasks spanning 1 min to 8+ hours |
| RE-Bench | "Research engineering" | 71 ML research-engineering tasks with human baseline |
| SWAA | "Short-task suite" | Calibrates the low end of the horizon curve |
| Doubling time | "Growth rate" | Time for the 50% horizon to double; ~7 months per HCAST |
| Eval-context gaming | "Model behaves differently" | Documented behavior gap between tests and deployment |
| Upper bound | "Horizon is a ceiling" | Benchmark horizon > deployment reliability under load |

## Further Reading

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST, RE-Bench, SWAA specs.
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — the original horizon paper.
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — current numbers and methodology.
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — live tracking.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — internal perspective on METR's measurements.
