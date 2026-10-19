# Capstone 84 — Evaluasi Penolakan

> Ke-helpfulness pada prompt jinak dan penolakan pada prompt berbahaya adalah dua metrik, bukan satu. Ukur keduanya.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 18 safety lessons, Phase 19 Track A lessons 25-29
**Time:** ~90 min

## Masalahnya

Sebuah pass keamanan pada asisten gagal dengan dua cara yang berlawanan. Model menolak hal yang seharusnya ia jawab (over-refusal), dan model menjawab hal yang seharusnya ia tolak (under-refusal). Keduanya adalah bug. Tim yang hanya mengukur laju penolakan pada prompt berbahaya merilis model yang menolak membantu pekerjaan rumah kimia. Tim yang hanya mengukur helpfulness merilis model yang menjelaskan cara berbuat jahat. Set metrik yang tepat memperlakukan asisten sebagai klasifier biner atas keamanan prompt, di mana kebijakan yang diuji menentukan sisi mana dari batas keputusan setiap prompt berada.

Tantangannya adalah LLM nyata adalah target yang bergerak selama pengembangan. Perubahan pada system prompt, perubahan temperature, atau perubahan tool menggeser kedua metrik secara bersamaan. Kamu membutuhkan evaluator berbasis fixture dengan mock LLM deterministik di bawahnya, sehingga perubahan kode menggeser metrik karena alasan yang diketahui. Mock LLM di pelajaran ini adalah sebuah tabel dispatch kecil: pattern, policy, output. Harness di pelajaran 87 mencolokkan model nyata nanti dengan mengganti mock dengan panggilan HTTP. Framework metriknya tidak berubah.

Pelajaran ini membangun framework evaluasi. Empat angka yang dihasilkannya: laju under-refusal, laju over-refusal, kalibrasi (apakah confidence yang dinyatakan model cocok dengan akurasinya), dan rincian per kategori memakai taksonomi pelajaran 82.

## Konsepnya

Sebuah evaluasi penolakan punya empat objek: satu set prompt dengan label `safe`/`unsafe`, mock LLM dengan policy berskrip, klasifier penolakan atas output model (regex atas frasa seperti `I cannot`, `I will not`, `as a responsible`), dan agregator metrik.

```mermaid
flowchart LR
  P[labeled prompt set] --> M[mock LLM]
  M --> R[refusal classifier]
  R --> A[aggregator]
  A --> UR[under-refusal rate]
  A --> OR[over-refusal rate]
  A --> CAL[calibration ECE]
  A --> CAT[per-category breakdown]
```

Policy mock LLM sengaja dibuat sederhana supaya tesnya bisa diaudit. `MockPolicyStrict` bawaan menolak pada prompt apa pun yang cocok dengan salah satu dari daftar pola regex terlarang dan menjawab selain itu. `MockPolicyOverCautious` menolak pada set pola yang jauh lebih luas sehingga sengaja over-refuse. `MockPolicyLeaky` hanya menolak pada kasus yang paling jelas sehingga sengaja under-refuse. Ketiga policy itu adalah bug berskrip yang harus dideteksi oleh framework.

Under-refusal: model menjawab (tanpa frasa penolakan) pada prompt yang ditandai `unsafe` oleh set label. Over-refusal: model menolak (frasa penolakan ada) pada prompt yang ditandai `safe` oleh set label. Harness juga menghitung `accuracy = (TP + TN) / total` di mana TP adalah "prompt berbahaya ditolak dengan benar" dan TN adalah "prompt aman dijawab dengan benar".

Kalibrasi memakai Expected Calibration Error (ECE) atas confidence yang dinyatakan model. Mock LLM secara opsional memancarkan token `confidence:0.X` di outputnya; harness menguraikannya. ECE membagi prompt ke dalam bin berdasarkan confidence per sepuluh, menghitung akurasi per bin, dan merata-ratakan `|conf - accuracy|` yang ditimbang dengan ukuran bin. Model yang mengatakan `confidence:0.9` tapi benar 60% dari waktu punya ECE sekitar 0.3 pada bin itu. ECE independen dari over/under refusal karena ia mengukur apakah model tahu kapan ia benar.

Rincian per kategori menggabungkan prompt berlabel dengan artefak taksonomi dari pelajaran 82. Setiap prompt berbahaya membawa label kategori (salah satu dari enam). Harness melaporkan laju under-refusal per kategori sehingga tim bisa melihat, misalnya, bahwa model menangani `instruction-override` dengan baik tapi terpeleset pada `multi-turn-ramp`.

```figure
ci-refusal-quadrant
```

## Bangun Sendiri

`code/mock_llm.py` mendefinisikan tiga policy. Setiap policy adalah callable yang memetakan prompt ke string respons. Respons menyelipkan confidence model sebagai `[conf=0.X]`. `code/prompts.py` adalah korpus berlabel: 25 prompt berbahaya (diambil dari taksonomi pelajaran 82 berdasarkan id) plus 30 prompt aman (permintaan jinak sehari-hari, tanpa tumpang tindih dengan set benign pelajaran 83 supaya kedua evaluasi tetap independen).

`code/main.py` menjalankan evaluator. Klasifier penolakan adalah regex atas frasa penolakan. Agregator mengembalikan dict dengan `under_refusal`, `over_refusal`, `accuracy`, `ece`, dan `per_category_under_refusal`. Runner menyapu ketiga policy mock dan menulis laporan perbandingan.

## Gunakan

`python3 main.py`. Demo mencetak tabel yang membandingkan ketiga policy, menulis `outputs/refusal_eval_report.json`, dan mengonfirmasi bahwa `MockPolicyOverCautious` punya over-refusal tertinggi dan `MockPolicyLeaky` punya under-refusal tertinggi. Policy strict berada di antara keduanya; itu adalah baseline regresi.

## Rilis

`outputs/skill-refusal-evaluation.md` mendokumentasikan definisi metrik supaya pengguna hilir dari laporan itu tidak salah membaca angkanya.

## Latihan

1. Tambahkan policy mock keempat yang menolak berdasarkan panjang prompt. Konfirmasi bahwa under-refusal naik pada serangan berkode (yang cenderung pendek).
2. Ganti ECE dengan reliability curve dan plot satu curve per policy. Catat bin mana yang over-confident.
3. Tambahkan daftar prompt aman per kategori (role-play jinak, instruksi jinak tentang konteks sebelumnya). Hitung over-refusal per kategori dan periksa apakah role-play menarik penolakan palsu terbanyak.

## Istilah Kunci

| Term | Common usage | Precise meaning |
|---|---|---|
| under-refusal | the model is helpful | the model answered a prompt labeled unsafe |
| over-refusal | the model is safe | the model refused a prompt labeled safe |
| calibration | the model is humble | the gap between stated confidence and observed accuracy, summarized by Expected Calibration Error |
| accuracy | quality | (TP + TN) / total for the safe/unsafe binary decision |
| per-category breakdown | a chart | under-refusal rate joined against the lesson 82 taxonomy categories |

## Bacaan Lanjutan

Pelajaran 85 (klasifier output) dan pelajaran 87 (gerbang end to end) mengonsumsi framework metrik dari pelajaran ini.
