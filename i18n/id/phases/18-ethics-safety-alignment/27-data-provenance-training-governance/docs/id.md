# Provenance Data dan Tata Kelola Data Training

> EU AI Act mewajibkan standar opt-out yang dapat dibaca mesin untuk GPAI pada Agustus 2025 (melalui pengecualian TDM dari EU Copyright Directive). California AB 2013 (ditandatangani 2024) — Transparansi data training Generative AI mewajibkan developer mempublikasikan ringkasan dataset dengan 12 field yang diwajibkan. Keselarasan DPA 2025 pada legitimate interest: Irish DPC (21 Mei 2025) menerima training LLM Meta pada konten publik pihak pertama dewasa UE/EEA dengan safeguards setelah opini EDPB; Cologne Higher Regional Court (23 Mei 2025) menolak injunction; Hamburg DPA menjatuhkan prosedur urgensi; UK ICO (23 September 2025) mengeluarkan respons regulasi positif terhadap safeguards pelatihan AI LinkedIn (transparansi, opt-out yang disederhanakan, perpanjangan jendela keberatan) dan terus memantau — bukan sebuah clearance formal. Brazilian ANPD (2 Juli 2024) menangguhkan pemrosesan Meta karena transparansi informasi yang tidak memadai; langkah pencegahan diangkat pada 30 Agustus 2024 setelah Meta mengirimkan rencana kepatuhan. Masalah irreversibilitas utama: kerangka cookie-consent dirancang untuk pelacakan real-time dan dapat dibatalkan; begitu data masuk ke bobot model, penghapusan secara bedah tidak mungkin — tidak ada hak GDPR right-to-erasure yang praktis untuk jaringan neural yang telah dilatih. Jendela kepatuhan ada pada waktu pengumpulan. Data Provenance Initiative (dataprovenance.org, Longpre, Mahari, Lee et al., "Consent in Crisis", Juli 2024): audit skala besar menunjukkan penurunan cepat dari data commons AI saat para publisher menambahkan pembatasan robots.txt.

**Type:** Learn
**Languages:** Python (stdlib, 12-field California AB 2013 scaffolding generator)
**Prerequisites:** Phase 18 · 24 (regulatory), Phase 18 · 26 (cards)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Jelaskan 12 field yang diwajibkan oleh California AB 2013 untuk transparansi data training Generative AI.
- Nyatakan posisi DPA 2025 tentang training LLM berbasis legitimate interest (Irish DPC, UK ICO, Hamburg, Cologne).
- Jelaskan masalah irreversibilitas: mengapa GDPR right-to-erasure tidak memiliki padanan praktis untuk jaringan neural yang telah dilatih.
- Nyatakan temuan "Consent in Crisis" dari Data Provenance Initiative.

## Masalahnya

Tata kelola data training adalah hulu dari setiap model card (Pelajaran 26) dan setiap kewajiban regulasi (Pelajaran 24). Pada 2024-2025, lanskap regulasi mengonsolidasikan pada tiga prinsip: infrastruktur opt-out, disclosure per-dataset, dan akomodasi legitimate interest untuk data yang tersedia publik. Penyedia yang tidak patuh pada waktu pengumpulan tidak dapat meremediasi di hilir.

## Konsepnya

### California AB 2013

Ditandatangani 2024. Dokumentasi harus diposting pada atau sebelum 1 Januari 2026 untuk sistem yang dirilis pada atau setelah 1 Januari 2022. Section 3111(a) mewajibkan developer mempublikasikan ringkasan tingkat tinggi dari dataset yang digunakan dalam training dengan 12 item undang-undang:
1. Sumber atau pemilik dataset.
2. Deskripsi bagaimana dataset memajukan tujuan yang dimaksudkan dari sistem AI.
3. Jumlah titik data dalam dataset (rentang umum dapat diterima; estimasi untuk dataset dinamis).
4. Deskripsi jenis titik data (tipe label untuk dataset berlabel; karakteristik umum untuk yang tak berlabel).
5. Apakah dataset menyertakan data yang dilindungi oleh hak cipta, merek dagang, atau paten, atau seluruhnya berada di domain publik.
6. Apakah dataset dibeli atau dilisensikan.
7. Apakah dataset menyertakan informasi pribadi (menurut Cal. Civ. Code §1798.140(v)).
8. Apakah dataset menyertakan informasi konsumen agregat (menurut Cal. Civ. Code §1798.140(b)).
9. Pembersihan, pemrosesan, atau modifikasi lain oleh developer, dengan tujuan yang dimaksudkan.
10. Periode waktu saat data dikumpulkan, dengan pemberitahuan jika pengumpulan sedang berlangsung.
11. Tanggal saat dataset pertama kali digunakan selama pengembangan.
12. Apakah sistem menggunakan atau terus-menerus menggunakan generasi data sintetis.

Item 12 (data sintetis) merupakan hal baru dibandingkan datasheets Gebru et al. 2018. Item 7 (informasi pribadi) memicu kewajiban Privacy Rights Act (CPRA). Undang-undang mengecualikan sistem keamanan/integritas, operasi pesawat terbang, dan sistem keamanan nasional khusus federal (Section 3111(b)).

### EU AI Act (Pelajaran 24) dan opt-out TDM

Pengecualian text-and-data-mining dari EU Copyright Directive mengizinkan training pada konten yang tersedia publik kecuali pemegang hak melakukan opt-out. EU AI Act GPAI Code of Practice bab Copyright mewajibkan para penyedia GPAI menghormati sinyal opt-out yang dapat dibaca mesin (robots.txt, klaim "No AI Training" dari C2PA, dll.).

### Konvergensi DPA 2025 pada legitimate interest

Irish DPC (21 Mei 2025): rencana Meta untuk melatih pada konten publik pihak pertama pengguna dewasa UE/EEA diterima dengan safeguards setelah opini EDPB. Cologne Higher Regional Court (23 Mei 2025) menolak injunction terhadap Meta: opt-out sudah cukup. Hamburg DPA menjatuhkan prosedur urgensi demi konsistensi lintas UE. UK ICO (23 September 2025) mengeluarkan respons regulasi positif — bukan clearance formal — terhadap keberlanjutan pelatihan AI oleh LinkedIn dengan safeguards serupa dan pemantauan berkelanjutan.

Prinsip konvergen: legitimate interest dapat membenarkan training pada konten publik pihak pertama dengan opt-out. Consent tidak diperlukan.

### Brazilian ANPD (Juni 2024)

Menangguhkan pemrosesan Meta atas data pengguna Brasil untuk pelatihan AI karena transparansi informasi yang tidak memadai. Hasil yang berbeda dari DPA EU — ANPD memprioritaskan transparansi di atas keterterimaan legitimate interest.

### Masalah irreversibilitas

Cookie-consent dirancang untuk pelacakan real-time dan dapat dibatalkan. Data training berbeda: begitu data masuk ke bobot model, penghapusan secara bedah tidak mungkin. Melatih ulang dari nol adalah satu-satunya remediasi lengkap, dan biayanya prohibitif mahal.

Remediasi parsial:
- **Unlearning.** Penghapusan aproksimasi; diukur oleh MIA (Pelajaran 22).
- **Lokalisasi berbasis influence function.** Identifikasi bobot yang paling dipengaruhi oleh data; perbarui secara selektif.
- **Fine-tune-suppression.** Latih model untuk menolak output yang diturunkan dari data.

Tidak ada yang menyelesaikan masalah ini sepenuhnya. Jendela kepatuhan ada pada waktu pengumpulan.

### Data Provenance Initiative

dataprovenance.org. Longpre, Mahari, Lee et al. "Consent in Crisis" (Juli 2024): audit skala besar atas commons data training AI. Temuan: para publisher menambahkan pembatasan robots.txt pada laju yang mempercepat. Commons yang terbuka untuk training menyusut dengan cepat. 2023 -> 2024 menyaksikan sekitar 25% dari sumber training teratas menambahkan pembatasan tertentu. Implikasi: ketersediaan data training masa depan bergantung pada paradigma akuisisi baru (lisensing, generasi sintetis, partisipasi yang diinsentivasi).

### Di mana ini cocok di Fase 18

Pelajaran 26 adalah dokumentasi tingkat model. Pelajaran 27 adalah tata kelola tingkat dataset. Bersama-sama mereka mendefinisikan lapisan transparansi. Pelajaran 28 memetakan ekosistem riset yang bekerja pada pertanyaan-pertanyaan ini.

```figure
an-provenance-oneway
```

## Gunakan

`code/main.py` menghasilkan sebuah kerangka ringkasan dataset 12 field yang patuh California AB 2013 untuk sebuah dataset mainan. Kamu dapat mengisi field-fieldnya dan mengamati field mana yang memicu kewajiban lanjutan privasi atau hak cipta.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-provenance-check.md`. Diberi sebuah dataset yang digunakan dalam training, ia memeriksa cakupan 12 field AB 2013, kepatuhan infrastruktur opt-out, keselarasan DPA, dan penilaian risiko irreversibilitas.

## Latihan

1. Jalankan `code/main.py`. Hasilkan ringkasan 12 field untuk sebuah dataset mainan dan identifikasi field mana yang kurang dispesifikasi.

2. Pengecualian TDM dari EU Copyright Directive dapat dibaca mesin. Usulkan format standar untuk sinyal opt-out dan bandingkan dengan robots.txt dan "No AI Training" dari C2PA.

3. Baca "Consent in Crisis" dari Data Provenance Initiative (Juli 2024). Jelaskan tiga kategori konten yang paling cepat dibatasi dan argumen satu konsekuensi ekonomi.

4. Keselarasan DPA 2025 menerima legitimate interest untuk training konten publik. Bangun sebuah skenario di mana legitimate interest tidak akan cukup dan identifikasi dasar hukum apa yang dibutuhkan sebuah penyedia sebagai gantinya.

5. Sketsa sebuah manifes provenance data training yang menyusun dengan field AB 2013 dan sebuah rantai provenance bertanda tangan C2PA untuk setiap dataset. Identifikasi satu hambatan teknis dan satu hambatan hukum.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| AB 2013 | "the California law" | Generative AI training-data transparency; 12 mandated fields |
| TDM exception | "text-and-data-mining" | EU Copyright Directive training-data exception with opt-out |
| Legitimate interest | "the EU basis" | GDPR Article 6 basis that may justify training on public content |
| Opt-out signal | "machine-readable no-train" | robots.txt, C2PA "No AI Training," TDM.Reservation |
| Irreversibility | "cannot un-train" | Data in model weights is not surgically removable |
| Unlearning | "approximate removal" | Post-training interventions to reduce model dependence on specific data |
| Consent in Crisis | "the DPI audit" | July 2024 finding of accelerating robots.txt restrictions |

## Bacaan Lanjutan

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — Generative AI training-data transparency law
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — Copyright chapter
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — DPI audit
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — regulatory context
