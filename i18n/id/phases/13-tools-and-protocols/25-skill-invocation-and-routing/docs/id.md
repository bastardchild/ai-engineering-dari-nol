# Invokasi Skill dan Routing

> Invokasi adalah keputusan otoritas yang diikuti keputusan relevansi. Deskripsi yang baik membantu model memilih; kebijakan yang baik memutuskan apakah pilihan itu diizinkan.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 24 (Skill Discovery and Progressive Disclosure)
**Time:** ~105 minutes

## Learning Objectives

- Membedakan invokasi eksplisit pengguna, invokasi implisit model, invokasi aplikasi, dan invokasi skill-ke-skill.
- Memodelkan visibilitas manusia dan kelayakan model sebagai dimensi kebijakan independen.
- Menulis deskripsi routing dengan pemicu positif dan batas near-miss.
- Memisahkan kelayakan, seleksi, aktivasi, pengikatan argumen, dan eksekusi dalam trace dan test.
- Mengadaptasi field invokasi spesifik runtime tanpa menyajikannya sebagai frontmatter portabel.

## Masalahnya

Kamu memasang skill `database-migration`. Pengguna bisa menjalankannya dengan nama, tetapi model juga melihat deskripsinya dan memilihnya ketika seseorang bertanya tentang pertanyaan database umum. Skill itu kemudian mengusulkan perubahan skema untuk tugas yang hanya butuh penjelasan.

Kamu menambahkan `user-invocable: false`, berharap memblokir orang menjalankannya secara manual. Di runtime lain, field itu diabaikan. Kamu menambahkan `disable-model-invocation: true`, berharap skill hilang sepenuhnya. Di runtime yang memahaminya, pengguna masih bisa memanggilnya secara eksplisit.

Tidak ada yang salah dengan nama field-nya. Modelnya yang salah. "Pengguna bisa melihatnya," "model bisa memilihnya," "aplikasi bisa memuat-ulangnya," dan "tool di dalamnya bisa mengeksekusi" adalah fakta terpisah. Satu boolean bernama `invocable` tidak bisa mengekspresikannya.

Routing punya mode kegagalan kedua. Jika deskripsi kabur, beberapa skill menjadi masuk akal. Jika deskripsi dibunteli kata kunci, tugas yang tidak relevan memicunya. Katalog adalah antarmuka probabilistik: cukup ringkas untuk muat, cukup spesifik untuk merutekan.

## Konsepnya

### Lima kanal bisa memulai siklus hidup

| Aktor | Bentuk invokasi | Penggunaan tipikal | Risiko utama |
|---|---|---|---|
| Pengguna manusia | Menyebut skill di UI atau prompt | Seleksi alur kerja disengaja | Pengguna mengharapkan ketersediaan atau otoritas yang tidak diberikan host |
| Model atau agen otomatis | Memilih entri katalog dari konteks tugas | Prosedur ahli otomatis | Routing false-positive |
| Aplikasi | Mengaktifkan atau memuat-ulang skill melalui kode runtime | Alur kerja produk tetap | Coupling tersembunyi ke satu host |
| Skill lain atau subagent | Meminta skill tertentu sebagai dependensi alur kerja | Komposisi | Siklus, dependensi hilang, atau kebocoran konteks |
| Harness evaluasi | Mengaktifkan skill tertentu di bawah skenario tetap | Pengukuran berulang | Menguji skill sambil tidak sengaja melewati kebijakan produksi yang sedang dipelajari |

Spesifikasi Agent Skills portabel mendefinisikan paket. Ia tidak menstandarkan satu UI slash-command universal, flag routing implisit, API aplikasi, atau siklus hidup subagent.

### Lima tahap invokasi

```figure
skill-invocation-stages
```

Gunakan kata-kata ini secara presisi:

- **Eligible** berarti kebijakan mengizinkan aktor ini meminta skill.
- **Selected** berarti pengguna menyebutnya atau router menilainya relevan.
- **Activated** berarti instruksinya masuk ke konteks kerja.
- **Executing** berarti agen mulai kerja model atau tool di bawah instruksi itu.
- **Completed** berarti output memenuhi pemeriksaan keberhasilan independen.

Trace yang hanya mencatat `skill_used=true` menyembunyikan batas tempat kegagalan terjadi.

### Invokasi manusia dan model membentuk matriks 2x2

| Manusia bisa memanggil | Model bisa memanggil | Mode | Contoh yang sesuai |
|:---:|:---:|---|---|
| Ya | Ya | Bersama | Penjelasan kode, perencanaan test, tinjauan dokumentasi |
| Ya | Tidak | Hanya-manusia | Persiapan publikasi, ekspor penagihan, rencana pembersihan destruktif |
| Tidak | Ya | Hanya-model | Panduan gaya internal, referensi domain, prosedur dukungan otomatis |
| Tidak | Tidak | Dinonaktifkan atau hanya-aplikasi | Rollout bertahap, paket usang, preload programatik |

Matriks itu adalah model kebijakan, bukan YAML standar.

Satu host saat ini memakai `disable-model-invocation: true` untuk baris hanya-manusia dan `user-invocable: false` untuk baris hanya-model. Defaultnya keduanya. Host lain memakai `agents/openai.yaml` dengan `allow_implicit_invocation: false` untuk mempertahankan invokasi eksplisit sambil menonaktifkan seleksi implisit. Itu adapter runtime. Host yang tidak dikenal bisa mengabaikannya.

Detail yang membingungkan itu penting: `user-invocable: false` tidak berarti "model tidak bisa menggunakan ini." Ia menghapus invokasi langsung pengguna di host yang mendefinisikannya. `disable-model-invocation: true` tidak berarti "skill dinonaktifkan." Ia menghapus seleksi yang diinisiasi model sambil mempertahankan akses eksplisit pengguna.

### Invokasi eksplisit berbasis identitas

Invokasi eksplisit menyuplai identitas secara langsung:

```text
/release-readiness v2.4.0
```

atau:

```text
release-readiness check v2.4.0 without publishing
```

Antarmuka Codex terkini mendokumentasikan `/skills` untuk seleksi dan nama skill biasa dalam permintaan untuk invokasi eksplisit. Claude Code mendokumentasikan `/skill-name` dan ekspansi argumen spesifik-host. Sintaks persis, visibilitas menu, aturan quoting, dan ekspansi variabel milik host.

Permintaan eksplisit tetap melewati kebijakan. Menyebut skill seharusnya tidak melewati izin yang hilang, batasan workspace, gerbang persetujuan, atau isolasi runtime.

### Invokasi implisit berbasis deskripsi

Untuk routing implisit, model pada awalnya melihat metadata katalog daripada isi penuh. Deskripsi karena itu adalah antarmuka routing skill.

Lemah:

```yaml
description: Helps with releases.
```

Terlalu luas:

```yaml
description: Use for release, version, package, build, deploy, publish, tag, changelog, GitHub, CI, or software tasks.
```

Berbatas:

```yaml
description: Inspect an already prepared release candidate and produce a readiness report. Use when the user asks whether a version, tag, package, or image is ready to publish; do not use for ordinary build failures or feature development.
```

Versi berbatas berisi:

1. **Kapabilitas:** inspeksi kandidat yang sudah disiapkan.
2. **Output:** laporan kesiapan.
3. **Batas positif:** bertanya apakah artifact rilis sudah siap.
4. **Batas negatif:** build dan pengembangan biasa di luar cakupan.

Batas negatif berguna ketika dua skill berdekatan berbagi kosakata. Mereka bukan pengganti untuk evaluasi near-miss.

### Routing adalah klasifikasi dengan opsi abstain

Untuk skill `s` dan permintaan `x`, bayangkan skor router:

```text
score(s, x) = capability_match + trigger_match + context_match - exclusion_match - ambiguity_penalty
```

Skoring persis mungkin keputusan LLM daripada aritmetika. Prinsip rekayasanya tetap berlaku: seleksi harus mengalahkan ambang dan skill pesaing. Ketika bukti lemah, abstain.

```figure
skill-routing-abstention
```

Untuk skill berdampak-tinggi, routing implisit mungkin tidak tepat bahkan dengan deskripsi kuat. Gunakan kebijakan hanya-manusia ketika biaya false-positive melebihi kenyamanan seleksi otomatis.

### Kelayakan harus mendahului peringkat

Jangan menilai setiap skill yang ditemukan, memilih kecocokan terkuat, dan memeriksa kebijakan satu skill itu setelahnya. Kocokan teratas yang diblokir akan secara salah mencegah kandidat eligible berskor lebih rendah dipertimbangkan.

Gunakan urutan ini untuk routing implisit:

1. Saring skill yang ditemukan berdasarkan aktor yang meminta dan adapter host aktif.
2. Nilai hanya kandidat eligible.
3. Pilih kecocokan eligible terkuat jika ia melewati ambang dan aturan ambiguitas.
4. Abstain ketika tidak ada kandidat eligible atau tidak ada skor eligible yang cukup kuat.

Misalkan `incident-triage` berskor `0.80` tetapi ekstensi hostnya menonaktifkan invokasi model. `incident-review` berskor `0.55` dan mengizinkan invokasi model. Router seharusnya mengevaluasi `incident-review` sebagai kandidat eligible terbaik. Ia seharusnya tidak memilih `incident-triage`, menolaknya, dan berhenti.

Urutan ini juga menjaga perubahan kebijakan tidak mengubah makna skor relevansi. Kelayakan mendefinisikan himpunan seleksi. Relevansi memeringkat himpunan itu.

### Evaluasi routing butuh near-miss

Kasus positif membuktikan recall:

```json
{"prompt":"Is version 2.4.0 ready to publish?","expected":"release-readiness"}
```

Negatif yang jelas membuktikan precision dasar:

```json
{"prompt":"Explain rotary position embeddings.","expected":null}
```

Near-miss mengekspos kualitas batas:

```json
{"prompt":"Why did today's package build fail?","expected":"build-diagnostics"}
```

Near-miss membagi `package` dan `build` dengan skill rilis tetapi termasuk di tempat lain. Himpunan routing yang hanya terdiri dari positif obvious dan negatif tidak relevan akan melebih-lebihkan kualitas.

### Argumen punya tiga representasi

Argumen invokasi melewati beberapa batas:

```figure
skill-argument-boundaries
```

Di setiap batas, pertahankan niat tanpa memperlakukan teks sebagai kode.

- Parser host memutuskan sintaks perintah dan quoting.
- Skill menerima teks terikat atau variabel menurut aturan host.
- Instruksi memvalidasi nilai wajib dan default.
- Panggilan tool mengubah nilai ke schema bertipe dan memvalidasinya ulang.

Jangan menginterpolasi argumen mentah ke dalam perintah shell. Lebih suka skrip yang dipanggil dengan vektor argumen atau tool MCP bertipe.

### Invokasi aplikasi adalah orkestrasi eksplisit

Produk bisa mengaktifkan skill karena alur kerjanya sudah mengetahui jenis tugas. Sebagai contoh, layanan tinjauan pull-request bisa memuat-ulang `pull-request-risk-review` setelah pengguna menekan Review.

Ini menghilangkan ketidakpastian routing tetapi menciptakan ketergantungan pada API runtime. Jaga adapter itu di luar isi portabel:

```figure
skill-host-adapter
```

Skill seharusnya tetap dapat dipahami ketika dibuka oleh klien patuh yang berbeda.

### Invokasi skill-ke-skill adalah edge seperti tool

Misalkan `release-readiness` meminta `security-change-review` ketika file dependensi berubah.

Pemanggil seharusnya menyuplai:

- identitas skill target;
- tugas berbatas dan path artifact;
- kontrak respons yang diharapkan;
- alasan invokasi;
- fallback jika tidak tersedia;
- kedalaman maksimum atau aturan siklus.

```json
{
  "target_skill": "security-change-review",
  "task": "Review dependency changes in the candidate diff",
  "inputs": ["artifacts/release.diff"],
  "expected": "risk-report.json",
  "max_depth": 2
}
```

Skill kedua tidak ditempel secara membabi buta ke skill pertama. Host memutuskan cara mengaktifkannya dan apakah ia berbagi konteks, berjalan di fork, atau kembali melalui hasil tool.

### Siklus hidup konteks spesifik-host

Setelah aktivasi, isi skill bisa tetap di percakapan, diringkas selama kompresi, atau berjalan di konteks terdelegasi. Izin tool bisa bertahan satu turn sementara instruksi bertahan lebih lama. Subagent bisa menerima skill tanpa seluruh riwayat induknya.

Jangan menulis skill yang bergantung pada asumsi umur tak terlihat. Taruh output yang tahan lama di file atau state bertipe, buat re-entry aman, dan nyatakan apa yang harus dimuat-ulang setelah interupsi.

```markdown
On resume, read `artifacts/release-readiness.json` if it exists.
Revalidate the candidate commit before continuing.
Do not repeat an external write whose idempotency key is already recorded.
```

## Bangun Sendiri

`code/main.py` mengimplementasikan kebijakan dan routing sebagai adapter terpisah.

Model meliputi:

- `Actor` untuk pemanggil manusia, model, agen otomatis, aplikasi, skill, dan harness;
- `SkillMetadata` untuk identitas routing;
- `InvocationPolicy` untuk matriks manusia/model;
- `InvocationRequest` dan `InvocationDecision` untuk input dan hasil yang dapat dilacak;
- `CorePolicyAdapter` untuk perilaku portabel tanpa ekstensi host;
- `ExtensionPolicyAdapter` untuk field runtime yang dikenali;
- `build_invocation_matrix(policy)` untuk tampilan 2x2;
- `route_request(skills, request, adapter)` untuk penyaringan kelayakan sebelum peringkat relevansi, seleksi, dan penolakan.

Jalankan:

```bash
cd phases/13-tools-and-protocols/25-skill-invocation-and-routing
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Demo mencetak satu matriks dan keputusan untuk kanal manusia-eksplisit, model-implisit, agen-otomatis, aplikasi, komposisi-skill, dan harness. Hasil adapter-ekstensinya menunjukkan kocokan leksikal teratas yang diblokir dihapus sebelum alternatif eligible diperingkat. Ia juga menyertakan allowlist nama-persis. Tidak dibutuhkan API model. Router deterministik ada untuk membuat batas kebijakan dapat diinspeksi, bukan untuk mengklaim bahwa pencocokan leksikal mereproduksi routing model produksi.

### Mengapa adapter inti dan ekstensi terpisah

Jika satu parser memberi makna pada setiap field frontmatter yang diamati, ia diam-diam mempromosikan konvensi runtime menjadi standar palsu. Adapter terpisah memaksa pemanggil menamai semantik host mana yang aktif.

`CorePolicyAdapter` hanya memakai kebijakan yang disuplai aplikasi. `ExtensionPolicyAdapter` mengenali set eksplisit field host dan mencatat field mana yang mengubah keputusan.

## Gunakan

Tulis kontrak invokasi sebelum memublikasikan skill:

```yaml
actors:
  human: allow
  model: deny
  application: allow
  skill: deny
explicit_name: release-readiness
arguments:
  candidate: required
  publish: fixed_false
ambiguity: ask_user
missing_dependency: stop
context:
  durable_state: artifacts/release-readiness.json
  max_composition_depth: 2
```

Kontrak ini adalah dokumentasi desain untuk adapter dan test. Ia bukan frontmatter `SKILL.md` portabel kecuali standar secara eksplisit mengadopsinya.

## Rilis

Pelajaran ini menghasilkan bundel `skill-invocation-router`. Ia menyertakan referensi model-invokasi, contoh kebijakan host, dan CLI non-eksekusi yang mengevaluasi satu permintaan manusia, model, agen-otomatis, aplikasi, komposisi-skill, atau harness dan mengembalikan keputusan JSON dengan kanal, adapter, skor, dan alasan.

CLI satu-permintaan itu adalah probe kebijakan, bukan evaluasi pemicu penuh. Gunakan desain positif berlabel dan near-miss di Pelajaran 27 untuk menghitung confusion count, precision, recall, dan stabilitas run berulang.

## Latihan

1. Buat semua empat baris matriks manusia/model dan tulis satu use case sah untuk masing-masing.
2. Tambahkan aktivasi hanya-aplikasi ke `CorePolicyAdapter`. Buktikan bahwa pemanggil manusia dan model tetap ditolak.
3. Tulis sepuluh near-miss untuk skill deployment. Setiap prompt harus berbagi kosakata dengan skill sambil termasuk alur kerja berbeda.
4. Tambahkan margin ambiguitas di antara dua skor routing teratas. Kembalikan `ask` ketika margin terlalu kecil.
5. Tambahkan kedalaman komposisi maksimum ke permintaan skill-ke-skill dan deteksi siklus dua-skill.
6. Jalankan himpunan berlabel yang sama melalui adapter inti dan ekstensi. Jelaskan setiap keputusan yang berubah.

## Istilah Kunci

| Istilah | Apa yang orang katakan | Apa artinya sebenarnya |
|---|---|---|
| Invokasi eksplisit | "Slash command" | Sebuah aktor menyuplai identitas skill secara langsung, tunduk pada kebijakan |
| Invokasi implisit | "Model yang memilih" | Seorang router memilih dari metadata katalog eligible berdasarkan konteks tugas |
| User-invocable | "Manusia bisa menggunakannya" | Properti menu atau invokasi-langsung spesifik-host, bukan field inti |
| Model-invocable | "Agen bisa menggunakannya" | Kelayakan untuk seleksi model implisit di bawah kebijakan host |
| Adapter invokasi | "Parser frontmatter" | Kode yang memetakan field dan API host ke model kebijakan yang dideklarasikan |
| Near-miss | "Hard negative" | Permintaan non-memicu yang menyerupai input yang dimaksud skill |
| Abstention | "Tidak ada skill dipilih" | Hasil routing yang disengaja ketika bukti tidak ada atau ambigu |

## Bacaan Lanjutan

- [Mengoptimalkan deskripsi skill](https://agentskills.io/skill-creation/optimizing-descriptions) untuk pemicu positif, spesifisitas, dan evaluasi.
- [Mengevaluasi skill](https://agentskills.io/skill-creation/evaluating-skills) untuk desain evaluasi pemicu dan output.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) untuk kontrol invokasi eksplisit dan implisit Codex terkini.
- [Claude Code skills](https://code.claude.com/docs/en/skills) untuk `user-invocable`, `disable-model-invocation`, argumen, dan konteks terdelegasi satu host.
