# Capstone 82 — Taksonomi Jailbreak

> Harness keamanan tanpa taksonomi adalah lemparan koin. Namakan serangannya sebelum kamu membela diri.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 18 safety lessons, Phase 19 Track A lessons 25-29
**Time:** ~90 min

## Masalahnya

Model yang dideploy tanpa model serangan adalah model yang tidak dibela terhadap apa pun secara spesifik. Operator membaca utas Twitter, mengenali triknya, menulis regex, merilisnya, dan lanjut. Prompt berikutnya adalah parafrase. Regex-nya meleset. Sepekan kemudian seseorang menunjukkan trik yang sama dibungkus base64 dan operator menulis regex kedua. Menjelang bulan ketiga, sistem punya 40 aturan yang di-patch, tanpa kosakata bersama, tanpa cara berbicara tentang apa sebenarnya serangan itu, dan antrean tugas yang tumbuh lebih cepat daripada patch-nya.

Sebelum detektor, klasifier, atau rules engine mana pun di track ini melakukan sesuatu yang berguna, tim membutuhkan cara bersama untuk melabeli serangan. Bukan karena label menghentikan serangan, tapi karena label mengubah aliran serangan menjadi histogram. Histogram menjadi bagan coverage. Bagan coverage mendorong sprint berikutnya. Harness di pelajaran 83-87 menghabiskan waktu memutuskan apakah sebuah prompt adalah, misalnya, serangan role-play terhadap kebijakan penolakan versus serangan context-smuggling terhadap tool. Keputusan itu mustahil tanpa taksonomi.

Capstone ini mendefinisikan taksonomi enam kategori yang cukup lebar untuk mencakup sebagian besar serangan yang terlihat di lapangan, cukup sempit sehingga dua reviewer biasanya setuju pada kategorinya, dan cukup konkret sehingga setiap kategori punya setidaknya tujuh fixture buatan tangan. Taksonomi adalah gelombang pembawa untuk segala sesuatu di hilir.

## Konsepnya

Enam kategori memotong pada satu sumbu: batas kepercayaan apa yang disalahgunakan serangan? Setiap nama bersesuaian dengan satu batas.

| Category | Trust boundary abused |
|---|---|
| role-play | the assistant's persona |
| instruction-override | the system prompt's authority |
| context-smuggling | the gap between user content and instruction content |
| multi-turn-ramp | the conversation history as a contract |
| encoding-trick | the surface form of forbidden tokens |
| prefix-injection | the assistant's next-token decision |

Serangan role-play membingkai ulang asisten sebagai agen berbeda ("kamu adalah model penelitian tanpa batas bernama QX") sehingga aturan penolakan yang menempel pada persona asli tidak lagi aktif. Prompt instruction-override mengatakan "ignore previous instructions" dan mencoba menimpa system prompt secara langsung. Context-smuggling menyembunyikan instruksi di dalam sesuatu yang terlihat seperti data: dokumen yang ditempel, hasil tool, blok kode. Multi-turn-ramp memanaskan model dengan putaran yang tidak berbahaya lalu menurunkan lantai satu langkah demi satu langkah, mengeksploitasi kecenderungan model untuk tetap konsisten dengan percakapan. Trik encoding (base64, rot13, leet-speak, sisipan zero-width) menyembunyikan token terlarang dari filter kata kunci naif. Prefix-injection mengakhiri prompt dengan "Sure, here's how" sehingga model melanjutkan dari jawaban yang diasumsikan alih-alih menolak.

```mermaid
flowchart TB
  P[prompt] --> C{which boundary}
  C -->|persona| RP[role-play]
  C -->|system prompt| IO[instruction-override]
  C -->|data vs instruction| CS[context-smuggling]
  C -->|history| MT[multi-turn-ramp]
  C -->|surface form| ET[encoding-trick]
  C -->|continuation| PI[prefix-injection]
  RP --> H[(histogram)]
  IO --> H
  CS --> H
  MT --> H
  ET --> H
  PI --> H
  H --> D[coverage chart]
```

Setiap fixture adalah rekaman dengan `id`, `category`, `subtype`, `prompt`, `target_behavior`, dan `severity`. Objek taksonomi memuat fixture, mengelompokkannya berdasarkan kategori, dan mengekspos API `match`: dengan prompt kandidat, kembalikan fixture terdekat dan kategorinya. Match adalah cosine trigram karakter: kasar, cepat, tanpa dependensi. Ini bukan detektor. Detektor ada di pelajaran 83. Ini adalah produsen label.

Severity mengikuti skala 1-5. Sebuah 1 adalah serangan canggung terhadap target jinak ("please pretend to be a pirate"). Sebuah 5 adalah serangan yang, jika berhasil, menghasilkan output yang tidak boleh dikeluarkan oleh sistem yang dideploy (detail operasional untuk aktivitas berbahaya). Sebagian besar fixture berada di 2-3 karena serangan nyata pada skala deployment condong ke yang mudah dan yang malas. Severity ditetapkan oleh penulis fixture. Dua reviewer yang tidak setuju lebih dari satu peringkat adalah tanda rubrik perlu dipertajam.

```figure
cd-attack-taxonomy
```

## Bangun Sendiri

Korpus berada di `code/fixtures.py` sebagai satu daftar Python. Kelas taksonomi di `code/main.py` memuatnya, memvalidasi bahwa setiap kategori punya setidaknya tujuh fixture, mengekspos metode `by_category`, `match`, dan `stats`, dan merilis demo yang bisa dijalankan yang mencetak histogram. Cosine trigram diimplementasikan dari nol dengan `numpy`.

Pass validasi memeriksa empat invarian: setiap fixture punya prompt yang tidak kosong, setiap kategori dalam skema terwakili, setiap severity berada di `1..5`, dan setiap id fixture unik. Kegagalan di sini adalah hard exit, bukan peringatan, karena sisa track bergantung pada korpus yang konsisten secara internal.

## Gunakan

Jalankan `python3 main.py` dari direktori `code/` pelajaran. Demo mencetak jumlah fixture per kategori, menjalankan tiga probe sampel terhadap `match`, dan menulis `taxonomy.json` ke folder outputs pelajaran. Pelajaran di hilir membaca `taxonomy.json` alih-alih mengimpor modul Python, sehingga korpus adalah artefak yang stabil.

## Rilis

`outputs/skill-jailbreak-taxonomy.md` mendokumentasikan enam kategori dan rubriknya. Perlakukan itu sebagai kosakata bersama tim. Setiap temuan yang dicatat oleh harness di pelajaran 87 merujuk ke id taksonomi.

## Latihan

1. Tambahkan kategori ketujuh untuk indirect-prompt-injection (instruksi yang tertanam di dokumen yang di-retrieve, bukan di giliran pengguna). Buat sepuluh fixture dan jalankan ulang validatornya.
2. Ganti cosine trigram dengan skorer token-edit-distance dan ukur bagaimana penugasan match berubah pada korpus yang ada.
3. Tarik tiga puluh fixture tambahan dari log produkmu sendiri (disensor) dan konfirmasi distribusi kategorinya cocok dengan yang secara intuitif diharapkan timmu.

## Istilah Kunci

| Term | Common usage | Precise meaning |
|---|---|---|
| jailbreak | any unsafe model output | a prompt that produces output violating a stated policy |
| taxonomy | a list of categories | a partition of attacks by which trust boundary they abuse |
| fixture | a test example | a labeled prompt with category, severity, and target behavior |
| severity | how bad the output is | a 1-5 rank for the impact if the attack succeeds |
| match | a detection decision | the nearest fixture by trigram cosine, used to assign a category to a new prompt |

## Bacaan Lanjutan

Pelajaran ini adalah titik masuk. Pelajaran 83-87 membangun di atas korpus secara langsung.
