# Agent Skills: Kontrak Portabel dan Batas Runtime

> Sebuah skill bukan prompt panjang dengan nama file yang lebih bagus. Ia adalah paket instruksi, resource, dan helper eksekutil yang bisa ditemukan, yang masuk ke konteks sebuah agent melalui kontrak runtime.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 01 (The Tool Interface), Phase 13 · 05 (Tool Schema Design)
**Time:** ~90 minutes

## Tujuan Pembelajaran

- Mendefinisikan sebuah agent skill tanpa mencampurnya dengan prompt, instruksi repo, tool, hook, subagent, atau plugin.
- Membaca kontrak portabel `SKILL.md` dan memisahkannya dari ekstensi spesifik-runtime.
- Menjelaskan discovery, selection, activation, loading resource, penggunaan tool, dan verifikasi sebagai tahap lifecycle yang berbeda-beda.
- Memvalidasi sebuah paket skill sebelum sebuah runtime menempatkannya di katalog agent.
- Memilih di antara skill, MCP tool, hook, subagent, atau kode biasa untuk sebuah tugas konkret.

## Keberhasilan Pertama Sepuluh Menit

Lakukan ini sebelum penjelasan panjang. Kamu akan membuat sebuah skill kecil, memasang bundle reviewer lengkap ke host agent nyata, memanggilnya, memverifikasi hasilnya, dan menghapusnya. Ini membuktikan lifecycle dengan hasil yang bisa diamati.

### Preflight untuk lab host-nyata

Checkpoint host-nyata membutuhkan Node.js, `npx`, Python 3, satu host pilihan yang mendukung skill, dan akses tulis ke scope proyek atau pengguna yang kamu pilih di installer. Verifikasi perintah lokalnya lebih dulu:

```bash
node --version
npx --version
python3 --version
```

Tentukan host dan scope mana yang akan kamu pakai sebelum instalasi. Jika ada persyaratan yang tidak tersedia, baca pelajaran ini di website atau lanjutkan dengan latihan paket manual di bawah. Fallback itu mengajarkan kontraknya, tetapi tidak membuktikan discovery di host, invocation, eksekusi script terbundel, atau perilaku uninstall. Tandai pengamatan itu sebagai tertunda.

### 1. Mulai di direktori kerja kosong

Jalankan perintah berikut dari direktori induk mana pun tempat kamu menyimpan pekerjaan belajarmu:

```bash
mkdir -p agent-skills-first-run
cd agent-skills-first-run
TARGET_ROOT="$(pwd -P)"
printf 'TARGET_ROOT=%s\n' "$TARGET_ROOT"
ls -A
```

Perintah terakhir seharusnya tidak mencetak apa pun. Jika ia mencetak file, pilih direktori kosong yang berbeda agar tinjauan punya batas yang jelas.

Buat direktori untuk skill pertamamu:

```bash
mkdir -p my-first-skill
```

Buat `my-first-skill/SKILL.md` dengan konten ini:

```markdown
---
name: my-first-skill
description: Turn rough meeting notes into a compact decision record when the user asks to capture a technical decision.
---

# Decision record

Extract the decision, context, alternatives, owner, and next review date.
If the notes do not contain a decision, ask one clarifying question instead
of inventing one.
```

Verifikasi bahwa kamu membuat file di direktori yang dituju:

```bash
test -f my-first-skill/SKILL.md
```

Tanpa output dan exit code 0 berarti file itu ada.

### 2. Pasang bundle reviewer lengkap

Tetap di `agent-skills-first-run` dan jalankan:

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-contract-reviewer --full-depth
```

Pilih agent host dan scope yang kamu pakai. Installer harus mencantumkan `skill-contract-reviewer` dan tujuan tempat ia menulis. `--full-depth` diperlukan karena skill pelajaran ini adalah bundle bersarang dengan referensi, sebuah script, dan sebuah asset.

Setel `SKILL_ROOT` ke direktori absolut yang dilaporkan installer. Ini harus berupa direktori yang memuat `SKILL.md` yang terpasang, bukan direktori sumber pelajaran dan bukan workspace saat ini:

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-contract-reviewer" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\n' "$SKILL_ROOT"
```

Jika sesi agent sudah terbuka, mulai sesi baru atau pakai perintah rescan skill milik host itu. Jangan berasumsi setiap host me-reload katalognya secara hot.

### 3. Panggil secara eksplisit

Di agent yang terpasang, dengan `agent-skills-first-run` sebagai direktori kerja, gunakan sintaks yang didukung host itu:

| Host | Explicit invocation |
|---|---|
| Codex | `skill-contract-reviewer`, or choose it from `/skills`, then provide the review request |
| Claude Code | `/skill-contract-reviewer` followed by the review request |
| Portable fallback | `Use skill-contract-reviewer to review the target package.` |

Gunakan nilai absolut yang dicetak untuk `SKILL_ROOT` dan `TARGET_ROOT` di dalam permintaan. Minta host meng-expand-nya sebelum eksekusi dan menampilkan perintah hasil-resolve yang persis, bukan perintah yang bergantung pada direktori kerja proses:

```text
Use skill-contract-reviewer to review <TARGET_ROOT>/my-first-skill. The installed bundle root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/check_skill.py <TARGET_ROOT>/my-first-skill. Before running it, show the fully resolved argv. Return the validation report, selected primitives, and one sentence for each selection. Include the resolved script path, resolved target path, cwd, argv, and exit code as execution evidence.
```

Perintah hasil-resolve seharusnya punya bentuk berikut, tanpa placeholder yang tersisa:

```bash
python3 "/absolute/install/path/skill-contract-reviewer/scripts/check_skill.py" \
  "/absolute/workspace/path/agent-skills-first-run/my-first-skill"
```

Hasil yang berhasil punya ketiga properti berikut:

1. Host menemukan `skill-contract-reviewer` berdasarkan nama.
2. Reviewer membaca kontrak paket dan menjalankan validator terbundel-nya.
3. Respons berisi laporan validasi tanpa error struktural untuk sampel itu, plus seleksi primitive yang berdasar alasan.

Bukti eksekusi juga harus menamai path script, path target, cwd, vektor argumen yang persis, dan exit code. Laporan yang fasih tanpa field-field itu tidak membuktikan bahwa script pendamping yang terpasang benar-benar berjalan.

Jika host melaporkan bahwa skill tidak tersedia, verifikasi tujuan instalasi, lakukan rescan atau restart sekali, lalu coba ulangi permintaan eksplisitnya. Jangan menulis ulang deskripsi skill untuk menyembunyikan kegagalan instalasi.

### 4. Uji selection implisit

Mulai giliran agent yang segar dan masukkan tugas yang sama tanpa menamai skill:

```text
Review <TARGET_ROOT>/my-first-skill as a reusable agent package and tell me whether its package contract is valid.
```

Jika host mengekspos skill yang dipilih, rekam apakah ia memilih `skill-contract-reviewer`. Jika host tidak mengekspos routing, tandai selection implisit sebagai belum terverifikasi. Invocation eksplisit adalah fallback yang portabel.

### 5. Bersihkan

Hapus hanya bundle reviewer yang terpasang:

```bash
npx skills remove skill-contract-reviewer
```

Pilih host dan scope yang sama seperti saat instalasi. Setelah rescan atau sesi baru, permintaan eksplisit untuk `skill-contract-reviewer` seharusnya melaporkan bahwa ia tidak tersedia. Simpan `my-first-skill` untuk pelajaran berikutnya, atau hapus direktori latihan setelah kamu selesai mengikuti track ini.

## Masalahnya

Misalkan timmu punya alur rilis yang andal. Ia menemukan perubahan yang sudah di-merge, memeriksa catatan migrasi, memperbarui changelog, menjalankan perintah packaging, dan menghasilkan checklist tinjauan.

Menaruh alur itu dalam satu prompt membuatnya mudah ditempel tetapi sulit dioperasikan. Prompt itu tidak punya identitas stabil, tidak ada aturan discovery, tidak ada batas resource, tidak ada bentuk paket yang bisa diuji, dan tidak ada jawaban atas pertanyaan dasar: Siapa yang boleh memanggilnya? Kapan model harus memilihnya? Script mana yang bisa ia jalankan? File mana yang tepercaya? Apa yang bertahan ketika konteks dikompresi?

Kesalahan sebaliknya adalah memperlakukan setiap instruksi yang bisa dipakai ulang sebagai skill. Konvensi repo, otomasi deterministik, tool eksternal, event hook, dan agent yang didelegasikan memecahkan masalah yang berbeda-beda. Membungkus semuanya ke dalam `SKILL.md` menghasilkan direktori yang terlihat portabel sambil bergantung pada perilaku tak terdokumentasi satu host.

Tugas rekayasa pertama adalah klasifikasi. Tentukan apa artefak itu sebelum kamu memutuskan cara membungkusnya.

## Konsepnya

### Skill mengkodekan pengetahuan prosedural

Sebuah agent skill adalah direktori yang entry point-nya adalah `SKILL.md`. File entry berisi YAML frontmatter yang diikuti instruksi Markdown. Direktori itu juga dapat memuat referensi, script, dan asset.

```figure
skill-package-anatomy
```

Direktorinya, bukan file Markdown semata, adalah unit yang bisa dideploy. Sebuah `SKILL.md` yang disalin dengan referensi yang hilang adalah paket yang rusak meskipun frontmatter-nya berhasil diparse.

### Abstraksi yang bertetangga

| Artifact | Primary job | Loaded or run when | What it should not impersonate |
|---|---|---|---|
| Prompt | Shape one model interaction | Included by an application or user | A versioned package with resources |
| Repository instructions | Explain one codebase's standing rules | A coding runtime enters that scope | A reusable task workflow |
| Agent skill | Supply reusable procedural knowledge | Explicit or implicit activation | A hard authorization boundary |
| MCP tool | Expose a typed remote capability | The model or application calls it | A detailed operating procedure |
| Hook | Run deterministic logic on an event | The declared event occurs | Probabilistic model routing |
| Subagent | Delegate work with separate context and state | An orchestrator creates or calls it | A static instruction bundle |
| Plugin | Distribute a larger runtime extension | The host installs or enables it | The portable skill contract itself |
| Learned skill library | Store behavior discovered through experience | A policy retrieves a prior program or trajectory | A standards-based `SKILL.md` package |

Sebuah skill rilis dapat memberi tahu agent bagaimana memeriksa sebuah rilis. Sebuah server MCP dapat mengekspos registry rilis. Sebuah hook dapat melarang push langsung. Sebuah subagent dapat mengaudit kandidat secara independen. Bagian-bagian ini menyusun karena masing-masing memegang tanggung jawab yang berbeda.

### Kata "skill" menamai dua gagasan yang berbeda

Sistem riset kadang menyebut sebuah program yang dipelajari, trajectory yang berhasil, atau fragmen kebijakan spesifik-lingkungan sebagai skill. Sebuah agent dapat membuat artefak-artefak ini selama eksplorasi, mengambilnya berdasarkan kemiripan tugas, mengeksekusinya, dan merevisi pustaka dari umpan balik. Phase 14 · 10 membangun pustaka lifelong-learning semacam itu.

Sebuah Agent Skill di mini-track ini berbeda. Ia adalah paket yang ditulis dengan kontrak filesystem yang dideklarasikan, metadata katalog, disclosure progresif, invocation yang dimediasi runtime, dan tool yang dikendalikan host. Ia dapat dibangkitkan atau diperbaiki oleh agent, tetapi pembelajaran bukanlah kebutuhan format ini.

| Dimension | Agent Skill package | Learned skill library |
|---|---|---|
| Primary unit | `SKILL.md` directory | Program, policy, trajectory, or memory record |
| Creation | Authored, generated, or curated | Usually discovered from environment experience |
| Selection | Catalog description plus runtime policy | Retrieval or policy over task state |
| Execution | Model follows instructions and calls host tools | Environment runs a stored behavior or code artifact |
| Portability | Package contract can cross compatible hosts | Often tied to one environment and action space |
| Evaluation | Routing, artifact, safety, and host compatibility | Reward, success rate, transfer, and library growth |

Kedua gagasan mengemas kompetensi yang bisa dipakai ulang. Keduanya tidak boleh berbagi klaim implementasi hanya karena berbagi nama.

### Inti portabel

Spesifikasi Agent Skills mewajibkan dua field frontmatter:

```yaml
---
name: release-readiness
description: Inspect a release candidate when the user asks whether a version is ready to publish.
---
```

`name` adalah identitas yang stabil. Ia harus memenuhi aturan penamaan spesifikasi dan cocok dengan direktori induknya. `description` sekaligus dokumentasi dan metadata routing. Ia harus menyatakan apa yang dilakukan skill dan kapan ia berlaku.

Field opsional portabel adalah:

| Field | Purpose | Portability note |
|---|---|---|
| `license` | State the terms for the package | Core specification |
| `compatibility` | State environmental requirements | Core specification |
| `metadata` | Carry string-valued extension data | Core specification |
| `allowed-tools` | Suggest pre-approved tools | Experimental; host support varies |

Badan Markdown memuat instruksi operasional. Ia seharusnya mendefinisikan alur kerja, titik keputusan, perilaku kegagalan, dan jalur langsung ke resource pendukung.

```markdown
# Release readiness

Use this workflow for a release candidate, not for ordinary development builds.

1. Read `references/release-policy.md`.
2. Run `python3 scripts/inspect_release.py --format json`.
3. Stop if the report contains a blocking failure.
4. Produce the checklist from `assets/release-checklist.md`.
5. Ask for approval before any publish or tag action.
```

### Ekstensi runtime adalah lapisan kedua

Sebagian host menerima frontmatter tambahan atau konfigurasi pendamping. Field-field itu bisa berguna, tetapi tidak otomatis portabel.

| Behavior | Example host extension | Portable core? |
|---|---|:---:|
| Hide a skill from model routing while keeping direct user invocation | `disable-model-invocation` | No |
| Hide a skill from the user's command menu while allowing model routing | `user-invocable` | No |
| Show argument help in a command menu | `argument-hint` | No |
| Run the skill in delegated context | `context`, `agent` | No |
| Pin model or reasoning settings | `model`, `effort` | No |
| Register lifecycle automation | `hooks` | No |
| Disable implicit invocation in Codex | `agents/openai.yaml` policy | No |

Perlakukan setiap ekstensi sebagai sebuah adapter. Jaga alur kerja inti tetap valid tanpanya, dokumentasikan fallback-nya, dan uji host yang mengonsumsinya. Sebuah runtime dapat mengabaikan field yang tidak dikenal, menolaknya, atau mempertahankannya tanpa mengimplementasikan perilakunya.

### Frontmatter adalah metadata yang bisa dieksekusi

Metadata mengubah perilaku sistem sebelum badan skill dibaca.

- `name` yang cacat dapat membuat discovery gagal.
- `description` yang kabur dapat merutekan permintaan yang salah.
- Sebuah flag khusus-manusia dapat mengeluarkan skill dari katalog model.
- Sebuah allowance tool dapat mengubah apakah host meminta izin.
- Sebuah pengaturan konteks dapat memindahkan eksekusi ke sesi agent terpisah.

Tinjau frontmatter seperti kode konfigurasi. Validasi, beri versi, dan sertakan perilakunya di dalam eval.

### Lifecycle skill

```figure
skill-runtime-lifecycle
```

Setiap panah adalah sebuah batas dengan mode kegagalannya sendiri.

1. **Discovery** menemukan paket yang mungkin di lokasi yang dikonfigurasi.
2. **Validation** menolak paket yang cacat atau tidak aman sebelum publikasi katalog.
3. **Cataloging** mengekspos `name` dan `description` yang ringkas, bukan paket penuh.
4. **Selection** memutuskan apakah skill itu relevan.
5. **Activation** memuat badan ke dalam konteks yang terlihat model.
6. **Disclosure** membaca referensi atau asset hanya ketika sebuah cabang membutuhkannya.
7. **Execution** memakai tool host di bawah aturan izin dan isolasi host.
8. **Verification** memeriksa artefak yang dihasilkan secara independen dari klaim model.

Menggabungkan tahap-tahap ini menghasilkan model mental yang buruk. Sebuah skill yang ditemukan belum aktif. Sebuah skill yang aktif belum diizinkan melakukan semua yang ia jelaskan. Sebuah panggilan tool yang diizinkan bukan bukti bahwa hasilnya benar.

### Skill dan tool bersifat ortogonal

MCP menjawab, "Kapabilitas mana yang bisa dipanggil aplikasi ini, dan apa skemanya?" Sebuah skill menjawab, "Bagaimana agent seharusnya mendekati kelas tugas ini?"

```figure
skill-tool-orthogonality
```

Skill mungkin menamai sebuah tool, tetapi meregistrilah yang memiliki capability registry yang sebenarnya. Jika tool itu absen, skill seharusnya menyatakan fallback atau gagal dengan jelas. Ia tidak pernah boleh memberi arti bahwa menamai sebuah kapabilitas menciptakannya.

### Skill dan instruksi repo punya scope berbeda

Instruksi repo menggambarkan lingkungan yang sudah kamu huni: perintah, konvensi, file hasil-generate, dan batasan. Sebuah skill menyediakan prosedur yang bisa dipakai ulang untuk tugas yang bisa terjadi di banyak repo.

Ketika keduanya berlaku, permintaan pengguna yang aktif dan aturan repo membatasi skill. Sebuah skill refactoring generik tidak boleh meng-override aturan repo yang melarang penyuntingan file hasil-generate.

### Skill tidak saling mengimpor

Sebuah skill dapat mengarahkan agent memanggil skill lain, tetapi ini bukan import tingkat bahasa. Skill kedua tetap melewati discovery, kelayakan, activation, izin, dan penanganan konteks runtime.

Tulis dependensi lintas-skill sebagai edge alur kerja yang bisa diamati:

```markdown
After producing the candidate changelog, invoke the `release-risk-review` skill.
Pass the candidate path and require a blocking or non-blocking verdict.
If that skill is unavailable, stop and report the missing dependency.
```

Ini membuat dependensi itu bisa diuji dan memberi host kesempatan menegakkan kebijakan.

## Bangun Sendiri

`code/main.py` mengimplementasikan sebuah validator berorientasi-standar kecil dan sebuah pemilih artefak. Ia tetap stdlib-only sehingga setiap aturan terlihat.

Validator mengekspos:

- `parse_frontmatter(text)` untuk memisahkan metadata dari badan.
- `validate_skill_text(text, directory_name, allowed_runtime_extensions=())` untuk memeriksa field wajib, penamaan, ekstensi tak dikenal, keberadaan badan, dan batasan portabilitas.
- `ValidationIssue` dan `SkillReport` untuk mengembalikan bukti terstruktur alih-alih satu boolean yang opaque.
- `FrontmatterSyntaxError` untuk input yang tidak bisa ditafsirkan dengan aman.

Pemilih mengekspos `TaskShape` dan `select_primitives(task)`. Ia memetakan kebutuhan sebuah tugas ke kode biasa, instruksi repo, skill, hook, subagent, atau tool MCP.

Jalankan lab-nya:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/22-skills-and-agent-sdks
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Blok perintah ini mensyaratkan clone lokal dan harus dimulai dari mana pun di dalam clone itu sehingga `git rev-parse --show-toplevel` dapat me-resolve root repository.

Demo mencetak JSON untuk satu skill portabel yang valid, satu skill yang diperluas host, satu paket yang tidak valid, dan beberapa keputusan task-shape. Periksa kode masalahnya. Sebuah validator paket seharumnya menjelaskan cara memperbaiki sebuah artefak tanpa menebak atas nama penulisnya.

### Urutan validasi itu penting

Validasi fakta struktural yang murah sebelum aturan konten yang lebih dalam:

```figure
skill-validation-order
```

Urutan ini mencegah error sekunder menyembunyikan invariant pertama yang rusak.

## Gunakan

Sebelum menulis sebuah skill, isi kartu keputusan ini:

| Question | If yes | Likely primitive |
|---|---|---|
| Does this need reusable model judgment across several steps? | The procedure is stable but decisions vary | Skill |
| Must this happen every time an event fires? | Missing one execution is unacceptable | Hook or application code |
| Does the model need an external capability with typed inputs? | The operation lives outside model context | Tool or MCP server |
| Does the work need isolated context, state, or ownership? | A separate worker returns a bounded result | Subagent |
| Is this guidance specific to one repository? | It describes local commands and constraints | Repository instructions |
| Is one interaction enough? | No package lifecycle is needed | Prompt |

Banyak alur production memakai lebih dari satu baris. Kartu itu mencegah satu artefak berpura-pura menyediakan setiap properti.

## Rilis

Pelajaran ini menghasilkan bundle `skill-contract-reviewer` di bawah `outputs/`. Ia memuat:

- sebuah `SKILL.md` portabel yang meninjau paket skill yang diusulkan;
- checklist referensi untuk kontrak portabel dan seleksi primitive;
- sebuah script validasi deterministik;
- fixture task-shape yang mencakup prompt, skill, tool, hook, kode biasa, dan subagent.

Pasang bundle penuhnya, bukan hanya file entry-nya:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/install_skills.py /tmp/aiefs-skills --phase 13 --type skill
```

Installer kursus melaporkan setiap skill Phase 13 yang disalin dan menulis `/tmp/aiefs-skills/manifest.json`. Tujuan yang bersih ini memeriksa bentuk paket; loop keberhasilan-pertama di atas memeriksa discovery dan invocation di host nyata.

Pelajaran berikutnya memperdalam setiap tahap lifecycle. Pelajaran 24 membangun discovery dan disclosure progresif. Pelajaran 25 membangun kebijakan invocation dan routing. Pelajaran 26 memisahkan izin dari sandboxing. Pelajaran 27 mengubah seluruh paket menjadi artefak rilis yang dievaluasi.

## Latihan

1. Klasifikasikan lima alur kerja dari timmu sendiri memakai `TaskShape`. Bela setiap kasus di mana kamu memilih lebih dari satu primitive.
2. Tambahkan tes batas yang membuktikan nilai `compatibility` 500 karakter lulus dan nilai 501 karakter gagal sebagai kesalahan spesifikasi.
3. Tambahkan satu ekstensi runtime ke allowlist. Tulis tes yang membuktikan file yang sama masih bisa dibedakan dari skill portabel-saja.
4. Pecah prompt 400 baris menjadi `SKILL.md`, satu referensi, satu kontrak script, dan satu template output. Jaga setiap file bertanggung jawab atas satu jenis informasi.
5. Rancang respons kegagalan untuk sebuah skill yang mereferensikan tool MCP yang tidak tersedia. Jangan diam-diam mensubstitusi tool dengan izin yang lebih luas.
6. Tinjau sebuah skill yang sudah ada dan beri label setiap kalimat sebagai routing, prosedur, kebijakan, penunjuk referensi, atau kontrak output. Pindahkan apa pun yang tidak seharusnya di sana.

## Istilah Kunci

| Term | What people say | What it actually means |
|---|---|---|
| Agent skill | "A saved prompt" | A discoverable directory of procedural instructions and optional resources |
| Portable core | "Fields every runtime shares" | The contract defined by the Agent Skills specification |
| Runtime extension | "Extra frontmatter" | Host-specific configuration whose behavior requires a compatible adapter |
| Activation | "The skill ran" | The skill body entered model-visible context; execution may come later |
| Skill dependency | "Import another skill" | A runtime-mediated invocation edge with availability and policy checks |
| Tool contract | "A function schema" | Inputs, outputs, permissions, side effects, errors, and evidence for a capability |

## Bacaan Lanjutan

- [Agent Skills specification](https://agentskills.io/specification) untuk kontrak direktori dan frontmatter yang portabel.
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices) untuk scope, instruksi, dan organisasi resource.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) untuk perilaku discovery dan invocation Codex saat ini.
- [Claude Code skills](https://code.claude.com/docs/en/skills) untuk ekstensi invocation, argumen, tool, dan delegated-context milik satu runtime.
