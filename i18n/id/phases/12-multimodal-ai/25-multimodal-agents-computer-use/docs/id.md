# Agen Multimodal dan Computer-Use (Capstone)

> Produk frontier 2026 adalah agen multimodal yang membaca screenshot, mengklik tombol, menavigasi UI web, mengisi formulir, dan menyelesaikan alur kerja end-to-end. SeeClick dan CogAgent (2024) membuktikan primitif GUI-grounding. Ferret-UI menambahkan mobile. ChartAgent memperkenalkan visual tool-use untuk grafik. VisualWebArena dan AgentVista (2026) adalah benchmark yang dikejar oleh frontier — dan bahkan Gemini 3 Pro serta Claude Opus 4.7 hanya mencetak ~30% pada tugas-tugas sulit AgentVista. Capstone ini menyatukan setiap benang Phase 12: persepsi (VLM resolusi-tinggi), penalaran (LLM dengan tool use), grounding (output koordinat), memori horison-panjang, dan evaluasi.

**Type:** Capstone
**Languages:** Python (stdlib, action schema + agent loop skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Time:** ~240 minutes

## Tujuan Pembelajaran

- Merancang loop agen multimodal: persepsi → penalaran → bertindak → mengamati → mengulang.
- Membangun skema output GUI grounding (koordinat klik, ketik teks, scroll, drag) yang bisa dieluarkan VLM sebagai JSON.
- Membandingkan agen screenshot-saja vs agen accessibility-tree vs agen hibrida.
- Menyiapkan evaluasi benchmark agen multimodal pada irisan kecil VisualWebArena.

## Masalahnya

Sebuah alur kerja situs pemesanan: "cari penerbangan ke Tokyo untuk 15 April, kursi lorong di bawah $800, pesankan."

Sebuah agen multimodal perlu:

1. Mengambil screenshot dari browser.
2. Mem-parse screenshot + URL + tujuan menjadi sebuah rencana.
3. Mengeluarkan aksi terstruktur: click (pada x,y), type "Tokyo" (pada elemen E), scroll down, select (radio button).
4. Menerapkan aksi ke browser.
5. Mengamati state baru (screenshot berikutnya).
6. Mengulang sampai tugas selesai.

Setiap langkah adalah panggilan VLM multimodal. Output VLM harus berupa JSON yang dapat di-parse. Kesalahan menumpuk di lintas langkah, jadi pemulihan itu penting.

## Konsepnya

### GUI grounding — primitifnya

GUI grounding adalah: diberi screenshot dan instrujsi bahasa alami, keluarkan koordinat (x, y) untuk diklik (atau aksi lain).

SeeClick (arXiv:2401.10935) adalah hasil open pertama berskala besar: fine-tune sebuah VLM pada data GUI sintetis + nyata, keluarkan koordinat sebagai token teks biasa. Bekerja.

CogAgent (arXiv:2312.08914) menambahkan encoding resolusi-tinggi 1120x1120 untuk UI padat. Skor: ~84% pada navigasi web.

Ferret-UI (arXiv:2404.05719) berfokus pada UI mobile, terintegrasi dengan data accessibility iOS.

Format output biasanya JSON:

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` membantu pemulihan: jika koordinat meleset di antara screenshot, petunjuk semantik memungkinkan sistem melakukan re-grounding.

### Skema aksi

Sebuah skema aksi tipikal punya 6-10 tipe aksi:

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

Agen mengeluarkan satu aksi per langkah. Wrapper browser mengeksekusi dan mengembalikan state baru.

### Screenshot-saja vs accessibility-tree

Dua mode input:

- Screenshot-saja: gambar penuh, tanpa info struktural. Paling umum; bekerja di aplikasi apa pun.
- Accessibility tree: DOM terstruktur / info accessibility iOS. Jauh lebih andal untuk grounding; bekerja di mana tree tersedia.
- Hibrida: keduanya, dengan tree sebagai grounder yang andal untuk aksi atomik dan screenshot untuk konteks semantik.

Agen produksi memakai hibrida jika memungkinkan. Automasi browser (Selenium + accessibility) selalu punya tree-nya; aplikasi desktop terkadang punya.

### Memori horison-panjang

Sebuah alur kerja 20 langkah menghasilkan 20 screenshot. Context VLM cepat penuh. Tiga strategi kompresi:

- Rantai-ringkasan: setiap 5 langkah, ringkas apa yang telah terjadi, buang screenshot lama.
- Skip-frame: simpan yang pertama, terakhir, dan setiap screenshot ke-3.
- Log yang direkam oleh tool: eksekusi aksi, simpan log teks tentang apa yang dilakukan; jangan melihat-ulang screenshot lama.

API computer-use Claude memakai pola log. Lebih sederhana, lebih andal.

### Visual tool use

ChartAgent (arXiv:2510.04514) memperkenalkan visual tool use untuk pemahaman grafik: crop, zoom, OCR, memanggil deteksi eksternal. Agen bisa mengeluarkan "crop to region (100, 200, 300, 400) then call OCR" sebagai sebuah tool call. Tool mengembalikan teks; VLM melanjutkan penalaran.

Pola ini menggeneralisasi: set-of-mark prompting, anotasi region, dan tool deteksi eksternal semuanya cocok dengan skema yang sama "keluarkan tool call, terima respons terstruktur."

### Benchmark 2026

- ScreenSpot-Pro. GUI grounding pada ~1k screenshot web. Open SOTA Qwen2.5-VL-72B ~85%. Frontier ~90%.
- VisualWebArena. Tugas web end-to-end (belanja, forum, iklan baris). Open SOTA ~20%. Gemini 3 Pro ~27%.
- AgentVista (arXiv:2602.23166). Benchmark 2026 tersulit. Alur kerja realistis di 12 domain. Model frontier mencetak 27-40%; model open 10-20%.
- WebArena / WebShop. Benchmark lebih lama; sudah jenuh oleh frontier.

### Mengapa masih sulit

Bottleneck performa agen:

1. Grounding visual pada skala halus. "Klik X kecil itu" sering gagal pada resolusi mobile.
2. Perencanaan horison-panjang. Setelah 10 aksi, agen menyimpang dari tujuannya.
3. Pemulihan kesalahan. Saat sebuah klik gagal (tombol salah), mendeteksi + memulihkan jarang menjadi data training.
4. Konteks lintas-halaman. Melompat di antara tab atau formulir panjang kehilangan state.

Arah riset: arsitektur memori, replanning eksplisit, verifikasi multimodal (kecocokan screenshot untuk keberhasilan aksi).

### Build-it capstone

Tugas capstone: membangun agen computer-use yang:

1. Membaca HTML + screenshot dari halaman mock situs pemesanan.
2. Merencanakan sekuens multi-langkah: search → select → fill form → submit.
3. Mengeluarkan aksi JSON yang cocok dengan skema aksi.
4. Mengevaluasi pada irisan 10 tugas yang tetap.

Pelajaran ini menyediakan kode scaffold yang mudah diperluas menjadi browser nyata.

```figure
mm-agent-loop
```

## Gunakan

`code/main.py` adalah scaffold capstone:

- Definisi skema aksi JSON (10 aksi).
- State browser mock sebagai dict.
- Kerangka loop agen: terima state, keluarkan aksi, terapkan, ulangi.
- Mini-benchmark 10-tugas (halaman sintetis) untuk mengukur tingkat keberhasilan end-to-end.
- Hook pemulihan-kesalahan untuk saat sebuah aksi gagal.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-multimodal-agent-designer.md`. Diberi produk computer-use (domain, set aksi, target evaluasi), merancang loop agen penuh, strategi memori, mode grounding, dan skor benchmark yang diharapkan.

## Latihan

1. Perluas skema aksi dengan tool `screenshot_region` (crop + zoom). Tugas apa yang diuntungkan?

2. Baca AgentVista (arXiv:2602.23166). Deskripsikan kategori tugas tersulit dan mengapa model frontier masih gagal.

3. Kompresi memori horison-panjang: rancang rantai-ringkasan dengan ≤4 screenshot yang tetap hidup, berapa pun jumlah yang di-log.

4. Bangun hook pemulihan-kesalahan: saat aksi gagal (tombol tidak ditemukan), apa yang dilakukan agen selanjutnya?

5. Bandingkan Claude 4.7 screenshot-saja dengan Qwen2.5-VL hibrida screenshot + accessibility-tree pada 10 tugas web. Mana yang menang di tugas mana?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GUI grounding | "Click coordinates" | Model outputs (x,y) for the target of an instruction on a screenshot |
| Action schema | "Tool definitions" | JSON description of valid actions (click, type, scroll, drag) |
| Accessibility tree | "Structured DOM" | Machine-readable UI hierarchy from browser/iOS APIs |
| Hybrid agent | "Screenshot + tree" | Uses both image and structured info; more reliable than either alone |
| Visual tool use | "Zoom/crop/detect" | Agent calls external vision tools (OCR, detection) mid-plan |
| Summary-chain | "Memory compression" | Periodic text summaries replace long screenshot history |
| VisualWebArena | "E2E web bench" | 2024 benchmark for end-to-end web tasks |
| AgentVista | "2026 hard bench" | 12-domain realistic workflows; even Gemini 3 Pro scores ~30% |

## Bacaan Lanjutan

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
