# Keluarga Qwen-VL dan Video Dynamic-FPS

> Keluarga Qwen-VL — Qwen-VL (2023), Qwen2-VL (2024), Qwen2.5-VL (2025), Qwen3-VL (2025) — adalah garis keturunan model vision-language terbuka paling berpengaruh pada 2026. Setiap generasi membuat satu taruhan arsitektur decisif yang disalin ekosistem terbuka lainnya dalam dua belas bulan: resolusi dinamis native melalui M-RoPE, sampling dynamic-FPS dengan time alignment absolut, window attention di ViT, dan format output agent terstruktur. Pada Qwen3-VL, resepnya telah distabilkan: encoder 2D-RoPE-ViT dengan input aspect-ratio-native, MLP projector ke basis bahasa Qwen3 yang besar, dan tahap training yang menekankan OCR, grounding, dan perilaku agent sebagai target first-class. Pelajaran ini membaca keluarga secara kronologis supaya kamu memahami mengapa setiap knob berada di tempatnya.

**Type:** Learn
**Languages:** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)
**Prerequisites:** Phase 12 · 06 (patch-n'-pack)
**Time:** ~120 minutes

## Tujuan Pembelajaran

- Menghitung rotasi tiga-sumbu M-RoPE (temporal, tinggi, lebar) dan menjelaskan mengapa ketiganya dibutuhkan.
- Memilih strategi sampling dynamic-FPS untuk sebuah video dan bernalar tentang token-per-detik vs akurasi deteksi-event.
- Menyebutkan empat peningkatan generasi Qwen-VL secara berurutan dan apa yang masing-masing mengaktifkan.
- Memasang format output agent JSON bergaya Qwen2.5-VL dan mem-parse tool call terstruktur dari respons VLM.

## Masalahnya

Qwen-VL dikirim pada Agustus 2023 sebagai tanggapan langsung terhadap LLaVA-1.5 dan BLIP-2. Kesenjangan yang ditarget tim Qwen tiga dimensi: resolusi, video, dan output terstruktur.

Resolusi: LLaVA-1.5 berjalan pada 336x336. Cukup untuk foto, tidak berguna untuk faktur berbahasa Mandarin atau tangkapan layar spreadsheet padat. Inovasi pertama Qwen-VL adalah 448x448 dan output bounding-box ter-grounding, memungkinkan model menunjuk ke sesuatu.

Video: Video-LLaMA menumpuk encoder per-frame dan memberikannya ke LLM. Itu bekerja untuk klip pendek, bukan untuk video multi-menit di mana sumbu temporal adalah sinyalnya. Tim Qwen menginginkan satu encoder yang memahami waktu.

Output terstruktur: LLaVA memancarkan teks bebas-form. Sebuah agent membutuhkan JSON. Qwen-VL melatih pada format output JSON eksplisit termasuk koordinat bounding-box sebagai teks.

Setiap generasi Qwen-VL memperluas salah satu dari tiga sumbu ini.

## Konsepnya

### Qwen-VL (Agustus 2023)

Generasi pertama: OpenCLIP ViT-bigG/14 sebagai encoder (2.5B params), Q-Former kompatibel-LLama (1-step dengan 256 query), basis Qwen-7B. Kontribusi:

- Resolusi 448x448 (kemudian SOTA untuk VLM terbuka).
- Grounding: dilatih pada pasangan gambar-teks dengan output coordinate-token eksplisit. "The cat is at <box>(112, 204), (280, 344)</box>".
- Training multilingual Mandarin + Inggris sejak awal.

Benchmark saat itu: kompetitif dengan GPT-4V pada Inggris, dominan pada Mandarin. Supervisi grounding adalah headline sebenarnya.

### Qwen2-VL (September 2024) — M-RoPE dan resolusi asli

Qwen2-VL mengganti tumpukan resolusi-tetap + Q-Former dengan encoder ViT resolusi-dinamis-native. Perubahan kunci:

- Resolusi dinamis native. ViT menerima HxW apa pun yang habis dibagi 28 (patch 14 dengan spatial merge 2x). Gambar pada 1120x672 (40x24 merged patch) menghasilkan 960 token visual. Tanpa resize, tanpa tiling, tanpa thumbnail.
- M-RoPE (Multimodal RoPE). Setiap token membawa posisi 3D (t, h, w) alih-alih 1D. Untuk gambar t=0, untuk video t = frame_index. RoPE memutar vektor query/key dengan frekuensi per sumbu. Tanpa tabel positional embedding.
- MLP projector. Buang Q-Former; gunakan MLP 2-layer pada merged patch token.
- Video dengan FPS dinamis. Video di-sample pada 1-2 FPS secara default, tetapi model menerima jumlah frame apa pun.

Hasilnya: Qwen2-VL-7B menyamai GPT-4o pada beberapa benchmark multimodal dan mengalahkannya pada DocVQA (94,5 vs 88,4). Perubahan arsitektur adalah langkah decisif.

### Qwen2.5-VL (Februari 2025) — FPS dinamis + waktu absolut

Pergeseran besar Qwen2.5-VL adalah video. FPS dinamis bukan sekadar "sample lebih banyak frame saat dibutuhkan." Paper memformalkan:

- Token waktu absolut. Alih-alih indeks posisi (frame 0, 1, 2...), gunakan timestamp aktual. "Pada 0:04, kucing melompat." Model melihat token `<time>0.04</time>` yang berselang-seling dengan token frame.
- FPS dinamis. Sample pada 1 FPS untuk footage lambat, 4+ FPS untuk aksi. Pengguna atau trainer memilih; M-RoPE beradaptasi.
- Window attention di ViT. Spatial attention dibatasi window (lokal di dalam blok) untuk throughput; attention global setiap beberapa layer.
- Format output JSON eksplisit. Dilatih pada data tool-call: "{\"tool\": \"click\", \"coords\": [380, 220]}". Agent-ready out of the box.
- Penskalaan MRoPE-v2. Posisi menskala dengan ukuran input maksimum sehingga video 10 menit tidak kehabisan rentang frekuensi.

Benchmark: Qwen2.5-VL-72B mengalahkan GPT-4o pada sebagian besar benchmark video, menyamai Gemini 2.0 pada dokumen, dan menetapkan SOTA model-terbuka untuk GUI grounding (ScreenSpot: akurasi 84% vs 38% untuk GPT-4o).

### Qwen3-VL (November 2025)

Qwen3-VL adalah peningkatan inkremental yang mengonsolidasi alih-alih menciptakan ulang: LLM backbone lebih besar (Qwen3-72B), data training diperluas, OCR membaik, penalaran lebih kuat melalui "thinking mode" Qwen3. ViT dan M-RoPE tetap. Paper berfokus pada peningkatan data dan training di atas arsitektur.

Pelajaran garis keturunan: pada 2025 arsitektur Qwen-VL telah distabilkan. Generasi tambahan menskala komputasi dan data, bukan primitif.

### M-RoPE secara matematis

RoPE klasik memutar query `q` berdimensi `d` oleh posisi `m` menggunakan koordinat berpasangan:

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-MRoPE membagi hidden dim menjadi tiga band. Katakan `d = 96`. Alokasikan 32 dim ke temporal, 32 ke tinggi, 32 ke lebar. Setiap band memutar dengan posisi sumbunya sendiri. Sebuah patch pada (t=5, h=10, w=20) mendapat rotasi `R_t(5)`, `R_h(10)`, `R_w(20)` yang diterapkan pada tiga bandnya.

Token teks menggunakan `t = text_index, h = 0, w = 0` (atau pilihan ternormalisasi), menjaga kompatibilitas. Frame video menggunakan `t = frame_time, h = baris, w = kolom`. Gambar tunggal menggunakan `t = 0`.

Manfaatnya: satu positional encoding menangani teks, gambar, dan video tanpa bercabang kode atau tabel posisi berbeda.

### Logika sampling dynamic-FPS

Diberi video dengan durasi `T` detik dan target-token budget `B`:

1. Hitung FPS maksimum yang kamu mampu: `fps_max = B / (T * tokens_per_frame)`.
2. Pilih target FPS dari `{1, 2, 4, 8}` yang memenuhi `fps <= fps_max`.
3. Jika gerakan tinggi (heuristik optical-flow atau permintaan eksplisit pengguna), pilih FPS lebih tinggi. Jika gerakan rendah, pilih lebih rendah.
4. Sample merata pada FPS yang dipilih; sisipkan token `<time>t</time>` antar frame.

Qwen2.5-VL melatih logika ini secara implisit; saat inference pengguna mengontrol melalui parameter `fps`. Urutan aksi 60 detik pada 4 FPS dengan 81 token per frame = 19440 token, dapat dikelola dalam konteks 32k.

### Output agent terstruktur

Pelatihan agent Qwen2.5-VL secara eksplisit menargetkan tool call terstruktur:

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

Parsing deterministik: JSON.parse atas output model. Bandingkan dengan "click at (1024, 512)" bebas-form yang membutuhkan regex dan penanganan ambiguitas. Pergeseran inilah mengapa skor ScreenSpot Qwen2.5-VL melompat dari 55% Qwen2-VL ke 84%.

```figure
mm-mrope-axes
```

## Gunakan

`code/main.py` mengimplementasikan:

- Perhitungan posisi M-RoPE untuk sekuens ter-pack yang mencampur teks, patch gambar, dan frame video.
- Sampler dynamic-FPS: diberi (durasi, budget, motion_level), pilih FPS dan hasilkan timestamp frame.
- Parser output JSON Qwen2.5-VL main-main yang menangani respons tool-call dengan field koordinat.

Jalankan, lalu rasakan perbedaannya saat kamu menukar FPS tetap dengan FPS dinamis pada video 5 menit.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-qwen-vl-pipeline-designer.md`. Diberi tugas video (monitoring, agen, pengenalan aksi, aksesibilitas), ia menghasilkan konfigurasi Qwen2.5-VL (frame budget, strategi FPS, flag window-attention, mode agent-output) dan perkiraan latensi. Gunakan ini setiap kali kamu mendeploy model keluarga Qwen-VL untuk produk video.

## Latihan

1. Hitung rotasi M-RoPE untuk patch pada (t=3, h=5, w=7) dengan hidden 48 (16 per band, theta dasar 10000). Tampilkan sudut rotasi untuk tiga pasangan pertama di setiap band.

2. Rekaman kamera keamanan 10 menit pada 1 FPS menghasilkan berapa frame? Pada resolusi 384 dengan pool 3x, berapa total token? Apakah konteks default 32k Qwen2.5-VL menanganinya?

3. Pilih FPS untuk reli tenis 30 detik vs demo resep 30 detik vs rekaman agen-UI 30 detik. Benarkan masing-masing dengan logika dynamic-FPS.

4. Qwen2.5-VL membuang Q-Former sepenuhnya. Mengapa MLP sederhana bekerja pada 2025 tetapi tidak pada 2023? (Petunjuk: skala data dan kualitas encoder.)

5. Parse tiga output tool-call JSON Qwen2.5-VL menjadi dict Python. Apa yang gagal untuk JSON rusak dan strategi pemulihan apa yang direkomendasikan cookbook Qwen?

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| M-RoPE | "Multimodal RoPE" | 3D rotary position embedding with temporal, height, and width bands in the hidden dim |
| Dynamic FPS | "Smart sampling" | Frame sampling rate chosen per video based on motion, duration, and token budget |
| Absolute time token | "Timestamp token" | `<time>t</time>` interleaved in the sequence so the model sees actual seconds not frame index |
| Window attention | "Local attention" | Spatial self-attention restricted to small windows for speed; global attention added periodically |
| Structured agent output | "JSON mode" | Training data supervision teaching the VLM to emit parseable JSON with coords and tool names |
| min_pixels / max_pixels | "Resolution bounds" | Per-request Qwen2.5-VL controls bounding total pixel count and therefore token count |
| Grounding | "Point-at-it" | Outputting bounding-box coordinates as text tokens; used since Qwen-VL v1 |

## Bacaan Lanjutan

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
