# Internal Serving Engine — PagedAttention, Continuous Batching, Chunked Prefill

> Throughput serving-engine modern bertumpu pada tiga default yang saling menguatkan, bukan satu trik. PagedAttention selalu aktif. Continuous batching menyuntikkan request baru ke batch aktif di antara iterasi decode. Chunked prefill mengiris prompt panjang agar token decode tidak pernah kelaparan. Nyalakan ketiganya dan Llama 3.3 70B FP8 pada satu H100 SXM5 mendorong 2.200-2.400 tok/detik pada 128 konkuren — sekitar 25% di atas default vLLM sendiri dan 3-4x loop PyTorch naif. Pelajaran ini membaca scheduler dan kernel attention vLLM — engine referensi untuk ketiga teknik — pada level yang bisa kamu gambarkan diagramnya, dan diakhiri dengan continuous batcher mainan di `code/main.py` yang menjadwalkan prefill dan decode seperti vLLM.

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Jelaskan PagedAttention sebagai alokator KV cache: blok, tabel blok, dan mengapa fragmentasi tetap di bawah 4% pada load produksi.
- Gambarkan continuous batching pada level iterasi: bagaimana sekuens yang selesai keluar dari batch dan yang baru bergabung tanpa menguras.
- Jelaskan chunked prefill dalam satu kalimat dan sebutkan metrik latensi mana yang ia lindungi (petunjuk: ekor TTFT, bukan throughput rata-rata).
- Cek kombinasi fitur vLLM terhadap matriks kompatibilitas untuk versimu sebelum menyalakan semua optimasi sekaligus.

## Masalahnya

Loop serve PyTorch naif menjalankan satu request dalam satu waktu: tokenize, prefill, decode hingga EOS, kembalikan. Pada satu pengguna ini bekerja. Pada seratus, ini antrean orang-orang sabar. Perbaikan obvious — static batching — mem-pad setiap request ke prompt terpanjang dalam window, mem-pad setiap decode ke output terpanjang yang diharapkan, dan menahan seluruh batch pada sekuens terlambat. Kamu membayar padding yang tidak pernah dipakai, dan request cepat menunggu yang lambat.

vLLM memecahkan tiga masalah sekaligus. PagedAttention menghentikan fragmentasi KV cache yang memakan 60-80% memori GPU seperti alokasi contiguous klasik. Continuous batching membiarkan request bergabung dan keluar batch di antara tiap iterasi decode, sehingga batch selalu penuh kerja nyata. Chunked prefill memecah prompt 32k-token menjadi irisan ~512-token yang disisipkan dengan decode, sehingga prompt panjang tidak membekukan setiap token decode di GPU.

Default produksi 2026 adalah ketiganya aktif. Kamu perlu memahami fungsi masing-masing karena mode kegagalannya semua ada di scheduler, bukan di model.

## Konsepnya

### PagedAttention sebagai sistem memori virtual

KV cache adalah `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element` per sekuens. Untuk Llama 3.3 70B pada 8192 token, itu sekitar 1,25 GB per sekuens dalam BF16. Kalau kamu mencadangkan 8192 slot untuk setiap request tetapi rata-rata request hanya memakai 1500 token, kamu membuang sekitar 82% HBM yang kamu cadangkan. Batching klasik membayar pemborosan ini.

PagedAttention meminjam ide dari memori virtual OS. KV cache tidak contiguous per sekuens. Ia dialokasikan dalam blok berukuran tetap (default 16 token). Setiap sekuens punya tabel blok yang memetakan posisi token logisnya ke ID blok fisik. Saat sekuens tumbuh melewati blok yang dialokasikan, satu blok lagi ditambahkan. Saat selesai, blok-bloknya kembali ke pool.

Fragmentasi turun dari 60-80% (klasik) ke di bawah 4% (PagedAttention). Kamu tidak menyalakan PagedAttention dengan flag — ia satu-satunya alokator yang dibawa vLLM. Kenopnya adalah `--gpu-memory-utilization` (default 0,9), yang memberi tahu vLLM berapa banyak HBM yang dicadangkan untuk blok KV setelah memuat bobot dan aktivasi.

### Continuous batching pada level iterasi

"Dynamic batching" lama menunggu window (misal 10 ms) terisi batch, lalu menjalankan prefill + decode + decode + decode hingga setiap sekuens selesai. Sekuens cepat selesai duluan lalu idle sementara GPU menyelesaikan yang lambat.

Continuous batching beroperasi di antara tiap langkah decode. Sebut himpunan sekuens berjalan sebagai daftar `RUNNING`. Pada tiap iterasi:

1. Sekuens di `RUNNING` yang baru mencapai EOS atau max_tokens dikeluarkan.
2. Scheduler melihat antrean tunggu. Kalau ada blok KV bebas, ia menerima sekuens baru (prefill atau lanjutan).
3. Forward pass berjalan di atas apa pun yang kini ada di `RUNNING`, mengemisikan satu token baru per sekuens.

Ukuran batch tidak pernah di-pad ke angka tetap. Sekuens pada posisi output berbeda berbagi satu forward fused. Pada 2026 vLLM ini disebut scheduler `V1`. Invarian kuncinya: scheduler berjalan sekali per iterasi decode, bukan sekali per request.

### Chunked prefill melindungi ekor TTFT

Prefill bersifat compute-bound. Prompt 32k-token pada Llama 3.3 70B butuh ~800 ms prefill murni pada satu H100. Selama prefill berjalan, token decode untuk setiap sekuens lain dalam batch menunggu. Dalam loop serving, latensi token-pertama (TTFT) satu prompt panjang menjadi blip inter-token latency (ITL) bagi puluhan pengguna lain.

Chunked prefill membagi prefill menjadi chunk berukuran tetap (default 512 token) dan menjadwalkan tiap chunk sebagai unit. Di antara chunk, scheduler bisa memajukan sekuens decode sebanyak satu token. Kamu menukar sedikit kenaikan latensi prefill absolut (beberapa ms per chunk) dengan jitter waktu-decode yang jauh lebih rendah. P99 ITL pada load campuran turun dari ~50 ms menjadi ~15 ms dalam benchmark yang dipublikasikan.

### Tiga default saling berinteraksi

Ketiga fitur saling mengasumsikan. PagedAttention memberi scheduler resource KV berbutir-halus untuk dipertukarkan. Continuous batching butuh resource berbutir-halus itu agar menerima sekuens baru tidak memaksa reshuffle global. Chunked prefill adalah keputusan yang dibuat scheduler pada daftar `RUNNING` yang sama — ia satu kebijakan scheduler lagi, bukan sistem terpisah.

Kamu tidak perlu tahu setiap flag. Kamu perlu tahu apa yang dioptimalkan scheduler: goodput di bawah budget blok-KV, dengan tunduk pada irisan chunked prefill.

### Cek matriks kompatibilitas

Cek setiap kombinasi fitur terhadap matriks kompatibilitas untuk versi vLLM persismu sebelum menyalakan semuanya sekaligus, karena apa yang bisa dikombinasikan berubah antar rilis. Pada v0.18.0 matriks fitur menandai speculative decoding kompatibel dengan chunked prefill dan prefix caching, dan halaman speculative-decoding mencantumkan dua ketidakcocokan yang diketahui: pipeline parallelism hingga v0.15.0, dan spekulasi draft-model hingga v0.10.0. Untuk metode draft itu sendiri, default 2026 sering kali EAGLE-3 (`"method": "eagle3"`), dibahas di Fase 17 · 05.

### Angka yang perlu kamu ingat

- Llama 3.3 70B FP8, H100 SXM5, 128 konkuren, ketiganya aktif: 2.200-2.400 tok/detik.
- Model sama, default vLLM (tanpa chunked prefill): ~1.800 tok/detik.
- Model sama, loop forward PyTorch naif: ~600 tok/detik.
- Pemborosan fragmentasi KV pada PagedAttention di load produksi: <4%.
- P99 ITL pada load campuran: ~15 ms dengan chunked prefill, ~50 ms tanpa.

### Seperti apa scheduler itu

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` adalah persis loop ini dalam Python stdlib dengan jumlah token dan latensi forward palsu. Menjalankannya menunjukkan bagaimana chunked prefill menjaga sekuens decode tetap hidup selama prefill panjang.

```figure
tensor-parallel
```

## Gunakan

`code/main.py` mensimulasikan scheduler ala-vLLM dengan fitur yang bisa di-toggle. Jalankan untuk melihat:

- Mode `NAIVE`: satu request dalam satu waktu, tanpa batching.
- Mode `STATIC`: pad dan tunggu, batching klasik.
- Mode `CONTINUOUS`: admisi dan pelepasan level-iterasi.
- Mode `CONTINUOUS + CHUNKED`: irisan prefill disisipkan dengan decode.

Output menunjukkan throughput total (token per detik virtual), rata-rata TTFT, dan P99 ITL. Baris `CONTINUOUS + CHUNKED` seharusnya dominan pada trafik campuran.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-vllm-scheduler-reader.md`. Dengan konfigurasi serving (ukuran batch, utilisasi memori KV, ukuran chunked prefill, konfigurasi spekulatif), ia menghasilkan diagnosis scheduler yang menyebut mana dari tiga default yang menjadi bottleneck dan apa yang perlu di-tune.

## Latihan

1. Jalankan `code/main.py`. Bandingkan `STATIC` dengan `CONTINUOUS` pada workload berisi campuran request pendek dan panjang. Dari mana selisih throughput berasal — efisiensi prefill, efisiensi decode, atau latensi ekor?
2. Ubah toy scheduler untuk menambahkan `--max-num-batched-tokens`. Berapa nilai tepat untuk H100 yang menjalankan Llama 3.3 70B FP8? (Petunjuk: ia fungsi dari ukuran blok KV dan jumlah blok bebas, bukan HBM mentah.)
3. Baca ulang catatan rilis vLLM v0.18.0. Kombinasi flag mana yang saling eksklusif? Daftarkan.
4. Hitung pemborosan fragmentasi KV cache untuk trace 1.000 request dengan rata-rata 1.500 token output, std 600 token, pada (a) alokasi contiguous per-request pada maks 8192, (b) PagedAttention dengan blok 16-token.
5. Jelaskan dalam satu paragraf mengapa chunked prefill membantu P99 ITL tetapi tidak membantu throughput secara isolasi. Dari mana kemenangan throughput berasal dalam praktik?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| PagedAttention | "the KV trick" | Fixed-size block allocator for KV cache; fragmentation <4% |
| Block table | "the page table" | Per-sequence map from logical token position to physical KV block |
| Continuous batching | "dynamic batching, but right" | Admit/release decisions made every decode iteration |
| Chunked prefill | "prefill splitting" | Break long prefill into 512-token slices interleaved with decode |
| TTFT | "first token time" | Prefill + queue + network; dominated by prefill at long prompts |
| ITL | "inter-token latency" | Time between consecutive decode tokens; dominated by batch size |
| Goodput | "throughput that meets SLO" | Tokens/sec where every request still hit TTFT and ITL targets |
| V1 scheduler | "the new scheduler" | vLLM's 2026 scheduler; runs continuous batching with chunked prefill |
| `--gpu-memory-utilization` | "the memory knob" | Fraction of HBM reserved for KV blocks after weights and activations |

## Further Reading

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — official source on chunked-prefill and speculative-decoding compatibility.
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 release cadence and version-specific behavior.
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — the original write-up that still defines how to think about the allocator.
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — fragmentation analysis and scheduler design.
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — detailed V1 scheduler walkthrough with flame graphs.
