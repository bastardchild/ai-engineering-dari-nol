# Serving Prefix-Cache — RadixAttention dan Reuse KV

> Perlakukan KV cache sebagai resource kelas-satu yang bisa dipakai ulang dan disimpan dalam radix tree, maka scheduling ikut berubah: alih-alih FCFS (first-come, first-served) seperti jadwal vLLM, scheduler yang sadar-cache memprioritaskan request dengan prefix bersama terpanjang — praktisnya traversal radix secara depth-first agar cabang panas tetap residen di HBM. SGLang adalah engine yang membangun serving di sekitar ide ini. Pada Llama 3.1 8B dengan prompt 1K ala-ShareGPT, SGLang mencapai ~16.200 tok/detik berbanding ~12.500 vLLM, keunggulan ~29%. Pada workload RAG yang kaya-prefix keunggulannya mencapai 6,4x. Pada workload berbentuk voice-cloning, hit rate cache melampaui 86%. Ter-deploy pada 400.000+ GPU pada 2026 di xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS. Jebakannya adalah angka 6,4x menguap saat urutan prefix tidak konsisten — urutan adalah tuas engineer.

**Type:** Learn
**Languages:** Python (stdlib, toy radix-tree cache + cache-aware scheduler)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 14 (Agentic RAG)
**Time:** ~75 minutes

## Tujuan Pembelajaran

- Gambarkan RadixAttention: bagaimana prefix disimpan dalam radix tree dan bagaimana blok KV dibagi lintas sekuens yang berakar pada cabang sama.
- Jelaskan scheduling yang sadar-cache dan mengapa FCFS salah untuk trafik kaya-prefix.
- Hitung speedup ekspektasi untuk workload dengan hit rate prefix-cache dan distribusi panjang prompt tertentu.
- Sebutkan disiplin pengurutan prompt yang membuat angka 6,4x menjadi nyata vs upside yang hilang.

## Masalahnya

Serving klasik memperlakukan prompt tiap request sebagai opaque. Bahkan ketika 5.000 request RAG semuanya diawali system prompt 2.000-token yang sama plus preamble retrieval yang sama, vLLM melakukan prefill atas prefix 2.000-token itu 5.000 kali. GPU mengerjakan pekerjaan yang sama berulang-ulang.

Observasinya: prompt dalam workload agenik dan RAG hampir selalu berbagi prefix panjang. System prompt, skema tool, contoh few-shot, header retrieval, riwayat percakapan — semua berulang lintas request. Kalau kamu menyimpan KV cache untuk prefix itu sekali dan memakainya ulang, kamu tidak perlu melakukan prefill lagi.

RadixAttention melakukan persis itu. Token diindeks dalam radix tree; tiap node memiliki blok KV untuk sekuens token pada path dari root. Request baru menelusuri tree: node mana pun yang tokennya cocok memakai ulang blok KV node itu. Biaya prefill menjadi proporsional terhadap sufiks "baru", bukan seluruh prompt.

Tantangannya adalah scheduling. Kalau dua request berbagi prefix 2.000-token dan satu request ketiga hanya berbagi 200 token dari prefix sama, kamu ingin melayani dua request berbagi-panjang itu bersamaan agar prefix panjang tetap di HBM. FCFS melakukan sebaliknya — ia melayani siapa pun yang datang duluan, berpotensi mengusir cabang panas sebelum request prefix-panjang berikutnya tiba.

## Konsepnya

### Radix tree sebagai indeks KV

Radix tree (compact trie) menyimpan sekuens token. Tiap node memiliki rentang token dan blok KV yang dihitung untuk rentang itu. Anak memperpanjang sekuens satu token atau lebih.

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

Request baru masuk dengan system prompt + "Context: <doc A>" + "Question: Carol". Scheduler menelusuri: prefix sistem cocok (124 blok dipakai ulang), cabang doc-A cocok (31 blok dipakai ulang), lalu mengalokasikan blok segar hanya untuk "Question: Carol" (4 blok). Biaya prefill: 4 blok token baru. Tanpa tree: 160 blok. Hemat ~40x pada prefill.

### Scheduling yang sadar-cache

Reuse berbasis radix-tree percuma kalau cache bergejolak. Dua kebijakan kunci:

1. **Dispatch depth-first**. Saat memilih request berikutnya dari antrean, utamakan request yang berakar pada cabang sama dengan himpunan berjalan saat ini. Ini menjaga cabang panas tetap pinned.
2. **LRU pada level cabang, bukan level blok**. Usir seluruh cabang (mulai dari daun yang paling lama tak dipakai) alih-alih blok individual, agar bentuk cache cocok dengan bentuk radix.

FCFS melanggar keduanya. Request yang berbagi 2.000 token antre di belakang request yang berbagi 50, lalu cabang 2.000-token diusir untuk menerima yang 50-token.

### Angka benchmark yang perlu kamu hafal

- Llama 3.1 8B, H100, prompt ShareGPT 1K: SGLang ~16.200 tok/detik vs vLLM ~12.500 (keunggulan ~29%).
- RAG kaya-prefix (sistem sama + dokumen sama, pertanyaan bervariasi): hingga 6,4x pada SGLang.
- Workload voice cloning: hit rate prefix-cache 86,4%.
- Hit rate produksi lintas pelanggan SGLang: 50-99% tergantung disiplin prompt.
- Ter-deploy pada 400.000+ GPU pada 2026.

### Jebakan urutan

Angka 6,4x bergantung pada urutan template-prompt yang konsisten. Kalau klienmu menyusun prompt sebagai `[system, tools, context, history, question]` pada sebagian request dan `[system, context, tools, history, question]` pada sebagian lain, tree tidak bisa menemukan prefix bersama. Apa yang tampak seperti prefix bersama bagi manusia adalah dua sekuens berbeda bagi radix tree.

Tuas engineer: template prompt-mu adalah kunci cache. Tetapkan urutannya. Taruh semua yang immutable (sistem, tool, skema) di depan. Taruh konteks retrieval berikutnya. Taruh pertanyaan pengguna terakhir. Jangan menyisipkan konten dinamis ke dalam prefix.

Kasus nyata dari riset: memindahkan konten dinamis keluar dari prefix yang bisa di-cache menaikkan satu deployment dari hit rate 7% menjadi 74% dalam satu perubahan.

### Di mana RadixAttention menang dan kalah

Menang:
- RAG (preamble retrieval sama, pertanyaan bervariasi).
- Agen (skema tool sama, query bervariasi).
- Chat dengan system prompt panjang.
- Workload suara/vision dengan preamble berulang.

Kalah (kembali ke throughput level-vLLM):
- Generasi single-shot dengan prompt unik (komplesi kode, chat terbuka tanpa system prompt).
- Prompt dinamis di mana setiap request menyisipkan konten unik ke dalam prefix.

### Mengapa ini masalah scheduler, bukan sekadar masalah kernel

Kamu bisa mengimplementasikan reuse KV sebagai trik kernel. Insight SGLang adalah reuse hanya menguntungkan jika scheduler menjaga cabang panas tetap residen. Kebijakan naif "pakai ulang kalau tersedia" akan mengaduk cache pada load campuran. Scheduler berindeks radix-tree inilah yang mengubah trik kernel menjadi keunggulan produksi 29%.

### Interplay dengan vLLM

Kedua sistem bukan kompetitor ketat. Pada 2026 vLLM menambahkan prefix caching (`--enable-prefix-caching`) dan router yang sadar-cache (vLLM Router dalam Rust). Selisih menyempit tetapi tidak hilang sepenuhnya — seluruh stack SGLang radix-first; vLLM menempelkannya belakangan. Untuk workload yang didominasi reuse prefix, SGLang tetap default. Untuk serving general-purpose tanpa pola prefix kuat, vLLM tetap setara atau lebih baik.

```figure
roofline
```

## Gunakan

`code/main.py` mengimplementasikan KV cache radix-tree mainan plus scheduler dengan dua kebijakan: FCFS dan sadar-cache. Menjalankan workload sama lewat keduanya, melaporkan hit rate prefix-cache dan delta throughput. Lalu menjalankan workload "urutan acak" untuk menunjukkan keruntuhan 6,4x.

## Rilis

Pelajaran ini menghasilkan `outputs/skill-radix-scheduler-advisor.md`. Dengan deskripsi workload (bentuk template-prompt, pola retrieval, jumlah tenant konkuren), ia menghasilkan resep pengurutan prompt dan keputusan go/no-go adopsi SGLang.

## Latihan

1. Jalankan `code/main.py`. Bandingkan FCFS dan sadar-cache pada workload sama. Dari mana delta berasal — hemat prefill, hemat decode, atau antrean?
2. Ubah workload agar prompt mengacak `[system, tools, context]` secara acak. Jalankan ulang. Apa yang terjadi pada hit rate? Mengapa?
3. Hitung biaya HBM untuk mempertahankan system prompt 2.000-token tetap residen sebagai satu cabang radix pada Llama 3.1 8B. Bandingkan dengan biaya batch 16-sekuens tanpa reuse prefix.
4. Baca paper SGLang RadixAttention. Jelaskan dalam tiga kalimat mengapa eviksi LRU berbentuk-tree mengalahkan LRU berbentuk-blok pada load kaya-prefix.
5. Pelanggan melaporkan hit rate cache hanya 8%. Sebutkan tiga kemungkinan penyebab dan diagnostik yang akan kamu jalankan untuk masing-masing.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| RadixAttention | "the SGLang thing" | KV cache indexed as a radix tree so shared prefixes reuse blocks |
| Radix tree | "compact trie" | Tree where each node owns a token range and its KV blocks |
| Cache-aware scheduler | "hot-branch-first" | Scheduler that prefers requests sharing the resident branch |
| Prefix-cache hit rate | "how much of your prompt was free" | Fraction of prompt tokens served from reused KV blocks |
| FCFS | "first-come first-served" | Default scheduling that breaks prefix locality |
| Branch-level LRU | "evict the leaf" | Eviction policy matched to radix shape |
| Prompt template ordering | "the cache key" | The prompt's component order determines what the tree can share |
| System prompt pinning | "resident prefix" | Keep the immutable system portion pinned to avoid eviction thrash |

## Further Reading

- [SGLang GitHub](https://github.com/sgl-project/sglang) — source and docs.
- [SGLang documentation](https://sgl-project.github.io/) — RadixAttention and scheduling details.
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — the design reference.
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — benchmark numbers and scheduler rationale.
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) — vLLM's own radix-like implementation, for comparison.
