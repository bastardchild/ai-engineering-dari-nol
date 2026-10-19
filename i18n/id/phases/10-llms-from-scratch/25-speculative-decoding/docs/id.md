# Decoding Spekulatif danEAGLE

> Sebuah perbatasanLLMmenghasilkan satu token memerlukan penerusan penuh pada miliaran parameter. Ketentuan forward pass tersebut terlalu banyak: sering kali model yang jauh lebih kecil dapat menebak 3-5 token berikutnya dengan benar, dan model besar hanya perlu *memverifikasi* tebakannya. Jika tebakannya benar, kamu mendapat 5 token dengan harga satu. Penguraian kode spekulatif (Leviathan dkk. 2023) membuat hal ini menjadi tepat, danEAGLE-3 (2025) mendorong tingkat penerimaan menjadi ~4,5 token per verifikasi — peningkatan 4-5x pada distribusi keluaran yang sesuai.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## Masalahnya

Dekode throughput untuk model kelas 70B aktifH100biasanya 40-80tokens/second.Setiap token memerlukan forward pass penuh untuk membaca semua bobot modelHBM. kamu tidak dapat memperkecil model tanpa mengubah keluarannya. kamu tidak dapat menambah ukuran batch melebihi memori. kamu terjebak - kecuali kamu dapat membiarkan model mengeluarkan lebih dari satu token per penerusan.

Generasi autoregresif pada dasarnya terlihat serial:`x_{t+1} = sample(p(· | x_{1:t}))`. Namun ada peluang konkurensi. Jika kamu memiliki prediktor murah yang mengatakan "4 token berikutnya mungkin [a, b, c, d]" kamu dapat memverifikasi semua 5 posisi dalam **satu forward pass dari model besar** dan menerima awalan pencocokan terpanjang.

Leviathan, Kalai, Matias (2023, "Inferensi Cepat dari Transformers melalui Penguraian Spekulatif") membuat hal ini tepat melalui cara yang cerdasaccept/rejectaturan yang mempertahankan distribusi pengambilan sampel model target. Distribusi keluaran yang sama, 2-4× lebih cepat.

## Konsepnya

### Pengaturan Dua Model

- **Model sasaran**`M_p`: model besar, lambat, dan berkualitas tinggi yang sebenarnya kamu inginkan sampelnya. Distribusi:`p(x)`.
- **Model draf**`M_q`: model kecil, cepat, dan berkualitas lebih rendah. Distribusi:`q(x)`. 5-30× lebih kecil.

Per langkah:

1. Draf model diusulkan`K`token secara otomatis:`x_1, x_2, ..., x_K ~ q`.
2. Model target berjalanONEmaju melewati semuanya`K+1`posisi secara paralel, menghasilkan`p(x_k)`untuk setiap token yang diusulkan.
3. Accept/rejectsetiap token dari kiri ke kanan melalui aturan pengambilan sampel penolakan yang dimodifikasi di bawah. Terima awalan yang paling lama cocok.
4. Jika ada token yang ditolak, ambil contoh penggantinya dari distribusi yang telah diperbaiki dan hentikan. Jika tidak, cicipi satu token bonus dari`p(· | x_1...x_K)`.

Jika drafnya cocok dengan target dengan sempurna, kamu mendapatkan token K+1 per target maju. Jika draft salah pada posisi 1 maka kamu hanya mendapat 1 token saja.

### Aturan Ketepatan

Penguraian kode spekulatif **terbukti setara dalam distribusi dengan pengambilan sampel dari p**. Aturan penolakan:

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

dimana`(p - q)+`menunjukkan bagian positif dari perbedaan poin. Ketika rancangan dan target disepakati (`p ≈ q`) penerimaannya mendekati 1. Jika tidak setuju, distribusi sisa dikonstruksikan sedemikian rupa sehingga keseluruhan sampel masih tetap sama persis`p`.

**Kasus serakah.** Untuk pengambilan sampel suhu=0, cukup periksa`argmax(p) == x_t`. Jika ya, terima; jika tidak, keluarkan`argmax(p)`dan berhenti.

### Percepatan yang Diharapkan

Jika tingkat penerimaan tingkat token model draf adalah`α`, token yang diharapkan dihasilkan per pass target-forward adalah:

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

Di`α = 0.8, K = 4`:`(1 - 0.8^5)/(1 - 0.8) = 3.36`token per penerusan. Satu target ke depan memerlukan biaya kira-kira`cost_q * K + cost_p`(K langkah draf ditambah satu verifikasi target). Jika`cost_p >> cost_q * K`rasio percepatannya adalah`3.36× / 1 = 3.36×`pada keluaran.

Satu-satunya parameter nyata adalah`α`, yang sepenuhnya bergantung pada penyelarasan draf-target. Draf yang bagus adalah segalanya.

### Pelatihan Draf: Distilasi

Model kecil yang acak menghasilkan draf yang buruk. Resep standarnya adalah menyaring dari target:

1. Pilih arsitektur kecil (~1 miliar untuk target 70 miliar, ~500 juta untuk target 7 miliar).
2. Jalankan model target pada korpus teks besar; menyimpan distribusi token berikutnya.
3. Latih rancangan tersebut dengan divergensi KL terhadap distribusi target (bukan terhadap token kebenaran dasar).

Hasilnya:`α`biasanya 0,6-0,8 pada pengkodean, 0,7-0,85 pada obrolan bahasa alami. Mempercepat 2-3× dalam produksi.

### EAGLE: Penyusunan Pohon + Penggunaan Kembali Fitur

Li, Wei, Zhang, Zhang (2024, "EAGLE: Pengambilan Sampel Spekulatif Memerlukan Pemikiran Ulang Ketidakpastian Fitur") mengamati dua inefisiensi dalam penguraian kode spekulatif standar:

1. Draf melakukan K langkah serial, masing-masing tumpukan penuh. Namun draf tersebut dapat menggunakan kembali fitur target (status tersembunyi) dari verifikasi terbaru — target telah menghitung representasi kaya yang diperoleh ulang dari awal oleh draf tersebut.
2. Draf tersebut menghasilkan rantai linier. Jika draf dapat menghasilkan *pohon* kandidat (setiap node memiliki beberapa tebakan), satu forward pass target dapat memverifikasi beberapa jalur kandidat secara paralel melalui masker perhatian pohon, dan memilih cabang terpanjang yang diterima.

EAGLE-1 perubahan:
- Draf masukan = keadaan tersembunyi akhir target pada posisi t, bukan token mentah.
- Rancangan arsitektur = 1 lapisan dekoder transformator (bukan model kecil yang terpisah).
- Output = pohon K = 4-8 kandidat per kedalaman, kedalaman 4-6.

EAGLE-2 (2024) menambahkan topologi pohon dinamis: pohon tumbuh lebih luas di tempat yang rancangannya tidak pasti dan tetap sempit di tempat yang yakin. Menaikkan`α_effective`tanpa meningkatkan biaya verifikasi.

EAGLE-3 (Li dkk. 2025, "EAGLE-3: Meningkatkan Akselerasi Inferensi Model Bahasa Besar melalui Tes Waktu Pelatihan") menghilangkan ketergantungan fitur lapisan atas yang tetap dan melatih draf dengan kerugian "simulasi waktu pengujian" baru — draf dilatih berdasarkan keluaran yang cocok dengan distribusi waktu pengujian target, bukan distribusi pelatihan yang dipaksakan oleh guru. Tingkat penerimaan naik dari 0,75 (EAGLE-2) hingga 0,82 (EAGLE-3), dan berartitokens/verifydari 3,0 hingga 4,5.

### Verifikasi Perhatian Pohon

Saat draf menghasilkan keluaran berupa pohon, model target akan memverifikasinya dalam satu penerusan menggunakan **topeng perhatian pohon** — masker sebab-akibat yang mengkodekan topologi pohon, bukan garis murni. Setiap token hanya melayani nenek moyangnya di pohon. Lulus verifikasi masih satu maju, satu matmul; topeng topologi hanya memerlukan beberapa entri KV tambahan.

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

Jika`a, b`bersaing dengan kandidat token pertama dan`c, d, e, f`adalah kandidat token kedua, keenam posisi diverifikasi dalam satu forward pass. Outputnya adalah awalan terpanjang di sepanjang jalur yang diterima.

### Saat Menang, Saat Tidak

**Menang:**
- Obrolan / penyelesaian dengan teks yang dapat diprediksi (kode, bahasa Inggris umum, keluaran terstruktur).`α`tinggi.
- Pengaturan dengan tidak terpakaiGPUkomputasi selama decode (fase terikat memori). Penyusunan pohon menggunakan FLOP yang tersedia.

**Kalah / tidak menang:**
- Keluaran yang sangat stokastik (penulisan kreatif pada suhu tinggi).`α`turun ke arah`1/|vocab|`.
- Penyajian batch dengan konkurensi yang sangat tinggi — batching sudah memenuhi FLOP, hanya ada sedikit ruang untuk verifikasi pohon.
- Model target yang sangat kecil dimana drafnya tidak jauh lebih kecil.

Toko produksi biasanya melaporkan 2-3× kecepatan jam dinding pada obrolan, 3-5× pada pembuatan kode, dan mendekati nol pada penulisan kreatif.

```figure
speculative-decoding
```

## Bangun Sendiri

`code/main.py`:

- Sebuah referensi`speculative_decode(target, draft, prompt, K, temperature)`yang menerapkan aturan penolakan yang tepat dan memverifikasi bahwa aturan tersebut mempertahankan distribusi target (KL empiris <0,01 vs pengambilan sampel target biasa).
- SebuahEAGLE-perancang pohon bergaya yang membangun pohon kedalaman-K dengan percabangan bagian atas.
- Pembuat topeng perhatian pohon yang menghasilkan pola sebab akibat yang tepat untuk pemverifikasi.
- Harness tingkat penerimaan yang menjalankan keduanya pada LM kecil (yang disulingGPT-2-kecil dari aGPT-2-target sedang).

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## Gunakan

- **vLLM** dan **SGLang** mengirimkan decoding spekulatif kelas satu. Di vLLM, lulus`--speculative-config`sebuahJSONkeberatan dengan`method`,`model`, dan`num_speculative_tokens`;EAGLE-3 adalah`"method": "eagle3"`.
- **NVIDIATensorRT-LLM** mendukung Medusa danEAGLEpohon secara asli.
- **Model draf referensi**:`Qwen/Qwen3-0.6B`(draf untuk Qwen3-32B),`meta-llama/Llama-3.2-1B-Instruct`(draf untuk Llama 3.x 70B).
- **Kepala Medusa** (Cai et al. 2024, "Medusa: SederhanaLLMKerangka Akselerasi Inferensi dengan Beberapa Kepala Decoding"): sebagai ganti model draf, tambahkan kepala prediksi paralel K ke target itu sendiri. Lebih mudah untuk diterapkan, penerimaannya sedikit lebih rendah dibandingkanEAGLE.

## Rilis

Pelajaran ini menghasilkan`outputs/skill-speculative-tuning.md`— keterampilan yang memprofilkan beban kerja model target dan memilih: model draf, K (panjang draf), lebar pohon, suhu, dan kapan harus kembali ke dekode biasa.

## Latihan

1. Terapkan aturan penolakan yang tepat dan verifikasi secara empiris. Jalankan 10 ribu sampel melalui`speculative_decode`dan melalui pengambilan sampel sasaran biasa; menghitung jarak TV antara dua distribusi output. Seharusnya <0,01.

2. Hitung rumus percepatan. Diberikan tetap`α`dan`K`, plot token yang diharapkan per target maju. Temukan K optimal untuk α ∈ {0.5, 0.7, 0.9}.

3. Latih draf kecil. Ambil 124MGPT-2 target dan suling 30MGPT-2 draft pada 100 juta token dengan kerugian KL. Ukur`α`pada teks yang ditahan. Diharapkan: 0,6-0,7.

4. MenerapkanEAGLE- penyusunan pohon bergaya. Daripada membuat rantai, buatlah rancangan keluaran 3 cabang teratas di setiap kedalaman. Bangun topeng perhatian pohon. Verifikasikan target menerima cabang terpanjang yang benar.

5. Ukur mode kegagalan. Jalankan dekode spekulatif pada suhu=1,5 (stokastisitas tinggi). Tunjukkan α runtuh dan algoritme lebih lambat dibandingkan dekode biasa karena overhead draft.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Target model | "The big model" | The slow, high-quality model you want samples from (p distribution) |
| Draft model | "The speculator" | The small, fast predictor (q distribution); 5-30x smaller |
| K / draft length | "Look-ahead" | Number of speculated tokens per verify pass |
| α / acceptance rate | "Hit rate" | Per-token probability that the draft's proposal is accepted |
| Exact rejection rule | "The accept test" | r < p/q compare that preserves target's distribution |
| Residual distribution | "Corrected p-q" | (p - q)+ / ||(p - q)+||_1, the distribution to sample from on rejection |
| Tree drafting | "Branching speculation" | Draft outputs a tree of candidates, verified in one pass with tree-structured attention mask |
| Tree attention mask | "Topological mask" | Causal mask encoding the tree topology so each node attends only to its ancestors |
| Medusa heads | "Parallel heads" | K extra prediction heads on the target itself; no separate draft model |
| EAGLE feature reuse | "Hidden-state draft" | Draft input is target's last hidden state, not raw tokens, shrinking the draft |
| Test-time simulation loss | "EAGLE-3 training" | Train draft on outputs matching target's test-time distribution, not teacher forcing |

## Bacaan Lebih Lanjut

- [Leviathan, Kalai, Matias, 2023 — "Inferensi Cepat dari Transformers melalui Penguraian Kode Spekulatif"](https://arxiv.org/abs/2211.17192) — aturan penolakan yang tepat dan analisis percepatan teoritis
- [Chen, Borgeaud, Irving dkk., 2023 — "Mempercepat Dekode Model Bahasa Besar dengan Pengambilan Sampel Spekulatif"](https://arxiv.org/abs/2302.01318) — makalah pengambilan sampel spekulatif secara bersamaan di DeepMind
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: SederhanaLLMKerangka Akselerasi Inferensi dengan Beberapa Kepala Decoding"](https://arxiv.org/abs/2401.10774) — alternatif kepala paralel untuk model rancangan
- [Li, Wei, Zhang, Zhang, 2024—"EAGLE: Pengambilan Sampel Spekulatif Memerlukan Pemikiran Ulang Ketidakpastian Fitur"](https://arxiv.org/abs/2401.15077) — penggunaan kembali fitur dan penyusunan pohon
- [Li dkk., 2024— "EAGLE-2: Inferensi Model Bahasa yang Lebih Cepat dengan Pohon Draf Dinamis"](https://arxiv.org/abs/2406.16858) — topologi pohon dinamis
- [Li dkk., 2025— "EAGLE-3: Meningkatkan Akselerasi Inferensi Model Bahasa Besar melalui Tes Waktu Pelatihan"](https://arxiv.org/abs/2503.01840) — pencocokan waktu ujian waktu kereta
- [Fu, Haotian, Peng dkk., 2024 - "Putuskan Ketergantungan Berurutan dariLLMInferensi Menggunakan Penguraian Lookahead"](https://arxiv.org/abs/2402.02057) —Jacobi/lookaheaddecoding, alternatif bebas spekulan
