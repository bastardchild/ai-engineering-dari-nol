# Pra-Pelatihan MiniGPT(Parameter 124 juta)

> GPT-2 Kecil memiliki 124 juta parameter. Itu berarti 12 lapisan transformator, 12 kepala perhatian, dan penyematan 768 dimensi. kamu dapat melatihnya dari awal dengan satu caraGPUdalam beberapa jam. Kebanyakan orang tidak pernah melakukan ini. Mereka menggunakan pos pemeriksaan terlatih. Namun jika kamu tidak melatihnya sendiri, kamu tidak benar-benar memahami apa yang terjadi di dalam model yang kamu gunakan untuk membuat produk.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10, Lessons 01-03 (Tokenizers, Building a Tokenizer, Data Pipelines)
**Time:** ~120 minutes

## Tujuan Pembelajaran

- Terapkan secara penuhGPT-2 arsitektur (124 juta parameter) dari awal: penyematan token, penyematan posisi, blok transformator, dan kepala model bahasa
- Melatih aGPTmodel pada korpus teks menggunakan prediksi token berikutnya dengan kehilangan entropi silang
- Menerapkan pembuatan teks autoregresif dengan pengambilan sampel suhu dantop-k/top-ppenyaringan
- Pantau kurva kerugian pelatihan dan validasi bahwa model mempelajari pola bahasa yang koheren

## Masalahnya

kamu tahu apa itu transformator. kamu telah membaca diagramnya. kamu dapat melafalkan "perhatian adalah semua yang kamu butuhkan" dan menggambar kotak berlabel "Perhatian Multi-Kepala" di papan tulis.

Semua itu tidak berarti kamu memahami apa yang terjadi saat model menghasilkan teks.

Ada 124.438.272 parameter di dalamnyaGPT-2 Kecil (dengan pengikat beban). Masing-masing ditetapkan dengan menjalankan loop pelatihan: penerusan maju, kerugian komputasi, penerusan mundur, pembaruan bobot. Dua belas blok transformator. Dua belas kepala perhatian per blok. Ruang penyematan 768 dimensi. Kosakata 50.257 token. Setiap kali model menghasilkan token, seluruh 124 juta parameter berpartisipasi dalam rantai perkalian matriks tunggal yang mengambil urutan ID token dan menghasilkan distribusi probabilitas pada token berikutnya.

Jika kamu belum pernah membuatnya sendiri, kamu sedang mengerjakan kotak hitam. kamu dapat menggunakanAPI. kamu dapat menyempurnakannya. Namun saat terjadi kesalahan -- saat model berhalusinasi, saat model berulang, saat model menolak mengikuti instruksi -- kamu tidak memiliki model mental untuk *mengapa*.

Pelajaran ini membangunGPT-2 Kecil dari awal. Tidak di PyTorch. Secara numpy. Setiap perkalian matriks terlihat. Setiap gradien dihitung oleh kode kamu. kamu akan melihat dengan tepat bagaimana 124 juta angka berkonspirasi untuk memprediksi kata berikutnya.

## Konsepnya

### ItuGPTArsitektur

GPTadalah model bahasa autoregresif. "Autoregresif" berarti menghasilkan satu token pada satu waktu, masing-masing dikondisikan pada semua token sebelumnya. Arsitekturnya adalah tumpukan blok dekoder transformator.

Berikut adalah grafik komputasi lengkap dari ID token hingga probabilitas token berikutnya:

1. ID Token masuk. Bentuk: (batch_size, seq_len).
2. Pencarian penyematan token. Setiap ID dipetakan ke vektor 768 dimensi. Bentuk: (batch_size, seq_len, 768).
3. Pencarian penyematan posisi. Setiap posisi (0, 1, 2, ...) dipetakan ke vektor 768 dimensi. Bentuk yang sama.
4. Tambahkan penyematan token + penyematan posisi.
5. Lewati 12 blok trafo.
6. Normalisasi lapisan akhir.
7. Proyeksi linier terhadap ukuran kosakata. Bentuk: (batch_size, seq_len, vocab_size).
8. Softmax untuk mendapatkan probabilitas.

Itulah keseluruhan modelnya. Tidak ada konvolusi. Tidak ada pengulangan. Hanya penyematan, perhatian, jaringan feedforward, dan norma lapisan yang ditumpuk 12 kali.

```mermaid
graph TD
    A["Token IDs\n(batch, seq_len)"] --> B["Token Embeddings\n(batch, seq_len, 768)"]
    A --> C["Position Embeddings\n(batch, seq_len, 768)"]
    B --> D["Add"]
    C --> D
    D --> E["Transformer Block 1"]
    E --> F["Transformer Block 2"]
    F --> G["..."]
    G --> H["Transformer Block 12"]
    H --> I["Layer Norm"]
    I --> J["Linear Head\n(768 -> 50257)"]
    J --> K["Softmax\nNext-token probabilities"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#0f3460,color:#fff
    style C fill:#1a1a2e,stroke:#0f3460,color:#fff
    style D fill:#1a1a2e,stroke:#16213e,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
    style I fill:#1a1a2e,stroke:#16213e,color:#fff
    style J fill:#1a1a2e,stroke:#0f3460,color:#fff
    style K fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### Blok Transformator

Masing-masing dari 12 blok mengikuti pola yang sama. Arsitektur pra-norma (GPT-2 menggunakan pre-norm, bukan post-norm seperti trafo asli):

1. LapisanNorma
2. Perhatian Diri Multi-Kepala
3. Koneksi sisa (tambahkan input kembali)
4. LapisanNorma
5. Jaringan Umpan-Maju (MLP)
6. Koneksi sisa (tambahkan input kembali)

Sambungan sisa sangat penting. Tanpanya, gradien akan hilang saat mencapai blok 1 selama propagasi mundur. Dengan mereka, gradien dapat mengalir langsung dari kerugian ke lapisan mana pun melalui jalur "lewati". Inilah sebabnya mengapa kamu dapat menumpuk 12, 32, atau bahkan 96 blok (GPT-4 dikabarkan menggunakan 120).

### Perhatian: Mekanisme Inti

Perhatian diri memungkinkan setiap token melihat setiap token sebelumnya dan memutuskan seberapa banyak yang harus diperhatikan pada masing-masing token. Ini perhitungannya.

Untuk setiap posisi token, hitung tiga vektor dari input:
- **Pertanyaan (Q)**: "Apa yang saya cari?"
- **Kunci (K)**: "Apa isi saya?"
- **Nilai (V)**: "Informasi apa yang saya bawa?"

```
Q = input @ W_q    (768 -> 768)
K = input @ W_k    (768 -> 768)
V = input @ W_v    (768 -> 768)

attention_scores = Q @ K^T / sqrt(d_k)
attention_scores = mask(attention_scores)   # causal mask: -inf for future positions
attention_weights = softmax(attention_scores)
output = attention_weights @ V
```

Topeng sebab akibat itulah yang membuatnyaGPTautoregresif. Posisi 5 dapat menempati posisi 0-5 tetapi tidak dapat menempati posisi 6, 7, 8, dan seterusnya. Hal ini mencegah model "mencurangi" dengan melihat token masa depan selama pelatihan.

**Perhatian multi-kepala** membagi ruang 768 dimensi menjadi 12 kepala dengan masing-masing 64 dimensi. Setiap kepala mempelajari pola perhatian yang berbeda. Satu kepala mungkin melacak hubungan sintaksis (kesepakatan subjek-kata kerja). Yang lain mungkin melacak kesamaan semantik (sinonim). Yang lain mungkin melacak kedekatan posisi (kata-kata terdekat). Keluaran dari ke-12 kepala digabungkan dan diproyeksikan kembali ke 768 dimensi.

```mermaid
graph LR
    subgraph MultiHead["Multi-Head Attention (12 heads)"]
        direction TB
        I["Input (768)"] --> S1["Split into 12 heads"]
        S1 --> H1["Head 1\n(64 dims)"]
        S1 --> H2["Head 2\n(64 dims)"]
        S1 --> H3["..."]
        S1 --> H12["Head 12\n(64 dims)"]
        H1 --> C["Concat (768)"]
        H2 --> C
        H3 --> C
        H12 --> C
        C --> O["Output Projection\n(768 -> 768)"]
    end

    subgraph SingleHead["Each Head Computes"]
        direction TB
        Q["Q = X @ W_q"] --> A["scores = Q @ K^T / 8"]
        K["K = X @ W_k"] --> A
        A --> M["Apply causal mask"]
        M --> SM["Softmax"]
        SM --> MUL["weights @ V"]
        V["V = X @ W_v"] --> MUL
    end

    style I fill:#1a1a2e,stroke:#e94560,color:#fff
    style O fill:#1a1a2e,stroke:#e94560,color:#fff
    style Q fill:#1a1a2e,stroke:#0f3460,color:#fff
    style K fill:#1a1a2e,stroke:#0f3460,color:#fff
    style V fill:#1a1a2e,stroke:#0f3460,color:#fff
```

Pembagian dengan sqrt(d_k) -- sqrt(64) = 8 -- bersifat penskalaan. Tanpanya, perkalian titik menjadi besar untuk vektor berdimensi tinggi, mendorong softmax ke wilayah yang gradiennya hampir nol. Ini adalah salah satu wawasan penting dalam makalah asli “Yang kamu Butuhkan Perhatian”.

### KV Cache: Mengapa Inferensi Cepat

Selama pelatihan, kamu memproses seluruh rangkaian sekaligus. Selama inferensi, kamu menghasilkan satu token dalam satu waktu. Tanpa pengoptimalan, menghasilkan token N memerlukan penghitungan ulang perhatian untuk semua N-1 token sebelumnya. Yaitu O(N^2) per token yang dihasilkan, atau total O(N^3) untuk rangkaian dengan panjang N.

KV Cache memecahkan masalah ini. Setelah menghitung K dan V untuk setiap token, simpanlah. Saat membuat token N+1, kamu hanya perlu menghitung Q untuk token baru dan mencari K dan V yang di-cache dari semua token sebelumnya. Hal ini mengurangi biaya per token dari O(N) menjadi O(1) untuk komputasi K dan V. Perhitungan skor perhatian masih O(N) karena kamu memperhatikan semua posisi sebelumnya, namun kamu menghindari perkalian matriks yang berlebihan pada input.

UntukGPT-2 dengan 12 lapisan dan 12 kepala, cache KV menyimpan 2 (K + V) x 12 lapisan x 12 kepala x 64 dim = 18.432 nilai per token. Untuk rangkaian token 1024, ukurannya sekitar 75MBFP32. Untuk Llama 3 405B dengan 128 lapisan, cache KV untuk satu urutan dapat melebihi 10 GB. Inilah sebabnya mengapa inferensi konteks panjang terikat pada memori.

### Prefill vs Decode: Dua Fase Inferensi

Saat kamu mengirim prompt ke anLLM, inferensi terjadi dalam dua fase berbeda.

**Isi awal** memproses seluruh permintaan kamu secara paralel. Semua token diketahui, sehingga model dapat menghitung perhatian untuk semua posisi secara bersamaan. Fase ini terikat pada komputasi -- theGPUsedang melakukan perkalian matriks pada throughput penuh. Untuk prompt 1000 token diA100, pengisian awal memerlukan waktu sekitar 20-50 md.

**Decode** menghasilkan token satu per satu. Setiap token baru bergantung pada semua token sebelumnya. Fase ini terikat pada memori -- hambatannya adalah membaca bobot model dan cache KVGPUmemori, bukan matematika matriks itu sendiri. ItuGPUInti komputasi sebagian besar menganggur menunggu pembacaan memori. UntukGPT-2, setiap langkah decode membutuhkan waktu yang hampir sama terlepas dari berapa banyak FLOP yang dibutuhkan matmuls, karena bandwidth memori adalah batasannya.

Perbedaan ini penting untuk sistem produksi. Isi ulang skala throughput denganGPUmenghitung (lebih lanjutFLOPS= pengisian awal lebih cepat). Dekode skala throughput dengan bandwidth memori (memori lebih cepat = dekode lebih cepat). Itulah sebabnyaNVIDIAituH100berfokus pada peningkatan bandwidth memori selamaA100-- secara langsung mempercepat pembuatan token.

```mermaid
graph LR
    subgraph Prefill["Phase 1: Prefill"]
        direction TB
        P1["Full prompt\n(all tokens known)"]
        P2["Parallel computation\n(compute-bound)"]
        P3["Builds KV Cache"]
        P1 --> P2 --> P3
    end

    subgraph Decode["Phase 2: Decode"]
        direction TB
        D1["Generate token N"]
        D2["Read KV Cache\n(memory-bound)"]
        D3["Append to KV Cache"]
        D4["Generate token N+1"]
        D1 --> D2 --> D3 --> D4
        D4 -.->|repeat| D1
    end

    Prefill --> Decode

    style P1 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P2 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P3 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D4 fill:#1a1a2e,stroke:#e94560,color:#fff
```

### Lingkaran Pelatihan

Pelatihan sebuahLLMadalah prediksi token berikutnya. Diberikan token [0, 1, 2, ..., N-1], prediksi token [1, 2, 3, ..., N]. Fungsi kerugian adalah entropi silang antara distribusi probabilitas prediksi model dan token aktual berikutnya.

Satu langkah pelatihan:

1. **Forward pass**: Jalankan batch melalui 12 blok. Dapatkan logit (skor pra-softmax) untuk setiap posisi.
2. **Hitung kerugian**: Entropi silang antara logit dan token target (input digeser satu posisi).
3. **Backward pass**: Hitung gradien untuk semua parameter 124M menggunakan backpropagation.
4. **Langkah pengoptimal**: Perbarui bobot.GPT-2 menggunakan Adam dengan pemanasan kecepatan pembelajaran dan peluruhan kosinus.

Jadwal kecepatan pembelajaran lebih penting dari yang kamu harapkan.GPT-2 memanas dari 0 hingga kecepatan pemelajaran puncak selama 2.000 langkah pertama, lalu meluruh mengikuti kurva kosinus. Memulai dengan kecepatan pembelajaran yang tinggi menyebabkan model menyimpang. Mempertahankan tingkat tinggi yang konstan menyebabkan osilasi dalam pelatihan selanjutnya. Pola pemanasan-lalu-peluruhan digunakan oleh setiap jurusanLLM.

### GPT-2 Kecil: Angka

| Component | Shape | Parameters |
|-----------|-------|------------|
| Token embeddings | (50257, 768) | 38,597,376 |
| Position embeddings | (1024, 768) | 786,432 |
| Per-block attention (W_q, W_k, W_v, W_out) | 4 x (768, 768) | 2,359,296 |
| Per-block FFN (up + down) | (768, 3072) + (3072, 768) | 4,718,592 |
| Per-block LayerNorms (2x) | 2 x 768 x 2 | 3,072 |
| Final LayerNorm | 768 x 2 | 1,536 |
| **Total per block** | | **7,080,960** |
| **Total (12 blocks)** | | **85,054,464 + 39,383,808 = 124,438,272** |

Proyeksi keluaran (logits head) berbagi bobot dengan matriks penyematan token. Hal ini disebut pengikatan bobot -- tindakan ini mengurangi jumlah parameter sebesar 38 juta dan meningkatkan performa karena memaksa model menggunakan ruang representasi yang sama untuk masukan dan keluaran.

## Bangun Sendiri

### Langkah 1: Lapisan Penyematan

Penyematan token memetakan masing-masing dari 50.257 kemungkinan token ke vektor 768 dimensi. Penyematan posisi menambahkan informasi tentang lokasi setiap token dalam urutan. Keduanya dijumlahkan.

```python
import numpy as np

class Embedding:
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, embed_dim) * 0.02

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        tok_emb = self.token_embed[token_ids]
        pos_emb = self.pos_embed[:seq_len]
        return tok_emb + pos_emb
```

Deviasi standar 0,02 untuk inisialisasi berasal dariGPT-2 kertas. Terlalu besar dan umpan awal ke depan menghasilkan nilai ekstrem yang mengganggu kestabilan latihan. Terlalu kecil dan keluaran awal hampir sama untuk semua masukan, sehingga sinyal gradien awal tidak berguna.

### Langkah 2: Perhatian Diri dengan Topeng Kausal

Perhatian satu kepala terlebih dahulu. Masker kausal menetapkan posisi masa depan ke tak terhingga negatif sebelum softmax, memastikan setiap posisi hanya dapat menangani dirinya sendiri dan posisi sebelumnya.

```python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, -1, -2 if Q.ndim == 4 else 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V
```

Implementasi softmax mengurangi maksimum sebelum melakukan eksponensial. Tanpa ini, exp(large_number) meluap hingga tak terbatas. Ini adalah trik stabilitas numerik yang tidak mengubah keluaran karena softmax(x - c) = softmax(x) untuk konstanta c apa pun.

### Langkah 3: Perhatian Multi-Kepala

Bagi masukan 768 dimensi menjadi 12 kepala yang masing-masing terdiri dari 64 dimensi. Setiap kepala menghitung perhatian secara independen. Gabungkan hasilnya dan proyeksikan kembali ke 768 dimensi.

```python
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_out = np.random.randn(embed_dim, embed_dim) * 0.02

    def forward(self, x, mask=None):
        batch, seq_len, d = x.shape
        Q = (x @ self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x @ self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x @ self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        attn_out = weights @ V

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)
        return attn_out @ self.W_out
```

Tarian reshape-transpose-reshape adalah bagian yang paling membingungkan dari perhatian multi-kepala. Inilah yang terjadi: tensor (batch, seq_len, 768) menjadi (batch, seq_len, 12, 64), lalu (batch, 12, seq_len, 64). Sekarang masing-masing dari 12 kepala memiliki matriksnya sendiri (seq_len, 64) untuk diperhatikan. Setelah diperhatikan, kita membalik prosesnya: (batch, 12, seq_len, 64) menjadi (batch, seq_len, 12, 64) menjadi (batch, seq_len, 768).

### Langkah 4: Blok Transformator

Satu blok trafo lengkap: LayerNorm, perhatian multi-head dengan sisa, LayerNorm, umpan maju dengan sisa.

```python
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class FeedForward:
    def __init__(self, embed_dim, ff_dim):
        self.W1 = np.random.randn(embed_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

    def forward(self, x):
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)  # GELU approximation: ReLU for simplicity
        return h @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    def forward(self, x, mask=None):
        x = x + self.attn.forward(self.ln1.forward(x), mask)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x
```

Jaringan umpan maju memperluas masukan 768 dimensi menjadi 3.072 dimensi (4x), menerapkan nonlinier, lalu memproyeksikan kembali ke 768. Pola ekspansi-kontraksi ini memberi model representasi internal yang "lebih luas" untuk digunakan di setiap posisi.GPT-2 kegunaanGELUaktivasi, tetapi kami menggunakan ReLU di sini untuk kesederhanaan -- perbedaannya kecil untuk memahami arsitekturnya.

### Langkah 5: PenuhGPTModel

Tumpuk 12 blok trafo. Tambahkan lapisan penyematan di depan dan proyeksi keluaran di belakang.

```python
class MiniGPT:
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072):
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)

        logits = x @ self.embedding.token_embed.T
        return logits

    def count_parameters(self):
        total = 0
        total += self.embedding.token_embed.size
        total += self.embedding.pos_embed.size
        for block in self.blocks:
            total += block.attn.W_q.size + block.attn.W_k.size
            total += block.attn.W_v.size + block.attn.W_out.size
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size
            total += block.ln1.gamma.size + block.ln1.beta.size
            total += block.ln2.gamma.size + block.ln2.beta.size
        total += self.ln_f.gamma.size + self.ln_f.beta.size
        return total
```

Perhatikan pengikatan beban:`logits = x @ self.embedding.token_embed.T`. Proyeksi keluaran menggunakan kembali matriks penyematan token (ditransposisikan). Ini bukan sekadar trik menghemat parameter. Artinya model menggunakan ruang vektor yang sama untuk memahami token (penyematan) dan memprediksinya (output).

### Langkah 6: Lingkaran Pelatihan

Untuk pelatihan nyata yang dijalankan pada parameter 124 juta, kamu memerlukan aGPUdan PyTorch. Loop pelatihan ini mendemonstrasikan mekanisme pada model kecil yang berjalan dalam numpy murni. Kami menggunakan model mungil (4 lapis, 4 kepala, 128 dim) agar mudah diatur.

```python
def cross_entropy_loss(logits, targets):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
    return loss


def train_mini_gpt(text, vocab_size=256, embed_dim=128, num_heads=4,
                   num_layers=4, seq_len=64, num_steps=200, lr=3e-4):
    tokens = np.array(list(text.encode("utf-8")[:2048]))
    model = MiniGPT(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=seq_len, ff_dim=embed_dim * 4
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training tokens: {len(tokens):,}")
    print(f"Config: {num_layers} layers, {num_heads} heads, {embed_dim} dims")
    print()

    for step in range(num_steps):
        start_idx = np.random.randint(0, max(1, len(tokens) - seq_len - 1))
        batch_tokens = tokens[start_idx:start_idx + seq_len + 1]

        input_ids = batch_tokens[:-1].reshape(1, -1)
        target_ids = batch_tokens[1:].reshape(1, -1)

        logits = model.forward(input_ids)
        loss = cross_entropy_loss(logits, target_ids)

        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    return model
```

Kerugian dimulai di dekat ln(vocab_size) -- untuk kosakata tingkat byte 256 token, yaitu ln(256) = 5,55. Model acak memberikan probabilitas yang sama untuk setiap token. Seiring berjalannya pelatihan, kerugian menurun karena model belajar memprediksi pola umum: "th" setelah "t", spasi setelah titik, dan seterusnya.

Dalam produksi, kamu akan menggunakan pengoptimal Adam dengan akumulasi gradien, pemanasan kecepatan pembelajaran, dan pemotongan gradien. Perulangan forward-pass-loss-backward-update identik. Pengoptimalnya lebih canggih.

### Langkah 7: Pembuatan Teks

Pembuatan menggunakan model terlatih untuk memprediksi satu token dalam satu waktu. Setiap prediksi diambil sampelnya dari distribusi keluaran (atau diambil dengan rakus sebagai argmax).

```python
def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / temperature
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(next_token)

    return tokens
```

Suhu mengontrol keacakan. Suhu 1.0 menggunakan distribusi mentah. Suhu 0,5 mempertajamnya (lebih deterministik -- model lebih sering memilih pilihan utamanya). Suhu 1,5 meratakannya (lebih acak -- token dengan probabilitas rendah mendapat peluang lebih besar). Suhu 0,0 adalah decoding serakah (selalu pilih token dengan probabilitas tertinggi).

Itu`tokens[-seq_len:]`jendela diperlukan karena model memiliki panjang konteks maksimum (1024 untukGPT-2). Setelah kamu melampauinya, kamu harus membuang token terlama. Ini adalah "jendela konteks" yang dibicarakan semua orang.

```figure
sampling-decoder
```

## Gunakan

### Pelatihan Penuh dan Demo Generasi

```python
corpus = """The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Self-attention computes relationships between all pairs of positions in a sequence.
Multi-head attention splits the representation into multiple subspaces.
Each attention head can learn different types of relationships.
The feedforward network provides nonlinear transformations at each position.
Residual connections enable gradient flow through deep networks.
Layer normalization stabilizes training by normalizing activations.
Position embeddings give the model information about token ordering.
The causal mask ensures autoregressive generation during training.
Pre-training on large text corpora teaches the model general language understanding.
Fine-tuning adapts the pre-trained model to specific downstream tasks."""

model = train_mini_gpt(corpus, num_steps=200)

prompt = list("The transformer".encode("utf-8"))
output_tokens = generate(model, prompt, max_new_tokens=100, temperature=0.8)
generated_text = bytes(output_tokens).decode("utf-8", errors="replace")
print(f"\nGenerated: {generated_text}")
```

Pada korpus kecil dengan model kecil, teks yang dihasilkan akan menjadi semi-koheren. Ia akan mempelajari beberapa pola tingkat byte dari teks pelatihan tetapi tidak dapat menggeneralisasi caranyaGPT-2 dengan data pelatihan 40GB dan arsitektur parameter 124M penuh. Intinya bukanlah kualitas keluaran. Intinya adalah kamu dapat melacak setiap langkah: menyematkan pencarian, komputasi perhatian, transformasi feedforward, proyeksi logit, softmax, dan pengambilan sampel. Setiap operasi terlihat.

## Rilis

Pelajaran ini menghasilkan`outputs/prompt-gpt-architecture-analyzer.md`-- prompt yang menganalisis pilihan arsitektur apa punGPTmodel -gaya. Berikan kartu model atau laporan teknis dan uraikan alokasi parameter, desain perhatian, dan keputusan penskalaan.

## Latihan

1. Ubah model untuk menggunakan 24 lapisan dan 16 kepala12/12.Hitung parameternya. Bagaimana cara menggandakan kedalaman dibandingkan dengan menggandakan lebar (dimensi penyematan)?

2. MenerapkanGELUfungsi aktivasi (GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))) dan ganti ReLU di jaringan feedforward. Jalankan pelatihan selama 500 langkah dengan setiap aktivasi dan bandingkan kerugian terakhir.

3. Tambahkan cache KV ke fungsi pembangkitan. Simpan tensor K dan V untuk setiap lapisan setelah forward pass pertama, dan gunakan kembali untuk token berikutnya. Ukur kecepatannya: hasilkan 200 token dengan dan tanpa cache dan bandingkan waktu jam dinding.

4. Terapkan pengambilan sampel top-k (hanya pertimbangkan k token dengan probabilitas tertinggi) dan pengambilan sampel top-p (pengambilan sampel inti: pertimbangkan kumpulan token terkecil yang probabilitas kumulatifnya melebihi p). Bandingkan kualitas keluaran pada suhu 0,8 dengan top-k=50 vs top-p=0,95.

5. Buat plotter kurva kerugian pelatihan. Latih model untuk 1000 langkah dan plot kerugian vs langkah. Identifikasi tiga fase: penurunan awal yang cepat (mempelajari byte umum), fase tengah yang lebih lambat (mempelajari pola byte), dan dataran tinggi (overfitting pada korpus kecil). Bentuk kurva ini sama baik kamu melatih model 128-dim atauGPT-4.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Autoregressive | "It generates one word at a time" | Each output token is conditioned on all previous tokens -- the model predicts P(token_n \| token_0, ..., token_{n-1}) |
| Causal mask | "It can't see the future" | An upper-triangular matrix of -infinity values that prevents attention to future positions during training |
| Multi-head attention | "Multiple attention patterns" | Splitting Q, K, V into parallel heads (e.g., 12 heads of 64 dims each for GPT-2) so each head can learn different relationship types |
| KV Cache | "Caching for speed" | Storing computed Key and Value tensors from previous tokens to avoid redundant computation during autoregressive generation |
| Prefill | "Processing the prompt" | The first inference phase where all prompt tokens are processed in parallel -- compute-bound on GPU FLOPS |
| Decode | "Generating tokens" | The second inference phase where tokens are generated one at a time -- memory-bound on GPU bandwidth |
| Weight tying | "Sharing embeddings" | Using the same matrix for input token embeddings and the output projection head -- saves 38M params in GPT-2 |
| Residual connection | "Skip connection" | Adding the input directly to the output of a sublayer (x + sublayer(x)) -- enables gradient flow in deep networks |
| Layer normalization | "Normalizing activations" | Normalizing across the feature dimension to mean 0 and variance 1, with learnable scale and bias parameters |
| Cross-entropy loss | "How wrong the predictions are" | -log(probability assigned to the correct next token), averaged over all positions -- the standard LLM training objective |

## Bacaan Lebih Lanjut

- [Radford et al., 2019 -- "Model Bahasa adalah Pembelajar Multitask Tanpa Pengawasan" (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) -- ituGPT-2 makalah yang memperkenalkan kelompok parameter 124M hingga 1,5B
- [Vaswani dkk., 2017 -- "Yang kamu Butuhkan Hanya Perhatian"](https://arxiv.org/abs/1706.03762) -- kertas trafo asli dengan perhatian perkalian titik berskala dan perhatian multi-kepala
- [Laporan Teknis Llama 3](https://arxiv.org/abs/2407.21783) -- bagaimana Meta menskalakannyaGPTarsitektur hingga parameter 405B dengan GPU 16K
- [Pope dkk., 2022 -- "Menskalakan Inferensi Transformator Secara Efisien"](https://arxiv.org/abs/2211.05102) -- makalah yang memformalkan pra-pengisian vs dekode dan analisis cache KV
