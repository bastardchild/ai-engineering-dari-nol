# Norms and Distances

> Fungsi jarak-mu mendefinisikan apa artinya "mirip". Pilih yang salah, dan semua yang di hilir rusak.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minutes

## Learning Objectives

- Mengimplementasikan fungsi jarak L1, L2, cosine, Mahalanobis, Jaccard, dan edit distance dari nol
- Memilih metrik jarak yang tepat untuk tugas ML tertentu dan menjelaskan mengapa alternatifnya gagal
- Menghubungkan norm L1 dan L2 dengan regularisasi LASSO dan Ridge serta wilayah batas geometrisnya
- Mendemonstrasikan bagaimana dataset yang sama menghasilkan tetangga terdekat yang berbeda di bawah metrik yang berbeda

## Masalahnya

Kamu punya dua vektor. Mungkin itu word embedding. Mungkin itu profil pengguna. Mungkin itu larik pixel. Kamu perlu tahu: seberapa dekat keduanya?

Jawabannya sepenuhnya bergantung pada fungsi jarak mana yang kamu pilih. Dua titik data bisa menjadi tetangga terdekat di bawah satu metrik dan jauh terpisah di bawah metrik lain. KNN classifier-mu, mesin rekomendasimu, vector database-mu, algoritma clustering-mu, fungsi loss-mu -- semuanya bergantung pada pilihan ini. Pilih yang salah, dan model-mu mengoptimalkan hal yang salah.

Tidak ada jarak terbaik universal. L2 bekerja untuk data spasial. Kemiripan cosine mendominasi NLP. Jaccard menangani set. Edit distance menangani string. Mahalanobis memperhitungkan korelasi. Wasserstein memindahkan massa probabilitas. Masing-masing mengkodekan asumsi berbeda tentang apa artinya "mirip".

Pelajaran ini membangun setiap fungsi jarak utama dari nol, menunjukkan kapan masing-masing adalah alat yang tepat, dan mendemonstrasikan bagaimana data yang sama menghasilkan tetangga terdekat yang sepenuhnya berbeda tergantung pada metrik yang kamu gunakan.

## Konsepnya

### Norm: mengukur magnitudo vektor

Sebuah norm mengukur "ukuran" sebuah vektor. Setiap fungsi jarak antara dua vektor dapat ditulis sebagai norm dari selisihnya: d(a, b) = ||a - b||. Jadi memahami norm berarti memahami jarak.

### Norm L1 (jarak Manhattan)

Norm L1 menjumlahkan nilai absolut dari semua komponen.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

Ini disebut jarak Manhattan karena mengukur seberapa jauh kamu berjalan di grid kota di mana kamu hanya bisa bergerak sepanjang sumbu. Tanpa diagonal.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

Kapan menggunakan L1:
- Data sparse dimensi tinggi (fitur teks, one-hot encodings)
- Saat kamu menginginkan ketahanan terhadap outlier (satu perbedaan besar tidak mendominasi)
- Masalah pemilihan fitur (regularisasi L1 mempromosikan sparsity)

Kaitan dengan regularisasi L1 (Lasso): menambahkan ||w||_1 ke fungsi loss-mu memberi penalti pada jumlah nilai bobot absolut. Ini mendorong bobot kecil menjadi tepat nol, melakukan pemilihan fitur otomatis. Penalti L1 menciptakan wilayah batas berbentuk berlian di ruang bobot, dan sudut-sudut berlian terletak di sumbu di mana beberapa bobot bernilai nol.

Kaitan dengan fungsi loss: Mean Absolute Error (MAE) adalah jarak L1 rata-rata antara prediksi dan target. Ia memberi penalti semua galat secara linier, membuatnya tahan terhadap outlier dibandingkan MSE.

### Norm L2 (jarak Euclidean)

Norm L2 adalah jarak garis lurus. Akar kuadrat dari jumlah komponen kuadrat.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

Ini adalah jarak yang kamu pelajari di kelas geometri. Pythagoras dalam n dimensi.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

Kapan menggunakan L2:
- Data kontinu dimensi rendah-sedang
- Saat skala fitur sebanding
- Jarak fisik (data spasial, pembacaan sensor
- Kemiripan gambar pada level pixel

Kaitan dengan regularisasi L2 (Ridge): menambahkan ||w||_2^2 ke fungsi loss-mu memberi penalti bobot besar. Berbeda dengan L1, ini tidak mendorong bobot ke nol. Ia mengecilkan semua bobot ke arah nol secara proporsional. Penalti L2 menciptakan wilayah batas melingkar, jadi tidak ada sudut di sumbu. Bobot menjadi kecil tapi jarang tepat nol.

Kaitan dengan fungsi loss: Mean Squared Error (MSE) adalah rata-rata dari jarak L2 yang dikuadratkan. Mengkuadratkan memberi penalti galat besar lebih berat daripada galat kecil.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Norm Lp: keluarga umum

L1 dan L2 adalah kasus khusus dari norm Lp:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

Nilai p yang berbeda menghasilkan "bola satuan" berbentuk berbeda (himpunan semua titik pada jarak 1 dari asal):

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### Norm L-infinity (jarak Chebyshev)

Saat p mendekati tak hingga, norm Lp konvergen ke komponen absolut maksimum.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

Jarak antara dua titik ditentukan oleh satu-satunya dimensi di mana keduanya paling berbeda. Semua dimensi lain diabaikan.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

Kapan menggunakan L-infinity:
- Saat deviasi kasus terburuk dalam satu dimensi tertentu penting
- Papan permainan (raja dalam catur bergerak secara L-infinity: satu langkah ke arah mana pun berbiaya 1)
- Toleransi manufaktur (setiap dimensi harus dalam spesifikasi)

### Kemiripan Cosine dan Jarak Cosine

Kemiripan cosine mengukur sudut antara dua vektor, mengabaikan magnitudo mereka.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

Rentangnya dari -1 (arah berlawanan) sampai +1 (arah sama). Vektor tegak lurus punya kemiripan cosine 0.

Jarak cosine mengubahnya menjadi jarak: cosine_distance = 1 - cosine_similarity. Ini berkisar dari 0 (arah identik) sampai 2 (arah berlawanan).

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Mengapa cosine mendominasi NLP dan embedding: dalam teks, panjang dokumen seharusnya tidak memengaruhi kemiripan. Sebuah dokumen tentang kucing yang dua kali lebih panjang dari dokumen lain tentang kucing seharusnya tetap "mirip". Kemiripan cosine mengabaikan magnitudo (panjang) dan hanya peduli pada arah. Dua dokumen dengan distribusi kata yang sama tapi panjang berbeda menunjuk ke arah yang sama dan mendapat kemiripan cosine 1.0.

Kapan menggunakan kemiripan cosine:
- Kemiripan teks (vktor TF-IDF, word embedding, sentence embedding)
- Domain apa pun di mana magnitudo adalah noise dan arah adalah sinyal
- Sistem rekomendasi (vektor preferensi pengguna)
- Pencarian embedding (vector database hampir selalu memakai cosine atau dot product)

### Kemiripan Dot Product vs Kemiripan Cosine

Dot product dari dua vektor adalah:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Kemiripan cosine adalah dinormalisasi dot product oleh kedua magnitudo. Ketika kedua vektor sudah dinormalisasi satuan (magnitudo = 1), dot product dan kemiripan cosine identik.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

Kapan keduanya berbeda: dot product menyertakan informasi magnitudo. Vektor dengan magnitudo lebih besar mendapat skor dot product lebih tinggi. Ini penting di beberapa sistem retrieval di mana kamu menginginkan item "populer" untuk peringkat lebih tinggi. Magnitudo bertindak sebagai sinyal kualitas atau kepentingan implisit.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

Dalam praktiknya:
- Gunakan kemiripan cosine saat kamu menginginkan kemiripan arah murni
- Gunakan dot product saat magnitudo membawa informasi bermakna
- Banyak vector database (Pinecone, Weaviate, Qdrant) membiarkanmu memilih di antara keduanya
- Jika embedding-mu sudah dinormalisasi L2, pilihannya tidak masalah

### Jarak Mahalanobis

Jarak Euclidean memperlakukan semua dimensi secara setara. Tapi jika fitur-mu berkorelasi atau punya skala berbeda, L2 memberikan hasil yang menyesatkan.

Jarak Mahalanobis memperhitungkan struktur kovarians dari data.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

di mana S adalah matriks kovarians dari data.

Secara intuitif: jarak Mahalanobis terlebih dahulu meng-dekorelasi dan menormalisasi data (whitening), lalu menghitung jarak L2 di ruang yang ditransformasi tersebut. Jika S adalah matriks identitas (fitur tidak berkorelasi, varians satuan), jarak Mahalanobis menyusut menjadi jarak Euclidean.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

Kapan menggunakan jarak Mahalanobis:
- Deteksi outlier (titik dengan jarak Mahalanobis besar dari mean adalah outlier)
- Klasifikasi saat fitur punya skala dan korelasi berbeda
- Saat kamu punya cukup data untuk mengestimasi matriks kovarians yang andal
- Quality control dalam manufaktur (pemantauan proses multivariat)

### Kemiripan Jaccard (untuk set)

Kemiripan Jaccard mengukur tumpang tindih antara dua set.

```
J(A, B) = |A intersect B| / |A union B|
```

Rentangnya dari 0 (tidak ada tumpang tindih) sampai 1 (set identik). Jarak Jaccard = 1 - kemiripan Jaccard.

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

Kapan menggunakan Jaccard:
- Membandingkan set tag, kategori, atau fitur
- Kemiripan dokumen berdasarkan keberadaan kata (bukan frekuensi)
- Deteksi near-duplicate (aproksimasi MinHash dari Jaccard)
- Membandingkan vektor fitur biner (data keberadaan/ketidakhadiran)
- Mengevaluasi model segmentasi (Intersection over Union = Jaccard)

### Edit Distance (jarak Levenshtein)

Edit distance menghitung jumlah minimum operasi karakter-tunggal yang diperlukan untuk mengubah satu string menjadi string lain. Operasinya adalah: sisip, hapus, atau ganti.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

Dihitung menggunakan dynamic programming. Isi matriks di mana entri (i, j) adalah edit distance antara i karakter pertama dari string A dan j karakter pertama dari string B.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

Kapan menggunakan edit distance:
- Pemeriksaan dan koreksi ejaan
- Penyejajaran sekuens DNA (dengan operasi berbobot)
- Pencarian string fuzzy
- Deduplikasi data teks yang berantakan

### KL Divergence (bukan jarak, tapi digunakan seperti jarak)

KL divergence mengukur seberapa berbeda satu distribusi probabilitas dari yang lain. Dibahas di Lesson 09, tapi ini termasuk dalam diskusi ini karena orang menggunakannya sebagai "jarak" meskipun bukan demikian.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

Sifat kritis: KL divergence TIDAK simetris.

```
D_KL(P || Q) != D_KL(Q || P)
```

Ini berarti ia gagal pada syarat dasar sebuah metrik jarak. Ia juga tidak memenuhi ketidaksamaan segitiga. Ini adalah divergence, bukan jarak.

Forward KL (D_KL(P || Q)) adalah "mean-seeking": Q mencoba menutupi semua mode dari P. Reverse KL (D_KL(Q || P)) adalah "mode-seeking": Q berfokus pada satu mode dari P.

Saat kamu melihat KL divergence:
- VAE (istilah KL di ELBO mendorong distribusi laten ke prior)
- Knowledge distillation (siswa mencocokkan distribusi guru)
- RLHF (penalti KL menjaga model yang di-fine-tune tetap dekat dengan model dasar)
- Metode policy gradient (membatasi update kebijakan)

### Jarak Wasserstein (Earth Mover's Distance)

Jarak Wasserstein mengukur "kerja" minimum yang diperlukan untuk mengubah satu distribusi probabilitas menjadi yang lain. Bayangkan begini: jika satu distribusi adalah tumpukan tanah dan yang lain adalah lubang, berapa banyak tanah yang harus kamu pindahkan dan seberapa jauh?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

Untuk distribusi 1D, ini menyederhanakan menjadi integral dari selisih absolut dari fungsi distribusi kumulatif:

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Mengapa Wasserstein penting:
- Ini adalah metrik yang sebenarnya (simetris, memenuhi ketidaksamaan segitiga)
- Ia menyediakan gradien bahkan ketika distribusi tidak bertumpang tindih (KL divergence menjadi tak hingga)
- Sifat ini menjadikannya sentral untuk Wasserstein GANs (WGANs), yang memecahkan ketidakstabilan training GAN asli

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

Kapan menggunakan Wasserstein:
- Training GAN (WGAN, WGAN-GP)
- Membandingkan distribusi yang mungkin tidak bertumpang tindih
- Masalah optimal transport
- Retrieval gambar (membandingkan histogram warna)

### Mengapa Tugas Berbeda Membutuhkan Jarak Berbeda

| Task | Best distance | Why |
|------|--------------|-----|
| Text similarity | Cosine | Magnitude is noise, direction is meaning |
| Image pixel comparison | L2 | Spatial relationships matter, features are comparable scale |
| Sparse high-dim features | L1 | Robust, does not amplify rare large differences |
| Set overlap (tags, categories) | Jaccard | Data is naturally set-valued, not vectorial |
| String matching | Edit distance | Operations map to human editing intuition |
| Outlier detection | Mahalanobis | Accounts for feature correlations and scales |
| Comparing distributions | KL divergence | Measures information lost by using Q instead of P |
| GAN training | Wasserstein | Provides gradients even when distributions do not overlap |
| Embeddings (vector DB) | Cosine or dot product | Embeddings are trained to encode meaning in direction |
| Recommendation | Dot product | Magnitude can encode popularity or confidence |
| DNA sequences | Weighted edit distance | Substitution costs vary by nucleotide pair |
| Manufacturing QC | L-infinity | Worst-case deviation in any dimension matters |

### Kaitan dengan Fungsi Loss

Fungsi loss adalah fungsi jarak yang diterapkan pada prediksi vs target.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### Kaitan dengan Regularisasi

Regularisasi menambahkan penalti norm pada bobot ke fungsi loss.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

Mengapa L1 menghasilkan sparsity tapi L2 tidak: bayangkan wilayah batas di ruang bobot 2D. L1 adalah berlian, L2 adalah lingkaran. Kontur fungsi loss (elips) paling mungkin menyentuh berlian di sebuah sudut, di mana satu bobot bernilai nol. Ia menyentuh lingkaran di titik yang halus, di mana kedua bobot tidak nol.

### Pencarian Tetangga Terdekat

Setiap fungsi jarak menyiratkan masalah pencarian tetangga terdekat: diberikan titik kueri, temukan titik terdekat dalam sebuah dataset.

Pencarian tetangga terdekat eksak berbiaya O(n * d) per kueri dalam dataset berisi n titik dengan d dimensi. Untuk dataset besar, ini terlalu lambat.

Algoritma Approximate Nearest Neighbor (ANN) menukar sedikit akurasi dengan peningkatan kecepatan yang masif:

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW (Hierarchical Navigable Small World) adalah algoritma dominan di vector database modern. Ia membangun graf multi-lapis di mana setiap simpul terhubung ke tetangga terdekat perkiraannya. Pencarian dimulai di lapisan atas (jarang, lompatan panjang) dan turun ke lapisan bawah (rapat, lompatan pendek).

```figure
norm-unit-balls
```

## Bangun Sendiri

### Langkah 1: Semua fungsi norm dan jarak

Lihat `code/distances.py` untuk implementasi lengkap. Setiap fungsi dibangun dari nol menggunakan hanya matematika dasar Python.

### Langkah 2: Data yang sama, jarak berbeda, tetangga berbeda

Demo di `distances.py` membuat dataset, memilih titik kueri, dan menunjukkan bagaimana tetangga terdekat berubah tergantung pada metrik jarak. Titik yang "paling dekat" di bawah L1 mungkin tidak paling dekat di bawah L2 atau cosine.

### Langkah 3: Pencarian kemiripan embedding

Kodenya menyertakan mock pencarian kemiripan embedding yang menemukan "dokumen" paling mirip dengan sebuah kueri menggunakan kemiripan cosine vs jarak L2, menunjukkan bahwa peringkatnya bisa berbeda.

## Gunakan

Penggunaan praktis yang paling umum: menemukan item serupa di vector database.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

Saat kamu memanggil `model.encode(text)` lalu mencari di vector database, inilah yang terjadi di balik layar. Model embedding memetakan teks ke vektor. Vector database menghitung kemiripan cosine (atau dot product) antara vektor kuerimu dan setiap vektor yang tersimpan, memakai algoritma ANN untuk menghindari memeriksa semuanya.

## Latihan

1. Hitung jarak L1, L2, dan L-infinity antara (1, 2, 3) dan (4, 0, 6). Verifikasi bahwa L-inf <= L2 <= L1 selalu berlaku untuk setiap pasangan titik. Buktikan mengapa urutan ini dijamin.

2. Buat dua vektor di mana kemiripan cosine tinggi (> 0.9) tapi jarak L2 besar (> 10). Jelaskan secara geometri apa yang terjadi. Lalu buat dua vektor di mana kemiripan cosine rendah (< 0.3) tapi jarak L2 kecil (< 0.5).

3. Implementasikan fungsi yang menerima dataset dan titik kueri serta mengembalikan tetangga terdekat di bawah jarak L1, L2, cosine, dan Mahalanobis. Temukan dataset di mana keempatnya tidak setuju tentang titik mana yang terdekat.

4. Hitung jarak Wasserstein antara [0.5, 0.5, 0, 0] dan [0, 0, 0.5, 0.5] dengan tangan menggunakan metode CDF. Lalu hitung antara [0.25, 0.25, 0.25, 0.25] dan [0, 0, 0.5, 0.5]. Yang mana lebih besar dan mengapa?

5. Implementasikan MinHash untuk kemiripan Jaccard aproksimatif. Hasilkan 100 set acak, hitung Jaccard eksak untuk semua pasangan, dan bandingkan dengan aproksimasi MinHash menggunakan 50, 100, dan 200 fungsi hash. Plot galat aproksimasinya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Norm | "Size of a vector" | A function that maps a vector to a non-negative scalar, satisfying triangle inequality, absolute homogeneity, and zero only for the zero vector |
| L1 norm | "Manhattan distance" | Sum of absolute component values. Produces sparsity in optimization. Robust to outliers |
| L2 norm | "Euclidean distance" | Square root of sum of squared components. The straight-line distance in Euclidean space |
| Lp norm | "Generalized norm" | The p-th root of the sum of p-th powers of absolute components. L1 and L2 are special cases |
| L-infinity norm | "Max norm" or "Chebyshev distance" | The maximum absolute component value. The limit of Lp as p approaches infinity |
| Cosine similarity | "Angle between vectors" | Dot product normalized by both magnitudes. Ranges from -1 to +1. Ignores vector length |
| Cosine distance | "1 minus cosine similarity" | Converts cosine similarity to a distance. Ranges from 0 to 2 |
| Dot product | "Unnormalized cosine" | Sum of component-wise products. Equals cosine similarity times both magnitudes |
| Mahalanobis distance | "Correlation-aware distance" | L2 distance in a space that has been whitened (decorrelated and normalized) using the data covariance matrix |
| Jaccard similarity | "Set overlap" | Size of intersection divided by size of union. For sets, not vectors |
| Edit distance | "Levenshtein distance" | Minimum insertions, deletions, and substitutions to transform one string into another |
| KL divergence | "Distance between distributions" | Not a true distance (not symmetric). Measures extra bits from using Q to encode P |
| Wasserstein distance | "Earth mover's distance" | Minimum work to transport mass from one distribution to another. A true metric |
| Approximate nearest neighbor | "ANN search" | Algorithms (HNSW, LSH, IVF) that find approximately closest points much faster than exact search |
| HNSW | "The vector DB algorithm" | Hierarchical Navigable Small World graph. Multi-layer graph for fast approximate nearest neighbor search |
| L1 regularization | "Lasso" | Adding the L1 norm of weights to the loss. Drives weights to zero (sparsity) |
| L2 regularization | "Ridge" or "weight decay" | Adding the squared L2 norm of weights to the loss. Shrinks weights toward zero without sparsity |
| Elastic Net | "L1 + L2" | Combines L1 and L2 regularization. Handles correlated feature groups better than either alone |

## Bacaan Lanjutan

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - pustaka Meta untuk pencarian ANN skala miliaran
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - paper yang memperkenalkan Earth Mover's distance ke GAN
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - algoritma ANN fundamental
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, di mana kemiripan cosine menjadi default untuk embedding
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - panduan praktis tentang metrik jarak dan algoritma tetangga di scikit-learn
