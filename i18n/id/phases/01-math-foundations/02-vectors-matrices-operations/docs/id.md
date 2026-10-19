# Vektor, Matriks & Operasi

> Setiap neural network hanyalah perkalian matriks dengan langkah tambahan.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## Tujuan Pembelajaran

- Bangun kelas Matrix dengan operasi element-wise, perkalian matriks, transpose, determinan, dan invers
- Bedakan perkalian element-wise dari perkalian matriks dan jelaskan kapan masing-masing berlaku
- Implementasikan satu dense layer neural network (`relu(W @ x + b)`) hanya memakai kelas Matrix dari-nol
- Jelaskan aturan broadcasting dan cara kerja penjumlahan bias dalam framework neural network

## Masalahnya

Kamu ingin membangun neural network. Kamu membaca kode dan melihat ini:

```
output = activation(weights @ input + bias)
```

`@` itu adalah perkalian matriks. `weights` adalah matriks. `input` adalah vektor. Kalau kamu tidak tahu apa yang dilakukan operasi-operasi itu, baris ini adalah sihir. Kalau kamu tahu, baris ini adalah seluruh forward pass sebuah layer dalam tiga operasi.

Setiap gambar yang diproses model-mu adalah matriks nilai piksel. Setiap embedding kata adalah vektor. Setiap layer dari setiap neural network adalah transformasi matriks. Kamu tidak bisa membangun sistem AI tanpa lancar operasi matriks, sama seperti kamu tidak bisa menulis kode tanpa memahami variabel.

Pelajaran ini membangun kelancaran itu dari nol.

## Konsepnya

### Vektor: daftar angka yang terurut

Vektor adalah daftar angka yang memiliki arah dan magnitudo. Di AI, vektor merepresentasikan titik data, fitur, atau parameter.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

Vektor 2D `[3, 4]` menunjuk ke koordinat (3, 4) pada sebuah bidang. Panjangnya (magnitudo) adalah 5 (segitiga 3-4-5).

### Matriks: kisi-kisi angka

Matriks adalah kisi 2D. Baris dan kolom. Matriks m x n memiliki m baris dan n kolom.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

Dalam neural network, matriks bobot mengubah vektor input menjadi vektor output. Layer dengan 784 input dan 128 output memakai matriks bobot 128x784.

### Mengapa shape penting

Perkalian matriks punya aturan ketat: `(m x n) @ (n x p) = (m x p)`. Dimensi-dimensi dalam harus cocok.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

Kalau kamu mendapat shape mismatch error di PyTorch, inilah alasannya.

### Peta operasi

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | Element-wise combine | Adding bias to output |
| Scalar multiply | Scale every element | Learning rate * gradients |
| Matrix multiply | Transform vectors | Layer forward pass |
| Transpose | Flip rows and columns | Backpropagation |
| Determinant | Single number summary | Checking invertibility |
| Inverse | Undo a transformation | Solving linear systems |
| Identity | Do-nothing matrix | Initialization, residual connections |

### Element-wise vs perkalian matriks

Perbedaan ini terus-menerus menjebak pemula.

Element-wise: kalikan posisi-posisi yang bersesuaian. Kedua matriks harus memiliki shape yang sama.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Perkalian matriks: dot product dari baris dan kolom. Dimensi-dimensi dalam harus cocok.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Operasi berbeda, hasil berbeda, aturan berbeda.

### Broadcasting

Ketika kamu menambahkan vektor bias ke matriks output, shape-nya tidak cocok. Broadcasting meregangkan array yang lebih kecil agar pas.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Setiap framework modern melakukan ini otomatis. Memahaminya mencegah kebingungan ketika shape tampak salah tetapi kode tetap berjalan.

```figure
vector-projection
```

## Bangun Sendiri

### Langkah 1: Kelas Vector

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Langkah 2: Kelas Matrix dengan operasi inti

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Langkah 3: Lihat cara kerjanya

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Langkah 4: Hubungkan ke neural network

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

Ini adalah satu dense layer: `output = relu(W @ x + b)`. Setiap dense layer di setiap neural network melakukan tepat hal ini.

## Gunakan

NumPy melakukan semua hal di atas dalam baris yang lebih sedikit dan jauh lebih cepat.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Operator `@` di Python memanggil `__matmul__`. NumPy mengimplementasikannya dengan rutin BLAS teroptimasi yang ditulis dalam C dan Fortran. Matematika yang sama, 100x lebih cepat.

Broadcasting di NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy otomatis me-broadcast bias 1D ke kedua baris. Beginilah cara kerja penjumlahan bias di setiap framework neural network.

## Rilis

Pelajaran ini menghasilkan sebuah prompt untuk mengajarkan operasi matriks lewat intuisi geometris. Lihat `outputs/prompt-matrix-operations.md`.

Kelas Matrix yang dibangun di sini adalah fondasi framework neural network mini yang kita bangun di Fase 3, Pelajaran 10.

## Latihan

1. **Verifikasi invers.** Kalikan `A @ A.inverse_2x2()` dan pastikan kamu mendapat matriks identitas. Coba dengan tiga matriks 2x2 berbeda. Apa yang terjadi ketika determinannya nol?

2. **Implementasikan invers 3x3.** Perluas kelas Matrix untuk menghitung invers matriks 3x3 memakai metode adjugat. Uji terhadap `np.linalg.inv` milik NumPy.

3. **Bangun network dua layer.** Hanya memakai kelas Matrix-mu (tanpa NumPy), buat neural network dua layer: input (3) -> hidden (4) -> output (2). Inisialisasi bobot acak, jalankan forward pass, dan verifikasi semua shape sudah benar.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Vector | "An arrow" | An ordered list of numbers. In AI: a point in high-dimensional space. |
| Matrix | "A table of numbers" | A linear transformation. It maps vectors from one space to another. |
| Matrix multiply | "Just multiply the numbers" | Dot products between every row of the first matrix and every column of the second. Order matters. |
| Transpose | "Flip it" | Swap rows and columns. Turns an m x n matrix into n x m. Critical in backpropagation. |
| Determinant | "Some number from the matrix" | Measures how much the matrix scales area (2D) or volume (3D). Zero means the transformation crushes a dimension. |
| Inverse | "Undo the matrix" | The matrix that reverses the transformation. Only exists when the determinant is not zero. |
| Identity matrix | "The boring matrix" | The matrix equivalent of multiplying by 1. Used in residual connections (ResNets). |
| Broadcasting | "Magic shape fixing" | Stretching a smaller array to match a larger one by repeating along missing dimensions. |
| Element-wise | "Regular multiplication" | Multiply matching positions. Both arrays must have the same shape (or be broadcastable). |

## Bacaan Lanjutan

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - intuisi visual untuk setiap operasi yang dibahas di sini
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - aturan persis yang diikuti NumPy
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - referensi ringkas aljabar linear khusus ML
