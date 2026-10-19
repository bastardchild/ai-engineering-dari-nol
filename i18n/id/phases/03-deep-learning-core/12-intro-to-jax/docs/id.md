# Pengantar JAX

> PyTorch memutasi tensor. TensorFlow membangun graf. JAX mengompilasi fungsi murni. Yang terakhir mengubah cara kamu berpikir tentang deep learning.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## Tujuan Pembelajaran

- Menulis kode neural network fungsi-murni memakai API fungsional JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Menjelaskan perbedaan desain kunci antara mutasi eager PyTorch dan model kompilasi fungsional JAX
- Menerapkan kompilasi jit dan vektoraliasi vmap untuk mempercepat loop training dibanding Python naif
- Melatih network sederhana di JAX dan mengontraskan manajemen state eksplisit dengan pendekatan berorientasi objek PyTorch

## Masalahnya

Kamu tahu cara membangun neural network di PyTorch. Kamu mendefinisikan `nn.Module`, memanggil `.backward()`, men-step optimizer. Ia bekerja. Jutaan orang memakainya.

Tetapi PyTorch punya batas yang tertanam dalam DNA-nya: ia menelusuri operasi secara eager, satu per satu, di Python. Setiap `tensor + tensor` adalah peluncuran kernel terpisah. Setiap langkah training menginterpretasi ulang kode Python yang sama. Ini baik-baik saja sampai kamu perlu melatih model 540-miliar-parameter di 2.048 TPU. Lalu overhead membunuhmu.

Google DeepMind melatih Gemini di JAX. Anthropic melatih Claude di JAX. Ini bukan operasi kecil -- ini run training neural network terbesar di Bumi. Mereka memilih JAX karena ia memperlakukan loop training-mu sebagai program yang bisa dikompilasi, bukan sekuens panggilan Python.

JAX adalah NumPy dengan tiga kekuatan super: diferensiasi otomatis, kompilasi JIT ke XLA, dan vektoraliasi otomatis. Kamu menulis fungsi yang memproses satu contoh. JAX memberimu fungsi yang memproses batch, menghitung gradien, mengompilasi ke kode mesin, dan berjalan di banyak device. Semua tanpa mengubah fungsi aslinya.

## Konsepnya

### Filosofi JAX

JAX adalah framework fungsional. Tanpa kelas, tanpa mutable state, tanpa metode `.backward()`. Sebaliknya:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

Ini bukan preferensi gaya. Ini batas kompiler. Kompilasi JIT membutuhkan fungsi murni -- input sama selalu menghasilkan output sama, tanpa efek samping. Restriksi itulah yang memungkinkan speedup 100x.

### jax.numpy: Permukaan yang Familier

JAX mengimplementasikan ulang API NumPy pada akselerator:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Nama fungsi sama. Aturan broadcasting sama. Semantik slicing sama. Tetapi array hidup di GPU/TPU, dan setiap operasi bisa ditelusuri kompiler.

Satu perbedaan krusial: array JAX immutable. Tidak ada `a[0] = 5`. Sebaliknya: `a = a.at[0].set(5)`. Ini terasa canggung seminggu, lalu klik -- immutability-lah yang membuat transformasi seperti `grad`, `jit`, dan `vmap` bisa dikomposisikan.

### jax.grad: Autodiff Fungsional

PyTorch menempelkan gradien pada tensor (`.grad`). JAX menempelkan gradien pada fungsi.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` menerima fungsi dan mengembalikan fungsi baru yang menghitung gradien. Tanpa panggilan `.backward()`. Tanpa computation graph yang disimpan pada tensor. Gradien hanyalah fungsi lain yang bisa kamu panggil, komposisikan, atau kompilasi-JIT.

Ini terkomposisi sewenang-wenang:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Turunan kedua. Turunan ketiga. Jacobian. Hessian. Semua dengan mengomposisikan `grad`. PyTorch juga bisa (`torch.autograd.functional.hessian`), tetapi ditempel. Di JAX, ia adalah fondasi.

Batasnya: `grad` hanya bekerja pada fungsi murni. Tidak ada print statement di dalam (mereka berjalan selama tracing, bukan eksekusi). Tidak ada mutasi state eksternal. Tidak ada pembangkitan angka acak tanpa manajemen key eksplisit.

### jit: Kompilasi ke XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

Pada panggilan pertama, JAX menelusuri fungsi -- ia merekam operasi apa yang terjadi, tanpa mengeksekusinya. Lalu ia menyerahkan trace itu ke XLA (Accelerated Linear Algebra), kompiler Google untuk TPU dan GPU. XLA memfusi operasi, mengeliminasi salinan memori redundan, dan menghasilkan kode mesin yang teroptimasi.

Panggilan berikutnya melewatkan Python sepenuhnya. Kode terkompilasi berjalan di akselerator secepat C++.

Kapan JIT membantu:

- Langkah training (komputasi sama diulang ribuan kali)
- Inference (model sama, input berbeda)
- Fungsi apa pun yang dipanggil lebih dari sekali dengan input berbentuk mirip

Kapan JIT merugikan:

- Fungsi dengan alur kontrol Python yang bergantung pada nilai (`if x > 0` di mana x adalah array yang ditelusuri)
- Komputasi sekali jalan (overhead kompilasi melebihi runtime)
- Debugging (tracing menyembunyikan eksekusi sebenarnya)

Restriksi alur kontrol itu nyata. `jax.lax.cond` menggantikan `if/else`. `jax.lax.scan` menggantikan loop `for`. Ini bukan opsional -- ini harga kompilasi.

### vmap: Vektoraliasi Otomatis

Kamu menulis fungsi yang memproses satu contoh:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` mengangkatnya untuk memproses batch:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` berarti: jangan batch atas `params` (bersama), batch atas sumbu 0 dari `x`. Tanpa loop `for` manual. Tanpa reshaping. Tanpa threading dimensi batch. JAX mencari tahu dimensi batch dan memvektorkan seluruh komputasi.

Ini bukan gula sintaksis. `vmap` menghasilkan kode tervektor yang terfusi yang berjalan 10-100x lebih cepat daripada loop Python. Dan ia terkomposisi dengan `jit` dan `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Gradien per contoh. Satu baris. Ini nyaris mustahil di PyTorch tanpa hack.

### pmap: Paralelisme Data Antar Device

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` mereplikasi fungsi ke semua device yang tersedia (GPU/TPU) dan membagi batch. Di dalam fungsi, `jax.lax.pmean` dan `jax.lax.psum` menyinkronkan gradien antar device.

Google melatih Gemini di ribuan chip TPU v5e memakai `pmap` (dan penerusnya `shard_map`). Model pemrogramannya: tulis versi satu-device, bungkus dengan `pmap`, selesai.

### Pytree: Struktur Data Universal

JAX beroperasi pada "pytree" -- kombinasi bersarang dari list, tuple, dict, dan array. Parameter modelmu adalah pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Setiap transformasi JAX -- `grad`, `jit`, `vmap` -- tahu cara menelusuri pytree. `jax.tree.map(f, tree)` menerapkan `f` ke setiap leaf. Beginilah optimizer mengupdate semua parameter sekaligus:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Tanpa metode `.parameters()`. Tanpa registrasi parameter. Struktur tree adalah modelnya.

### Fungsional vs Berorientasi Objek

PyTorch menyimpan state di dalam objek:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX memakai fungsi murni dengan state eksplisit:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Param diberikan masuk. Tidak ada yang disimpan. Tidak ada yang dimutasi. Ini membuat setiap fungsi bisa diuji, dikomposisikan, dan dikompilasi. Artinya juga kamu mengelola param sendiri -- atau memakai pustaka seperti Flax atau Equinox.

### Ekosistem JAX

JAX memberimu primitif. Pustaka memberimu ergonomi:

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax adalah pustaka optimizer standar. Ia memisahkan transformasi gradien (Adam, SGD, clipping) dari update parameter, sehingga mudah dikomposisikan:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Kapan Memakai JAX vs PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class (Google built both) | Community-maintained (torch_xla) |
| GPU support | Good (CUDA via XLA) | Best-in-class (native CUDA) |
| Debugging | Hard (tracing + compilation) | Easy (eager, line-by-line) |
| Ecosystem | Research-focused (Flax, Equinox) | Massive (HuggingFace, torchvision, etc.) |
| Hiring | Niche (Google/DeepMind/Anthropic) | Mainstream (everywhere) |
| Large-scale training | Superior (XLA, pmap, mesh) | Good (FSDP, DeepSpeed) |
| Prototyping speed | Slower (functional overhead) | Faster (mutate and go) |
| Production inference | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Who uses it | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

Jawaban jujur: pakai PyTorch kecuali kamu punya alasan spesifik memakai JAX. Alasan itu adalah -- akses TPU, butuh gradien per contoh, training multi-device pada skala raksasa, atau bekerja di Google/DeepMind/Anthropic.

### Angka Acak di JAX

JAX tidak punya state acak global. Setiap operasi acak membutuhkan key PRNG eksplisit:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Ini menyebalkan pada awalnya. Tetapi ia menjamin reproduksibilitas antar device dan kompilasi -- properti yang tidak bisa dijamin `torch.manual_seed` PyTorch dalam setting multi-GPU.

```figure
batchnorm-effect
```

## Bangun Sendiri

### Langkah 1: Setup dan Data

Kita akan melatih MLP 3 lapis pada MNIST memakai JAX dan Optax. 784 input, dua hidden layer berisi 256 dan 128 neuron, 10 kelas output.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Langkah 2: Inisialisasi Parameter

Tanpa kelas. Hanya fungsi yang mengembalikan pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

Inisialisasi He, dikerjakan manual. Tiga key PRNG dipecah dari satu seed. Setiap bobot adalah array immutable dalam dict bersarang.

### Langkah 3: Forward Pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Fungsi murni. Param masuk, prediksi keluar. Tanpa `self`, tanpa state tersimpan. `loss_fn` menghitung cross-entropy dari nol -- softmax, log, rerata negatif.

### Langkah 4: Langkah Training yang Dikompilasi-JIT

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` mengembalikan nilai loss sekaligus gradien dalam satu pass. Dekorator `@jax.jit` mengompilasi kedua fungsi ke XLA. Setelah panggilan pertama, setiap langkah training berjalan tanpa menyentuh Python.

### Langkah 5: Loop Training

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 epoch. ~97% akurasi uji. Epoch pertama lambat (kompilasi JIT). Epoch 2-10 cepat.

Perhatikan apa yang hilang: tanpa `.zero_grad()`, tanpa `.backward()`, tanpa `.step()`. Seluruh update adalah satu panggilan fungsi terkomposisi. Gradien dihitung, ditransformasi Adam, dan diterapkan ke parameter -- semua di dalam `train_step`.

## Gunakan

### Flax: Standar Google

Flax adalah pustaka neural network JAX paling umum. Ia menambahkan `nn.Module` kembali, tetapi dengan manajemen state eksplisit:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

Struktur sama dengan PyTorch, tetapi `params` terpisah dari model. `model.init()` menciptakan param. `model.apply(params, x)` menjalankan forward pass. Objek model tidak punya state.

### Equinox: Alternatif yang Pythonic

Equinox (oleh Patrick Kidger) merepresentasikan model sebagai pytree:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Model itu sendiri adalah pytree. Tanpa `.apply()`. Parameter hanyalah leaf model. Ini lebih dekat dengan cara JAX berpikir.

### Optax: Optimizer yang Composable

Optax memisahkan transformasi gradien dari update:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping, warmup learning rate, weight decay -- semua dikomposisikan sebagai rantai transform. Setiap transform melihat gradien, memodifikasinya, dan meneruskannya ke berikutnya. Tanpa kelas optimizer monolitik.

## Rilis

**Instalasi:**

```bash
pip install jax jaxlib optax flax
```

Untuk dukungan GPU:

```bash
pip install jax[cuda12]
```

Untuk TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Jebakan performa:**

- Panggilan JIT pertama lambat (kompilasi). Panaskan sebelum benchmarking.
- Hindari loop Python atas array JAX di dalam JIT. Pakai `jax.lax.scan` atau `jax.lax.fori_loop`.
- `jax.debug.print()` bekerja di dalam JIT. `print()` biasa tidak.
- Buat profil dengan `jax.profiler` atau TensorBoard. Kompilasi XLA bisa menyembunyikan bottleneck.
- JAX mengalokasikan 75% memori GPU secara default. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` untuk menonaktifkan.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**Pelajaran ini menghasilkan:**

- `outputs/prompt-jax-optimizer.md` -- prompt untuk memilih konfigurasi optimizer JAX yang tepat
- `outputs/skill-jax-patterns.md` -- skill yang membahas pola fungsional dalam JAX

## Latihan

1. Tambahkan dropout ke MLP. Di JAX, dropout membutuhkan key PRNG -- teruskan key melalui forward pass dan pecah untuk setiap lapis dropout. Bandingkan akurasi uji dengan dan tanpa.
2. Pakai `jax.vmap` untuk menghitung gradien per contoh untuk batch berisi 32 gambar MNIST. Hitung norma gradien setiap contoh. Contoh mana yang gradiennya terbesar, dan mengapa?
3. Ganti fungsi forward manual dengan `mlp_forward(params, x)` generik yang bekerja untuk berapa pun lapis. Pakai `jax.tree.leaves` untuk menentukan kedalaman otomatis.
4. Benchmark langkah training dengan dan tanpa `@jax.jit`. Ukur waktu 100 langkah masing-masing. Seberapa besar speedup pada hardware-mu? Berapa overhead kompilasi pada panggilan pertama?
5. Implementasikan gradient clipping dengan mengomposisikan `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Latih dengan dan tanpa clipping. Plot norma gradien selama training untuk melihat efeknya.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| XLA | "The thing that makes JAX fast" | Accelerated Linear Algebra -- a compiler that fuses operations and generates optimized GPU/TPU kernels from a computation graph |
| JIT | "Just-in-time compilation" | JAX traces the function on first call, compiles to XLA, then runs the compiled version on subsequent calls |
| Pure function | "No side effects" | A function where the output depends only on inputs -- no global state, no mutation, no randomness without explicit keys |
| vmap | "Auto-batching" | Transforms a function that processes one example into one that processes a batch, without rewriting |
| pmap | "Auto-parallelism" | Replicates a function across multiple devices and splits the input batch |
| Pytree | "Nested dict of arrays" | Any nested structure of lists, tuples, dicts, and arrays that JAX can traverse and transform |
| Tracing | "Recording the computation" | JAX executes the function with abstract values to build a computation graph, without computing real results |
| Functional autodiff | "grad of a function" | Computing derivatives by transforming functions, not by attaching gradient storage to tensors |
| Optax | "JAX's optimizer library" | A composable library of gradient transformations -- Adam, SGD, clipping, scheduling -- that chain together |
| Flax | "JAX's nn.Module" | Google's neural network library for JAX, adding layer abstractions while keeping state explicit |

## Bacaan Lanjutan

- Dokumentasi JAX: https://jax.readthedocs.io/ -- dok resmi, dengan tutorial luar biasa tentang grad, jit, dan vmap
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- makalah orisinal yang menjelaskan filosofi desain
- Dokumentasi Flax: https://flax.readthedocs.io/ -- pustaka neural network Google untuk JAX
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- alternatif Flax yang Pythonic
- DeepMind, "Optax: composable gradient transformation and optimisation" -- pustaka optimizer standar
- "You Don't Know JAX" (Colin Raffel, 2020) -- panduan praktis jebakan dan pola JAX, dari salah satu penulis T5
