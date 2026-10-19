# Penyempurnaan dengan LoRA & QLoRA

> Penyempurnaan penuh model 7B memerlukan 56GBVRAM. kamu tidak memilikinya. Begitu pula sebagian besar perusahaan. LoRA memungkinkan kamu menyempurnakan model yang sama dalam 6 GB dengan melatih kurang dari 1% parameter. Ini bukan kompromi -- ini cocok dengan kualitas penyesuaian penuh pada sebagian besar tugas. Seluruh ekosistem penyempurnaan sumber terbuka dijalankan dengan trik yang satu ini.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 06 (Instruction Tuning / SFT)
**Time:** ~75 minutes
**Terkait:** Fase 10 mencakupSFT/DPOloop dari awal. Pelajaran ini menghubungkannya dengan tahun 2026PEFTperangkat (PEFT,TRL, Unsloth, Axolotl, Pabrik LLaMA).

## Tujuan Pembelajaran

- Menerapkan LoRA dengan memasukkan matriks adaptor tingkat rendah (A dan B) ke dalam lapisan perhatian model yang telah dilatih sebelumnya
- Hitung penghematan parameter LoRA vs penyesuaian penuh: peringkat r dengan dimensi d_model melatih parameter 2*r*d alih-alih d^2
- Menyempurnakan model menggunakan QLoRA (basis terkuantisasi 4-bit + adaptor LoRA) agar sesuai dengan konsumenGPUmemori
- Gabungkan kembali bobot LoRA ke model dasar untuk penerapan dan bandingkan kecepatan inferensi dengan dan tanpa adaptor

## Masalahnya

kamu memiliki model dasar. Lama 3 8B. kamu menginginkannya menjawab tiket dukungan pelanggan dengan suara perusahaan kamu.SFTadalah jawabannya. TapiSFTmempunyai masalah biaya.

Penyempurnaan penuh memperbarui setiap parameter dalam model. Llama 3 8B memiliki 8 miliar parameter. Di fp16, setiap parameter membutuhkan 2 byte. Itu 16GB hanya untuk memuat beban. Selama pelatihan, kamu juga memerlukan gradien (16 GB), status pengoptimal untuk Adam (32 GB untuk momentum + varians), dan aktivasi. Total: kira-kira 56GBVRAMuntuk model 8B tunggal.

SebuahA10080GB hampir tidak bisa memuat ini. Harga dua A100$3-4/hour on cloud providers. Training for 3 epochs on 50,000 examples takes 6-10 hours. That's $30-40 per percobaan. Jalankan 10 eksperimen untuk mendapatkan hyperparameter yang benar dan kamu telah menghabiskan $400 sebelum menerapkan apa pun.

Skalakan ini ke Llama 3 70B dan angkanya menjadi tidak masuk akal. 140GB untuk beban saja. kamu membutuhkan sebuah cluster. $100+ per eksperimen.

Ada masalah yang lebih dalam juga. Penyempurnaan penuh mengubah setiap bobot dalam model. Jika kamu menyempurnakan data dukungan pelanggan, kamu mungkin menurunkan kemampuan umum model. Ini disebut bencana lupa. Model menjadi lebih baik dalam tugas kamu dan lebih buruk dalam segala hal lainnya.

kamu memerlukan metode yang melatih lebih sedikit parameter, menggunakan lebih sedikit memori, dan tidak merusak pengetahuan model yang sudah ada.

## Konsepnya

### LoRA: Adaptasi Tingkat Rendah

Edward Hu dan rekannya di Microsoft menerbitkan LoRA pada bulan Juni 2021. Wawasan makalah ini: pembaruan bobot selama penyesuaian memiliki peringkat intrinsik yang rendah. kamu tidak perlu memperbarui 16,7 juta parameter dalam matriks bobot 4096x4096. Informasi berguna dalam pembaruan dapat ditangkap oleh matriks peringkat 16 atau 32.

Berikut perhitungannya. Lapisan linier standar menghitung:

```
y = Wx
```

Dimana W adalah matriks d_out x d_in. Untuk proyeksi perhatian 4096x4096, itu berarti 16.777.216 parameter.

LoRA membekukan W dan menambahkan dekomposisi tingkat rendah:

```
y = Wx + BAx
```

Dimana B adalah (d_out x r) dan A adalah (r x d_in). Peringkat r jauh lebih kecil dari d -- biasanya 8, 16, atau 32.

Untuk r=16 pada lapisan 4096x4096:
- Parameter asli: 4096 x 4096 = 16.777.216
- Parameter LoRA: (4096 x 16) + (16 x 4096) = 65.536 + 65.536 = 131.072
- Pengurangan : 131.072 / 16.777.216 = 0,78%

kamu melatih 0,78% parameter dan mendapatkan kualitas 95-100%.

```mermaid
graph LR
    X["Input x"] --> W["Frozen W (d x d)"]
    X --> A["A (r x d)"]
    A --> B["B (d x r)"]
    W --> Plus["+ (merge)"]
    B --> Plus
    Plus --> Y["Output y"]

    style W fill:#1a1a2e,stroke:#e94560,color:#fff
    style A fill:#0f3460,stroke:#16213e,color:#fff
    style B fill:#0f3460,stroke:#16213e,color:#fff
```

A diinisialisasi dengan Gaussian acak. B diinisialisasi ke nol. Artinya kontribusi LoRA dimulai dari nol -- model memulai pelatihan dari perilaku aslinya dan secara bertahap mempelajari adaptasinya.

### Faktor Penskalaan: Alfa

LoRA memperkenalkan faktor penskalaan alfa yang mengontrol seberapa besar pengaruh pembaruan peringkat rendah terhadap keluaran:

```
y = Wx + (alpha / r) * BAx
```

Ketika alpha = r, skalanya adalah 1x. Jika alpha = 2r (default umum), penskalaannya adalah 2x. Hyperparameter ini mengontrol kecepatan pembelajaran jalur LoRA secara independen dari kecepatan pembelajaran dasar.

Panduan praktis:
- alpha = 2 * rank adalah konvensi umum komunitas (makalah asli menggunakan alpha = rank di sebagian besar eksperimen)
- alpha = rank memberikan penskalaan 1x, konservatif tetapi stabil
- Alfa yang lebih tinggi berarti pembaruan yang lebih besar per langkah, yang dapat mempercepat konvergensi atau menyebabkan ketidakstabilan

### Tempat Mendaftar LoRA

Sebuah transformator memiliki banyak lapisan linier. kamu tidak perlu menambahkan LoRA ke semuanya. Makalah asli menguji kombinasi yang berbeda:

| Target Layers | Trainable Params (7B) | Quality |
|--------------|----------------------|---------|
| q_proj only | 4.7M | Good |
| q_proj + v_proj | 9.4M | Better |
| q_proj + k_proj + v_proj + o_proj | 18.9M | Best for attention |
| All linear (attention + MLP) | 37.7M | Marginal gain, 2x params |

Titik terbaik untuk sebagian besar tugas: q_proj + v_proj. Ini menargetkan proyeksi kueri dan nilai dalam perhatian diri, yang mengontrol apa yang diperhatikan model dan informasi apa yang diekstraksi. MenambahkanMLPlapisan membantu tugas-tugas kompleks seperti pembuatan kode tetapi menggandakan jumlah parameter untuk mengurangi hasil pada tugas-tugas sederhana.

### Seleksi Peringkat

Peringkat r mengontrol ekspresi adaptasi:

| Rank | Trainable Params (per layer) | Best For |
|------|---------------------------|----------|
| 4 | 32,768 | Simple classification, sentiment |
| 8 | 65,536 | Single-domain Q&A, summarization |
| 16 | 131,072 | Multi-domain tasks, instruction following |
| 32 | 262,144 | Complex reasoning, code generation |
| 64 | 524,288 | Diminishing returns for most tasks |
| 128 | 1,048,576 | Rarely justified |

Hu dkk. menunjukkan bahwa r=4 sudah mencakup sebagian besar adaptasi untuk tugas-tugas sederhana. r=8 dan r=16 adalah pilihan paling umum dalam praktiknya. Melampaui r=64 jarang meningkatkan kualitas dan mulai kehilangan keunggulan memori LoRA.

### QLoRA: Kuantisasi 4-Bit + LoRA

Tim Dettmers dan rekannya di Universitas Washington menerbitkan QLoRA pada Mei 2023. Idenya: mengkuantisasi model dasar beku ke presisi 4-bit, lalu memasang adaptor LoRA di fp16 di atasnya.

Ini mengubah persamaan memori secara dramatis:

| Method | Weight Memory (7B) | Training Memory (7B) | GPU Required |
|--------|-------------------|---------------------|-------------|
| Full fine-tune (fp16) | 14GB | ~56GB | 1x A100 80GB |
| LoRA (fp16 base) | 14GB | ~18GB | 1x A100 40GB |
| QLoRA (4-bit base) | 3.5GB | ~6GB | 1x RTX 3090 24GB |

QLoRA memberikan tiga kontribusi teknis:

**NF4(Normal Float 4-bit)**: Tipe data baru yang dirancang khusus untuk bobot jaringan saraf. Bobot jaringan saraf mengikuti distribusi yang kira-kira normal.NF4menempatkan 16 tingkat kuantisasinya pada kuantil distribusi normal standar. Ini secara teori merupakan informasi yang optimal untuk data yang terdistribusi normal. Ia kehilangan lebih sedikit informasi dibandingkan kuantisasi 4-bit yang seragam (INT4) atau Float4 standar.

**Kuantisasi ganda**: Konstanta kuantisasi itu sendiri memerlukan memori. Setiap blok dengan 64 bobot memerlukan faktor skala fp32 (4 byte). Untuk model 7B, itu tambahan 0,4 GB. Kuantisasi ganda mengkuantisasi konstanta ini menjadi fp8, mengurangi overhead menjadi 0,1 GB. Kecil tapi bertambah.

**Pengoptimal halaman**: Selama pelatihan, status pengoptimal (momentum dan varians Adam) dapat melebihiGPUmemori dalam urutan yang panjang. Penggunaan pengoptimal halamanNVIDIAmemori terpadu untuk status pengoptimal halaman secara otomatisCPU RAMkapanGPUmemori habis, dan halamankan kembali bila diperlukan. Hal ini mencegahOOMcrash dengan mengorbankan beberapa throughput.

### Pertanyaan Kualitas

Apakah mengurangi parameter atau mengkuantisasi basis akan merusak kualitas? Hasil dari beberapa makalah:

| Method | MMLU (5-shot) | MT-Bench | HumanEval |
|--------|--------------|----------|-----------|
| Full fine-tune (Llama 2 7B) | 48.3 | 6.72 | 14.6 |
| LoRA r=16 | 47.9 | 6.68 | 14.0 |
| QLoRA r=16 (NF4) | 47.5 | 6.61 | 13.4 |
| QLoRA r=64 (NF4) | 48.1 | 6.70 | 14.2 |

LoRA pada r=16 berada dalam 1% dari penyesuaian penuh pada sebagian besar tolok ukur. QLoRA pada r=16 kehilangan sepersekian persen lagi. QLoRA pada r=64 pada dasarnya cocok dengan penyempurnaan penuh sambil menggunakan memori 90% lebih sedikit.

### Biaya Dunia Nyata

Menyempurnakan Llama 3 8B pada 50.000 contoh (3 epoch):

| Method | GPU | Time | Cost |
|--------|-----|------|------|
| Full fine-tune | 2x A100 80GB | 8 hours | ~$32 |
| LoRA r=16 | 1x A100 40GB | 4 hours | ~$8 |
| QLoRA r=16 | 1x RTX 4090 24GB | 6 hours | ~$5 |
| QLoRA r=16 (Unsloth) | 1x RTX 4090 24GB | 2.5 hours | ~$2 |
| QLoRA r=16 | 1x T4 16GB | 12 hours | ~$4 |

QLoRA pada satu konsumenGPUbiayanya lebih murah daripada makan siang. Inilah sebabnya mengapa komunitas fine-tuning open-weight meledak pada tahun 2023 dan mengapa setiap kerangka pelatihan di bawah ini mengirimkan QLoRA secara default pada tahun 2026.

### Tahun 2026PEFTtumpukan

| Framework | What it is | Pick when |
|-----------|-----------|-----------|
| **Hugging Face PEFT** | The canonical LoRA/QLoRA/DoRA/IA3 library | You want raw control and your training loop is already on `transformers.Trainer` |
| **TRL** | HF's reinforcement-from-feedback trainers (SFT, DPO, GRPO, PPO, ORPO) | You need DPO/GRPO after SFT; built on top of PEFT |
| **Unsloth** | Triton-kernel rewrite of the forward/backward pass | You want 2-5x speedup + half the VRAM with no accuracy loss; Llama/Mistral/Qwen family |
| **Axolotl** | YAML-config wrapper over PEFT + TRL + DeepSpeed + Unsloth | You want reproducible, version-controlled training runs |
| **LLaMA-Factory** | GUI/CLI/API over PEFT + TRL | You want zero-code fine-tuning; 100+ model families supported |
| **torchtune** | Native PyTorch recipes, no `transformers` dep | You want minimal deps and your org already standardizes on PyTorch |

Aturan praktisnya: penggunaan penelitian atau eksperimen satu kali →PEFT. Jalur produksi berulang → Axolotl dengan kernel Unsloth diaktifkan. Pembuatan prototipe sekali pakai → Pabrik LLaMA.

### Menggabungkan Adaptor

Setelah pelatihan, kamu memiliki dua hal: model dasar yang dibekukan dan adaptor LoRA kecil (biasanya 10-100MB). kamu dapat:

1. **Pisahkan keduanya**: Muat model dasar, muat adaptor di atas. Tukar adaptor untuk tugas yang berbeda. Inilah cara kamu menyajikan beberapa varian yang disesuaikan dari satu model dasar.

2. **Gabungkan keduanya secara permanen**: Hitung W' = W + (alpha/r) * BA dan simpan hasilnya sebagai model lengkap baru. Model yang digabungkan ukurannya sama dengan aslinya. Tidak ada overhead inferensi. Tidak ada adaptor untuk dikelola.

Untuk melayani banyak tugas (adaptor dukungan pelanggan, adaptor kode, adaptor terjemahan), pisahkan semuanya. Untuk menerapkan satu model khusus, gabungkan.

Teknik penggabungan tingkat lanjut untuk menggabungkan beberapa adaptor:

- **TIES-Penggabungan** (Yadav dkk. 2023): Memangkas parameter berkekuatan kecil, menyelesaikan konflik tanda, lalu menggabungkan. Mengurangi interferensi antar adaptor.
- **DARE** (Yu dkk. 2023): Menghapus parameter adaptor secara acak sebelum menggabungkan dan mengubah skala sisanya. Sangat efektif dalam menggabungkan kemampuan.
- **Aritmatika tugas**: Cukup tambahkan atau kurangi bobot adaptor. Menambahkan adaptor "kode" dan adaptor "matematika" sering kali menghasilkan model yang bagus dalam keduanya.

### KapanNOTuntuk Menyempurnakan

Penyempurnaan adalah pilihan ketiga, bukan yang pertama.

**Pertama: rekayasa cepat.** Tulis perintah sistem yang lebih baik. Tambahkan beberapa contoh contoh. Gunakan rantai pemikiran. Ini tidak memerlukan biaya apa pun dan hanya membutuhkan waktu beberapa menit. Jika perintah tersebut membuat kamu mencapai 80%, kamu mungkin tidak perlu melakukan penyesuaian.

**Kedua:RAG.** Jika model perlu mengetahui data spesifik kamu (dokumen, basis pengetahuan, katalog produk), pengambilan lebih murah dan lebih mudah dikelola daripada memasukkannya ke dalam bobot. Lihat Pelajaran 06.

**Ketiga: penyempurnaan.** Gunakan ini saat kamu memerlukan model untuk mengadopsi gaya, format, atau pola penalaran tertentu yang tidak dapat dicapai melalui dorongan. Saat kamu membutuhkan keluaran terstruktur yang konsisten. Saat kamu perlu menyaring model yang lebih besar menjadi model yang lebih kecil. Saat latensi penting dan kamu tidak mampu membeli token tambahan dari beberapa kali perintah.

```mermaid
graph TD
    Start["Need better model behavior?"] --> PE["Try prompt engineering"]
    PE -->|"Works"| Done["Ship it"]
    PE -->|"Not enough"| RAG["Need external knowledge?"]
    RAG -->|"Yes"| RAGBuild["Build RAG pipeline"]
    RAG -->|"No, need style/format change"| FT["Fine-tune with LoRA/QLoRA"]
    RAGBuild -->|"Works"| Done
    RAGBuild -->|"Also need style change"| FT
    FT --> Done

    style Start fill:#1a1a2e,stroke:#e94560,color:#fff
    style Done fill:#0f3460,stroke:#16213e,color:#fff
```

```figure
lora-params
```

## Bangun Sendiri

Kami menerapkan LoRA dari awal di PyTorch murni. Tidak ada perpustakaan. Tidak ada keajaiban. kamu akan membuat lapisan LoRA, memasukkannya ke dalam model, melatihnya, dan menggabungkan kembali bobotnya.

### Langkah 1: Lapisan LoRA

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
```

A diinisialisasi dengan nilai acak berskala. B diinisialisasi ke nol. Produk BA dimulai dari nol, sehingga model dimulai dengan perilaku aslinya.

### Langkah 2: Lapisan Linier Terbungkus LoRA

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

Lapisan linier asli dibekukan. Hanya parameter LoRA (A dan B) yang dapat dilatih.

### Langkah 3: Menyuntikkan LoRA ke dalam Model

```python
def inject_lora(model, target_modules, rank=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers
```

Pertama, bekukan setiap parameter dalam model. Kemudian telusuri pohon model, temukan lapisan linier yang cocok dengan nama target kamu, dan gantikan dengan versi yang dibungkus LoRA. Matriks LoRA A dan B adalah satu-satunya parameter yang dapat dilatih di keseluruhan model.

### Langkah 4: Hitung Parameter

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }
```

### Langkah 5: Gabungkan Bobot Kembali

```python
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)
```

Setelah penggabungan, lapisan LoRA hilang. Modelnya berukuran sama dengan aslinya dengan adaptasi yang dimasukkan ke dalam bobotnya. Tidak ada overhead inferensi.

### Langkah 6: Simulasi Kuantisasi QLoRA

```python
def quantize_to_nf4(tensor, block_size=64):
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
```

Ini mensimulasikan kuantisasi 4-bit dengan memetakan bobot ke dalam 16 level diskrit dalam blok 64. Produksi QLoRA menggunakan perpustakaan bitsandbytes untuk kebenarannya.NF4padaGPU.

### Langkah 7: Lingkaran Pelatihan

```python
def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses
```

### Langkah 8: Demo Penuh

```python
def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }
```

Demo ini membuat model kecil, memasukkan LoRA ke dalam dua lapisan, melatihnya, dan menggabungkan kembali bobotnya. Jumlah parameter turun dari dapat dilatih penuh menjadi ~1% dapat dilatih selama pelatihan LoRA, kemudian kembali ke arsitektur asli setelah penggabungan.

## Gunakan

Dengan ekosistem Hugging Face, LoRA pada model nyata membutuhkan sekitar 20 baris:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Untuk QLoRA, tambahkan kuantisasi bitsandbytes:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = get_peft_model(model, lora_config)
```

Itu saja. Lingkaran pelatihan yang sama. Saluran data yang sama. Model dasar sekarang ada dalam 4-bit, adaptor LoRA dilatih dalam fp16, dan semuanya muat dalam 6GB.

Untuk pelatihan dengan Hugging Face Trainer:

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-adapter")
```

Adaptor yang disimpan berukuran 10-100MB. Model dasar tetap tidak tersentuh. kamu dapat berbagi adaptor di Hugging Face Hub tanpa mendistribusikan ulang model lengkapnya.

## Rilis

Pelajaran ini menghasilkan:
- `outputs/prompt-lora-advisor.md`-- perintah yang membantu kamu menentukan peringkat LoRA, modul target, dan hyperparameter untuk tugas spesifik kamu
- `outputs/skill-fine-tuning-guide.md`-- keterampilan yang mengajarkan agen pohon keputusan tentang kapan dan bagaimana melakukan penyesuaian

## Latihan

1. **Studi ablasi peringkat.** Jalankan demo dengan peringkat 2, 4, 8, 16, 32, dan 64. Plot kekalahan akhir vs. peringkat. Temukan titik pengembalian yang semakin berkurang di mana penggandaan peringkat tidak lagi mengurangi separuh kerugian. Untuk tugas klasifikasi sederhana pada fitur 256-dim, nilai ini seharusnya berada di sekitar r=8-16.

2. **Perbandingan modul target.** Ubah inject_lora untuk hanya menargetkan lapisan "0", hanya lapisan "2", hanya lapisan "4", dan ketiganya. Latih setiap varian selama 20 epoch. Bandingkan kecepatan konvergensi dan kerugian akhir. Ini mencerminkan keputusan sebenarnya dalam menargetkan q_proj vs v_proj vs semua lapisan linier.

3. **Analisis kesalahan kuantisasi.** Ambil matriks bobot model terlatih sebelum dan sesudah quantize_to_nf4 / dequantize_from_nf4. Hitung rata-rata kesalahan kuadrat, kesalahan absolut maksimum, dan korelasi antara bobot asli dan bobot yang direkonstruksi. Bereksperimenlah dengan nilai block_size 32, 64, 128, dan 256.

4. **Pelayanan multi-adaptor.** Latih dua adaptor LoRA pada subkumpulan data yang berbeda (indeks genap vs indeks ganjil). Simpan kedua adaptor. Muat model dasar satu kali, lalu tukar adaptor dan verifikasi bahwa masing-masing menghasilkan keluaran berbeda pada masukan yang sama. Beginilah cara sistem produksi melayani beberapa model yang disesuaikan dari satu basis.

5. **Inferensi gabungan vs. tidak digabungkan.** Bandingkan output model LoRA sebelum dan sesudah merge_lora_weights pada 100 input yang sama. Verifikasi bahwa keluarannya identik (dalam toleransi floating-point 1e-5). Maka kecepatan inferensi patokan untuk keduanya -- penggabungan seharusnya sedikit lebih cepat karena ini merupakan perkalian matriks tunggal, bukan dua.

## Istilah-Istilah Utama

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| LoRA | "Efficient fine-tuning" | Low-Rank Adaptation: freeze base weights, train two small matrices A and B whose product approximates the full weight update |
| QLoRA | "Fine-tune on a laptop" | Quantized LoRA: load the base model in 4-bit NF4, train LoRA adapters in fp16 on top, enabling 7B fine-tuning in 6GB VRAM |
| Rank (r) | "How much the model can learn" | The inner dimension of the A and B matrices; controls expressiveness vs. parameter count |
| Alpha | "LoRA learning rate" | Scaling factor applied to the LoRA output; alpha/r scales the adaptation's contribution to the final output |
| NF4 | "4-bit quantization" | Normal Float 4: a 4-bit data type with quantization levels at normal distribution quantiles, optimal for neural network weights |
| Adapter | "The small trained part" | The LoRA A and B matrices saved as a separate file (10-100MB), loadable on top of any copy of the base model |
| Target modules | "Which layers to LoRA" | The specific linear layers (q_proj, v_proj, etc.) where LoRA adapters are injected |
| Merging | "Bake it in" | Computing W + (alpha/r) * BA and replacing the original weight, eliminating the adapter overhead at inference |
| Paged optimizers | "Don't OOM during training" | Offloading optimizer states (Adam momentum, variance) to CPU when GPU memory is exhausted |
| Catastrophic forgetting | "Fine-tuning broke everything else" | When updating all weights causes the model to lose previously learned capabilities |

## Bacaan Lebih Lanjut

- Hu et al., "LoRA: Adaptasi Tingkat Rendah dari Model Bahasa Besar" (2021) -- makalah asli yang memperkenalkan metode dekomposisi tingkat rendah, diuji padaGPT-3 175B dengan peringkat serendah 4
- Dettmers dkk., "QLoRA: Penyempurnaan Model Bahasa Terkuantisasi yang Efisien" (2023) -- memperkenalkanNF4, kuantisasi ganda, dan pengoptimal halaman, memungkinkan penyesuaian 65B pada satu memori 48GBGPU
- PEFTdokumentasi perpustakaan (huggingface.co/docs/peft) -- perpustakaan standar untuk LoRA, QLoRA, dan metode efisien parameter lainnya di ekosistem Hugging Face
- Yadav dkk., "TIES-Penggabungan: Menyelesaikan Interferensi Saat Menggabungkan Model" (2023) -- teknik untuk menggabungkan beberapa adaptor LoRA tanpa penurunan kualitas
- [Rafailov dkk., "Optimasi Preferensi Langsung: Model Bahasa kamu Diam-diam adalah Model Hadiah" (NeurIPS 2023)](https://arxiv.org/abs/2305.18290) --DPOderivasi; tahap penyesuaian preferensi yang terjadi setelahnyaSFT, tidak diperlukan model penghargaan.
- [TRLdokumentasi](https://huggingface.co/docs/trl/) -- referensi resmi untuk`SFTTrainer`,`DPOTrainer`,`KTOTrainer`, dan permukaan integrasi denganPEFT/bitsandbytes/Unsloth.
- [Dokumentasi tidak senonoh](https://docs.unsloth.ai/) -- kernel yang menyatu yang menggandakan throughput fine-tuning dan membagi separuh memori; lapisan kinerja di bawahTRL.
- [Dokumentasi Axolotl](https://axolotl-ai-cloud.github.io/axolotl/) --YAML-dikonfigurasi multi-GPU SFT/DPO/QLoRApelatih; alternatif config-as-code untuk skrip tulisan tangan.
