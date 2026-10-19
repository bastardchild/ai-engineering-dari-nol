# CNN dan RNN untuk Teks

> Konvolusi mempelajari n-gram. Rekurensi mengingat. Keduanya digantikan attention. Keduanya masih penting pada hardware terbatas.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 minutes

## Masalahnya

TF-IDF dan Word2Vec menghasilkan vektor datar yang mengabaikan urutan kata. Pengklasifikasi yang dibangun di atasnya tidak bisa membedakan `dog bites man` dari `man bites dog`. Urutan kata kadang membawa sinyal.

Dua keluarga arsitektur mengisi celah itu sebelum transformer tiba.

**Convolutional net untuk teks (TextCNN).** Terapkan konvolusi 1D atas sekuens word embedding. Filter selebar 3 adalah detektor trigram yang terpelajari: ia merentang tiga kata dan mengeluarkan skor. Tumpuk lebar berbeda (2, 3, 4, 5) untuk mendeteksi pola multi-skala. Max-pool menjadi representasi berukuran tetap. Datar, paralel, cepat.

**Recurrent net (RNN, LSTM, GRU).** Proses token satu per satu, mempertahankan hidden state yang membawa informasi ke depan. Sekuensial, membawa memori, panjang input fleksibel. Mendominasi pemodelan sekuens dari 2014 sampai 2017, lalu attention terjadi.

Pelajaran ini membangun keduanya, lalu menyebutkan kegagalan yang memotivasi attention.

## Konsepnya

**TextCNN** (Kim, 2014). Token di-embedding. Konvolusi 1D selebar-`k` menggeser filter atas `k`-gram embedding yang berurutan, menghasilkan feature map. Max-pooling global atas map itu memilih aktivasi terkuat. Gabungkan output max-pool dari beberapa lebar filter. Umpankan ke head pengklasifikasi.

Mengapa berhasil. Filter adalah n-gram yang terpelajari. Max-pooling invarian-posisi, sehingga "not good" memicu fitur sama di awal maupun tengah ulasan. Tiga lebar filter dengan 100 filter masing-masing memberimu 300 detektor n-gram terpelajari. Training paralel; tanpa dependensi sekuensial.

**RNN.** Pada tiap langkah waktu `t`, hidden state `h_t = f(W * x_t + U * h_{t-1} + b)`. `W`, `U`, `b` dibagi lintas waktu. Hidden state pada waktu `T` adalah ringkasan seluruh prefiks. Untuk klasifikasi, pool lintas `h_1 ... h_T` (max, mean, atau last).

RNN polos menderita gradien hilang. **LSTM** menambahkan gate yang memutuskan apa yang dilupakan, apa yang disimpan, dan apa yang dikeluarkan, menstabilkan gradien melalui sekuens panjang. **GRU** menyederhanakan LSTM menjadi dua gate; performanya mirip dengan parameter lebih sedikit.

**RNN bidirectional** menjalankan satu RNN maju dan satu lagi mundur, menggabungkan hidden state. Representasi tiap token melihat konteks kiri dan kanan. Esensial untuk tugas tagging.

```figure
rnn-unroll
```

## Bangun Sendiri

### Langkah 1: TextCNN di PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` membentuk ulang `[batch, seq_len, embed_dim]` menjadi `[batch, embed_dim, seq_len]` karena `nn.Conv1d` memperlakukan sumbu tengah sebagai channel. Output pooled berukuran tetap apa pun panjang inputnya.

### Langkah 2: pengklasifikasi LSTM

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Max-pool atas sekuens, bukan pool last-state. Untuk klasifikasi, max-pooling biasanya mengalahkan pengambilan hidden state terakhir karena informasi di akhir sekuens panjang cenderung mendominasi state terakhir.

### Langkah 3: demo gradien hilang (intuisi)

RNN polos tanpa gating tidak bisa mempelajari dependensi jarak jauh. Perhatikan tugas mainan: prediksi apakah token `A` muncul di mana pun dalam sekuens. Jika `A` di posisi 1 dan sekuensnya 100 token panjangnya, gradien dari loss harus mengalir mundur melalui 99 perkalian bobot rekuren. Jika bobotnya kurang dari 1, gradien hilang. Jika lebih dari 1, ia meledak.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM memperbaikinya dengan **cell state** yang mengalir melalui jaringan dengan interaksi hanya aditif (forget gate menskalanya secara multiplikatif, tetapi gradien tetap mengalir sepanjang "jalan tol"). GRU melakukan hal mirip dengan parameter lebih sedikit. Keduanya memberimu training stabil melalui sekuens 100+ langkah.

### Langkah 4: mengapa ini pun belum cukup

Tiga masalah bertahan bahkan dengan LSTM.

1. **Bottleneck sekuensial.** Melatih RNN pada sekuens panjang 1000 butuh 1000 langkah forward/backward serial. Tidak bisa diparalelkan lintas waktu.
2. **Vektor konteks berukuran tetap dalam setup encoder-decoder.** Decoder hanya melihat hidden state final encoder, terkompresi atas seluruh input. Input panjang kehilangan detail. Pelajaran 09 membahas ini langsung.
3. **Langit-langit akurasi dependensi-jauh.** LSTM mengungguli RNN polos tetapi masih kesulitan mempropagasikan informasi spesifik lintas 200+ langkah.

Attention memecahkan ketiganya. Transformer membuang rekurensi sepenuhnya. Pelajaran 10 adalah pivotnya.

## Gunakan

`nn.LSTM`, `nn.GRU`, dan `nn.Conv1d` PyTorch siap produksi. Kode training-nya standar.

Hugging Face menyediakan embedding pra-terlatih yang kamu pasang sebagai layer input:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Checklist pakai-ketika-cocok-dengan-konstrain.

- **Inference edge / on-device.** TextCNN dengan embedding GloVe 10-100x lebih kecil daripada transformer. Jika target deploymu ponsel, inilah stack-nya.
- **Klasifikasi streaming / online.** RNN memproses satu token per satu; transformer butuh sekuens penuh. Untuk teks masuk real-time, LSTM masih menang.
- **Model mungil untuk baseline.** Iterasi cepat pada tugas baru. Latih TextCNN dalam 5 menit di CPU.
- **Pelabelan sekuens dengan data terbatas.** BiLSTM-CRF (pelajaran 06) masih arsitektur NER setingkat produksi untuk 1k-10k kalimat berlabel.

Selebihnya serahkan ke transformer.

## Rilis

Simpan sebagai `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## Latihan

1. **Mudah.** Latih TextCNN pada dataset mainan 3-kelas (kamu yang mengarang datanya). Verifikasi bahwa lebar filter (2, 3, 4) mengungguli lebar tunggal (3) pada rata-rata F1.
2. **Sedang.** Implementasikan pooling max, mean, dan last-state untuk pengklasifikasi LSTM. Bandingkan pada dataset kecil; dokumentasikan pooling mana yang menang dan hipotesiskan alasannya.
3. **Sulit.** Bangun tagger NER BiLSTM-CRF (gabungkan pelajaran 06 dan yang ini). Latih pada CoNLL-2003. Bandingkan dengan baseline CRF-saja dari pelajaran 06 dan dengan fine-tune BERT. Laporkan waktu training, memori, dan F1.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| TextCNN | CNN for text | Stack of 1D convolutions over word embeddings with global max-pool. Kim (2014). |
| RNN | Recurrent net | Hidden state updated at each time step: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | Gated RNN | Adds input / forget / output gates + a cell state. Trains stably through long sequences. |
| GRU | Simpler LSTM | Two gates instead of three. Similar accuracy, fewer parameters. |
| Bidirectional | Both directions | Forward + backward RNN concatenated. Every token sees both sides of its context. |
| Vanishing gradient | Training signal dies | Repeated multiplication by <1 weights in plain RNNs makes early-step gradients effectively zero. |

## Bacaan Lanjutan

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — the TextCNN paper. Eight pages. Readable.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — the LSTM paper. Unexpectedly lucid.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — the diagrams that made LSTMs accessible to everyone.
