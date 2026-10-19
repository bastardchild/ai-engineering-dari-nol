# Model Sequence-to-Sequence

> Dua RNN yang berpura-pura menjadi penerjemah. Bottleneck yang mereka tabrak adalah alasan attention ada.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75 minutes

## Masalahnya

Klasifikasi memetakan sekuens bervariabel panjang ke satu label. Penerjemahan memetakan sekuens bervariabel panjang ke sekuens bervariabel panjang lainnya. Input dan output hidup di vocabulary berbeda, mungkin bahasa berbeda, tanpa jaminan paritas panjang.

Arsitektur seq2seq (Sutskever, Vinyals, Le, 2014) memecahkannya dengan resep yang sengaja sederhana. Dua RNN. Satu membaca kalimat sumber dan menghasilkan vektor konteks berukuran tetap. Satunya lagi membaca vektor itu dan menghasilkan kalimat target token per token. Kode yang sama yang kamu tulis untuk pelajaran 08, dilem dengan cara berbeda.

Ini layak dipelajari karena dua alasan. Pertama, bottleneck vektor-konteks adalah kegagalan paling berguna secara pedagogis di NLP. Ia memotivasi semua hal yang dikuasai attention dan transformer. Kedua, resep training-nya (teacher forcing, scheduled sampling, beam search saat inference) masih berlaku untuk setiap sistem generasi modern termasuk LLM.

## Konsepnya

**Encoder.** RNN yang membaca kalimat sumber. Hidden state finalnya adalah **vektor konteks** — ringkasan berukuran tetap dari seluruh input. Katanya tak ada yang hilang selain sumbernya.

**Decoder.** RNN lain yang diinisialisasi dari vektor konteks. Pada tiap langkah ia mengambil token yang dihasilkan sebelumnya sebagai input dan menghasilkan distribusi atas vocabulary target. Sampel atau argmax untuk memilih token berikutnya. Umpankan kembali. Ulangi sampai token `<EOS>` dihasilkan atau panjang maksimum tercapai.

**Training:** loss cross-entropy pada tiap langkah decoder, dijumlahkan atas sekuens. Backprop through time standar melalui kedua jaringan.

**Teacher forcing.** Selama training, input decoder pada langkah `t` adalah token *ground-truth* pada posisi `t-1`, bukan prediksi decoder sendiri sebelumnya. Ini menstabilkan training; tanpanya, kesalahan awal berantai dan model tak pernah belajar. Saat inference, kamu harus memakai prediksi model sendiri, sehingga selalu ada gap distribusi train/inference. Gap itu disebut **exposure bias**.

**Bottleneck-nya.** Semua yang dipelajari encoder tentang sumber harus diperas ke satu vektor konteks itu. Kalimat panjang kehilangan detail. Kata jarang kabur. Penyusunan ulang (chat noir vs. black cat) harus dihafal, bukan dihitung.

Attention (pelajaran 10) memperbaikinya dengan membiarkan decoder melihat *setiap* hidden state encoder, bukan hanya yang terakhir. Itulah seluruh tawarannya.

```figure
lstm-gates
```

## Bangun Sendiri

### Langkah 1: encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` berbentuk `[batch, seq_len, hidden_dim]` — satu hidden state per posisi input. `hidden` berbentuk `[1, batch, hidden_dim]` — langkah terakhir. Pelajaran 08 bilang "pool atas outputs untuk klasifikasi." Di sini kita menyimpan hidden state terakhir sebagai vektor konteks, dan mengabaikan output per langkah.

### Langkah 2: decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder dipanggil satu langkah per satu. Input: batch token tunggal dan hidden state kini. Output: logit vocabulary untuk token berikutnya dan hidden state yang diperbarui.

### Langkah 3: loop training dengan teacher forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

Dua kenop yang patut disebut. `ignore_index=0` melewatkan loss pada token padding. `teacher_forcing_ratio` adalah probabilitas memakai token asli vs. prediksi model pada tiap langkah. Mulai dari 1.0 (teacher forcing penuh) dan turunkan bertahap ke ~0.5 selama training untuk menutup gap exposure-bias.

### Langkah 4: loop inference (greedy)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Decoding greedy memilih token berprobabilitas tertinggi pada tiap langkah. Ia bisa tersesat: begitu kamu berkomitmen pada satu token, kamu tak bisa menariknya kembali. **Beam search** mempertahankan `k` sekuens parsial teratas tetap hidup dan memilih yang berskor tertinggi di akhir. Lebar beam 3-5 itu standar.

### Langkah 5: bottleneck, didemonstrasikan

Latih model pada tugas salin mainan: sumber `[a, b, c, d, e]`, target `[a, b, c, d, e]`. Tambah panjang sekuens. Amati akurasi.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

Satu hidden state GRU tidak bisa menghafal input 40-token secara lossless. Informasinya ada di setiap langkah encoder, tetapi decoder hanya melihat state terakhir. Attention memperbaikinya secara langsung.

## Gunakan

PyTorch punya `nn.Transformer` dan template seq2seq berbasis `nn.LSTM`. Pustaka `transformers` Hugging Face menyediakan model encoder-decoder penuh (BART, T5, mBART, NLLB) yang dilatih pada miliaran token.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

Encoder-decoder modern membuang RNN demi transformer. Bentuk tingkat-tingginya (encoder, decoder, hasilkan-token-per-token) identik dengan paper seq2seq 2014. Mekanisme di dalam tiap blok berbeda.

### Kapan masih meraih seq2seq berbasis RNN

Hampir tidak pernah, untuk proyek baru. Pengecualian spesifik:

- Penerjemahan streaming di mana kamu mengonsumsi input satu token per satu dengan memori terbatas.
- Generasi teks on-device di mana biaya memori transformer prohibitif.
- Pedagogi. Memahami bottleneck encoder-decoder adalah jalan tercepat memahami mengapa transformer menang.

### Exposure bias dan mitigasinya

- **Scheduled sampling.** Turunkan rasio teacher forcing selama training supaya model belajar pulih dari kesalahannya sendiri.
- **Minimum risk training.** Latih pada skor BLEU level-kalimat alih-alih cross-entropy level-token. Lebih dekat ke yang sebenarnya kamu inginkan.
- **Fine-tuning reinforcement learning.** Beri reward pada generator sekuens dengan metrik. Dipakai dalam RLHF LLM modern.

Ketiganya masih berlaku untuk generasi berbasis transformer.

## Rilis

Simpan sebagai `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Latihan

1. **Mudah.** Implementasikan tugas salin mainan. Latih seq2seq GRU pada pasangan input-output di mana target sama dengan sumber. Ukur akurasi pada panjang 5, 10, 20. Reproduksi bottleneck-nya.
2. **Sedang.** Tambahkan decoding beam search dengan lebar beam 3. Ukur BLEU pada korpus paralel kecil melawan greedy. Dokumentasikan di mana beam search menang (biasanya token-token akhir) dan di mana ia tak ada bedanya.
3. **Sulit.** Fine-tune `facebook/bart-base` pada dataset parafrasa 10k-pasangan. Bandingkan output beam-4 model fine-tune dengan model dasar pada input held-out. Laporkan BLEU dan pilih 10 contoh kualitatif.

## Istilah Kunci

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Encoder | Input RNN | Reads source. Produces per-step hidden states and a final context vector. |
| Decoder | Output RNN | Initialized from context vector. Generates target tokens one at a time. |
| Context vector | The summary | Final encoder hidden state. Fixed size. The bottleneck attention solves. |
| Teacher forcing | Use true tokens | Feed the ground-truth previous token at training time. Stabilizes learning. |
| Exposure bias | Train/test gap | Model trained on true tokens never practiced recovering from its own mistakes. |
| Beam search | Better decoding | Keep top-k partial sequences alive at each step instead of committing greedily. |

## Bacaan Lanjutan

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — the original seq2seq paper. Four pages.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — introduced the GRU and the encoder-decoder framing.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — the attention paper. Read immediately after this lesson.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — buildable seq2seq + attention code.
