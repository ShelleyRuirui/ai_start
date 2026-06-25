# ASR — Bilingual (English + Chinese) Speech Recognition with Keras + TensorFlow

A CTC-based automatic speech recognition model supporting **English and Chinese** (mixed-language) built with the Keras functional API and TensorFlow 2.

## Architecture (DeepSpeech2-style)

```
Input: (batch, time, 40 MFCCs)
  → Conv2D(32) + BatchNorm + MaxPool
  → Conv2D(64) + BatchNorm + MaxPool
  → Flatten frequency dim
  → Bidirectional LSTM(128) → Bidirectional LSTM(128)
  → Dense(128) + Dropout
  → Dense(num_classes) + Softmax  →  CTC Decode → Text
```

- **Output vocabulary**: English letters, digits, punctuation + up to ~6000 common Chinese characters + CTC blank.
- **CTC loss** handles variable-length input/output alignment.
- **3.2M+ parameters** (depends on vocabulary size).

## Project Structure

| File | Purpose |
|------|---------|
| [`config.py`](config.py) | Audio, model, and training hyper-parameters. |
| [`tokenizer.py`](tokenizer.py) | Bilingual character-level tokenizer (build, save, load, encode, decode). |
| [`feature_extraction.py`](feature_extraction.py) | Load audio → MFCC → normalize → pad/truncate. |
| [`model.py`](model.py) | Keras functional API model definition + CTC loss. |
| [`train.py`](train.py) | Custom training loop with CTC loss, checkpointing, early stopping. |
| [`predict.py`](predict.py) | Inference (greedy or beam-search decoding). |
| [`requirements.txt`](requirements.txt) | Python dependencies. |

## Setup

```bash
# Install dependencies (ideally inside a virtual environment)
pip install -r requirements.txt
```

## Data Format

Prepare a **tab-separated transcripts file** (one utterance per line). Both English and Chinese are supported, including mixed-language:

```
sample001.wav	hello world
sample002.wav	你好世界
sample003.wav	how are you 今天天气怎么样
sample004.wav	it is a sunny day 阳光明媚
```

Audio files should be in a separate directory. Supported formats: `wav`, `flac`, `mp3`, `m4a`, `ogg`.

## Training

### Step 1: Build vocabulary (optional — done automatically during training)

```bash
python tokenizer.py --transcripts /path/to/transcripts.txt --output vocab.json
```

### Step 2: Train the model

```bash
cd projects/asr

python train.py \
    --data_dir /path/to/audio/files \
    --transcripts /path/to/transcripts.txt \
    --batch_size 16 \
    --epochs 100 \
    --lr 0.001 \
    --checkpoint asr_model.h5 \
    --val_split 0.1 \
    --max_vocab 6000 \
    --min_freq 2
```

If you already have a pre-built vocabulary:

```bash
python train.py \
    --data_dir /path/to/audio \
    --transcripts /path/to/transcripts.txt \
    --vocab vocab.json
```

### Optional flags

| Flag | Description |
|------|-------------|
| `--mixed_precision` | Enable mixed-precision training (faster on GPU). |
| `--max_vocab` | Maximum vocabulary size (default: 6000). |
| `--min_freq` | Minimum frequency for Chinese characters (default: 2). |

Training logs are written to `./logs/asr/` and can be viewed with TensorBoard:

```bash
tensorboard --logdir ./logs/asr
```

### Outputs

- `asr_model.h5` — best model checkpoint
- `asr_model_vocab.json` — vocabulary file (saved alongside the model)

## Inference

### Single file

```bash
python predict.py \
    --model asr_model.h5 \
    --vocab asr_model_vocab.json \
    --audio sample.wav
```

### Batch directory

```bash
python predict.py \
    --model asr_model.h5 \
    --vocab asr_model_vocab.json \
    --audio_dir ./test_audio/ \
    --output predictions.txt
```

### Decoding methods

- `--decode greedy` — argmax at each time step, collapse repeats, remove blanks (default, fast).
- `--decode beam_search` — beam-search decoding (slower but more accurate).

## Configuration

All hyper-parameters are in [`config.py`](config.py). Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | Audio sample rate (Hz). |
| `n_mfcc` | 40 | Number of MFCC coefficients. |
| `rnn_units` | [128, 128] | LSTM units per layer. |
| `learning_rate` | 1e-3 | Adam learning rate. |
| `early_stop_patience` | 10 | Stop if no val loss improvement for N epochs. |
| `max_vocab` | 6000 | Maximum vocabulary size (Chinese chars + English + punctuation). |

## Tokenizer

The [`BilingualTokenizer`](tokenizer.py) handles:

- **Building** a vocabulary from a corpus of transcripts (collects English letters, digits, punctuation + frequent Chinese characters).
- **Saving/loading** vocabulary as JSON.
- **Encoding** text → integer ID sequence (for training).
- **Decoding** integer ID sequence → text (for inference).

Index 0 is always reserved for the CTC blank token.

## Requirements

- Python 3.9+
- TensorFlow ≥ 2.12.0
- librosa ≥ 0.10.0
- soundfile ≥ 0.12.0
- jieba ≥ 0.42.0
- numpy, scipy, matplotlib
