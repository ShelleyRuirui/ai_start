# Music Composer — Genre-Conditioned LSTM Music Generation

A deep learning project that learns to compose **Jazz** and **Blues** music using a conditional LSTM model built with Keras + TensorFlow. Given a seed melody and a genre label, the model autoregressively generates new musical note sequences and exports them as MIDI files.

## Architecture

The model uses a **two-input conditional LSTM** architecture:

```
Input 1: Note sequence (64 tokens) ──► Embedding(64) ──► LSTM(64) ──► LSTM(64) ──► Dropout
                                                                                        │
Input 2: Genre label (0/1)     ──► Embedding(3, 64) ──► Flatten ──► Dense(128, tanh) ──┤
                                                                                        │
                                                                                   Concatenate
                                                                                        │
                                                                                   Dense(128, tanh)
                                                                                        │
                                                                                   Dense(vocab_size, softmax)
                                                                                        │
                                                                              ┌──► Next token ◄──┐
                                                                              │    (autoregressive loop)
                                                                              └──────────────────┘
```

- **Note embedding**: 64-dim, maps each token (note/chord/rest) to a dense vector.
- **LSTM layers**: Two stacked LSTMs (64 units each) capture temporal dependencies in melodies.
- **Genre conditioning**: Genre label (0=jazz, 1=blues) is embedded and projected through a Dense layer, then concatenated with the LSTM output.
- **Output**: Softmax over the vocabulary (~9000 tokens) — predicts the next note token.
- **Total params**: ~1.84M

## Project Structure

```
music_composer/
├── dataset/
│   ├── download_midkar_blues.py   # Download Blues MIDIs from midkar.com
│   ├── README.md                  # Dataset preparation notes
│   ├── requirements.txt           # Requests + BeautifulSoup
│   ├── jazz/                      # Jazz MIDI files (manually downloaded)
│   └── blues/                     # Blues MIDI files (via download script)
│
├── train/
│   ├── preprocess.py              # Parse MIDI → token sequences → sliding windows
│   ├── train.py                   # Build & train the conditional LSTM model
│   ├── train_data.npz             # Preprocessed training data
│   ├── train_steps.txt            # Training log (epoch losses)
│   ├── music_lstm_genre_model.h5  # Final trained model
│   └── checkpoints/               # Per-epoch model checkpoints
│
├── output/
│   ├── generate_music.py          # Autoregressive generation → MIDI export
│   ├── jazz_generated.mid         # Generated Jazz sample
│   └── blues_generated.mid        # Generated Blues sample
│
└── README.md                      # This file
```

## How It Works

### 1. Data Preparation ([`preprocess.py`](train/preprocess.py))

Parses MIDI files from `dataset/jazz/` and `dataset/blues/` using `music21`:

- Extracts **notes** (`NOTE_pitch_duration`), **chords** (`CHORD_p1_p2_..._duration`), and **rests** (`REST_duration`) as string tokens.
- Filters out low-frequency tokens (occurring < 15 times) — maps them to `<OOV>` (out-of-vocabulary).
- Builds a vocabulary of ~9000 tokens.
- Creates training samples via **sliding windows** of length 64: each sample is `(sequence of 64 tokens, genre label) → next token`.

### 2. Training ([`train.py`](train/train.py))

- **Loss**: `sparse_categorical_crossentropy` (avoids one-hot OOM for 9000 classes).
- **Optimizer**: Adam (lr=0.002).
- **Callbacks**:
  - `ReduceLROnPlateau`: halve LR if loss plateaus for 3 epochs.
  - `EarlyStopping`: stop if no improvement for 5 epochs.
  - `ModelCheckpoint`: save every epoch.
- Trained on **250,400 samples** across 12 epochs (best loss: ~4.38).

### 3. Generation ([`generate_music.py`](output/generate_music.py))

- Seeds the model with a random 64-token window from the training data.
- **Autoregressive loop**: predicts the next token, appends it, slides the window, repeats.
- **Temperature sampling** (default: 0.85) controls randomness — lower = more deterministic, higher = more creative.
- Converts the generated token sequence back to a MIDI file using `music21`.

## Usage

### Setup

```bash
# Install dependencies
pip install tensorflow music21 numpy

# For dataset download script
pip install requests beautifulsoup4
```

### Preprocess MIDI files

```bash
cd projects/music_composer/train
python preprocess.py
```

Expects MIDI files in `../dataset/jazz/` and `../dataset/blues/`.

### Train the model

```bash
python train.py
```

Outputs:
- `music_lstm_genre_model.h5` — final model
- `checkpoints/model_epoch_*.h5` — per-epoch checkpoints

### Generate music

```bash
cd projects/music_composer/output
python generate_music.py
```

Outputs:
- `jazz_generated.mid`
- `blues_generated.mid`

### Download Blues dataset

```bash
cd projects/music_composer
python dataset/download_midkar_blues.py --out blues
```

## Configuration

Key parameters you can tune (in each script):

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `SEQ_LENGTH` | `preprocess.py` / `generate_music.py` | 64 | Sliding window size (number of tokens). |
| `MIN_TOKEN_COUNT` | `preprocess.py` | 15 | Minimum frequency to keep a token in vocabulary. |
| `TEMPERATURE` | `generate_music.py` | 0.85 | Sampling temperature (0 = greedy, >1 = more random). |
| `GEN_TOTAL_TOKENS` | `generate_music.py` | 600 | Number of tokens to generate per piece. |
| `lstm_dim` | `train.py` | 64 | LSTM hidden units. |
| `learning_rate` | `train.py` | 0.002 | Adam learning rate. |

## Training Results

```
Epoch 1: loss=4.78, acc=23.1%
Epoch 4: loss=4.44, acc=25.3%
Epoch 7: loss=4.38, acc=25.8%  ← best
Epoch 8+: overfitting begins (loss climbs)
```

The model achieves ~25-26% accuracy (next-token prediction), which is reasonable for a vocabulary of ~9000 tokens with musical structure.

## Dependencies

- Python 3.9+
- TensorFlow ≥ 2.x
- music21
- numpy
- requests, beautifulsoup4 (dataset download only)

## References

- Jazz dataset: [Jazz-ML-Dataset](https://github.com/SaiKayala/Jazz-Ml-Dataset)
- Blues dataset: [midkar.com Blues MIDIs](https://midkar.com/Blues/Blues_MIDIs.html)
