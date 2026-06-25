"""
Training script for the bilingual (English + Chinese) ASR model.

Usage:
    # Build vocabulary from transcripts, then train
    python train.py --data_dir /path/to/audio --transcripts /path/to/transcripts.txt

    # Or use a pre-built vocabulary
    python train.py --data_dir /path/to/audio --transcripts /path/to/transcripts.txt \\
                    --vocab vocab.json

Data format expectation:
    - Audio files in a directory (wav/flac/...).
    - A transcripts file where each line is:
          <filename>\t<transcript>
      e.g.:
          sample001.wav	hello world
          sample002.wav	你好世界
          sample003.wav	how are you 今天天气怎么样

The script:
    1. Builds or loads a bilingual character-level vocabulary.
    2. Loads audio & extracts MFCC features.
    3. Builds the model with the correct output dimension.
    4. Trains with CTC loss.
    5. Saves the best checkpoint + vocabulary.
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Local imports
from config import (
    AudioConfig,
    ModelConfig,
    TrainingConfig,
    audio_config,
    model_config,
    training_config,
)
from feature_extraction import extract_mfcc, load_audio, normalize_features
from model import build_asr_model, ctc_loss_function, get_model_summary
from tokenizer import BilingualTokenizer, build_vocab_from_transcripts

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def parse_transcripts(transcript_path: str) -> List[Tuple[str, str]]:
    """
    Read a transcript file.

    Format per line:  <relative_path>\t<transcript>
    Returns list of (audio_path, transcript_text).
    """
    entries = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            audio_rel, text = parts
            entries.append((audio_rel, text.strip()))
    return entries


def compute_input_length(
    waveform_len: int,
    hop_length: int,
    conv_pool_sizes: List[int],
) -> int:
    """
    Compute the output time steps after conv + pooling layers.

    Each MaxPooling2D with pool_size=(p, 1) divides time by p.
    """
    T = (waveform_len + hop_length - 1) // hop_length  # MFCC time steps
    for p in conv_pool_sizes:
        T = (T + p - 1) // p  # ceil division
    return max(T, 1)


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------


class ASRDataGenerator(tf.keras.utils.Sequence):
    """
    Generates batches of (features, labels) for CTC training.

    Yields:
        inputs: (batch, time, n_mfcc) float32
        labels: SparseTensor of ground-truth label sequences
        input_length: (batch,) int32 – length after conv downsampling
        label_length: (batch,) int32 – length of each label sequence
    """

    def __init__(
        self,
        entries: List[Tuple[str, str]],
        audio_dir: str,
        tokenizer: BilingualTokenizer,
        audio_cfg: AudioConfig,
        model_cfg: ModelConfig,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.entries = entries
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.audio_cfg = audio_cfg
        self.model_cfg = model_cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.entries))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.entries) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_entries = [self.entries[i] for i in batch_indexes]
        return self._generate_batch(batch_entries)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_batch(self, batch_entries):
        cfg = self.audio_cfg
        mcfg = self.model_cfg

        batch_features = []
        batch_labels = []
        batch_input_lengths = []
        batch_label_lengths = []

        for rel_path, text in batch_entries:
            full_path = os.path.join(self.audio_dir, rel_path)
            if not os.path.exists(full_path):
                continue

            # Load audio
            waveform, sr = load_audio(full_path, target_sr=cfg.sample_rate)

            # Extract MFCCs
            features = extract_mfcc(
                waveform,
                sr=sr,
                n_mfcc=cfg.n_mfcc,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.win_length,
            )
            features = normalize_features(features)

            # Convert text to label sequence using the bilingual tokenizer
            label_seq = self.tokenizer.encode(text)

            # Compute input length after conv downsampling
            input_len = compute_input_length(
                len(waveform), cfg.hop_length, mcfg.conv_pool_sizes
            )

            batch_features.append(features)
            batch_labels.append(np.array(label_seq, dtype=np.int32))
            batch_input_lengths.append(input_len)
            batch_label_lengths.append(len(label_seq))

        if not batch_features:
            # Fallback: return a dummy batch
            return (
                np.zeros((1, 1, cfg.n_mfcc), dtype=np.float32),
                tf.SparseTensor(
                    indices=[[0, 0]], values=[1], dense_shape=[1, 1]
                ),
                np.array([1], dtype=np.int32),
                np.array([1], dtype=np.int32),
            )

        # Pad features to same time steps within the batch
        max_T = max(f.shape[0] for f in batch_features)
        padded_features = np.zeros(
            (len(batch_features), max_T, cfg.n_mfcc), dtype=np.float32
        )
        for i, f in enumerate(batch_features):
            T = f.shape[0]
            padded_features[i, :T, :] = f

        # Build SparseTensor for labels
        indices = []
        values = []
        max_label_len = max(len(l) for l in batch_labels)
        dense_shape = [len(batch_labels), max_label_len]
        for i, label in enumerate(batch_labels):
            for j, val in enumerate(label):
                indices.append([i, j])
                values.append(int(val))
        sparse_labels = tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=dense_shape,
        )

        return (
            padded_features,
            sparse_labels,
            np.array(batch_input_lengths, dtype=np.int32),
            np.array(batch_label_lengths, dtype=np.int32),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train bilingual ASR model")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing audio files.",
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        required=True,
        help="Path to transcripts file (tsv: filename\\ttext).",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="Path to pre-built vocabulary JSON. If not provided, builds from transcripts.",
    )
    parser.add_argument(
        "--max_vocab",
        type=int,
        default=6000,
        help="Maximum vocabulary size (default: 6000).",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=2,
        help="Minimum frequency for Chinese characters (default: 2).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=training_config.batch_size,
        help=f"Batch size (default: {training_config.batch_size}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=training_config.epochs,
        help=f"Number of epochs (default: {training_config.epochs}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=training_config.learning_rate,
        help=f"Learning rate (default: {training_config.learning_rate}).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=training_config.checkpoint_path,
        help=f"Model save path (default: {training_config.checkpoint_path}).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=training_config.val_split,
        help=f"Validation split ratio (default: {training_config.val_split}).",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision training (faster on GPUs).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Mixed precision
    # ------------------------------------------------------------------
    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled.")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading transcripts from: {args.transcripts}")
    entries = parse_transcripts(args.transcripts)
    print(f"Found {len(entries)} entries.")

    if len(entries) == 0:
        print("ERROR: No entries found. Check your transcripts file.")
        return

    # ------------------------------------------------------------------
    # Build or load vocabulary
    # ------------------------------------------------------------------
    if args.vocab and os.path.exists(args.vocab):
        print(f"Loading vocabulary from: {args.vocab}")
        tokenizer = BilingualTokenizer.load(args.vocab)
    else:
        print("Building vocabulary from transcripts...")
        texts = [text for _, text in entries]
        tokenizer = BilingualTokenizer.build_from_texts(
            texts, min_freq=args.min_freq, max_vocab=args.max_vocab
        )
        # Save vocabulary alongside the model
        vocab_path = args.checkpoint.replace(".h5", "_vocab.json")
        tokenizer.save(vocab_path)

    print(f"Vocabulary size: {len(tokenizer)} (plus CTC blank = {len(tokenizer) + 1} classes)")

    # Update model config with actual vocabulary size
    model_config.num_classes = len(tokenizer) + 1  # +1 for CTC blank
    model_config.blank_label = 0

    # Shuffle and split
    random.shuffle(entries)
    val_size = int(len(entries) * args.val_split)
    train_entries = entries[val_size:]
    val_entries = entries[:val_size]
    print(f"Train samples: {len(train_entries)}, Val samples: {len(val_entries)}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    print("Building ASR model...")
    model = build_asr_model(input_dim=audio_config.n_mfcc)
    print(get_model_summary(model))

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # ------------------------------------------------------------------
    # Data generators
    # ------------------------------------------------------------------
    train_gen = ASRDataGenerator(
        entries=train_entries,
        audio_dir=args.data_dir,
        tokenizer=tokenizer,
        audio_cfg=audio_config,
        model_cfg=model_config,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_gen = ASRDataGenerator(
        entries=val_entries,
        audio_dir=args.data_dir,
        tokenizer=tokenizer,
        audio_cfg=audio_config,
        model_cfg=model_config,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")

    # ------------------------------------------------------------------
    # Checkpoint & early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = args.checkpoint

    # TensorBoard writer
    log_dir = training_config.log_dir
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "train")
    )
    val_summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "val")
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        train_loss.reset_state()
        for batch_idx in range(len(train_gen)):
            features, sparse_labels, input_lengths, label_lengths = train_gen[
                batch_idx
            ]

            with tf.GradientTape() as tape:
                logits = model(features, training=True)
                loss = ctc_loss_function(
                    y_true=sparse_labels,
                    y_pred=logits,
                    label_length=label_lengths,
                    input_length=input_lengths,
                )

            grads = tape.gradient(loss, model.trainable_variables)
            # Gradient clipping
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss.update_state(loss)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Batch {batch_idx + 1:3d}/{len(train_gen)} | "
                    f"Loss: {loss:.4f}"
                )

        # --- Validation ---
        val_loss.reset_state()
        for batch_idx in range(len(val_gen)):
            features, sparse_labels, input_lengths, label_lengths = val_gen[
                batch_idx
            ]
            logits = model(features, training=False)
            loss = ctc_loss_function(
                y_true=sparse_labels,
                y_pred=logits,
                label_length=label_lengths,
                input_length=input_lengths,
            )
            val_loss.update_state(loss)

        # --- Logging ---
        print(
            f"\nEpoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss.result():.4f} | "
            f"Val Loss: {val_loss.result():.4f}\n"
        )

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch)
        with val_summary_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), step=epoch)

        # --- Checkpoint ---
        current_val_loss = val_loss.result().numpy()
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            model.save(checkpoint_path)
            print(f"  -> Model saved to {checkpoint_path} (val_loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> No improvement for {patience_counter} epoch(s).")

        # --- Early stopping ---
        if patience_counter >= training_config.early_stop_patience:
            print(
                f"Early stopping triggered after {epoch} epochs. "
                f"Best val_loss: {best_val_loss:.4f}"
            )
            break

    print("Training complete.")
    print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
