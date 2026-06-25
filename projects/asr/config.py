"""
Configuration for the ASR model.

All hyper-parameters are centralised here so they can be tuned easily.

Vocabulary is now handled by ``BilingualTokenizer`` (see tokenizer.py),
which supports English + Chinese characters.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AudioConfig:
    """Audio preprocessing parameters."""
    sample_rate: int = 16000
    n_mfcc: int = 40
    n_fft: int = 512
    hop_length: int = 160          # 10 ms at 16 kHz
    win_length: int = 400          # 25 ms at 16 kHz
    max_duration: float = 10.0     # seconds – pad / truncate to this


@dataclass
class ModelConfig:
    """Model architecture parameters."""
    # --- Spectrogram / feature shape ---
    n_mfcc: int = 40

    # --- Conv front-end ---
    conv_filters: List[int] = field(default_factory=lambda: [32, 64])
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3])
    conv_strides: List[int] = field(default_factory=lambda: [1, 1])
    conv_pool_sizes: List[int] = field(default_factory=lambda: [2, 2])

    # --- RNN back-end ---
    rnn_units: List[int] = field(default_factory=lambda: [128, 128])
    rnn_dropout: float = 0.2
    rnn_bidirectional: bool = True

    # --- Dense / output ---
    dense_units: int = 128
    dropout_rate: float = 0.2

    # --- Vocabulary ---
    # The vocabulary is now managed by BilingualTokenizer (tokenizer.py).
    # Set these after building/loading a tokenizer.
    num_classes: int = 0       # will be set from tokenizer.vocab_size + 1 (blank)
    blank_label: int = 0       # CTC blank index

    # Path to saved vocabulary JSON (optional – used at training time)
    vocab_path: str = "vocab.json"


@dataclass
class TrainingConfig:
    """Training hyper-parameters."""
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-3
    # Early stopping
    early_stop_patience: int = 10
    # Model checkpoint path
    checkpoint_path: str = "asr_model.h5"
    # TensorBoard log dir
    log_dir: str = "./logs/asr"
    # Train / val split ratio
    val_split: float = 0.1


# ---------------------------------------------------------------------------
# Singleton instances (imported elsewhere)
# ---------------------------------------------------------------------------
audio_config = AudioConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
