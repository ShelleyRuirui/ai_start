"""
ASR model definition using Keras functional API.

Architecture (DeepSpeech2-style):
  1. Optional conv front-end (2 x Conv2D + MaxPool) to downsample frequency dim.
  2. Reshape to (time, features) for RNN.
  3. Stacked Bidirectional LSTM layers.
  4. Dense projection + softmax over vocabulary (+ CTC blank).

The model outputs a probability distribution over characters at each time step.
Training uses CTC loss (handled in train.py).
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers

from config import ModelConfig, model_config


def build_asr_model(
    input_dim: int,
    config: Optional[ModelConfig] = None,
) -> Model:
    """
    Build and return a CTC-based ASR model.

    Args:
        input_dim: Number of feature dimensions per time step (e.g. n_mfcc).
        config: ModelConfig instance. Uses default if None.

    Returns:
        A Keras Model that accepts (batch, time, input_dim) and outputs
        (batch, time, num_classes) log-probabilities.
    """
    if config is None:
        config = model_config

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    # Shape: (batch, time, freq) – we treat freq as the "channel" dimension
    # for Conv1D, or we can use Conv2D with a reshaped view.
    audio_input = layers.Input(
        shape=(None, input_dim), name="audio_input", dtype=tf.float32
    )

    # ------------------------------------------------------------------
    # Conv front-end (applied as Conv1D along time axis)
    # ------------------------------------------------------------------
    x = audio_input
    # Add a channel dimension for Conv1D: (batch, time, freq, 1)
    x = layers.Reshape((-1, input_dim, 1), name="expand_dims")(x)

    for i, (filters, kernel, stride, pool) in enumerate(
        zip(
            config.conv_filters,
            config.conv_kernel_sizes,
            config.conv_strides,
            config.conv_pool_sizes,
        )
    ):
        # Conv2D: kernel over (time, freq)
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(kernel, 3),  # (time_kernel, freq_kernel)
            strides=(stride, 1),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-5),
            name=f"conv_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.MaxPooling2D(
            pool_size=(pool, 1),  # pool only along time
            name=f"pool_{i}",
        )(x)

    # Squeeze the frequency dimension: now (batch, time', freq')
    # After conv layers the freq dim may have changed; we flatten it.
    # But since we used kernel (k, 3) with padding="same" and stride 1 on freq,
    # freq dim stays input_dim. We'll just squeeze the last dim.
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]), name="flatten_freq")(x)

    # ------------------------------------------------------------------
    # RNN back-end (Bidirectional LSTMs)
    # ------------------------------------------------------------------
    for i, units in enumerate(config.rnn_units):
        return_sequences = i < len(config.rnn_units) - 1 or True
        if config.rnn_bidirectional:
            x = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=True,
                    dropout=config.rnn_dropout,
                    kernel_regularizer=regularizers.l2(1e-5),
                ),
                name=f"bidirectional_lstm_{i}",
            )(x)
        else:
            x = layers.LSTM(
                units,
                return_sequences=True,
                dropout=config.rnn_dropout,
                kernel_regularizer=regularizers.l2(1e-5),
                name=f"lstm_{i}",
            )(x)

    # ------------------------------------------------------------------
    # Dense projection
    # ------------------------------------------------------------------
    x = layers.Dense(
        config.dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-5),
        name="dense",
    )(x)
    x = layers.Dropout(config.dropout_rate, name="dropout")(x)

    # ------------------------------------------------------------------
    # Output: log-softmax over vocabulary (+ blank)
    # ------------------------------------------------------------------
    output = layers.Dense(
        config.num_classes,
        activation="softmax",
        name="output",
    )(x)

    model = Model(inputs=audio_input, outputs=output, name="ASR_Model")
    return model


def ctc_loss_function(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    label_length: tf.Tensor,
    input_length: tf.Tensor,
) -> tf.Tensor:
    """
    Compute CTC loss.

    Args:
        y_true: Sparse tensor of ground-truth label sequences.
        y_pred: (batch, time, num_classes) – model output probabilities.
        label_length: (batch,) – length of each label sequence.
        input_length: (batch,) – length of each input sequence (after conv
                      downsampling).

    Returns:
        Scalar loss.
    """
    # Convert dense predictions to log-probabilities for CTC
    # y_pred is already softmax; CTC expects logits or log-probabilities.
    log_probs = tf.math.log(y_pred + 1e-7)

    # Compute CTC loss
    batch_size = tf.shape(y_pred)[0]
    input_length_t = tf.cast(input_length, tf.int32)
    label_length_t = tf.cast(label_length, tf.int32)

    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=log_probs,
        label_length=label_length_t,
        logit_length=input_length_t,
        blank_index=0,  # index 0 is reserved for blank
    )
    return tf.reduce_mean(loss)


def get_model_summary(model: Model) -> str:
    """Return the model summary as a string."""
    string_list = []
    model.summary(print_fn=lambda s: string_list.append(s))
    return "\n".join(string_list)


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    m = build_asr_model(input_dim=model_config.n_mfcc)
    print(get_model_summary(m))
    print(f"Number of classes (vocab + blank): {model_config.num_classes}")
    print(f"Vocabulary: {model_config.vocab}")
