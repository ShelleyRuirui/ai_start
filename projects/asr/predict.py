"""
Inference / prediction script for the bilingual (English + Chinese) ASR model.

Usage:
    python predict.py --model asr_model.h5 --vocab vocab.json --audio sample.wav
    python predict.py --model asr_model.h5 --vocab vocab.json \\
                      --audio_dir ./test_audio/ --output predictions.txt
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf

from config import AudioConfig, ModelConfig, audio_config, model_config
from feature_extraction import extract_mfcc, load_audio, normalize_features
from tokenizer import BilingualTokenizer


def load_trained_model(model_path: str) -> tf.keras.Model:
    """
    Load a saved Keras ASR model.

    Args:
        model_path: Path to the .h5 model file.

    Returns:
        Loaded Keras model.
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Model loaded from: {model_path}")
    return model


def decode_predictions(
    predictions: np.ndarray,
    tokenizer: BilingualTokenizer,
    method: str = "greedy",
) -> List[str]:
    """
    Decode model output probabilities into text strings.

    Args:
        predictions: (batch, time, num_classes) – softmax probabilities.
        tokenizer: BilingualTokenizer for decoding IDs to text.
        method: 'greedy' or 'beam_search'.

    Returns:
        List of decoded text strings.
    """
    if method == "greedy":
        # Greedy decoding: argmax at each time step, then collapse repeats
        # and remove blanks.
        decoded_texts = []
        argmax_indices = np.argmax(predictions, axis=-1)  # (batch, time)

        for batch_idx in range(argmax_indices.shape[0]):
            chars = []
            prev = tokenizer.blank_id
            for idx in argmax_indices[batch_idx]:
                if idx != prev and idx != tokenizer.blank_id:
                    ch = tokenizer.id_to_char.get(int(idx), "")
                    chars.append(ch)
                prev = idx
            decoded_texts.append("".join(chars))
        return decoded_texts

    elif method == "beam_search":
        # Beam search decoding using tf.nn.ctc_beam_search_decoder
        log_probs = tf.math.log(predictions + 1e-7)
        # Transpose to (time, batch, classes) as required by the decoder
        decoded, _ = tf.nn.ctc_beam_search_decoder(
            tf.transpose(log_probs, perm=[1, 0, 2]),
            sequence_length=[predictions.shape[1]] * predictions.shape[0],
            beam_width=100,
            top_paths=1,
        )
        decoded_dense = tf.sparse.to_dense(decoded[0])
        decoded_texts = []
        for row in decoded_dense.numpy():
            text = "".join(
                tokenizer.id_to_char.get(int(idx), "")
                for idx in row
                if idx != tokenizer.blank_id
            )
            decoded_texts.append(text)
        return decoded_texts

    else:
        raise ValueError(f"Unknown decoding method: {method}")


def predict_single(
    model: tf.keras.Model,
    audio_path: str,
    tokenizer: BilingualTokenizer,
    audio_cfg: AudioConfig = audio_config,
    decode_method: str = "greedy",
) -> str:
    """
    Run ASR on a single audio file.

    Args:
        model: Trained Keras model.
        audio_path: Path to audio file.
        tokenizer: BilingualTokenizer for decoding.
        audio_cfg: Audio configuration.
        decode_method: 'greedy' or 'beam_search'.

    Returns:
        Transcribed text.
    """
    # Load and preprocess
    waveform, sr = load_audio(audio_path, target_sr=audio_cfg.sample_rate)
    features = extract_mfcc(
        waveform,
        sr=sr,
        n_mfcc=audio_cfg.n_mfcc,
        n_fft=audio_cfg.n_fft,
        hop_length=audio_cfg.hop_length,
        win_length=audio_cfg.win_length,
    )
    features = normalize_features(features)

    # Add batch dimension: (1, T, D)
    features = np.expand_dims(features, axis=0).astype(np.float32)

    # Predict
    predictions = model.predict(features, verbose=0)

    # Decode
    texts = decode_predictions(
        predictions,
        tokenizer=tokenizer,
        method=decode_method,
    )
    return texts[0]


def predict_batch(
    model: tf.keras.Model,
    audio_dir: str,
    tokenizer: BilingualTokenizer,
    audio_cfg: AudioConfig = audio_config,
    decode_method: str = "greedy",
    output_path: Optional[str] = None,
) -> List[str]:
    """
    Run ASR on all audio files in a directory.

    Args:
        model: Trained Keras model.
        audio_dir: Directory containing audio files.
        tokenizer: BilingualTokenizer for decoding.
        audio_cfg: Audio configuration.
        decode_method: 'greedy' or 'beam_search'.
        output_path: Optional path to save predictions.

    Returns:
        List of (filename, transcript) strings.
    """
    supported_extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    audio_files = sorted(
        [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if Path(f).suffix.lower() in supported_extensions
        ]
    )

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return []

    results = []
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        print(f"Transcribing: {filename} ...")
        try:
            text = predict_single(
                model, audio_path, tokenizer, audio_cfg, decode_method
            )
            results.append((filename, text))
            print(f"  -> {text}")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append((filename, f"[ERROR: {e}]"))

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for filename, text in results:
                f.write(f"{filename}\t{text}\n")
        print(f"\nPredictions saved to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Bilingual ASR Inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.h5).",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to vocabulary JSON file.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to a single audio file.",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=None,
        help="Directory with audio files for batch inference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for batch predictions (tsv).",
    )
    parser.add_argument(
        "--decode",
        type=str,
        default="greedy",
        choices=["greedy", "beam_search"],
        help="Decoding method (default: greedy).",
    )
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading vocabulary from: {args.vocab}")
    tokenizer = BilingualTokenizer.load(args.vocab)
    print(f"Vocabulary size: {len(tokenizer)}")

    # Load model
    model = load_trained_model(args.model)

    if args.audio:
        text = predict_single(
            model, args.audio, tokenizer, decode_method=args.decode
        )
        print(f"\nTranscription: {text}")

    if args.audio_dir:
        predict_batch(
            model,
            args.audio_dir,
            tokenizer,
            decode_method=args.decode,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
