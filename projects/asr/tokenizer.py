"""
Bilingual (English + Chinese) character-level tokenizer for CTC ASR.

Vocabulary construction:
  - Index 0: CTC blank token (reserved, never emitted by the model).
  - Indices 1+: printable ASCII characters + common Chinese characters +
                 punctuation marks used in both languages.

The tokenizer can:
  - Build a vocabulary from a corpus of transcripts.
  - Load/save a pre-built vocabulary from a file.
  - Encode text → integer sequence.
  - Decode integer sequence → text.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

# ---------------------------------------------------------------------------
# Default character sets
# ---------------------------------------------------------------------------

# English lowercase letters + space
ENGLISH_CHARS = " abcdefghijklmnopqrstuvwxyz"

# Digits
DIGITS = "0123456789"

# Common punctuation (English + Chinese)
PUNCTUATION = ".,!?;:'\"-()[]{}<>/\\@#$%^&*_~`+=|"

# Chinese punctuation
CN_PUNCTUATION = "，。！？；：、""''（）【】《》—…·～"

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class BilingualTokenizer:
    """
    Character-level tokenizer for English + Chinese ASR.

    The vocabulary includes:
      - CTC blank (index 0, reserved)
      - English letters, digits, common punctuation
      - Chinese characters (loaded from a corpus or a pre-built vocab file)

    Usage:
        # Build from transcripts
        tokenizer = BilingualTokenizer.build_from_texts(transcripts)

        # Or load pre-built
        tokenizer = BilingualTokenizer.load("vocab.json")

        # Encode / Decode
        ids = tokenizer.encode("hello 世界")
        text = tokenizer.decode(ids)
    """

    def __init__(
        self,
        char_to_id: Dict[str, int],
        id_to_char: Optional[Dict[int, str]] = None,
    ):
        """
        Args:
            char_to_id: Mapping from character string to integer ID.
                        ID 0 MUST be reserved for CTC blank.
            id_to_char: Optional reverse mapping. Built automatically if not provided.
        """
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char or {v: k for k, v in char_to_id.items()}
        self.blank_id = 0
        self.vocab_size = len(char_to_id)

    # ------------------------------------------------------------------
    # Build from data
    # ------------------------------------------------------------------

    @classmethod
    def build_from_texts(
        cls,
        texts: List[str],
        min_freq: int = 2,
        max_vocab: int = 6000,
        include_chars: Optional[Set[str]] = None,
    ) -> "BilingualTokenizer":
        """
        Build a tokenizer from a list of transcript strings.

        The fixed character set (English, digits, punctuation) is always included.
        Chinese characters are collected from the corpus, filtered by frequency,
        and capped at ``max_vocab`` total vocabulary size.

        Args:
            texts: List of transcript strings (can mix English and Chinese).
            min_freq: Minimum frequency for a Chinese character to be included.
            max_vocab: Maximum total vocabulary size (excluding blank).
            include_chars: Additional characters to force-include.

        Returns:
            A BilingualTokenizer instance.
        """
        # Fixed set: always include these
        fixed_chars: Set[str] = set(ENGLISH_CHARS + DIGITS + PUNCTUATION + CN_PUNCTUATION)
        if include_chars:
            fixed_chars |= include_chars

        # Count character frequencies in the corpus
        freq: Dict[str, int] = {}
        for text in texts:
            for ch in text:
                freq[ch] = freq.get(ch, 0) + 1

        # Collect Chinese characters (CJK Unified Ideographs block)
        # Unicode range: \u4e00-\u9fff (common), \u3400-\u4dbf (extension A)
        cn_chars: Dict[str, int] = {}
        for ch, count in freq.items():
            if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
                if count >= min_freq:
                    cn_chars[ch] = count

        # Sort Chinese characters by frequency (descending)
        sorted_cn = sorted(cn_chars.items(), key=lambda x: -x[1])

        # Build vocabulary: fixed chars first, then Chinese chars up to max_vocab
        vocab: Set[str] = set(fixed_chars)
        remaining_slots = max_vocab - len(vocab)
        for ch, _ in sorted_cn:
            if ch not in vocab:
                if remaining_slots <= 0:
                    break
                vocab.add(ch)
                remaining_slots -= 1

        # Sort for deterministic ordering
        sorted_vocab = sorted(vocab)

        # Assign IDs: index 0 = CTC blank, indices 1+ = characters
        char_to_id: Dict[str, int] = {}
        # Blank is index 0 (not in the vocab set)
        for idx, ch in enumerate(sorted_vocab, start=1):
            char_to_id[ch] = idx

        return cls(char_to_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the vocabulary to a JSON file."""
        path = Path(path)
        data = {
            "char_to_id": self.char_to_id,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {path} ({self.vocab_size} tokens)")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BilingualTokenizer":
        """Load a vocabulary from a JSON file saved by ``save()``."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        char_to_id = data["char_to_id"]
        # Ensure blank is at index 0
        if 0 in char_to_id.values():
            # Remove blank if it was accidentally saved
            char_to_id = {k: v for k, v in char_to_id.items() if v != 0}
        return cls(char_to_id)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """
        Convert a text string to a list of integer token IDs.

        Characters not in the vocabulary are silently skipped.

        Args:
            text: Input string (e.g. "hello 世界").

        Returns:
            List of integer IDs (1-based; 0 is CTC blank).
        """
        ids = []
        for ch in text:
            idx = self.char_to_id.get(ch)
            if idx is not None:
                ids.append(idx)
        return ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a list of texts."""
        return [self.encode(t) for t in texts]

    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of token IDs back to a text string.

        Args:
            ids: List of integer IDs (0 = blank, ignored in output).

        Returns:
            Decoded string.
        """
        chars = []
        for idx in ids:
            if idx == self.blank_id:
                continue
            ch = self.id_to_char.get(idx)
            if ch is not None:
                chars.append(ch)
        return "".join(chars)

    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        """Decode a batch of ID sequences."""
        return [self.decode(ids) for ids in batch_ids]

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"BilingualTokenizer(vocab_size={self.vocab_size}, "
            f"blank_id={self.blank_id})"
        )


# ---------------------------------------------------------------------------
# Utility: build vocab from a transcripts file
# ---------------------------------------------------------------------------


def build_vocab_from_transcripts(
    transcript_path: str,
    output_path: str = "vocab.json",
    min_freq: int = 2,
    max_vocab: int = 6000,
) -> BilingualTokenizer:
    """
    Convenience function: read a transcripts file, build a vocabulary,
    and save it.

    Args:
        transcript_path: Path to TSV file (filename\\ttext per line).
        output_path: Where to save the vocabulary JSON.
        min_freq: Minimum Chinese character frequency.
        max_vocab: Maximum vocabulary size.

    Returns:
        The built tokenizer.
    """
    texts = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) == 2:
                texts.append(parts[1].strip())

    print(f"Building vocabulary from {len(texts)} transcripts ...")
    tokenizer = BilingualTokenizer.build_from_texts(
        texts, min_freq=min_freq, max_vocab=max_vocab
    )
    tokenizer.save(output_path)
    return tokenizer


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "hello world",
        "你好世界",
        "how are you 今天天气怎么样",
        "it is a sunny day 阳光明媚",
    ]
    tok = BilingualTokenizer.build_from_texts(sample_texts, min_freq=1, max_vocab=1000)
    print(tok)
    print(f"Vocab size: {len(tok)}")

    for text in sample_texts:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        print(f"  '{text}' -> {ids} -> '{decoded}'")
