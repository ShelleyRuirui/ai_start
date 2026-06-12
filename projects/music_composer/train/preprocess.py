import warnings

import os
import numpy as np
from music21 import converter, note, chord

# ===================== 全局配置 =====================
# 屏蔽全部无关网络警告
warnings.filterwarnings("ignore")

GENRE_MAP = {"jazz": 0, "blues": 1}
SEQ_LENGTH = 128
DATASET_ROOT = "../dataset"
REST_SYMBOL = "REST"
# ====================================================

all_token_sequences = []
all_genre_labels = []

def extract_full_music_sequence(midi_file_path):
    try:
        midi_stream = converter.parse(midi_file_path)
    except Exception as e:
        print(f"[跳过损坏文件] {midi_file_path} | Error: {str(e)}")
        return []

    token_list = []
    for elem in midi_stream.flat:
        try:
            duration = elem.quarterLength
        except:
            continue

        # 单个音符
        if isinstance(elem, note.Note):
            pitch = elem.pitch.midi
            token = f"NOTE_{pitch}_{duration}"
            token_list.append(token)
        # 完整和弦
        elif isinstance(elem, chord.Chord):
            pitches = sorted([n.pitch.midi for n in elem.notes])
            pitch_str = "_".join([str(p) for p in pitches])
            token = f"CHORD_{pitch_str}_{duration}"
            token_list.append(token)
        # 休止符判断，不导入rest模块
        elif elem.__class__.__name__ == "Rest":
            token = f"{REST_SYMBOL}_{duration}"
            token_list.append(token)
    return token_list

# 第一步：先统计所有midi文件，方便显示进度
all_midi_files = []
all_midi_genre_ids = []
for genre_name, genre_id in GENRE_MAP.items():
    genre_dir = os.path.join(DATASET_ROOT, genre_name)
    if not os.path.exists(genre_dir):
        print(f"警告：文件夹不存在 {genre_dir}，跳过该曲风")
        continue
    for file in os.listdir(genre_dir):
        if file.endswith((".mid", ".midi")):
            full_path = os.path.join(genre_dir, file)
            all_midi_files.append(full_path)
            all_midi_genre_ids.append(genre_id)

total_file_num = len(all_midi_files)
print(f"总共待解析 MIDI 文件数量：{total_file_num}\n")

# 遍历解析，实时打印进度
for idx, (midi_path, g_id) in enumerate(zip(all_midi_files, all_midi_genre_ids)):
    current = idx + 1
    print(f"[{current}/{total_file_num}] 正在解析：{midi_path}")
    seq_tokens = extract_full_music_sequence(midi_path)
    if len(seq_tokens) > SEQ_LENGTH:
        all_token_sequences.append(seq_tokens)
        all_genre_labels.append(g_id)

# 构建词表映射
all_tokens = []
for seq in all_token_sequences:
    all_tokens.extend(seq)
unique_tokens = sorted(set(all_tokens))
vocab_size = len(unique_tokens)
token_to_idx = {t: i for i, t in enumerate(unique_tokens)}
idx_to_token = {i: t for i, t in enumerate(unique_tokens)}

# 滑动窗口生成训练集
X_seq = []
X_genre = []
y_target = []

print("\n开始构造训练滑动窗口样本...")
for seq, g_label in zip(all_token_sequences, all_genre_labels):
    seq_indexed = [token_to_idx[t] for t in seq]
    for i in range(len(seq_indexed) - SEQ_LENGTH):
        X_seq.append(seq_indexed[i:i+SEQ_LENGTH])
        X_genre.append(g_label)
        y_target.append(seq_indexed[i + SEQ_LENGTH])

# 转为数组保存
X_seq = np.array(X_seq)
X_genre = np.array(X_genre)
y_target = np.array(y_target)

np.savez_compressed(
    "train_data.npz",
    X_seq=X_seq,
    X_genre=X_genre,
    y_target=y_target,
    token_to_idx=token_to_idx,
    idx_to_token=idx_to_token,
    vocab_size=vocab_size,
    SEQ_LENGTH=SEQ_LENGTH,
    REST_SYMBOL=REST_SYMBOL
)

print("=" * 50)
print(f"预处理完成")
print(f"有效乐曲总数: {len(all_token_sequences)}")
print(f"训练样本总数: {len(X_seq)}")
print(f"词汇表总大小: {vocab_size}")
print(f"数据已保存至 train_data.npz")
print("=" * 50)