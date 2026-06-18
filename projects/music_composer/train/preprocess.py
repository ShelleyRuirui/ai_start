import warnings
import os
import numpy as np
from collections import Counter
from music21 import converter, note, chord

# ===================== 全局配置 =====================
# 屏蔽全部无关网络警告
warnings.filterwarnings("ignore")

GENRE_MAP = {"jazz": 0, "blues": 1}
SEQ_LENGTH = 64
DATASET_ROOT = "../dataset"
REST_SYMBOL = "REST"
# 词表压缩配置：保留出现次数 >=3 的token，可自行调整
MIN_TOKEN_COUNT = 15
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

# ========== 新增：统计token频次、过滤低频，缩小vocab ==========
# 收集全部token
all_tokens_flat = []
for seq in all_token_sequences:
    all_tokens_flat.extend(seq)

# 统计每个token出现次数
token_counter = Counter(all_tokens_flat)
# 只保留出现次数 >= MIN_TOKEN_COUNT 的高频token
valid_tokens = [tok for tok, cnt in token_counter.items() if cnt >= MIN_TOKEN_COUNT]

# 构建新映射，增加未知标记OOV（低频token统一映射到0）
token_to_idx = {"<OOV>": 0}
for idx, tok in enumerate(valid_tokens, start=1):
    token_to_idx[tok] = idx
idx_to_token = {v: k for k, v in token_to_idx.items()}
vocab_size = len(token_to_idx)

print(f"\n词表压缩完成：")
print(f"原始全部唯一token数：{len(token_counter)}")
print(f"过滤后有效高频token数：{vocab_size}")
print(f"出现少于{MIN_TOKEN_COUNT}次的音符/和弦统一归为<OOV>\n")
# ==============================================================

# 滑动窗口生成训练集
X_seq = []
X_genre = []
y_target = []

print("开始构造训练滑动窗口样本...")
for seq, g_label in zip(all_token_sequences, all_genre_labels):
    # 低频token自动替换为<OOV>编号0
    seq_indexed = [token_to_idx.get(tok, 0) for tok in seq]
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
print(f"词汇表压缩后大小: {vocab_size}")
print(f"数据已保存至 train_data.npz")
print("=" * 50)