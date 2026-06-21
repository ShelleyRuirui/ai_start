# 屏蔽无关警告
import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
from tensorflow.keras.models import load_model
from music21 import note, chord, stream

# ===================== 配置区 =====================
MODEL_PATH = "../train/checkpoints/model_epoch_07_loss_4.38.h5"
DATA_NPZ = "../train/train_data.npz"
SEQ_LENGTH = 64
GEN_TOTAL_TOKENS = 600
TEMPERATURE = 0.85
# ==================================================

# 处理分数时值工具函数
def parse_duration(dur_str):
    if "/" in dur_str:
        a, b = dur_str.split("/")
        return float(a) / float(b)
    return float(dur_str)

# 加载词表映射
data = np.load(DATA_NPZ, allow_pickle=True)
token_to_idx = data["token_to_idx"].item()
idx_to_token = data["idx_to_token"].item()
vocab_size = int(data["vocab_size"])
X_all_seq = data["X_seq"]
del data

# 加载训练模型
model = load_model(MODEL_PATH)
print("✅ 模型加载完成：model_epoch_07_loss_4.38.h5")

# 温度采样函数
def temp_sample(logits, temp):
    logits = logits / temp
    exp_log = np.exp(logits)
    prob = exp_log / np.sum(exp_log)
    return np.random.choice(len(prob), p=prob)

# 自回归生成token序列
def generate_tokens(seed_window, genre_id, gen_len):
    cur_seq = seed_window.copy()
    genre_input = np.array([genre_id])
    for _ in range(gen_len):
        input_win = np.array([cur_seq[-SEQ_LENGTH:]])
        pred = model.predict([input_win, genre_input], verbose=0)
        next_tok = temp_sample(pred[0], TEMPERATURE)
        cur_seq.append(next_tok)
    return cur_seq

# token转midi：OOV替换音符，避免堆积空白休止
def tokens_to_midi(token_id_list, save_name):
    midi_stream = stream.Stream()
    for tid in token_id_list:
        tok = idx_to_token.get(tid, "<OOV>")
        parts = tok.split("_")
        if tok.startswith("NOTE"):
            pitch = int(parts[1])
            dur = parse_duration(parts[2])
            n = note.Note(pitch)
            n.quarterLength = dur
            midi_stream.append(n)
        elif tok.startswith("CHORD"):
            dur_str = parts[-1]
            dur = parse_duration(dur_str)
            pitch_list = [int(p) for p in parts[1:-1]]
            c = chord.Chord(pitch_list)
            c.quarterLength = dur
            midi_stream.append(c)
        elif tok.startswith("REST"):
            dur = parse_duration(parts[1])
            r = note.Rest()
            r.quarterLength = dur
            midi_stream.append(r)
        else:
            # OOV不用休止，用中音C填充，消除大片静音
            n = note.Note(60)
            n.quarterLength = 0.25
            midi_stream.append(n)
    midi_stream.write("midi", fp=save_name)
    print(f"🎵 已输出MIDI文件: {save_name}")

# ========== 方案一核心：取片段末尾64个，抛弃前面空白前奏 ==========
# Jazz种子
raw_jazz = X_all_seq[random.randint(0, len(X_all_seq)-1)].tolist()
seed_jazz = raw_jazz[-SEQ_LENGTH:]   # 只保留最后64位，切掉开头空白

# Blues种子（另一条完全不同的乐曲片段）
raw_blues = X_all_seq[random.randint(0, len(X_all_seq)-1)].tolist()
seed_blues = raw_blues[-SEQ_LENGTH:]

# 生成Jazz
print("\n===== 正在生成 JAZZ 乐曲 =====")
jazz_tokens = generate_tokens(seed_jazz, genre_id=0, gen_len=GEN_TOTAL_TOKENS)
tokens_to_midi(jazz_tokens, "jazz_generated.mid")

# 生成Blues
print("\n===== 正在生成 BLUES 乐曲 =====")
blues_tokens = generate_tokens(seed_blues, genre_id=1, gen_len=GEN_TOTAL_TOKENS)
tokens_to_midi(blues_tokens, "blues_generated.mid")

print("\n🎉 两首乐曲全部生成完成！")
print("输出文件：jazz_generated.mid 、 blues_generated.mid")