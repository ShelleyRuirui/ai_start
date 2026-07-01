"""
Generate an architecture diagram for the Music Composer LSTM model.
Clean layout with no overlapping arrows/boxes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

# ---------- Chinese font ----------
chinese_font = None
for f in fm.fontManager.ttflist:
    if "PingFang" in f.name or "STHeiti" in f.name or "Songti" in f.name:
        chinese_font = f.name
        break
if chinese_font:
    plt.rcParams["font.family"] = chinese_font
plt.rcParams["font.size"] = 11

# ---------- dimensions ----------
FIG_W = 12
FIG_H = 10

# Colors
C_NOTE   = "#4A90D9"   # blue
C_EMBED  = "#50C878"   # green
C_LSTM   = "#FF8C42"   # orange
C_GENRE  = "#9B59B6"   # purple
C_DENSE  = "#E74C3C"   # red
C_OUT    = "#2C3E50"   # dark
C_CONCAT = "#F1C40F"   # yellow
C_BG     = "#F8F9FA"

def box(ax, x, y, w, h, text, color, alpha=0.9, fs=10, tc="white"):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.08",
                       facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=tc)

def arrow(ax, x1, y1, x2, y2, c="#555", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=lw))

# ---------- figure ----------
fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_facecolor(C_BG)
ax.axis("off")

ax.text(FIG_W/2, FIG_H-0.2, "Music Composer - 条件LSTM模型架构图",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#2C3E50")

# === Layout ===
# Left column: note path (x=3.0)
# Right column: genre path (x=9.0)
# Center: concat + output (x=6.0)
LX, RX, CX = 3.0, 9.0, 6.0
BW = 3.5  # box width
BH = 0.55 # box height
STEP = 0.95

# ---- Left: Note Sequence (top to bottom) ----
y = 8.2
box(ax, LX, y, BW, BH, "输入1: 音符序列 (64个token)", C_NOTE)
y1 = y
y -= STEP
box(ax, LX, y, BW, BH, "Embedding (vocab_size -> 64维)", C_EMBED)
y2 = y
y -= STEP
box(ax, LX, y, BW, BH, "LSTM (64, return_sequences=True)", C_LSTM)
y3 = y
y -= STEP
box(ax, LX, y, BW, BH, "LSTM (64)", C_LSTM)
y4 = y
y -= STEP
box(ax, LX, y, BW, BH, "Dropout (0.1)", C_LSTM, alpha=0.7)
y_left_bottom = y

# Left arrows
arrow(ax, LX, y1-BH/2, LX, y2+BH/2)
arrow(ax, LX, y2-BH/2, LX, y3+BH/2)
arrow(ax, LX, y3-BH/2, LX, y4+BH/2)
arrow(ax, LX, y4-BH/2, LX, y_left_bottom+BH/2)

# ---- Right: Genre Path (top to bottom) ----
y = 8.2
box(ax, RX, y, BW, BH, "输入2: 风格标签 (0=Jazz / 1=Blues)", C_GENRE)
ry1 = y
y -= STEP
box(ax, RX, y, BW, BH, "Embedding (3 -> 64维)", C_EMBED)
ry2 = y
y -= STEP
box(ax, RX, y, BW, BH, "Flatten", C_DENSE, alpha=0.7)
ry3 = y
y -= STEP
box(ax, RX, y, BW, BH, "Dense (128, tanh)", C_DENSE)
ry4 = y

# Right arrows
arrow(ax, RX, ry1-BH/2, RX, ry2+BH/2)
arrow(ax, RX, ry2-BH/2, RX, ry3+BH/2)
arrow(ax, RX, ry3-BH/2, RX, ry4+BH/2)

# ---- Concat (center, below both branches) ----
# Place concat below the lowest of the two branches
concat_y = min(y_left_bottom, ry4) - STEP * 0.6

# Arrows from left and right to concat
# Left: from Dropout bottom to concat left side
arrow(ax, LX, y_left_bottom-BH/2, CX-1.0, concat_y+BH/2)
# Right: from Dense bottom to concat right side
arrow(ax, RX, ry4-BH/2, CX+1.0, concat_y+BH/2)

box(ax, CX, concat_y, 5.0, BH, "Concatenate (192维)", C_CONCAT, tc="#2C3E50")

# ---- Dense + Output (center, below concat) ----
y = concat_y - STEP
box(ax, CX, y, 5.0, BH, "Dense (128, tanh)", C_DENSE)
arrow(ax, CX, concat_y-BH/2, CX, y+BH/2)

y -= STEP
box(ax, CX, y, 5.0, BH, "Dense (vocab_size, softmax)", C_OUT)
arrow(ax, CX, y+STEP-BH/2, CX, y+BH/2)

# Output label
y -= 0.8
box(ax, CX, y, 3.5, 0.5, "下一个音符 (预测)", "#E67E22", fs=12)
arrow(ax, CX, y+0.8-BH/2, CX, y+BH/2)

# ---- Autoregressive loop (left side) ----
loop_x = 0.8
loop_y = y + 0.5
ax.annotate("", xy=(LX-BW/2-0.1, loop_y), xytext=(loop_x, loop_y),
            arrowprops=dict(arrowstyle="->", color="#E67E22", lw=2, linestyle="dashed"))
ax.text(loop_x-0.2, loop_y+0.4, "自回归循环\n(滑动窗口)", fontsize=8,
        color="#E67E22", ha="center", va="bottom", fontweight="bold")

# ---- Legend ----
leg_y = 0.3
items = [
    ("输入层", C_NOTE), ("Embedding", C_EMBED), ("LSTM", C_LSTM),
    ("风格编码", C_GENRE), ("全连接层", C_DENSE), ("拼接", C_CONCAT), ("输出层", C_OUT),
]
for i, (label, color) in enumerate(items):
    xp = 1.0 + i * 1.4
    r = FancyBboxPatch((xp-0.45, leg_y-0.15), 0.9, 0.3,
                        boxstyle="round,pad=0.05", facecolor=color,
                        edgecolor="none", alpha=0.85)
    ax.add_patch(r)
    ax.text(xp, leg_y, label, ha="center", va="center", fontsize=8,
            fontweight="bold", color="white" if color != C_CONCAT else "#2C3E50")

# ---------- Save ----------
out = "architecture_diagram.png"
plt.tight_layout()
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_BG)
print(f"Saved to {out}")
plt.close()
