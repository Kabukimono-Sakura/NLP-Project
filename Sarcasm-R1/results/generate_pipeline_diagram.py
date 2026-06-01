#!/usr/bin/env python
"""Generate Sarcasm-R1 pipeline architecture diagram for PPT presentation.

Run: python results/generate_pipeline_diagram.py
Output: results/figures/pipeline_diagram.png
"""

from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Color scheme - professional, cohesive
C_INPUT = "#E8F0FE"      # Light blue - data
C_INPUT_BORDER = "#4285F4"
C_PROCESS = "#FFF3E0"    # Light orange - processing
C_PROCESS_BORDER = "#FF8C00"
C_MODEL = "#E8F5E9"      # Light green - model
C_MODEL_BORDER = "#2E7D32"
C_REWARD = "#F3E5F5"     # Light purple - reward
C_REWARD_BORDER = "#7B1FA2"
C_REASON = "#FFF8E1"     # Light yellow - reasoning
C_REASON_BORDER = "#F9A825"
C_OUTPUT = "#FFEBEE"     # Light red - output
C_OUTPUT_BORDER = "#C62828"
C_ARROW = "#455A64"
C_TEXT = "#212121"
C_SUBTEXT = "#616161"

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")


def draw_box(x, y, w, h, label, sublabel="", color=C_INPUT, border=C_INPUT_BORDER,
             fontsize=11, sublabel_size=9, bold=True):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor=border,
        linewidth=2.0, zorder=3
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.18, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=weight, color=C_TEXT, zorder=5)
        ax.text(x + w / 2, y + h / 2 - 0.22, sublabel,
                ha="center", va="center", fontsize=sublabel_size,
                color=C_SUBTEXT, zorder=5, style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=weight, color=C_TEXT, zorder=5)


def draw_arrow(x1, y1, x2, y2, label="", style="-|>", color=C_ARROW, lw=2.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=2)


def draw_curved_arrow(x1, y1, x2, y2, color=C_ARROW, lw=2.0, connectionstyle="arc3,rad=0.3"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=connectionstyle),
                zorder=2)


# =====================================================================
# Title
# =====================================================================
ax.text(8, 9.6, "Sarcasm-R1: GRPO + Three-Dimensional Reasoning Pipeline",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#1A237E")

# =====================================================================
# Row 1: Data Input (left side)
# =====================================================================
draw_box(0.3, 8.0, 2.8, 1.1, "SemEval 2018 Task 3", "2,862 tweets",
         C_INPUT, C_INPUT_BORDER)
draw_box(0.3, 6.5, 2.8, 1.1, "MUStARD", "690 dialogues",
         C_INPUT, C_INPUT_BORDER)

# Arrow: data -> processing
draw_arrow(3.1, 8.55, 4.0, 8.55)
draw_arrow(3.1, 7.05, 4.0, 7.65)

# =====================================================================
# Data Processing
# =====================================================================
draw_box(4.0, 7.2, 2.6, 1.6, "Data Processing", "Standardized CSV",
         C_PROCESS, C_PROCESS_BORDER, fontsize=12)

# Arrow: processing -> combined
draw_arrow(6.6, 8.0, 7.4, 8.0)

# =====================================================================
# Combined Training Data
# =====================================================================
draw_box(7.4, 7.2, 2.4, 1.6, "Training Data", "3,552 samples",
         C_INPUT, C_INPUT_BORDER, fontsize=12)

# =====================================================================
# Row 2: GRPO Training (center)
# =====================================================================
draw_arrow(8.6, 7.2, 8.6, 6.15)

draw_box(5.8, 4.6, 5.6, 1.5, "GRPO Training", "Qwen2.5-7B-Instruct  |  200 steps  |  8 generations/prompt",
         C_MODEL, C_MODEL_BORDER, fontsize=13, sublabel_size=10)

# =====================================================================
# Reward Functions (left of GRPO)
# =====================================================================
draw_arrow(5.8, 5.35, 4.8, 5.75)

# Reward box with 3 sub-components
reward_y = 3.8
draw_box(0.3, reward_y + 1.5, 4.4, 2.4, "", "",
         C_REWARD, C_REWARD_BORDER, fontsize=10)
ax.text(2.5, reward_y + 3.55, "Multi-Component Reward",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color=C_REWARD_BORDER, zorder=5)

# Sub-rewards inside
reward_items = [
    ("Accuracy Reward", "+25 / -10", 0),
    ("Format Reward", "Structure bonus", 1),
    ("Dimension Coverage", "+2/dim +3 bonus", 2),
]
for i, (name, desc, idx) in enumerate(reward_items):
    ry = reward_y + 3.15 - idx * 0.75
    ax.text(1.0, ry, f"  {name}", ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=C_TEXT, zorder=5)
    ax.text(4.4, ry, desc, ha="right", va="center",
            fontsize=8.5, color=C_SUBTEXT, zorder=5)

# Arrow: reward -> GRPO (feedback loop)
draw_curved_arrow(4.7, reward_y + 2.7, 5.8, 5.1,
                  color="#7B1FA2", lw=2.5, connectionstyle="arc3,rad=-0.2")
ax.text(4.9, 4.2, "reward\nsignal", ha="center", va="center",
        fontsize=8, color="#7B1FA2", style="italic")

# =====================================================================
# Three-Dimensional Reasoning (right of GRPO)
# =====================================================================
reason_x = 11.8
reason_y = 3.8
draw_box(reason_x, reason_y + 1.5, 3.8, 2.4, "", "",
         C_REASON, C_REASON_BORDER)
ax.text(reason_x + 1.9, reason_y + 3.55, "Three-Dimensional Reasoning",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color=C_REASON_BORDER, zorder=5)

dims = [
    ("Rhetoric", "Irony, Hyperbole, Contrast"),
    ("Context", "Situational, Knowledge mismatch"),
    ("Emotion", "Sentiment gap, Emotional reversal"),
]
for i, (name, desc) in enumerate(dims):
    ry = reason_y + 3.15 - i * 0.75
    ax.text(reason_x + 0.4, ry, f"  {name}:", ha="left", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT, zorder=5)
    ax.text(reason_x + 1.7, ry - 0.02, desc, ha="left", va="center",
            fontsize=7.8, color=C_SUBTEXT, zorder=5)

# Arrow: reasoning -> GRPO
draw_arrow(reason_x, 5.35, 11.4, 5.35)

# =====================================================================
# Row 3: Output
# =====================================================================
draw_arrow(8.6, 4.6, 8.6, 3.4)

draw_box(5.8, 2.0, 5.6, 1.3, "Structured Reasoning Output", "{reasoning} {sarcastic / non_sarcastic}",
         C_OUTPUT, C_OUTPUT_BORDER, fontsize=12, sublabel_size=9)

# =====================================================================
# Row 4: Evaluation
# =====================================================================
draw_arrow(8.6, 2.0, 8.6, 1.2)

draw_box(6.3, 0.2, 4.6, 0.9, "Evaluation: Accuracy 79.8%  |  Macro-F1 71.8%",
         "", C_OUTPUT, "#1565C0", fontsize=12, sublabel_size=9)

# =====================================================================
# Legend / bottom note
# =====================================================================
ax.text(0.3, 0.15, "Data", ha="left", va="center", fontsize=9,
        color=C_INPUT_BORDER, fontweight="bold")
ax.text(1.3, 0.15, "Process", ha="left", va="center", fontsize=9,
        color=C_PROCESS_BORDER, fontweight="bold")
ax.text(2.6, 0.15, "Model", ha="left", va="center", fontsize=9,
        color=C_MODEL_BORDER, fontweight="bold")
ax.text(3.6, 0.15, "Reward", ha="left", va="center", fontsize=9,
        color=C_REWARD_BORDER, fontweight="bold")
ax.text(4.7, 0.15, "Reasoning", ha="left", va="center", fontsize=9,
        color=C_REASON_BORDER, fontweight="bold")
ax.text(6.0, 0.15, "Result", ha="left", va="center", fontsize=9,
        color=C_OUTPUT_BORDER, fontweight="bold")

plt.tight_layout(pad=0.5)
output_path = os.path.join(FIG_DIR, "pipeline_diagram.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"[OK] {output_path}")
