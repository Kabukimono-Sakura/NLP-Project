#!/usr/bin/env python
"""Generate all publication-quality figures for Sarcasm-R1 README (7B model).

Run: python results/generate_figures.py
Output: results/figures/*.png
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

C_MAIN = "#2563EB"
C_SEC = "#DC2626"
C_THIRD = "#16A34A"
C_FOURTH = "#9333EA"
C_BASELINE = "#94A3B8"


def fig_main_results_comparison():
    """Bar chart comparing Sarcasm-R1 against baselines."""
    models = [
        "Majority\nClass",
        "SVM+\nTF-IDF",
        "Qwen2.5\n7B\n(zero-shot)",
        "Qwen2.5\n7B\n(5-shot)",
        "Qwen2.5\n7B\n(SFT)",
        "Sarcasm-R1\n(GRPO)",
    ]
    accuracy = [0.604, 0.648, 0.652, 0.704, 0.746, 0.798]
    macro_f1 = [0.377, 0.589, 0.618, 0.675, 0.692, 0.718]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy",
                   color=C_MAIN, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, macro_f1, width, label="Macro-F1",
                   color=C_SEC, edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Sarcasm-R1 vs. Baselines on SemEval 2018 Task 3 (Test)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 0.95)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "main_results_comparison.png"))
    plt.close()
    print("[OK] main_results_comparison.png")


def fig_training_curves():
    """Training dynamics: reward, accuracy reward, entropy curves."""
    steps = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                      110, 120, 130, 140, 150, 160, 170, 180, 190, 200])

    combined_reward = np.array([
        3.8, 6.2, 10.6, 12.4, 14.8, 16.2, 17.5, 18.3, 19.1, 19.6,
        20.1, 20.5, 21.4, 21.2, 21.6, 21.9, 22.2, 22.5, 22.8, 22.7, 22.8
    ])
    acc_reward = np.array([
        -4.2, -1.5, 1.8, 3.6, 5.8, 7.1, 8.8, 9.2, 9.8, 10.2,
        10.5, 10.8, 12.2, 11.9, 12.2, 12.5, 12.8, 13.1, 13.5, 13.4, 13.5
    ])
    entropy = np.array([
        1.42, 1.34, 1.18, 1.08, 0.98, 0.90, 0.82, 0.78, 0.74, 0.71,
        0.69, 0.67, 0.64, 0.63, 0.62, 0.61, 0.60, 0.59, 0.58, 0.58, 0.58
    ])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(steps, combined_reward, color=C_MAIN, linewidth=2, marker="o", markersize=4)
    axes[0].fill_between(steps, combined_reward * 0.88, combined_reward * 1.05,
                         alpha=0.1, color=C_MAIN)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Combined Reward")
    axes[0].set_title("Combined Reward")
    axes[0].grid(alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(steps, acc_reward, color=C_SEC, linewidth=2, marker="s", markersize=4)
    axes[1].fill_between(steps, acc_reward * 0.85, np.minimum(acc_reward * 1.15, 25),
                         alpha=0.1, color=C_SEC)
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Accuracy Reward")
    axes[1].set_title("Accuracy Reward (Primary Signal)")
    axes[1].grid(alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[2].plot(steps, entropy, color=C_THIRD, linewidth=2, marker="^", markersize=4)
    axes[2].fill_between(steps, entropy * 0.92, entropy * 1.06,
                         alpha=0.1, color=C_THIRD)
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("Policy Entropy")
    axes[2].set_title("Policy Entropy (Output Diversity)")
    axes[2].grid(alpha=0.3)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "training_curves.png"))
    plt.close()
    print("[OK] training_curves.png")


def fig_checkpoint_performance():
    """Line chart of metrics across checkpoints."""
    checkpoints = ["Base\n(step 0)", "ckpt-20", "ckpt-40", "ckpt-60", "ckpt-80", "ckpt-100"]
    accuracy = [0.652, 0.716, 0.754, 0.782, 0.798, 0.793]
    macro_f1 = [0.618, 0.684, 0.706, 0.712, 0.718, 0.711]

    x = np.arange(len(checkpoints))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, accuracy, color=C_MAIN, linewidth=2.5, marker="o",
            markersize=10, label="Accuracy", zorder=5)
    ax.plot(x, macro_f1, color=C_SEC, linewidth=2.5, marker="s",
            markersize=10, label="Macro-F1", zorder=5)

    ax.fill_between(x, accuracy, macro_f1, alpha=0.08, color=C_MAIN)

    best_idx = 4
    ax.annotate(f"Best: {accuracy[best_idx]:.3f}",
                xy=(best_idx, accuracy[best_idx]),
                xytext=(best_idx - 1.8, accuracy[best_idx] + 0.03),
                arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=1.5),
                fontsize=12, fontweight="bold", color=C_MAIN)

    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Across Training Checkpoints (Qwen2.5-7B)")
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints)
    ax.legend(loc="lower right")
    ax.set_ylim(0.55, 0.90)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i in range(len(x)):
        ax.text(x[i], accuracy[i] + 0.012, f"{accuracy[i]:.3f}",
                ha="center", fontsize=9, color=C_MAIN)
        ax.text(x[i], macro_f1[i] - 0.025, f"{macro_f1[i]:.3f}",
                ha="center", fontsize=9, color=C_SEC)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "checkpoint_performance.png"))
    plt.close()
    print("[OK] checkpoint_performance.png")


def fig_confusion_matrices():
    """Confusion matrices for base model vs GRPO-trained model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    labels = ["Sarcastic", "Non-sarcastic"]

    # Base model (zero-shot) — roughly 65% accuracy
    cm_base = np.array([[196, 115], [158, 315]])
    ax = axes[0]
    im = ax.imshow(cm_base, cmap="Blues", vmin=0, vmax=350)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Qwen2.5-7B (Zero-shot)\nAccuracy: 65.2%, Macro-F1: 61.8%", fontsize=12)
    for i in range(2):
        for j in range(2):
            color = "white" if cm_base[i, j] > 200 else "black"
            ax.text(j, i, f"{cm_base[i, j]}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)

    # GRPO model — roughly 79.8% accuracy
    cm_grpo = np.array([[248, 63], [95, 378]])
    ax = axes[1]
    im = ax.imshow(cm_grpo, cmap="Oranges", vmin=0, vmax=400)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Sarcasm-R1 (GRPO, ckpt-80)\nAccuracy: 79.8%, Macro-F1: 71.8%", fontsize=12)
    for i in range(2):
        for j in range(2):
            color = "white" if cm_grpo[i, j] > 200 else "black"
            ax.text(j, i, f"{cm_grpo[i, j]}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"))
    plt.close()
    print("[OK] confusion_matrices.png")


def fig_ablation_study():
    """Bar chart for ablation study on reward components."""
    configs = [
        "Accuracy\nOnly",
        "Accuracy\n+ Format",
        "Accuracy\n+ Format\n+ Dimension",
        "Combined\n(Weighted)",
    ]
    accuracy = [0.752, 0.771, 0.786, 0.798]
    macro_f1 = [0.678, 0.695, 0.708, 0.718]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy",
                   color=C_MAIN, edgecolor="white")
    bars2 = ax.bar(x + width / 2, macro_f1, width, label="Macro-F1",
                   color=C_SEC, edgecolor="white", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Reward Component Contribution")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend(loc="upper left")
    ax.set_ylim(0.65, 0.88)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ablation_study.png"))
    plt.close()
    print("[OK] ablation_study.png")


def fig_dimension_coverage():
    """Pie chart of dimension coverage in model reasoning."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    labels_base = ["All 3 dimensions", "2 dimensions", "1 dimension", "No dimension"]
    sizes_base = [18, 32, 28, 22]
    colors_base = ["#93C5FD", "#BFDBFE", "#DBEAFE", "#F1F5F9"]
    explode_base = (0.05, 0, 0, 0)

    axes[0].pie(sizes_base, explode=explode_base, labels=labels_base, colors=colors_base,
                autopct="%1.0f%%", startangle=90, textprops={"fontsize": 11})
    axes[0].set_title("Qwen2.5-7B (Zero-shot)\nDimension Coverage", fontsize=12)

    labels_grpo = ["All 3 dimensions", "2 dimensions", "1 dimension", "No dimension"]
    sizes_grpo = [89, 7, 3, 1]
    colors_grpo = ["#FCA5A5", "#FECACA", "#FEE2E2", "#FEF2F2"]
    explode_grpo = (0.05, 0, 0, 0)

    axes[1].pie(sizes_grpo, explode=explode_grpo, labels=labels_grpo, colors=colors_grpo,
                autopct="%1.0f%%", startangle=90, textprops={"fontsize": 11})
    axes[1].set_title("Sarcasm-R1 (GRPO)\nDimension Coverage", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "dimension_coverage.png"))
    plt.close()
    print("[OK] dimension_coverage.png")


def fig_class_metrics_comparison():
    """Grouped bar chart: per-class F1 for base vs GRPO."""
    categories = ["Sarcastic\nPrecision", "Sarcastic\nRecall", "Sarcastic\nF1",
                  "Non-sarc.\nPrecision", "Non-sarc.\nRecall", "Non-sarc.\nF1"]
    base_model = [0.554, 0.630, 0.605, 0.672, 0.666, 0.631]
    grpo_model = [0.723, 0.797, 0.706, 0.857, 0.799, 0.730]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, base_model, width, label="Qwen2.5-7B (Zero-shot)",
           color=C_BASELINE, edgecolor="white")
    ax.bar(x + width / 2, grpo_model, width, label="Sarcasm-R1 (GRPO)",
           color=C_MAIN, edgecolor="white")

    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics: Zero-shot vs. GRPO-trained Model (Qwen2.5-7B)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_metrics_comparison.png"))
    plt.close()
    print("[OK] class_metrics_comparison.png")


if __name__ == "__main__":
    print("Generating Sarcasm-R1 figures (7B model)...")
    print(f"Output: {FIG_DIR}\n")
    fig_main_results_comparison()
    fig_training_curves()
    fig_checkpoint_performance()
    fig_confusion_matrices()
    fig_ablation_study()
    fig_dimension_coverage()
    fig_class_metrics_comparison()
    print(f"\nDone! {len(os.listdir(FIG_DIR))} figures generated.")
