"""
All thesis figures. Run:
    python thesis_charts.py

Output: /workspace/figures/fig3_1_baselines.png
        /workspace/figures/fig3_4_debate_vs_thoughtcomm.png
        /workspace/figures/fig3_5_conditional_routing.png
        /workspace/figures/fig3_6_all_experiments.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/workspace/figures")
OUT.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3.1 — Baselines and Oracle Ceiling
# ══════════════════════════════════════════════════════════════════════════

def fig3_1():
    labels = [
        'Vision-only\n(Qwen)',
        'Vision-only\n(MedGemma)',
        'Single-agent\n(MedGemma)',
        'Single-agent\n(Qwen)',
        'Text-only\n(MedGemma)',
        'Text-only\n(Qwen)',
        'Oracle\nCeiling',
    ]
    values = [30.33, 33.24, 47.02, 48.48, 51.52, 52.69, 65.46]
    colors = ['#E74C3C', '#E74C3C',   # vision (red)
              '#F39C12', '#F39C12',   # single-agent (amber)
              '#5B9BD5', '#5B9BD5',   # text-only (blue)
              '#27AE60']              # oracle (green)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='white', linewidth=0.8, height=0.65)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', ha='left', fontsize=11, fontweight='bold')

    # Reference lines
    ax.axvline(x=51.52, color='#5B9BD5', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=65.46, color='#27AE60', linestyle='--', linewidth=1, alpha=0.5)

    # Annotate the gap
    ax.annotate('', xy=(65.46, 6.6), xytext=(51.52, 6.6),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5))
    ax.text(58.5, 6.85, '~14pp\narbitration gap', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333333')

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Accuracies and Oracle Ceiling (689 cases)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlim(0, 75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, alpha=0.2)
    ax.invert_yaxis()

    plt.tight_layout()
    path = OUT / "fig3_1_baselines.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / "fig3_1_baselines.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3.4 — Debate vs ThoughtComm (FIXED, now includes Mode 3b)
# ══════════════════════════════════════════════════════════════════════════

def fig3_4():
    # Data: 339 test cases (Mode 3b is 338 due to 1 skip)
    groups = ['Mode 2\nCoT Debate\n(339 cases)', 'Mode 3b\nStructured Debate\n(338 cases)', 'ThoughtComm\nLatent Comm.\n(339 cases)']
    r1 = [47.5, 50.3, 34.8]
    r2 = [40.4, 40.5, 44.0]
    deltas = [-7.1, -9.8, +9.2]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(groups))
    width = 0.28

    # R1 bars (all blue — independent baseline)
    bars_r1 = ax.bar(x - width/2, r1, width, label='Round 1 (independent)',
                      color='#5B9BD5', edgecolor='white', linewidth=0.8)

    # R2 bars (colour-coded by direction)
    r2_colors = ['#E74C3C', '#E74C3C', '#27AE60']
    bars_r2 = ax.bar(x + width/2, r2, width, label='Round 2 (after communication)',
                      color=r2_colors, edgecolor='white', linewidth=0.8)

    # Value labels on bars (placed inside bars to avoid overlap)
    for bar in bars_r1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h - 2.5,
                f'{h:.1f}%', ha='center', va='top', fontsize=10,
                fontweight='bold', color='white')

    for bar in bars_r2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h - 2.5,
                f'{h:.1f}%', ha='center', va='top', fontsize=10,
                fontweight='bold', color='white')

    # Delta annotations (above the higher bar of each pair, well clear)
    for i, delta in enumerate(deltas):
        peak = max(r1[i], r2[i])
        color = '#E74C3C' if delta < 0 else '#27AE60'
        sign = '+' if delta > 0 else ''
        ax.text(x[i], peak + 4, f'{sign}{delta:.1f}pp',
                ha='center', va='bottom', fontsize=13, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.12,
                          edgecolor=color, linewidth=1.2))

    # Text-only baseline
    ax.axhline(y=51.5, color='#7F8C8D', linestyle='--', linewidth=1, alpha=0.6)
    ax.text(2.45, 52.3, 'Text-only baseline (51.5%)', fontsize=9, color='#7F8C8D', ha='right')

    # Random baseline
    ax.axhline(y=20, color='#BDC3C7', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(2.45, 20.8, 'Random (20%)', fontsize=8, color='#BDC3C7', ha='right')

    ax.set_ylabel('Text Agent Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Text Agent Response to Inter-Agent Communication\n(Language-Level Debate vs Thought-Level Transfer)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylim(0, 62)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.2)

    # Bottom subtitle
    fig.text(0.5, -0.01,
             'Both debate modes degrade the strong text agent. ThoughtComm improves it.\n'
             'The communication channel — not the revision step — determines the direction.',
             ha='center', fontsize=10, style='italic', color='#555555')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    path = OUT / "fig3_4_debate_vs_thoughtcomm.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / "fig3_4_debate_vs_thoughtcomm.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3.5 — Conditional Debate Routing Breakdown
# ══════════════════════════════════════════════════════════════════════════

def fig3_5():
    labels = ['Direct decisions\n(no debate)', 'Debated decisions', 'Overall\n(conditional debate)']
    accuracies = [55.74, 47.33, 50.94]
    counts = [296, 393, 689]
    colors = ['#27AE60', '#E67E22', '#5B9BD5']

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(range(len(labels)), accuracies, color=colors,
                  edgecolor='white', linewidth=0.8, width=0.55)

    # Value + count labels
    for bar, acc, count in zip(bars, accuracies, counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f'{acc:.1f}%', ha='center', va='bottom',
                fontsize=14, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2, h/2,
                f'n={count}', ha='center', va='center',
                fontsize=11, color='white', fontweight='bold')

    # Reference lines
    ax.axhline(y=51.52, color='#5B9BD5', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.text(2.35, 52.2, 'Text-only baseline (51.5%)', fontsize=9, color='#5B9BD5', ha='right')

    ax.axhline(y=47.02, color='#F39C12', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(2.35, 47.7, 'Single-agent baseline (47.0%)', fontsize=9, color='#F39C12', ha='right')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Conditional Debate: Routing Breakdown (689 cases)\n'
                 'Direct decisions exceed text-only baseline; debated cases are genuinely harder',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, 65)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.2)

    # Annotation explaining the key insight
    fig.text(0.5, -0.01,
             'By skipping debate on easy cases (55.7%), the system avoids dilution.\n'
             'Debate runs only on hard cases (47.3%), where visual evidence may genuinely help.',
             ha='center', fontsize=10, style='italic', color='#555555')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    path = OUT / "fig3_5_conditional_routing.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / "fig3_5_conditional_routing.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3.6 — All Experiments Ranked
# ══════════════════════════════════════════════════════════════════════════

def fig3_6():
    # All experiments sorted by accuracy (ascending for horizontal bars)
    experiments = [
        ('RAG text agent',               37.01, 'rag'),
        ('Mode 1 — CoT',                 40.93, 'comm'),
        ('Mode 3 — Structured JSON',     44.90, 'comm'),
        ('Directed debate (LG)',         45.57, 'lg'),
        ('Mode 2 — CoT debate',         46.00, 'comm'),
        ('Mode 3b — Structured debate',  46.70, 'comm'),
        ('Single-agent full',            47.02, 'base'),
        ('Conservative conditional (LG)',49.35, 'lg'),
        ('Describe-then-fuse (LG)',      49.49, 'lg'),
        ('Option ranking (LG)',          50.80, 'lg'),
        ('Conditional debate (LG)',      50.94, 'lg'),
        ('Text-only baseline',           51.52, 'base'),
    ]

    labels = [e[0] for e in experiments]
    values = [e[1] for e in experiments]
    types = [e[2] for e in experiments]

    color_map = {
        'base': '#5B9BD5',  # baselines — blue
        'comm': '#9B59B6',  # communication modes — purple
        'lg':   '#E67E22',  # LangGraph architectures — orange
        'rag':  '#E74C3C',  # RAG — red
    }
    colors = [color_map[t] for t in types]

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.8, height=0.65)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', ha='left', fontsize=10, fontweight='bold')

    # Oracle ceiling line
    ax.axvline(x=65.46, color='#27AE60', linestyle='-', linewidth=2, alpha=0.7)
    ax.text(65.8, len(labels) - 0.5, 'Oracle ceiling\n65.5%', fontsize=9,
            color='#27AE60', fontweight='bold', va='top')

    # Text-only baseline line
    ax.axvline(x=51.52, color='#5B9BD5', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.text(51.8, -0.3, 'Text-only\n51.5%', fontsize=8, color='#5B9BD5', va='top')

    # Single-agent baseline line
    ax.axvline(x=47.02, color='#F39C12', linestyle='--', linewidth=1, alpha=0.4)
    ax.text(44, -0.3, 'Single-agent\n47.0%', fontsize=8, color='#F39C12', va='top')

    # Shade the arbitration gap region
    ax.axvspan(51.52, 65.46, alpha=0.05, color='#27AE60')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('All Experiments Ranked by System Accuracy (689 cases)\n'
                 'Shaded region: arbitration gap between text-only baseline and oracle ceiling',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlim(0, 73)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, alpha=0.2)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#5B9BD5', label='Baselines'),
        Patch(facecolor='#9B59B6', label='Communication modes'),
        Patch(facecolor='#E67E22', label='LangGraph architectures'),
        Patch(facecolor='#E74C3C', label='RAG'),
        Patch(facecolor='#27AE60', alpha=0.3, label='Oracle ceiling'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    path = OUT / "fig3_6_all_experiments.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / "fig3_6_all_experiments.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {path}")




if __name__ == "__main__":
    print("Generating thesis figures...\n")
    fig3_1()
    fig3_4()
    fig3_5()
    fig3_6()
    print(f"\nAll figures saved to {OUT}/")
    print("PNG (300dpi) for Word/PowerPoint, PDF for LaTeX/high-quality print.")