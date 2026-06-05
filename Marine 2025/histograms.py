"""
Generate publication-quality class distribution bar charts.
Run from the Marine 2025 directory:
    python publication_graphs.py
"""

import matplotlib.pyplot as plt
import matplotlib
from Various.configurationFile import VISUALIZATIONS_PATH

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'figure.dpi': 600,
})

DATASET_SIZE = 504


def make_bar_chart(classes, counts, colors, title, output_name):
    percentages = [f'{c / DATASET_SIZE * 100:.1f}%' for c in counts]
    labels = [f'{c} ({p})' for c, p in zip(counts, percentages)]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(range(len(classes)), counts, color=colors,
                  edgecolor='black', linewidth=0.6, width=0.65)

    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                label, ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=13)
    ax.set_ylabel('Image Count', fontsize=16)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=12)
    ax.set_ylim(0, max(counts) * 1.15)

    ax.text(0.98, 0.95, f'Dataset Size = {DATASET_SIZE}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.8))

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    output_path = VISUALIZATIONS_PATH / output_name
    fig.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f'Saved: {output_path} + .pdf')
    plt.close(fig)


if __name__ == '__main__':
    make_bar_chart(
        classes=['Background\n/Other', 'Clean\nHull', 'Slime\n/Algae',
                'Barnacles\n/Molluscs', 'Calcareous\nDeposits'],
        counts=[418, 392, 374, 296, 81],
        colors=['#2B8AFF', '#00FF00', '#FFFF6A', '#9D29B1', '#FF5733'],
        title='Class Occurrence Frequency (Before Merge)',
        output_name='fig1_class_distribution_before_merge.png',
    )

    # Figure 2: use CLASS_DICTIONARY_new colors
    make_bar_chart(
        classes=['Background\n/Other', 'Clean\nHull', 'Soft\nFouling',
                'Hard\nFouling'],
        counts=[418, 392, 374, 305],
        colors=['#9D29B1', '#00FF00', '#FFFF6A', '#FF5733'],
        title='Class Occurrence Frequency (After Merge)',
        output_name='fig2_class_distribution_after_merge.png',
    )