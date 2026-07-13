"""
Failure-stage distribution plots comparing mix vs large tasks.

Three chart variants:
  - plot_grouped_stacked_bar: side-by-side stacked bars for mix vs large
  - plot_compact_horizontal: horizontal stacked bars for N=2 vs N=3 conditions
  - plot_compact_horizontal_3_levels: horizontal bars for N=2 / N=3 / N=4

All read from error_ana/result/ JSONL files via get_data().
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import read_jsonl
from matplotlib.patches import Patch

def plot_grouped_stacked_bar(data_mix, data_large):
    models = ['Gemini-2.5 Pro', 'Gemini-2.5 Flash', 'GPT-5.1', 'GPT-4o']

    df_mix = pd.DataFrame(data_mix, index=models)
    df_large = pd.DataFrame(data_large, index=models)

    df_mix_pct = df_mix.div(df_mix.sum(axis=1), axis=0) * 100
    df_large_pct = df_large.div(df_large.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = [
        '#d62728',  # Perception
        '#ff7f0e',  # Decomposition
        '#1f77b4',  # Composition
        '#9467bd'   # Application
    ]
    error_types = df_mix.columns.tolist()

    x = np.arange(len(models))
    width = 0.35

    bottom_mix = np.zeros(len(models))
    bottom_large = np.zeros(len(models))

    for i, (col, color) in enumerate(zip(error_types, colors)):
        p1 = ax.bar(x - width/2 - 0.02, df_mix_pct[col], width, bottom=bottom_mix,
                    color=color, alpha=0.55, edgecolor='white', label=col if i == 0 else "")

        p2 = ax.bar(x + width/2 + 0.02, df_large_pct[col], width, bottom=bottom_large,
                    color=color, alpha=1.0, edgecolor='white')

        bottom_mix += df_mix_pct[col]
        bottom_large += df_large_pct[col]

        for bar in p1:
            height = bar.get_height()
            if height > 5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2,
                        f'{height:.0f}%', ha='center', va='center', color='white', fontsize=8, weight='bold')

        for bar in p2:
            height = bar.get_height()
            if height > 5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2,
                        f'{height:.0f}%', ha='center', va='center', color='white', fontsize=8, weight='bold')

    ax.set_title('Failure Mode Distribution: Mix vs. Large Scale', fontsize=16, pad=20)
    ax.set_ylabel('Percentage of Failure Cases (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)

    legend_elements_color = [Patch(facecolor=colors[i], label=error_types[i]) for i in range(len(colors))]
    legend1 = ax.legend(handles=legend_elements_color, title='Error Type',
                        bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.add_artist(legend1)

    legend_elements_style = [
        Patch(facecolor='gray', alpha=0.55, edgecolor='white', label='Mix (Standard)'),
        Patch(facecolor='gray', alpha=1.0, edgecolor='white', label='Large (Complex)')
    ]
    ax.legend(handles=legend_elements_style, title='Task Setting',
              bbox_to_anchor=(1.01, 0.75), loc='upper left')

    plt.tight_layout()
    plt.savefig('/data/yongka/analogy/spatial/figures2/multi_model_failure_distribution_compare.png', dpi=300)
    print('Plot showing...')
    plt.show()

def plot_compact_horizontal(data_mix, data_large, data_larger):
    models = ['Gemini-2.5 Pro', 'Gemini-2.5 Flash', 'GPT-5.1', 'GPT-4o']
    models = models[::-1]

    df_mix = pd.DataFrame(data_mix, index=models[::-1]).reindex(models)
    df_large = pd.DataFrame(data_large, index=models[::-1]).reindex(models)

    df_mix_pct = df_mix.div(df_mix.sum(axis=1), axis=0) * 100
    df_large_pct = df_large.div(df_large.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [
        '#d62728',  # Perception
        '#ff7f0e',  # Decomposition
        '#1f77b4',  # Composition
        '#9467bd'   # Application
    ]
    error_types = df_mix.columns.tolist()

    y = np.arange(len(models))
    height = 0.4

    left_mix = np.zeros(len(models))
    left_large = np.zeros(len(models))

    for i, (col, color) in enumerate(zip(error_types, colors)):
        p1 = ax.barh(y + height/2 + 0.02, df_mix_pct[col], height, left=left_mix,
                     color=color, alpha=0.55, edgecolor='white', linewidth=0.8)

        p2 = ax.barh(y - height/2 - 0.02, df_large_pct[col], height, left=left_large,
                     color=color, alpha=1.0, edgecolor='white', linewidth=0.8)

        left_mix += df_mix_pct[col]
        left_large += df_large_pct[col]

        for bar in p1:
            w = bar.get_width()
            if w > 6:
                ax.text(bar.get_x() + w/2, bar.get_y() + bar.get_height()/2,
                        f'{w:.0f}%', ha='center', va='center', color='white',
                        fontsize=10, weight='bold')

        for bar in p2:
            w = bar.get_width()
            if w > 6:
                ax.text(bar.get_x() + w/2, bar.get_y() + bar.get_height()/2,
                        f'{w:.0f}%', ha='center', va='center', color='white',
                        fontsize=10, weight='bold')

    ax.set_title('Error Distribution by Number of Atomic Transformations', fontsize=18, pad=20, fontfamily='serif', weight='bold')
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(models, rotation=30, fontsize=12, fontfamily='serif', weight='bold')

    legend_elements_color = [Patch(facecolor=colors[i], label=error_types[i]) for i in range(len(colors))]
    legend_elements_style = [
        Patch(facecolor='gray', alpha=0.55, edgecolor='white', label='N=2'),
        Patch(facecolor='gray', alpha=1.0, edgecolor='white', label='N=3')
    ]

    ax.legend(handles=legend_elements_color + legend_elements_style,
              loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=15)

    plt.tight_layout()
    plt.savefig('/data/yongka/analogy/spatial/figures2/multi_model_failure_distribution_compare2.png', dpi=300)
    print('Saved compact horizontal plot to /data/yongka/analogy/spatial/figures2/multi_model_failure_distribution_compare2.png')
    plt.show()

def plot_compact_horizontal_3_levels(data_mix, data_large, data_larger):
    models = ['Gemini-2.5 Pro', 'Gemini-2.5 Flash', 'GPT-5.1', 'GPT-4o']
    models = models[::-1]

    df_mix = pd.DataFrame(data_mix, index=models[::-1]).reindex(models)
    df_large = pd.DataFrame(data_large, index=models[::-1]).reindex(models)
    df_larger = pd.DataFrame(data_larger, index=models[::-1]).reindex(models)

    df_mix_pct = df_mix.div(df_mix.sum(axis=1), axis=0) * 100
    df_large_pct = df_large.div(df_large.sum(axis=1), axis=0) * 100
    df_larger_pct = df_larger.div(df_larger.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 9))

    colors = [
        '#d62728',  # Perception
        '#ff7f0e',  # Decomposition
        '#1f77b4',  # Composition
        '#9467bd'   # Application
    ]
    error_types = df_mix.columns.tolist()

    y = np.arange(len(models))
    height = 0.28
    gap = 0.03
    alphas = [0.2, 0.4, 0.6]

    left_mix = np.zeros(len(models))
    left_large = np.zeros(len(models))
    left_larger = np.zeros(len(models))

    for i, (col, color) in enumerate(zip(error_types, colors)):
        p1 = ax.barh(y + height + gap, df_mix_pct[col], height, left=left_mix,
                     color=color, alpha=alphas[0], edgecolor='white', linewidth=0.8)

        p2 = ax.barh(y, df_large_pct[col], height, left=left_large,
                     color=color, alpha=alphas[1], edgecolor='white', linewidth=0.8)

        p3 = ax.barh(y - height - gap, df_larger_pct[col], height, left=left_larger,
                     color=color, alpha=alphas[2], edgecolor='white', linewidth=0.8)

        left_mix += df_mix_pct[col]
        left_large += df_large_pct[col]
        left_larger += df_larger_pct[col]

        def add_labels(bars, alpha_val):
            for bar in bars:
                w = bar.get_width()
                if w > 6:
                    ax.text(bar.get_x() + w/2, bar.get_y() + bar.get_height()/2,
                            f'{w:.0f}%', ha='center', va='center', color='black',
                            fontsize=16, weight='bold')

        add_labels(p1, alphas[0])
        add_labels(p2, alphas[1])
        add_labels(p3, alphas[2])

    ax.set_xlim(0, 100)
    ax.tick_params(axis='x', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(models, rotation=45, fontsize=20, fontfamily='serif', weight='bold')

    legend_elements_color = [Patch(facecolor=colors[i], label=error_types[i]) for i in range(len(colors))]
    legend_elements_style = [
        Patch(facecolor='gray', alpha=alphas[0], edgecolor='white', label='N=2'),
        Patch(facecolor='gray', alpha=alphas[1], edgecolor='white', label='N=3'),
        Patch(facecolor='gray', alpha=alphas[2], edgecolor='white', label='N=4')
    ]

    legend1 = ax.legend(
        handles=legend_elements_color,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
        frameon=False,
        fontsize=20,
        columnspacing=2.0
    )
    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=legend_elements_style,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        frameon=False,
        fontsize=20,
        columnspacing=2.0
    )

    plt.tight_layout()
    saved_path = '/data/yongka/analogy/spatial/figures2/error_dis_large.png'
    print(f'Saving plot to {saved_path}...')
    plt.savefig(saved_path, dpi=300)
    plt.show()
    fig.savefig('/data/yongka/analogy/spatial/figures2/error_dis_large.pdf', format='pdf')
    print('Plot saved to /data/yongka/analogy/spatial/figures2/error_dis_large.pdf')

def get_data(operations, task):
    models = ['gemini2.5pro', 'gemini2.5flash', 'gpt5', 'gpt']
    data_folder = '/data/yongka/analogy/spatial/error_ana/result'
    data = {
        'Perception Failure':    [0, 0, 0, 0],
        'Decomposition Failure': [0, 0, 0, 0],
        'Composition Failure':   [0, 0, 0, 0],
        'Application Failure':   [0, 0, 0, 0]
    }
    for operation in operations:
        for i, model in enumerate(models):
            file_path = f'{data_folder}/{task}/{operation}_{model}.jsonl'
            result = read_jsonl(file_path)
            for item in result:
                error_type = item.get('failure_stage', None)
                if error_type is None:
                    continue
                if error_type == 1:
                    data['Perception Failure'][i] += 1
                elif error_type == 2:
                    data['Decomposition Failure'][i] += 1
                elif error_type == 3:
                    data['Composition Failure'][i] += 1
                elif error_type == 4:
                    data['Application Failure'][i] += 1
    return data

if __name__ == '__main__':
    data_mix = get_data(['union', 'intersection'], 'mix')
    data_large = get_data(['union', 'intersection'], 'large')
    data_larger = get_data(['union', 'intersection'], 'large_hard')
    plot_compact_horizontal_3_levels(data_mix, data_large, data_larger)
