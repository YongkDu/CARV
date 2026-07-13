"""
Failure-stage distribution plots for the mix task.

Generates two chart types from error_ana/result/ JSONL files:
  - Vertical stacked bar chart (plot_multi_model_failure_distribution)
  - Horizontal stacked bar chart (plot_multi_model_failure_distribution_horizontal)

Each bar segment represents the percentage of failures attributed to one of four
diagnosis stages: Perception, Decomposition, Composition, or Application.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import read_jsonl

def plot_multi_model_failure_distribution(data):
    models = ['Gemini-2.5 Pro', 'Gemini-2.5 Flash', 'GPT-5.1', 'GPT-4o', 'Qwen-32B']

    df = pd.DataFrame(data, index=models)
    df_pct = df.div(df.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = [
        '#d62728',  # Perception
        '#ff7f0e',  # Decomposition
        '#1f77b4',  # Composition
        '#9467bd'   # Application
    ]

    df_pct.plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.6, edgecolor='white')

    ax.set_title('Error Distribution Across Models', fontsize=18, pad=20, fontfamily='serif', weight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xticks(rotation=30, fontsize=11, weight='bold', fontfamily='serif')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(reversed(handles), reversed(labels), title='Failure Mode', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    for c in ax.containers:
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 5 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=10, color='white', weight='bold')

    plt.tight_layout()
    plt.savefig('/data/yongka/analogy/spatial/figures2/multi_model_failure_distribution_mix3.png', dpi=300)
    print('Plot saved to /data/yongka/analogy/spatial/figures2/multi_model_failure_distribution_mix3.png')
    plt.show()

def plot_multi_model_failure_distribution_horizontal(data):
    models = ['Gemini-2.5 Pro', 'Gemini-2.5 Flash', 'GPT-5.1', 'GPT-4o', 'Qwen2.5VL-32B', 'Qwen3VL-30B']

    df = pd.DataFrame(data, index=models)
    df = df.iloc[::-1]
    df_pct = df.div(df.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [
        '#d62728',  # Perception
        '#ff7f0e',  # Decomposition
        '#1f77b4',  # Composition
        '#9467bd'   # Application
    ]

    df_pct.plot(kind='barh', stacked=True, color=colors, ax=ax, width=0.6, edgecolor='white')

    ax.set_xlim(0, 100)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.yticks(fontsize=20, rotation=45, weight='bold', fontfamily='serif')
    plt.xticks(fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels,
               title='',
               bbox_to_anchor=(0.45, -0.18),
               loc='upper center',
               ncol=2,
               frameon=False,
               columnspacing=0.8,
               fontsize=22)

    for c in ax.containers:
        labels = [f'{v.get_width():.1f}%' if v.get_width() > 5 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=16, color='white', weight='bold')

    plt.tight_layout()
    plt.savefig('/data/yongka/analogy/spatial/figures2/multi_model_failure_distribution_mix5.png', dpi=300)
    print('Plot showing... (Saved to /data/yongka/analogy/spatial/figures2/multi_model_failure_distribution_mix5.png)')
    plt.show()
    fig.savefig('/data/yongka/analogy/spatial/figures2/multi_model_failure_distribution_mix5.pdf', format='pdf')
    print('Plot saved to /data/yongka/analogy/spatial/figures2/multi_model_failure_distribution_mix5.pdf')

def get_data(operations):
    models = ['gemini2.5pro', 'gemini2.5flash', 'gpt5', 'gpt', 'qwen32b', 'qwen3-30b']
    data_folder = '/data/yongka/analogy/spatial/error_ana/result'
    data = {
        'Perception Failure':    [0, 0, 0, 0, 0, 0],
        'Decomposition Failure': [0, 0, 0, 0, 0, 0],
        'Composition Failure':   [0, 0, 0, 0, 0, 0],
        'Application Failure':   [0, 0, 0, 0, 0, 0]
    }
    for task in ['mix']:
        for operation in operations:
            for i, model in enumerate(models):
                file_path = f'{data_folder}/{task}/{operation}_{model}.jsonl'
                result = read_jsonl(file_path)
                for item in result:
                    error_type = item['failure_stage']
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
    data = get_data(['union', 'intersection'])
    plot_multi_model_failure_distribution_horizontal(data)
