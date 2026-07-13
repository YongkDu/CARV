"""
Logistic regression with conditional error rate and radar chart (mix task).

Variant of plot_comb_redar.py that uses compute_probability (conditional error rate
P(Y=1|X_feature=1)) instead of log-odds coefficients, together with global min-max
rescaling before the merged radar chart.

Usage:
    Configure OUTPUT_DIR, LABEL_DIR, and fig_root in __main__, then run:
        python plot_comb_redar2.py
"""
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
import sys
from sklearn.linear_model import LogisticRegression
import json
import os
from utils import read_jsonl
import matplotlib.gridspec as gridspec

def load_and_format_data(raw_data_list):
    """Build binary feature vectors over property pairs and failure labels for logistic regression."""
    all_properties = ['subject', 'position', 'color', 'number', 'object']
    all_pairs = list(itertools.combinations(sorted(all_properties), 2))
    pair_to_index = {pair: i for i, pair in enumerate(all_pairs)}

    data_records = []

    for item in raw_data_list:
        p1_list = item.get('property_1', [])
        p2_list = item.get('property_2', [])

        failure_label = 1 if item.get('correctness', False) is False else 0
        feature_vector = [0] * len(all_pairs)

        all_props_in_sample = set(p1_list + p2_list)
        # transform p1_list and p2_list into pairs
        p1_pairs = list(itertools.combinations(sorted(p1_list), 2))
        p2_pairs = list(itertools.combinations(sorted(p2_list), 2))

        for pair in p1_pairs + p2_pairs:
            if pair in pair_to_index:
                index = pair_to_index[pair]
                feature_vector[index] = 1

        data_records.append({
            'features': feature_vector,
            'failure': failure_label
        })

    X = np.array([record['features'] for record in data_records])
    Y = np.array([record['failure'] for record in data_records])

    return X, Y, [f'{p[0]}-{p[1]}' for p in all_pairs]

def run_logistic_regression(X, Y, feature_cols):
    """Fit a logistic regression and return a DataFrame of feature weights."""
    clf = LogisticRegression()
    clf.fit(X, Y)
    weights = pd.DataFrame({
        "feature": feature_cols,
        "weight": clf.coef_[0]
    })
    print(weights)
    return weights

def compute_probability(X, Y, feature_cols):
    """Compute conditional error rate P(Y=1 | X_feature=1) for each property pair feature."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_cols)
    Y = np.array(Y)
    results = []
    for col in feature_cols:
        mask = (X[col] == 1)
        count = mask.sum()
        if count == 0:
            prob = 0.0
        else:
            prob = Y[mask].mean()
        results.append({
            "feature": col,
            "weight": prob
        })
    weights = pd.DataFrame(results)
    return weights

def reform_data(label_file, response_file):
    reformed_data = []
    with open(label_file, "r") as f:
        label_data = json.load(f)
    response_data = read_jsonl(response_file)
    for i, item in enumerate(label_data):
        reformed_data.append({
            'property_1': item['changed_properties1'],
            'property_2': item['changed_properties2'],
            'correctness': None
        })
        correctness_analysis = response_data[i].get('correctness_analysis', {})
        correctness = correctness_analysis.get('correctness', None)
        if correctness is not None:
            reformed_data[-1]['correctness'] = True if correctness == True else False
        if reformed_data[-1]['correctness'] is None:
            reformed_data.pop()
    return reformed_data

def plot_radar_chart_merged(model_dic, save_path):
    """
    Plot a radar chart from model weights (single DataFrame per model, not split by operation).
    """
    data_merged = {}

    for model_name, df_results in model_dic.items():
        if 'feature' in df_results.columns:
            series = df_results.set_index('feature')['weight']
        else:
            series = df_results['weight']

        valid_features = [f for f in series.index if isinstance(f, str) and len(f.split('-')) == 2]
        data_merged[model_name] = series[valid_features]

    df_plot = pd.DataFrame(data_merged)
    df_plot = df_plot.sort_index()

    categories = df_plot.index.tolist()
    categories = [cat.replace('-', '\n') for cat in categories]
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd',
              '#8c564b', '#e377c2']

    for i, model_name in enumerate(df_plot.columns):
        values = df_plot[model_name].tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=2, linestyle='-', label=model_name, color=colors[i % len(colors)])
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=20, fontweight='bold', fontfamily='serif')
    ax.tick_params(axis='x', pad=10)

    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1.0"], color="grey", size=12)
    plt.ylim(0.0, 1.0)

    ax.grid(color='gray', linestyle='--', alpha=0.3)

    leg = plt.legend(loc='upper right', bbox_to_anchor=(2, 1.1), frameon=False, fontsize=20)
    for line in leg.get_lines():
        line.set_linewidth(6.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Radar chart saved to {save_path}")
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')

def rescale_model_dict_global(model_dict):
    """Apply global min-max scaling across all model weights to normalize for radar chart comparison."""
    import copy
    new_dict = copy.deepcopy(model_dict)

    all_values = []
    for model_name, df in new_dict.items():
        if 'weight' in df.columns:
            all_values.extend(df['weight'].tolist())

    global_min = min(all_values)
    global_max = max(all_values)

    print(f"Global Min: {global_min}, Global Max: {global_max}")

    if global_max == global_min:
        return new_dict

    for model_name, df in new_dict.items():
        if 'weight' in df.columns:
            df['weight'] = (df['weight'] - global_min) / (global_max - global_min)

    return new_dict

if __name__ == '__main__':
    OUTPUT_DIR = "/data/yongka/analogy/spatial/output_base_new/mix"
    LABEL_DIR = "/data/yongka/analogy/spatial/data_labels_all/mix"
    fig_root = "/data/yongka/analogy/spatial/figures2"

    modelname_map = {
        'gemini2.5pro': 'Gemini-2.5 Pro',
        'gemini2.5flash': 'Gemini-2.5 Flash',
        'qwen32b': 'Qwen2.5VL-32B',
        'gpt5': 'GPT-5.1',
        'gpt': 'GPT-4o',
        'qwen3-30b': 'Qwen3VL-30B'
    }
    model_dict = {}
    for model in ['gemini2.5pro', 'gemini2.5flash', 'gpt5', 'qwen32b', 'gpt', 'qwen3-30b']:
        reformed_data = []
        for operation in ['union', 'intersection']:
            for diff in ['easy']:
                label_file = f"{LABEL_DIR}/{operation}/{diff}_500.json"
                response_file = f"{OUTPUT_DIR}/{operation}/{model}_{diff}_500.jsonl"
                reformed_data += reform_data(label_file, response_file)

        X, Y, feature_cols = load_and_format_data(reformed_data)
        results_df = run_logistic_regression(X, Y, feature_cols)
        model_dict[modelname_map[model]] = results_df
    if model_dict is not None:
        model_dict = rescale_model_dict_global(model_dict)
        plot_radar_chart_merged(model_dict, f"{fig_root}/radar2.png")
        print(f"\nVisualization saved as 'radar2.png'.")
    else:
        print("\nCould not visualize: logistic regression did not converge. Check data distribution and sample size.")
