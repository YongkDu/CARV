"""
Logistic regression analysis with 4-in-1 heatmaps and radar chart (mix task).

Fits logistic regression per model and operation, then visualizes regression
coefficients as:
  - Single heatmap with union (lower triangle) vs intersection (upper triangle)
  - 4-in-1 heatmap grid (2x2) for comparing models
  - 4-in-1 compact row layout (1x4)
  - Merged radar chart showing average difficulty across property combinations

Usage:
    Configure OUTPUT_DIR, LABEL_DIR, and fig_root in __main__, then run:
        python plot_comb_redar.py
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
    return

def plot_regression_heatmap(results_dic, save_path):
    """
    Plot logistic regression coefficients as a heatmap with union in the lower triangle
    and intersection in the upper triangle.
    """
    results_df_union = results_dic['union']
    results_df_intersection = results_dic['intersection']
    all_props = set()
    for row_name in results_df_union['feature']:
        props = row_name.split('-')
        if len(props) == 2:
            all_props.update(props)
    all_props = sorted(list(all_props))

    coef_matrix = pd.DataFrame(np.nan, index=all_props, columns=all_props)

    for idx, row in results_df_union.iterrows():
        props = row['feature'].split('-')
        if len(props) != 2 or props[0] not in all_props or props[1] not in all_props:
            continue
        prop1, prop2 = props[0], props[1]
        coef = row['weight']
        coef_matrix.loc[prop2, prop1] = coef
    for idx, row in results_df_intersection.iterrows():
        props = row['feature'].split('-')
        if len(props) != 2 or props[0] not in all_props or props[1] not in all_props:
            continue
        prop1, prop2 = props[0], props[1]
        coef = row['weight']
        coef_matrix.loc[prop1, prop2] = coef

    mask = np.triu(np.ones_like(coef_matrix, dtype=bool), k=0)
    plt.figure(figsize=(10, 9))

    sns.heatmap(coef_matrix,
                annot=True,
                cmap='Reds',
                vmax=coef_matrix.max().max() * 1.1,
                center=0,
                linewidths=.5,
                linecolor='gray',
                cbar_kws={'label': 'Log-Odds Coefficient (Difficulty Contribution)'})

    plt.title('Contribution of Property Pairs to Model Failure (Logistic Regression)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

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

def plot_regression_heatmap_4in1(model_dic, save_path):
    """
    Plot regression coefficient heatmaps for multiple models in a 2x2 grid.
    Lower triangle: Union task; Upper triangle: Intersection task.
    """
    map_data = []
    all_props = set()
    for model_name, results_dic in model_dic.items():
        results_df_union = results_dic['union']
        for row_name in results_df_union['feature']:
            props = row_name.split('-')
            if len(props) == 2:
                all_props.update(props)

    all_props = sorted(list(all_props))
    n = len(all_props)

    global_min = 0
    global_max = 0

    for model_name, results_dic in model_dic.items():
        results_df_union = results_dic['union']
        results_df_intersection = results_dic['intersection']

        coef_matrix = pd.DataFrame(np.nan, index=all_props, columns=all_props)

        for idx, row in results_df_union.iterrows():
            props = row['feature'].split('-')
            if len(props) != 2:
                continue
            prop1, prop2 = props[0], props[1]
            if prop1 in all_props and prop2 in all_props:
                coef_matrix.loc[prop2, prop1] = row['weight']

        for idx, row in results_df_intersection.iterrows():
            props = row['feature'].split('-')
            if len(props) != 2:
                continue
            prop1, prop2 = props[0], props[1]
            if prop1 in all_props and prop2 in all_props:
                coef_matrix.loc[prop1, prop2] = row['weight']
        map_data.append((model_name, coef_matrix))
        vals = coef_matrix.values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            global_max = max(global_max, vals.max())
            global_min = min(global_min, vals.min())

    limit = max(abs(global_min), abs(global_max))
    vmin, vmax = -limit, limit

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.15, hspace=0.2)

    axes = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]),
            plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])]
    cbar_ax = plt.subplot(gs[:, 2])

    for ax, (model_name, mat) in zip(axes, map_data):
        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            annot_kws={"size": 9},
            cmap='RdBu_r',
            center=0,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            linecolor='#f0f0f0',
            cbar=False
        )

        ax.set_title(model_name, fontsize=14, fontweight='bold', pad=10, fontfamily='serif')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right', fontsize=10, fontfamily='serif', weight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, fontfamily='serif', weight='bold')

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.outline.set_visible(False)

    plt.suptitle("Difficulty Contribution of Property Combinations Across Models", fontsize=18, y=0.96, fontweight='bold', fontfamily='serif')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')

def plot_regression_heatmap_4in1_compact(model_dic, save_path):
    """
    Plots model regression coefficient heatmaps in a single row (1x4).
    Lower triangle: Union task; Upper triangle: Intersection task.
    Compact version: No annotations, smaller cells.
    """
    map_data = []
    all_props = set()
    for model_name, results_dic in model_dic.items():
        results_df_union = results_dic['union']
        for row_name in results_df_union['feature']:
            props = row_name.split('-')
            if len(props) == 2:
                all_props.update(props)

    all_props = sorted(list(all_props))

    global_min = 0
    global_max = 0

    for model_name, results_dic in model_dic.items():
        results_df_union = results_dic['union']
        results_df_intersection = results_dic['intersection']

        coef_matrix = pd.DataFrame(np.nan, index=all_props, columns=all_props)

        for idx, row in results_df_union.iterrows():
            props = row['feature'].split('-')
            if len(props) == 2:
                prop1, prop2 = props[0], props[1]
                if prop1 in all_props and prop2 in all_props:
                    coef_matrix.loc[prop2, prop1] = row['weight']

        for idx, row in results_df_intersection.iterrows():
            props = row['feature'].split('-')
            if len(props) == 2:
                prop1, prop2 = props[0], props[1]
                if prop1 in all_props and prop2 in all_props:
                    coef_matrix.loc[prop1, prop2] = row['weight']

        map_data.append((model_name, coef_matrix))

        vals = coef_matrix.values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            global_max = max(global_max, vals.max())
            global_min = min(global_min, vals.min())

    limit = max(abs(global_min), abs(global_max))
    vmin, vmax = -limit, limit

    fig = plt.figure(figsize=(24, 5.5))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.08], wspace=0.1)

    axes = [plt.subplot(gs[0, i]) for i in range(4)]
    cbar_ax = plt.subplot(gs[0, 4])

    for ax, (model_name, mat) in zip(axes, map_data):
        sns.heatmap(
            mat,
            ax=ax,
            annot=False,
            cmap='RdBu_r',
            center=0,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            linecolor='#f8f8f8',
            cbar=False
        )

        ax.set_title(model_name, fontsize=16, fontweight='bold', pad=12, fontfamily='serif')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12, fontfamily='serif', weight='bold')

        if ax == axes[0]:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12, fontfamily='serif', weight='bold')
        else:
            ax.set_yticks([])

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=10)

    plt.suptitle("Difficulty Contribution of Property Combinations Across Models", fontsize=20, y=1.05, fontweight='bold', fontfamily='serif')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Compact heatmap saved to {save_path}")
    plt.show()
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')

def plot_radar_chart_merged(model_dic, save_path):
    """
    Merge Union and Intersection regression coefficients (average) and plot a radar chart.
    Compares models across all property pair combinations.
    """
    data_merged = {}
    all_features = set()

    for model_name, results_dic in model_dic.items():
        df_union = results_dic['union'].set_index('feature')['weight']
        df_inter = results_dic['intersection'].set_index('feature')['weight']

        features = set(df_union.index).union(set(df_inter.index))
        valid_features = [f for f in features if len(f.split('-')) == 2]
        all_features.update(valid_features)

        s1, s2 = df_union.align(df_inter, join='outer', fill_value=0)
        avg_weights = (s1 + s2) / 2

        data_merged[model_name] = avg_weights[valid_features]

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

    plt.yticks([-0.5, 0, 0.5, 1.0], ["-0.5", "0", "0.5", "1.0"], color="grey", size=12)
    plt.ylim(-1.0, 1.5)

    ax.grid(color='gray', linestyle='--', alpha=0.3)

    leg = plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.25), frameon=False, fontsize=20)
    for line in leg.get_lines():
        line.set_linewidth(6.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Radar chart saved to {save_path}")
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')

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
        result_dict = {}

        for operation in ['union', 'intersection']:
            reformed_data = []
            for diff in ['easy']:
                label_file = f"{LABEL_DIR}/{operation}/{diff}_500.json"
                response_file = f"{OUTPUT_DIR}/{operation}/{model}_{diff}_500.jsonl"
                reformed_data += reform_data(label_file, response_file)

            X, Y, feature_cols = load_and_format_data(reformed_data)
            results_df = run_logistic_regression(X, Y, feature_cols)
            result_dict[f"{operation}"] = results_df
        model_dict[modelname_map[model]] = result_dict
    if model_dict is not None:
        plot_radar_chart_merged(model_dict, f"{fig_root}/radar.png")
        print(f"\nVisualization saved as 'radar.png'.")
    else:
        print("\nCould not visualize: logistic regression did not converge. Check data distribution and sample size.")
