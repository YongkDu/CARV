"""
Logistic regression analysis of property-pair difficulty (per-model heatmaps).

For each model, loads inference output JSONL files and label JSON files, reformats
items as binary feature vectors over all property pairs, fits a logistic regression,
and plots the resulting coefficients as a heatmap. Coefficients indicate how much
each property combination contributes to model failure probability.

Usage:
    Configure OUTPUT_DIR, LABEL_DIR, and fig_root in __main__, then run:
        python plot_comb.py
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

def plot_regression_heatmap(results_df, save_path):
    """Plot logistic regression coefficients as a symmetric heatmap."""
    all_props = set()
    for row_name in results_df['feature']:
        props = row_name.split('-')
        if len(props) == 2:
            all_props.update(props)
    all_props = sorted(list(all_props))

    coef_matrix = pd.DataFrame(np.nan, index=all_props, columns=all_props)

    for idx, row in results_df.iterrows():
        props = row['feature'].split('-')
        if len(props) != 2 or props[0] not in all_props or props[1] not in all_props:
            continue

        prop1, prop2 = props[0], props[1]
        coef = row['weight']
        coef_matrix.loc[prop1, prop2] = coef
        coef_matrix.loc[prop2, prop1] = coef

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

if __name__ == '__main__':
    OUTPUT_DIR = "/data/yongka/analogy/spatial/output_base_new/large"
    LABEL_DIR = "/data/yongka/analogy/spatial/data_labels_all/large"
    fig_root = "/data/yongka/analogy/spatial/figures"

    for model in ['gemini2.5pro', 'gemini2.5flash', 'gpt5', 'gpt']:
        reformed_data = []
        for operation in ['union', 'intersection']:
            for diff in ['easy', 'hard']:
                label_file = f"{LABEL_DIR}/{operation}/{diff}_500.json"
                response_file = f"{OUTPUT_DIR}/{operation}/{model}_{diff}_500.jsonl"
                reformed_data += reform_data(label_file, response_file)

            X, Y, feature_cols = load_and_format_data(reformed_data)
            results_df = run_logistic_regression(X, Y, feature_cols)

            if results_df is not None:
                plot_regression_heatmap(results_df, f"{fig_root}/heatmap_{operation}_{model}_large2.png")
                print(f"\nVisualization saved as 'heatmap_{model}_large2.png'.")
            else:
                print("\nCould not visualize: logistic regression did not converge. Check data distribution and sample size.")
