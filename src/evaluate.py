"""
evaluate.py
-----------
Aggregates results from all model scripts into a single comparison
table and generates a summary bar chart. Run this after
models_classical.py, models_deep.py, and models_roberta.py have
each produced their CSVs in results/.

Usage:
    python src/evaluate.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def safe_read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    rows = []

    tuned = safe_read(os.path.join(RESULTS_DIR, 'classical_tuned_test.csv'))
    if tuned is not None:
        for _, r in tuned.iterrows():
            rows.append({'Model': r['Model'] + ' (tuned, TF-IDF)', 'Test_F1': r['TEST_F1']})

    cnn = safe_read(os.path.join(RESULTS_DIR, 'cnn_results.csv'))
    if cnn is not None:
        test_row = cnn[cnn['Tag'] == 'CNN | TEST']
        if not test_row.empty:
            rows.append({'Model': 'CNN', 'Test_F1': test_row.iloc[0]['F1']})

    roberta = safe_read(os.path.join(RESULTS_DIR, 'roberta_results.csv'))
    if roberta is not None:
        for _, r in roberta.iterrows():
            rows.append({'Model': r['Variant'], 'Test_F1': r['Test_F1']})

    if not rows:
        print('No result CSVs found in results/. Run the model scripts first.')
        return

    df = pd.DataFrame(rows).sort_values('Test_F1', ascending=False).reset_index(drop=True)
    print('\n=== Overall Model Comparison (Test macro-F1) ===')
    print(df.to_string(index=False))

    df.to_csv(os.path.join(RESULTS_DIR, 'overall_comparison.csv'), index=False)

    # plot
    plt.figure(figsize=(10, max(4, len(df) * 0.4)))
    plt.barh(df['Model'], df['Test_F1'], color='steelblue')
    plt.xlabel('Test macro-F1')
    plt.title('Model Comparison — Clickbait Detection')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'model_comparison.png')
    plt.savefig(out_path, dpi=150)
    print(f'\nSaved comparison table to results/overall_comparison.csv')
    print(f'Saved comparison chart to {out_path}')


if __name__ == '__main__':
    main()
