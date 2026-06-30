"""
models_classical.py
-------------------
Trains and evaluates classical ML classifiers on the Webis 2017 dataset.
Covers four experimental conditions reported in the dissertation:

  1. Handcrafted features (baseline + case-based feature subsets)
  2. TF-IDF representations (postText and targetTitle)
  3. Tuned models via GridSearchCV
  4. Five balanced training sets via majority-class undersampling

Requires feature CSVs produced by preprocessing.py and the raw JSONL data
for TF-IDF experiments (to recreate text splits).

Usage:
    python src/models_classical.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.base import clone

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print('XGBoost not found; skipping XGBoost models.')

DATA_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data')
RANDOM_STATE = 47

# ── feature column indices (within a single field's 8-feature block) ────────
# Order: chars, words, begins_q, adj_count, exclaim_count, question_count,
#        slang_count, uppercase_ratio
COUNT_IDX    = [0, 1]        # chars, words
QUESTION_IDX = [2, 5]        # begins_q, question_count
STYLE_IDX    = [4, 7, 6, 3]  # exclaim, uppercase_ratio, slang, adj


# ── helpers ─────────────────────────────────────────────────────────────────

def load_features():
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train_features.csv'))
    X_val   = pd.read_csv(os.path.join(DATA_DIR, 'X_val_features.csv'))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, 'X_test_features.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
    y_val   = pd.read_csv(os.path.join(DATA_DIR, 'y_val.csv')).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_raw_corpus():
    """Re-load raw text for TF-IDF experiments."""
    train_inst  = pd.read_json(os.path.join(DATA_DIR, 'train', 'instances.jsonl'), lines=True)
    train_truth = pd.read_json(os.path.join(DATA_DIR, 'train', 'truth.jsonl'),    lines=True)
    test_inst   = pd.read_json(os.path.join(DATA_DIR, 'test',  'instances.jsonl'), lines=True)
    test_truth  = pd.read_json(os.path.join(DATA_DIR, 'test',  'truth.jsonl'),    lines=True)

    train_df = pd.merge(train_inst, train_truth, on='id')
    test_df  = pd.merge(test_inst,  test_truth,  on='id')

    for df in [train_df, test_df]:
        df['postText']    = df['postText'].apply(
            lambda x: x[0] if isinstance(x, list) else x
        ).fillna('').astype(str).str.lower()
        df['targetTitle'] = df['targetTitle'].fillna('').astype(str).str.lower()
        df['truthClass']  = df['truthClass'].apply(lambda x: 1 if x == 'clickbait' else 0)

    return train_df, test_df


def summarize(tag, rep):
    return {
        'Model': tag,
        'Accuracy':  round(rep.get('accuracy', np.nan), 4),
        'Precision': round(rep['macro avg']['precision'], 4),
        'Recall':    round(rep['macro avg']['recall'], 4),
        'F1':        round(rep['macro avg']['f1-score'], 4),
    }


def eval_model(model, X_tr, y_tr, X_val, y_val, X_te, y_te, name):
    model.fit(X_tr, y_tr)
    rep_val  = classification_report(y_val, model.predict(X_val), output_dict=True, zero_division=0)
    rep_test = classification_report(y_te,  model.predict(X_te),  output_dict=True, zero_division=0)
    cm       = confusion_matrix(y_te, model.predict(X_te))
    print(f'\n  {name}  |  val F1={rep_val["macro avg"]["f1-score"]:.3f}'
          f'  test F1={rep_test["macro avg"]["f1-score"]:.3f}')
    return summarize(f'{name} | VAL', rep_val), summarize(f'{name} | TEST', rep_test), cm


def get_base_models():
    models = [
        ('LogReg', Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                                       n_jobs=-1, random_state=RANDOM_STATE))
        ])),
        ('LinearSVM', Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', LinearSVC(class_weight='balanced', random_state=RANDOM_STATE))
        ])),
        ('RandomForest', RandomForestClassifier(
            n_estimators=300, class_weight='balanced',
            n_jobs=-1, random_state=RANDOM_STATE
        )),
        ('DecisionTree', DecisionTreeClassifier(
            class_weight='balanced', random_state=RANDOM_STATE
        )),
        ('GaussianNB', Pipeline([
            ('scaler', StandardScaler(with_mean=True)),
            ('clf', GaussianNB())
        ])),
    ]
    if HAS_XGB:
        models.append(('XGBoost', xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.1, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            n_jobs=-1, random_state=RANDOM_STATE, eval_metric='logloss',
            verbosity=0
        )))
    return models


# ── experiment 1: handcrafted features ──────────────────────────────────────

def run_handcrafted(X_train, X_val, X_test, y_train, y_val, y_test):
    print('\n' + '='*60)
    print('EXPERIMENT 1 — HANDCRAFTED FEATURES (postText + targetTitle)')
    print('='*60)

    models = get_base_models()
    rows_val, rows_test = [], []

    # postText block (first 8 cols) vs targetTitle block (last 8 cols)
    for block_label, X_tr, X_v, X_te in [
        ('postText',     X_train.iloc[:, :8].values, X_val.iloc[:, :8].values, X_test.iloc[:, :8].values),
        ('targetTitle',  X_train.iloc[:, 8:].values, X_val.iloc[:, 8:].values, X_test.iloc[:, 8:].values),
        ('combined',     X_train.values,              X_val.values,              X_test.values),
    ]:
        print(f'\n  -- {block_label} --')
        for name, mdl in models:
            rv, rt, _ = eval_model(clone(mdl), X_tr, y_train, X_v, y_val, X_te, y_test,
                                   f'{name} ({block_label})')
            rows_val.append(rv)
            rows_test.append(rt)

    return pd.DataFrame(rows_val), pd.DataFrame(rows_test)


# ── experiment 2: case-based feature subsets ────────────────────────────────

def run_case_based(X_train, X_val, X_test, y_train, y_val, y_test):
    print('\n' + '='*60)
    print('EXPERIMENT 2 — CASE-BASED FEATURE SUBSETS')
    print('='*60)

    # use postText block for case analysis (first 8 cols)
    Xtr = X_train.iloc[:, :8].values
    Xvl = X_val.iloc[:, :8].values
    Xte = X_test.iloc[:, :8].values

    cases = {
        'Count-based':    ([0, 1],   Xtr[:, [0,1]],   Xvl[:, [0,1]],   Xte[:, [0,1]]),
        'Question-based': ([2, 5],   Xtr[:, [2,5]],   Xvl[:, [2,5]],   Xte[:, [2,5]]),
        'Style-based':    ([4,7,6,3],Xtr[:,[4,7,6,3]],Xvl[:,[4,7,6,3]],Xte[:,[4,7,6,3]]),
        'No features':    (None,     np.zeros((Xtr.shape[0],1)), np.zeros((Xvl.shape[0],1)), np.zeros((Xte.shape[0],1))),
        'postText+Title': (None,     X_train.values, X_val.values, X_test.values),
    }

    models = get_base_models()
    rows_val, rows_test = [], []

    for case_name, (_, X_tr, X_v, X_te) in cases.items():
        print(f'\n  -- {case_name} --')
        for name, mdl in models:
            rv, rt, _ = eval_model(clone(mdl), X_tr, y_train, X_v, y_val, X_te, y_test,
                                   f'{name} ({case_name})')
            rows_val.append(rv)
            rows_test.append(rt)

    return pd.DataFrame(rows_val), pd.DataFrame(rows_test)


# ── experiment 3: TF-IDF ────────────────────────────────────────────────────

def run_tfidf(train_df, test_df):
    print('\n' + '='*60)
    print('EXPERIMENT 3 — TF-IDF vs HANDCRAFTED')
    print('='*60)

    y_all = train_df['truthClass'].values

    rows_val, rows_test = [], []
    models = get_base_models()

    for field, col in [('postText', 'postText'), ('targetTitle', 'targetTitle')]:
        raw_text = train_df[col].values
        test_text = test_df[col].values

        raw_tr, raw_vl, y_tr, y_vl = train_test_split(
            raw_text, y_all, test_size=0.2, random_state=RANDOM_STATE,
            stratify=y_all
        )

        vec = TfidfVectorizer(max_features=5000)
        X_tr = vec.fit_transform(raw_tr)
        X_vl = vec.transform(raw_vl)
        X_te = vec.transform(test_text)
        y_te = test_df['truthClass'].values

        print(f'\n  -- TF-IDF ({field}) --')
        for name, mdl in models:
            if 'gaussiannb' in name.lower() or 'GaussianNB' in name:
                # GaussianNB needs dense; use MultinomialNB for sparse TF-IDF instead
                clf = MultinomialNB()
                tag = f'MultinomialNB (TF-IDF {field})'
            else:
                clf = clone(mdl)
                tag = f'{name} (TF-IDF {field})'
            rv, rt, _ = eval_model(clf, X_tr, y_tr, X_vl, y_vl, X_te, y_te, tag)
            rows_val.append(rv)
            rows_test.append(rt)

    return pd.DataFrame(rows_val), pd.DataFrame(rows_test)


# ── experiment 4: hyperparameter tuning ─────────────────────────────────────

def run_tuned(X_train, X_val, X_test, y_train, y_val, y_test):
    print('\n' + '='*60)
    print('EXPERIMENT 4 — TUNED MODELS (5-fold CV on targetTitle TF-IDF)')
    print('='*60)

    # Load raw title text for TF-IDF
    train_df, test_df = load_raw_corpus()
    raw_tr, raw_vl, y_tr, y_vl = train_test_split(
        train_df['targetTitle'].values, train_df['truthClass'].values,
        test_size=0.2, random_state=RANDOM_STATE,
        stratify=train_df['truthClass'].values
    )
    vec = TfidfVectorizer(max_features=5000)
    X_tr_tfidf = vec.fit_transform(raw_tr)
    X_vl_tfidf = vec.transform(raw_vl)
    X_te_tfidf = vec.transform(test_df['targetTitle'].values)
    y_te = test_df['truthClass'].values

    param_grids = {
        'LogReg':       {'clf__C': [0.1, 1.0, 3.0]},
        'LinearSVM':    {'clf__C': [0.5, 1.0, 2.0]},
        'RandomForest': {'n_estimators': [300, 600], 'max_depth': [None, 20, 40]},
        'DecisionTree': {'max_depth': [None, 10, 20, 40], 'min_samples_split': [2, 5, 10]},
    }
    if HAS_XGB:
        param_grids['XGBoost'] = {
            'n_estimators': [300, 600],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1]
        }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, mdl in get_base_models():
        grid = param_grids.get(name)
        if grid is None:
            continue
        gs = GridSearchCV(clone(mdl), grid, scoring='f1_macro', cv=cv,
                          n_jobs=-1, refit=True, verbose=0)
        gs.fit(X_tr_tfidf, y_tr)
        yv = gs.predict(X_vl_tfidf)
        yt = gs.predict(X_te_tfidf)
        rep_v = classification_report(y_vl, yv, output_dict=True, zero_division=0)
        rep_t = classification_report(y_te, yt, output_dict=True, zero_division=0)
        rows.append({
            'Model':       name,
            'BestParams':  str(gs.best_params_),
            'VAL_F1':      round(rep_v['macro avg']['f1-score'], 4),
            'TEST_F1':     round(rep_t['macro avg']['f1-score'], 4),
            'VAL_Acc':     round(rep_v['accuracy'], 4),
            'TEST_Acc':    round(rep_t['accuracy'], 4),
        })
        print(f'  {name}: best={gs.best_params_}  test F1={rows[-1]["TEST_F1"]}')

    return pd.DataFrame(rows).sort_values('TEST_F1', ascending=False)


# ── experiment 5: balanced datasets ─────────────────────────────────────────

def run_balanced(train_df, test_df):
    print('\n' + '='*60)
    print('EXPERIMENT 5 — FIVE BALANCED TRAINING SETS (undersampling)')
    print('='*60)

    df0 = train_df[train_df['truthClass'] == 0]
    df1 = train_df[train_df['truthClass'] == 1]
    n1  = len(df1)

    seeds   = [11, 22, 33, 44, 55]
    models  = get_base_models()
    all_rows = []

    for seed in seeds:
        df0_sample = df0.sample(n=n1, random_state=seed, replace=False)
        df_bal = pd.concat([df0_sample, df1], ignore_index=True).sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)

        tr, vl = train_test_split(
            df_bal, test_size=0.2, random_state=RANDOM_STATE,
            stratify=df_bal['truthClass']
        )

        vec = TfidfVectorizer(max_features=5000)
        X_tr = vec.fit_transform(tr['targetTitle'])
        X_vl = vec.transform(vl['targetTitle'])
        X_te = vec.transform(test_df['targetTitle'])
        y_tr = tr['truthClass'].values
        y_vl = vl['truthClass'].values
        y_te = test_df['truthClass'].values

        for name, mdl in models:
            clf = clone(mdl)
            clf.fit(X_tr, y_tr)
            rep_t = classification_report(
                y_te, clf.predict(X_te), output_dict=True, zero_division=0
            )
            all_rows.append({
                'Seed':     seed,
                'Model':    name,
                'TEST_Acc': round(rep_t['accuracy'], 4),
                'TEST_F1':  round(rep_t['macro avg']['f1-score'], 4),
            })

    df_all = pd.DataFrame(all_rows)
    summary = (
        df_all.groupby('Model')[['TEST_Acc', 'TEST_F1']]
        .agg(['mean', 'std'])
        .round(4)
    )
    print('\n  Results averaged across 5 seeds:')
    print(summary.to_string())
    return df_all, summary


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print('Loading feature matrices...')
    X_train, X_val, X_test, y_train, y_val, y_test = load_features()
    train_df, test_df = load_raw_corpus()

    val_hand, test_hand   = run_handcrafted(X_train, X_val, X_test, y_train, y_val, y_test)
    val_case, test_case   = run_case_based(X_train, X_val, X_test, y_train, y_val, y_test)
    val_tfidf, test_tfidf = run_tfidf(train_df, test_df)
    df_tuned              = run_tuned(X_train, X_val, X_test, y_train, y_val, y_test)
    df_bal, bal_summary   = run_balanced(train_df, test_df)

    # save results
    out = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out, exist_ok=True)

    test_hand.sort_values('F1', ascending=False).to_csv(
        os.path.join(out, 'classical_handcrafted_test.csv'), index=False)
    test_case.sort_values('F1', ascending=False).to_csv(
        os.path.join(out, 'classical_casebased_test.csv'), index=False)
    test_tfidf.sort_values('F1', ascending=False).to_csv(
        os.path.join(out, 'classical_tfidf_test.csv'), index=False)
    df_tuned.to_csv(
        os.path.join(out, 'classical_tuned_test.csv'), index=False)
    bal_summary.to_csv(
        os.path.join(out, 'balanced_summary.csv'))

    print(f'\nAll results saved to {out}/')


if __name__ == '__main__':
    main()
